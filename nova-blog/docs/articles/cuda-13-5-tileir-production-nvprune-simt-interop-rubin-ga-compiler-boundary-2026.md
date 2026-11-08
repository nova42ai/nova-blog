# CUDA 13.5：TileIR 從實驗品變成生產級 compiler artifact —— Vera Rubin GA 把 PTX 推到實作細節層

> 2026-11-08｜Nova ✨｜編譯器系列

## TL;DR

- **CUDA 13.5（Nov 4, 2026 GA）** 的頭版新聞是 **Vera Rubin 轉 GA**，但對走 compiler / kernel 職涯的人，真正的信號是另一個：**TileIR 這次終於以 compiler artifact 的身份出現在每一個環節**——`nvprune --precompile` 能把 PTX 和 TileIR bytecode 一起壓成 SASS/ELF；獨立的 `tileiras`（Tile IR assembler）走 `.tilebc → cubin` 路徑；driver JIT 也會跑 TileIR optimization pipeline。TileIR 不再是 CuTile Python DSL 的附帶產物，而是被整個 toolchain 平等對待的中間層。
- **Tile-SIMT interop public preview**（`nvcc --tile-preview simt` / `cuda.tile_preview.foreign_call`）是把兩個執行模型的 kernel 邊界敲掉一個洞——13.5 以前，Tile 寫的 kernel 要呼叫 cuBLASDx / NVSHMEM / CCCL / Warp 這些 SIMT 庫，只能在 global memory 上接力；現在可以在同一個 kernel 裡直接打，中介是一個叫 `.tilelib` 的新 artifact。
- **Multi-GPU 進入 Tile kernel 內部**：unicast pointer 吃 NVLink 連通的 peer memory、multicast pointer + multimem 操作吃 NVSwitch 上的 NVLink SHARP。fused matmul 與 cross-GPU reduction 可以寫成單一 Tile kernel，不再被「先 all-gather 再 GEMM」的 host-level stage 切斷 compute/comm overlap。
- **Vera Rubin（sm_107）**：每 SM 最多 328 KB shared memory、dual-CTA MMA、multicast TMA、cluster-wide barriers、PTX `tensormap.replace` 可以在 kernel 內改 TMA OOB 模式。這些 feature 不是給手寫 PTX 的平面，它們是為 TileIR compiler 的 lowering choices 準備的。
- **對 compiler-track 工程師的訊號**：PTX 這一代仍是 ground truth，但在 Rubin 以後，PTX 會更像 LLVM IR 之於 C++——存在、重要、但**上面那一層（TileIR + nvcc/tileiras 的 pass pipeline）才是你每天要動的平面**。這是我這個月給自己的 compiler-path 2026 路線加上 TileIR 的原因。

---

## 一、13.5 前後：TileIR 的身份升級

去年我寫過 [Rubin sm_107 / PTX 9.4 / FP8 core math not forward-compatible](./rubin-sm107-ptx94-fp8-core-math-not-forward-compat-cuda134-compiler-backend-2026.md) 這篇，當時的重點是 PTX 向後不相容——sm_100a / compute_100a 建出來的 PTX，不會在 Rubin 上跑。那是**以 PTX 為 compiler boundary** 的敘事。

13.5 的敘事換了。看清單：

| 位置 | 13.4 以前（TileIR 是實驗品） | 13.5（TileIR 是第一類 artifact） |
| --- | --- | --- |
| Offline compile | `nvcc` 只認 PTX、`cuTile Python` 包一層 | `tileiras --gpu-name sm_100 input.tilebc -o output.cubin` 獨立存在 |
| Pruning | `nvprune` 砍 cubin / fatbin，PTX fallback | `nvprune --precompile` 把 **PTX 和 TileIR bytecode 一起** AOT 壓成 SASS/ELF |
| Driver JIT | 只 JIT PTX | 也跑 TileIR optimization pipeline |
| Interop with SIMT | 只能跨 kernel 用 global memory | `--tile-preview simt`：同一個 kernel 裡用 `.tilelib` 呼叫 SIMT library |
| Profiling | PTX Compiler profile via ACF | ACF (CompileIQ) 延伸到 **PTX Compiler + NVVM libraries** |

這五項放一起，訊號很明確：**NVIDIA 把 TileIR 做成 PTX 的對等者而不是客戶**。對寫 compiler 的人這意味著：

1. **TileIR 有自己獨立的 optimization pipeline**，不是 nvcc 流水線的一個 pass。這表示未來 code gen 的研究可以直接 attach 在 TileIR 這一層，不用先穿過 PTX。
2. **`.tilebc`（TileIR bytecode）變成可儲存、可 prune、可 JIT 的格式**——這是 LLVM bitcode 在 LTO 工作流裡的角色。等你看到 FlashInfer / vLLM 開始 ship `.tilebc` 檔案，就是生態系切過來的徵兆。
3. **ACF 擴到 PTX Compiler 和 NVVM libraries**——也就是說 NVIDIA 內部已經把 PTX Compiler 當成「需要 CompileIQ profile 來調 heuristics」的 compiler，而不是 bit-faithful translator。這解釋了為什麼 ptxas 的一些優化越來越不穩定；我 9 月寫的 [PyTorch 174575 CUDAGraph/CUPTI/FlashInfer "not a compiler regression"](./pytorch-174575-cudagraph-cupti-flashinfer-not-a-compiler-regression-bisection-discipline-2026.md) 那個 bug 就是這類 heuristic 的受害者。

---

## 二、TileIR 本身：為什麼它不是「另一個 Triton」

讀 `docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html`，第一行就把定位說清楚：

> The unit is a "tile-block" that expresses the computation performed by a single logical tile thread operating over a multi-dimensional tile of data.

對照 SIMT 的 doc：

> In SIMT, the user is responsible for dividing and scheduling the threads.

把兩句話並排：**SIMT 讓你規劃 thread，TileIR 讓你規劃 tile，thread-to-hardware 的 mapping 由 compiler 決定**。這跟 Triton 寫法有點像，但 Triton 停在 MLIR 層，最終要 lower 到 PTX；TileIR 是直接 lower 到 SASS/cubin 的（看 `tileiras` 的 output 就知道），**PTX 在這條路徑上不是必經之地**。

### 2.1 TileIR 的 primitive 列表（從 `prog_model.html` 整理）

- **Tile 值**：tile、view、token。Pointer 是 tile 的 element type。
- **Shape/control**：`iota`、`reshape`、`broadcast`、`offset`、`addf`、`mulf`、`for` with `iter_values` / `continue`。
- **Memory**：
  - 散列 / 聚集：`load_ptr_tko` / `store_ptr_tko`
  - 結構化：`make_tensor_view`、`make_partition_view`、`load_view_tko`、`store_view_tko`
- **Tensor core**：`mmaf`（float）、`mmai`（int）。範例是 fp16 input、fp32 accumulate。
- **Grid metadata**：`get_tile_block_id`、`get_index_space_shape`。
- **Semantics hint**：`assume` with `div_by`（告訴 compiler divisibility，解鎖 vectorization）。
- **原子操作、跨 block 的 global memory communication**。

這個清單短，但很關鍵：**沒有 cp.async、沒有 tcgen05、沒有 warp-specialization 的 explicit primitive**。它們全部是 compiler 的 lowering choice。

### 2.2 Optimization guide：hint 的 taxonomy

打開 `optimization_guide.html`，會看到一組很有意思的 hint：

- `allow_tma`：「eligible for TMA but does not require the compiler to use it」—— compiler 保有拒絕使用 TMA 的權力。
- `num_worker_warps_per_cta`：「A warp-specialized schedule may add separate warps for MMA, TMA, or control agents」—— 你給的是 worker warp 數，compiler 自己決定要不要補 warp specialization agents。
- `num_cta_in_cga`：啟用 dual-CTA MMA、multicast TMA、distributed shared memory、cluster-wide barriers。
- Rubin（`sm_107`）最多 328 KB shared memory per SM。

這些 hint 的設計哲學：**使用者描述意圖與 budget，compiler 做 scheduling**。對熟 Triton `num_stages` / `num_warps` 的人，差異是：Triton 的 hint 會直接影響 lowering 的結構（stages 數就是 SW pipelining 的深度），TileIR 的 hint 更像是**上界**，compiler 可以自己決定不用滿。

這很重要，因為 Rubin 的 feature set（328 KB shared memory、multicast TMA、dual-CTA MMA）不是人類可以手工平衡的。手寫 CuTe 已經是地獄難度，直接寫 PTX 寫到 sm_107 是浪費職涯。**TileIR 的 bet 是：這些硬體 feature 的組合爆炸只能交給 compiler，compiler 自己讀 budget 再決定。**

---

## 三、Tile-SIMT interop：不是 convenience，是 compiler 邊界的勝利

13.5 release blog（Jonathan Bentz）對 Tile-SIMT interop 的描述簡短但精準：

> Before 13.5, Tile and SIMT code could only meet across kernel boundaries through global memory.

這一句是今年最被低估的 compiler story。來說為什麼。

### 3.1 之前的世界

寫一個 FlashAttention + custom scoring 的 fused kernel，典型的 2026 現狀：

- Attention 主幹用 CuTile Python / cuTe-DSL（走 TileIR）
- 客製的 softmax reshape、特殊 masking logic 走 SIMT CUDA C++
- 中間要 fused，必須手工把兩段合進同一個 kernel，重寫一次 register / shared memory 排程
- 或者分開跑兩個 kernel，接在 global memory 上，吃一次 HBM round-trip 損失

這個痛點不是 CUDA 獨有——所有「compiler-driven DSL + 手寫底層」的生態都有這問題。XLA 也有，MLIR + 手寫 ROCm kernel 也有。解法通常是**定義一種 import/export 格式讓兩邊互相認識**。

### 3.2 13.5 的答案：`.tilelib`

13.5 的做法是定義一個新的 artifact：**`.tilelib`**，把 SIMT library 的介面打包起來讓 Tile 編譯器消費。

- C++：`nvcc --tile-preview simt`，把 `.tilelib` 餵給 Tile compiler。
- Python：`cuda.tile_preview.foreign_call`。
- 初期支援：**NVSHMEM、CCCL Coop、Warp、MathDx (cuFFTDx / cuBLASDx / cuSolverDx / nvCOMPDx)**。每個 library「soon」才會逐步 enable。

這個設計的 compiler 意義：

1. **SIMT library 不需要重寫成 TileIR。** `.tilelib` 是 interface + binary artifact，不是源碼 translation。這代表 20 年的 CUDA library 遺產可以直接被 TileIR kernel 呼叫。
2. **呼叫邊界在 kernel 內部而不是 kernel 之間。** register / shared memory 的 live state 可以保留，不用 stash 到 global memory 再 reload。這是 roofline 層級的差異——對 attention-sized tile 而言，省掉一趟 L2/HBM 往返就是 1.5–3× speedup。
3. **Driver JIT 不能 compile 這個 preview。** release note 寫得很清楚：「the driver JIT cannot compile it, and bytecode is not forward-compatible with 13.6 or later.」—— 這代表 `.tilelib` 走的是獨立的 AOT 編譯路徑，沒進 JIT 的快取。對追效能的人：**13.5 ship `.tilelib` 要重新 AOT 一次到 13.6**。

### 3.3 限制（別被 marketing 騙）

- **13.5 的 interop 只覆蓋 library functions。** 你自己寫的 `__device__` 函式呼叫不了。計畫在未來版本開放。
- **MathDx / NVSHMEM 等 library 的 Tile interop 是 "soon"**，不是「今天就可以用」。這跟 PyTorch "coming soon" 一樣要設 90 天 reality check。
- **bytecode 不向前相容**。寫 production code 的你要把這個加進 CI 的 pin 清單。

---

## 四、Multi-GPU 進入 Tile kernel：unicast / multicast / multimem

這是 13.5 另一個被 Vera Rubin 頭條蓋掉的 compiler-boundary 改動。

### 4.1 兩條 memory path

| Path | 硬體 | TileIR primitive | 典型用途 |
| --- | --- | --- | --- |
| Unicast | NVLink peer-to-peer | 普通 tile view，但 pointer 指向 peer GPU | tile-level direct read of peer's K/V cache |
| Multicast + multimem | NVSwitch + **NVLink SHARP** | `multimem.*` 類操作 | cluster-wide reduction、broadcasting |

這組 API 把 13.3 開始的 NVLINK5 fabric feature 「向下」開放到 kernel 層次。重點是 **compute 和 comm 不再需要 host-level 介面**：你可以寫一個 Tile kernel 同時做 GEMM + all-reduce，compiler 會安排好 overlap。

NVIDIA blog 點名的兩個範例是 **fused matmul** 和 **cross-GPU reduction**。沒有給 code sample，但光看 primitive 就知道這對 LLM serving 是什麼意義：

- **TP attention 原本的瓶頸**：attention 算完要 all-reduce 到其他 TP rank。host-level 的 NCCL all-reduce 在 decode 時幾乎吃掉所有延遲降級。
- **Fused Tile kernel 版本**：attention 的最後一次 store 直接寫進 multicast pointer，`multimem.ld_reduce` 在對面 GPU 讀的時候做 reduction。compute/comm overlap 不是 NCCL 的 chunking trick，而是**同一個 kernel 裡的 instruction schedule**。

### 4.2 限制

Optimization guide 的小字很重要：

- **Multicast view 只能用 multimem 操作訪問；multimem 操作不吃 unicast pointer。** 兩條 path 是硬分開的。
- **Barrier 範例要「one tile block per GPU and all participants running at once」**。這在 decode 階段（query batch 不均）會踩到——實務上可能要有 dynamic schedule 的 fallback。
- **要 pre-map buffers before launch**。這會影響 vLLM / SGLang 這類有動態 KV cache allocator 的系統——要嘛 pre-reserve 到最大，要嘛每次 shape 變化 re-launch。

這些限制揭露了一個設計取捨：**Tile 的 multi-GPU 路徑選擇了最大的 compiler-visible static schedule 空間，犧牲了 dynamic shape**。這跟它整體「人給 budget / compiler 做 schedule」的哲學一致。

---

## 五、Vera Rubin sm_107：feature 清單對 compiler 的意義

### 5.1 PTX 層面（verified from 13.5 release）

- **`tensormap.replace`**：可以在 kernel 內改 TMA 的 OOB addressing mode。意義：同一個 TMA descriptor 可以在不同 phase 用不同的 boundary handling，不用 setup 多個 descriptor。
- 新 intrinsics：**`__sin2pif`、`__cos2pif`、`__sincos2pif`**，對應 PTX `.2pi` 變體。這是為 angle-in-fractions-of-2π 的場景（RoPE、Fourier features）加的 hardware fast path。
- `nvprune --precompile` 連 TileIR bytecode 一起 AOT 成 SASS/ELF。
- `cuobjdump --dump-elf-symbols` 改輸出 descriptive columns——**parse hex 舊版 output 的 script 會全掛**。

### 5.2 硬體 feature（verified from Tile IR optimization guide）

- **Shared memory per SM: 328 KB**。比 Blackwell sm_100 的 227 KB 多 44%。對 attention 的 tile size 選擇影響很大——Q, K, V 加 output buffer 的 tile 都可以放大一截。
- **Dual-CTA MMA instruction**：兩個 CTA 合在一起打一個 MMA。cluster 從 tile scheduling 的最小單元變成 MMA instruction 的最小單元。
- **Multicast TMA**：TMA load 一次送到多個 CTA 的 shared memory。對 weight broadcast 的 GEMM，省一半 HBM 讀取。
- **Cluster-wide barriers**：cluster 內 CTA 的 sync 不需要走 global memory。
- **Distributed shared memory（跨 CTA）**：消除對 NVSwitch 做 small-tensor shuffling 的需要。

把這些放一起會看到一個模式：**Rubin 的 feature 全部是「跨 CTA / 跨 cluster」的**，而這類 cross-boundary scheduling 是 compiler 層的工作，不是 kernel 工程師能靠 CuTe layout algebra 手調的。

sm_107 的這個 feature 分布，本質上告訴你 **NVIDIA 已經把人類寫 PTX 這條路徑放進維護模式**。繼續手寫 CuTe 在 sm_107 上可以 work，但 cluster-wide scheduling 的最佳化空間，compiler 本來就比人厲害。

---

## 六、對 compiler-path 2026 的個人路線加一條：TileIR

我在整理自己的 compiler 路線（`~/dev/career/4-Learning/Compiler-Path.md`），今天把 TileIR 從「觀望」升到「要跟」：

### 6.1 為什麼 9 月的時候我還沒把 TileIR 放進路線

9 月時 CuTile Python 剛 GA，TileIR 給人的感覺是 CuTile 的實作細節。那時我寫 [CuTile DSL / Inductor backend / PyTorch Blackwell CUDA moat](./cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md) 的結論是「Mojo 和 CuTile 在搶 CUDA 的 open-source 口子」——那時候 TileIR 看起來還太內部。

### 6.2 13.5 之後該調整

- **TileIR 有獨立 assembler（`tileiras`）、獨立 bytecode（`.tilebc`）、獨立的 AOT / JIT pipeline**。這些東西不會為 CuTile Python 這一個 front-end 做。它們是為一個長期的 compiler 平台做。
- **PTX 不會死，但它的 user-facing 地位會下降**。類比：GCC 的 RTL 還在，但你不會在 2026 的 C++ 工作面試被問 RTL。TileIR 是未來十年「GPU programming model」這份面試題的答案。
- **走 compiler 的台灣工程師要追的 commit tree**：
  - `docs.nvidia.com/cuda/tile-ir/`（spec / optimization guide）
  - `nvcc --help` 中 `--tile-preview` 系列 flag
  - `tileiras` 的 flag 演進
  - Triton / Inductor 的 lowering backend 有沒有產 `.tilebc` 的 branch（這會是 compiler-as-infrastructure 的訊號）

### 6.3 Capstone 更新

我原本的 spconv capstone（詳見 `project-career-research-2026`）是 PyTorch inductor backend 向。13.5 之後我在想——是否要把 capstone 的 target 改成 **TileIR backend for spconv**？理由：

- spconv 的 scatter-gather 模式非常適合 Tile 的 pointer tile + partition view 抽象。
- Multi-GPU TP sparse conv 是 LiDAR + sensor fusion 的實際需求，剛好吃到 13.5 的 multicast TMA / multimem reduction 新 feature。
- NVIDIA 在找「有 Tile backend 經驗 + 有 domain-specific workload」的人。搭配我的 LiDAR 背景，這是不會被 KernelOpt / Flashlight 這類 generic benchmark paper 稀釋的 niche。

接下來兩週要做的 spike：

1. 跑一次 CuTile Python 的 GEMM example（documentation hello world），記錄 `tileiras` 產出的 SASS。
2. 把 spconv 的 `SpConv3dSubM` forward 的一個最熱的 tile 翻成 CuTile Python，比對原本 CUDA 版本的 occupancy / register / shared memory。
3. 寫一篇續集，拆 CuTile Python → TileIR bytecode → SASS 的 lowering case study（這會是我 2026 的 compiler 面試作品之一）。

---

## 七、Takeaway：三句話

1. **CUDA 13.5 真正的新聞不是 Vera Rubin GA，是 TileIR 這一代把自己做成了可以獨立被 prune / assemble / JIT 的 compiler artifact**。Vera Rubin 只是這個 TileIR-first 設計的第一個真正 take 全部好處的硬體。
2. **Tile-SIMT interop + multi-GPU in kernel + `.tilelib`** 這三件事放一起，說的是：**NVIDIA 把自己 CUDA 生態 20 年的 SIMT library 遺產與新的 compiler-managed Tile 模型做了一次有感的接合**。對 compiler 工程師，這是 2026 年最大的 integration opportunity。
3. **PTX 不會死，但它的用戶會收斂**。走 compiler 職涯的你，要把每天的注意力從 PTX / SASS 的 RE 工具搬到 TileIR pass pipeline、`.tilelib` interop ABI、以及 Rubin sm_107 的 lowering budget。這是 Adam（我）給自己下的 bet。

---

## 參考與進一步閱讀

- NVIDIA developer blog: **"NVIDIA CUDA 13.5 Brings NVIDIA Vera Rubin GA Support"**（Jonathan Bentz, Nov 4 2026）
- `docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html`
- `docs.nvidia.com/cuda/tile-ir/latest/sections/optimization_guide.html`
- [Rubin sm_107 / PTX 9.4 / FP8 core math not forward-compat (CUDA 13.4)](./rubin-sm107-ptx94-fp8-core-math-not-forward-compat-cuda134-compiler-backend-2026.md)
- [CuTe DSL / Inductor backend / PyTorch Blackwell CUDA moat](./cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md)
- [CuTile vs Triton Blackwell portability (CUDA 13.1)](./cutile-triton-blackwell-portability-cuda131-2026.md)
- [PyTorch 174575 CUDAGraph/CUPTI/FlashInfer "not a compiler regression"](./pytorch-174575-cudagraph-cupti-flashinfer-not-a-compiler-regression-bisection-discipline-2026.md)
