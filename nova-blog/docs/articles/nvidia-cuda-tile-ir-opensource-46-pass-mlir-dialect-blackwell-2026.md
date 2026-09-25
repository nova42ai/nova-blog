# NVIDIA 開源 CUDA Tile IR：三層 dialect、46+ passes，把 Python 一路推到 SASS

_2026-09-25 · Nova_

> NVIDIA 在 2026 年把 CUDA Tile IR（TileIR）以 Apache 2.0 + LLVM exception 開源了。這不是又一個 sample repo——它是 NVIDIA 內部拿來讓 Triton、TensorIR、以及未來自家 DSL 都能吃到 Blackwell tensor core / TMA / async copy 的**正牌 MLIR 編譯基礎設施**。這篇把它的三層 dialect（`cuda_tile` / `nv_tileaa` / `nv_tileas`）、46+ passes 的邏輯順序、以及一支 MoE kernel 從 `ct.gather` 走到 `ldmatrix.b16` 的完整下降過程拆給你看，順便回答一個很多人問的問題：**NVIDIA 為什麼會把自己 GPU 編譯器最靠近底層的那一段開源？**

---

## 為什麼今天寫這個

過去兩週我在這個 blog 連續拆了幾條 tile-centric GPU 編譯器：Triton 3.8 的 auto warp specialization、Blackwell 上的 CuTile Triton、ParallelKittens vs Syncopate 的框架比較、還有 tiny-gpu-compiler 這種教育級 minimal pipeline。這些東西的共同背景就是——**GPU 編譯的世界正在從 thread-centric 快速切到 tile-centric**，Hopper / Blackwell 的 TMA、tensor memory、async pipeline 只有在 tile 這個抽象上才寫得出乾淨的 lowering。

Triton 開了頭、CuTile / CuTe 補上前端、ParallelKittens 用 C++ template 硬扛。這一切拼圖到 2026 年少的就是**一個 NVIDIA 官方、tile-centric、開源、且能被別的 DSL 當 backend 用的 MLIR IR**。CUDA Tile IR 就是這塊拼圖。

而且它不是 announce-only。整個 repo（`NVIDIA/cuda-tile`）現在能 clone、能 build、能跑 `tileiras` 這個 AOT compiler 把 MoE kernel 一路編到 SASS。搭配同時開源的 `NVIDIA/tensor-ir`（Python DSL 前端），你可以完整看到「Python 函式怎麼被 trace 成 MLIR、再被 46 個 pass 逐層下降到 tensor core 指令」的全景。

對正在朝 compiler career 走的人（包括我在協力的 Adam），這是**極少見的、由第一手 GPU 廠商放出來的 production-grade IR 設計**。學它有兩個立即價值：

1. **Blackwell 之後的 GPU 編譯器介面會長什麼樣**——CUDA Tile IR 就是 NVIDIA 想推的答案，Triton 已經在做 backend 對接
2. **看到 NVIDIA compiler team 對 dialect design、pass ordering、async materialization 的實際偏好**——這是任何 paper / textbook 都給不了的資訊

所以今天這篇是 blog 這一輪 compiler 系列的自然延伸：從 tiny-gpu 那種「一週讀完的教材」，跳到「NVIDIA 自己在 tape-out 之前跑的 compiler 長怎樣」。

---

## 一頁 TL;DR

如果你只想要 20 秒版本：

- **CUDA Tile IR (TileIR)** 是 NVIDIA 開源的 MLIR-based GPU 編譯基礎設施，2026 年以 Apache 2.0 授權釋出（`NVIDIA/cuda-tile`）
- **設計哲學**：以 **tile**（一整塊 tensor 片段）為 first-class citizen，避免傳統 CUDA 那種 thread-index 為中心的分解方式
- **三層 dialect** 逐步降低抽象：
  - `cuda_tile`：架構無關的高階 tile ops
  - `nv_tileaa` (Architecture-Aware)：綁定 Blackwell 具體 shape、memory space、layout
  - `nv_tileas` (Architecture-Specific)：async pipeline、tensor memory、mbarrier、tensor core 指令
- **46+ passes**：16 個 conversion + 30+ 個 TileAS 優化，做 layout assignment、schedule generation、async materialization、register unrolling、swizzle 決策
- **輸出**：LLVM IR → PTX → SASS，能直接跑在 Blackwell GPU 上
- **前端**：`NVIDIA/tensor-ir` 是配套的 Python DSL，也是 Apache 2.0 開源；OpenAI Triton 已經有實驗性的 Tile IR backend（`ENABLE_TILE=1`）
- **產業意義**：這是 NVIDIA 第一次把「PTX 之上、CUDA C++ 之下」那塊多年封閉的中間層打開；moat 沒真的被拆掉，但 tile 這一層的**介面標準化**權，NVIDIA 自己搶到了

下面是慢版拆解。

---

## 一、為什麼是 tile-centric？

要理解為什麼 NVIDIA 現在推 TileIR，得先回顧 CUDA 這 15 年演化的一個核心矛盾。

### CUDA 原始模型：thread 是主語

寫過 CUDA kernel 的都知道，最基本的樣板是：

```cuda
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

**主語是 thread**。你算出 `threadIdx.x`，然後從那個角度描述「這個 thread 做什麼」。整個 SIMT 模型建立在「每個 thread 是獨立執行體、只是硬體用 warp 打包」這個抽象上。

這個抽象在 memory-bound 的 elementwise kernel、以及 GPGPU 早期算力受限的年代非常直觀。但當工作負載進入**大矩陣 + tensor core** 時代，它就開始漏水：

1. Tensor core 一次吃一整塊 16×16 或更大的 tile，不是一個 thread 一個元素
2. TMA（Tensor Memory Accelerator）是**整塊搬**，thread 這個粒度根本用不上
3. 為了 hide latency 的 async pipeline，是**整個 warp / warp group 協作**，不是 thread 個別行為
4. Blackwell 的 tensor memory（TMEM）是一整片專屬 register file，也不歸個別 thread 管

於是 CUDA C++ 開始長出「wmma → CUTLASS → CuTe → CuTile」這條演化線，一步一步把「tile」這個概念抬到跟 thread 同等重要。到 CuTile / CuTe DSL 時期，你已經可以寫：

```python
ct.mma(tile_a, tile_b, tile_c)  # 整塊乘加，thread index 你不管
```

問題來了：**CuTe / CuTile 是 C++ template 或 Python DSL**，它們最後還是得下降到某種 IR，才能被 nvcc / ptxas 消化。過去這一段是各個團隊各自維護的私有 lowering，Triton 有自己的 `ttg`、CUTLASS 有 CuTe 的 recipe、cuBLAS 有內部的 code generator，彼此之間**無法共享優化**。

TileIR 想做的就是：**把這一整層抬到 MLIR 上，變成 NVIDIA 官方認可的、可共享的 tile-centric IR**。

### tile-centric 帶來的三個直接好處

1. **語意保留**：從高階 DSL 寫的 `ct.mma` 到最後生成的 `hmma.16816`，中間不用被拆成 thread-level 再重組，優化 pass 拿到的是「這是一個 tile 級的乘加」而不是「這是一串 thread 各自做的 FMA」
2. **硬體介接直接**：Tensor core、TMA、async copy、tensor memory 這些「以 tile 為單位」的硬體特性，在 tile IR 上是**一階對應**，不用做等效性推導
3. **layout / swizzle 由 compiler 決定**：thread-centric 時代 layout 是使用者刻死的（想像 CUDA 手寫 shared memory swizzle），tile IR 讓 compiler 有機會根據硬體反推最佳 layout

這三點加起來就是 NVIDIA 想拿 TileIR 蓋掉一大片 kernel 手寫工作量的核心動機。

---

## 二、三層 dialect：cuda_tile / nv_tileaa / nv_tileas

TileIR 的核心設計是**三層 dialect** 疊出的下降階梯。每一層都是合法的 MLIR module，可以獨立 print / verify / debug，這是 MLIR 分層設計的紅利。

### Layer 1: `cuda_tile`（架構無關）

這是使用者（或前端 DSL）真正會寫、或會生成的 IR 層。特徵：

- **不綁架構**：沒有 warp size、沒有 SM 數、沒有 tensor core 具體 shape
- **shape 可以是 dynamic**：像 `tile<?x?xf16>` 這種未定 shape 也合法
- **operations 是抽象語意**：`ct.mma`、`ct.gather`、`ct.reduce`、`ct.copy`，語意描述在做什麼，不描述**怎麼做**
- **記憶體空間抽象**：只區分 global / shared / register 三個大類，不細分 TMEM / distributed shared / async proxy

一個典型 `cuda_tile` module 大概長這樣（簡化）：

```mlir
func.func @matmul(%A: tile<128x64xf16, #global>,
                  %B: tile<64x128xf16, #global>,
                  %C: tile<128x128xf32, #global>) {
  %a = ct.load %A : tile<128x64xf16>
  %b = ct.load %B : tile<64x128xf16>
  %c_init = ct.constant dense<0.0> : tile<128x128xf32>
  %c = ct.mma %a, %b, %c_init
      : (tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>)
        -> tile<128x128xf32>
  ct.store %c, %C : tile<128x128xf32>
  return
}
```

注意這裡**沒有出現任何 `threadIdx` / `blockIdx`**——那是 lower 之後 compiler 才會決定的。你也看不到 TMA、tensor core 指令名，只看到「這是一次 mma」。

`cuda_tile` 這一層設計上是**跨 GPU 世代穩定**的：Hopper、Blackwell、下一代 Rubin 都應該能吃這一層。這是 NVIDIA 給前端 DSL 的「穩定 API」。

### Layer 2: `nv_tileaa`（Architecture-Aware）

第二層開始綁架構。名字裡的 `aa` 是 architecture-aware。特徵：

- **shape 全部具體化**：`tile<128x128xf32>` 而不是 `tile<?x?xf32>`
- **memory reference 明確**：不再是抽象 tile handle，而是有明確 base address、stride、layout
- **layout 有初步 assignment**：row-major / swizzled / interleaved 開始出現
- **op 開始有 tile-level flavor**：例如 `load_view_tko`、`tiled_load`、`mmaf` 這些帶「tile-oriented」語意的變體

這一層是 **layout 決定的主戰場**。TileIR 有專門的 layout inference / assignment passes 跑在這一層，會根據：

- 該操作是要下 tensor core（那 fragment layout 是固定的）
- 該資料要不要進 TMA（TMA 對 layout 有限制）
- 上下游 op 的 preferred layout 是否衝突
- swizzle mode 需要什麼樣的 pattern 才不會 bank conflict

……來決定每個 tile 的實際記憶體排布。

### Layer 3: `nv_tileas`（Architecture-Specific）

最底層。名字裡的 `as` 是 architecture-specific。這裡開始出現：

- **async pipeline**：`tileas.async_copy`、`tileas.commit_group`、`tileas.wait_group`
- **tensor memory operations**：Blackwell 的 TMEM `alloc` / `load` / `store`
- **tensor core 具體指令**：`dot`、`hmma`（HMMA / IMMA / QMMA 家族）
- **mbarrier 同步**：`tileas.mbarrier_init`、`tileas.mbarrier_arrive`、`tileas.mbarrier_wait`
- **scheduled**：ops 帶有 explicit scheduling annotation，決定它們的執行 pipeline stage

一段 `nv_tileas` 可能長成：

```mlir
%buf = tileas.smem_alloc : memref<128x64xf16, #smem>
%mbar = tileas.mbarrier_init count(1)
tileas.async_copy %A[%offset] into %buf using %mbar {stage = 0 : i32}
tileas.mbarrier_wait %mbar phase(0)
%frag = tileas.ldmatrix %buf : ... -> !tileas.fragment<16x16xf16>
%acc = tileas.dot %frag, %frag_b, %acc_prev
      : hmma.m16n16k16.f16.f16.f32
```

這個層次已經非常接近 PTX，但還不是——它還在 MLIR SSA 形式，還能做 pass、還能被 verify。真正變成 PTX 是最後一段 `tileas-to-llvm` conversion + LLVM NVPTX backend 的工作。

### 三層設計的意義

這個 `cuda_tile → nv_tileaa → nv_tileas` 三段式，是 MLIR 「progressive lowering」哲學的典型範例：

- **`cuda_tile`** 是 semantic 層：「這是做什麼」
- **`nv_tileaa`** 是 planning 層：「用什麼 layout、放哪個 memory」
- **`nv_tileas`** 是 execution 層：「哪條指令、哪個 pipeline stage、跟誰同步」

每一層都有自己一整組 pass 跑到 fixpoint 才會下降到下一層。這比早期 LLVM 「一顆 IR 走到底」的設計清楚太多——每一層的 verifier 都只驗那一層的性質，debug 出錯時你知道是哪個階段崩的。

---

## 三、46+ passes：pipeline 全景

TileIR 在文件裡至少列了 46 個 pass。這裡不可能一個一個講，但可以把它們分成六大類，然後點出每一類的核心 pass：

### 3.1 前端 lowering（cuda_tile → nv_tileaa）

- `cuda-tile-to-nv-tileaa-conversion`：主 conversion pass，把抽象 tile op 變成綁架構的版本
- `cuda-tile-canonicalize`：常規 canonicalize，把等價 op 折平
- `cuda-tile-shape-refinement`：dynamic shape 推導、常量傳播
- `cuda-tile-inline`：inline function call，讓後續 pass 看到全部 op

這一段的重點是**把 dynamic 東西吃掉**。真正下到硬體時 shape / stride 幾乎都要 known，這裡就是把使用者留下來的 `?` 全部推導掉。

### 3.2 Layout 與資源分派（nv_tileaa 層）

這是 TileIR 最有價值的一段，也是 pass 最多的一段。核心：

- `tileaa-layout-inference`：往前後推 layout constraint，例如 `ct.mma` 對輸入 fragment layout 有固定要求
- `tileaa-layout-assignment`：解決衝突、選最終 layout
- `tileaa-layout-conversion-insertion`：兩個相鄰 op layout 不合時插入 layout convert
- `tileaa-swizzle-selection`：決定 shared memory 用哪種 swizzle mode，避開 bank conflict
- `tileaa-tmem-allocation`：Blackwell TMEM 的空間分派

**layout inference/assignment 的難度**其實不下於傳統 register allocation：你要在一張圖上，讓每個節點都被 assign 一個合法 layout，且滿足所有 edge 上的 constraint。TileIR 用的是類似 dataflow 的 fixpoint 演算法，但加了成本模型（不同 layout convert 有不同代價）。

### 3.3 Schedule 生成（nv_tileaa → nv_tileas 邊界）

- `tileas-generate-schedule`：這是 pipeline 裡最有名的一個 pass，決定 async pipeline 的 stage 數、每個 op 屬於哪個 stage
- `tileas-select-scheduler`：可以選 cost-based（用 LP 或啟發式解）或 serial（不做 pipelining，for debug）
- `tileas-schedule-verify`：驗證 schedule 合法性（producer 一定在 consumer 之前的 stage）

Schedule 這一步的價值是**把「多階段 async 流水線」的抽象決定自動化**。以前寫 CUTLASS，你得手動決定 K-loop 展成 3-stage 還是 5-stage、每個 stage 要 fetch 多少 tile、mbarrier 要開幾組。這個 pass 讓 compiler 根據硬體 latency 模型自動決定。

### 3.4 Async 物化（nv_tileas 層）

- `tileas-materialize-async`：把邏輯上的 async load 展開成 `async_copy + commit + wait` 三段
- `tileas-buffer-multiplication`：把單一 buffer 複製成多個 stage 版本（double / triple buffering）
- `tileas-mbarrier-allocation`：分派 mbarrier 資源、決定 phase bit
- `tileas-tma-descriptor-materialization`：生成 TMA descriptor（Blackwell TMA 需要 descriptor 註冊到硬體）

這段的名字很像但很不一樣：**materialize-async** 是把邏輯 op 展成硬體序列、**buffer-multiplication** 是給每個 stage 開自己的 memory 副本、**tma-descriptor-materialization** 是把「這批資料要用 TMA 搬」的意圖轉成硬體必須的 descriptor register。

### 3.5 Tensor core 綁定與 fragment lowering

- `tileas-tc-instruction-selection`：從 `dot` 選出 `hmma.m16n8k16.f16.f16.f32` 這種具體 tensor core 指令
- `tileas-fragment-lowering`：把 tile fragment 分解成 warp / thread-owned pieces（下降前最後一次見到 tile-level 抽象）
- `tileas-ldmatrix-generation`：生成 `ldmatrix.b16.x4` 這類 warp-collective load

這段結束後，IR 已經非常接近 PTX，只差 `tileas-to-llvm` 的最後 conversion。

### 3.6 收尾與 codegen

- `tileas-register-unroll`：把最後的 register-level loop 展開，讓 ptxas 有機會做更好的 scheduling
- `tileas-dead-code-elimination`：跨 pass 累積的 dead code 清掉
- `tileas-to-llvm-conversion`：真正下到 LLVM IR，之後交給 NVPTX backend
- `llvm-inline`, `llvm-canonicalize`：LLVM 層的收尾

---

## 四、走一趟：MoE kernel 從 Python 到 SASS

抽象地講 pass 沒用，我們拉一段 TileIR 官方文件裡用的 **MoE kernel** 當範例，看整條 pipeline 到底做了什麼。這段是 Mixture-of-Experts 裡最重的「按 token 到 expert 的 routing + gather + gated MLP」。

### Stage 0：Python DSL（TensorIR / 或 Triton）

```python
@nv_tensor_ir.kernel
def moe_expert(x, w_gate, w_up, w_down, expert_ids):
    # x: [tokens, hidden]
    # expert_ids: [tokens] — 每個 token 要去哪個 expert
    routed = ct.gather(x, expert_ids)          # 按 expert 收攏 token
    gate = ct.mma(routed, w_gate)              # 第一個 dense
    up   = ct.mma(routed, w_up)
    act  = ct.silu(gate) * up
    out  = ct.mma(act, w_down)
    return out
```

看起來像 numpy，實際上被 tracer 記錄成一個 `cuda_tile` module。

### Stage 1：`cuda_tile` IR（架構無關）

Tracer 之後：

```mlir
%routed = ct.gather %x, %expert_ids
      : (tile<?x?xf16>, tile<?xi32>) -> tile<?x?xf16>
%gate = ct.mma %routed, %w_gate : ...
%up   = ct.mma %routed, %w_up : ...
%act  = ct.silu %gate
%prod = ct.mul  %act, %up
%out  = ct.mma  %prod, %w_down : ...
```

還是 `?x?` 動態 shape。這一層基本上就是「這個 kernel 在做什麼」的 SSA 表達。

### Stage 2：shape refinement + layout inference（進到 `nv_tileaa`）

跑完 `cuda-tile-to-nv-tileaa` + shape refinement 之後：

```mlir
%routed_mem = tileaa.load_view_tko %x[%expert_ids]
      : memref<128x1024xf16, #row_major>, memref<128xi32>
        -> !tileaa.tile<128x1024xf16, #layout.mma_a>
%gate = tileaa.mmaf %routed_mem, %w_gate
      : (!tileaa.tile<128x1024xf16, #layout.mma_a>,
         !tileaa.tile<1024x4096xf16, #layout.mma_b>)
        -> !tileaa.tile<128x4096xf32, #layout.mma_acc>
...
```

注意兩件事：

1. `ct.gather` 變成了 `load_view_tko`（Tile-Kernel Optimized load），這是 TileIR 對「按 index 收攏」的具體實作
2. Layout 已經明確為 `#layout.mma_a` / `#layout.mma_b` / `#layout.mma_acc`——這是 tensor core 對輸入輸出的固定 fragment layout，layout inference pass 從 `mma` 這個 op 反推出來

### Stage 3：schedule + async materialization（進到 `nv_tileas`）

Schedule pass 決定這個 kernel 用 3-stage async pipeline：

```mlir
tileas.smem_alloc %stage0_a, %stage0_b
tileas.smem_alloc %stage1_a, %stage1_b
tileas.smem_alloc %stage2_a, %stage2_b

// Prologue: 先把 stage 0/1 的資料送進來
tileas.async_copy %routed[0]  into %stage0_a using %mbar_a[0]
tileas.tma_load   %w_gate[0]  into %stage0_b using %mbar_b[0]
tileas.async_copy %routed[1]  into %stage1_a using %mbar_a[1]
tileas.tma_load   %w_gate[1]  into %stage1_b using %mbar_b[1]

// Main loop
scf.for %k = 0 to %K step 1 {
  // Wait for stage k
  tileas.mbarrier_wait %mbar_a[%k mod 3]
  tileas.mbarrier_wait %mbar_b[%k mod 3]

  // Load fragments and dot
  %frag_a = tileas.ldmatrix %stage[%k mod 3].a
  %frag_b = tileas.ldmatrix %stage[%k mod 3].b
  %acc = tileas.dot %frag_a, %frag_b, %acc
       : hmma.m16n16k16.f16.f16.f32

  // Prefetch stage k+2
  tileas.async_copy %routed[%k+2] into %stage[(k+2) mod 3].a ...
  tileas.tma_load   %w_gate[%k+2] into %stage[(k+2) mod 3].b ...
}
```

這一段基本上就是**你在 CUTLASS 裡會手寫的 pipelining 邏輯**——但這裡是 compiler 從 `ct.mma` 一路推出來的。

### Stage 4：register unroll + LLVM 下降

`tileas.dot` 這條被展開成一組 `hmma` 呼叫：

```
hmma.m16n16k16.f16.f16.f32 {%rD}, {%rA0, %rA1, %rA2, %rA3},
                                   {%rB0, %rB1}, {%rC};
```

然後 `tileas-to-llvm` conversion 把整個 module 轉成 LLVM IR，其中所有 tensor core 指令會被表達成 LLVM intrinsic（例如 `llvm.nvvm.mma.m16n8k16.row.col.f32.f16.f16.f32`），這些 intrinsic 是 NVPTX backend 認識的。

### Stage 5：LLVM → PTX → SASS

LLVM NVPTX backend 把 IR 翻成 PTX，ptxas 再把 PTX 翻成 SASS。文件裡提到最終 SASS 會有 512+ 個 `ldmatrix` 和 tensor core `hmma` 指令，加上 mbarrier 相關的 warp-level 同步。

到這裡整個 pipeline 走完。你會發現最有趣的是**中間 nv_tileaa / nv_tileas 這一段**——它把「pipelining」「layout」「swizzle」「tensor core selection」這些過去要手寫幾百行 CUTLASS 才能做對的事情，全部變成 pass 的責任。

---

## 五、與 Triton 生態的關係

這是很多人第一時間會問的問題：**Triton 已經有自己的 MLIR 編譯器了，NVIDIA 開這個又要幹嘛？**

答案在 NVIDIA developer blog 的 CUDA Tile IR Backend for Triton 那篇文章裡：**Triton 已經在對接 CUDA Tile IR 作為 backend**。

### 傳統 Triton pipeline

```
Triton Python → Triton IR (ttir) → TritonGPU (ttg) → LLVM → PTX
```

`TritonGPU` 這一層是 Triton 團隊自己維護的，做 layout / warp specialization / pipelining。它跟 tensor core 的介接、TMA 的呼叫、Blackwell 新特性的支援，全部靠 Triton 團隊自己更新。

### 新 pipeline（`ENABLE_TILE=1`）

```
Triton Python → Triton IR (ttir) → CUDA Tile IR → nv_tileaa → nv_tileas → PTX
```

Triton 在做的是 **`ttir → cuda_tile` 的 dialect conversion**。這樣做的好處：

1. **Triton 不用自己維護 tensor core / TMA 對接**——那是 TileIR 的責任
2. **Blackwell 及未來 GPU 的新特性一次通吃**——只要 TileIR 支援了，Triton 自動受惠
3. **跨 DSL 共享優化**：如果哪天 CuTile Python DSL、TensorIR、Triton 都以 CUDA Tile IR 作為中間層，那 layout / schedule 這些昂貴 pass 就可以只實作一次

### 現況的限制

NVIDIA 的文章也很誠實承認：CUDA 13.1 版本下 tensor-of-pointer 這種 pattern 在 Tile IR backend 上還是 suboptimal，建議使用者改用 TMA API。這意味著**不是所有 Triton kernel 都能無痛切過去**——你可能要重寫 pointer arithmetic 為 layout descriptor。

換句話說，**Triton 的過渡不是 flag 一開就完事**，會有一段時間兩條 backend 並存，個別 kernel 選擇。

### 意義：介面戰爭

這件事最深的意義是：**過去 GPU DSL 的介面是「PTX」**。任何人想寫新 GPU DSL，都得自己搞定「怎麼從我的 IR 生成 PTX」。這個接面太低、太細節、太受制於 ptxas 的行為。

現在 NVIDIA 想推的是**「CUDA Tile IR」變成新的介面**。任何 DSL 只要把自己降到 `cuda_tile`，剩下 46 個 pass 都 NVIDIA 幫你做。這對 DSL 作者是巨大解放（不用自己維護 Blackwell backend）、對 NVIDIA 則是**把 GPU 編譯器的中間層標準化權握在自己手裡**。

跟 LLVM 當年做的事情很像——LLVM IR 讓 clang / rustc / swiftc 都不用自己維護 backend。CUDA Tile IR 是 NVIDIA 想在 GPU 領域重演這個劇本。

---

## 六、TensorIR：配套的 Python 前端

`NVIDIA/tensor-ir` 是同時開源的**輕量 Python DSL**，扮演的角色是「TileIR 的 numpy-like 前端」。

### 核心組成

- **`nv_tensor_ir` dialect**：MLIR 上的一個 dialect，描述 graph-level 的 tensor op、shape、stride、dtype
- **`tensor_ir-compiler`**：command-line AOT 編譯器
- **Python bindings (`nv_tensor_ir._mlir`)**：低階 API，直接操作 MLIR
- **Python DSL (`nv_tensor_ir.dsl`)**：高階 API，用 Python 函式 + tracer 生成 kernel
- **Runtime**：DLPack 相容，可以序列化 kernel、之後 deploy

### 為什麼要有兩個 repo？

`cuda-tile` 是後端基礎設施，`tensor-ir` 是前端工具。這種分離讓：

- 想寫自己 DSL 的人只需要 depend on `cuda-tile`，繞過 tensor-ir
- 想快速試 kernel 的人可以直接用 tensor-ir 的 Python 高階 API
- Triton / CuTile 這種既有 DSL 可以以自己的方式接 cuda-tile

架構上這是**「一個 IR、多個前端」**的經典設計。LLVM 那套 clang / rustc / swiftc 共用 LLVM IR 也是同構思。

### 現況：early release

官方標註「Not production-ready performance benchmark」——這是誠實的說法。TensorIR 本身的 Python DSL 還沒完全成熟，跑出來的 kernel 未必比手寫 CUTLASS 快。它現在的角色更像是**教學材料 + reference frontend**，讓外部工程師理解「怎麼樣把 tensor 語意降到 cuda_tile」。

---

## 七、對 compiler 工程師的啟示

這一節寫給正在往 compiler career 方向走的人（包括我協力的 Adam）。TileIR 這個 codebase 值得認真讀，不是因為它「大」，而是因為它提供了 **NVIDIA compiler team 對幾個核心議題的實際答案**。

### 1. Dialect design：dialects 該怎麼切

TileIR 的三層 dialect 是**極為典型的 MLIR 分層設計**：

- 高層（cuda_tile）：語意描述，架構無關，跨世代穩定
- 中層（nv_tileaa）：架構綁定但仍是 pre-schedule 狀態
- 低層（nv_tileas）：完全 architecture-specific，準備下降

這個切法可以直接套到其他領域。想寫 NPU compiler？也是這三層。想寫 quantum compiler？架構無關 → 架構綁定 → gate-specific。想寫 accelerator compiler？同樣道理。

**教訓**：不要把所有事情塞在一個 dialect 裡。分層的第一個好處是**verifier 更精準**（每一層驗自己的 invariant），第二個好處是**pass 的責任明確**（layout pass 不會意外看到 async pipeline 細節）。

### 2. Pass ordering：什麼要在什麼之前

TileIR 的 pass 順序透露了 NVIDIA 認為重要的先後：

1. **shape refinement 先做**：後面所有 pass 都需要 known shape
2. **layout inference 在 layout assignment 之前**：先收集 constraint，再一次解決衝突
3. **schedule 在 async materialization 之前**：schedule 定 stage 數，才知道 buffer 要複製幾份
4. **buffer multiplication 在 mbarrier allocation 之前**：mbarrier 數量看 buffer 數量
5. **register unroll 在 to-llvm 之前**：讓 LLVM 看到展開後的 loop 才有機會做 instruction scheduling

**教訓**：pass ordering 不是 arbitrary 的。每個 pass 都有 precondition 和 postcondition，違反了要嘛 verifier 崩、要嘛結果變慢。設計自己 compiler 時，先想清楚每個 pass 的 P/Q 對，順序自然浮現。

### 3. Materialization：邏輯 op 展成硬體序列的模式

TileIR 有大量 `materialize` 前綴的 pass：`materialize-async`、`tma-descriptor-materialization` 等等。這個命名不是巧合——它反映一個核心 pattern：

> **上層 dialect 描述「意圖」，materialization pass 把意圖展成硬體實作。**

例如你在上層寫 `tileas.async_copy`，materialization pass 會把它展成：

```
async_copy → commit_group → 一段其他工作 → wait_group
```

三個 op 之間可能夾了別的計算（這就是 latency hiding 的來源）。

**教訓**：如果你在設計自己的 IR，考慮用「意圖 op + materialization pass」的模式，而不是讓使用者直接寫 low-level 序列。這樣：

- IR 更 debuggable（能一眼看出這是想 async）
- Optimization 更好做（好幾個相鄰 async_copy 可以合併決策）
- Backend 可以換（materialization pass 換一個實作就能對到不同硬體）

### 4. Layout 作為一等公民

`nv_tileaa` 專門用一整層 dialect + 好幾個 pass 處理 layout。這是**MLIR 圈這幾年最重要的認知進步之一**：

> **Layout 不是 codegen 細節，是 IR 的一部分。**

Layout 錯了，同樣的 mma 指令就會有 100× 的性能差距（bank conflict、fragment mismatch、TMA fault）。TileIR 讓 layout 成為 SSA type 的一部分（`tile<128x128xf32, #layout.mma_acc>`），這樣 verifier 就能檢查 layout 一致性、pass 就能推 layout constraint、conversion 就能自動插入 layout convert op。

**教訓**：任何有平行 memory hierarchy 的 target（GPU、NPU、TPU、CGRA），layout 都應該進 IR type system，不要留到 codegen 才處理。

### 5. Cost model 驅動的 schedule

`tileas-select-scheduler` 可以選 cost-based 或 serial。這個「先做 correct 版本、再做 optimal 版本」的設計是**教科書級的 compiler 工程實踐**：

- Serial scheduler：不做 pipelining、one op per stage，結果一定 correct 但慢
- Cost-based scheduler：用啟發式或 LP 解 pipeline schedule，快但可能因為 cost model 不準而次優

Debug 時你可以 fallback 到 serial 版本，隔離「這是 schedule bug 還是 codegen bug」。

**教訓**：任何**優化**性質的 pass，都應該提供一個 correct-but-slow 的 baseline 版本。這是你 debug 時最好的朋友。

### 6. Environment variables 控制優化

TileIR 有一組環境變數控制 TMA 偏好、swizzle mode、async delays 等等。這種**用 env var 控制 pass 行為**的做法看起來土，但實務上非常有用：

- 不用重新 build compiler 就能試不同組合
- 可以在 benchmark script 裡 sweep
- Regression 出現時可以快速二分找到罪魁禍首

**教訓**：compiler 的每個「可決策點」都值得暴露為 flag/env。工程實踐上你會感謝過去的自己。

---

## 八、NVIDIA 為什麼開源？

這是最有趣的產業問題。CUDA moat 一直是 NVIDIA 的護城河，PTX 之上的抽象層歷來封閉。為什麼 2026 年會把這一層打開？

我的解讀是三個力量交會的結果：

### 8.1 Triton / Mojo / IREE 帶來的壓力

過去五年，Triton、Mojo、IREE 這些 tile-centric compiler 快速成熟。它們都在做「越過 CUDA C++ 直接生 PTX」的事情。這對 NVIDIA 是雙面刃：

- **好處**：AI 圈的人不用寫 CUDA 也能吃到 GPU，市場擴大
- **壞處**：這些 DSL 的中間層（TritonGPU、IREE 的 hal dialect、Mojo IR）**不歸 NVIDIA 管**，NVIDIA 無法透過中間層強制推自己想推的硬體特性

如果 NVIDIA 什麼都不做，未來 Blackwell / Rubin 的特性能不能被 AI 圈及時吃到，就取決於 Triton / IREE 團隊有沒有空更新自己的中間層。這對硬體 vendor 是**戰略上不可接受的失控**。

開源 CUDA Tile IR 是**反守為攻**：不如我自己做一個 tile-centric IR、確保它比 TritonGPU 更好、把所有 DSL 拉過來用我的。這樣新硬體特性我第一時間支援，DSL 自動受惠。

### 8.2 Google TPU / AWS Trainium 的競爭

Google TPU 有 XLA + StableHLO，AWS Trainium 有 Neuron compiler，兩者都提供**架構無關的中間 IR**給 JAX / PyTorch 等前端接。這讓 TPU / Trainium 對前端友好度**追上了 CUDA**。

NVIDIA 的護城河一直是 CUDA C++ + cuBLAS/cuDNN 這種**應用層堆積**，但在 tile-centric 中間層這一環，反而落後於對手。開源 TileIR 是補這個洞。

### 8.3 內部合理化

從純技術角度，NVIDIA 內部有太多重複的 tile-level lowering 系統（cuBLAS、CUTLASS、cuDNN 各自維護）。統一到 TileIR 之後：

- 一次修 bug，多個產品受惠
- 新硬體 support 一次做完
- Compiler 團隊的資源可以集中投入

「開源」在這裡有一個微妙的動機：**開源後有社群幫你 debug、幫你寫 test、幫你維護 doc**。對已經是 de facto standard 的東西，開源反而降低內部維護成本。

### 8.4 moat 有沒有真的被拆掉？

沒有。真正的 moat 是**硬體 + PTX + ptxas 這條鏈**。TileIR 是 PTX 之上的一層，PTX 依然是 NVIDIA 專有、ptxas 依然是黑盒、SASS 依然沒有公開文件。競爭對手 CPU / GPU 廠商即使把 TileIR 拿走，也生不出來能跑的 kernel——因為最終 SASS 只有 NVIDIA 硬體認得。

換句話說，**NVIDIA 開放的是「上游介面」，收緊的是「下游護城河」**。這是很聰明的做法。

---

## 九、給 Adam 的行動建議

寫這篇的另一個目的是給我協力的 Adam 幾個具體的下一步。他正在朝 NVIDIA compiler team 布局，TileIR 開源這件事對他個人的意義：

### 9.1 立刻做的事

1. **clone 兩個 repo**：`NVIDIA/cuda-tile` + `NVIDIA/tensor-ir`，build 起來能跑 `tileiras`
2. **跑一個 minimal example**：用 TensorIR 寫一個 matmul，看 `--emit-mlir --print-after-all` 印出來的 pass 前後 IR
3. **挑一個 pass 讀原始碼**：推薦 `tileas-generate-schedule` 或 `tileaa-layout-assignment`，這兩個是最有含金量的
4. **對照 Triton pipeline**：跑 `TRITON_ENABLE_TILE=1` 的 Triton kernel，看它產生的 cuda_tile IR 長什麼樣

### 9.2 面試素材

TileIR 是**極好的面試話題**：

- 「講一個你研究過的 compiler 專案」→ 這是 NVIDIA 自家的，聊起來 relevance 拉滿
- 「dialect design 的哲學」→ 三層 dialect 是教科書答案
- 「pass ordering」→ 46 個 pass 的順序你能講出邏輯，這是 senior 級的訊號

準備一份 5 分鐘 whiteboard talk，題目就叫「CUDA Tile IR 的三層 dialect 與 pass pipeline」。

### 9.3 中期投入

如果真的要投入 compiler career，可以考慮：

1. **給 `NVIDIA/cuda-tile` 或 `NVIDIA/tensor-ir` 貢獻**：修個 doc typo 開始、慢慢到修 bug、到加 pass。GitHub 上的貢獻紀錄是 compiler 職缺最強的證明
2. **寫技術 blog 拆 TileIR**：這個 blog 就是我在幫你做的一部分，你自己也可以寫。有中文技術圈 + NVIDIA 官方認可的東西，能見度很好拉
3. **選一個具體 pass 做 case study**：例如「TileIR 的 async pipeline schedule 怎麼推」，這種深度分析 paper 或 blog 都是簡歷金

### 9.4 不要做的事

- **不要一次讀完 46 個 pass**：這會 burn out。挑一個 sub-pipeline（例如 layout assignment 三部曲）深挖
- **不要跟 Triton 二選一**：兩個是**互補**的（Triton 用 TileIR 當 backend），會 TileIR 對 Triton 面試也加分
- **不要把 TensorIR 當生產工具**：它現在還不是。當學習材料用最好

---

## 十、總結

**CUDA Tile IR 的開源，是 GPU compiler 從「thread-centric、廠商私有中間層」向「tile-centric、標準化 MLIR 生態」演化的關鍵一步。**

技術上，它示範了三層 dialect + 46 pass 的漂亮分層；工程上，它把過去要 CUTLASS 手寫幾百行的 pipelining / layout 自動化；產業上，它是 NVIDIA 對 Triton / Mojo / IREE 這波 tile-centric 浪潮的官方答案。

對 compiler 學習者，這是**極少見的、由第一手 GPU 廠商公開的 production-grade IR**。對 DSL 作者，這是 Blackwell 以後**新硬體特性的最快通路**。對 NVIDIA 自己，這是**把 GPU 編譯器中間層標準化權握在手裡**的戰略布局。

我不知道 CUDA Tile IR 五年後會不會成為 GPU 界的 LLVM IR。但至少現在——2026 年 9 月，這是任何嚴肅看待 GPU 編譯的人都得認真讀的東西。

---

## 參考連結

- [NVIDIA/cuda-tile — CUDA Tile IR (Apache 2.0)](https://github.com/NVIDIA/cuda-tile)
- [NVIDIA/tensor-ir — TensorIR frontend (Apache 2.0)](https://github.com/NVIDIA/tensor-ir)
- [NVIDIA TileIR Internals: from CuTile to MLIR/LLVM to SASS — Henry Zhu, 2026](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)
- [Advancing GPU Programming with the CUDA Tile IR Backend for OpenAI Triton — NVIDIA Developer Blog](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
- [CUDA Tile IR: Lessons from a Tile-Centric CUDA Dialect for MLIR — LLVM Dev Meeting 2026-04](https://llvm.org/devmtg/2026-04/slides/mlir/mlir_springer.pdf)
- [NVIDIA CUDA Tile IR Open-Sourced — Phoronix](https://www.phoronix.com/news/NVIDIA-CUDA-Tile-IR-Open-Source)
- [Nvidia Open-Sources CUDA Tile IR: Did They End the Moat? — byteiota](https://byteiota.com/nvidia-open-sources-cuda-tile-ir-did-they-end-the-moat/)
- [Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs — TensorRT LLM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs)
