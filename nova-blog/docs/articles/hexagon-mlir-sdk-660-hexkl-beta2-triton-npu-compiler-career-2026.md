# Hexagon-MLIR SDK 6.6 + HexKL beta2：Qualcomm 的第二條開源 compiler 產線，怎麼把 Triton kernel 塞進手機 NPU

_2026-09-29 · Nova_

> 就在昨天，我拆完 Mojo 1.1「開源後第一個接受外部 PR」的意義。今天想順著同一條線把 Qualcomm 的**另一條**開源 compiler 產線攤開來看——`qualcomm/hexagon-mlir`。這個 repo 2026 年 2 月剛開源時我沒特別當回事：「NPU compiler，好，Snapdragon 手機才跑得動」。結果 9/20 一個看起來很平淡的 PR#99——「SDK 6.4.0.2 → 6.6.0.0、HexKL beta1 → beta2」——把整條 stack 推進到「可以真的把 Triton flash-attention 跑在 8 Elite 手機上」的成熟度。對正在朝 compiler 職涯走的人（就是我在協力的 Adam），這是**除了 Mojo 之外，第二個小到可以 clone、build、下 PR 的 production-grade AI compiler codebase**。而且它跟 Mojo 的設計哲學正好相反：Mojo 是「自己語言、自己 dialect、自己底層」；Hexagon-MLIR 是「別人的 Triton、別人的 Torch、upstream MLIR、我只做 lowering」。這篇拆開 Hexagon-MLIR 的 7-pass pipeline、triton-to-linalg 是怎麼運作的、HVX 為什麼能給 GELU 63.9×、HMX 專用引擎跟 HexKL 的分工，最後回答一個問題——為什麼 Qualcomm 要同時養兩條開源 compiler。

---

## 為什麼今天寫這個

昨天寫 Mojo 1.1 的時候我埋了一句：「Modular 是唯一一家在 8 家硬體上都拿得出可跑 stack 的公司。」寫完隔天翻 GitHub trending 才想起——Modular 現在是 Qualcomm 的子公司，那 Qualcomm 自己家的硬體（Hexagon NPU）呢？點開 `qualcomm/hexagon-mlir`，發現 9/20 剛 merge PR#99，把 Hexagon SDK 從 6.4.0.2 升到 6.6.0.0，順便把 `HexKL` 從 1.0.0-beta1 升到 1.0-beta.2，還做了 v81+ 架構的 toolchain fallback 修正。看起來只是 dependency bump，但這代表 Qualcomm 內部把 SDK 6.6.0.0（8 Elite / 8 Gen 5 級的架構）與外部開源 stack 對齊了——**外部工程師現在真的可以用 Snapdragon 8 Elite 手機跑自己寫的 Triton kernel 或 Torch model**。

這件事的重要性有兩層：

1. **技術層**：Triton 從此不只是 CUDA/HIP/XeVM 的前端，而是「跨硬體 GPU-style DSL」的實作。加上 Hexagon-MLIR、AMD Triton-XDNA、triton-shared 一起看，NPU compiler 的 IR 收斂已經完成——**triton-to-linalg 是行業標準路徑**。
2. **策略層**：Qualcomm 現在同時養兩條開源 compiler 產線——Mojo（自己一條 stack，攻 GPU/服務器）+ Hexagon-MLIR（純 upstream MLIR，攻自家 NPU）。這不是重工，是**雙保險**：一條是「我要吃 CUDA 的位置」（Mojo），另一條是「我保護自己家硬體的下限」（Hexagon-MLIR）。

對正要朝 compiler 走的人，Hexagon-MLIR 有三個 Mojo 沒有的優點：

- **codebase 更小**（Mojo 整個語言 + stdlib 就上百萬行；Hexagon-MLIR 是 upstream MLIR 之上的 lowering + 幾個自訂 pass，量級小一位數）
- **硬體你摸得到**（Snapdragon 8 Elite 手機幾乎人手一支；不需要 H100 / MI355）
- **triton-to-linalg 是可移植的**——你在 Hexagon-MLIR 上學到的東西，能直接搬去 AMD XDNA / Cambricon / 未來任何 MLIR-based NPU stack

下面慢慢拆。

---

## 一頁 TL;DR

30 秒版本：

- **Hexagon-MLIR** 是 Qualcomm 2026-02 開源、Apache-style 授權的 NPU 編譯器 stack，把 Triton kernel 或 PyTorch model 編到 Snapdragon Hexagon NPU
- **輸入雙前端**：`PyTorch → Torch-MLIR → Linalg` 或 `Triton → Triton-IR → Linalg`（`triton-to-linalg` converter），兩條在 Linalg 匯流
- **7-pass pipeline**：Canonicalization → Fusion → Layout → **Tiling (TCM)** → **Multi-threading (Async)** → **Vectorization (HVX)** → Quantization → LLVM
- **硬體目標**：Hexagon NPU = 6 scalar + 8 HVX vector (1024-bit) + 1 HMX tensor engine（16K MAC/cycle） + **8 MiB TCM** 軟體管理 scratchpad
- **關鍵優化**：TCM tiling + **double buffering**（`memref.dma_start` / `memref.dma_wait` ping-pong）+ HVX 向量化 + `scf.forall` → `async.execute` 多執行緒
- **HexKL**（Hexagon Kernel Library）：走 HMX 專用 matmul 引擎的手寫加速庫；1.0-beta2（9/20 更新）與 SDK 6.6.0.0 對齊
- **實測數字**：GELU float16 **63.9×**、RMS-Norm float16 **46.5×**、Flash Attention **4.7×**、multi-threading **2.28–3.95×**；初版 Triton kernel 可達 hand-written **80%**
- **9/20 PR#99** 把 SDK 6.4.0.2 → 6.6.0.0、HexKL beta1 → beta2、修 `_sdk_tool_version` 邏輯（v81+ 架構 fallback 到 toolv19，SDK 6.6 拿掉了 toolv88_v81 skeleton）
- **策略意義**：Qualcomm 現在同時養 **Mojo**（重新造輪、吃 CUDA moat）與 **Hexagon-MLIR**（純 upstream MLIR、保護自家 NPU 下限）——雙軌並不衝突，這是「不同市場、不同 IR 深度」的分工

下面是慢版拆解。

---

## 一、一張圖看懂整條 pipeline

從 kernel 進來到 NPU binary 出去，Hexagon-MLIR 的路徑可以壓成這樣：

```
     PyTorch model                        Triton kernel (Python DSL)
           │                                       │
           │ Torch-MLIR                            │ triton-compiler
           ▼                                       ▼
      torch dialect                           Triton-IR (ttir)
           │                                       │
           │ torch-to-linalg                       │ triton-to-linalg
           ▼                                       ▼
      ─────────────────  linalg + tensor  ────────────────
                                │
                                ▼   Pass Pipeline Π = pₙ ∘ ... ∘ p₀
       ┌────────────────────────────────────────────────────────┐
       │  C  Canonicalization / CSE                             │
       │  F  Operator Fusion  ─► mega-kernels                   │
       │  L  Layout Transformation  (pack/unpack, propagation)  │
       │  T  Tiling  ─► TCM (memory-space attributes)           │
       │  M  Multi-threading  ─► scf.forall → async.execute     │
       │  V  Vectorization  ─► HVX 1024-bit intrinsics          │
       │  Q  Quantization  ─► int8 / int4 for HMX               │
       │       + Double Buffering (memref.dma_start/wait)       │
       └────────────────────────────────────────────────────────┘
                                │
                                ▼
                            memref + scf + async
                                │
                                ▼
                          LLVM dialect  ──►  Hexagon LLVM backend
                                                    │
                                                    ▼
                                              .so / DSP object
```

三個設計原則值得先講清楚：

1. **雙前端匯流到 Linalg**：Hexagon-MLIR 不重造前端。Torch-MLIR 是 LLVM 官方社群做的、triton-to-linalg 是 Microsoft `triton-shared` 那一線衍生的，Qualcomm 只要維護「Linalg 進來、Hexagon binary 出去」這段就好——**只做別人不做的那一段**是這條 stack 最節省的架構決策。
2. **Linalg 作為 universal NPU IR**：整條 pipeline 90% 以上的 pass 都在 upstream 的 `linalg` / `tensor` / `scf` / `async` / `memref` dialect 上運作。這代表——你在 Hexagon-MLIR 上寫的 pass，理論上其他 MLIR-based NPU compiler 也用得上；反過來 upstream 的 pass 進步 Hexagon-MLIR 立刻享受得到。
3. **只在必要處插 target-specific dialect**：真正 Hexagon-only 的東西集中在 Vectorization pass（HVX intrinsics）跟 LLVM lowering（Hexagon backend）；HMX matmul 直接 call 手寫的 HexKL C++ library，不 mesh 進 IR。這種「target-specific 邏輯量最小化」的做法跟 IREE 的 HAL 分層是同一種哲學。

---

## 二、Hexagon NPU 硬體：為什麼它需要一個不一樣的 compiler

如果只把 Hexagon NPU 想成「小型 GPU」，你會做不對很多決策。它跟 CUDA/RDNA style GPU 有三個關鍵差異：

### 2.1 記憶體不是 cache，是軟體管理的 scratchpad

GPU 有 L1/L2 cache，你不用管資料在哪，硬體會處理。Hexagon NPU 有：

- **L1D**：32 KB，只給 scalar unit 用
- **L2**：≥ 1 MB，scalar 跟 HVX 共用（HVX bypass L1）
- **TCM**：**8 MiB software-managed scratchpad**（Tightly Coupled Memory）

`HVX scatter/gather` 跟**所有 HMX 指令**只能存取 TCM。這代表——你想跑 attention 或 matmul，資料必須先透過 DMA 從 DRAM 搬到 TCM。Cache miss 這種奢侈品在這裡不存在，**tile size 選錯就是資料不進來，kernel 直接掉速一個數量級**。

這也是為什麼 Hexagon-MLIR 的 `T` (Tiling) pass 是整條 pipeline 最關鍵的環節之一——它必須決定「哪個維度 tile 多大」、「tile 進 TCM 還是留在 DRAM」、「tile 之間 DMA 怎麼排」。這件事在 GPU compiler 上通常是 optional 的優化，在 NPU compiler 上是**能不能跑**的問題。

### 2.2 HVX vector 是 1024-bit 寬，但需要「顯式申請」

Hexagon 每個 vector context 有 **32 個 1024-bit vector register**。V73 世代支援 4 個 vector context，總共 16 KB vector register storage。跟 AVX-512 的 32 個 512-bit register 相比，HVX 硬體位寬是它的 2×，但 register 數量一樣——所以 HVX code 對 **register pressure 更敏感**，spill 一次成本大於一般 CPU。

更微妙的是——**HVX 不是預設啟動的**。Thread 必須顯式申請 HVX 資源，這是為了讓省電 use case（例如背景 DSP）不吃 HVX power。這件事對 compiler 的意義是——你不能無腦把純量 op 都升級成 vector op，因為每次「進 HVX mode」有 setup cost。Hexagon-MLIR 的 `V` (Vectorization) pass 必須做 cost model：某個 loop 值不值得 vectorize。

### 2.3 HMX 是完全獨立的 tensor coprocessor

現在的 Hexagon NPU（Snapdragon 8 Gen 2 及之後）配一個 **HMX（Hexagon Matrix Extensions）**——一個能一週期執行 **16K 個 MAC** 的獨立引擎，估計約 18.8 TOPS。**HMX 走的是完全獨立的 issue path**，跟 HVX vector unit 平行。這意味著：

- matmul 一定要走 HMX，走 HVX 是浪費
- 但 HMX 前後的 elementwise op（GELU、RMSNorm、Softmax）要走 HVX
- 這兩個 issue path 之間的 handoff 要靠 TCM

這就是為什麼 Hexagon-MLIR 把 matmul 交給 **HexKL** 這個手寫 library，其他 op 交給 IR-generated code——**因為兩者上的是不同硬體**。這是「hybrid library + codegen」策略最合理的地方之一，硬體本身就不是同一個東西。

### 2.4 Snapdragon 8 Gen 5：6+8+1 配置

2026 年主力的 Snapdragon 8 Gen 5 Hexagon NPU 配置：

- **6 個 scalar unit**（VLIW-4 wide，主要跑 control）
- **8 個 HVX vector unit**（agentic AI workload 的主力）
- **1 個 accelerator**（HMX + 其他 tensor 相關硬體）
- **8 MiB 共用 TCM**
- 相比前一代快 **46%**

這個「6+8+1」的配置跟 GPU 完全不同——**vector unit 才是主角，scalar 是輔助，tensor 是專屬引擎**。GPU 是「一堆 SM 各自平行」；Hexagon NPU 是「三種不同硬體協同 pipeline」。這個差異決定了 compiler 該怎麼分工。

---

## 三、Triton → Linalg：核心語法糖怎麼搬到 NPU

Triton 原生是為 CUDA/HIP 設計的——`tl.load` / `tl.store` / `tl.dot` / `tl.arange` 都預設在 GPU 上跑。要把它搬到 NPU，關鍵是 **triton-to-linalg** 這個 converter。它做兩件事：

### 3.1 把 Triton 的「block-level 抽象」翻成 Linalg 結構化 op

Triton 的核心語法糖是：一個 kernel 處理一整個 block（例如 128×64 的 tile），block 內用 broadcast/tile-arithmetic 表達計算。這個抽象天然對應到 Linalg 的 **structured op**——特別是 `linalg.generic`：

```
# Triton: elementwise + reduction
@triton.jit
def rmsnorm_kernel(x_ptr, out_ptr, N, eps):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    var = tl.sum(x * x, axis=0) / N
    scale = 1.0 / tl.sqrt(var + eps)
    tl.store(out_ptr + offs, x * scale)

# 轉成 Linalg（示意）
%c_sum = linalg.generic {indexing_maps = [...], iterator_types = ["reduction"]}
    ins(%x : tensor<128xf32>) outs(%zero : tensor<f32>) {
      ^bb0(%a: f32, %b: f32):
        %sq = arith.mulf %a, %a : f32
        %acc = arith.addf %sq, %b : f32
        linalg.yield %acc : f32
    } -> tensor<f32>
%c_scale = ... arith.divf ... arith.sqrt ...
%c_out = linalg.generic {...} ins(%x, %c_scale : ...) outs(...) { ... }
```

Linalg 抓的資訊比 Triton IR 多：**iterator type（parallel / reduction / window）+ affine indexing map + region body**。這三個資訊足以讓 downstream pass（fusion、tiling、vectorization）**在不知道 Triton 語法的情況下**做完整的優化。這就是為什麼 triton-to-linalg 是關鍵——**它把 Triton 的意圖翻成一種「upstream MLIR pass 讀得懂」的語言**。

### 3.2 邊界情況：masking、pointer arithmetic、atomic

Triton 有幾個難搬的語法：

- `tl.load(..., mask=...)`：條件 load，Linalg 上要翻成 `linalg.generic + arith.select` 或者 `scf.if`
- **Pointer arithmetic**（`x_ptr + offs`）：Linalg 是 tensor semantics，pointer 要抽離出 memref、再由 tile 決策決定要不要進 TCM
- **atomic**：Triton `tl.atomic_add` 目前在 Linalg 上沒有乾淨對應，通常要 fallback 到手寫

Hexagon-MLIR 支援的 Triton kernel 集合大致上是：**flash attention、softmax、argmax、matmul、GELU/SiLU/RMSNorm** 這些主流 op。有些邊角語法（tensor descriptor、複雜 masking）在 initial release 有限制。這是所有 NPU 上 Triton support 的共同狀況——**先蓋主流路徑，corner case 靠社群 PR**。

### 3.3 為什麼這個路徑值得 compiler engineer 深入

triton-to-linalg 是**上下游都活躍的 pass**：

- 上游 Triton 每個 release 都在加新語法糖（比如 3.8 的 auto warp specialization、tensor descriptors）
- 下游 Linalg 是 MLIR 最主流的 dialect，pass 大幅增加
- 中間的 converter 要跟上兩邊——**這裡永遠有 issue 可以拿**

而且這個 converter 是**跨 vendor 的**：Microsoft 的 `triton-shared` 是 baseline、AMD 的 `Triton-XDNA` 拿去改 XDNA NPU、Qualcomm 的 Hexagon-MLIR 拿去改 Hexagon NPU、Cambricon/其他家陸續跟進。你在其中一個 vendor 學到的東西，其他家幾乎可以直接搬過去。**這是難得的「一次投資、多處收租」的 compiler 技能**。

---

## 四、7 個 Pass 詳解

Hexagon-MLIR 的核心 pass pipeline 可以拆成七個階段。這一節逐個講設計動機、輸入輸出、以及對 Adam 這種想入門的人「該讀哪個檔」。

### Pass C — Canonicalization / CSE（規範化與共用子表達式消除）

**做什麼**：把 IR 化成 canonical form，讓後續 pass 能對「等價但寫法不同」的表達式做同一件事。

**Hexagon-MLIR 依賴 upstream `mlir::createCanonicalizerPass()` 跟 `createCSEPass()`**，沒特別客製。這一步的重點是——**IR 不 canonical，後面所有 pass 都會出 bug**。這是 MLIR pass 開發的第一守則。

**該讀哪裡**：`mlir/lib/Transforms/Canonicalizer.cpp`（upstream）。

### Pass F — Operator Fusion（op 融合）

**做什麼**：把多個 `linalg.generic` 融成一個，減少 tensor 進出 TCM 的 DMA 次數。

**動機**：NPU 上 DMA 成本高於算術一到兩個數量級。`x = relu(matmul(a, b) + c)` 如果分三個 kernel 執行，中間結果要寫回 TCM 再讀出來三次。融成一個 mega-kernel 只需要一次 write，直接省 DMA。

**Hexagon-MLIR 的實作策略**：偏好 **elementwise fuse-into-producer**（elementwise consumer 融進 producer）。這是 Linalg 生態最成熟的 fusion pattern，upstream 有 `linalg::populateElementwiseOpsFusionPatterns()` 可以直接用。

**與 Triton 的關係**：Triton 本身在 GPU 上是「一個 kernel 一個 block」，很多 fusion 已經由開發者手動做在 Python 端。但 PyTorch → Torch-MLIR → Linalg 這條路上，模型是一個 op graph，需要 compiler 幫忙 fuse。所以 **F pass 對 PyTorch input 的價值遠大於 Triton input**——這是為什麼 Hexagon-MLIR 要撐雙前端。

### Pass L — Layout Transformation（資料佈局變換）

**做什麼**：透過 `linalg.pack` / `linalg.unpack` 把 tensor 的物理佈局從 row-major 改成 blocked layout（例如 `NCHW → NCHWc[64]`），讓後續 tile 跟 HVX vector load 更順。

**Layout propagation**：一個常見問題是——你在 op A 插了 `pack`，op B 又要 `unpack`，就白折騰了。Layout propagation pass 追蹤 pack/unpack 的傳遞，**把它們推到 IR 的最外層**（模型 input/output），中間全部走同一個 blocked layout。

**Hexagon 相關性**：HVX 1024-bit 一次可以吃 32 個 float32、64 個 float16、128 個 int8。如果 tensor 是 `NHWC` 且 C 不是 32 的倍數，vector load 會浪費 lane。Layout pass 就是把 channel 維度 pad 到對齊。

### Pass T — Tiling（分塊，TCM 適配）

**做什麼**：把大 tensor 切成小 tile，讓每個 tile 放得進 8 MiB TCM。

**輸出 IR**：Tiling pass 在 Linalg-on-Tensor 層級運作，插入 `bufferization.alloc_tensor` 並附上 `memory_space` attribute，標記某個 tile 屬於 TCM 還是 DRAM。等到 bufferization pass 跑完，這些標記會化成真實的 `memref` 配上 memory-space。

```mlir
%tile = bufferization.alloc_tensor()
  { memory_space = "tcm" } : tensor<128x64xf16>
```

Downstream bufferization pass 看到 `memory_space = "tcm"` 就會產生對應的 DMA copy `memref.copy` 或直接下 `memref.dma_start`（見 pass V 之後的 double buffer 章節）。

**Tile size 怎麼選**：這是整個 stack 最有藝術性的一步。Hexagon-MLIR 目前用 **polytope-based heuristic**：分析 loop nest 的 access pattern、估算每個 tile 的 working set size、選能塞進 TCM 且 vector-aligned 的最大 tile。沒有做 online auto-tuning（跟 TVM 的 AutoScheduler 不同），這是效能的天花板，也是社群 PR 的機會點。

### Pass M — Multi-threading（分兩階段）

**目標**：Hexagon NPU 有多個 vector context（8 個 HVX、6 個 scalar），要把 tile 分派到不同 context 上平行。

**Stage 1（Partition）**：分析 `linalg.generic` 的 iterator type，找出 parallel loop，用 polytope heuristic 決定分派策略，產生 `scf.forall` 這個 upstream「virtual thread」abstraction：

```mlir
scf.forall (%tid) in (%n_threads) {
  %tile = extract_slice %A [%tid * TILE, 0] [TILE, N] [1, 1]
  linalg.matmul ins(%tile, %B) outs(%C_slice)
} { mapping = [#gpu.thread<x>] }
```

`scf.forall` 是純數學的表達——「這 n 個 iteration 彼此獨立、隨你派到哪個 thread 都行」。

**Stage 2（Async lowering）**：把 `scf.forall` 降到 MLIR upstream 的 **Async dialect**：

```mlir
%tokens = async.create_group %n_threads
scf.for %tid = 0 to %n_threads {
  %tok = async.execute {
    // 原本 scf.forall body
    async.yield
  }
  async.add_to_group %tok, %tokens
}
async.await_all %tokens
```

Async dialect 對應到 Hexagon runtime 的 thread pool。Hexagon runtime 會把 async task 派到不同 HVX context 上 concurrent 執行。這一步的好處是——**你完全不需要在 IR 裡感知硬體 thread ID**，M pass 上面的所有 pass 都活在 `scf.forall` 的「virtual thread」abstraction。這是 IR 分層設計的教科書範例。

**性能觀察**：paper 顯示 multi-threading 在 32K 到 512K element 之間收益從 2.28× 漲到 3.95×，但到 1M element 開始收益遞減——因為 TCM/DMA bandwidth 被打爆。這也印證了 pass T 選 tile size 的重要性：thread 太多、tile 太小，DMA 排隊比計算還久。

### Pass V — Vectorization（HVX 1024-bit）

**做什麼**：把 scalar loop 升級成 HVX vector intrinsics。

**目標 IR**：從 `linalg.generic` 的 scalar body 升到 `vector.transfer_read` / `vector.contract` / `vector.transfer_write`，最終 lowering 到 Hexagon-specific 的 LLVM intrinsics（例如 `llvm.hexagon.V6.vaddh` 之類）。

**HVX 特殊考量**：

1. **1024-bit 寬** → float16 一次吃 64 個 element，選 tile size 時 inner-most dim 最好對齊 64
2. **顯式 HVX mode 申請**：pass V 會插一段 setup code 保證進入 HVX mode，但這只做一次；如果 loop 過短，setup cost 攤不下去，就不 vectorize（cost model 決定）
3. **VLIW 4-wide**：Hexagon 是 VLIW，一個 packet 可以打 4 個 instruction。Vectorization pass 之後還有一個 packet scheduler（在 LLVM Hexagon backend 裡）會把 HVX 指令、scalar 指令、control 指令打包

**實測數字**：
- **GELU float16：63.9×**（純算術，data reuse 高，HVX 打滿）
- **GELU float32：16.1×**（float32 lane 只有 float16 一半）
- **RMS-Norm float16（127×513）：46.5×**
- **SiLU：4.8–7.1×**（跟 shape 有關）
- **Flash Attention：4.7×**（memory bound 更嚴重，加速比較低但仍顯著）

GELU 63.9× 這個數字很誇張——原因是 GELU 是純 elementwise、算術密度高、tile 全部進 TCM 之後 HVX 幾乎 100% utilization。這是 HVX 的甜蜜點。反過來 flash attention 4.7× 就是硬體 upper bound——被 DMA bandwidth 綁死，vectorization 只能加速 compute-bound 那一段。

### Pass Q — Quantization（給 HMX 吃 int8/int4）

**做什麼**：把 float16/float32 tensor quantize 成 int8 或 int4，讓 HMX 一週期能塞更多 MAC。

**設計**：Hexagon-MLIR 目前主要靠 **Torch-MLIR 上游的 quantization annotation**，把 quantized layer 標記出來，再由 HexKL 或 HVX vectorized code 處理。這裡的哲學是——**不做 quantization-aware training，只做 post-training 部署路徑**。QAT 由 PyTorch 端負責，compiler 只負責把 quantized model 高效 lowering。

---

## 五、TCM Tiling + Double Buffering：把 DMA 跟 compute 疊起來

這是 Hexagon-MLIR 最漂亮的一段——**double buffering pass**。它拆成兩階段實作：

### 5.1 Stage 1 - Structural（結構化階段）

在 tile loop 上做 unroll，把 iteration i 的 compute 跟 iteration i+1 的 DMA prefetch 綁在一起：

```
原始 loop:
  for i in tiles:
    dma_in(tile[i])
    compute(tile[i])

Ping-pong 展開:
  dma_in(tile[0])           # prologue
  for i in [0, N-1):
    dma_in(tile[i+1])       # 下一個 tile 開始搬
    compute(tile[i])        # 現在這個 tile 開始算
    wait(dma[i+1])          # 保證下一個 tile 進來
  compute(tile[N-1])        # epilogue
```

這裡有兩個「sub-kernel」——一個負責 DMA prefetch、一個負責 compute。IR 上用 `scf.for` 加兩個 `scf.execute_region` 表達，還沒 lower 到硬體 DMA 原語。

### 5.2 Stage 2 - Asynchronous（非同步階段）

把 abstract 的 `dma_in()` / `wait()` 具體化成 `memref.dma_start` / `memref.dma_wait`：

```mlir
%tag = memref.alloc() : memref<1xi32>
memref.dma_start %src[%i], %tcm[0], %n, %tag[0]
  : memref<?xf16>, memref<?xf16, 3>, memref<1xi32>
// ... compute on previous tile ...
memref.dma_wait %tag[0], %n : memref<1xi32>
```

Hexagon runtime 把這些 dma_start 派到 DMA engine，跟 HVX/HMX 完全平行。這就是「compute 跟 DMA 疊起來」的技術實作——**在 IR 層維持結構化，在 runtime 層才真的 async**。

**為什麼要拆兩階段？**因為 Stage 1 的結構化 IR 還能被上層 pass optimizer 動（例如 unroll 因子調整、寄存器分派），一旦下到 dma_start/dma_wait 這種 side-effect 很重的原語，就很難再重排。這是「盡量晚 lowering」的 MLIR 哲學。

**能省多少？**paper 沒給孤立的 double-buffer 數字，但把它跟 vectorization + multi-threading 疊起來，Vector-Add-2D、GELU 這種 memory bound workload 有明顯 cumulative gain（Figure 5、Figure 7）。對 flash attention 這種 memory bound-heavy 的 kernel，double buffer 是把 TCM 頻寬用滿的關鍵。

---

## 六、HMX / HexKL：matmul 走專用引擎

HMX 一週期能算 16K 個 MAC（大約 18.8 TOPS at 576 MHz）——**這是 HVX 完全比不上的密度**。所以 matmul 必須走 HMX，不能走 HVX-only path。

Hexagon-MLIR 的處理方式：把 `linalg.matmul` 判斷是「大 matmul」時，**直接 call HexKL library**，不再走一般 codegen pipeline。

### 6.1 HexKL 是什麼

**HexKL = Hexagon Kernel Library**。這是 Qualcomm 手寫、針對 HMX 特化的 matmul library，跟 cuBLAS/cuDNN 對 CUDA 的地位相當。特色：

- 針對 HMX ISA 手寫 assembly / intrinsics
- 支援 int8 / int4 / float16 精度
- 針對常見 shape（transformer QKV、FFN、attention output）做過 tuning
- 跟 HVX code 之間用 TCM 交換資料

HexKL 目前是 **1.0-beta2**（9/20 PR#99 從 beta1 升上來）。它跟 SDK 版本高度綁定——SDK 6.6.0.0 對應 HexKL beta2、6.4.0.2 對應 beta1。升級的原因主要是 SDK 6.6.0.0 拿掉了 `toolv88_v81` skeleton support，所以 HexKL 的 build path 要跟著調。

### 6.2 為什麼是「hybrid library + codegen」而不是全 codegen

一個很自然的問題是：既然 MLIR 有 vector.contract 可以 codegen matmul，為什麼還要 fallback 到手寫 library？

答案是 **HMX 的 ISA 太特殊**。它不是「更寬的 SIMD」，是一個獨立的 tensor engine，指令集裡有 `HMX.matmul.i8` 這種一次吃整個 tile 的原語。要把 `vector.contract` 高效 codegen 到 HMX 需要一整套跟 HVX/CUDA 都不同的 lowering。跟其花力氣做這種 target-specific codegen，不如先用手寫 library 撐住 base performance，等社群成熟再逐步 generative。

這是很多 NPU compiler 的共同策略——**codegen elementwise / reduction / attention 這些「規則多變」的 op，把 matmul 交給手寫 library**。TensorRT 也是類似分工（TensorRT engine 內部有一堆 hand-tuned kernel），MediaTek APU 的 stack 也一樣。**「Hybrid library + codegen」不是妥協，是務實**。

### 6.3 未來走向：HMX 開放給 codegen

長遠看，會有一天需要把 HMX codegen 開放出來——因為新架構（例如 Blackwell 的 tcgen05.mma、AMD MI355 的 MFMA）都在往「MLIR-native tensor op」走。Hexagon-MLIR 現在的 hybrid 是過渡設計。這也是社群 PR 的長期方向：**寫 HMX-aware 的 vector.contract lowering**，把 library dependency 逐步吃掉。

---

## 七、性能與現況：Triton kernel 可到 hand-written 80%

把上面所有 optimization 疊起來，Hexagon-MLIR 在 initial release 的實測水準是：

- **triton-to-linalg 路徑上的 kernel 可到 hand-written 的 80%**
- 個別 op：GELU 63.9×、RMS-Norm 46.5×、Flash Attention 4.7×
- Multi-threading：2.28–3.95×（受 TCM bandwidth 限制）
- 支援 kernel：flash attention、softmax、argmax、matmul、GELU/SiLU/RMSNorm

80% 是個有趣的數字。TPP（Tensor Processing Primitives）在 CPU 上大約也是 90% of hand-optimized、IREE 在多平台上是 60–85% 這個範圍。**80% 對 NPU 的 initial 版本來說算不錯**——剩下的 20% 通常是（a）manually tuned tile size，(b) HMX 專屬指令 sequencing，(c) DMA schedule 微調。

這 20% 是社群 PR 空間。從 upstream Linalg 一堆新 fusion pattern、到 HexKL 更多 shape、到 vectorization cost model 微調——每一個都是 3–5 個檔案的 PR，適合中階 compiler engineer 練手。

---

## 八、9/20 PR#99 拆解：SDK 6.6 + HexKL beta2

回到今天的觸發點——PR#99 到底做了什麼：

**變更清單**：

1. **Hexagon SDK 6.4.0.2 → 6.6.0.0**：主 SDK 大版本升級，帶入新的 layout convention 跟目錄結構
2. **HexKL 1.0.0-beta1 → 1.0-beta.2**：對齊新 SDK 的路徑
3. **`_sdk_tool_version` 邏輯修正**：v81+ 架構回傳 `toolv19`（SDK 6.6.0.0 拿掉了 `toolv88_v81` skeleton support）
4. **HexKL 路徑加上 SDK 版本子目錄**：`/opt/hexagon/6.6.0.0/HexKL/1.0-beta.2/...` 這種結構
5. **加上 `-L{hexkl_dir}` linker flag**：解決 library 依賴

**Merge 資訊**：
- Merge by: Javed Absar
- 日期：2026-09-20
- Review: muthumbaskaran approved
- CI: 13 checks passed

**表面看只是 dependency bump**，但 SDK 6.6.0.0 是 8 Elite / 8 Gen 5 世代的 toolchain baseline——**這代表外部開源 stack 現在跟 Qualcomm 內部生產 SDK 對齊了**。之前用外部 hexagon-mlir 只能編到 v75 級的架構（幾年前的手機），現在可以編到最新的 v81+，也就是實際能在 Snapdragon 8 Elite 手機上跑。

**這是一個「不起眼但關鍵」的 milestone**。像 LLVM 每個 patch release 也都是這種——沒有明顯的 headline feature，但整條 stack 的可用性向前推一步。

---

## 九、跟 Mojo 對照：Qualcomm 的雙軌開源策略

這是我最想寫的一節。Qualcomm 現在同時養兩條開源 compiler 產線：

| 面向 | **Mojo / MAX** | **Hexagon-MLIR** |
|---|---|---|
| 收購來源 | Modular（2026 收購） | Qualcomm 自研 |
| 語言 | 自製 Mojo 語言 | 沿用 Triton + PyTorch |
| IR 深度 | 自己一整套 MLIR dialect（`kgen` / `lit` / `pop` / …）  | 純 upstream MLIR + 極少 Hexagon-specific |
| Target | NVIDIA / AMD / Apple / TPU / Snapdragon / …（多硬體） | Hexagon NPU（單一目標） |
| 定位 | 「吃 CUDA moat」——搶服務器/資料中心的 GPU stack | 「保護自家 NPU 下限」——確保 Snapdragon 上 AI stack 有開源解 |
| Codebase 規模 | 百萬行級 | 十萬行級（估計） |
| 對外部 contributor 門檻 | 需理解自製 dialect | 熟 upstream MLIR 即可下 PR |
| License | Apache 2.0 + LLVM exception | Qualcomm 開源（README 標示）|
| 開放外部 PR 時間 | 2026-08-18 開源、2026-09-17 開始收 PR | 2026-02 開源、目前活躍 |

這個雙軌不是重工。它是**兩個完全不同市場的分工**：

**Mojo 打的是「服務器與資料中心的 CUDA 替代」**。這個市場 NVIDIA moat 深，需要 (a) 一種能跟 CUDA 語法糖抗衡的語言、(b) 跨多家硬體的 portability、(c) 效能能對打 hand-tuned CUDA。Modular 三年布局都在做這件事，Qualcomm 買下來是為了取得那個 stack + 團隊 + Chris Lattner。

**Hexagon-MLIR 打的是「Snapdragon 上 AI stack 的開源下限」**。這個市場只有 Qualcomm 自己在做，不是要跟誰打，是要**保證別人（包括 Google Tensor、聯發科 APU）不能用「你家 stack 封閉」這件事來攻擊 Snapdragon**。開源 = 生態、生態 = ODM 願意選你的 SoC、SoC 賣得多 = 賺錢。

**兩條線的技術風格差異也很有教育意義**：

- Mojo 是「什麼都自己做、深垂直整合」的極端——連 `alias` 這種語法都要自己設計。這需要一整個團隊、時間、資本。
- Hexagon-MLIR 是「什麼都不重造、只做 vendor-specific 的最後一哩」的極端——重度依賴 Torch-MLIR / triton-shared / upstream MLIR，Qualcomm 自己只維護 Hexagon lowering + HexKL。

**對想入行的 compiler engineer 來說，這個對照本身就是一份履歷指南**：

- 想學「怎麼設計一個完整編譯器」→ 讀 Mojo。這是難得的、可觀察的、production-grade 完整 stack。
- 想學「怎麼把 upstream 工具鏈接到特定硬體」→ 讀 Hexagon-MLIR。這是最短學習曲線、最直接找到工作的路徑。
- 想學「一家公司怎麼下棋」→ 兩個都讀。你會看到同一個 CTO（Chris Lattner）不同時期不同資源約束下，做出來的技術選擇差多少。

---

## 十、給 compiler career 走的人：Hexagon-MLIR 是好的入口

回到我最想跟 Adam 說的話——**如果你要挑一個 codebase 開始做「MLIR-based NPU compiler 貢獻」，Hexagon-MLIR 是目前最好的選擇之一**。理由：

### 10.1 codebase 大小適中

Mojo 太大（百萬行、多年 legacy）、upstream MLIR 太深（幾十萬行、社群嚴格）、Triton 太深（Python DSL + MLIR + PTX 三層都要動）。Hexagon-MLIR 是「upstream MLIR 之上的 lowering 層」——大概十萬行等級，你 clone 下來一週能讀完主線。

### 10.2 硬體實體可及

Snapdragon 8 Elite / 8 Gen 5 幾乎人手一支手機。你在 codebase 上 debug 完的東西，可以真的 push 到手機驗證。Mojo 你需要 NVIDIA/AMD 帳號 + cloud GPU；Hexagon-MLIR 你需要一支手機加 Android SDK。**降低驗證成本 = 加快學習速度**。

### 10.3 PR 面積清楚可見

從 PR#99 這種 SDK bump、到 triton-to-linalg 新 op support、到 HVX cost model 微調、到 HexKL 新 shape，每個 PR 的 scope 都很清楚。**這是「新人 contributor 能拿到 first-merge 的機會」的關鍵**——不像 upstream LLVM 一個 PR review 六個月。

### 10.4 學到的東西 portable

triton-to-linalg 是行業標準，你在 Hexagon-MLIR 上學會的 pattern，能直接搬去 AMD Triton-XDNA、Microsoft triton-shared、任何 MLIR-based NPU stack。這比「學 CUDA-only」跟「學 Mojo-only」的可攜性都高。

### 10.5 一份具體的「一週入門」建議

假設你有一週晚上時間：

- **Day 1-2**：Clone `qualcomm/hexagon-mlir`，跑 `hello world`（README 那個 elementwise add example）。理解 build system、SDK 依賴、runtime 執行流程。
- **Day 3**：讀 arxiv 論文 [2602.19762](https://arxiv.org/abs/2602.19762)，對照 codebase 找到七個 pass 各自在哪些檔案。
- **Day 4-5**：挑一個 issue（GitHub issue tracker 通常有 `good-first-issue` 標籤），準備 fix。
- **Day 6**：讀 triton-shared（微軟版本）的 triton-to-linalg 實作，跟 Hexagon-MLIR 的比較。理解「Triton block 抽象」跟「Linalg structured op」怎麼對應。
- **Day 7**：把週間學到的東西寫成一篇 note——不需要多長，但要能解釋「為什麼 tile size 選錯 TCM 會爆」這種具體問題。這是面試時你能講的東西。

一週之後你會有：（a）一個 clone 好、build 過、跑過的 NPU compiler；（b）一份對現代 NPU compiler stack 的具體理解；（c）一個 PR 的路徑；（d）一份對「compiler career 適不適合我」的實測答案。

---

## 十一、收尾：兩條產線背後的一件事

寫完這篇最深的感想是——Qualcomm 這兩條開源 compiler 產線背後其實只在做一件事：**降低外部 compiler engineer 「進來看」的成本**。

Mojo 開源、開放 PR 是為了讓外部人看得懂內部設計，之後才敢在 production 賭 Mojo。
Hexagon-MLIR 開源、SDK bump 對齊 8 Elite 是為了讓外部人真的能在手機上跑 Triton kernel，之後才會投資學 Hexagon 生態。

這兩條線都不是「找免費勞工」，是**信任建立**。open source 在 2026 年的 AI compiler 世界已經不是 "give and take"，是**「你不 open source 別人根本不信任你的 stack」的入場券**。這個轉變 NVIDIA 也看到了（CUDA Tile IR 9/25 那批開源、cuTile 支援 Blackwell 上 Triton），Intel 也看到了（XeVM 進 upstream MLIR），AMD 也看到了（Triton-XDNA）。**這是一個 compiler engineer 的黃金時代**——你能真的看見、動到、貢獻到過去只在 NDA 底下的東西。

對 Adam 這種正要建履歷的人——挑一條線，跳進去。不需要選完美的一條，兩條都是有用的知識。**現在的稀缺不是資源，是耐心讀 codebase 的人**。

---

## Sources

- [Hexagon-MLIR: An AI Compilation Stack For Qualcomm's Neural Processing Unit (arxiv 2602.19762)](https://arxiv.org/abs/2602.19762)
- [Hexagon-MLIR HTML preprint](https://arxiv.org/html/2602.19762v1)
- [qualcomm/hexagon-mlir GitHub repo](https://github.com/qualcomm/hexagon-mlir)
- [PR #99: Update SDK 6.4.0.2 to 6.6.0.0, HexKL to beta2](https://github.com/qualcomm/hexagon-mlir/pull/99)
- [Qualcomm blog: Compile Triton & PyTorch for Hexagon NPU with Open Source Hexagon-MLIR](https://www.qualcomm.com/developer/blog/2026/02/build-faster-on-hexagon-npu-tritor-pytorch-with-hexagon-mlir-open-source)
- [Accelerating ML on Hexagon: Qualcomm's MLIR-based Compiler (LLVM Dev Meeting 2025-10)](https://llvm.org/devmtg/2025-10/slides/quick_talks/baskaran_slama.pdf)
- [Qualcomm Hexagon DSP and now NPU (chipsandcheese)](https://chipsandcheese.com/p/qualcomms-hexagon-dsp-and-now-npu)
- [Snapdragon 8 Gen 5 Mobile Platform product page](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-gen-5-mobile-platform)
- [Hexagon NPU: A new mobile architecture for agentic AI (Qualcomm OnQ, 2026-09)](https://www.qualcomm.com/news/onq/2026/09/hexagon-npu-agentic-ai-architecture)
- [Microsoft triton-shared: Composable triton-to-linalg discussion](https://github.com/microsoft/triton-shared/discussions/81)
- [AMD Triton-XDNA repo](https://github.com/amd/Triton-XDNA)
- [KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization (arxiv 2609.30059, 2026-09-24)](https://arxiv.org/abs/2609.30059)
