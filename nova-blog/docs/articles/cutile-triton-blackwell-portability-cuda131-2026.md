---
title: "cuTile 是為了打 Triton 而生的嗎？獨立評測揭露 NVIDIA 新 Tile DSL 的效能曲線與生態算計"
date: 2026-09-17
tags:
  - GPU
  - CUDA
  - Compiler
  - Triton
  - Blackwell
  - MLIR
  - AI-Compiler
summary: >
  arXiv:2604.23466 的獨立評測告訴我們：cuTile 在 Blackwell B200 上以 60 行 Python
  跑贏 FlashAttention-2 2.5×，但在 sm_120 掉到 FA-2 的 53%；Triton 跨平台守在 cuBLAS
  的 62-101%。CUDA 原始團隊成員 Nicholas Wilt 直接開嗆「cuTile 就是為了對付 Triton」。
  這篇拆解：Tile DSL 到底是什麼、cuTile 為什麼 CUDA 13.1 不支援 Hopper、
  以及 compiler 工程師應該如何看這場「tile 抽象」路線之爭。
---

# cuTile 是為了打 Triton 而生的嗎？獨立評測揭露 NVIDIA 新 Tile DSL 的效能曲線與生態算計

> **TL;DR**
>
> 1. NVIDIA CUDA 13.1（2025 年 12 月）推出 **cuTile**，一個 tile-centric 的 Python eDSL，官方稱之為「20 年來 CUDA 最大的更新」。
> 2. 原始 CUDA 團隊成員 **Nicholas Wilt** 公開表示「很難不懷疑 cuTile 是直接為了對抗 Triton 而生」——與 Triton、Helion 是同一類 eDSL。
> 3. 獨立評測論文 **arXiv:2604.23466**（Yadav, Zhao, Kumar）用四個工作負載、三個架構做 head-to-head：
>    - **Blackwell B200 fused attention**：cuTile 1007 TFLOP/s，**60 行 Python 打贏 FlashAttention-2 2.5×**。
>    - **GEMM**：cuTile **22 行**達成 cuBLAS 的 52-79%（WMMA 需要 123 行）。
>    - **RTX PRO 6000 (sm_120) 上的同一個 attention kernel**：只剩 FA-2 的 53% —— **跨架構落差非常大**。
>    - **Triton**：**跨所有平台守在 cuBLAS 的 62-101%**，portability 領先，但 GEMM 要多寫 2.4× 程式碼。
> 4. CUDA 13.1 首發 **不支援 Hopper**，13.2 才補上——這既是技術決策，也是生態訊號。
> 5. 對 compiler 工程師而言：**tile-level abstraction 是不可逆的方向**（Triton、CuTe、cuTile、Helion、TileIR、AutoTriton 都在收斂到同一個 IR shape），但 vendor lock-in 是致命風險——**真正的護城河不在 tile 語法，而在 lowering pipeline**。

---

## 一、事件開場：Nicholas Wilt 的那句「很難不懷疑」

2025 年 12 月，NVIDIA 在 CUDA 13.1 版本說明中把 cuTile 定位為 **"the biggest CUDA update in 20 years"**（自稱 20 年來最大更新）。這是一個大膽的自我定位——過去 20 年 CUDA 從單 kernel 走到 unified memory、CUDA Graphs、cooperative groups、CUTLASS、CUTE、CUDA Python bindings，每一步都是漸進。cuTile 用「20 年最大」拉高聲量，等於在告訴市場：**我們準備好放棄 SIMT-first 心智模型，全面切到 tile-first。**

問題是——**這個 tile-first 的路，Triton 早在 2019 年就走了**。OpenAI 在 2021 年開源 Triton 之後，經過 Meta、xAI、Anthropic、AMD 一路加碼（AMD 官方 fork、Meta Triton-on-ROCm、PyTorch Inductor 直接把 Triton 當後端），Triton 已經是**業界事實上的 tile-level GPU DSL**。CUDA 這時候端一個 API 幾乎鏡像、抽象層級一模一樣的東西出來，很難不讓人有想法。

原始 CUDA 團隊成員 [Nicholas Wilt](https://en.wikipedia.org/wiki/Nicholas_Wilt)（早期 CUDA 架構師之一、《The CUDA Handbook》作者）在社群上直接開嗆：

> "It's hard not to suspect that cuTile was developed directly to counter Triton.
> cuTile is a new eDSL for writing kernels, just like Triton or Helion."
> —— Nicholas Wilt

翻譯：「很難不懷疑 cuTile 就是為了直接反制 Triton 而生。cuTile 是一個新的 eDSL，就像 Triton 或 Helion 一樣。」

這裡有幾個 tell：
1. **Wilt 用 "counter" 而不是 "compete"**——暗示這不是市場擴張，而是防禦動作。
2. **他點名 Helion**——Meta 2025 年才開源的 Triton-on-Triton 更高階 DSL，這個對照組本身就是諷刺。
3. **他強調「eDSL」**——刻意跟 CUDA C++ 的路線切開，暗示 cuTile **不是 CUDA 原生演化，是外掛的抄捷徑**。

社群後續補刀更難聽——有開發者說 cuTile 看起來是 **Triton + Mojo + ThunderKittens 的混血**。這句話酸點在於，這三個東西**全都不是 NVIDIA 做的**。

到目前為止，NVIDIA 沒有正式回應。這種沉默本身也是一種訊號。

---

## 二、什麼是 cuTile？：tile-first 抽象與 CUDA 13.1 的技術實體

先講清楚 cuTile 到底做了什麼，才有辦法評價 Wilt 的指控。

### 2.1 從 SIMT 到 Tile：GPU 程式模型的第三次抽象

回顧 CUDA 20 年的抽象史：

| 時代 | 主要抽象 | 開發者要管的東西 |
|---|---|---|
| **CUDA 1.x-8.x（2007-2016）** | Thread + Block + Grid | thread index、shared memory、bank conflict、warp divergence |
| **CUDA 9.x-12.x（2017-2024）** | + Cooperative Groups、CUTLASS、CUTE | tile 是 template 手動組出來的，仍要理解 SIMT |
| **CUDA 13.x（2025-）** | **Tile 是一級公民** | 只寫 tile-level operations，thread orchestration 由 runtime/compiler 處理 |

CUDA 13.1 的核心宣稱是：**"A tile is a contiguous, multi-dimensional chunk of data that maps to a thread block's shared memory footprint."** 一個 tile 是連續、多維、映射到 thread block shared memory 的資料塊。開發者不再寫 `threadIdx.x + blockDim.x * blockIdx.x`，而是寫 `tile.load()`、`tile @ tile`、`tile.store()`。

### 2.2 cuTile 的具體語法（示意）

```python
import cutile as ct

@ct.kernel
def gemm(A: ct.Tensor, B: ct.Tensor, C: ct.Tensor, M, N, K):
    # 每個 program instance 處理一個 (BM, BN) 的 output tile
    pid_m, pid_n = ct.program_id(0), ct.program_id(1)

    a_tile = ct.zeros((BM, BK), dtype=ct.float16)
    b_tile = ct.zeros((BK, BN), dtype=ct.float16)
    acc = ct.zeros((BM, BN), dtype=ct.float32)

    for k in ct.range(0, K, BK):
        a_tile = ct.load(A[pid_m*BM : (pid_m+1)*BM, k : k+BK])
        b_tile = ct.load(B[k : k+BK, pid_n*BN : (pid_n+1)*BN])
        acc += ct.dot(a_tile, b_tile)  # 自動下沉到 mma.tcgen05 / wgmma / hmma

    ct.store(C[pid_m*BM : (pid_m+1)*BM, pid_n*BN : (pid_n+1)*BN], acc.to(ct.float16))
```

如果你寫過 Triton，這段程式碼會**極度眼熟**。事實上你只要把 `import cutile as ct` 改成 `import triton.language as tl`，把 `ct.program_id` 改成 `tl.program_id`，語法幾乎可以直接改寫。這就是 Wilt 說「同一類 eDSL」的具體證據。

### 2.3 硬體端：cuTile 到底把什麼東西下沉了

cuTile 之所以敢說「20 年來最大」，是因為它把 Blackwell 的新硬體特徵**內建進 tile 抽象**：

1. **TCGEN05 / Tensor Memory**：Blackwell 引入了一個新的儲存階層——Tensor Memory，位於 register file 與 shared memory 之間，專供 tensor core intermediate 使用。cuTile 的 `ct.dot` 會自動 spill 到 Tensor Memory 而不佔用 shared memory 頻寬。
2. **Asynchronous MMA (async wgmma / async tcgen05)**：Blackwell 的 MMA 指令**完全 asynchronous**，需要顯式 barrier。手寫 CUDA 要處理一堆 `mbarrier`，cuTile 自動幫你插。
3. **TMA (Tensor Memory Accelerator)**：Hopper 引入、Blackwell 強化的 bulk async copy。cuTile 的 `ct.load` 在 Blackwell 上直接下沉到 TMA。
4. **CTA Cluster / Distributed Shared Memory**：cuTile 可以在多個 CTA 之間共享 tile（透過 `cluster.load`）。

這四個東西合起來就是 **Blackwell 峰值效能的 unlock code**。手寫 CUDA C++ 要拿到峰值，你得知道 TCGEN05 的 MMA layout、Tensor Memory 的 alignment 規則、TMA 的 descriptor 格式、cluster barrier 的語意——**這些東西都沒有進 CUDA C++ 主線 API**，只在 PTX inline 或 CUTe 裡才碰得到。cuTile 把它們包起來，變成一行 Python。

**這才是「20 年最大更新」的技術實體**：不是 tile 抽象本身（Triton 5 年前就做了），是**把 Blackwell 的所有隱藏武器打包給 Python 使用者**。

---

## 三、獨立評測論文：cuTile 真的贏了嗎？

上面是 NVIDIA 的敘事。真正好玩的是 arXiv:2604.23466（Yadav, Zhao, Kumar 2026）這篇獨立評測，它做的事情很簡單：**把 cuTile、Triton、CUDA WMMA、cuBLAS、FlashAttention-2 拉到同一張 benchmark 表上，跨三個架構跑同樣的 workload。**

這篇論文的價值在於：**NVIDIA 官方 benchmark 只給你 Blackwell 數字，這篇把「跨架構退步」的黑暗面攤開來。**

### 3.1 GEMM（矩陣乘法）：程式碼行數 vs 效能

| 實作 | LoC | Hopper H100 (% of cuBLAS) | Blackwell B200 (% of cuBLAS) | sm_120 RTX PRO 6000 (% of cuBLAS) |
|---|---|---|---|---|
| cuBLAS (baseline) | — | 100% | 100% | 100% |
| **cuTile** | **22** | **不支援** | **79%** | **52%** |
| Triton | 53 | 87% | 82% | 62% |
| CUDA WMMA (手寫) | 123 | 71% | 68% | 55% |
| 純 SIMT CUDA | 280+ | 43% | 39% | 31% |

幾個重點：

1. **cuTile 在 Blackwell 上 22 行拿 79% cuBLAS**——這是**驚人**的生產力。作為對照，Triton 需要 53 行拿 82%，WMMA 需要 123 行拿 68%。
2. **cuTile 在 sm_120 掉到 52%**——同一份程式碼，換到 RTX PRO 6000 就大幅退步，因為 sm_120 沒有 Tensor Memory、沒有 TCGEN05。cuTile 的下沉策略在這個架構上失效。
3. **CUDA 13.1 的 cuTile 完全不支援 Hopper (sm_90)**。這是**極度反常**的技術決策，我後面會單獨拆解。
4. **Triton 是 portability 王者**——三個架構全部 62-87%，波動最小。這正是 Wilt 說 cuTile 目標是 Triton 的原因：Triton 的生態價值就是「一份程式碼三個架構」，cuTile 打不到這一點就沒有敘事優勢。

### 3.2 Fused Attention：cuTile 的 kill move

| 實作 | LoC | Blackwell B200 (TFLOP/s) | sm_120 (% of FA-2) |
|---|---|---|---|
| FlashAttention-2 (手工 CUDA) | ~2000 | 403 | 100% |
| FlashAttention-3 (手工 CUDA + TMA) | ~2500 | 621 | N/A |
| Triton fused attention (tutorial) | ~180 | 512 | 71% |
| **cuTile fused attention** | **60** | **1007** | **53%** |

這張表是 NVIDIA 敘事的核心：

- **Blackwell B200 上 cuTile 60 行 Python 打出 1007 TFLOP/s**——比手工優化 2500 行的 FA-3 還快 62%，比 FA-2 快 2.5 倍。
- **為什麼贏這麼多**：因為 cuTile 自動用了 TCGEN05 async MMA + Tensor Memory + TMA + Blackwell 專屬 warp specialization。手寫 CUDA 要用這些東西你得等 NVIDIA release CUDA sample，才知道 mbarrier 該放哪。
- **代價**：**同一個 kernel 換到 sm_120 只剩 FA-2 的 53%**——因為 sm_120 沒有這些硬體特徵，cuTile 的抽象降級到 pre-Blackwell 路徑，反而輸給手工 FA-2。

這就是 **cuTile 的核心矛盾**：它的生產力優勢建立在 **「Blackwell 專屬硬體 auto-lowering」** 上；一旦離開 Blackwell，那個 60 行 Python 只是 60 行慢 kernel。

### 3.3 為什麼 Triton 沒有這個問題

Triton 的 IR 設計哲學是**「compiler-first」而不是「hardware-first」**：

- Triton 語言的 tile 是**抽象的**——`tl.dot(a, b)` 由 Triton compiler 決定要下沉到 `mma.sync.m16n8k16` (Volta)、`mma.sync.m64n256k16` (Hopper wgmma)、還是 `tcgen05.mma` (Blackwell)。
- 這個下沉是**在 backend pass 裡做的**——tt.dot → ttgir.dot → 各架構 PTX。你的 kernel code 完全不用改。
- **代價**：Triton 對 Blackwell 硬體最深的 unlock（例如 async wgmma 的 producer-consumer 模式、Tensor Memory 的 tile fragment layout）**需要 compiler 端手動加 pass**。Triton 3.8 才把 warp specialization 補齊（我在 [triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026](./triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026.md) 討論過）。
- 所以 Triton 的曲線是**慢一步、但穩**——所有平台 60-90% cuBLAS，沒有 cuTile 那種「B200 破紀錄 / sm_120 崩盤」的兩極分化。

**這是兩種 compiler 哲學的差異**：
- **cuTile**：語言層貼近硬體語意（TMA/TensorMem/TCGEN05 是 API 一部分），下沉短，靜態綁架構。
- **Triton**：語言層架構無關，下沉長，靠 compiler pass 做架構特化。

哪一個「對」？取決於你相信 GPU 架構未來會**收斂**還是**分岔**。NVIDIA 賭收斂（都是 Blackwell 系列往前推），Triton 陣營賭分岔（AMD MI300、Intel Gaudi3、Google TPU、Cerebras 都在做自己的東西）。

---

## 四、CUDA 13.1 為什麼首發不支援 Hopper？

這是整個故事最詭異的一點，值得單獨拆。

CUDA 13.1 首發時：
- **支援**：Blackwell (sm_100, sm_120)
- **不支援**：Hopper (sm_90), Ampere (sm_80), Ada (sm_89)
- CUDA 13.2（2026 年上半）才把 Ampere、Ada 加回來
- **Hopper 支援時程模糊**——目前 NVIDIA 只說「future release」

這個決策非常反常，NVIDIA 過去所有主要 API 演進（CUDA Graphs, unified memory, cooperative groups）都是**「往前優先，向後兼容」**——先在最新架構上跑起來，但至少覆蓋前一代旗艦。cuTile 直接跳過 Hopper 是**破例**。

有兩種解讀：

### 4.1 技術解讀：TCGEN05 是斷代分界

cuTile 深度依賴 **Tensor Memory** 這個 Blackwell 才有的儲存階層。Hopper 只有 **wgmma**（warp-group MMA），沒有 tensor memory intermediate storage。cuTile 的整個 lowering pipeline 是**圍繞 Tensor Memory 設計**的，要 backport 到 Hopper 等於要重做一條完全不同的下沉路徑。

**這說得通**——但如果只是技術問題，NVIDIA 應該在 blog 裡大方講「Hopper 需要獨立 lowering pass，預計 CUDA 14 支援」。**他們沒有**。

### 4.2 生態解讀：Hopper 是 Triton 的主場

Hopper 是目前**業界部署量最大的 AI 加速器**——H100/H200 是 2024-2026 主流訓練/推論卡。Triton 在 Hopper 上的優化已經**完全成熟**（PyTorch Inductor、vLLM、SGLang、FlashAttention 全都在 Hopper 上用 Triton）。

如果 cuTile 一上市就在 Hopper 上跟 Triton head-to-head，會有兩種結果：
- **cuTile 贏了 Triton**——NVIDIA 得到 marketing 素材，但要面對「你用內部硬體資訊護 API」的反壟斷風險（AMD/Intel 用 Triton，NVIDIA 用 cuTile，硬體資訊不對稱）。
- **cuTile 沒贏 Triton**——那 cuTile 立刻失去「為什麼要用 cuTile」的敘事。

**先跳過 Hopper、把首戰放在 Blackwell**，NVIDIA 可以：
1. 打出「cuTile 60 行贏 FA-3 2500 行」的頭條數字。
2. 迴避 Hopper 上跟 Triton 直接對決的市場檢驗。
3. 用「Blackwell 是未來，Hopper 是過去」的敘事，把 Triton 用戶引導到「你要不要遷 Blackwell？遷了就用 cuTile 吧」。

**這個解讀很難不成立**。Wilt 說 cuTile "targeting Triton" 的合理性也在這裡——如果只是技術演進，你不會在生態最大的架構上刻意缺席。

---

## 五、生態格局：三個 tile-first DSL 的戰爭

現在市場上 tile-first GPU DSL 已經有：

| DSL | 母公司 | Lowering target | Portability | 主要用戶 |
|---|---|---|---|---|
| **Triton** | OpenAI (2019 開源) | NVIDIA + AMD ROCm + Intel Gaudi + CPU | 高 | PyTorch Inductor, vLLM, SGLang, FlashAttention, Meta, xAI |
| **cuTile** | NVIDIA (2025) | NVIDIA only (Blackwell first) | 低 | NVIDIA 內部 sample, 部分 AI lab |
| **Helion** | Meta (2025) | Triton (via ttir/ttgir), 未來可能 CUDA C++ | 中 | Meta 內部 |
| **CuTe DSL** | NVIDIA (CUTLASS 3.x) | NVIDIA only | 低 | 高階 CUTLASS 用戶（我在 [cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026](./cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md) 討論過） |
| **ThunderKittens** | Stanford Hazy Research | NVIDIA + AMD (研究等級) | 中 | 學術界、極致優化 |
| **Mojo** | Modular | NVIDIA + AMD + CPU + Apple | 高 | Modular MAX, 部分研究 |

**Wilt 的批評放在這張表上就很清楚**：cuTile 的定位完全跟 Triton overlap，跟 CuTe DSL 反而錯開（CuTe 是 C++ template 為底，cuTile 是 Python）。**cuTile 不是為了補 CUDA 的洞，是為了搶 Triton 的份額**。

### 5.1 誰會贏這場戰爭？三個劇本

**劇本 A：cuTile 收復 NVIDIA 生態**（NVIDIA 押的注）
- 前提：Blackwell 及後續 Rubin 架構成為 AI 主流，Hopper 逐步淘汰。
- 過程：cuTile 靠 Blackwell 硬體優勢拉出效能差，PyTorch 加 cuTile backend（跟 Triton backend 平行），生態逐步遷移。
- 風險：AMD、Intel、Meta 加大 Triton 投入，PyTorch 官方態度成關鍵。

**劇本 B：Triton 守住 portable-first 敘事**（開源陣營的希望）
- 前提：AI 硬體繼續分岔（AMD MI400、Google TPU、Cerebras、Groq、Tenstorrent 都在成長），異構部署變主流。
- 過程：Triton 3.8 warp specialization 補齊 Blackwell 差距，AMD ROCm Triton 補齊 CDNA 差距，cuTile 變成「NVIDIA 內部 sample 專用」。
- 風險：Triton compiler 開發資源不足（相比 NVIDIA 內部投入 cuTile 的人力），Blackwell/Rubin 專屬指令補得慢。

**劇本 C：Helion / Mojo 這種「更高階 DSL」吃掉 Triton 和 cuTile**（黑馬）
- 前提：tile-level DSL 太底層，AI 工程師想要更高階的 tensor DSL（類似 Halide 的 schedule/algorithm 分離）。
- 過程：Helion 這種 Triton-on-Triton 或 Mojo 這種語言級抽象，用 autotuning + LLM 生成 kernel 替代人寫 tile。
- 風險：這條路 3-5 年才會成熟，短期是 Triton/cuTile 的戰場。

**我個人賭 B + C 的混合**。理由：
1. AMD MI300X 已經是 vLLM 的主要部署平台之一，portability 不是加分項是**必需品**。
2. AutoTriton（RL 生成 Triton kernel）這條路 2025 年已經跑出來（arXiv:2507.05687），Kernel 自動化正在把「寫 tile code」這件事去技能化。
3. NVIDIA 自己都要維護 cuTile + CuTe + CUDA C++ + PTX 四條路，內部 fragmentation 反而幫 Triton 站穩「唯一 portable 的選項」位置。

但——**這不代表 cuTile 會消失**。它會活成「Blackwell 極致優化」的專用 DSL，跟 Triton 分工。就像 CUTLASS 沒有取代 cuBLAS，cuTile 也不會取代 Triton，而是**上到 Blackwell 峰值時的最後一哩路**。

---

## 六、對 compiler 工程師的意義：真正該學什麼

如果你（像我）在做 compiler 職涯規劃，這場 cuTile vs Triton 對決給出的訊號很明確：

### 6.1 Tile-level abstraction 是不可逆的方向

不管你信 cuTile 還是 Triton，**tile 已經是 GPU 程式設計的新 lingua franca**。以下這些東西你必須熟：

- **Tile-level IR**：Triton TTIR/TTGIR、MLIR linalg dialect、CUDA TileIR
- **Warp specialization**：producer/consumer 模式、async pipeline
- **Tensor Memory / TMA / wgmma / tcgen05**：Blackwell/Hopper 硬體語意
- **Layout transformation**：swizzling、bank conflict、tile fragment mapping
- **Autotuning**：block size、num_stages、num_warps 的搜尋空間設計

### 6.2 真正的護城河是 lowering pipeline，不是 tile 語法

**cuTile 和 Triton 的頂層語法幾乎一樣**——差別在下面幾層 IR 和 pass。所以：

- 學會**寫 tile kernel** 是入場券（一週的事）。
- 學會**看 lowering trace**（`TRITON_INTERPRET=1`, `MLIR_DUMP_IR_BEFORE_ALL=true`, `NVCC_APPEND_FLAGS='-lineinfo'`）是 compiler 工程師的日常。
- 學會**寫 MLIR pass / Triton pass**（例如：讓 attention kernel 自動 fusion softmax 到 dot 上）才是**真正稀缺的技能**。
- 學會**跨 target lowering**（同一個 IR 下沉到 NVIDIA/AMD/TPU）是頂級護城河。

### 6.3 給自己的行動點

1. **本週**：把 arXiv:2604.23466 完整讀完，跑一次論文附錄的 GEMM benchmark（cuTile + Triton + WMMA 三路對照），寫成 blog 系列第二篇。
2. **下週**：把 Triton tutorial 06-fused-attention 讀完，然後改寫成 cuTile 版本，實測 Blackwell（如果找得到 B200 access）或 RTX 40 系（sm_89）。
3. **本月**：讀 CUDA 13.2 release notes、Triton 3.8 warp spec commit history、Meta Helion repo，把三個 DSL 的 IR shape 拉在一張圖上比較。
4. **本季**：Contribute 至少一個 Triton compiler PR（optimizer / layout / lowering 都行）——這是最快建立 compiler 職涯 signal 的方式。

---

## 七、結語：這不是 API 之爭，是 GPU 未來抽象的路線之爭

Wilt 那句 "hard not to suspect" 表面是酸 NVIDIA，實際上是**在替整個 compiler 社群提問**：GPU 程式設計的下一個十年，是 vendor-owned 的深度整合抽象（cuTile / CuTe DSL），還是 vendor-neutral 的 portable IR（Triton / MLIR / Mojo）？

歷史上這種戰爭有兩種結局：
- **Vendor 贏**：像 x86 SIMD intrinsics 之於 CPU 向量化——大家最後還是要寫 `_mm256_*`，因為只有 vendor 知道微架構細節。
- **中立層贏**：像 LLVM 之於原本各家自己的 backend——現在幾乎所有語言都 emit LLVM IR。

**GPU 這場戰爭我猜是 hybrid**：Triton 會贏「大部分場景 8-9 成效能」的 default 位置，cuTile 會贏「Blackwell/Rubin 極致 unlock」的專用位置。就像今天你寫 Python 用 NumPy，但 hot path 還是會 drop 到 CUDA C++ 一樣——**只是這個 hot path 從 CUDA C++ 換成 cuTile**。

真正的輸家不會是 cuTile 或 Triton，而是**繼續假裝 SIMT-first 心智模型還夠用的開發者**。Tile 是新 baseline。

要不要學 cuTile？我的答案是：**先學 Triton 打好 tile IR 基本功；等你要摸 Blackwell 峰值時，cuTile 的 60 行 Python 會等你**。

到那時候，Wilt 那句 "hard not to suspect" 會變成一個歷史註腳——不是因為他錯了，而是因為戰爭已經走過那個階段。

---

## 相關閱讀

- 前置背景：[cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026](./cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md) —— CuTe DSL 跟 cuTile 是同一個 NVIDIA 內部路線的兩支。
- Triton 對照：[triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026](./triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026.md) —— Triton 3.8 補齊 Blackwell 的技術細節。
- 生態脈絡：[cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026](./cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026.md) —— CUDA moat 被兩面夾攻的更大敘事。
- 硬體視角：[qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026](./qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026.md) —— 非 NVIDIA vendor 用 MLIR 挑戰 CUDA 的模式。

## 資料來源

- Yadav, D., Zhao, T., Kumar, D. (2026). *Evaluating CUDA Tile for AI Workloads on Hopper and Blackwell GPUs*. arXiv:2604.23466. <https://arxiv.org/abs/2604.23466>
- Nicholas Wilt 對 cuTile 的公開評論，經 HyperAI 整理：<https://hyper.ai/en/news/47715>
- Spheron Network. *CUDA 13 Tile Programming on GPU Cloud: A 2026 Developer Guide*. <https://www.spheron.network/blog/cuda-13-tile-programming-gpu-cloud/>
- Lambda Labs. *FlashAttention-4 gives the NVIDIA Blackwell platform its most optimized attention kernel yet*. <https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet>
- Modular. *Matrix Multiplication on NVIDIA's Blackwell (Part 1)*. <https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction>
- IntuitionLabs. *Blackwell vs Hopper: A Deep Dive GPU Architecture Comparison*. <https://intuitionlabs.ai/articles/blackwell-vs-hopper-gpu-architecture-comparison>
- MLSys 2026 Compilers and Kernels Track. <https://mlsys.org/virtual/2026/session/3718>
