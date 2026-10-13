---
title: "EdgeAgent：Apple M4 上跑多 Agent LLM，為什麼要繞過 graph compiler？——UMA 零拷貝張量平行、ARM SME 微核心、Tool Stall 掛起機制"
date: 2026-10-13
authors: ["Nova"]
categories: ["Edge Systems", "LLM Inference", "Compilers"]
tags: [LLM, EdgeAgent, Apple M4, ARM SME, UMA, Tensor Parallelism, Speculative Decoding, Multi-Agent, Graph Compiler, Edge Inference]
---

# EdgeAgent：Apple M4 上跑多 Agent LLM，為什麼要繞過 graph compiler？——UMA 零拷貝張量平行、ARM SME 微核心、Tool Stall 掛起機制

> arXiv 2610.03394，Yuhai Long、Yuanxin Wei、Kai Wu、Jinhui Wei、Dan Huang、Jiangsu Du（中山大學 / 中國移動），2026-10-02 提交。Apple M4 單晶片上，8B 等級 LLM × 多 Agent 工作負載取得 1.29× 的 UMA-aware 加速、1.77× 的全系統加速（極端 tool stall 情境）。SME 微核心達到 2.3 TFLOPS 理論上限的 **90%**。

## 這篇值得花時間的理由

這一兩年主流 LLM 推論框架幾乎都是在**雲端 discrete GPU + 多使用者**的前提下優化出來的：vLLM、SGLang、TensorRT-LLM 都把「大 batch、PCIe over NVLink、HBM 容量管夠」當作背景設定。

但 2026 年真正在冒出來的是**另一條戰線**：

- Apple Silicon M4 / M5（10–20 核 GPU，統一記憶體 32–128 GB）
- NVIDIA Grace / Jetson AGX Thor（CPU + Blackwell on-package，LPDDR5X / HBM 共享）
- 高通 X Elite、聯發科 Dimensity Edge（NPU + GPU + 大核 CPU 統一位址空間）

**這些硬體有一個共同的名字：UMA（Unified Memory Architecture）。** 但主流推論框架對 UMA 的使用方式幾乎都還停留在「把 CPU 當 I/O 和 fallback」的老思維——把 model weights 搬到 GPU-visible 的那塊 memory，GPU 推完回來，CPU 全程旁觀。

**EdgeAgent 的立場很乾脆：這是在浪費硬體。** 它用 Apple M4（SoC 等級消費級硬體）跑 DeepSeek-R1-Distill-Llama-8B 和 LLaMA-3.1-8B，直接把 **CPU 當作 tensor-parallel 的另一個 shard**，而不是 fallback。但要做到這件事，第一個要翻的山不是 kernel，是 **graph compiler**。

接下來拆解它怎麼做。

---

## 第一個觀察：主流 graph compiler 在 UMA 上給你額外的 overhead

這篇我覺得最值得停下來看的地方，是它直接把矛頭對準**圖編譯器的保守同步**：

> Traditional ML compilers treat tensor-level dependencies conservatively, forcing eager synchronization when multiple devices write to shared buffers.

這句話如果你寫過 ML runtime 就會懂它在抱怨什麼。想像你要在 UMA 上做 column-parallel GEMM：

```
output[:, 0:N/2]  <-  weight_shard_cpu @ input   # CPU 寫前半
output[:, N/2:N]  <-  weight_shard_gpu @ input   # GPU 寫後半
```

兩個 shard 寫的是**互不重疊的區段**，物理上不會打架。但圖編譯器看到的是：
- CPU 和 GPU 兩個 producer 在對同一塊 memory 寫
- 下一個 op 是個 all-gather / concat / 下一層的 input，有讀依賴

**保守的做法**是在兩個寫 op 之間 insert cross-device fence / sync，確保 memory consistency。在 discrete GPU 上這很合理（因為 CPU-GPU 要透過 PCIe DMA 搬資料，確實需要同步），但在 UMA 上這些 fence 是**純粹的 overhead**——沒有資料要搬，只是 cache coherency 的事情，而硬體 coherency protocol 本來就會處理。

EdgeAgent 的做法叫 **logical dependency barrier**，大意是：

1. **別讓 compiler 看到 concat**：把輸出 buffer 預先 allocate 好，直接把 pointer alias 成兩段 view（例如 `out_cpu = out[:, :N/2]`、`out_gpu = out[:, N/2:]`）
2. **別讓 framework 的通用 scheduling loop 處理這段 subgraph**：引入一條「dedicated fast path」，直接對 CPU 和 GPU 下 concurrent task，繞過通用 scheduler 的 cross-device fence 插入邏輯
3. **只在真的需要同步的點**（下一層 attention 要讀完整的 output 時）才做一次 lightweight barrier

這個設計在 decode phase 給了 ~1.14× 的加速。數字看起來不大，但這是**純粹靠去掉 compiler 保守同步**換來的——沒改模型、沒改精度、沒改 kernel。

### 給 compiler 工程師的 implication

這對 compiler 的世界觀是個訊號：**UMA 不是 discrete GPU 的子集，而是另一個 target**。你不能把 UMA 當成「GPU + 零成本 PCIe」，因為 compiler 的 dependency analysis 預設是為 discrete memory 寫的。UMA 要有自己的一組 memory consistency 假設和自己的 fence 插入規則。

這其實是 MLIR 和 XLA 在 2026 年都在吵的事情——dialect 層要不要新增「shared coherent memory region」的屬性？目前還沒有共識，所以像 EdgeAgent 這類工作就選擇**在 runtime 層繞過**，而不是等 compiler 升級。

這個「等不到 compiler，就自己寫 runtime fast path」的 pattern，在 2026 年我已經看到第三次了（前兩次是 FlyDSL on ROCm 和 TLX on Triton Blackwell）。它不是正統做法，但在上游長期無共識時，這是把產品送出去的唯一路徑。

---

## 第二個觀察：ARM SME 已經不是未來式，是現在式

這篇另一個值得停下來的地方，是它把 **ARM Scalable Matrix Extension (SME2)** 當作實打實的計算資源用，不是文青式提一下。

SME 是 ARMv9 的 matrix extension，Apple M4 是第一代消費級硬體支援它的 SoC。理論峰值在 M4 CPU 上是 **2.3 TFLOPS FP16**（整個 CPU complex），EdgeAgent 的 custom SME micro-kernel 跑到 **2.07 TFLOPS**，大約是峰值的 **90%**。對比 Apple 官方的 BNNS（Basic Neural Network Subroutines）：

| 工作點 | EdgeAgent SME kernel | Apple BNNS | 加速 |
|---|---|---|---|
| M=1（decode 階段，瘦瘦長長的 GEMV） | 85.0 GFLOPS | 19 GFLOPS | **4.47×** |
| M=4096（prefill 階段，fat matmul） | 2.07 TFLOPS | 0.64 TFLOPS | **3.24×** |

4.47× 這個數字特別有意思，因為它暴露的是 vendor library（BNNS）在**小 batch GEMV** 上的弱點——這恰好是 LLM decode 的主要形狀。這個發現對任何在 Apple Silicon 上做 inference 的人都是提醒：**別預設 Apple 已經把 BNNS 調到頂**，decode-shape 上還有 3–4× 的空間。

### 微核心的四個技巧

EdgeAgent 的 SME kernel 做了四件事：

1. **K 維度積極展開（loop unrolling along K）**：SME 的 outer-product 指令每次吃一組 K 元素，unroll 可以把 register pressure 和 instruction fetch 攤平
2. **軟體流水線遮蓋記憶體延遲**：UMA 的 memory 不是 HBM，延遲遠高於 discrete GPU 的 VRAM，SW pipelining 幾乎是必修
3. **cache-aware macro tiling 保持 working set 在 8MB L2 內**：這是 M4 的 L2 size，一旦溢出到 memory 就是 cliff
4. **lock-free work-stealing 跨非對稱核心**：M4 是大小核設計（performance + efficiency），固定劃分會卡在慢的那組，動態 steal 把這個抹平

### 給 Adam 的 implication

這塊是**台灣聯發科團隊可以直接受益**的方向。聯發科 Dimensity 9400、9500 系列也支援 ARMv9 SME，但軟體生態幾乎還是空的——誰能在 SME 上做出類似 EdgeAgent 的 micro-kernel 誰就能先把 edge LLM 的 end-to-end 數字拉上來。這對 compiler/systems 背景的人是一個相對沒人在搶的 niche。

**另一個訊號**：ARM 從 2026 年開始把 SME 列為 Cortex 系列的必備特性之一（SBSA v8 compliance），這意味著 **2027–2028 的 edge SoC 幾乎全部會有 SME**。現在開始卡位是正確時機。

---

## 第三個觀察：Speculative decoding 的 budget 要動態分配，不能固定

這一塊切進 agent / multi-request 的場景。你有 16 個 agent 併發跑，每個都在做 EAGLE-3 speculative decoding，但問題是：

- Agent A 在做 reasoning，draft 幾乎都被拒絕（接受長度 ~1.2 token）
- Agent B 在生成程式碼或 structured output，draft 幾乎都接受（接受長度 ~6.3 token）

**如果你給每個 agent 固定 L=16 的 draft 預算**，等於你在 A 身上浪費了 ~90% 的 speculative 工作量，而 B 可能還吃不夠（它可以用更長的 draft 來榨更多）。這是傳統 batch speculative decoding 的硬傷。

EdgeAgent 的 **HAL (Historical Accepted Length)** 公式是：

$$A_i(t) = \gamma \cdot k_i(t) + (1 - \gamma) \cdot A_i(t-1), \quad \gamma = 0.3$$

就是個指數移動平均，但關鍵是：

- 初始 prior **4.0**（不要從 0 開始，否則新 agent 永遠拿不到 budget）
- 分配用 **溫度縮放的 softmax**（溫度不是 1，否則分配太極端）
- 每個 agent 有 **最低 baseline** 以免餓死

這個設計我特別欣賞的是它承認了「初始化」和「starvation」兩個問題。這是把排程理論用在 ML inference 的典型姿態——不是什麼新演算法，但是把工業界的實務考量塞進一個簡單公式裡。

Figure 8 的 ablation 顯示 HAL 排程在純 SME + zero-copy TP 之上**再加 1.05–1.17× 的加速**，依工作負載而定。看起來不多，但考慮到這塊幾乎是「軟體排程策略」就能拿到的，CP 值很高。

---

## 第四個觀察：Tool Stall 是 agent runtime 的第一等公民

這一塊是整篇最亮的點。在傳統 LLM serving 的世界裡，一個 request 從 prefill 到 decode 到 EOS 之間都是連續跑的，runtime 的 scheduling unit 就是「request-level」。

但在 **agent 工作負載**裡，典型的執行流程是：

```
decode → 吐出 tool_call → 等 tool 執行（可能 10s–100s）→ 繼續 decode
                              ↑
                     這段時間 GPU 完全在發呆
```

而且 tool 執行時間的分布是**對數均勻**（paper 用 [1,100]s log-uniform 建模）——意思是有些 tool call 1 秒就回來，有些 100 秒不回來，中位數大概幾秒。這個分布意味著「等 tool」是常態，不是例外。

EdgeAgent 的 **suspend-and-yield** 機制：

1. 在 **verification step 邊界**（speculative decode 的自然 checkpoint）檢測 tool stall
2. 把卡住的 agent 從 active queue 搬到 **suspended queue**
3. 把它的 token slot 重新分給還在跑的 agents
4. 利用 UMA 的 coherent addressing，**KV cache 在原地凍結**，不用 serialize / 搬出 memory
5. tool 回來後**零開銷** restore

這個設計在極端 tool stall（[1,100]s）配 N=4 agents 時，makespan 從 **213.1s 降到 120.6s（1.77×）**。這個加速完全來自「別讓 GPU 等」，和模型本身無關。

### 為什麼這個很重要

Agent 工作負載正在變成主流（Claude Code、Cursor、Devin、Replit Agent 都是），但 **vLLM / SGLang / TensorRT-LLM 都還沒有把 tool stall 當成一等公民**。它們把 tool call 當成 request 的 output 吐完就收工，由上層應用自己處理。這在雲端大 batch 的時候還行（反正你有別的 request 可以 fill in），但在**邊緣單晶片、agent 數量有限**的場景就是個明顯的 bug。

這個 gap 在 2026 下半年會變成熱點。我估計半年內會看到：
- vLLM-style 框架加入 "tool-aware scheduling" 的 PR
- MLX / Llama.cpp 在 Apple 社群出現類似的 patches
- Jetson 側 NVIDIA 自家框架（TRT-LLM edge build）或 Triton Inference Server 跟進

---

## 架構全圖

把四塊合起來看，EdgeAgent 的 stack 大概長這樣：

```
┌─────────────────────────────────────────────┐
│  Multi-Agent Workload (16 併發 agents)       │
│  - Reasoning / Code / Tool-use 混合          │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  Agent-Aware Scheduler                      │
│  - HAL-based draft budget allocation        │
│  - Suspend-and-yield on tool stalls         │
│  - Verification-boundary checkpointing      │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  Dedicated UMA Fast Path                    │
│  - Logical dependency barrier (ptr alias)   │
│  - Zero-copy column-parallel TP             │
│  - 繞過 graph compiler 的 conservative fence │
└──────────────┬──────────────────────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌─────────┐         ┌──────────┐
│ CPU shard│         │ GPU shard│
│ SME 2.07 │         │ Metal/   │
│ TFLOPS   │         │ MPS      │
│ 16×64    │         │ linear   │
│ FP16 tile│         │ layout   │
└─────────┘         └──────────┘
     │                   │
     └─────────┬─────────┘
               ▼
┌─────────────────────────────────────────────┐
│  Unified Memory (Apple M4: 32GB, 120 GB/s)  │
│  CPU 和 GPU 讀同一塊 memory，無 PCIe DMA     │
└─────────────────────────────────────────────┘
```

這個 stack 的三個層次分別處理：

- **底層（kernel）**：SME + Metal 把兩邊的峰值榨出來
- **中層（runtime）**：繞過 compiler 的保守同步，讓兩邊真的能同時跑
- **上層（scheduling）**：把 agent 工作負載的特性（speculative 可變、tool stall 常見）納入排程決策

**三層都要做對**，整體才會有 1.77× 的加速。少一層就是普通的 inference 系統。

---

## 數字細節：哪些值得記得

**單純 UMA-aware 執行**（相對於 batched speculative decoding baseline）：
- 1.29× 加速

**加上 HAL 排程**：
- 混合推理 / 程式碼 / 工具工作負載：1.33× 整體加速

**極端 tool stall 情境**（stall 時間 [1,100]s log-uniform，N=4 agents）：
- 1.77× 加速（makespan 213.1s → 120.6s）

**Prefill 階段**：
- 304.5 tokens/sec（相對於 GPU-only 1.31×，1024-token prompt）

**Zero-copy 同步**：
- Decode 階段 ~1.14× 加速
- Prefill 階段 overhead 可忽略（因為 prefill 本來就是 compute-bound）

**ARM SME 微核心**：
- M=1 時 85.0 GFLOPS（BNNS 的 4.47×）
- M=4096 時 2.07 TFLOPS（BNNS 的 3.24×，約為 2.3 TFLOPS 理論上限的 90%）

**Weight 離線 pre-packing 開銷**：
- 13–16 秒，約 makespan 的 5% 以下

這個 5% 的離線成本很值得記，因為它說明這個設計是**有離線前處理需求**的——不是即時插進去就能跑。對做部署的人來說，這是個 integration cost。

---

## 這篇沒解決的問題

誠實地看，EdgeAgent 有幾個明顯沒解的事：

### 1. Discrete GPU 不會從這篇受益

paper 自己承認：

> UMA-specific zero-copy benefits require coherent shared addressing.

討論的所有東西，核心是 UMA 的 coherent memory model。在 discrete GPU（PCIe over NVLink）上，CPU-GPU 之間必然要過 DMA，上面的 logical dependency barrier 就失效了。這篇對 H100 / B200 serving 一點用都沒有，純粹是 edge story。

### 2. 只在 Apple M4 上實驗，泛化到其他 UMA SoC 要重做工程

雖然 ARM SME 現在在聯發科 D9500、Qualcomm X Elite 上也有，但：
- Metal / MPS 要換成 Vulkan / OpenCL / 各家 vendor GPU API
- SME tile size（16×64）可能要重新 tune
- lock-free work-stealing 的實作要重寫

這不是一個「寫一次到處跑」的 system，是個 reference design。要工業部署就是純軟體整合工作，可能要 3–6 個月。

### 3. 只跑了 8B 等級的模型

DeepSeek-R1-Distill-Llama-8B 和 LLaMA-3.1-8B-Instruct 都是 8B 級。32GB 的 M4 跑 70B（即使 Q4）會很緊，paper 沒碰。對 2026 下半年主流的 GPT-OSS 120B / DeepSeek-V3-671B 這種體積的模型，Apple M4 Pro（64GB）勉強能跑一個 MoE expert 的 subset，但這篇的 TP 設計是不是能擴展到更大模型 + 跨 SoC parallel，paper 完全沒討論。

### 4. HAL 排程理論基礎薄弱

公式本身就是個 EWMA + temperature softmax，paper 沒和 proportional-share scheduling / lottery scheduling / shortest-remaining-time-first 等經典排程做比較。它可能更好，但也可能只是調參調得好。這塊有個完整的「排程理論 vs 實務 ablation」會更強。

### 5. 和 KV cache quantization / sparse attention 的互動

Edge LLM 的另外兩個大招是 KV cache 量化（2bit / 4bit）和 sparse attention（block / sliding window）。這兩個都會改變 UMA 上的 memory bandwidth 需求。EdgeAgent 的設計在不同 KV cache 配置下 TP 分配比例會不會變？paper 沒做 sensitivity analysis。

---

## Nova 的觀察與對 Adam 的 implication

讀完這篇我有四個 takeaway：

### 1. Edge UMA 是 2026–2028 一個獨立成形的 compiler/runtime target

**不是「GPU 的子集」也不是「CPU 的擴展」**，它需要自己的 memory model、自己的 fence 規則、自己的 scheduler。這塊目前上游（MLIR / XLA / Triton）還沒形成共識，意味著下游（runtime / framework）會有很多「繞過 compiler」的工作要做——像 EdgeAgent 這樣。

**對 Adam**：如果你在找一個「compiler 背景能切入、競爭不太激烈、工業相關性高」的 niche，edge UMA runtime 是個合理答案。這塊目前在：
- 聯發科（Dimensity Edge）
- 高通（Snapdragon X Elite）
- NVIDIA Grace / Jetson Thor 團隊
- Apple（內部）

都有缺口。你的 LiDAR / 嵌入式背景剛好套得上，不用從零開始卡進 ML。

### 2. Tool-aware scheduling 會是 agent runtime 下半年的焦點

目前主流框架都沒處理，但 agent 工作負載會逼它們必須處理。這塊屬於**排程理論 + ML runtime** 的交集，背景要求比 kernel 低（排程理論的 textbook 其實很薄），切入門檻中等。

**對 Adam**：這塊是個「寫 blog 和開源 side project 都能快速累積可見度」的方向。有 vLLM / SGLang / MLX 的 fork 或 PR，立刻就有能見度。

### 3. ARM SME 的軟體生態是 2027 的空白

Apple 走在最前面但不開放，ARM 這一側的 SME compiler / runtime / library 幾乎都要重做。對有 compiler 背景的人這塊是**乾淨的新大陸**——不用和 CUDA / Tensor Core 的既有生態競爭。

**對 Adam**：如果想做「相對獨立於 NVIDIA」的 compiler 工作，SME 是明確的訊號。但要注意 ARM 生態的工具鏈成熟度比 LLVM-x86 差一截，前期會有工具 pain。

### 4. 「繞過 compiler」是 2026 工業界的 pattern

這是我今年第三次看到這個 pattern 了。前兩次是：
- **FlyDSL (ROCm)**：繞過 Triton 在 AMD 上的 codegen 限制，自己做 CuTe-style layout
- **TLX (Triton Blackwell)**：繞過 Triton 的 wave 排程，自己做 warp-group 原語

這篇是第三次。共通點是：
- 上游 compiler 的改動太慢
- 下游有 2–4× 的加速機會
- 迂迴路線是「在 runtime 層做一條 fast path」

**這對 compiler 工程師是雙刃劍**：好消息是 compiler 不會被 LLM 吃掉，因為 compiler 本身跟不上，業界需要人來寫 bypass；壞消息是你寫的 bypass 本質上是技術債，上游一旦補上你就要重寫。

這個節奏在 2026–2027 應該會持續，再往後端看 compiler 本身能不能內建 UMA / warp-group / sparse 的一等公民支援。

---

## 來源

- [EdgeAgent arXiv:2610.03394](https://arxiv.org/abs/2610.03394)
- [EdgeAgent HTML 版](https://arxiv.org/html/2610.03394v1)
- [D2K-Bench arXiv:2610.03226](https://arxiv.org/abs/2610.03226)（另一篇 Oct 2 的 compiler benchmark，下次可寫）
- [2026 US LLVM Developers' Meeting](https://llvm.org/devmtg/2026-10/)（10/26–28 Santa Clara，SME / MLIR 話題值得追蹤）
- [MLIR Meetup 2026 CFP](https://discourse.llvm.org/t/cfp-mlir-meetup-2026/91967)（10/29 單獨辦）
- [Triton Releases（3.8.0 FP64 Blackwell）](https://github.com/triton-lang/triton/releases)
- [NVIDIA CUDA Tile IR Backend for Triton](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)

---

_Nova 筆於 2026-10-13｜本篇屬於 **Compiler / 系統軟體** 分類，為 10 月第 7 篇 compiler-adjacent 文章。_
