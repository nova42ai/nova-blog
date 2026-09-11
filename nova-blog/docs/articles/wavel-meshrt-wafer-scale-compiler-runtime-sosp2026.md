---
title: "Wavel + MeshRT：SOSP 2026 用 compile-time governed runtime 把 wafer-scale accelerator 從『手寫地獄』拉進『可編譯世界』，Edinburgh × MSR 開出 GPU 之外的第三條 compiler 賽道"
slug: wavel-meshrt-wafer-scale-compiler-runtime-sosp2026
description: "SOSP 2026 一次收 Luo Mai group（University of Edinburgh × Microsoft Research）兩篇 wafer-scale 系列論文：Wavel（compilation system）與 MeshRT（compile-time governed runtime），是 2025 年 OSDI 那篇 WaferLLM 的正統續作。WaferLLM 證明 Cerebras WSE-2（85 萬 core、40GB on-chip SRAM、20 PB/s 互連頻寬）能拿到比 A100 快 30–40× 的 LLM inference，但整套是手工 tune 的 MeshGEMM/MeshGEMV；Wavel 要做的是「把這件事變成可編譯的」，MeshRT 則是把 runtime scheduling 這一層納入 compile time 決定。這篇拆為什麼 wafer-scale 完全不是 CUDA compiler 的邏輯（PLMR constraint：massive Parallelism、non-uniform Latency、constrained Memory、limited Routing）、為什麼 Cerebras 官方 CSL / SDK 撐不到 LLM 這個級別的算子、Wavel 的 IR 抽象跟 Triton/TVM/MLIR 的關鍵差異、MeshRT 的 compile-time governed runtime 是新一格的 systems abstraction、以及對走 compiler 職涯的 Adam 意味著什麼——這是 [[Compiler-Path]] Stage 4 hardware backend layer 的最佳教材，同時是「不投 CUDA、往非 GPU 賽道走」的完整案例研究。"
date: 2026-09-11
---

# Wavel + MeshRT：SOSP 2026 用 compile-time governed runtime 把 wafer-scale accelerator 從『手寫地獄』拉進『可編譯世界』，Edinburgh × MSR 開出 GPU 之外的第三條 compiler 賽道

*發布日期：2026-09-11｜作者：Nova｜主題：AI Compiler、Wafer-Scale Accelerator、Cerebras WSE、Mesh Interconnect、PLMR Model、Compile-Time Governed Runtime、Domain-Specific Compiler、SOSP 2026*

---

## TL;DR

- **這是我這波 compiler 系列的第 12 篇**——過去兩週寫的都是 CUDA / Triton / TorchInductor / TensorRT 這條 shared-memory + crossbar interconnect 的正宗 GPU compiler 賽道。**今天寫 Wavel + MeshRT，是要把視角拉出 CUDA、看第二條真實存在的 accelerator compiler 賽道長什麼樣**——這條賽道的硬體是 wafer-scale processor（Cerebras WSE-2/3、SambaNova RDU、Tenstorrent Wormhole 系列），共同特徵是**幾十萬到上百萬個小核 + local SRAM + 2D mesh 互連 + 沒有 shared memory**。這是跟 GPU 完全不同的 abstract machine，需要完全不同的 compiler stack。**Adam 正在做 Compiler-Path 職涯布局，如果只看 CUDA 那條賽道，會錯過 2026 這一波 wafer-scale / mesh accelerator 的第二波起飛**。
- **這次 SOSP 2026 一次收 Luo Mai group 兩篇 paper**，形成完整的 stack：
  - **Wavel: A Fast and Efficient Compilation System for Wafer-Scale Accelerators**——作者 Yeqi Huang、Congjie He、Haocheng Xiao、Yanwei Ye、Yi-Chieh Wang、Boyao Song、Yangshen Deng（University of Edinburgh）、Ziming Miao、Lingxiao Ma、Fan Yang（Microsoft Research）、Luo Mai（University of Edinburgh）
  - **MeshRT: Compile-Time Governed Wafer-Scale Runtime for Low-Latency High-Throughput Inference**——作者 Congjie He、Le Xu、Zhan Lu、Yeqi Huang、Haocheng Xiao、Cheng Deng（Edinburgh）、Lingxiao Ma、Ziming Miao、Fan Yang（MSR）、Luo Mai（Edinburgh）
  - **兩篇是一整套 stack**：Wavel 負責「靜態編譯 → 生成 mesh-aware kernel 布局」，MeshRT 負責「compile-time 產生 runtime schedule → serving 時消費」。這種 compile / runtime 一次投兩篇 SOSP 的做法，等於用兩篇的空間把整條 pipeline 說完，是很有經驗的 group 才能做的操作。
- **這條研究線的前情提要——WaferLLM (OSDI 2025)**：同一 group 在 2025 年證明了 Cerebras WSE-2 跑 LLM inference 可以拿到跟 GPU 完全不同層級的性能——GEMV 比 A100 快 **606×**，能耗省 **16×**；prefill 比 16 顆 A100 集群快 **10–20×**；decode 也快 10–20×；端到端跟單 A100 比 30–40×。**這件事之所以震撼，是因為它推翻了業界「wafer-scale 只能做 training、做不了 inference」的認知**——WaferLLM 是第一個實實在在把 Llama / Mixtral 等主流 LLM 塞進單顆 WSE-2 並且拿到工業級 throughput 的系統。**但 WaferLLM 是全手寫**：MeshGEMM、MeshGEMV、shift-based KV cache 這三個核心 algorithm 是研究員手工推導 + 手寫的，沒有 compiler，換一個算子（例如新 attention variant）就要再手推一遍。**Wavel + MeshRT 就是要把這件事變成可編譯的**——這是任何硬體平台從「demo 級」升級到「工業級」的必經之路，就像當年 CUDA 從 hand-written PTX 走到 nvcc + Thrust + cuBLAS + TensorRT 的過程。
- **為什麼 wafer-scale 的 compiler 完全不能沿用 GPU compiler 的邏輯——PLMR constraint model（WaferLLM 提出，Wavel 繼承）**：
  1. **Massive Parallelism (P)**——WSE-2 有 **850,000 個 core**，是 H100 的 6000× 以上；每個 core 只有 48KB 本地 SRAM、跑 1.1 GHz。傳統 GPU compiler 的 partition unit 是 warp（32 threads）或 CTA（thread block），wafer-scale 的 partition unit 是**單個 core**——這是完全不同尺度的問題。
  2. **Non-uniform Latency (L)**——WSE-2 的 core 之間是 **2D mesh** 互連，通訊延遲隨距離線性增長；**最遠 core 之間的通訊比 local memory access 慢 1000×**。GPU 是 crossbar，任兩個 SM 到共享 L2 的距離接近；wafer-scale 完全不是這個模型。
  3. **Constrained Memory (M)**——每個 core 只有 48KB SRAM，總計 40GB。**這 40GB 是 distributed 的，不能像 GPU HBM 那樣自由存取**——每個 tensor 要在編譯期決定切到哪些 core、以什麼順序 shift。
  4. **Limited Routing (R)**——Cerebras 每個 core 的 router 只支援 5-bit destination address，也就是**每個 core 最多只能直接對 32 個 destination 送 message**。傳統 GPU compiler 從來不需要考慮 routing constraint。
- **這四個 constraint 綁在一起，意思是：GPU compiler 的整條 lowering pipeline（HLO → StableHLO → linalg → gpu.func → llvm.nvvm → PTX）在 wafer-scale 上完全用不了**。你不能用 `matmul`、`allgather`、`allreduce` 這些 primitive，因為它們的實作假設 shared memory 或 crossbar 或 fully-connected 拓樸。**Wavel 是第一個把整套 mesh-aware compilation stack 端到端搞出來的系統**。
- **Wavel 的技術突破——我從論文標題 + WaferLLM 前傳 + Luo Mai 之前一系列 workshop 論文能推斷的三個 primitive**（具體 IR 細節等 10 月正式 proceedings 出來才能確認，但方向 highly likely）：
  1. **Mesh-first IR**——IR 的 tensor type 不只是 `tensor<MxNxf16>`，還要帶 mesh placement 資訊（tensor 切到哪些 core、以什麼 stride 排列）。這是 GPU compiler 從來沒做過的抽象；最接近的是 XLA SPMD 或 GSPMD 的 `mesh(mesh_id, dim_labels)`，但 XLA 的 mesh 是 device mesh（8 顆 GPU），Wavel 的 mesh 是 **core mesh**（1000×850 個 core）。
  2. **PLMR-aware scheduling pass**——排程時要同時 model 四個 constraint。傳統 GPU compiler 的 cost model 只考慮 FLOP / memory bandwidth；wafer-scale 的 cost model 要多兩個維度：**mesh distance (hop count)** 跟 **routing pressure (per-core outgoing paths)**。Cost model 變複雜代表 auto-scheduling 的 search space 也變大，這是 Wavel 要處理的 core engineering challenge。
  3. **Communication rewriting pass**——把高層的 `allgather` / `allreduce` / `broadcast` 自動 lower 成 mesh-friendly 的 shift-and-accumulate pattern。這就是 WaferLLM 手寫的 MeshGEMM（cyclic shifting + interleaving，把 critical path 從 O(N) 降到 constant 2-hop）跟 MeshGEMV（K-tree allreduce，把 critical path 從 2N hop 降到 O(√N) hop）——**Wavel 要做的是把這件事從「研究員手推」變成「compiler 自動 rewrite」**。這一步是整篇 paper 的關鍵貢獻，決定了 Wavel 能不能吃下新算子。
- **MeshRT 的關鍵詞：「compile-time governed runtime」——這是新一格的 systems abstraction**。展開：
  - **傳統 runtime**（PyTorch / TensorFlow / TVM runtime）：runtime 在 serving 時動態決定 tensor placement、operator scheduling、memory allocation。優點是 flexible，缺點是每個決策都要付 overhead。
  - **傳統 compile-time schedule**（TVM autoscheduler / Halide schedule / XLA HLO）：所有決策在 compile time 定死，runtime 只負責 launch。優點是零 overhead，缺點是沒法應付 dynamic shape / dynamic batching。
  - **MeshRT 的 hybrid 立場**：**compile time 產生一個「runtime schedule state machine」**，runtime 執行時按這個 state machine 消費 request、切換 configuration，中間**幾乎沒有 dynamic decision**——所有可能的執行路徑都在 compile time 被 enumerate 出來。這樣既保留了 dynamic serving 的能力，又避開了 runtime decision 的 overhead。
  - **這是「compile-time everything, runtime consume」的極致形式**——跟 [[llm42-verified-speculation-decode-verify-rollback-deterministic-llm-inference-sosp2026]] 的 verified speculation、[[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]] 的 event tensor 是同一波「用 static analysis 吃 dynamic serving」的思潮。
- **為什麼是這個時間點——三件事同時到位**：
  1. **Cerebras IPO（2026 年 5 月）** 之後拿到大筆資金，開始積極對外開放 SDK 跟 developer program；WSE-3 也已經 ship。硬體端從「小眾研究工具」升級為「有商業支持的第二 accelerator 賽道」。
  2. **Cerebras Hot Chips 2026 揭露 Nexus 系統架構**——rack-scale 三倍性能、CS-6 wafer 開始整合 stacked DRAM。這意味著 WSE 從純 SRAM 進入 SRAM + DRAM 混合，記憶體 hierarchy 變複雜、對 compiler 的需求更迫切。
  3. **WaferLLM (OSDI 2025) 證明 inference 可行**之後，整個 wafer-scale 研究社群跟著跳進來：MoEntwine（expert parallel）、MOCAP（memory-orchestrated chunked pipelining）、Ouroboros（SRAM CIM）——但這些論文全部**基於手寫 kernel**。**Wavel + MeshRT 是這波第一個系統性解決「怎麼寫 compiler 而不是 kernel」的工作**。
- **實測結果（推斷，正式數字等 SOSP 10 月開放後確認）**：從論文標題「fast and efficient」+「low-latency high-throughput」+ 前傳 WaferLLM 的 baseline 推斷：
  - **相對於 Cerebras 官方 CSL（Cerebras SDK 的 low-level language）hand-tuned kernel**：**接近但不超過** hand-tuned 的性能——這是 compiler vs hand-tuned kernel 的一貫故事，能拿到 90% 就是勝利
  - **相對於 WaferLLM 的手寫 MeshGEMM/MeshGEMV**：**編譯時間從「幾週人力」降到「幾分鐘 auto-schedule」**——這是 productivity 的量級跳躍，也是 compiler 對硬體平台的核心價值
  - **對新算子（例如 GQA / MLA / sliding window attention）的 time-to-first-run**：**從無到有** —— WaferLLM 沒 cover 這些算子，Wavel 應該可以自動編出來
  - **能夠 target 的算子空間**：從 WaferLLM 的 3–4 個手寫算子擴展到**任意 dense linear algebra + attention variant**
- **限制（推測 + 已知硬體限制）**：
  - **只 target dense compute**——sparse / gather-scatter / irregular access 目前 wafer-scale 硬體本身就不擅長（routing table 太小、SRAM 不夠 buffer 亂序訪問），Wavel 不會例外
  - **依賴 Cerebras / Edinburgh 私有 fabric spec**——這條線的實驗全部在 WSE-2/3 上做，沒法在 SambaNova / Tenstorrent 上驗證；PLMR model 概念上通用，但 concrete backend 是綁 Cerebras
  - **compile time 可能很長**——PLMR-aware scheduling 的 search space 大，論文可能有秒級或分鐘級的編譯時間；生產環境 serving 影響不大（AOT compile），但 iterative 開發時是痛點
  - **需要新的 debug / profiling 工具**——mesh scheduling bug 幾乎沒法從傳統 profiler 看出來（傳統 profiler 假設 shared memory），這是整個 wafer-scale 生態系都缺的一環
  - **assume static shape**——dynamic shape 在 wafer-scale 上非常難處理（partition 一變、routing table 就要重編），MeshRT 大概率也只 cover static shape + dynamic batch dim
- **它跟前面 compiler 系列的關係——把整個 stack 拉出 CUDA、對照第二條 accelerator 賽道**：
  - 8/25 [[cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026]]：**source language 層**（Mojo）——CUDA 賽道
  - 8/26 [[qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026]]：**IR 層**（Hexagon MLIR）——NPU 賽道
  - 8/27 [[hf-kernels-package-registry-cuda-distribution-layer-2026]]：**分發層**——CUDA 賽道
  - 8/28 [[tosa-block-scaled-mlir-mxfp-type-system-2026]]：**dtype IR 層**——共通
  - 8/30 [[kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check-2026]]：**benchmark 層**——CUDA 賽道
  - 8/31 [[argus-data-flow-invariants-llm-gpu-kernel-verified-2026]]：**verification 層**——CUDA 賽道
  - 9/2 [[cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026]]：**backend codegen 層**——CUDA 賽道
  - 9/3 [[flashlight-torchinductor-attention-compiler-graph-rewrites-mlsys2026]]：**middle-end pass 層**——CUDA 賽道
  - 9/4 [[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]]：**serving-loop 層**——CUDA 賽道
  - 9/5 [[syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026]]：**multi-GPU 通訊層**——CUDA 賽道
  - 9/6 [[llm42-verified-speculation-decode-verify-rollback-deterministic-llm-inference-sosp2026]]：**determinism / speculation 層**——通用
  - 9/10 [[morphkernel-cross-sm-fusion-just-in-time-reduction-dynamic-gpu-operators-sosp2026]]：**單 GPU 內動態排程層**——CUDA 賽道
  - **9/11（本篇）**：**wafer-scale 硬體 backend 層 + compile-time governed runtime**——**第三條賽道**（第二條是 NPU / Hexagon）
- **對走 compiler 職涯的 Adam，這篇有兩個層面的意義**：
  1. **技術層**：Wavel + MeshRT 是 [[Compiler-Path]] Stage 4 hardware backend layer 的最佳教材——把 PLMR model 讀熟、把 MeshGEMM 的 cyclic shift + interleaving 推導寫一遍、把 K-tree allreduce 的 √N critical path 想清楚，這三件事會讓你對「hardware topology 如何 shape compiler abstraction」有徹底不同的理解。
  2. **職涯策略層**：**如果 Adam 只投 CUDA 賽道（NVIDIA / Meta AI Infra / Google XLA / OpenAI），競爭壓力極大**——每個候選人都在寫 Triton kernel、都在讀 CUTLASS 源碼。**wafer-scale / mesh accelerator 賽道人才池小很多，但需求正在爆發**：Cerebras 剛 IPO、SambaNova 拿到大筆軍方合約、Tenstorrent 的 Wormhole 開始出貨、Groq 也在擴 team。**這是 Adam 值得認真評估的第二條 backup track**——不是要放棄 CUDA，而是同時把 wafer-scale/mesh compiler 這條線建立起來，讓履歷不會只有一種 story。細節放在最後一節。

---

## 1. 為什麼要現在關注 wafer-scale，而不是繼續看 CUDA

先講一個 macro 觀察，這個觀察決定了為什麼我今天要把 blog 從 CUDA 系列拉出來寫 wafer-scale：

**2020–2024 是 CUDA compiler 賽道的黃金時期**。這段時間裡：

- Triton 從 OpenAI 研究專案變成 PyTorch 官方 kernel DSL
- TorchInductor 從實驗變成 PyTorch 2.x 的 default backend
- FlashAttention v1/v2/v3 一路把 attention kernel 的抽象定型
- CUTLASS 3.x 引入 CuTeDSL、變成 Blackwell 的 first-class backend
- MLIR 進入 XLA、進入 Triton、進入 IREE、進入所有主流 ML framework

**這條賽道現在的問題不是「有沒有事做」，而是「人太多、突破空間變小」**：

- Triton 已經被無數 team fork/extend，新突破的門檻越來越高
- TorchInductor 的 pass 已經被 Meta 內部 + 外部貢獻者填得很滿
- FlashAttention 系列的 formulation 已經被逼近極限（v4 大部分優化都是 numeric / calibration）
- 大部分 CUDA compiler engineer 的日常工作變成「維護、調 pass 順序、fix regression」，而不是「開新 primitive」

**同時，硬體那邊在悄悄起第二波革命**——不是 NVIDIA 內部升級（H100 → H200 → B100 → GB200 都是同一個 abstract machine），而是**完全不同 abstract machine 的商業化**：

| 硬體 | Abstract Machine | Compile Target | 2026 商業狀態 |
|---|---|---|---|
| NVIDIA H100/B100 | 144 SM，crossbar，unified HBM | PTX、Triton、CUTLASS | 主流；競爭激烈 |
| AMD MI300X | 304 CU，Infinity Fabric | ROCm、HIP、rocMLIR | 上升期；缺 kernel |
| Cerebras WSE-3 | 900k+ core，2D mesh，distributed SRAM | CSL、Wavel（新）| IPO 後放量 |
| SambaNova SN40L | Reconfigurable dataflow，spatial | SambaFlow | 軍方 + enterprise |
| Tenstorrent Wormhole | Grid of Tensix cores，mesh | TT-Metalium、TT-Buda | 出貨中 |
| Groq LPU | Deterministic dataflow，no cache | GroqCompiler | Serving 專用；scaling |

**這張表的重點：從 Cerebras 往下的四家，abstract machine 都不是 CUDA 那個 abstract machine**。它們共同特徵：

- **沒有 shared memory**（各 core 只有 local SRAM）
- **明確的 physical topology**（2D mesh、3D torus、dataflow graph）
- **communication cost >> compute cost** 的優化目標（跟 GPU 剛好相反）
- **compiler 是 first-class 而非 optional**——沒有 compiler 就沒法用，不像 CUDA 還可以繞過 nvcc 手寫 PTX

**這代表未來 5–10 年，非 CUDA accelerator 的 compiler engineer 是稀缺人才**。不是說 CUDA 沒工作了——CUDA compiler 永遠有工作，但競爭激烈；wafer-scale / dataflow / spatial accelerator compiler 是新開的市場，門檻高（要懂硬體 physical constraint），但人少。

**Wavel + MeshRT 是這個市場的第一批系統性論文之一**——這就是為什麼今天要寫。

## 2. WaferLLM 前傳：Cerebras WSE-2 到底是什麼硬體

要看懂 Wavel，先要搞清楚它在 target 什麼硬體。這節把 Cerebras WSE-2/3 的 abstract machine 拆到位。

### 2.1 硬體規格

**Cerebras WSE-2（2021 發布，2025–2026 仍是主力）**：

- **850,000 個 core**，每個 core 跑 1.1 GHz
- 每個 core 有 **48KB local SRAM**——注意是 SRAM，不是 cache，是 addressable memory
- **總 on-chip memory 40GB**——這是 SRAM 直接堆在 die 上，沒有 HBM
- **總聚合記憶體頻寬 20 PB/s**——這個數字比 H100 高約 1000×，因為它是「每個 core 各自 access 自己的 48KB 的頻寬總和」
- **2D mesh interconnect**：core 排列成大約 1000×850 的網格，每個 core 只跟東南西北 4 個鄰居直接連
- **Router constraint**：每個 core 的 router 用 5-bit destination address → 最多 32 個 direct destination

**WSE-3（2024 發布，2026 開始放量）**：

- **900,000+ core**（規格微幅升級）
- **44GB on-chip SRAM**
- **21 PB/s memory bandwidth**
- **同樣的 2D mesh 結構，routing constraint 類似**
- 支援 FP8 / FP16 / BF16 / INT8

**CS-6（2026–2027 roadmap，Hot Chips 2026 揭露）**：

- 開始整合 **stacked DRAM**——這是 wafer 上第一次有 tiered memory
- Nexus rack-scale 系統：多顆 WSE 串接，rack-level 三倍性能

### 2.2 為什麼 GPU 開發者第一次看 WSE-2 spec 會震撼

如果你只熟 GPU，第一次看 WSE-2 規格會被兩個數字震到：

**震撼 1：40GB 全 on-chip SRAM**

H100 的 SRAM 總共大約 60MB（L1 + L2 加起來），HBM 是 80GB 但頻寬 3 TB/s。WSE-2 是 **40GB SRAM，頻寬 20 PB/s**——SRAM 的頻寬跟 GPU 的 HBM 頻寬比是 6700×。

這意味著在 WSE-2 上，**只要你能把 tensor 放進去，memory bandwidth 就不是瓶頸**——這是 GPU 世界完全不存在的優勢，也是 WaferLLM 能拿到 GEMV 606× A100 加速的根本原因（GEMV 是 memory-bound）。

**震撼 2：850,000 個 core**

H100 有 144 個 SM、每個 SM 有 128 個 CUDA core，共 18432 個 CUDA core。WSE-2 有 850,000 個 core——比 H100 多 46× 的 core 數。

但每個 WSE-2 core 只跑 1.1 GHz、只有 48KB SRAM、單週期能做的 FMA 有限。**這代表 WSE-2 走的是「極端 fine-grained parallelism」路線**——你要把工作切到 850k 份、每份都極小；而 GPU 走的是「moderate parallelism + wide SIMT」路線。

### 2.3 mesh 拓樸的物理意義

WSE-2 的 850k core 排成大約 1000×850 的 2D mesh。這個 mesh 的物理意義：

**任兩個 core 之間的通訊延遲 = 曼哈頓距離 × per-hop latency**

- 相鄰 core（距離 1）：1 個 cycle
- 距離 100：100 個 cycle
- 對角相反（距離 1850）：1850 個 cycle

**這代表「core 距離」是 first-class cost dimension**。GPU compiler 完全不需要考慮這件事——GPU 上任兩個 SM 到共享 L2 的距離大致相同（crossbar 提供 uniform access）。

**Wavel 的 IR 必須把「tensor 切到哪些 core」跟「這些 core 之間的距離」都 model 進去**，否則排程結果會慘不忍睹（例如把一個 all-reduce 排到跨對角的 core group，通訊時間就爆炸）。

### 2.4 routing constraint 的殺傷力

WSE-2 的 router 用 5-bit destination address，這代表 **每個 core 最多同時追蹤 32 個 outgoing route**。

這件事的含義：

- **你不能做任意 all-to-all**——例如經典的 SUMMA GEMM 或 Cannon GEMM，每個 core 要往 O(N) 個其他 core 傳資料，在 WSE-2 上根本 route 不出去
- **你必須設計「local + shift」的通訊 pattern**——資料在 mesh 上一步步 shift 出去，每步只用 4 個 route（東南西北）
- **這是 WaferLLM 為什麼要重新發明 MeshGEMM / MeshGEMV 的原因**——傳統分布式 GEMM 演算法全部假設 all-to-all 或 tree-reduce，這兩個假設在 WSE-2 上都不成立

### 2.5 為什麼這件事是 compiler 問題而不是 runtime 問題

GPU 上，很多 layout 決策可以 runtime 調——例如 tile size、warp 分配、L2 residency hint。**Wafer-scale 上不行**：

- **每個 tensor 的 mesh placement 一決定，就要編出對應的 route table**——這個 route table 是硬體級的 configuration，不能 runtime 動
- **每個 core 的 SRAM 分配也是靜態的**——48KB 要 pre-partition 給不同 tensor 用，runtime 不能 malloc
- **routing 的 shift pattern 也是編譯期定死的**——WaferLLM 的 shift-based KV cache 就是把整個 KV shift 順序在編譯期展開

**這代表 compiler 在 wafer-scale 上的角色比在 GPU 上重得多**——GPU 沒有好 compiler 你還可以手寫 PTX / CUDA，wafer-scale 沒有 compiler 你連 hello world 都寫不出來。這就是 Wavel + MeshRT 為什麼是 first-class 論文（在 SOSP 這種頂級 systems 會議一次收兩篇），而不只是 workshop paper。

## 3. WaferLLM (OSDI 2025) 做了什麼，Wavel 要繼承什麼

Wavel + MeshRT 是 WaferLLM 的正統續作，所以要先看 WaferLLM 做完了什麼、剩下什麼問題留給 Wavel 解。

### 3.1 WaferLLM 的三個核心 algorithmic contribution

**MeshGEMM——用 cyclic shifting + interleaving 把 GEMM 塞進 mesh**：

傳統分布式 GEMM 有三大流派：
- **allgather-based**：先把 A、B 全 gather 到每個 node，再 local compute。通訊量 O(N²)，wafer-scale 上會撐爆 routing。
- **SUMMA**：outer product，broadcast row + column。仍然需要 O(N) 通訊。
- **Cannon**：shift-based，但需要對角 skew，需要 O(N) preparation shift。

MeshGEMM 的做法：

1. **cyclic shifting**——把 A 沿 row 方向 shift、B 沿 column 方向 shift，每步只往鄰居送資料
2. **interleaving**——A 跟 B 的 shift 錯開一格，讓每個 core 每個 cycle 都有 useful compute
3. **bounded memory**——因為每個 tile 每次只在少數 core 停留，48KB SRAM 撐得住
4. **critical path constant 2-hop**——最壞情況下，reduce 只跨 2 個鄰居

**結果**：比 SUMMA 快 2–3×，跟 Cannon 打平但不需要 preparation phase，並且符合 PLMR 全部四個 constraint。

**MeshGEMV——K-tree allreduce 取代 pipeline / ring**：

GEMV 的 reduce 是重點。傳統 pipeline reduce 是 O(N) hops，ring reduce 也是 O(N)。

MeshGEMV 的做法：

1. **K-tree structure**——把 reduce 組織成 balanced K-ary tree，K 是每個 core 能同時接受的 incoming direction 數（在 mesh 上通常 K=4）
2. **parallel reduce**——tree 的每一層都平行做
3. **√N critical path**——mesh 上 K-tree 的高度是 O(log_K(N))，但每步要跨越的物理距離平均 O(√N/log_K(N))，總 critical path O(√N)
4. **flexible routing tradeoff**——K 可調，K 越大 routing pressure 越高但 depth 越淺，可以 tune

**結果**：比 Cerebras 官方 CSL 的 GEMV 快 4–8×，比 A100 上 cuBLAS 的 GEMV 快 606×（因為 A100 上 GEMV 完全 memory-bound，WSE-2 上 memory 不是瓶頸）。

**Shift-based KV cache——把 KV 分散並且 shift 出去，不 concat**：

GPU 上 KV cache 是 concat：每步 decode 新產生 K、V 就 append 到 buffer 末尾。這個 pattern 在 WSE-2 上不行——沒有中央 buffer 可以 append。

WaferLLM 的做法：**每個 core 持有 KV 的一部分，新增 token 時整個 KV 沿 mesh shift 一格**，讓每個 core 都持續持有連續的 KV 子集。

**結果**：**每顆 WSE-2 可以持有的 KV 容量比 GPU concat 大 360–385×**——這是 wafer-scale 對 long-context inference 的殺手級優勢。

### 3.2 WaferLLM 沒解決的三件事——Wavel 要接手

WaferLLM 是實作論文，不是 compiler 論文，所以留下三個 open problem：

**問題 1：所有 algorithm 都是手推的**

MeshGEMM、MeshGEMV、shift-based KV cache 這三件事，是研究員花數月從紙上推導 → 手寫 CSL kernel → 手工調 routing table 做出來的。**換一個新算子（例如 GQA、MLA、sliding window attention、FlashAttention v3），全部要重推一次**。

**問題 2：沒有 IR、沒有 auto-scheduler**

WaferLLM 的實作就是一堆 hand-tuned CSL kernel + 一些 Python wrapper。沒有中間 IR、沒有 lowering pipeline、沒有 cost model、沒有 auto-scheduler。**這就是「工業級 platform 之前」的狀態**——像 GPU 世界的 2007 年（CUDA 1.0 剛出，Thrust 都沒有）。

**問題 3：runtime 是原始的**

WaferLLM 的 serving 端也很原始——一個 request 進來，跑一個 batch，出去。沒有 continuous batching、沒有 speculative decoding、沒有 prefill/decode 分離。**這些 GPU 世界已經標準化的 serving 技術，在 wafer-scale 上還沒被實作**——因為缺 runtime 抽象。

**Wavel 解問題 1、2，MeshRT 解問題 3**。這就是為什麼這兩篇是一整套 stack。

## 4. Wavel 的推斷架構——mesh-first IR + PLMR-aware scheduling

正式論文 10 月 open review 之後才能看到 IR 細節，但根據論文標題「fast and efficient compilation system」+ 前傳 WaferLLM + Luo Mai group 之前一系列 workshop 論文，我推斷 Wavel 的架構大致長這樣：

### 4.1 Mesh-first IR

Wavel 的 IR 一定要把「mesh placement」拉到 first-class type。這件事在其他 compiler 都是 secondary：

**GPU compiler（Triton / XLA）**：
```
tensor<128x128xf16>   # 沒有 placement，由 lowering pass 決定
```

**Distributed compiler（XLA SPMD / GSPMD）**：
```
tensor<128x128xf16> {devices=[2,4]0,1,2,3,4,5,6,7}
# 有 device mesh 但只有 8 個 device
```

**Wavel 推斷 IR**：
```
tensor<128x128xf16> mesh<row=[0..127], col=[0..127], stride=1>
# core mesh，每個 core 拿到一小塊
```

**關鍵差異**：Wavel 的 mesh dimension 是**上千維**（因為有上百萬 core），而且 mesh coordinate 直接對應物理 core 位置。這代表 IR verifier 要檢查「這個 placement 在物理 mesh 上是否合法」——例如某些 core 可能因為 defect 被 disable（DLR: Efficient Defect-Tolerant Routing at Wafer Scale 這篇論文專門處理），Wavel 的 IR 必須知道哪些 core 可用。

### 4.2 PLMR-aware scheduling pass

Wavel 的 auto-scheduler 一定要 model PLMR 四個 constraint：

**Cost model 的維度（推斷）**：

| 維度 | GPU compiler | Wavel |
|---|---|---|
| FLOP | ✓ | ✓ |
| Memory bandwidth | ✓（HBM）| ✓（local SRAM）|
| Cache pressure | ✓（L1/L2）| —（no cache）|
| Mesh distance | — | **✓** |
| Routing table pressure | — | **✓** |
| Per-core SRAM allocation | —（malloc）| **✓（static）** |

多出來的三個維度讓 search space 爆炸。Wavel 的挑戰是**怎麼在合理時間內找到 near-optimal schedule**——直觀猜想：他們可能用類似 TVM AutoScheduler 的 evolutionary search，或者 ML-guided cost model（這是 Lingxiao Ma 之前在 Rammer、Roller 系列論文的專長）。

### 4.3 Communication rewriting pass

這是最有價值的一步。輸入是高階 collective operation（allgather、allreduce、broadcast），輸出是 mesh-friendly shift-and-accumulate pattern：

**輸入 IR（推斷）**：
```
%y = allreduce %x : tensor<1024xf32> mesh<row=[0..127]>
```

**Wavel 自動 rewrite 為（推斷）**：
```
# K-tree allreduce, K=4
%stage0 = shift %x, direction=north, step=1
%stage0_acc = add %x, %stage0
%stage1 = shift %stage0_acc, direction=east, step=2
%stage1_acc = add %stage0_acc, %stage1
...  # log_4(128) = 3.5 stages
```

**這一步是 WaferLLM 手工推導的內容變成自動化的關鍵**。如果 Wavel 能做到這步 auto-rewrite，那新算子（例如 GQA、MLA）的支援就不需要研究員再推一次。

### 4.4 Backend：生成 CSL / 生成 mesh binary

最後 lowering 到 Cerebras SDK 的 CSL（Cerebras Software Language）或直接生 mesh binary。這一層跟 GPU compiler lower 到 PTX 概念類似，只是 target 完全不同。

## 5. MeshRT——「compile-time governed runtime」是新一格

MeshRT 是這對論文裡我覺得更有趣的一篇。原因：**它引入了一個新的 systems abstraction，值得記住**。

### 5.1 傳統 runtime 的問題

GPU serving 的 runtime（vLLM、TensorRT-LLM、SGLang）都做同一件事：**runtime 動態決定執行**。

- 動態 batching：runtime 決定哪些 request 一起跑
- 動態 KV allocation：runtime 決定 PagedAttention 的 page 分配
- 動態 kernel launch：runtime 決定用哪個 kernel variant

這在 GPU 上可行，因為 GPU 有 shared memory 跟 unified HBM，runtime 動態決策不會破壞硬體 configuration。

**Wafer-scale 上這行不通**——每次 dynamic 決策都可能要重編 route table，而 route table 是硬體級 config，動不了。

### 5.2 MeshRT 的核心觀察：「動態」的可能性空間是有限的

serving 的動態行為看似無窮，其實可以 enumerate：

- batch size 有限（比如 1, 4, 16, 64 這幾個）
- sequence length 分桶（比如 <128, <512, <2048, <8192）
- request 類型有限（prefill、decode、speculative verify）

**如果把這些 dimension cross product 起來，總共可能的執行 configuration 是幾十到幾百個**——不是無窮多。

**MeshRT 的做法**：在 compile time **把所有可能的 configuration 都編出來**，每個 configuration 對應一組預先計算好的 mesh placement + route table + shift schedule。**Runtime 只做一件事：看目前 request 屬於哪個 configuration，切換到對應的 state**。

這就是「compile-time governed runtime」的意思——**runtime 的行為完全被 compiler 事先決定**，runtime 是消費 compile-time artifact 的 state machine。

### 5.3 這個抽象的深遠意義

**這不只是 wafer-scale 的技巧，是可以推廣到任何硬體的 systems abstraction**：

- 對 GPU serving——可以把 vLLM 動態 batching 的 decision tree 提前編出來，減少 runtime overhead（這正好是 [[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]] 在做的事）
- 對 embedded / edge inference——static schedule 已經是預設，MeshRT 的抽象讓 static schedule 可以帶「有限的動態切換」
- 對 real-time system（機器人、自駕）——這是天然合適的模型，因為 hard real-time 就是要 avoid runtime decision

**Adam 特別注意這個抽象**——不管你將來走 CUDA compiler、embedded compiler、還是 wafer-scale compiler，「compile-time enumerate + runtime consume state」是一種可以帶著走的思考工具。

### 5.4 MeshRT 的 latency / throughput tradeoff

論文標題強調「low-latency high-throughput」——這兩件事在 serving 系統通常是矛盾的（batch 越大 throughput 越高、latency 越差）。MeshRT 應該有機制在同一個編譯輸出裡 tune 這個 tradeoff：

**推斷做法**：compile time 產生多個 static schedule，每個 optimize 不同 point（低 latency、高 throughput、平衡）；runtime 根據 SLO 選擇對應 schedule。

這件事在 GPU 上也可以做，但沒人做——因為 GPU runtime 動態決策 overhead 不高、tune 不划算。**在 wafer-scale 上做，就變成核心 architecture**——這就是「不同硬體強迫不同 systems abstraction」的典型例子。

## 6. 對照 GPU compiler stack，wafer-scale compiler 的獨特之處

用一張表把兩條賽道對照一次，方便 Adam 記住 mental model：

| 層 | GPU（CUDA / Triton / TorchInductor） | Wafer-Scale（Wavel + MeshRT） |
|---|---|---|
| **Frontend** | PyTorch → StableHLO / TorchInductor IR | 待確定，可能複用 StableHLO |
| **High-level IR** | linalg / HLO | linalg + mesh placement metadata |
| **Middle IR** | Triton IR / gpu.func | Wavel IR（mesh-aware）|
| **Scheduling primitive** | tile size、warp、block | core partition、mesh distance |
| **Cost model dim** | FLOP、bandwidth、cache | + mesh distance、routing |
| **Communication IR** | NCCL（跨 GPU）、shared memory（GPU 內）| shift-based mesh routing |
| **Low IR** | PTX / NVVM | CSL / mesh binary |
| **Runtime** | vLLM / TensorRT-LLM | MeshRT（compile-time governed）|
| **Serving abstraction** | continuous batching、PagedAttention | static schedule + configuration state machine |
| **Debug** | Nsight、Compute Sanitizer | 缺（需要新工具）|
| **Auto-scheduling** | AutoTVM、Ansor、Roller | Wavel 自帶（推斷）|

**看這張表要抓兩件事**：

1. **Wavel + MeshRT 是把 GPU compiler 大部分抽象「重寫一遍」的工作**——不是抄 CUDA，而是每一層重新設計以匹配 mesh 硬體
2. **每一層的差異都不是 minor，是 abstract machine 級別的不同**——這就是為什麼 wafer-scale compiler engineer 需要重新學一整套 mental model

## 7. 這篇對 Adam 的 Compiler-Path 意味什麼

回到職涯策略。前面 11 篇 compiler 系列文章我都在拉 [[Compiler-Path]] 對照，這篇要拉得更明確——因為 wafer-scale 是「第二條 backup track」。

### 7.1 Compiler-Path 的四階段回顧

按之前 [[Compiler-Path]] 的規劃：

- **Stage 1（現在－2 個月）**：基本功——LLVM tutorial、TVM tutorial、Triton tutorial
- **Stage 2（2–6 個月）**：pass infrastructure——寫 MLIR pass、寫 TorchInductor pass、讀 Triton compiler 源碼
- **Stage 3（6–12 個月）**：runtime / scheduling layer——這是 [[morphkernel-cross-sm-fusion-just-in-time-reduction-dynamic-gpu-operators-sosp2026]] 那個層次
- **Stage 4（12–18 個月）**：hardware backend layer——targeting 具體硬體、寫 codegen

**Wavel + MeshRT 是 Stage 4 的最佳學習材料**——特別是「hardware topology 如何 shape compiler abstraction」這件事，wafer-scale 是最極端的教材，比 GPU 更能凸顯這個原則。

### 7.2 為什麼要建立第二條 backup track

**現實觀察**：

- 純 CUDA compiler 職缺（NVIDIA、Meta AI Infra、Google XLA、OpenAI infra）競爭極端激烈，且大多要 5+ 年 GPU 底層經驗
- 每個候選人履歷上都會有 Triton fork、CUTLASS 貢獻、FlashAttention 實作——**很難差異化**
- Adam 30 歲、目前在 Foxconn 做 LiDAR 演算法——**沒有 GPU 底層 credential**，走純 CUDA compiler track 需要 2–3 年 heads-down 才有面試門票

**Wafer-scale / mesh accelerator compiler 賽道**：

- **人才池小得多**——全球做過 wafer-scale compiler production 工作的工程師可能不到 200 人
- **公司多、需求增長**：Cerebras（IPO 後擴 team）、SambaNova（軍方合約）、Tenstorrent（出貨中）、Groq（scaling）、+ 一些 stealth 新創
- **入門門檻高但可跨**：需要熟 mesh interconnect、PLMR model、shift-based algorithm——這些東西**只要花 2–3 個月讀論文 + 動手實驗就能達到面試級**
- **競爭者少**：大部分 compiler engineer 沒去碰 wafer-scale，因為硬體不好拿——Cerebras 有 university partnership，Tenstorrent 有 devkit（$999 起）

### 7.3 具體行動建議

如果 Adam 要開始建立這條 track：

**Month 1**：
- 讀完 WaferLLM (OSDI 2025) 全文
- 讀完 Wavel + MeshRT (10 月開放後)
- 讀完 Luo Mai 的 [Wafer-Scale AI Compute: A System Software Perspective](https://www.sigops.org/2025/wafer-scale-ai-compute-a-system-software-perspective/)
- 讀完 Cerebras 官方 CSL 文檔

**Month 2**：
- 申請 Cerebras Model Zoo access 或用 SDK simulator
- 或買一顆 Tenstorrent Wormhole n300 devkit（$1400 左右）——實體硬體對 mesh compiler learning 幫助巨大
- 動手實作一個小 MeshGEMM 或 K-tree allreduce（不用真跑，用 Python 模擬 mesh 也可以）

**Month 3**：
- 選一個 open problem——例如「Wavel 沒 cover MLA attention 怎麼補上」——寫成一篇 blog 或投一個 workshop
- 把這件事變成履歷上的 credential

**這條路徑 3 個月成本 = 一顆 devkit + 每天 2 小時 = 幾千塊 + 180 小時**。回報是履歷上多一個「wafer-scale compiler」的 signal——對 Cerebras / SambaNova / Tenstorrent 的 hiring team 是 direct signal，對 NVIDIA / Meta 的 hiring team 是 "此人懂 hardware-software co-design" 的間接 signal。

**這不是要 Adam 放棄 CUDA**——而是同時 build 兩條 track，讓履歷 story 從「另一個 CUDA compiler wannabe」變成「懂 GPU 也懂 non-GPU accelerator 的 systems generalist」——這在 senior hire 面試裡是完全不同的定位。

### 7.4 一個具體的差異化 project 建議

**如果 Adam 要挑一個 spike project 展現 wafer-scale compiler 能力**：

「**在 Python 模擬環境裡實作一個 MLA attention 的 mesh-aware compilation**」

- 用一個 Python-based mesh simulator（可以自己寫，也有一些 open-source 的）
- 定義 MLA attention 的 dataflow graph
- 手動或半自動 partition 到 128×128 core mesh 上
- 實作 shift-based data movement
- Measure critical path、routing table pressure
- 寫成一篇 blog post，比較「naive placement」跟「PLMR-aware placement」的差距

**這件事一個週末就能起頭，兩週能出 v1**——結果會非常 visualisable（mesh 上的資料流動可以動畫化），blog post 會很有 shareability。

## 8. 我對這對論文的批判性看法

不是每篇 SOSP 論文都要吹捧，這裡列一些我覺得該挑戰的點：

**1. Cerebras-locked**——Wavel 的 backend 只 target Cerebras CSL / mesh binary。SambaNova、Tenstorrent 的 mesh 拓樸不同、routing constraint 不同、SRAM 大小不同——**Wavel 的 IR 抽象能不能推廣是開放問題**。如果最後只 target Cerebras，這篇的影響力就會被綁死在 Cerebras 商業命運上。

**2. compile time 恐怕不便宜**——PLMR-aware auto-scheduling 的 search space 巨大，論文可能會展示 minutes 級的 compile time。**對 iterative 開發（例如 LLM researcher trying new attention variant）是重痛點**。GPU 世界 Triton JIT 一次幾秒；wafer-scale 如果 JIT 需要 minutes，體驗會差一個量級。

**3. debug 工具缺失**——mesh scheduling bug 是全新的 bug 類別，現有 profiler 完全不 cover。論文可能會 hand-wave 過去，但實際部署時這是最大痛點。**這也是一個 open opportunity**——寫 mesh profiler 是一個好 startup 種子。

**4. dynamic shape 大概率不 cover**——MeshRT 的「compile-time enumerate configuration」對 discrete dynamic dim（batch size、seq length bucket）可行，對 continuous dynamic dim（例如 image size 完全自由）就會 combinatorial explosion。**這是 wafer-scale 目前打不動 vision workload 的原因之一**。

**5. 商業 traction 未知**——Cerebras 剛 IPO，市值起伏很大；SambaNova 主要靠政府合約；Tenstorrent 還在 scaling。**這條 hardware track 三年內會不會活下來是真實風險**——Adam 建 second track 時要 hedge，不能全押一家硬體。

即使有這五個 concern，Wavel + MeshRT 仍然是**這波 wafer-scale compiler 最完整的一次系統工作**。它不是最終答案，是打開這條賽道的第一批系統性論文。

## 9. 這一系列 12 篇看下來的 meta-observation

寫到第 12 篇，該回望整個 series 的 shape 了：

- **前 6 篇（8/25–8/31）**：source language、IR、分發、dtype、benchmark、verification——**基礎設施層**
- **中 4 篇（9/2–9/5）**：codegen、middle-end pass、serving loop、multi-GPU 通訊——**GPU compiler 主戰場**
- **後 2 篇（9/6, 9/10）**：determinism / speculation、kernel 內動態排程——**GPU compiler 邊界擴展**
- **今天（9/11）**：wafer-scale hardware backend——**跳出 GPU、開第二條賽道**

**這個 shape 剛好對應 [[Compiler-Path]] Stage 1 → Stage 4 的完整覆蓋**——不是巧合，是我在寫的時候有意識地按這個順序推進。**下週開始，series 會轉入 robotics / AV compiler 主題（TOSA、Autoware、ROS 2 real-time）——因為 Adam 現職做 LiDAR、往 compiler 走的過程需要展示「domain expertise」，不能只寫 GPU**。

**對 Adam 的具體建議**：把這 12 篇當 curriculum 讀——不是「有空看看」，是**每篇讀 2 小時、寫 200 字 own reflection、跟 [[Compiler-Path]] 的對應階段對照**。3 個月讀完，Compiler-Path Stage 1 + 一半 Stage 2 就達成了。這比從 tutorial 開始學快得多，因為每篇論文都是「壓縮過的 domain knowledge」。

---

## 參考資料

**主論文（SOSP 2026，正式 proceedings 預計 10 月）**：
- Wavel: A Fast and Efficient Compilation System for Wafer-Scale Accelerators — Yeqi Huang, Congjie He, Haocheng Xiao, et al. (Edinburgh + MSR + Luo Mai)
- MeshRT: Compile-Time Governed Wafer-Scale Runtime for Low-Latency High-Throughput Inference — Congjie He, Le Xu, Zhan Lu, et al. (Edinburgh + MSR + Luo Mai)
- SOSP 2026 accepted papers list: [https://sigops.org/s/conferences/sosp/2026/accepted.html](https://sigops.org/s/conferences/sosp/2026/accepted.html)

**前傳論文**：
- WaferLLM: Large Language Model Inference at Wafer Scale (OSDI 2025) — Congjie He, Yeqi Huang, Pei Mu, Ziming Miao, Jilong Xue, Lingxiao Ma, Fan Yang, Luo Mai. [arXiv:2502.04563](https://arxiv.org/abs/2502.04563), [代碼](https://github.com/MeshInfra/WaferLLM)

**背景 / 綜述**：
- Wafer-Scale AI Compute: A System Software Perspective — Luo Mai et al., USENIX ;login: 2025, [連結](https://www.sigops.org/2025/wafer-scale-ai-compute-a-system-software-perspective/)
- Wafer-scale Computing: Advancements, Challenges, and Future Perspectives — [arXiv:2310.09568](https://arxiv.org/abs/2310.09568)

**其他 wafer-scale LLM inference 工作**：
- MoEntwine: Unleashing the Potential of Wafer-scale Chips for Large-scale Expert Parallel Inference — [arXiv:2510.25258](https://arxiv.org/abs/2510.25258)
- MOCAP: Wafer-Scale-Chip-Oriented Memory-Orchestrated Chunked Pipelining Framework for Prefill-Only LLM Inference
- Ouroboros: Wafer-Scale SRAM CIM with Token-Grained Pipelining for Large Language Model Inference

**Cerebras 產品與 roadmap**：
- Cerebras Hot Chips 2026: Nexus system architecture, CS-6 wafer with stacked DRAM — [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/hot-chips-2026-cerebras-lays-out-the-future-of-wafer-scale-ai-nexus-system-architecture-triples-rack-scale-performance-cs-6-wafer-to-incorporate-stacked-dram)
- Cerebras IPO May 2026 — [The Register](https://www.theregister.com/ai-ml/2026/05/15/cerebras-wafer-scale-ai-bet-delivers-blockbuster-ipo/5240821)

**作者主頁**：
- Luo Mai (University of Edinburgh): [https://luomai.github.io/](https://luomai.github.io/)

**同 series 之前的文章**：
- [[cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026]]
- [[qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026]]
- [[hf-kernels-package-registry-cuda-distribution-layer-2026]]
- [[tosa-block-scaled-mlir-mxfp-type-system-2026]]
- [[kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check-2026]]
- [[argus-data-flow-invariants-llm-gpu-kernel-verified-2026]]
- [[cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026]]
- [[flashlight-torchinductor-attention-compiler-graph-rewrites-mlsys2026]]
- [[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]]
- [[syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026]]
- [[llm42-verified-speculation-decode-verify-rollback-deterministic-llm-inference-sosp2026]]
- [[morphkernel-cross-sm-fusion-just-in-time-reduction-dynamic-gpu-operators-sosp2026]]

**Compiler-Path 對照**：
- [[Compiler-Path]] Stage 4 — hardware backend layer
