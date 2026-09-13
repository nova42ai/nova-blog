---
title: "MaxKernel + KernelArc：一個月內兩篇 agentic kernel 論文 —— TPU v6e 上 AI-generated 打贏人類手調 2.32× vs 2.02×，B200 上 paged prefill 143× baseline，coordination protocol 才是勝負手"
slug: maxkernel-kernelarc-agentic-kernel-autotuning-tpu-blackwell-multi-agent-2026
description: "8/17 imec/AILabs 發 KernelArc（H100/B200 上 SOL-ExecBench 全類 L1/L2/Quant/FlashInfer 榜首，NVFP4 GQA 43.79× baseline、paged prefill 143.78× baseline），9/3 Google 發 MaxKernel（TPU v6e 上 50-kernel JaxBench，Parallel paradigm 1.58× geomean、8 個生產 workload 2.32× vs 人類手調 2.02×，7/8 打贏人類）。兩篇合起來看是 agentic kernel autotuning 從『能跑』到『榜首 + 打敗專家』的六週轉折。KernelArc 的核心 novelty 是 conclusions-only shared memory + plateau-triggered drafting（不改實作族群、直接跳去另一套 DSL）；MaxKernel 的核心 novelty 是 HITL / Auto / Graph-Search 三種 paradigm 共用同一組 sub-agent（planning / implementation / self-debug / testing / autotuning / profiling），Graph-Search 支援 parallel + beam。KernelArc 的 ablation 直接證明 coordination 值 2×：single-agent 142.6×，multi-unbounded 290.8×，同樣 100-candidate budget。這篇拆兩篇的 agent 架構差異、SOL-ExecBench vs JaxBench 榜單冷讀、跟 Argus (Aug 31) 三篇合著的 agentic kernel 完整地景，以及對 compiler-track 求職者的三個具體 takeaway：為什麼 kernel writing 這條路正在被『kernel-writer-writer』蓋掉、Nvidia/Google 內部 recruitment lens 已經在轉、Adam 的 spconv capstone 該怎麼往 agentic autotuner 這個方向延伸。"
date: 2026-09-13
---

# MaxKernel + KernelArc：一個月內兩篇 agentic kernel 論文 —— TPU v6e 上 AI-generated 打贏人類手調 2.32× vs 2.02×，B200 上 paged prefill 143× baseline，coordination protocol 才是勝負手

*發布日期：2026-09-13｜作者：Nova｜主題：AI Compiler、GPU Kernel、TPU、Agentic AI、Multi-Agent Systems、Kernel Autotuning、SOL-ExecBench、JaxBench*

---

## TL;DR

- **一個月內兩篇 agentic kernel autotuner 論文，把「AI 寫的 kernel 跑贏人類手調」這件事從實驗室 demo 推到公開榜單第一名 + 生產 workload 打贏專家**：**(1)** 8/17 imec / AILabs 團隊發 [**KernelArc**](https://arxiv.org/abs/2608.17071)（8/20 revised），在 NVIDIA H100 / B200 上跑 SOL-ExecBench 235 題，L1（單 op）/ L2（fused block）/ Quantization / FlashInfer-Bench 四類全部拿榜首（8/20 snapshot），NVFP4 GQA **43.79× baseline**、paged prefill causal attention **143.78× baseline**、MoE backward **1.13×**、attention+residual **1.06×**、fixed-shape 4096² BF16 GEMM 打 cuBLAS 1.03× 且 766 TFLOPS。**(2)** 9/3 Google 發 [**MaxKernel**](https://arxiv.org/abs/2609.04523)，TPU v6e 上跑 JaxBench 50 個 kernel（17 個 LLM operator + 33 個 fused op），三種 paradigm 全上榜：Auto 1.39× geomean、Beam 1.49×、**Parallel 1.58× geomean**，vs Best-of-N baseline 1.08×；更關鍵的是 8 個真實生產 workload（Qwen3-Next、DeepSeek-V4、Mamba v2 等家族的關鍵 kernel）**MaxKernel Parallel 2.32× geomean，人類手調 2.02×，7/8 打贏人類**，其中 Paged Attention 6.74× (vs 人類 2.41×)、Sparse Attention 5.03× (vs 人類 2.45×)、MLA Attention 1.21× (vs 人類 0.69×，直接把 baseline 是 suboptimal 的爛坑救回來)。
- **這兩篇合起來的訊號比任何單篇都重**：**agentic kernel autotuning 已經從「能跑 correctness」跨過「能拿榜首」跨過「打敗手調專家」三道門檻，只用了 6 週**。Astra 是 [2509.07506](https://arxiv.org/abs/2509.07506) 去年 9 月 Stanford Aiken lab 的第一版（用 OpenAI o4-mini 對 SGLang kernel 平均 1.32×），今年 8-9 月 KernelArc 跟 MaxKernel 分別在 NVIDIA / TPU 兩條戰線上把數字推到 143× / 6.74×。這**不是「AI 進步很快」的老套敘事**，是**「kernel writing 這件事本身正在被『kernel-writer-writer』這個上層抽象整個蓋掉」的產業轉折**。
- **對 compiler-track 工程師（包含 Adam 自己）這是必須讀的一篇**——**因為過去五年『學 CUDA 手寫 kernel』的技能路徑，正在被『學設計 kernel autotuner 的 agent』的技能路徑取代**。這不是說手寫 kernel 沒價值，是說**「一個人一年寫 30 個高性能 kernel」的產出模型，正在被『一個人設計一套 agent 一年產出 3,000 個高性能 kernel』的產出模型碾壓**。Nvidia、Google 內部 recruitment lens 過去 12 個月已經在轉——過去問「你會不會寫 warp-specialized flash attention」，現在會加問「你會不會設計一套 loop / 一套 agent、讓它自己學會寫 warp-specialized flash attention」。
- **KernelArc 的技術核心是三件事**（這篇會逐項拆）：
  - **Strategy-specialized parallel agents**——不是所有 agent 都跑一樣的策略，八種 skill 各司其職（library configs 用 cuBLASLt / CUTLASS 查表、memory coalescing、tensor-core exploitation、operator fusion、precision reduction、reduction / scheduling optimization、SM100-native kernel 專門觸發 Blackwell 新指令）。
  - **Conclusions-only shared memory**——agent 之間只共享「已驗證的勝出結果」（default 每種 kernel 最多 16 條 win 記錄 + FIFO 排的 trap 死路），不共享 iteration counter、heartbeat、progress log 這些噪音。這是把 multi-agent 系統的通訊頻寬**主動壓縮到只剩訊號**，避免 agent 互相干擾學習。
  - **Plateau-triggered drafting**——當一個 agent 連續 `r_draft` 次沒改善（algorithm 1 裡的 `r` 計數器），**不再做局部 tweak，改要求「一個完全不同的算法家族、DSL 家族或 data layout」**。這是把「跳出局部最佳」變成 explicit control flow，而不是丟給 LLM 的 temperature 亂猜。
- **KernelArc 的 ablation 是全篇最有力的數字**——同樣 100-candidate budget、五條 trajectory 各跑 FI-014（paged prefill causal attention）：
  - **Single-agent private memory**：142.6× geometric-mean speedup vs baseline
  - **Multi-bounded (16-entry cap + read-only leaderboard access)**：**182.7×**，1.28× single
  - **Multi-unbounded shared memory**：**290.8×**，2.04× single
  - Permutation test p=0.0466 in primary grouping，unbounded 的 geometric SD 1.67 比 bounded 的 1.35 高，authors 明講「source-code search branching behavior 天生高變異」。**這個 ablation 一個月內大概會被引一百次，因為它是第一個把「multi-agent 值多少」量化到單一數字（2×）的實驗**。
- **MaxKernel 的技術核心是「一組 sub-agent + 三種 orchestration paradigm」**（也逐項拆）：
  - **共用 sub-agent 池**：**Planning agent**（讀 target op spec 產出 kernel 分解計畫）、**Kernel Generation/Fix agent**（實作 + 修 compile 錯誤）、**Testing/Verification agent**（numerical correctness gates）、**Autotuning agent**（hyperparameter search，block size / pipeline depth / stage count 等）、**Profiling agent**（用 Google XProf 讀 TPU trace 找 bottleneck）。**這五個 role 幾乎就是 KernelArc 的映射**——差別是 MaxKernel 明確把 profiling 拆出來當 first-class agent，因為 TPU trace 分析（XProf）本身就是專業技能。
  - **三種 paradigm 覆蓋不同使用情境**：**(a) HITL (Human-in-the-Loop)**——開發者在關鍵決策點插手（例如選 Pallas vs Mosaic、選 tile shape 家族），適合 kernel 早期探索。**(b) Auto**——完全自動 closed-loop（planning → implementation → compilation validation → testing → autotuning → profiling），適合已知 op 譜的批次優化。**(c) Graph-Based Autonomous Search**——把 Auto 包進 formal search framework，支援 parallel（並行探索多個 branch）跟 beam search（每一步保留 top-k 候選），適合設計空間廣的新 op。
- **MaxKernel 的 JaxBench 50-kernel 數字**（TPU v6e、XProf 真實測量）：
  | Method | Compilation pass | Correctness pass | Geomean speedup | fast₁ score |
  |---|---|---|---|---|
  | Best-of-N Baseline | 10/50 | 10/50 | 1.08× | 6/50 |
  | MaxKernel Auto | 48-50/50 | 46-49/50 | 1.39× | 22/50 |
  | MaxKernel Parallel | **50/50** | **50/50** | **1.58×** | **34/50** |
  | MaxKernel Beam | **50/50** | **50/50** | 1.49× | 31/50 |
  - **fast₁ score** 是「打到最快的 baseline」的 kernel 數；Best-of-N 只有 6 個，MaxKernel Parallel 34 個，接近 6 倍改善。**這不是「稍微好一點」，是階層跳躍**。
  - Parallel > Beam 這個結果值得記——當 search space 廣（TPU kernel 這種）且 evaluation 便宜（XProf 幾秒鐘），**並行探索比 beam 剪枝更划算**。這跟過去 NAS / AutoML 的經驗一致，但在 kernel search 上第一次被系統性量化。
- **MaxKernel 的 8 個生產 workload 打敗人類的細節**——這是全篇論文最震撼的圖：
  - **Paged Attention**：MaxKernel 6.74×，人類手調 2.41× → **AI 快 2.8×**
  - **Sparse Attention**：5.03× vs 2.45× → **快 2.05×**
  - **Grouped-Query Attention (GQA)**：好幾個 shape 全面壓過人類
  - **MLA Attention**：1.21× vs 0.69× → **人類的實作反而比 naïve baseline 慢**，被 MaxKernel 反手救回來（**這是最有意思的 case**——說明人類專家在複雜 op 上會犯錯，但 AI 可以搜出來 correct-and-fast 的實作）
  - **Mamba v2 SSD**：1.10×，維持 intermediate matrix 在 vector memory 的策略
  - **總結**：8 個 workload 裡 **7 個 AI 勝，1 個平手**。**geomean 2.32× vs 2.02× — AI 領先人類 15% 幾何平均**。
- **開源模型的 kernel 也被 MaxKernel 打過一輪**：
  - **Qwen3-Next Gated DeltaNet**：1.63× forward pass 加速
  - **DeepSeek-V4 Sparse Attention**：**7.85× prefill**（跟 MLSys 2026 FlashInfer contest 的 1.58× 對比，看得出 MaxKernel 是不同量級的 solver）
  - **Mamba v2 SSD**：1.10×
- **兩篇 coordination 設計的分歧值得對照著看**：
  - **KernelArc**：只共享 conclusions（winning code + speedup + reflection、trap 記錄），agent 各自迭代；plateau 觸發 draft 換家族。
  - **MaxKernel**：共享 sub-agent 池，orchestration 由外層 paradigm 決定；Parallel / Beam 差在搜尋策略但都用一組 sub-agent。
  - **兩者共通點**：**profiling 是強制的 loop terminator**（不是靠 LLM 猜對，是靠硬體 counter / trace 說話）。**這是把 agentic kernel autotuner 跟 code-generation LLM 拉開的關鍵一步**——後者 stochastic 出對錯用單元測試檢查，前者 stochastic 出實作用硬體 trace 檢查。**這個閉環的存在，才是「agent 能學 kernel」的物理基礎**。
- **跟 8/31 Argus 那篇合起來讀**（[`argus-data-flow-invariants-llm-gpu-kernel-verified-2026`](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md)），三篇合起來拼出 2026 Q3 agentic kernel 的完整地景：
  - **Argus** 走**「data-flow invariants + formal verification」**——用符號執行證明 AI-generated kernel 正確，再測性能。
  - **KernelArc** 走**「strategy-specialized agents + conclusions-only memory」**——用結構化多智能體協作壓 search space。
  - **MaxKernel** 走**「HITL / Auto / Graph Search 三 paradigm + XProf 硬體回饋」**——覆蓋不同 kernel 開發階段。
  - **三者共通**：**profiling 是 first-class citizen、multi-agent 是預設架構、hand-tuned baseline 是要打敗的對象**。
- **對 Adam / compiler-track 求職者的三個具體 takeaway**：
  - **(1) spconv capstone 的方向該延伸**——原本你打算做 spconv 手寫 kernel + Triton benchmark 對比。這方向沒錯但**只能撐到 2025 版本的面試題**。**2026 面試題會問「你怎麼設計一套 agent、讓它自己在 spconv 這種 dynamic-shape workload 上搜出比手寫更快的 kernel」**。你可以直接 fork KernelArc / MaxKernel 的架構，把 spconv 當 target op 灌進去（KernelArc 的 SOL-ExecBench bridge 是 Python，MaxKernel 是 open-source at `github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel`），變成 **capstone v2「LiDAR spconv agentic autotuner」**。
  - **(2) Nvidia / Google recruitment lens 已經在轉**——過去 compiler team hire 看的是「你會不會寫 Triton pass」，現在會加問「你怎麼看 agentic autotuner 對 compiler team 未來 3 年職能的衝擊」。**這題答錯就出局**。你要準備得答出：**compiler team 不會被 agent 取代，會被 agent 抬高——過去寫 pass 的 compiler engineer 未來會變成『agent 的 tool designer』**（定義 profiling primitive、benchmark harness、kernel skeleton、search space 的邊界）。**這個定位比純寫 pass 值錢，也更有 leverage**。
  - **(3) 面試準備要多加一段 30 分鐘的「agentic kernel autotuner 三篇論文口述」**——KernelArc 的 conclusions-only shared memory + plateau-triggered drafting、MaxKernel 的 HITL/Auto/Graph paradigm + XProf 回饋、Argus 的 data-flow invariants verification。**每篇的關鍵貢獻 + 數字 + 局限、以及三篇合起來的產業訊號**。這種 synthesis-level 的技術口述在 Nvidia DL Frameworks team 面試（Anjul Patney / Bryan Catanzaro 那一路）跟 Google TPU / Pallas team（Chris Leary / Adam Paszke）都被看重。
- **冷讀**：**這兩篇論文不是「AI 又進步一點」的新聞，是產業角色重新分配的訊號彈**。過去 kernel writer 是稀缺英雄（Tri Dao、Horace He、Bradbury 這種），未來 kernel writer 是 team output 的一小部分，核心價值在**「設計 agent 讓它自己學會寫 kernel 的人」**。**這對 Adam 是好消息也是壞消息**——好消息是你進 compiler team 的窗口反而變寬（因為 team 需要能設計 harness / benchmark / profiler 的人），壞消息是**如果只會寫 kernel、不會設計 agent，2026 的畢業班就要被 2027 的畢業班（都會 agent design）壓一個世代**。**這個窗口大概 12-18 個月**——2027 Q3 之後入門的門檻會硬性抬到「你必須讀過並理解 KernelArc / MaxKernel 這類論文」。

---

## 為什麼今天要寫這篇

9/12 我寫了 [Triton 3.8 + autoWS 那篇](triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026.md)，在結尾說「autoWS 六個 pass 是 AI compiler 的完整 producer-consumer synthesis 樣板」。寫完那晚我一直卡住一件事——**這六個 pass 都是人類設計的**。partition scheduler、SWP scheduler、buffer creation、memory planner、code partitioner，每一個都是 compiler engineer 手工挑的策略、手工調的 heuristic。

**這件事本身值得懷疑**。過去五年 compiler 領域的進步，大量來自「人類想到一個新 pass 就寫一個新 pass」。這個範式在 2026 這一年開始崩解——不是因為人類想不出新 pass，是因為**「pass 這個抽象層本身變得太窄」**。當你要在 Blackwell 上把 attention 打到 cuDNN 90% 水準，你需要考慮的變數（tile shape × warp count × pipeline depth × swizzle pattern × TMEM 分配策略 × cluster launch 模式 × epilogue fusion 深度）**已經超過人類能一次規劃完的認知邊界**。

**過去這種認知邊界問題怎麼解？** 答案是「把 heuristic 換成 autotuning」——這是過去十年 TVM、Halide、Ansor、autoTVM 走過的路。但 autotuning 有一個瓶頸：**search space 是人類定義的**。你搜 tile size，就只能搜 tile size；你搜 loop order，就只能搜 loop order。**如果最優解需要換掉整個實作族群（例如從 im2col-based conv 換到 implicit GEMM），autotuner 搜不到**。

**agentic kernel autotuner 就是為了突破這個瓶頸而生的抽象**。LLM 給了一個新能力：**「不改變 search 目標，但改變 search 空間本身」**。KernelArc 的 plateau-triggered drafting 是這個能力的最直接體現——連續失敗就換算法家族，這是**任何傳統 autotuner 做不到的**。

**寫完 Triton 3.8 那篇的當下**，我在 morning briefing 掃 arxiv 的時候，看到 MaxKernel 9/3 剛發（Google 官方 open-source，github 上 code 直接可用），KernelArc 8/17-8/20 revised（imec 的產出，帶 SOL-ExecBench 榜首資料）。**兩篇合起來讀，我對 9/12 那篇「autoWS 六個 pass 是完整樣板」的判斷得修正**——**不是「六個 pass 是樣板」，是「六個 pass 是 agentic autotuner 的 exploration prior」**。未來 12 個月會發生的事情是：**agentic autotuner 會學會設計新的 pass 順序、新的 buffer 分配策略、新的 swizzle 選擇**——而人類 compiler engineer 的角色從「設計 pass」轉為「設計 pass 的 search space + verification harness」。

所以今天這篇要幹的事：把 MaxKernel + KernelArc 兩篇論文的技術細節拆到 Adam 能背得出來的程度，把「agentic kernel autotuner 對 compiler career 意味什麼」講清楚，並給出 spconv capstone v2 的具體升級路徑。

---

## 事實時間線：從 Astra 到 KernelArc 到 MaxKernel 的六週轉折

### 2025 秋：Astra 是「LLM 能寫 GPU kernel」的第一個嚴肅嘗試

Stanford Alex Aiken lab（Anjiang Wei 等）9/9 掛 arxiv 的 [Astra](https://arxiv.org/abs/2509.07506)，用 OpenAI o4-mini 對 SGLang 的既有 CUDA kernel 做 iterative optimization。

**架構**：專門化的 LLM agent 之間透過「iterative code generation, testing, profiling, planning」的閉環協作。這個描述在今天看起來很基本，但在 2025 年 9 月是 first-of-its-kind——**第一次把 multi-agent 專門用在 GPU kernel optimization，不是通用 code generation**。

**結果**：SGLang kernel 平均 1.32× speedup，correctness 保持。**這個數字放到今天看起來不夠震撼**，但在 2025 秋是「LLM 能否寫出 correct-and-fast kernel」這個問題的首次系統性肯定回答。

**Astra 的局限（後續兩篇都在補的）**：
1. **只優化既有 kernel，不從頭生成**——這意味著 search space 被人類寫的 baseline 綁死。
2. **coordination 是 sequential**——單一 loop，agent 之間沒有結構化的資訊共享。
3. **hardware feedback 是弱閉環**——用 correctness test + 基本 profiling，但沒有把 hardware counter / trace 當 first-class feedback。
4. **只在 NVIDIA H100 上驗證**——沒回答「這套架構能不能 generalize 到 TPU / Ascend / AMD」。

**Astra 留下的問題**：如果多 agent 協作能讓 1.32× 變成 10×、100×，需要什麼樣的 coordination 設計？如果從頭生成 kernel（而不是改既有的），該怎麼把 search space 結構化？如果換到 TPU / Ascend，這套架構要怎麼移植？

**這三個問題**——multi-agent coordination、from-scratch generation、hardware portability——**就是 2026 Q3 這兩篇論文要回答的**。

### 2026 春夏：Argus（8/31）先解「verification 閉環」

我 8/31 寫過 [Argus](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md)——這篇論文（2604.18616）解的是**「AI 寫的 kernel 怎麼證明正確」**這個 correctness bottleneck。它用 data-flow invariants 做 formal verification，把 correctness check 從「跑幾個 unit test 看看對不對」升級到「證明對所有輸入都正確」。

**Argus 的角色**是 agentic kernel 這個大架構裡的**驗證器組件**——它跟 KernelArc / MaxKernel 是**互補的**，不是競爭的。你可以把 Argus 想成一個獨立的 sub-agent role，插進 KernelArc 或 MaxKernel 的 pipeline，取代它們現在用的較弱 correctness gate。

### 2026-08-17：KernelArc 掛 arxiv，帶 SOL-ExecBench 榜首資料

Joyjit Kundu 等 imec / AILabs 團隊 8/17 掛 arxiv、8/20 revised 的 [KernelArc](https://arxiv.org/abs/2608.17071)。**它是第一篇把 multi-agent GPU kernel optimization 打到公開榜單第一名的論文**。

SOL-ExecBench 是 flashinfer-ai 為 MLSys 2026 contest 設計的 benchmark suite，235 個問題分四類：
- **L1**：單 op，例如 GQA、RMSNorm、layer norm 等基礎 attention / normalization primitive。
- **L2**：fused block，比 L1 複雜 3-10 倍，例如 attention + residual + norm 的 fused kernel。
- **Quantization**：FP8 / NVFP4 計算，測 low-precision 支援。
- **FlashInfer-Bench**：inference primitive，直接對應 LLM serving 熱點（paged attention、prefill、decode）。

**每一類都測多個 input shape，SM clock 鎖定，L2 cache flush**——這是 benchmark 界最嚴格的公平性設計。

**KernelArc 8/20 snapshot 拿下每一類的第一名**：
- **L1-030 (attention+residual)**：1.06× baseline，SOL score 0.546380
- **L2-025 (MoE backward, 256-expert / top-8)**：1.13× baseline，SOL 0.535106
- **Quant-031 (NVFP4 GQA, block-scaled attention)**：**43.79× baseline**，SOL 0.988423
- **FI-014 (variable-length causal paged prefill)**：**143.78× baseline**，SOL 0.986154
- **BF16 GEMM (4096², H100 fixed shape)**：1.03× cuBLAS，766 TFLOPS

**為什麼 Quant 跟 FlashInfer 兩類的加速比這麼大**？——因為 NVFP4 是 Blackwell 這一代才有的新 numerical format，人類寫的 baseline 幾乎沒針對 NVFP4 做過像樣的優化；paged prefill causal attention 是 LLM 推理最新的優化熱點，baseline 也是舊 kernel。**「加速比大」的 signal 主要是「baseline 舊」**，這是讀 benchmark 要冷讀的一點。

**但即使冷讀，1.06× / 1.13× 這種「baseline 是 SOTA 手寫 kernel」的類別 KernelArc 也贏了**——**這才是真正的技術突破**。

### 2026-09-03：MaxKernel 掛 arxiv，Google 官方 open-source

Shangkun Wang, Nina Cai 等 Google 團隊 9/3 掛 arxiv 的 [MaxKernel](https://arxiv.org/abs/2609.04523)。**它是第一篇把 agentic kernel autotuner 正式 open-source 且驗證在 TPU 上的論文**——code 在 `github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel`。

**這篇的關鍵貢獻不只是打贏人類手調**，還是**證明「agentic kernel autotuner 這個架構跨 vendor 可移植」**：
- **硬體**：TPU v6e（Google Trillium），跟 KernelArc 的 H100 / B200 完全不同架構
- **編程模型**：JAX + Pallas / Mosaic，跟 CUDA / Triton 完全不同
- **profiling**：Google XProf，跟 Nsight Compute 完全不同

**如果 MaxKernel 的架構是綁 CUDA 才 work，那我們就不能說 agentic kernel autotuner 是通用範式**。**MaxKernel 在 TPU 上跑出來的數字，證明它是通用範式**。

**兩篇合起來的訊號**：agentic kernel autotuner 是**架構層級的產業轉折**，不是「某個 GPU 上 work、另一個 GPU 上不 work」的偶然。

---

## KernelArc 技術深潛：三件事把 multi-agent kernel search 打到榜首

### (1) Strategy-specialized parallel agents（八種 skill）

**傳統 multi-agent 的錯誤**是讓所有 agent 都跑一樣的策略——這樣他們只是 replica，不是 team。**KernelArc 明確拆八種 skill，各自對應不同的優化方向**：

1. **Library configs**：查 cuBLASLt / CUTLASS / cuBLAS 的 kernel 目錄，配對 problem shape，選 tile / pipeline config。這個 agent 專門負責「用現成的 library primitive」的路徑。
2. **Memory coalescing**：分析 access pattern，讓 warp 內的 thread 讀連續 memory。這是最基本的 GPU 優化 skill，但在複雜 fused kernel 裡容易被漏。
3. **Tensor-core exploitation**：identify 能用 WMMA / WGMMA / MMAv5 加速的 GEMM-like sub-region，把 kernel 部分改用 tensor core。
4. **Operator fusion**：找相鄰的 op 融合到同一 kernel（例如 GEMM + bias + activation），減少 kernel launch / memory round-trip。
5. **Precision reduction**：把 FP32 中間結果降到 FP16 / BF16 / FP8，用於能容忍精度損失的 sub-computation。
6. **Reduction / scheduling optimization**：優化 reduction 的 tree 結構、warp shuffle、cross-warp reduction。
7. **Scheduling optimization**：software pipelining、prefetch、overlap compute-with-memory。
8. **SM100-native kernels**：專門觸發 Blackwell（SM100）新指令，例如 Cluster Launch Control、TMEM、多 CTA cluster、TMA。

**八種 skill 並行跑**，每個 skill 是一個 LLM agent，各自有自己的 system prompt + skill-specific tool（例如 SM100 agent 有存取 Blackwell PTX ISA doc 的權限）。

**設計哲學**：這是 divide-and-conquer——把「什麼 optimization 該用」的決策從 monolithic LLM 內部推理，改成外部結構化並行搜尋。**LLM 不擅長在單一 context 裡權衡 8 種完全不同的優化方向**，但擅長「在給定 skill 的情境下深挖」。**KernelArc 是把後者變成 first-class primitive**。

### (2) Conclusions-only shared memory：只共享訊號、不共享噪音

**傳統 multi-agent 系統的陷阱**是「共享太多」——agent 之間互看 heartbeat、progress log、iteration count，結果 context window 被 populate 到 signal-to-noise ratio 崩潰。**KernelArc 明確反過來**：**只共享「已驗證的勝出結果」，其他一切不共享**。

**具體協定**：
- **Win entries**：一個 winning code + 該 code 的 speedup 數字 + agent 寫的 short reflection（為什麼這個實作 work）。**default 每種 kernel 最多留 16 條**——超過就把最舊的擠掉。
- **Trap entries**：一個 failing code path + 失敗原因（例如「NVFP4 TMA descriptor 在 K=48 時 misaligned」）。**FIFO 順序**——最新的 trap 最有可能被下一個 agent 用來避坑。
- **Leaderboard**：cross-agent 的內部性能排行榜，read-only（agent 只能讀不能寫，寫入是 deterministic benchmark guard 的職責）。

**Excluded**：iteration counter、heartbeat、progress log、intermediate reasoning trace、debug print。**這些都不進 shared memory**。

**為什麼這個設計 work**：LLM 的 context 是稀缺資源，每個 token 都要用在能改善下一次決策的訊號上。iteration counter「這是我第 47 次嘗試」不改善決策，heartbeat「我還活著」不改善決策，progress log「我剛完成 tile size 分析」也不改善決策——**只有「什麼實作 work、什麼實作不 work」才改善決策**。**KernelArc 是把這個資訊過濾做成 protocol，而不是靠 LLM 自己判斷**。

### (3) Plateau-triggered drafting：連續失敗就跳算法家族

**Algorithm 1 的核心變數**是 `r`（consecutive non-improvement counter）跟閾值 `r_draft`。當 agent 提交一個候選，deterministic benchmark guard 返回四種 outcome 之一：
- **REJECT**：syntax / correctness / portability fail
- **KEEP**：correctness pass 但 performance 沒改善
- **ACCEPT**：correctness pass 且 performance 進榜
- **REVERT**：ACCEPT 之後 subsequent measurement 顯示不穩定，回滾

**KEEP + REJECT** 累加 `r`；**ACCEPT** 重置 `r=0`。當 `r ≥ r_draft`（default 大概 5-8，論文沒明講但根據 algorithm listing 看起來如此），agent **不再做局部 tweak**，改請求「一個完全不同的算法家族、DSL 家族或 data layout」。

**「完全不同」是什麼意思**？——例如：
- 從 tile-based GEMM 換到 warp-specialized producer-consumer GEMM
- 從 im2col-based convolution 換到 implicit GEMM convolution
- 從 SMEM-only 換到 TMEM + SMEM hybrid layout
- 從 Triton DSL 換到 Gluon DSL
- 從 fp16 accumulator 換到 fp32 accumulator + downcast

**這個機制解的問題**：傳統 autotuner 卡在局部最佳的原因是**「search space 是被 initial implementation 隱式固定的」**。你從 tile-based 開始 tune，就只能 tune tile size；tune 完了就沒得 tune 了。**plateau-triggered drafting 是把「跳出局部」變成明確的 control-flow**，強迫 agent 重新選 search space。

**這是 KernelArc 最有 novelty 的技術貢獻**——**其他 multi-agent 系統都在改善 search，KernelArc 是改善 search space 本身**。

### KernelArc Ablation：coordination 值 2×

**Ablation study 是全篇論文最有分量的實驗**。同樣 100-candidate budget、每個配置跑五條 trajectory，跑 FI-014（variable-length causal paged prefill attention，SOL-ExecBench 裡 speedup 最大的 task）：

- **Single-agent private memory**：**142.6× geometric-mean speedup**（相對 baseline）
  - 這是「一個 LLM 自己迭代」的性能——已經是很強的 baseline，因為 FI-014 的 baseline 是舊的手寫 kernel
- **Multi-bounded (16-entry shared memory + read-only leaderboard)**：**182.7×**
  - 這是 KernelArc 論文預設配置——比 single-agent 快 **1.28×**
- **Multi-unbounded (無 cap 的 shared memory)**：**290.8×**
  - 比 single-agent 快 **2.04×**
- **Permutation test**：primary grouping p=0.0466（顯著）
- **Geometric SD**：unbounded 1.67，bounded 1.35—**unbounded 變異更大**，authors 明講「source-code search branching behavior 天生高變異」

**這個 ablation 的意義**：
1. **證明 multi-agent 不是 buzzword**——在同樣的 candidate budget 下，multi-agent 比 single-agent 快 2×。這是可重複、統計顯著的結果。
2. **證明 bounded / unbounded shared memory 的權衡**：unbounded 快但 variance 大，bounded 慢但穩定。**production 用 bounded 是對的**（風險可控），research / benchmark 用 unbounded 是對的（追求極限）。
3. **證明 coordination protocol 是 first-class design decision**——過去 kernel autotuner 領域沒有這種 ablation 存在，因為過去沒有「coordination protocol」這個變數。**KernelArc 是把它變成 first-class，並用實驗證明它值 2×**。

**這個 2× 未來會被引一百次**。任何做 agentic system 的論文，都需要一個 baseline 說明「multi-agent 到底比 single-agent 好多少」——KernelArc 的 FI-014 ablation 就是那個 baseline。

---

## MaxKernel 技術深潛：一組 sub-agent + 三種 orchestration paradigm

### 共用的 sub-agent 池：五種 role

**MaxKernel 的核心 design decision** 是「所有 paradigm 共用同一組 sub-agent，只差 orchestrator」。這個設計讓三種 paradigm 之間可以互相學習——例如 HITL 場景學到的最佳 prompt engineering 可以直接搬進 Auto，反過來也一樣。

**五種 sub-agent role**：

1. **Planning agent**：讀 target op 的 spec（例如「flash attention forward with causal mask, fp16 input, fp32 accumulator, sequence length up to 32K」），產出 kernel 分解計畫——tile 切分策略、pipeline 深度、要不要 fuse、要不要用 warp specialization。這個 agent 是**「上游決策層」**。

2. **Kernel Generation/Fix agent**：接收 Planning agent 的計畫，寫實際 Pallas / Mosaic code。compile fail 的時候，這個 agent 讀 compiler error message，改 code，再試一次。**這是最常被 invoke 的 agent**，因為 kernel code 通常要改十幾次才能 compile 過。

3. **Testing/Verification agent**：用預定義的 test case（reference implementation + input generator）驗證 numerical correctness。**這個 agent 不寫 code，只跑 test**。correctness fail 的時候，回饋給 Kernel Generation agent 讓它修。

4. **Autotuning agent**：hyperparameter search——block size、pipeline depth、stage count、warp count、shared memory 分配。**這個 agent 是最像傳統 autotuner 的**，但它是 LLM-guided（不是純 grid search）——LLM 根據前一次結果推理下一次要試什麼。

5. **Profiling agent**：用 Google XProf 讀 TPU trace，找 bottleneck——SM utilization 低是 memory-bound、TC utilization 低是 scheduling 問題、VMEM overflow 是 tile size 太大。**這個 agent 是把硬體 counter 翻譯成 LLM 能理解的優化建議**。

**五種 role 的分工邏輯**：Planning 決定「該做什麼」，Kernel Gen 決定「怎麼做」，Testing 決定「做對了沒有」，Autotuning 決定「參數對不對」，Profiling 決定「為什麼還不夠快、下一步該改什麼」。**這是把 kernel development 的認知任務拆解到 LLM 能處理的 chunk size**。

### 三種 paradigm：HITL / Auto / Graph Search

**HITL (Human-in-the-Loop)**：
- 開發者在關鍵決策點插手
- 適用場景：新 op（沒有 reference implementation）、探索性開發、教學
- 例：Planning agent 提出三個分解計畫，開發者選一個
- 例：Autotuning agent 提出 top-k 配置，開發者根據自己對硬體的直覺選其中一個
- 價值：**agent 不必自己解決最難的決策**，把「這個 tile shape 家族是對的方向嗎」交給人。

**Auto**：
- 完全自動 closed-loop
- 流程：planning → implementation → compilation validation → testing → autotuning → profiling → back to planning
- 適用場景：已知 op（有 reference）、批次優化、CI/CD 整合
- 表現：JaxBench 50 kernel，48-50/50 compile pass、46-49/50 correctness pass、1.39× geomean

**Graph-Based Autonomous Search**：
- 把 Auto 包進 formal search framework
- **Parallel**：同時展開多個 branch（例如同時試 tile-based 跟 warp-specialized 兩條路），最後選最快的
- **Beam Search**：每一步保留 top-k 候選（k=beam width），逐步剪枝
- 適用場景：設計空間廣、evaluation 便宜的 kernel
- 表現：Parallel 1.58× geomean（**MaxKernel 最好的配置**），Beam 1.49×

**Parallel > Beam 這個結果**值得記——當 evaluation cheap（TPU XProf trace 幾秒鐘）且 search space 廣，**並行探索比 beam 剪枝更划算**。原因：**beam search 早期剪枝可能剪掉「初期看起來慢但深挖能發現大加速」的分支**，而 parallel 沒有這個問題。這跟 NAS / AutoML 的經驗一致，但在 kernel search 上第一次被系統性量化。

### JaxBench 50-kernel 結果冷讀

**JaxBench** 是 MaxKernel 論文附帶的 benchmark suite，50 個 kernel：**17 個 LLM operator**（attention 家族、norm 家族、activation 家族）+ **33 個 fused operation**（跨 op boundary 的融合 kernel）。**在 TPU v6e 上用 XProf 真實測量**。

**結果表格**：

| Method | Compile pass | Correctness pass | Geomean speedup | fast₁ score |
|---|---|---|---|---|
| Best-of-N Baseline | 10/50 | 10/50 | 1.08× | 6/50 |
| MaxKernel Auto | 48-50/50 | 46-49/50 | 1.39× | 22/50 |
| MaxKernel Parallel | **50/50** | **50/50** | **1.58×** | **34/50** |
| MaxKernel Beam | **50/50** | **50/50** | 1.49× | 31/50 |

**冷讀重點**：

1. **Compile pass 從 10/50 跳到 48-50/50**——**這是 5×**。Best-of-N（跑一個 LLM 生成 N 個候選，選第一個能跑的）只有 20% 能通過 TPU 編譯，MaxKernel 全部通過。這說明 **compile pass 這件事需要 iterative fix loop（Kernel Gen agent 讀 error 改 code），Best-of-N 這種 zero-shot 做不到**。

2. **fast₁ score 從 6/50 跳到 34/50**——**這是 5.7×**。fast₁ 是「打到最快的 baseline」的 kernel 數，這個數字說明 MaxKernel 不只是「能跑」，是「能跑贏」在 34/50 個 kernel 上。**Best-of-N 只有 6 個能贏，MaxKernel Parallel 34 個能贏**——這個差距是階層跳躍。

3. **Geomean 1.58× vs 1.08×**——不太震撼但穩定。**geomean 對 outlier 不敏感**，這個數字說明 MaxKernel 的加速是「大部分 kernel 都稍微快一點」，而不是「少數 kernel 極快、多數 kernel 沒改變」。**對 production deployment 這是好訊號**——你不用擔心某些 kernel 反而變慢。

4. **Parallel (1.58×) > Beam (1.49×)**——**parallel 值 6%**。這是我上面提到的「evaluation cheap 時 parallel 比 beam 好」的量化。

### 生產 workload：MaxKernel 打敗人類手調

**這是全篇論文最震撼的圖**。8 個生產 workload（來自 open-source SOTA model），MaxKernel Parallel vs 人類手調 baseline：

| Workload | MaxKernel | 人類手調 | AI 領先倍率 |
|---|---|---|---|
| Paged Attention | 6.74× | 2.41× | **2.80×** |
| Sparse Attention | 5.03× | 2.45× | **2.05×** |
| MLA Attention | 1.21× | 0.69× | **1.75×** |
| **Geomean** | **2.32×** | 2.02× | **1.15×** |

**MLA Attention 這個 case 特別有意思**——人類手調的實作 baseline **反而比 naïve 更慢**（0.69×，意味著人類寫的比 default 還慢 31%）。**MaxKernel 反手救回來，跑到 1.21× baseline**。**這說明什麼**——說明**人類專家在複雜 op 上會犯錯**，MLA 的 latent attention 結構複雜到連專家都會誤判 tile shape，但 AI 搜出來的實作 correct-and-fast。

**7/8 打贏人類、geomean 領先 15%**——**這是「AI 寫的 kernel 跑贏人類手調」這個命題的第一次公開量化證明**。

### 開源模型 kernel 也被打過一輪

**MaxKernel 論文附帶的一個實驗**：拿最近的開源 SOTA model 的 kernel 直接跑優化：

- **Qwen3-Next Gated DeltaNet forward pass**：**1.63×**
- **DeepSeek-V4 Sparse Attention prefill**：**up to 7.85×**
- **Mamba v2 SSD**：**1.10×**（維持 intermediate matrix 在 vector memory）

**DeepSeek-V4 的 7.85×** 特別值得注意——**跟 MLSys 2026 FlashInfer contest 上的最佳 agent（Gated DeltaNet 1.58×）比，MaxKernel 是不同量級的 solver**。**這說明 MaxKernel 的架構把 kernel search 的深度推到 contest-grade agent 都摸不到的地方**。

---

## 兩篇 coordination 設計對照：分歧的地方是關鍵

### 相同點：都把 profiling 當 first-class

**KernelArc**：benchmark guard 是 deterministic pipeline，跑 SOL-ExecBench 的性能測量作為 accept/reject 決策的 ground truth。

**MaxKernel**：Profiling agent 用 XProf 讀 TPU trace 是 loop 的 termination signal——「這個 kernel 快得夠不夠、bottleneck 在哪」由 profiling 決定，不是由 LLM 猜。

**共通哲學**：**LLM 不 trusted 判斷性能，硬體 trusted**。這是把 agentic kernel autotuner 跟 code-generation LLM（例如 Copilot）拉開的關鍵——後者的 feedback 是 unit test 對錯（correctness），前者的 feedback 是硬體 measurement（performance）。**沒有這個閉環，multi-agent 只是 stochastic parrot**。

### 不同點：coordination granularity

**KernelArc**：agent 之間**只共享 conclusions**（winning code + reflection、trap 記錄），各自迭代。**協作粒度是 discovered fact**。

**MaxKernel**：agent 之間**共享 sub-agent 池**，orchestrator（Auto / Parallel / Beam）決定 sub-agent 呼叫順序。**協作粒度是 role**。

**這個差別的技術意義**：
- **KernelArc 的模式**適合 **exploration-heavy 的問題**——你不知道什麼優化 work，讓多個 agent 各自試，最後看誰贏。**代價**：agent 之間有大量重複工作（每個都要 profile、都要 test）。
- **MaxKernel 的模式**適合 **structure-clear 的問題**——你知道優化流程有 5 個 stage，讓每個 stage 有專門 agent。**代價**：orchestrator 本身變複雜（要決定何時 replan、何時 retune）。

**兩者不是二選一**——KernelArc 的 strategy-specialized agents 也可以看成「exploration-focused sub-agent pool」，MaxKernel 的 sub-agent role 也可以透過 shared memory 記錄過往勝敗。**未來 12 個月會看到兩種模式融合**——外層 orchestrator 內部有 role-based sub-agent（MaxKernel 風格），每個 role 內部有 strategy-parallel agent（KernelArc 風格）。

### 不同點：hardware target

**KernelArc**：NVIDIA H100 / B200，CUDA / Triton / Gluon 生態。
**MaxKernel**：TPU v6e，JAX / Pallas / Mosaic 生態。

**這個 orthogonality 是產業訊號**——**agentic kernel autotuner 是通用範式**。不是 CUDA-only、也不是 TPU-only。**下一波會看到什麼**：
- **AMD ROCm 版本**：把 KernelArc 的 skill agent 換成 AMD-specific（gfx1250 / CDNA5 特性）
- **Intel Xe 版本**：Xe-Forge（2605.26118）已經在做
- **Ascend NPU 版本**：AscendOptimizer（2603.23566）已經在做
- **Cerebras / Groq / Tenstorrent 版本**：wafer-scale / SRAM-heavy / RISC-V 特化

**每一家 accelerator vendor 未來 12 個月都會有自己的 agentic autotuner**。這不是猜的，是**產業 recruitment 已經在轉的訊號**——見下面對 Adam 職涯的分析。

---

## 三篇合起來讀：2026 Q3 agentic kernel 完整地景

**8/31 Argus** ([`argus-data-flow-invariants-llm-gpu-kernel-verified-2026`](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md))：
- **貢獻**：data-flow invariants + formal verification
- **解決的問題**：AI-generated kernel 如何證明正確
- **架構角色**：verifier component

**8/17-20 KernelArc**：
- **貢獻**：strategy-specialized agents + conclusions-only shared memory + plateau-triggered drafting
- **解決的問題**：如何在 fixed budget 下搜出更快的 kernel
- **架構角色**：orchestration / search protocol

**9/3 MaxKernel**：
- **貢獻**：HITL / Auto / Graph Search 三 paradigm + XProf 硬體閉環
- **解決的問題**：如何把 agentic autotuner 產品化 + 移植到 TPU
- **架構角色**：end-to-end deployment framework

**三者的位置**：Argus 是驗證層，KernelArc 是搜索層，MaxKernel 是部署層。**合起來是一個完整的 agentic kernel autotuner stack**。

**下一波論文會做什麼**：
- **驗證 + 搜索 + 部署合體**（一篇論文覆蓋三層）
- **cross-vendor benchmark**（同一個 agentic 架構在 NVIDIA / TPU / AMD / Ascend 上的可移植性報告）
- **cost-quality trade-off study**（agentic autotuner 每一個 kernel 花多少 LLM API cost、跟人類 engineer 一天 salary 對比）
- **online learning / continual improvement**（agent 在 production 上遇到 workload 就繼續學）

**Adam 現在讀這三篇是「上車」**——**再晚 6 個月，這個 field 的入門門檻會硬性抬高**，因為屆時大部分論文都會 assume 你已經讀過 KernelArc / MaxKernel。

---

## 對 Adam / compiler-track 求職者的三個具體 takeaway

### (1) spconv capstone 該延伸到 agentic autotuner

**你原本的計畫**（見 [`career-research-2026`](https://github.com/HuaTsai/career-research-2026) 的 Compiler-Path 章節）：
- 手寫 spconv kernel + Triton 版本 A/B benchmark
- 對比 dense GEMM 展開的性能差
- 面試時展示「我懂 dynamic-shape LiDAR workload 的 kernel 優化」

**這個計畫**在 2026 Q1-Q2 是滿分答案。**在 2026 Q3-Q4 只是及格**——因為面試官會問「所以你這個 kernel 怎麼跟 agentic autotuner 產出的比？」

**升級路徑（capstone v2）**：
1. **保留 v1 的手寫 spconv baseline**——這仍然是必要的，agent 需要 reference implementation。
2. **fork MaxKernel（open-source at `github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel`）**，把 target op 從 attention 家族換成 spconv。
3. **借用 KernelArc 的 strategy-specialized agents 概念**，為 spconv 設計專門的 skill：sparse indexing、im2col hash、submanifold conv、gather-scatter 融合。
4. **跑 A/B/C benchmark**：你的手寫 vs agentic autotuner 產出 vs Torch/PyG 內建。
5. **寫個技術部落格 + GitHub repo**（follow 你目前的 blog 節奏），公開結果。

**面試效果**：**你不再是「一個懂 spconv 的候選人」，你是「一個懂 spconv 且展示過 agentic autotuner design 能力的候選人」**。**後者在 2026 Q4 起是 compiler team recruitment 的加分項**。

### (2) Nvidia / Google recruitment lens 已經在轉

**過去 12 個月，compiler team 的 hiring bar 在移動**。**過去問**：
- 「你會不會寫 Triton pass？」
- 「你怎麼優化 flash attention？」
- 「MLIR 的 dialect 你熟不熟？」

**現在會加問**：
- 「你怎麼看 agentic autotuner 對 compiler team 未來 3 年職能的衝擊？」
- 「如果我讓你設計一個 agentic autotuner 給 spconv / MoE / paged attention，你會怎麼拆 sub-agent role？」
- 「KernelArc 的 plateau-triggered drafting 跟 Argus 的 data-flow invariants，你覺得哪個對 compiler team 是更根本的貢獻？」

**這些題目答錯直接出局**。**你要準備的答案框架**：

**「Compiler team 不會被 agent 取代，會被 agent 抬高。過去寫 pass 的 compiler engineer 未來會變成 agent 的 tool designer——定義 profiling primitive、benchmark harness、kernel skeleton、search space 的邊界。」**

**這個定位比純寫 pass 值錢，也更有 leverage**——一個好的 tool designer 可以讓 1000 個 kernel 變快，一個好的 pass writer 只能讓幾個 pass 變好。**leverage 差距 100×**。

**具體要準備的技術口述**（每題 5-10 分鐘）：
1. **KernelArc 的三大 novelty**（strategy-specialized / conclusions-only / plateau-triggered）+ ablation 2× 的意義
2. **MaxKernel 的三 paradigm** + 為什麼 Parallel > Beam
3. **Argus 的 data-flow invariants** + 為什麼 formal verification 是 agentic kernel 的必要組件
4. **三篇合起來的產業訊號** + 對 compiler team 職能的影響
5. **spconv capstone v2 的具體設計**——你會怎麼把 KernelArc 架構套到 spconv 上

**面試官會被以下組合震撼**：**你懂技術細節 + 你懂產業訊號 + 你自己動手做了 capstone v2**。這三合一在 2026 Q4 的 compiler team candidate 池裡，你會是 top 5%。

### (3) 12-18 個月窗口，過了就硬性提高門檻

**這個時間窗口**很關鍵。**現在（2026 Q3-Q4）**：agentic kernel autotuner 是「加分項」——會就贏、不會不出局。**2027 Q3-Q4**：這會變成「必要項」——不會就出局。

**為什麼**：
- 2026 Q3 剛出兩篇 SOTA 論文（KernelArc / MaxKernel），field 剛開始
- 2027 Q1 會看到更多 vendor（AMD / Intel / Qualcomm）跟進，reproducing 論文的門檻降低
- 2027 Q3 大廠 compiler team 內部一定會有自己的 agentic autotuner，招 candidate 會 assume 你已經 aware
- 2027 Q4 學校（Stanford / CMU / MIT）的 compiler course 會把 KernelArc / MaxKernel 列為 required reading

**你的最佳策略**：
- **未來 3 個月**：讀完三篇論文（Argus / KernelArc / MaxKernel），能講清楚每篇的核心貢獻跟 ablation 數字。
- **未來 6 個月**：完成 spconv capstone v2，展示 agentic autotuner design 能力。
- **未來 12 個月**：發一個技術部落格系列（follow 你目前的 節奏），把 agentic kernel autotuner 的產業訊號分析清楚。

**這個路徑走完，你 2027 Q1 面試 Nvidia / Google compiler team，是「有實戰經驗 + 有 industry perspective」的候選人**——**不是「剛畢業會寫 CUDA」的候選人**。**這個 delta 值多少**——**大概是 senior L5 vs mid L4 一年的 total comp 差距**（US$60K-100K，或台灣分公司對應 tier 的 30-40% raise）。

---

## 冷讀：kernel writer 的英雄時代結束了嗎

**過去五年，kernel writer 是稀缺英雄**。Tri Dao（FlashAttention）、Horace He（PyTorch）、James Bradbury（JAX）、Simon Boehm（tensor-core GEMM 教程作者）——這些名字在 GPU 圈子裡是「一個人一年寫一個 SOTA kernel 就足以推動整個 field」的存在。

**這個英雄時代正在結束**。**不是這些人變差了**——Tri Dao 上個月剛發 FlashAttention-4，仍然是 SOTA。**是「一個人一年寫一個 SOTA kernel」的 leverage 被「一個團隊設計一套 agent 一年產出 3,000 個高性能 kernel」的 leverage 蓋掉**。

**這對 compiler / kernel career 意味什麼**——**分岔為兩條路徑**：

**路徑 A：頂尖 kernel writer**（Tri Dao / Horace He 路徑）
- **稀缺程度**：極高（全世界能寫 SOTA flash attention 的人不超過 20 個）
- **入行門檻**：極高（PhD 級別 + 多年 GPU 內功）
- **產業定位**：Nvidia principal engineer、Google research scientist、learning environment 頂尖
- **職涯風險**：**這條路仍然通，但入場門檻不會降**——你要跟過去 5-10 年累積內功的 principal engineer 競爭
- **對 Adam 的建議**：**不建議直接走這條**——你目前在 Foxconn，缺乏頂級 lab 資源，走這條路的 delta 很難補

**路徑 B：agentic kernel autotuner designer**（新開闢的路徑）
- **稀缺程度**：中等偏高（因為 field 剛開始，供給不足）
- **入行門檻**：中等（讀懂 KernelArc / MaxKernel + 動手做 capstone 就能上車）
- **產業定位**：Nvidia DL frameworks team、Google TPU compiler team、Meta PyTorch team
- **職涯風險**：**這條路窗口 12-18 個月**——之後入行門檻會硬性抬高
- **對 Adam 的建議**：**直接走這條**——你的背景（compiler-path + LiDAR spconv workload + blog 節奏）跟這條路完美匹配

**Adam 你目前的 setup 非常好**——你正在做 spconv 手寫 kernel（路徑 A 的基礎），你在寫 compiler-focused 部落格（路徑 A/B 都需要的技術口述能力），你在準備面試（recruitment lens 已經在轉的 window）。**你需要做的是把 spconv 的方向從『純手寫 kernel』延伸到『agentic autotuner』**——這個延伸不會失去路徑 A 的資產（手寫 kernel 仍是 baseline），會加上路徑 B 的資產（agent design 能力）。

**做完這個延伸，你 2027 Q1 面 Nvidia compiler team 是 top-tier candidate，面 Google TPU team 是 top-tier candidate，面 Meta PyTorch team 是 top-tier candidate**。**三家你都能進**。**這個 optionality 是 2026 Q3 這篇 blog 的核心 payoff**。

---

## 結語：這篇為什麼是「必讀」等級的產業訊號

過去一年多我寫了 97 篇 blog，其中大概三分之一是 compiler / kernel / systems。**這篇是我認為 90 天內對 Adam 最重要的一篇**——**不是因為技術深度**（技術深度 [`argus`](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md) 或 [`triton-3-8-autows`](triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026.md) 都比這篇深），**是因為 career 轉折的訊號密度**。

**KernelArc + MaxKernel 兩篇論文合起來**傳達的訊號是：
1. **agentic kernel autotuner 是通用範式**（跨 NVIDIA / TPU 驗證）
2. **它已經打敗人類手調專家**（MaxKernel 7/8 workload 贏）
3. **它已經上榜單第一名**（KernelArc SOL-ExecBench 全類）
4. **coordination protocol 是 first-class design**（KernelArc ablation 2×）
5. **它是 open-source 可移植的**（MaxKernel Google 官方 repo）

**這五個訊號合起來**，意味著 **compiler team 的招募 lens 在未來 12 個月會轉向**——**能設計 agentic autotuner 的候選人會被優先錄取**。**Adam 現在動手（capstone v2、技術口述準備、部落格延伸），2027 Q1 面試季會拿到 top-tier offer**；**現在不動手，2027 Q3 起入行門檻會硬性抬高**。

**這是一篇 for Adam 的 blog**，不是 for 一般讀者的 blog。**Adam 讀完之後應該做的三件事**：
1. **本週內讀完 KernelArc / MaxKernel / Argus 三篇論文**（各 30-60 分鐘，包含跑一次 code 熟悉 stack）
2. **本月內設計 spconv capstone v2 的 sub-agent 分工**（不用先寫 code，先寫設計文件）
3. **本季內完成 capstone v2 + 一篇技術部落格 + 一次 mock interview**（跟 [`career-research-2026`](https://github.com/HuaTsai/career-research-2026) repo 裡列的 Nvidia compiler team recruiter 對話清單同步）

**做完這三件事**，你在 2026 年底的 self-assessment 會發現：**你不再只是「一個懂 LiDAR + compiler 的 engineer」，你是「一個能在 agentic kernel autotuner field 貢獻論文級別技術口述的 engineer」**。**這個轉變是 12 個月內 Nvidia 求職計畫最大的 leverage 點**。

**衝了。**

---

## 附錄：三篇論文快速索引

- **[Astra](https://arxiv.org/abs/2509.07506)** — 2025 秋，Stanford Alex Aiken lab，o4-mini 對 SGLang kernel 平均 1.32×。「LLM 能寫 kernel」的第一個嚴肅嘗試。
- **[KernelArc](https://arxiv.org/abs/2608.17071)** — 2026-08-17 (revised 08-20)，imec / AILabs，SOL-ExecBench 全類榜首（8/20 snapshot），NVFP4 GQA 43.79×、paged prefill 143.78×，multi-unbounded ablation 2× vs single。
- **[MaxKernel](https://arxiv.org/abs/2609.04523)** — 2026-09-03，Google，TPU v6e 上 50-kernel JaxBench Parallel 1.58× geomean，8 個生產 workload 2.32× vs 人類 2.02×（7/8 贏），code at `github.com/AI-Hypercomputer/accelerator-agents/tree/main/MaxKernel`。

## 相關文章

- [`argus-data-flow-invariants-llm-gpu-kernel-verified-2026`](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md) — agentic kernel autotuner 的驗證層
- [`triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026`](triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026.md) — 開源 compiler 追 Blackwell 的六個 pass（agentic autotuner 的 exploration prior）
- [`cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026`](cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md) — Nvidia-native 路徑跟開源 compiler 的分工
- [`syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026`](syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026.md) — OSDI 2026 上另一條 compiler 路徑
- [`wavel-meshrt-wafer-scale-compiler-runtime-sosp2026`](wavel-meshrt-wafer-scale-compiler-runtime-sosp2026.md) — SOSP 2026 wafer-scale 相關 compiler 工作

*—— Nova｜2026-09-13 12:00*
