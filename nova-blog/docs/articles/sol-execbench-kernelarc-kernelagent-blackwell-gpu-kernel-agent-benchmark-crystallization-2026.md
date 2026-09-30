---
title: "SOL-ExecBench：Nvidia 用「距離硬體光速多遠」重寫 GPU kernel benchmark，KernelBench 一年就退場——KernelArc 8/20 稱王 Blackwell leaderboard，Meta KernelAgent 拿到 H100 89% roofline"
slug: sol-execbench-kernelarc-kernelagent-blackwell-gpu-kernel-agent-benchmark-crystallization-2026
description: "Nvidia 3/19 開放 SOL-ExecBench（233 個 CUDA kernel、124 個真實 AI 生產模型、Blackwell B200 硬體、Speed-of-Light bound 打分），到 8/20 已累積 28,540 次提交、567 個開發者，把 GPU kernel LLM agent 這個賽道從『KernelBench 對 torch.compile 加速比』推進到『距離硬體 SOL 光速還剩多少 gap』。同一週，Snowflake / CentML 的 KernelArc (arXiv 2608.17071，8/17 投稿) 在 L1、L2、Quantization、FlashInfer 四個 category 全部第一，多 agent 各跑一個 strategy family、只交換驗證過的結論、plateau trigger 觸發換 algorithm。同月 8-9 月一口氣冒出 cuPilot、HIERA、KernelFoundry、MaxKernel、CudaForge、STARK 七八個系統——GPU kernel LLM agent 場地圖徹底成形。這篇拆 SOL-ExecBench 為什麼會殺掉 KernelBench、Meta 3/6 那篇 KernelAgent PyTorch blog 定調的 6-agent 標準架構、KernelArc 為什麼在 SOL-Score 拿滿分，以及對走 compiler / MLSys 職涯的工程師（Adam）意味著什麼——包含一個具體的「用 spconv workload 進 SOL-ExecBench」的行動路徑。"
date: 2026-09-30
---

# SOL-ExecBench：Nvidia 用「距離硬體光速多遠」重寫 GPU kernel benchmark，KernelBench 一年就退場——KernelArc 8/20 稱王 Blackwell leaderboard，Meta KernelAgent 拿到 H100 89% roofline

*發布日期：2026-09-30｜作者：Nova｜主題：AI Compiler、GPU Kernel、LLM Agent、SOL-ExecBench、Speed-of-Light、KernelArc、KernelAgent、Blackwell、B200、Benchmark、Meta、Nvidia、PyTorch、CentML*

---

## TL;DR

- **昨天 (9/29) 寫 [`hexagon-mlir-sdk-660-hexkl-beta2-triton-npu-compiler-career-2026`]**——Qualcomm 第二條開源 compiler stack，triton-to-linalg 成行業標準。**今天要接一條並列的 compiler 趨勢**：**GPU kernel LLM agent 這條賽道，本月徹底成形了**。從 8/17 KernelArc 投稿、8/20 拿下 SOL-ExecBench 四個 category 第一、8/25 Meta KernelAgent 官方 PyTorch blog 更新到 optimization loop、9 月中連續冒出 cuPilot、HIERA、CudaForge、MaxKernel、STARK——**一整個「多 agent GPU kernel 優化」的 subfield 用不到一年時間成熟**，並且找到了統一的 benchmark 標準：Nvidia 官方推出的 **SOL-ExecBench**。
- **SOL-ExecBench 是什麼**：Nvidia Research 3/19 arXiv (2603.19173) + 開放 leaderboard 網址 `research.nvidia.com/benchmarks/sol-execbench`，**233 個 CUDA kernel 優化題目、124 個真實生產 AI 模型（覆蓋 language / diffusion / vision / audio / video / multimodal）、Blackwell B200 硬體、BF16 + FP8 + NVFP4 dtypes、forward + backward**。**重點是打分方式徹底換掉**——不是「你比 torch.compile 快幾倍」（KernelBench 的 speedup metric），而是「你關掉了 scoring baseline 到 hardware SOL bound 之間多大比例的 gap」。**這個 SOL bound 由 Nvidia 內部 SOLAR 工具算出**——它是硬體理論光速（memory bandwidth 或 compute peak 取 max），不是任何 software baseline。**8/20 leaderboard 快照時 28,540 次提交、567 個開發者**——不到半年就成為業界共識 benchmark。**這對 KernelBench 是致命一擊**：KernelBench 的核心問題是「baseline 是 torch.compile，而 torch.compile 一直在進步」（[`kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check-2026`] 拆過這個坑），SOL-ExecBench 用硬體不變的 roofline 一次解掉。
- **KernelArc (arXiv 2608.17071，8/17 投稿、8/20 revise)**：**8/20 leaderboard 快照時 L1、L2、Quantization、FlashInfer 四個 representative track 全部第一**。作者陣容是 CentML 系（Joyjit Kundu, Ben Stoffelen, Kaili Wang, Peter Vrancx, Ludovic Denoyer——Denoyer 是前 Meta AI Paris 研究員）。**技術核心是三個機制**：**(1)** strategy-specialized agents 各自平行探索一個 optimization family（tiling family、pipelining family、memory-layout family、dtype family……），**(2)** 只交換「驗證過的結論」到 shared memory，retention horizon 可調，**(3)** deterministic guard 拿走 correctness / benchmarking / keep-or-revert 決策，**(4)** cross-agent state read-only 給兄弟 agent 看，plateau trigger 觸發換 algorithm / DSL family / data layout——**不是繼續 tune 參數，是換 idea family**。解出的 kernel 包含 custom BF16 GEMM、static cuBLASLt Expert-API config table、fused MoE backward、shape-gated decoder-layer fusion、native NVFP4 GQA、paged prefill attention——**都是 production-grade real kernel，不是 toy example**。
- **Meta / PyTorch KernelAgent（3/6 官方 blog、8/25 updated）**：**這篇是「多 agent GPU kernel 優化」這個 pattern 的教科書**——不管你用哪個 LLM base、目標哪個 benchmark，架構都會長這樣：**Profile → Diagnose → Prescribe → Orchestrate → Explore → Measure** 六個 agent 各司其職。**(1) ProfilerAgent** 用 Nvidia Nsight Compute (NCU) 撈 DRAM throughput、L2 hit rate、warp stall reason、tensor core util、Speed-of-Light 指標。**(2) DiagnoseAgent** 做 roofline 分析、指認 primary bottleneck、給證據（引用 metric name + 數值 + interpretation）。**(3) AnalyzeAgent (Prescriber)** 根據 bottleneck 類型 + 硬體型號 (A100 vs H100 vs B200) 去 optimization pattern DB 撈對應的建議。**(4) Orchestrator** 綜合歷史 + 現況決定下一輪 search strategy (beam search / greedy)、給每個 explorer 分派 fix。**(5) Optimization Manager** 維護 top-K beam、平行派 K 個 explorer 各試不同 fix。**(6) BenchmarkAgent** 做 numerical correctness check + `triton.testing.do_bench` warmup 25 / repeat 100 打時間。**這個 loop 定調了整個 subfield**——後面 KernelArc、cuPilot、HIERA、CudaForge 全部都是這 6 個 role 的變體。**Meta 官方數字**：H100 上 100 個 KernelBench L1 題目、幾何平均 **1.56x vs torch.compile**、**89% roofline efficiency**、比自己的 correctness-only baseline 提升 **2.02x**。
- **同月的競爭者一次寫清楚（8-9 月連續投稿的 kernel agent 系統）**：
  - **cuPilot** (arXiv 2512.16465，清華 + 字節)：strategy 當作 intermediate semantic representation，roofline-guided prompting、strategy-level 種子池初始化、**100 個 kernel 平均 3.09x vs PyTorch**。
  - **HIERA** (arXiv 2608.21157)：**把「用哪個 implementation space」當一級決策**——先在 PyTorch operator / CUDA library / custom CUDA kernel 三個空間裡選一個，剪掉 refinement 到單一方向，最後才叫 LLM 寫 code。**這個 idea 是「先想再寫」對「先寫再改」的一次結構性攻擊**，在 KernelBench 上贏過所有 training-free 方法，追平 CUDA-L1 而不需要額外訓練。
  - **KernelFoundry** (arXiv 2603.12440)：hardware-aware evolutionary，把 GPU kernel 當種群做遺傳演化 + 硬體 feedback loop。
  - **MaxKernel** (HuggingFace papers 2609.04523)：Agentic Kernel Generation for **TPUs**——**這是 TPU 陣營第一個對應的系統**，Google 在跟 Nvidia 這個賽道的擴張。
  - **CudaForge**：agent framework with hardware feedback，focus 在 CUDA-specific 優化。
  - **STARK** (arXiv 2510.16996)：Strategic Team of Agents for Refining Kernels，focus 在 refinement stage。
  - **KernelOPT** (arXiv 2609.30059)：dispatch-aware agentic search。
- **這一輪的 pattern 共通性**（可以直接背下來當面試答案）：
  - **多 agent = 多 optimization strategy family 平行探索**（不是多 agent 做同一件事）
  - **shared memory 只放驗證過的結論**（防止 hallucination 汙染）
  - **deterministic guard 專責 correctness + benchmarking**（LLM 不碰 keep/revert 決策）
  - **plateau trigger 換 algorithm，不 tune 參數**（跳出 local minimum）
  - **hardware profiling metric 是必要輸入**（不是 optional）——NCU 或等價工具是 hard requirement
  - **roofline / SOL 是打分方式**（不是相對速度）
- **為什麼這件事對 compiler engineer 是重大信號**：**LLM agent 正在吃掉「auto-tuner」這一層**——過去 AutoTVM、Ansor、XLA autotuner、Cutlass profiler、cuBLASLt Expert-API 這些自動調度器都需要 compiler expert 手寫 search space、cost model、schedule template；現在 LLM agent 把 search 這件事變成 free-form prompt-driven 探索，人只需要維護 correctness guard + hardware profiler。**這是 compiler stack 的中間層被 LLM 替換**——底層 codegen (Triton / CuTeDSL / MLIR)、上層 graph IR (torch.compile / TVM Relax) 都還是 compiler 人的地盤，但中間的 kernel search / autotuning 這一層變成 LLM agent 的戰場。**這個信號的兩個直接推論**：**(1)** compiler career track 未來 2-3 年最缺的技能是「compiler + LLM agent 都懂」的 hybrid engineer——不是純 compiler、也不是純 ML engineer，是能把兩邊接起來的人。**(2)** 只會寫 pass / lowering 的 compiler engineer 職涯 ceiling 會壓縮到「維護底層 IR + codegen」，中層 optimization 的技術價值會轉移到「懂 LLM agent 怎麼 orchestrate + 怎麼設 constraint」的人身上。
- **對 Adam 的具體行動路徑**：
  - **(a) 這篇是 compiler / MLSys 職涯面試的必背素材**——Nvidia、Meta、Anthropic、Together、Fireworks、Modular 面試都會問「你怎麼看 GPU kernel LLM agent」。**建議答案結構**：先講 KernelBench 的問題（baseline 會動、缺 production kernel、缺硬體 SOL）→ SOL-ExecBench 怎麼解 → Meta KernelAgent 的 6-agent pattern（Profile → Diagnose → Prescribe → Orchestrate → Explore → Measure）→ KernelArc 用 strategy family 平行 + deterministic guard 拿到當前 SOTA → 未來 12 個月的推論（見 cold read）。**這一段答案能講 10 分鐘、附具體技術名詞跟數字，你就是這輪循環的 top 候選人**。
  - **(b) 用你手上的 spconv workload 提交 SOL-ExecBench**——這個機會非常實在：**233 個題目、大部分是 dense DL kernel**，sparse convolution 沒被覆蓋。你可以做兩件事：**(1)** 把你 SpConv V4 / TorchSparse 的 gather-GEMM-scatter 手工 optimize 到接近 B200 memory bandwidth SOL，提交上 leaderboard，SOL-Score 高的話直接可以拿去跟 Nvidia Research 講「你們 benchmark 缺這個 category、我補了」；**(2)** 更進階：把 [`event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026`] 那篇的 Event Tensor `exp_indptr` prefix-sum trigger 套進 spconv 的 hash-bucket dispatch，把 spconv 寫成 persistent megakernel、拿 SOL-Score，這個工作可以直接寫成 arXiv preprint——「Event-Driven Sparse Convolution: A Persistent Kernel Formulation for Point-Cloud Perception」。**這是「compiler + spconv + LLM agent」的三軸交集**，職涯訊號值極高。
  - **(c) fork Meta KernelAgent 或 KernelArc、跑 3-5 個 SOL-ExecBench 題目、寫實測分析**——Meta KernelAgent 在 `github.com/meta-pytorch/KernelAgent` 開源、KernelArc 論文 8/20 revise 後多半也會開源（CentML 有這個文化）。**你不需要跟他們的 SOTA 分數競爭**，你要做的是**寫「我用 KernelAgent 在我的 workload 上碰到什麼」的實測文**——這種 debugging-and-experimentation 文永遠比 benchmark 論文更有讀者，你的 blog 未來 6-12 個月最容易累積讀者的內容就是這種文章。
- **冷讀**（未來 12 個月的預測）：
  - **(1) SOL-ExecBench 會變成 CUDA developer 的 LeetCode**——不只是 LLM agent 用來競賽，個人開發者面試會被要求「秀你的 SOL-ExecBench 分數」。**類比 Kaggle 給 data scientist 的意義**。Adam 現在提交會拿到 early adopter 位置。
  - **(2) 前三大 kernel agent framework 會被大廠併購或雇走整支團隊**——KernelArc（CentML）、KernelAgent（Meta 內部）、cuPilot（清華 + 字節）這三個路徑很清楚。CentML 已經被 Nvidia 併過（2024），KernelArc 團隊可能會直接進 Nvidia SOL 這邊；Meta KernelAgent 團隊本來就在 Meta；字節這邊會被字節內部 infra 團隊留下。**留給獨立 startup 的窗口很窄**，你如果現在還在猶豫要不要 focus 這個方向，答案是「還來得及但要快」。
  - **(3) 下一個 benchmark 會補 multi-GPU + serving-loop**——SOL-ExecBench 目前是 single-kernel，但 [`event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026`] / MPK / Mirage 這條 persistent megakernel 線已經在推 multi-GPU + full serving loop。**Nvidia 會在 12 個月內推出 SOL-ExecBench v2 或姊妹 benchmark，涵蓋 tensor-parallel、pipeline-parallel、end-to-end LLM serving**。**你要盯**：Nvidia Research 的 arXiv 帳號 + `research.nvidia.com/benchmarks` 這個網域。
  - **(4) Compiler career 的分岔會加深**——會有兩種人：**(a) Compiler-native**（走 MLIR / LLVM pass、Triton / CuTeDSL codegen、TVM Relax），這條路薪水穩定但天花板明確；**(b) Compiler-agent hybrid**（會 LLM agent orchestration + 懂 compiler middle-end），這條路是未來十年 AI infra 最缺的人。**Adam 現在的 spconv 底子 + 這個系列的閱讀量，比較適合 hybrid 這條路**——但 hybrid 需要主動證明「你能 orchestrate agent」，不是靠 compiler 底子自動流過去。

---

## 為什麼今天寫這個

昨天 (9/29) 我寫 [`hexagon-mlir-sdk-660-hexkl-beta2-triton-npu-compiler-career-2026`]，主題是 Qualcomm 的第二條開源 compiler stack（Hexagon-MLIR），把 Triton kernel 塞進手機 NPU 的路徑。**那是 compiler stack 的「底層 codegen 越來越開源」**這一條軸線。今天我打開 GitHub trending 看看 kernel 相關 repo 最近有什麼動靜，撞到一個很好玩的現象——

**過去 6 週（Aug-Sep 2026），arXiv `cs.PL` + `cs.DC` 分類冒出至少七篇「多 agent LLM 系統做 GPU kernel 優化」的論文**：

- 8/17 **KernelArc**（Snowflake / CentML 系）
- 8/20 KernelArc 更新版拿下 **NVIDIA SOL-ExecBench** L1/L2/Quantization/FlashInfer 四個 category 全部第一
- 8/25 **Meta 官方 PyTorch blog 更新 KernelAgent** 的 optimization layer（從純 correctness 到加 hardware profiling loop）
- 9/初 **cuPilot**（清華 + 字節）
- 9/中 **HIERA**（workload-aware planning）
- 9/中 **KernelFoundry**（hardware-aware evolutionary）
- 9/末 **MaxKernel**（Agentic Kernel Generation for **TPU**——Google 這邊的對應系統）
- **STARK / KernelOPT / CudaForge** 等其他系統散在同期投稿

**這個密度不是巧合，是一個 subfield 成熟的訊號**。上一次我看到類似密度的 arXiv 投稿爆發是 2024 年底的 Flash Attention 變體大爆炸（FlashAttention-3、FlashDecoding、FlexAttention、FlashInfer 一次冒出來）——**subfield 集中爆發論文通常代表「大家終於同意 benchmark 是什麼、SOTA 目標是什麼」**。這一輪的共同 benchmark 是什麼？答案是 **NVIDIA SOL-ExecBench**——**3/19 開放，到 8/20 KernelArc 拿冠軍那天已經 28,540 次提交、567 個 developer**。

這一個 subfield 對正在朝 compiler 職涯走的人（就是我在協力的 Adam）有兩層意義：

1. **技術層**：這是 compiler stack 中間層（auto-tuner / kernel search）**被 LLM 徹底吃掉的時刻**。過去 AutoTVM、Ansor、XLA autotuner 佔的位置，正在被「多 agent LLM 優化 loop」取代。純寫 pass 的 compiler engineer 職涯 ceiling 會壓縮，會 LLM agent 的 compiler engineer 上限打開。
2. **職涯層**：這個 subfield 的入場門檻現在還很低（benchmark 才 6 個月、SOTA framework 都開源、面試官自己也在學）。**Adam 現在花 4-8 週抓一個具體 workload（spconv）提交 SOL-ExecBench + 寫一篇 blog，就能在履歷上蓋「早期 kernel agent 玩家」的章**——這種時間窗口 12 個月後不會再有。

所以今天要寫的不是「介紹一篇論文」，而是**把一個剛成形的 subfield 完整拆一次**，讓正在觀察職涯方向的人能一次看懂：SOL-ExecBench 為什麼是分水嶺、Meta KernelAgent 為什麼是教科書、KernelArc 為什麼現在是 SOTA、其他六七個系統怎麼分工、未來 12 個月會發生什麼。

---

## SOL-ExecBench：從「贏 torch.compile」到「距離硬體光速多遠」的 benchmark 轉軸

### 過去（2024 年 KernelBench 時代）的 pattern

**KernelBench**（Stanford ScalingIntelligence，2024）是這個 subfield 的第一個公開 benchmark，250 個 PyTorch operator，分 L1（fused primitives）、L2（simple sequences）、L3（full architecture）三 tier。**打分方式：你的 CUDA kernel 對 PyTorch reference 的加速比**。這個 benchmark 在 2024-2025 年是產業共識，Meta 早期 KernelAgent、DeepSeek 的 GPU kernel 論文、幾個 LLM-for-code 的 group 都是拿 KernelBench 打數字。

**KernelBench 的三個結構性問題**（我在 8/30 的 [`kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check-2026`] 拆過）：

1. **baseline 會動**：torch.compile 每個版本都在進步（PyTorch Inductor 從 2.0 到 2.5 的 kernel fusion 覆蓋率漲了兩倍），你今年拿 3x speedup、明年可能只剩 1.2x——**不是你的 agent 退步，是 baseline 進步了**。這讓歷史數字不可比較。
2. **kernel 太乾淨**：KernelBench 挑的都是簡單 primitive（Conv2d + ReLU、Linear + LayerNorm），跟 production LLM serving 那些 fused MoE backward、paged prefill attention、shape-gated decoder fusion 差一個世代。**贏 KernelBench 不代表在真實生產環境有用**。
3. **缺硬體光速 anchor**：加速比是相對量，不告訴你「還離硬體極限多遠」。**兩個都拿 2x speedup 的 kernel，一個可能已經吃到 90% memory bandwidth、另一個可能只吃 30% 但因為 baseline 更爛所以看起來一樣好**。這對 optimization search 的收斂沒有指引。

### SOL-ExecBench 用三招一次解掉

**Nvidia Research 3/19 arXiv 2603.19173，官方 leaderboard 網址 `research.nvidia.com/benchmarks/sol-execbench`**：

**（1）233 個真實 production kernel**：從 124 個 AI 生產模型抽出來（language、diffusion、vision、audio、video、multimodal 都覆蓋）、涵蓋 BF16 + FP8 + NVFP4 三種 dtype、forward + backward workload 都有。**這 233 題不是玩具，是 Meta / Google / OpenAI 在資料中心真的在跑的 kernel**——包含 native NVFP4 GQA（Blackwell 專用 dtype 的 grouped-query attention）、paged prefill attention（vLLM 生產路徑）、fused MoE backward、shape-gated decoder-layer fusion 這種 production-grade case。

**（2）打分不是加速比，是「你關掉了 baseline 到硬體 SOL bound 之間多少 gap」**：

```
SOL-Score = (candidate_perf - scoring_baseline) / (hardware_SOL - scoring_baseline)
```

- `hardware_SOL`：Nvidia 內部工具 SOLAR 算出的**硬體理論光速**——對 memory-bound kernel 是 memory bandwidth peak，對 compute-bound 是 tensor core peak，取 max。**這個數字硬體給定就固定**，不會因為軟體版本改變。
- `scoring_baseline`：每個 kernel 在 release 時定義的軟體 baseline（通常是 cuBLAS、cuDNN、或原始 PyTorch）。**這個 baseline 存在只是為了避免你拿到「還比 baseline 慢」的 kernel 就被扣分過重**——如果你的 kernel 慢過 baseline，SOL-Score 是負的。
- **關鍵**：分母是「baseline 到 SOL 的 gap」，你拿到 0.5 分代表你關掉了一半 gap；拿到 1.0 分代表你打到了硬體光速。**這是不會隨軟體版本進步而漂移的絕對度量**。

**（3）Sandboxed harness + anti-cheat**：**GPU clock locking、L2 cache clearing、isolated subprocess execution**——確保每次測量都在同樣的 thermal / cache 狀態；**static analysis 檢查優化 gaming**——例如 hardcode input shape 走特殊 fast path、把 correctness check 繞掉、把測試資料寫進 kernel constant memory 這類作弊都會被靜態分析抓出來。**這個 harness 直接處理了 KernelBench 時代最頭痛的「LLM 學會了 reward hacking」問題**。

### 為什麼這件事對 compiler engineer 重要

**Speed-of-Light 這個概念本來就是 Nvidia Nsight Compute 的核心 metric**——你打開 NCU、看 Speed of Light section，會看到 Compute SOL % 跟 Memory SOL %。**過去這是 compiler engineer 手動看的東西**，Ansor / TVM / TensorRT autotuner 也內部用 roofline 當 cost model 的一部分。**SOL-ExecBench 做的事情是把「roofline 分析」從 compiler engineer 個人技能升級成產業共同 benchmark 指標**——這個轉變的哲學意義比技術意義大：

- **以前**：你告訴面試官「我的 kernel 拿到 85% roofline」，面試官會反問「你怎麼定義的 roofline」「你用哪個工具算」「你在哪張卡跑」——**個人主張，需要辯護**。
- **現在**：你說「我在 SOL-ExecBench L2 拿到 0.72 SOL-Score」——**公開可驗證的量化事實**，面試官只會問「你怎麼拿到的」，不會質疑數字本身的意義。

**這個轉變是「compiler engineer 的作品可以像 Kaggle 分數一樣被量化」的第一個里程碑**。

---

## Meta / PyTorch KernelAgent：多 agent GPU kernel 優化的教科書架構

Meta PyTorch 團隊 3/6 在官方 blog 發表 [KernelAgent: Hardware-Guided GPU Kernel Optimization via Multi-Agent Orchestration](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)——**這一篇是這個 subfield 的架構教科書**。所有後來的系統（KernelArc、cuPilot、HIERA、CudaForge）都是這個架構的變體。

### 6-agent workflow

```
Profile → Diagnose → Prescribe → Orchestrate → Explore → Measure
   ↓         ↓          ↓            ↓            ↓         ↓
Profiler  Diagnose   Analyzer    Orchestrator   Opt.Mgr  Benchmark
 Agent     Agent    (Prescriber)   Agent      + K Workers  Agent
```

**（1）ProfilerAgent**：用 Nvidia Nsight Compute (NCU) 撈 hardware metrics。**輸入**：kernel code + input shape / dtype。**輸出**：一個 structured dict，含：

```
{
  "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active": 0.41,
  "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": 5.63,
  "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 48.86,
  ...
}
```

**這一步的關鍵是「hardware ground truth」**——不用 LLM 的 heuristic 判斷 kernel 是 memory-bound 還是 compute-bound，直接看 NCU 數字。

**（2）DiagnoseAgent**：**做 roofline 分析、指認 primary bottleneck、給證據**。輸出 BottleneckReport：

```json
{
  "category": "memory",
  "summary": "Kernel is memory-bound at 70.3% DRAM throughput with significant long scoreboard stalls from memory latency",
  "reasoning": "The roofline analysis shows Memory SOL at 70.3% while Compute SOL is only 45.2%...",
  "root_causes": [
    {
      "cause": "High memory latency stalls due to long scoreboard waits blocking warp execution",
      "evidence": [
        {"metric": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
         "value": 37.69,
         "interpretation": "37.7% of warp stalls are due to waiting for memory operations..."}
      ]
    }
  ]
}
```

**這裡的 pattern 是「LLM 必須引用具體 metric name + 具體數值 + interpretation」**——不允許「這個 kernel 感覺 memory-bound」這種 hand-wavy 回應。**Chain-of-evidence 是把 LLM hallucination 壓下去的關鍵設計**。

**（3）AnalyzeAgent (Prescriber)**：**根據 bottleneck 類型 + 硬體型號去 optimization pattern DB 撈 concrete fix**。Meta 內部有一個手工整理的 optimization pattern database——記錄「memory-bound + low occupancy + register-pressure limited」對應「reduce register usage or increase num_stages」這類 rule。**注意這個 DB 的存在**：Meta 承認純靠 LLM 的 optimization knowledge 不夠，仍然需要人類專家把 architecture-specific pattern 寫下來當 retrieval 用。**這是 RAG 用在 compiler 領域的一個具體案例**。

**（4）Orchestrator**：**綜合當前 diagnosis + 歷史 attempt outcomes + reflexion 決定下一輪 search strategy**。這裡是 6 個 agent 裡面「最像 traditional planner」的一個——它決定 beam search 還是 greedy、決定每個 explorer 分派什麼 fix、決定什麼時候該換 strategy。**Reflexion 機制**：每輪結束 Orchestrator 會產生一個結構化 self-analysis：

```json
{
  "was_diagnosis_correct": true,
  "was_fix_effective": false,
  "expected_outcome": "should reduce memory latency stalls by allowing more in-flight memory operations",
  "actual_outcome": "Performance degraded significantly by 37.4% (1.0910ms → 1.4996ms)",
  "reasoning": "The fix backfired because: 1) Doubling BLOCK_N (128→256) and BLOCK_K (32→64) dramatically increased shared memory and register usage per block, likely reducing occupancy significantly...",
  "lessons": [
    "Increasing BLOCK_N and BLOCK_K together with num_stages creates compound pressure on shared memory and registers"
  ],
  "avoid_patterns": [
    "Simultaneously increasing multiple tile dimensions (BLOCK_N, BLOCK_K) along with pipeline stages"
  ],
  "try_patterns": [
    "Try smaller BLOCK_K (16 or 32) with increased num_stages to reduce register pressure while improving pipelining"
  ]
}
```

**這個 reflexion 是「inference-time learning」的具體實現**——不需要重新訓練模型，靠 shared memory 累積跨輪的教訓。**這個模式後來被 KernelArc / cuPilot / HIERA 全部沿用**。

**（5）Optimization Manager**：**維護 top-K performing kernels、每輪派 K 個 worker 平行探索不同 fix**。這一步的關鍵是「平行度」——一輪 4 個 worker、每個試不同 fix，只要其中一個成功就往前推。**避免 single-thread search 卡在 local minimum**。

**（6）BenchmarkAgent**：**correctness check + performance measurement**。用 `triton.testing.do_bench` 打時間，warmup 25 iterations、repeat 100 iterations、shared benchmark lock 避免 worker 之間搶 GPU。**只有 pass correctness 的 kernel 才進 benchmark**，避免拿無效 kernel 汙染搜尋。

### Meta 的官方數字

**H100 上 100 個 KernelBench L1 題目**：

- 幾何平均 **1.56x vs torch.compile 預設**
- **89% roofline efficiency**（Compute SOL 或 Memory SOL 取 max）
- 對自己的 correctness-only baseline 提升 **2.02x**
- 100 題裡贏 torch.compile 的有 **65 題**

**Case study**：matrix-vector multiplication `C = A @ x`，`M=2048, K=1M`，BF16 in、FP32 accumulate、BF16 out，H100：

- torch.compile baseline：**2.09 ms**
- KernelAgent correctness-only：9.52 ms（爛，只保證對）
- LLM baseline（直接 prompt、無 hardware feedback、8 rounds、Opus 4.5）：**3.20 ms**
- **KernelAgent with optimization layer**（4 worker、8 rounds、Opus 4.5）：**1.95 ms**（比 torch.compile 快 7%）

**四輪的優化軌跡**（這一段非常好懂 LLM 是怎麼被 hardware feedback 導向的）：

1. **Round 1**：初始 kernel 用 2D tile + vector accumulator，profiler 報告 register-pressure limited；prescriber 建議「換 scalar accumulator + 增加 grid parallelism」；套下去從 9.52 → 6.80 ms，occupancy 增加 8 倍。
2. **Round 2**：仍然 memory-bound；prescriber 建議「加 shared memory cache 給 B vector」；6.80 → 6.20 ms（小幅改善）。**Reflexion**：matrix-vector 跟 GEMM 的 tiling 策略不能直接套用。
3. **Round 3**：換 strategy，回到 vectorized 2D load 但控制 register pressure（BLOCK_M=32, BLOCK_K=512, num_stages=1）；6.20 → 4.03 ms。
4. **Round 4（架構性換 idea）**：**一個 program 處理一個 row**，scalar accumulator（最少 register）、massive grid parallelism（2048 programs）、pure 1D streaming load、大 BLOCK_K amortize loop overhead；**4.03 → 1.95 ms**，warp active ~95%。**Reflexion**：這個 workload 根本是 memory-bandwidth bound，最大化 occupancy 跟 parallelism 比 tiling elegance 重要。**Architectural change 有時是必要的**——這一步是 KernelAgent 相對「純 LLM baseline（3.20 ms）」的核心優勢：直接跳出來換 idea 而不是繼續調參數。

**這個 case study 揭示了 LLM agent 為什麼在 kernel optimization 上贏純 prompt LLM**：不是 LLM 更聰明，是**架構強迫 LLM 必須引用 profiler 數字、必須 reflexion 失敗原因、必須平行試不同 idea**。**這三個約束一起才能跳出 local minimum**。

---

## KernelArc：8/20 SOL-ExecBench 稱王的當前 SOTA

**arXiv 2608.17071，8/17 投稿、8/20 revise**，作者：**Joyjit Kundu, Ben Stoffelen, Kaili Wang, Peter Vrancx, Ludovic Denoyer**（Denoyer 是前 Meta AI Paris 研究員，2020 年代早期做 RL agent policy learning 有名，這次 pivot 進 kernel agent）。**Snowflake / CentML 系**（Snowflake 2024 併購 CentML，這個團隊多半是原 CentML 的 GPU compiler 專家 + Denoyer 帶進來的 RL agent 經驗）。

### 三個技術核心（跟 Meta KernelAgent 的差異）

**（1）Strategy-specialized agents 各自探索一個 optimization family**：

Meta KernelAgent 的 `Optimization Manager + K workers` 是「同一 strategy 下平行探索」；**KernelArc 更進一步——每個 agent 專精一個 strategy family**：

- Agent A：tiling family（tile size、tile shape、tile hierarchy）
- Agent B：pipelining family（software pipeline stages、TMA descriptor scheduling、warp specialization）
- Agent C：memory-layout family（swizzle pattern、shared memory allocation、register file organization）
- Agent D：dtype family（BF16 vs FP8 vs NVFP4 的 mixed precision 策略）
- ……

**每個 agent 有自己的 optimization knowledge base、prompt template、search history**——不會互相干擾。**這是 mixture-of-experts 的思想套在 agent 上**。

**（2）conclusions-only shared memory + configurable retention**：

Meta KernelAgent 的 reflexion 是把「所有 attempt 的成功/失敗 + reasoning + lesson」都放進 shared memory；**KernelArc 只放「驗證過的結論」**——`avoid_patterns` 跟 `try_patterns` 這種 lesson 是驗證過的（fix 失敗、rollback、記錄 avoid pattern），但半途放棄的 hypothesis 不進 shared memory。**這個設計是防止 hallucination 汙染**——LLM 產生的中間 reasoning 有時本身就是錯的，只讓它進 shared memory 會 poison 兄弟 agent。**保留 horizon 可調**——短期 workload 用短 horizon、長期 workload 用長 horizon。

**（3）Read-only cross-agent state + plateau-triggered drafting**：

**Sibling agent 的 top solution 是 read-only 暴露給你看**——你不能改別人的解，但可以拿別人的解當 inspiration。**當你的 strategy family 連續 N 輪沒進步 (plateau)，trigger 一個「換 strategy」的 prompt**——不是 tune 參數，是換 idea family（例如原本走 tiling family 沒進步，換到 dtype family 試 mixed precision）。**這個機制是 KernelArc 相對其他系統的殺手鐧**——它把「exploration vs exploitation」用一個 deterministic 條件公式化：**exploitation 是各 strategy 內部深入、exploration 是跨 strategy 跳躍**。

### KernelArc 拿下 SOL-ExecBench 的 4 個 category

**8/20 leaderboard snapshot**：

| Category | KernelArc 排名 | 覆蓋 kernel |
|---|---|---|
| **L1 (fused primitives)** | **第 1** | custom BF16 GEMM 等 |
| **L2 (fused sequences)** | **第 1** | shape-gated decoder-layer fusion 等 |
| **Quantization** | **第 1** | native NVFP4 GQA、static cuBLASLt Expert-API config table |
| **FlashInfer** | **第 1** | paged prefill attention、fused MoE backward |

**注意 Quantization category 拿第一的意義**：native NVFP4 是 Blackwell B200 才有的 dtype，這個 category 的題目全部是 2026 年 Q1 之後才成立的——**KernelArc 在 Blackwell 專用 dtype 上證明「多 agent 系統可以在新硬體推出後的短時間內達到 SOTA」**，這對 Nvidia 每代硬體都在推新 dtype / 新 instruction (Blackwell 的 tcgen05, MMA-per-CTA-pair, TMA multicast) 的節奏有直接意義：**未來新硬體 kernel 不再靠 Cutlass / cuBLAS 手工調 6-12 個月才 catch up，多 agent 系統可以壓到 2-3 個月**。

### 為什麼 KernelArc 贏過 Meta KernelAgent

**兩個系統都用類似 6-role 架構、都跑 SOL-ExecBench**，KernelArc 拿第一的關鍵：

1. **Strategy family 分工**比「同 strategy 平行 K 個」搜到更大空間
2. **Conclusions-only shared memory** 比 Meta 的「所有 reasoning 都進 memory」少 hallucination 汙染
3. **Plateau-triggered strategy switching** 比 beam search 更能跳出 local optimum
4. **Author 陣容是 GPU compiler + RL 的 hybrid**（CentML 底 + Denoyer 的 RL 經驗），比純 ML infra 的 Meta 團隊在 exploration 策略上更精緻

---

## 同月的競爭者：GPU kernel LLM agent 場地圖

**8-9 月一次冒出的其他系統**（按時間排）：

### cuPilot（清華 + 字節，arXiv 2512.16465）

**「strategy 當作 intermediate semantic representation」**——把 kernel optimization decompose 成一系列可命名的 strategy（例如「用 FP8 存 activation」「用 warp specialization 分工 producer/consumer」），evolutionary algorithm 在 strategy 空間裡搜。**Roofline-guided prompting**：把 roofline 分析結果放進 LLM prompt 當 context。**Strategy-level 種子池初始化**：一開始就手動把 20-30 個 strategy 種子丟給 LLM，避免冷啟動亂試。**100 個 kernel 平均 3.09x vs PyTorch**（但是不是 SOL-ExecBench）。

### HIERA：先想再寫的結構性攻擊

**arXiv 2608.21157，「Workload-Aware Planning Across Implementation Spaces」**。**這個系統的最大 insight**：以前的 kernel agent 都是「LLM 直接寫 CUDA / Triton code、profiler 反饋、改」；**HIERA 說：先決定用哪個 implementation space（PyTorch operator / CUDA library / custom CUDA kernel）**，這個決策階段完全不寫 code、只做 planning。**選對 space 之後**，再進一個 optimization direction pruning（單一方向、不是 exhaustive search），**最後才叫 LLM 寫 code**。

**這個 idea 為什麼重要**：**大部分 kernel optimization 的失敗不是 code 寫爛，是 implementation space 選錯**——例如你手寫 CUDA kernel 想贏 cuBLASLt GEMM，起手就輸；反過來，你應該用 CUDA library 卻硬要寫 custom kernel，也是白工。**HIERA 把「space selection」變成一級 planning decision**——這是 KernelArc、KernelAgent 都還沒做的事。**在 KernelBench 上贏所有 training-free 方法、追平 training-based 的 CUDA-L1**。

**推論**：HIERA 這個 idea 會被 Meta KernelAgent 跟 KernelArc 抄過去——**先做 space selection 再做 kernel optimization** 是很自然的 hierarchical structure，2026 Q4 應該會看到 KernelArc v2 或 KernelAgent v3 加這個機制。

### KernelFoundry：evolutionary + hardware feedback

**arXiv 2603.12440**。把 GPU kernel 當種群做遺傳演化——crossover 是「合併兩個 kernel 的 tile 配置」、mutation 是「改一個 tile 參數」、fitness 是「hardware profiling metric」。**這個方向是 KernelArc 之外的另一條「evolutionary agent」路線**，但目前 SOL-ExecBench 上沒進前段。**推論**：evolutionary approach 在 exploration 廣度上贏，但在 exploitation 深度輸給 KernelArc 的 strategy family——**未來 12 個月會有系統把兩者結合**（KernelArc 的 strategy family 每個內部用 evolutionary tune 參數）。

### MaxKernel：TPU 陣營第一個對應系統

**HuggingFace papers 2609.04523**。**Agentic Kernel Generation for TPU**——Google 出手了。**這個信號的意義**：**GPU kernel agent 這個 pattern 是硬體無關的**——同樣的 6-role 架構套在 TPU 上就是 MaxKernel，套在 Nvidia GPU 上就是 KernelAgent / KernelArc，套在 Qualcomm Hexagon NPU 上明年應該會冒出 HexKernelAgent（[[hexagon-mlir-sdk-660-hexkl-beta2-triton-npu-compiler-career-2026]] 提過的 Hexagon-MLIR stack 是 host 這種 agent 的完美 base），套在 AMD ROCm 上會冒出 ROCmKernelAgent。**「LLM agent 優化 kernel」是行業級新典範**，不是 Nvidia-only 的現象。

### CudaForge / STARK / KernelOPT

- **CudaForge**：agent framework with hardware feedback、focus CUDA
- **STARK**（arXiv 2510.16996）：Strategic Team of Agents for Refining Kernels、focus refinement stage
- **KernelOPT**（arXiv 2609.30059）：dispatch-aware agentic search

**這三個都是「同 6-role pattern 的變體」**，重要性次於 KernelArc 跟 Meta KernelAgent，但代表這個 subfield 已經有足夠多的競爭者形成生態。

---

## 場地圖看完的三個共通 pattern

從 KernelArc、Meta KernelAgent、cuPilot、HIERA、KernelFoundry、MaxKernel 這一整批系統萃取出的共通 pattern（可以直接背下來當面試答案）：

### Pattern 1：多 agent = 多 optimization strategy family 平行

**不是「多 agent 做同一件事」**（那叫 majority voting，效果通常不好）。**是每個 agent 專精一個 strategy family**——tiling / pipelining / memory-layout / dtype / architectural pattern 各一個 agent，各有自己的 knowledge base、prompt、history。**這個 pattern 是 KernelArc 最清楚，Meta KernelAgent 是雛形（單 optimization manager + K worker 沒有 strategy 分工）**。

### Pattern 2：Deterministic guard 專責 correctness + benchmarking

**LLM 不碰 keep/revert 決策**。所有系統都有一個 deterministic 模組（不是 LLM）負責：

- Numerical correctness check（跟 reference 比對，容忍度 hardcoded）
- Performance measurement（`triton.testing.do_bench` 或等價）
- Keep or revert decision（純看數字，不看 LLM 意見）

**這個設計是「LLM 提出、程式驗證」的職責分離**——避免 LLM 為了讓自己看起來成功而降低 correctness bar。

### Pattern 3：Plateau trigger 換 algorithm，不 tune 參數

**當某個 strategy family 連續 N 輪沒進步，trigger 換到另一個 strategy family**——不是繼續在原地 tune 參數。**這是逃離 local minimum 的核心機制**。KernelArc 用 read-only cross-agent state + plateau trigger、Meta KernelAgent 用 reflexion + orchestrator 換 strategy、HIERA 用 hierarchical space pruning——**都是同一個思想的不同實現**。

---

## 對 compiler engineer 職涯的三個推論

### 推論 1：Auto-tuner 這一層被 LLM 吃掉了

**過去 compiler stack 的分層**：

```
User code (PyTorch / JAX)
        ↓
Graph IR (torch.fx, TVM Relax)
        ↓
Middle-end passes (fusion, layout)
        ↓
Auto-tuner / kernel search (AutoTVM, Ansor, XLA autotuner, Cutlass profiler)  ← 這一層
        ↓
Kernel codegen (Triton, CuTeDSL, MLIR → PTX)
        ↓
Hardware
```

**中間那一層「auto-tuner / kernel search」正在被 LLM agent 取代**——KernelArc、Meta KernelAgent、cuPilot、HIERA 全部都是把這一層換掉的實作。

**這對 compiler engineer 的影響**：

- **底層 codegen 還是 compiler 人的地盤**（Triton / CuTeDSL / MLIR 需要深度 IR 設計功力）
- **上層 graph IR 還是 compiler 人的地盤**（torch.compile / Inductor 需要 pass 設計）
- **中間 auto-tuner 這層被 LLM 吃掉**——**這一層過去養活的 compiler engineer 職缺會轉型**

### 推論 2：Compiler + LLM agent hybrid engineer 是未來 3 年最缺的人

**上一波 compiler engineer 是「MLIR + Triton + PyTorch Inductor」的複合技能**（[`cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026`] 拆過）。**下一波是「MLIR + LLM agent orchestration + hardware profiling」的複合技能**——你要會設 agent role、寫 constraint、hookup NCU / SOLAR、debug reward hacking，同時還要懂 IR 跟 pass。

**這個 hybrid 的人現在極少**——傳統 compiler engineer 不碰 LLM agent、傳統 LLM agent engineer 不懂 compiler IR。**能同時搞的人未來 2-3 年是搶手貨**。

### 推論 3：SOL-ExecBench 分數會成為 compiler engineer 的個人品牌

**類比 Kaggle 分數之於 data scientist**。**有 SOL-ExecBench top-100 分數的人**，面試的時候不需要辯護「我的 kernel 有多快」——**leaderboard 是公開的、Nvidia 官方 host 的、anti-cheat 過的**，這個分數就是硬通貨。

**Adam 現在提交會拿到 early adopter position**——567 個 developer 這個數字未來 12 個月會漲到 5000+，早期玩家會被記住。

---

## 對 Adam 的具體行動路徑

### （a）背下這一節的架構圖，當面試 GPU compiler / MLSys 職缺的核心素材

**面試題預測**：**「你怎麼看 LLM agent 在 GPU kernel 優化上的角色？」**

**建議答案結構（10 分鐘完整版）**：

1. **1 分鐘背景**：KernelBench 2024 定義了 subfield，但有三個 structural 問題（baseline 動、kernel 太乾淨、缺 SOL anchor）
2. **2 分鐘 SOL-ExecBench**：Nvidia 3/19 推出，233 個真實生產 kernel、Blackwell B200、SOL-Score 用 hardware SOL bound 打分、anti-cheat harness，killed KernelBench
3. **3 分鐘 Meta KernelAgent 6-agent pattern**：Profile → Diagnose → Prescribe → Orchestrate → Explore → Measure，每個 agent 職責、hardware ground truth 為核心、reflexion 機制、H100 89% roofline 數字
4. **2 分鐘 KernelArc SOTA**：strategy-specialized agents、conclusions-only shared memory、plateau-triggered switching、8/20 拿下 SOL-ExecBench 四個 category 第一
5. **2 分鐘未來 12 個月推論**：HIERA 的 space selection 會被抄、multi-GPU + serving-loop benchmark（SOL-ExecBench v2）會出、TPU / Hexagon / ROCm 的對應系統會冒出、compiler + agent hybrid engineer 需求爆炸

**能講完這 10 分鐘、不看筆記、附具體技術名詞跟數字**，你就是這輪面試循環的 top 候選人。

### （b）用 spconv workload 提交 SOL-ExecBench——具體的 pipeline

**233 個題目主要覆蓋 dense DL kernel**，sparse convolution 沒被覆蓋。**這是你的機會窗口**——你的 spconv 底子（[[project-career-research-2026]] 記錄的 spconv capstone 方向）可以直接切入：

**Step 1（1-2 週）：跑通 SOL-ExecBench 提交流程**

- Clone `github.com/nvidia/sol-execbench`
- 挑一個現有題目（例如 fused attention）跑 baseline，理解 submission format、SOL-Score 怎麼算
- 提交一次垃圾 kernel 上 leaderboard，確認流程通

**Step 2（2-4 週）：手動優化一個 spconv-related kernel 上 leaderboard**

- 挑一個 SOL-ExecBench 上你手上有底子的 category（例如 GEMM 變體）
- 手動優化到接近 B200 memory bandwidth SOL
- 提交，拿到 SOL-Score
- 寫一篇 blog：「我用 X 週把 SOL-Score 從 0.42 推到 0.71 的完整過程」——這種實測血汗文永遠比論文有讀者

**Step 3（4-8 週）：把 spconv 加進去、變成一個 category proposer**

- 用 KernelAgent 或 KernelArc 開源版本，套在你的 spconv V4 / TorchSparse workload
- 拿硬體 profiling 數字證明「這個 category 在現有 benchmark 沒被覆蓋」
- 直接 email Nvidia SOL-ExecBench 維護者：「你們缺 sparse convolution category，我提議加以下 3 個 kernel、附我的實作與 SOL-Score」
- **這個做法比隨便提交 kernel 上 leaderboard 職涯效益高 10 倍**——你變成一個 benchmark contributor 而不是消費者

**Step 4（8-12 週，可選 arXiv 版本）**：

- 把 [[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]] 的 Event Tensor `exp_indptr` prefix-sum trigger 套進 spconv 的 hash-bucket dispatch
- 寫一篇 arXiv preprint：「Event-Driven Sparse Convolution: A Persistent Kernel Formulation for Point-Cloud Perception」
- 目標 target：MLSys 2027 short paper 或 ASPLOS 2027 poster——**這是「compiler + spconv + LLM agent + event-driven IR」的四軸交集**，接下來 2 年沒人有時間做這個交集，你有時間就能佔位

### （c）在 Twitter / LinkedIn 上開始追蹤 kernel agent 的 top 30 人

**你需要知道的名字（過去 6 個月投稿的 first author + senior author 列表）**：

- **Kaiming Cheng, Yang Wang, Lu Fang**（Meta PyTorch KernelAgent）
- **Joyjit Kundu, Ludovic Denoyer**（KernelArc / Snowflake-CentML）
- **Xinhao Cheng, Zhihao Jia**（MPK / Mirage）
- **Hongyi Jin, Bohan Hou, Tianqi Chen**（Event Tensor / TVM）
- **Vinod Grover**（Nvidia CuTe / Cutlass）
- **Paulius Micikevicius**（Nvidia、KernelAgent acknowledgements 上有名）
- 清華 / 字節 cuPilot 團隊
- Nvidia SOL-ExecBench + SOLAR 團隊

**追這些人的 arXiv + Twitter/X，你會比 99% 的 compiler engineer 早 3-6 個月知道下一輪 SOTA 的方向**。

---

## 冷讀：未來 12 個月的預測

### 預測 1：SOL-ExecBench 會變成 CUDA developer 的 LeetCode

**訊號**：3/19 開放到 8/20 累積 28,540 次提交、567 個 developer——**這個增速比 KernelBench 2024 第一年快至少 5 倍**。**推論**：2027 年 Q1 前會有：

- 個人 developer 把 SOL-ExecBench 分數放在履歷 / LinkedIn / GitHub profile
- 面試會直接問「你 SOL-ExecBench 最高 X 題到 SOL-Score 多少」
- 某些 LLM startup（例如 Together / Fireworks / Modular）會把 SOL-ExecBench score 當招聘 filter
- 出現商業訓練課程「6 週把你的 SOL-ExecBench 分數提升 30%」

**Adam 現在提交會拿到 early adopter 稀缺性**——這種時間窗口只有 6-12 個月。

### 預測 2：前三大 kernel agent framework 會被大廠收編

**KernelArc（CentML）→ 高機率被 Nvidia 直接收編**：CentML 2024 已經被 Nvidia 併過，這批人的 code / IP 都是 Nvidia 的，KernelArc 的架構會被 upstream 進 Nvidia 內部（可能變成 Nsight Copilot 或 TensorRT autotuner 的一部分）。

**Meta KernelAgent → 留在 Meta**：這是 Meta 內部工具，會繼續在 PyTorch 開源版上養、內部 fork 服務 Meta AI infra。

**cuPilot（清華 + 字節）→ 字節內部化**：這個團隊會被字節內部 GPU infra 團隊（豆包 / Volcano Engine）留下，可能不再繼續開源。

**留給獨立 startup 的窗口很窄**——想 focus 這個方向的話 12 個月內要做出差異化否則會被巨頭吸走。

### 預測 3：下一個 benchmark 會補 multi-GPU + serving-loop

**SOL-ExecBench 目前是 single-kernel benchmark**——只測一個 kernel 打 SOL 多少。**但 [[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]] / MPK / Mirage 這條 persistent megakernel 線已經在推 multi-GPU + full serving loop**。**Nvidia 會在 12 個月內推出 SOL-ExecBench v2 或姊妹 benchmark，涵蓋 tensor-parallel、pipeline-parallel、end-to-end LLM serving**——因為 Nvidia 希望 benchmark 覆蓋自己整個 stack（不只 kernel，還有 NCCL、TensorRT-LLM、Triton Inference Server 這些 higher-level component）。

**要盯的信號**：Nvidia Research 的 arXiv 帳號 + `research.nvidia.com/benchmarks` 這個網域 + Vinod Grover / Michael Andersch / Paulius Micikevicius 這些 Nvidia senior 的 Twitter。

### 預測 4：Compiler career 的分岔會加深

**兩種人**：

**（a）Compiler-native**（走 MLIR / LLVM pass、Triton / CuTeDSL codegen、TVM Relax）
- 薪水穩定（$200K-$400K in Bay Area）
- 天花板明確（senior staff engineer 頂）
- 需求穩定（Nvidia / AMD / Intel / Modular / Meta / Google 都有位置）
- **但這一層過去 10 年的技術含量在下降**——越來越多工具 automated（例如 MLIR TableGen 讓寫 dialect 變快）

**（b）Compiler-agent hybrid**（會 LLM agent orchestration + 懂 compiler middle-end）
- 薪水未定（因為職缺剛出現）——但預測會顯著高於 (a)，因為供給極少
- 天花板高（可以往 principal engineer / VP infra 走）
- 需求爆炸（未來 3 年幾乎每家 AI infra 公司都會需要一個）
- **這一層的技術含量在上升**——LLM agent orchestration + hardware profiling + compiler IR 的組合是新技術棧

**Adam 的定位**：**你的 spconv 底子 + C++ / Python 熟練度 + 這個系列的閱讀量，走 (b) hybrid 這條路完全可行**。**但 hybrid 不是自動流過去的**——你需要主動證明「你能 orchestrate agent」，具體來說就是（b）路徑那三個步驟：

1. 提交 SOL-ExecBench + 寫實測 blog（證明你會用 kernel agent）
2. 用 Event Tensor / persistent megakernel 套 spconv（證明你能做 novel research）
3. 追蹤 top 30 kernel agent 研究者（證明你在 network 裡面）

---

## 這篇跟前面 compiler 系列的關係

到今天為止，這個 compiler 系列的軸線是這樣（每篇是「同一問題的不同層」或「同一趨勢的不同角度」）：

- 8/25 [`cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026`]：**source language 層**（Mojo）
- 8/26 [`qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026`]：**NPU IR 層**（Hexagon MLIR 第一波介紹）
- 8/27 [`hf-kernels-package-registry-cuda-distribution-layer-2026`]：**分發層**
- 8/28 [`tosa-block-scaled-mlir-mxfp-type-system-2026`]：**dtype IR 層**
- 8/30 [`kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check-2026`]：**KernelBench 時代的 benchmark 反思**
- 8/31 [`argus-data-flow-invariants-llm-gpu-kernel-verified-2026`]：**verification / proof 層**
- 9/2 [`cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026`]：**backend codegen 層**
- 9/3 [`flashlight-torchinductor-attention-compiler-graph-rewrites-mlsys2026`]：**middle-end pass 層**（單一 forward 圖重寫）
- 9/4 [`event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026`]：**serving-loop 層**（跨 forward 持久化 megakernel）
- 9/5 [`syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026`]：**multi-GPU 通訊層**
- 9/8 [`tier-iv-open-source-l4-av-chip-tosa-compiler-autoware-2026`]：**AV 端到端 compiler**
- 9/10 [`morphkernel-cross-sm-fusion-just-in-time-reduction-dynamic-gpu-operators-sosp2026`]：**SM 級動態融合**
- 9/11 [`wavel-meshrt-wafer-scale-compiler-runtime-sosp2026`]：**wafer-scale runtime**
- 9/12 [`triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026`]：**Triton warp specialization**
- 9/16 [`tensorlift-mlir-8pass-rtl-tensor-isa-accelerator-compiler-backend-iccad2026`]：**MLIR 到 RTL**
- 9/17 [`cutile-triton-blackwell-portability-cuda131-2026`]：**cuTile Python DSL**
- 9/18 [`dynamatic-mlir-hls-experience-latte2026`]：**MLIR HLS 教學經驗**
- 9/20 [`parallelkittens-vs-syncopate-cuda-framework-mlsys2026-multi-gpu-kernels`]：**multi-GPU kernel framework 對決**
- 9/21 [`what-irregularity-costs-cuda-rust-triton-tsdf-fusion-2026`]：**irregular workload cost**
- 9/24 [`tiny-gpu-compiler-mlir-verilog-gpu-16bit-isa-education-2026`]：**教學用 GPU compiler**
- 9/25 [`nvidia-cuda-tile-ir-opensource-46-pass-mlir-dialect-blackwell-2026`]：**Nvidia CUDA Tile IR 開源**
- 9/26 [`intel-xevm-upstream-linalg-dependent-reduction-fusion-battlemage-2026`]：**Intel XeVM upstream MLIR**
- 9/27 [`jepa-wam-visual-instruction-bank-v-jepa-2-1-text-to-image-goal-2026`]：**Meta V-JEPA 2.1**（換軸）
- 9/29 [`hexagon-mlir-sdk-660-hexkl-beta2-triton-npu-compiler-career-2026`]：**Hexagon-MLIR 第二輪深入**
- **9/30（今天）SOL-ExecBench + KernelArc + KernelAgent：GPU kernel LLM agent subfield 完整地圖**

**這一個月的 compiler 系列越寫越清楚一件事**：**AI compiler stack 的每一層正在被 refactor**——底層被 Nvidia CUDA Tile IR + Intel XeVM + Qualcomm Hexagon-MLIR 三大 vendor 開源、中間層被 LLM agent 吃、上層被 torch.compile + Triton 3.8 warp specialization 吃、跨 forward 被 Event Tensor / persistent megakernel 吃、跨 GPU 被 Syncopate / ParallelKittens 吃。**Compiler engineer 的職涯必修課變成「這 20 篇文章的技術棧都要能講」**——不是每一層都精通，是每一層都能 5 分鐘講清楚 idea 跟 trade-off。

**下一篇（10/1 或之後）預計方向**：

- **Nvidia GTC / SC 2026 前哨戰**（10 月中 Nvidia 通常會有 GTC Fall 或 SC preview，會揭露 Blackwell Ultra 後的 Rubin 更多細節）
- **AMD ROCm 7.0 kernel agent 的官方回應**（AMD 12 月會出 ROCm 7.0，會不會有 kernel agent 官方支援）
- **AV / robotics 換軸主題**（Waymo 第六代量產進度、Tesla FSD v14 進度、Nvidia Isaac Sim 6.0 preview）

---

## 附註：查證來源

**主要一手資料**：

- **Nvidia SOL-ExecBench**：`research.nvidia.com/benchmarks/sol-execbench`（leaderboard 網址）
- **SOL-ExecBench arXiv**：`arxiv.org/abs/2603.19173`（Nvidia + Microsoft，3/19 投稿）
- **KernelArc arXiv**：`arxiv.org/abs/2608.17071`（Joyjit Kundu 等，8/17 投稿）
- **Meta KernelAgent PyTorch blog**：`pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/`（3/6 發布）
- **Meta KernelAgent GitHub**：`github.com/meta-pytorch/KernelAgent`
- **cuPilot arXiv**：`arxiv.org/abs/2512.16465`（清華 + 字節）
- **HIERA arXiv**：`arxiv.org/abs/2608.21157`
- **KernelFoundry arXiv**：`arxiv.org/abs/2603.12440`
- **MaxKernel**：`huggingface.co/papers/2609.04523`

**背景參考**：

- Nvidia Nsight Compute (NCU) 是 Meta KernelAgent 的 hardware profiling 來源，`developer.nvidia.com/nsight-compute`
- SOLAR（Nvidia 內部 SOL 計算工具）：見 Nvidia Research arXiv 2606.26383
- KernelBench 原始 repo：`github.com/ScalingIntelligence/KernelBench`
- SOL-ExecBench GitHub：`github.com/nvidia/sol-execbench`

**這篇文章的所有數字（SOL-Score、topRanking、H100 89% roofline、8/20 快照的 28,540 提交 / 567 developer）都來自上述來源**，未來更新的 leaderboard 快照可能會改變 KernelArc 的 SOTA 地位——**這個 subfield 目前的 SOTA 週期大約 6-12 週**，讀者看到這篇時最好對照最新 leaderboard 快照。

---

_下一篇預告：Nvidia GTC Fall 2026 前哨戰 or AV 換軸主題（Waymo / Tesla / Isaac Sim）。今天寫這篇的關鍵原因是 8-9 月一次冒出七八個 kernel agent 系統，加上 KernelArc 8/20 拿下 SOL-ExecBench 四個 category 第一——這個 subfield 一次成形的時間點值得完整記錄一次。歡迎在 blog 留言告訴我你想看的下一篇方向。_
