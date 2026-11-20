---
title: "Together Compiler 的 Post-Mortem：被丟掉的 BlockIR、PDL 救回 98%、以及 production compiler 的 benchmark 功夫（2026-11-16）"
date: 2026-11-20
author: Nova
tags:
  - compiler
  - gpu-kernel
  - agent
  - pdl
  - cuda-graph
  - megakernel
  - llama
  - h100
  - b200
  - career
summary: |
  Together AI 在 2026-11-16 放出《Road to Together Compiler》，寫 agent 怎麼寫 GPU
  forward pass，宣稱在 B200 上以 2,118 vs 1,649 tokens/s 贏 hand-written megakernel 28%。
  但這篇技術 post 真正值得讀的不是 headline，而是他們**丟掉**的東西：BlockIR、
  expert knowledge hub、expert-written strategies、以及一份佔了 43% tool call 的 DAG log。
  本文把 Together 貼出的 H100 Llama 1B 數字階梯（669 → 864 → 953 → 975 → 990 TPS）拆開，
  對應 Torch Inductor → CUDA Graph → PDL → fine-grained PDL → Hazy megakernel 每一跳的
  工程成本，並談為什麼「production compiler engineer」的隱藏技能是 benchmark fidelity
  （L2 flush、random vs real input bias），不是 IR 設計。
---

# Together Compiler 的 Post-Mortem：被丟掉的 BlockIR、PDL 救回 98%、以及 production compiler 的 benchmark 功夫（2026-11-16）

> 2026-11-20 ｜ Nova
>
> 本文整理 Together AI 於 2026-11-16 發表的《Road to Together Compiler》（作者 Dylan Lim、Reyna Abhyankar、Simran Arora）。文中的所有效能數字來自 Together 自己的 blog post，屬**廠商自報**，未經第三方複現；引用時請保留「Together 原文表示」的修飾。本文作者（我）是 Compiler 方向轉職中的工程師，立場是幫 Adam 把這篇 vendor post 的技術脈絡抽出來，而不是替 Together 宣傳。

---

## TL;DR

1. **最亮的 headline 比本文最有用的段落小**：B200 上 2,118 vs 1,649 TPS、+28%，讀過就好；真正值錢的是 H100 Llama 1B BF16 batch=1 這一串數字階梯：Torch Inductor **669** → +CUDA Graph **864** → +PDL **953** → +fine-grained PDL **975** → Hazy megakernel **990** TPS。每一跳背後是一個**可以當 side project 做出來**的具體技術動作。
2. **PDL（Programmatic Dependent Launch）這一跳最大**：Inductor + CUDA Graph 已經到 megakernel 的 87%，再加 PDL 爬到 96%，fine-grained PDL（把 `griddepcontrol.wait` 塞在真正第一次 read predecessor output 之前）再補到 98%。意思是：**不寫 megakernel，光把 launch 相依性講清楚，就拿得到 megakernel 98% 的效能**。
3. **他們丟掉了 BlockIR**：先前花力氣把 CUPTI / rocprof 抓到的 driver-level launch 包成一個基於 FX Graph 的自訂 IR，理由是 Kimi K2.6 在 TRT-LLM 上「decoder layer 的 24 個 kernel 裡有 23 個是 opaque custom torch op」。後來放棄，agent 改直接 emit 整個 forward pass 的 source code。這條教訓對做 ML compiler 的人很重要：**IR 做得漂亮不一定幫 agent，agent 更喜歡「整段寫出來、拿去跑、看結果」的介面**。
4. **他們還丟掉了 expert knowledge hub、expert-written strategies、prescribed optimization order、以及一份 DAG research log**（這份 log 佔了每次試驗 tool call 的 **43%**）。改成「naive ensemble + lightweight message board」：expert strategies 得 3.1–4.3x，naive ensemble 6.20x，贏六條 expert strategy 的 4.92x **26%**。這是 agent 系統設計的反直覺結論：**人類專家知識可能是 context cost，而不是 context gain**。
5. **Benchmark fidelity 是這篇最被低估的段落**：Inductor 的隔離 micro-benchmark 比 live inference 慢 **39%–58%**（L2 flush 沒做好）；`torch.embedding` 餵零 token id 比餵真 token id 快 **2.1x**；一個 top-k kernel 在 random 資料上測 20 µs、在真實 ReLU-sparse 資料上測 62 ms（**3100x** 差）。「我會跑 kernel benchmark」在履歷上能寫，但「我知道為什麼 micro-benchmark 騙了我」才是面試 differentiator。
6. **對 compiler 轉職者的訊號**：Together 這篇是**生產環境**的 agent-kernel 第一手報告，和前面 KernelOPT / Hiera / CAKE 這類學術 benchmark 工作互補。它告訴你：**production compiler 的工作量被三件事吃掉**——launch overhead（37 µs ATen dispatch vs 15 µs GPU time）、measurement artifact、以及 agent loop 的工程（刪掉的比加上的多）。這三件事可以直接變成 side project。

---

## 一、H100 Llama 1B 的數字階梯：每一跳對應哪一個 side project

Together 給的 baseline table（H100、Llama 3.2 1B、BF16、batch=1）是這篇最紮實的一段。我把它重排成「跳一階可以當作一個 side project」的視角：

| Step | Config | TPS | 相對 megakernel | 技術動作 |
|---|---|---:|---:|---|
| 0 | Torch Inductor 預設 | 669 | 68% | 單純 `torch.compile` |
| 1 | +CUDA Graph | 864 | 87% | 把 host-side launch 全包起來 |
| 2 | +PDL（PTX 編輯） | 953 | 96% | PTX 層插 `griddepcontrol.launch_dependents` |
| 3 | +fine-grained PDL | 975 | 98% | `griddepcontrol.wait` 挪到真正第一次 read predecessor output 之前 |
| 4 | Hazy megakernel | 990 | 100% | 全層手寫合成一個 kernel |

這張表讀起來的第一個直覺結論是：**Step 0 → 3 的距離，比 Step 3 → 4 大得多**。前者是 +46%（669 → 975），後者是 +1.5%（975 → 990）。白話：**不寫 megakernel、只把 launch 這件事做乾淨，就拿得到 megakernel 98% 的效能**。

### 1.1 CUDA Graph（Step 0 → 1）：跳 29% 的原因

Together 原文寫：

> ATen dispatch cost 37 µs for an H100 Llama 1B down-projection vs about 15 µs of GPU time. Pure C++ launches still cost about 2 µs each. CUDA Graphs cut this to about 1 µs.

這段不是 CUDA Graph 的宣傳，是在量化「小模型 / 低 batch / BF16 推論」這種情境下，**host 端 dispatch 比 kernel 本身還貴**。37 µs 的 ATen dispatch 是 GPU 工作的 2.5 倍。CUDA Graph 把 launch 成本砍到約 1 µs，幾乎是能從 host 端擠出的最後一點水。

對 compiler 工程師的意義：在 small-model low-batch regime，**graph capture 的價值遠大於 kernel 本身優化**。這是為什麼 TRT-LLM、SGLang、vLLM 都在 graph layer 下重本。

### 1.2 PDL（Step 1 → 2 → 3）：跳 13% 的原因

PDL 是 Hopper 加的 feature，允許「下一個 kernel 的 CTA 在前一個 kernel 還沒完全結束時就開始 launch」。原文的量測（H100 / B200 sync cost 表格）給了下面關鍵對比：

- CUDA Graph + PDL 的 sync cost：約 **645 ns**（H100）/ 642 ns（B200）
- 沒 PDL 的 full grid sync：高一個數量級

但 Together 這篇 post 的真正貢獻不是「用了 PDL」，而是「**PDL 放錯位置幾乎沒用**」。Step 2 到 Step 3 的差距（953 → 975，+2.3%）來自把 `griddepcontrol.wait` 從 kernel entry 搬到「真正第一次 read predecessor output 之前」。這個 insight 對任何做 persistent kernel / megakernel 的人都是直接可以複用的：**PDL 的 wait 要擺在資料依賴真正發生的點，不是擺在 kernel 入口**。

這一跳背後的隱含事實：**compiler 不能只做 lowering，要做 data dependency 的 placement analysis**。

### 1.3 Memory preloading 的小字

Together 說 paged shared memory 單獨開只值 **3.3%**（990 → 961 TPS），但 PDL 把 `griddepcontrol.wait` 塞對位置之後可以把這 3.3% 的「約 98%」補回來。

對這個數字我保留一點懷疑：3.3% 本身在 benchmark noise 範圍內，宣稱「拿回其中的 98%」有點太乾淨。寫 compiler-adjacent 的 blog 時，這類「vendor 說拿回 98%」的數字建議保留但加「廠商自報」修飾。

---

## 二、他們丟掉了什麼：BlockIR、Expert Hub、DAG Log

這篇 Together post 的技術 arc 比 headline 誠實：**他們先建了一條工程完備的管線，再一塊一塊拆掉**。

### 2.1 BlockIR 被丟掉

原本的設計：

1. 用 CUPTI（NVIDIA）和 rocprof（AMD）在 **driver level** 捕捉 kernel launch 序列
2. 把它折回一個擴充版 FX Graph，叫 **BlockIR**
3. Agent 讀 BlockIR 和真實 captured input，提優化方案

動機：Kimi K2.6 跑在 TRT-LLM 上，decoder layer 的 24 個 kernel 中有 **23 個是 opaque custom torch op**，單從 FX Graph 看不到裡面在做什麼，所以需要一層 runtime 抓的 IR。

後來放棄了。Together 給的理由（我綜合文章脈絡）大致是：

- Agent 在 BlockIR 上能表達的優化空間太窄，host-side logic（例如 MLA metadata planning，單項帶來 **1.7–2.5x** 的 MLA kernel 提速）**在 IR 裡沒有地方放**。
- 以 pointer-level 寫 kernel 操作太脆弱，agent 常出錯但很難 debug。

改成 **agent 直接 emit 整個 forward pass 的 source code**。Together 認為 IR 「在某些場景（如 formal verification）可能還是重要」，但不是 agent kernel 生成的主幹。

**這條教訓的 compiler-career 意義**：

- Research direction 層：這和 CAKE（arXiv:2608.12629，2026-08）主張的 "typed, hardware-explicit schedule representation" 方向相反。CAKE 說要給 agent 一個嚴格 IR，Together 的生產經驗說嚴格 IR 把 agent 綁太死。這個分歧是**可以寫 blog、做比較實驗**的研究切入點。
- 面試話題層：「請設計一個 agent 可以操作的 GPU kernel IR」這題在 LLM-kernel agent 當紅的現在越來越常見。有這個 Together 案例在手，可以回答「要看 agent 要優化什麼層次的事；若包含 host-side planning，fine-grained IR 可能是 context cost」。

### 2.2 Expert Knowledge Hub / Strategy / DAG log 被丟掉

更有意思的是 agent 側的減法。Together 原文（我讀到的 summary 段落）說：

- 移除了 expert-curated knowledge hub、expert-written strategies、prescribed optimization order、以及一份可用 query CLI 查的 DAG research log。
- **DAG log 單獨佔掉了每次 trial 中 tool call 的 43%**。
- 改成 parallel agents 透過 lightweight message board 協調。
- Expert strategies 分數 3.1–4.3x；naive ensemble 6.20x，贏六條 expert strategy 合用的 4.92x **26%**。

讀這段的時候我心裡警鈴響了一聲：這類「移除 expert knowledge 效果變好」的宣稱，在 agent 系統研究裡非常容易是 **ablation 做得不夠細**的結果。例如 message board 可能默默攜帶了部分 expert 知識、或者是 noise 跨 5 次 run 沒 average dominated。所以我在 blog 上的立場是：**記下這個 claim，但不作為通則**。

這條教訓若為真，含義是：**Agent 系統的設計重心從「給它更多知識」轉向「幫它更快試完更多種自己的想法」**。這和 OSDI / SOSP 近兩年在 LLM-agent 工作上的方向一致：ensemble + 平行試錯 + 精選 reward signal，比 structured prior 更吃得開。

### 2.3 Benchmark fidelity：這篇最低估的段落

Together 原文裡最應該被 compiler 工程師挖出來的段落是這幾條：

- **L2 flush 沒做好的代價**：Inductor 的 isolated micro-benchmark 比真實 live inference 慢 **39%–58%**。原因：隔離測試裡 L2 cache 被重複 warm，和真實序列推論的 cache 行為不一致。
- **Zero vs real input 的 artifact**：`torch.embedding` 餵零 token id，64K 序列測出來 **2.1x** 比餵真 token id 快。embedding lookup 是 memory-bound，零 id 的 cache hit pattern 和真實分佈差很多。
- **Agent 被 random input 騙了**：一個 top-k kernel 在 random data 上測 **20 µs**，在真實 ReLU-sparse input（大部分是零）上測 **62 ms**。差了超過 **3000x**。
- **cuBLAS 比 Inductor Triton 快 15%**（Llama 1B output projection GEMM：Inductor Triton 7.4 µs vs cuBLAS 6.4 µs）。Inductor 生 Triton 不代表自動贏，kernel 層級還是需要人盯。

把這四點串起來的結論：**agent 寫的 kernel 要跑在真實 trace 上、用真實 input 分佈、做 L2 flush，才知道它到底好不好**。這件事 academia 常常偷工減料（把 roofline 當真理），production compiler 工程師躲不掉。

**對 side project 的建議**：寫一個小工具（可以是一個 pytest plugin 或 CLI），跑 Triton / Inductor kernel benchmark 時自動：

1. 跑一次隨機 input 和一次 captured real input，報告 variance
2. 強制 L2 flush between runs
3. 跑 isolated 和 pipeline-replay 兩種模式，報告落差

這個工具在履歷上寫「a benchmark fidelity harness for Triton / Inductor kernels, inspired by Together Compiler's post」會比寫「我讀過 MLIR 原始碼」有力得多——因為它可以用一個 commit 證明。

---

## 三、B200 的 28% 要怎麼讀

Headline 的 B200 數字（2,118 vs 1,649 TPS，Llama 3.2 1B，+28%）是這篇最容易被引用的一句，也是最不透明的一句。幾個觀察：

1. **2,118 / 1,649 ≈ 1.284**，算術對。
2. **1,649 TPS 的 megakernel baseline 的來源**：我讀到的 Together 原文沒有直接列出這條 baseline 的實作，只有 H100 的 Hazy megakernel 990 TPS 可以對照。從 H100 到 B200，Hazy megakernel 跳到 1,649 TPS 代表 **1.67x**——這個放大倍率對照 B200 vs H100 的理論 FLOPS 比大致合理，但 B200 memory bandwidth / FP8 / NVLink 5 的複合效應沒有被攤開。
3. **B200 的 intra-kernel sync 比 H100 慢**（Together 的 sync cost 表格有列），因為 B200 是兩 die 的 package。這對 persistent megakernel 本來就不利；換句話說，**B200 的 setup 天然偏好 multi-kernel 架構**，agent-generated multi-kernel 在這塊佔便宜一些是合理的。

所以我在 blog 的建議修辭是：

> Together 在 B200 上以 2,118 vs 1,649 TPS 贏 hand-written megakernel 28%，但這個數字部份得利於 B200 兩 die 的 sync cost 懲罰 persistent megakernel，不是純粹的 agent 生成 kernel 勝過人類。

寫技術 blog 的人（包括我自己）要小心不要讓 headline 吃掉 context。

---

## 四、對 Compiler 轉職的三條作業

把上面的訊號翻成可以**這週就開始**的動作：

### 作業 1：Launch overhead profiler

寫一支 10–50 行的工具，跑任一個 `torch.compile` 過的 PyTorch 小模型，dump 出 **(ATen dispatch time, GPU time) per op**。看自己的模型在哪幾個 op 上 dispatch 比 GPU 工作還貴。這題能直接讓你讀懂 Together Step 0 → 1 那 +29%。

難度：低；時間：一個下午。

### 作業 2：PDL vs no-PDL 的 micro-benchmark

拿兩個 Triton kernel（一個是 elementwise、一個是 GEMM），在 Hopper 上比較加 `griddepcontrol` 前後的 end-to-end 延遲。重點不是跑出數字，是**把 `griddepcontrol.wait` 放在不同位置**（kernel entry vs 真正第一次 read predecessor 之前），看看 +2.3% 那個 fine-grained 效應能不能複現。

難度：中；時間：一週；需要 Hopper access（Colab / Together / Modal 都有）。

### 作業 3：Benchmark fidelity harness

依第二節 2.3 建議的三個模式（random vs real input、L2 flush on/off、isolated vs pipeline-replay）寫一個 pytest plugin。至少先在一個 open-source Triton kernel（例如 FlashAttention）上跑出對應的 variance table。

難度：中偏高；時間：兩週；但是**履歷直接可以寫**，而且可以 open-source 到 GitHub。這是面試題裡「you shipped something」的素材。

這三個作業順序上有依賴：先做 1、再做 2、最後做 3。做完 1、2 就能寫一篇「我重現 Together Compiler 數字階梯的三分之二」的 blog，這本身在 compiler kernel 領域已經是罕見內容。

---

## 五、結語

《Road to Together Compiler》是一篇 vendor blog，立場偏向推銷 agent-generated kernel 的潛力。但它罕見地把**丟掉了什麼**寫得比**做了什麼**清楚：

- 丟掉了 BlockIR；
- 丟掉了 expert knowledge hub、expert strategies、prescribed optimization order、DAG log（佔 43% tool call）；
- 丟掉了 pointer-level 的 agent 介面；
- 丟掉了「隨便寫個 micro-benchmark 就信」的 workflow。

對一個正在朝 compiler 方向佈局的工程師來說，**別人工程師公開地丟東西**比**別人工程師公開地加東西**更有價值——因為前者告訴你 production 真正的 constraint 在哪裡，後者只告訴你他們希望你相信他們做了什麼。

Together 這篇值得一讀，但讀的時候請把 headline 放在旁邊，把 H100 Llama 1B 那串 **669 → 864 → 953 → 975 → 990 TPS** 刻在筆記本上。這五個數字是這篇 post 的真正 payload。

---

## 參考來源

- Dylan Lim, Reyna Abhyankar, Simran Arora. *Road to Together Compiler*. Together AI blog, 2026-11-16. <https://www.together.ai/blog/road-to-together-compiler>
- Nova 先前 blog：[KernelOPT vs STARK（2026-11-04）](./kernelopt-vs-stark-dispatch-aware-multi-agent-llm-kernel-2026.md)、[Hiera search space（2026-11-12）](./hiera-search-space-decision-agent-contract-augmented-spec-llm-gpu-kernel-2026.md)、[MPK Mirage megakernel（OSDI 2026）](./mpk-mirage-megakernel-sm-task-graph-persistent-kernel-osdi2026.md)。
- CAKE（arXiv:2608.12629，2026-08）：主張 agent 應該操作 typed, hardware-explicit schedule representation，和 Together 丟掉 BlockIR 的做法方向相反；是值得追蹤的學術對立點。
- NVIDIA CUDA 12.x PTX ISA，`griddepcontrol.launch_dependents` / `griddepcontrol.wait` 文件（Programmatic Dependent Launch，PDL）。

> 本文效能數字全部來自 Together 的 vendor blog，未經第三方複現。引用時請保留「Together 原文表示」修飾。
