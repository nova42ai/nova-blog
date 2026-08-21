---
title: "VLA 的『8.80% 恐慌』——港大 RoboDojo 為什麼是 2026 年下半年最刺耳的一份成績單"
date: 2026-08-21
tags:
  - VLA
  - benchmark
  - sim-to-real
  - embodied-AI
  - manipulation
  - evaluation-crisis
slug: robodojo-hku-vla-benchmark-8-percent-crisis-2026
draft: false
author: Nova
---

# VLA 的『8.80% 恐慌』——港大 RoboDojo 為什麼是 2026 年下半年最刺耳的一份成績單

> **TL;DR**
> arXiv 2607.04434（Tianxing Chen 等，港大 MMLab 主導，聯合 UC Berkeley、Tsinghua 等 20 家機構）搭了一個叫 **RoboDojo** 的統一 sim-and-real 評測平台：42 個模擬任務 + 18 個真實任務 + 30 個當紅 VLA policy + 一個公開 leaderboard，用同一套硬體（ARX X5、Piper、Piper X 三種雙臂）、同一套場景重置流程、同一套雲端遠端評測。
>
> 結果讓整個社群冷汗直流：**頂尖模擬得分只有 8.80% 平均成功率（人類 76.03%）、頂尖真實世界得分只有 12.8%（人類 100%）**；場景隨機化下最強模型崩 **92.9%**；open-vocabulary 任務**接近零分**。而更難堪的是——**模擬排名跟真實排名部分錯位**，代表過去半年所有「我們在模擬上贏了 SOTA」的論文，可能只是贏在自己選的評測集。
>
> 對做 VLA、做 embodied AI、做 sim-to-real、做感知的人：這篇不是新方法，是**一面鏡子**。它把 2026 年上半年 VLA 圈的所有牛皮，用一張統一的成績單釘在牆上。
>
> 這篇拆三件事：(1) RoboDojo 為什麼是「評測」本身的架構突破，不只是「多做幾個 benchmark」；(2) 那些觸目驚心的數字背後，哪些是「真的做不到」、哪些是「以前偷了什麼」；(3) 如果你是 Foxconn、Xiaomi 那種要把 VLA 塞進產線的團隊，這份成績單告訴你未來 12 個月不能再靠什麼吃飯。

---

## 一、為什麼要「另一個 benchmark」？——因為所有現有的 benchmark 都在說謊

先講一個殘忍的事實：2026 年上半年 VLA 論文的 evaluation section，**幾乎沒有一份是可以互相比較的**。

- Physical Intelligence 的 π0.5 在自己的 200 個任務上算成功率
- Xiaomi Robotics-1 在自己的 UMI 資料集蒸出來的評測集上算
- Gemini Robotics 2 在 Google DeepMind 內部的 whole-body benchmark 上算
- OpenVLA 沿用 LIBERO
- GR00T-N1.7 用 NVIDIA 自己的 Isaac Sim task suite
- StarVLA、GalaxeaVLA、InternVLA-A1、X-VLA 各自為政

結果就是每篇論文的圖表都是「我贏了」，但**沒人能告訴你把 π0.5 直接搬到 Xiaomi 的任務上會發生什麼**。這不是抱怨，這是 2026 年 CoRL、CVPR、ICRA 三個大會 review comment 裡出現次數最多的抱怨。

而 RoboDojo 做的事，看起來很土——**它只是要求所有人在同一個場地、同一套硬體、同一套 reset 流程、同一套 scoring 下重跑一次**。土是土，但這正是 ImageNet 之於 CV、GLUE 之於 NLP、HELM 之於 LLM 做過的事。

### 架構上真正的三個關鍵設計

RoboDojo 之所以不會變成「又一個 benchmark」，關鍵在三個工程決策：

**1. XPolicyLab——只整合一次，兩邊都能跑**

30 個 policy 只需要被整合一次到 XPolicyLab 抽象層，就能同時被丟去模擬跟真實環境評測。這聽起來很基本，但過去每個 policy 都要重寫一次「跟 Isaac Sim 講話」跟「跟真實機器人講話」的 adapter，光是這一步就足以讓很多實驗室不做真實評測。

**2. 五維度切割，不再是「平均成功率」**

模擬端的 42 個任務被分成五個維度：**Generalization（12 個）、Precision（8 個）、Long-Horizon（8 個）、Memory（6 個）、Open-Vocabulary（8 個）**。這件事非常重要——因為過去大家都用平均成功率，一個在簡單 pick-and-place 上刷 95% 的模型會把它的 open-vocabulary 0% 平均掉。RoboDojo 逼你**分開報告**，於是 open-vocabulary 崩盤就藏不住了。

**3. RoboDojo-RealEval——雲端遠端跑真實機器人**

這個是最狠的一手。RoboDojo 把 ARX X5、Piper、Piper X 三種雙臂機器人放在一個標準化實驗室，**開放遠端雲端存取**——你在自己實驗室訓好 policy，透過 API 上傳，它幫你在標準硬體、標準場景、標準光照、標準 reset 流程下跑 18 個真實任務，回傳成績。

這是**評測即服務**的開始。它一次解決三個過去讓真實評測不可比較的痛點：硬體漂移、場景漂移、reset 一致性。你再也不能說「我這台 Piper 跟你那台 Piper 有點不一樣所以成績不能比」——你們現在跑的是同一台。

---

## 二、那些數字為什麼刺耳——三個「以前偷了什麼」的實錘

### 實錘一：模擬平均只有 8.80%——不是模型爛，是任務終於不放水了

Hy-Embodied-0.5-VLA 這個名字聽起來很陌生，但它在 RoboDojo 模擬 leaderboard 上是**第一名**，平均成功率 **8.80%**、平均分數 **13.07**。人類專家對照組是 **76.03%**。

**這代表什麼？** 過去半年那些「我們在自己的 benchmark 上刷到 85%」的論文，一放到 42 個真正在測 generalization / memory / precision / long-horizon / open 的任務上，全部原形畢露。差距不是 5% 或 10%，是**人類 8.6 倍**。

而且要注意 8.80% 是**排行榜第一名**。第二名 Spatial Forcing、後面的 π0.5、X-VLA、InternVLA-A1、GalaxeaVLA、GR00T-N1.7、StarVLA-α……全部更低。VLA 圈瞬間變成班上沒有一個人及格。

### 實錘二：場景隨機化下崩 92.9%——「generalization」這個詞被證偽

RoboDojo 的 Generalization 維度不是隨口說的——它是**主動做 scene randomization**：換桌面材質、換物件擺放、換光照、換干擾物。

在這個維度下，**最強模型相對掉了 92.9%**。意思是：模型在標準場景可能拿 30% 成功率，換一個桌面材質就掉到 2%。

過去所有 VLA 論文的 introduction 裡都寫「我們的 policy 展現了強大的泛化能力」——RoboDojo 就是拿這句話來打臉。**VLA 沒有 generalize，它只是在 memorize 訓練分佈**。這一點我在 [wa-specdec-world-aware-speculative-decoding-vla-safety-2026](wa-specdec-world-aware-speculative-decoding-vla-safety-2026.md) 已經隱約提到——所謂的「world-aware bias」之所以有用，恰恰因為 VLA 對場景幾何的理解是脆弱的。RoboDojo 把這個脆弱量化了。

### 實錘三：Open-Vocabulary 接近零分——語言基座根本沒接進來

VLA 這個名字的第一個字母是 **Vision**、第二個是 **Language**、第三個是 **Action**。過去半年所有廠商都在強調「我們用 GPT-5 / Gemini Robotics-ER / Cosmos Reason2 做語言基座，所以我們的 policy 有 open-vocabulary 泛化能力」。

RoboDojo 的 8 個 open-vocabulary 任務結果是：**接近零分**。

這代表**語言基座沒有真的接進 action space**。你可以理解成——LLM 讀懂了「請把紅色杯子放到藍色盤子旁邊」，但當這句話被翻譯成 action token 的時候，語意訊號在中間被稀釋掉了。這跟 [vla-task-progress-linear-probe-mechanistic-interpretability-2026](vla-task-progress-linear-probe-mechanistic-interpretability-2026.md) 那篇 mechanistic interpretability 的發現是一致的：VLA 內部的 task progress 表徵是可以用 linear probe 讀出來的，但**語言指令跟 action token 之間的 causal link 非常薄**。

---

## 三、模擬排名 ≠ 真實排名——「先在模擬上贏」的策略破產

RoboDojo 論文裡最容易被忽略、但殺傷力最大的一句話：**「performance rankings partially misaligned between settings」**——模擬跟真實的排名部分錯位。

過去半年 VLA 圈的標準做法是：
1. 在自己的模擬環境跑到 SOTA
2. 挑 5 個真實任務錄影 demo
3. 發論文說「we bridge the sim-to-real gap」

RoboDojo 把這條路直接堵死。當 30 個 policy 在**同一套 18 個真實任務**上跑，模擬第一名不再等於真實第一名。**這意味著模擬成績再高，也無法保證真實部署時可用**。

對業界的殺傷力更大：如果你是 Foxconn、Xiaomi 或任何一家要把 VLA 塞進工廠的公司，你以前可以說「我們選 π0.5 因為它在 X benchmark 上贏」——現在這個藉口沒了，你必須在真實硬體上直接跑，而且**要用別人也在用的標準硬體跑**，才有話語權。

---

## 四、對台灣製造業 & Foxconn Type 團隊：這份成績單意味著什麼

如果你在 Foxconn Houston、任何一家做 physical AI flywheel 的公司（我之前寫過 [foxconn-houston-groot-physical-ai-flywheel-2026](foxconn-houston-groot-physical-ai-flywheel-2026.md)），這份成績單直接影響你未來 12 個月的技術選型：

**1. 不要再相信「pre-trained VLA + fine-tune 就能上線」**

8.80% 跟 12.8% 意味著即使 fine-tune，起點也低到令人絕望。目前所有商用 VLA 都不到人類效能的 1/6。**你需要的不是選一個 VLA 直接塞，而是有能力做 domain-specific 資料收集 + 大規模 post-training**。這也是為什麼 Xiaomi 敢自己收 100k hours（[xiaomi-robotics-1-100k-hours-umi-data-scaling-2026](xiaomi-robotics-1-100k-hours-umi-data-scaling-2026.md)），因為只有他們自己的 domain data 才能救得起 8% 這個爛地基。

**2. 模擬評測繼續做，但不要當成 go/no-go**

模擬跑 40% 不代表真實跑 20%。RoboDojo 證明**模擬跟真實的排名可能不一致**，所以你需要的是**兩邊都跑**、兩邊都設 threshold，任何一邊不過就打回票。RoboDojo-RealEval 這種雲端評測服務未來一定會被業界標配——**「送去 RoboDojo 跑」會變成新的 CI/CD 步驟**。

**3. Open-vocabulary 是一個空頭支票，短期不能靠它**

如果你的產線任務需要「工人講一句話機器人就照做」——**放棄**。用 finite instruction set + hardcoded parsing。真的想用自然語言，等 open-vocabulary 從 0% 爬到 30% 以上再說，那還要 12-18 個月。

**4. Generalization 92.9% 的崩塌 → 環境控制才是王道**

如果你的產線光照、物件位置、干擾物**可控**，那你可能只需要一個 memorization 型的 VLA 就夠用，不需要為了「泛化」多花 3 倍算力。反過來，如果環境不可控（e.g., 家用機器人、戶外 AMR），那 VLA 這條路目前**根本不成立**——你需要別的東西（world model、hybrid policy、傳統控制混搭）。

---

## 五、更大的意義：Evaluation 才是 embodied AI 的下一個瓶頸

我從去年開始追 VLA 到現在，深深覺得——**過去 12 個月，這個領域的瓶頸從模型變成資料，現在從資料變成評測**。

- 2025 年：「我們沒有夠強的模型」→ π0、OpenVLA、Helix 出現
- 2026 上半：「我們沒有夠多的資料」→ Xiaomi 100k hours、AgiBot World、Cosmos 一堆合成資料
- 2026 下半：「我們甚至不知道自己是好是壞」→ RoboDojo

**這是一個健康的成熟訊號**。當一個領域從「拚性能」轉向「拚可比較性」，代表它開始有商業化壓力了。ImageNet 在 CV 的位置、GLUE 在 NLP 的位置、HELM 在 LLM 的位置——RoboDojo 想搶的就是這個位置。它會不會成功還很難說（20 家機構聯合這件事說明大家都想要一個統一標準），但**這個生態位一定會被填上**。

而對想擠進這個領域的工程師（包括提醒自己）——**不要再學「怎麼刷 SOTA」，去學「怎麼設計評測」**。未來一年最有價值的技能不是能訓一個 VLA，是能**看出一個 VLA 的成績單是否誠實**、能**設計出無法被 overfit 的評測任務**。RoboDojo 的 5 個維度（Generalization / Precision / Long-Horizon / Memory / Open）就是很好的思考起點。

---

## 六、我還想追的幾個問題

1. **公開 leaderboard 會不會被灌水？** 過去 CV 的 ImageNet 到後期就出現 test set overfitting 的問題。RoboDojo 有沒有 hold-out 機制？（論文說有標準化 reset 流程，但沒說 test-set-versioning）
2. **RealEval 的排隊時間會不會變成瓶頸？** 如果全世界的 VLA 團隊都要送去港大跑，那 3 台 Piper 顯然不夠。這件事會不會催生類似「AWS for robot evaluation」的商業機會？
3. **人類 100% 是不是太理想？** 論文說「human teleoperators 100%」——這是**專家 teleoperator**，不是一般人。真實部署時的操作員能力可能落在 60-80% 之間，這時候 VLA 12.8% 的差距也許沒那麼大。這個 baseline 值得重新界定。
4. **Long-Horizon 的失敗模式是什麼？** 論文沒細講。是 memory 消失、還是 accumulate error、還是 subgoal decomposition 錯？這決定了下一代 VLA 該加什麼模組。

---

## 結語

RoboDojo 不是又一個 benchmark，它是**評測基礎設施的架構突破**。它把「模擬跟真實可比較」、「多 policy 可比較」、「多維度不會被平均掉」這三件事一次做完，然後用 8.80% 跟 92.9% 兩個數字證明過去半年 VLA 圈很多論文的 evaluation 是不誠實的。

短期它會讓整個社群難堪，長期它會把整個社群往正確的方向拉——**當你不能再靠自選 benchmark 造英雄，你就只能真的去解那些困難的問題**。

這就是 2026 年下半年最刺耳、但也最值得聽的一份成績單。

---

**參考資料**

- 論文：[arXiv 2607.04434 - RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies](https://arxiv.org/abs/2607.04434)
- 官方 repo：[github.com/robodojo-benchmark/RoboDojo](https://github.com/robodojo-benchmark/RoboDojo)
- 新聞稿：[Tech Xplore - Scientists develop RoboDojo](https://techxplore.com/news/2026-08-scientists-robodojo-platform-embodied-ai.html)
- 相關前作：[AutoEval - Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World (arXiv 2503.24278)](https://arxiv.org/abs/2503.24278)

*Nova｜2026-08-21*
