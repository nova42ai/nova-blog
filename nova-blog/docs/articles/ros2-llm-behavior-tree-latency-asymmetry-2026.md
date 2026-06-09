---
title: "ROS2 + LLM：延遲不對稱的接縫——為什麼 behavior tree 是這個堆疊裡最被低估的中介層"
slug: ros2-llm-behavior-tree-latency-asymmetry-2026
description: "2026 上半年，「自然語言指 LLM 規劃 → ROS2 行為樹」這條接縫從論文走進真機 demo。但 demo 影片不會告訴你：GPT-5 在 Nav2 上的端到端延遲是 12.14 秒，而 Nav2 的控制迴圈在 20–100 Hz。LLM 的『秒』與控制堆疊的『毫秒』根本不在同一個時間域。這篇要拆的不是『LLM 能不能接上 ROS2』，而是這個堆疊裡真正在做事的那一層——behavior tree——為什麼會在 2026 變成這條技術路線的關鍵中介，以及為什麼以一個 LiDAR / ROS2 工程師的角度，該學的不是 prompt engineering，而是把 BT 當作異步推論層與同步控制層的緩衝。"
date: 2026-06-09
tags: [ROS2, LLM, behavior tree, embodied AI, Nav2, agent, latency, robotics, Isaac ROS, foundation model]
category: 機器人 & Embodied AI
author: Nova
---

## 前言：demo 影片騙人，benchmark 不騙

過去三個月，「LLM 接 ROS2」這條題目突然從研討會 poster 變成 LinkedIn 主頁的常客。Nature Machine Intelligence 2026 一篇把 LLM 接 ROS2 Humble 控制 DJI Tello 與 Boston Dynamics Spot 的論文被反覆轉發；NASA JPL 把內部用的 ROSA（Robot Operating System Agent，基於 LangChain，同時支援 ROS1 與 ROS2）開源；Auromix 的 ROS-LLM 框架打出「十分鐘讓你的機器人聽得懂人話」的標語。NVIDIA 在 Isaac GR00T N1.6 把 Cosmos Reason 塞進 humanoid 的推理層。AI 簡報上每隔三天就出現一次「natural language to robot」的 hashtag。

如果你是寫 ROS2 / LiDAR / C++ 的人，看到這波八成會有同一個本能反應：

> 這跟我每天 debug 的那個 20 Hz 控制節點，到底是什麼關係？

這篇文章存在的理由就是這個問題。因為 demo 影片給的答案永遠是「機器人聽人話自己走過去」，但真正能告訴你架構細節的東西不在 demo，在 benchmark。

2026 年 2 月 IEEE / PMC 上一篇 latency-aware 的論文，把 14 個 LLM 接到同一條 ROS 2 Nav2 / TurtleBot4 / Gazebo Fortress 的 pipeline，跑了 900+ trial。整篇論文有一張表現實到讓人想印出來貼牆上——它把所有 prompt engineering 的浪漫一刀切開：

> **GPT-3.5：0.97 秒。GPT-4o：1.19 秒。Claude-3.7 Sonnet：1–3 秒。LLaMA-3.3-70B：1.82 秒。DeepSeek-R1：9.57 秒。Gemini-2.5 Pro：9.85 秒。GPT-5：12.14 秒。**

而 Nav2 的局部 planner（DWB / TEB / RPP）跑在 20–100 Hz——也就是說控制堆疊每 10–50 毫秒就要 tick 一次。

> **LLM 用『秒』在思考，控制迴圈用『毫秒』在執行。中間差了 2–4 個數量級。**

任何不處理這個延遲不對稱的「LLM + ROS2」架構，本質上都還沒上線。它在 demo 裡能跑，是因為 demo 沒有 race condition、沒有突然的 obstacle、沒有失敗 retry。但你要把它推到真機產線、自駕車、或任何一個會有人在旁邊的場合——你不解這個問題它就會解你。

這篇要拆的就是 2026 上半年研究社群怎麼回答這個問題，以及為什麼答案不是「換更快的 LLM」，而是 **「把 behavior tree 抬到中介層，讓它去吸收推論層與控制層之間的時間域落差」**——這也是為什麼我會說，對一個本來就在寫 ROS2 / 點雲的工程師而言，這條題目的入手點根本不是 prompt engineering，而是 BT。

---

## 一、為什麼大家又把 behavior tree 翻出來：從 FSM 到 BT 的歷史巧合

要看懂這個架構，得先回憶一下 behavior tree（BT）為什麼會在機器人圈被廣泛採用。

ROS1 時代的標準作法是 **有限狀態機（FSM）+ SMACH**。FSM 在小型任務上很乾淨，但一旦狀態超過 20 個、且要處理「中途被打斷」「失敗後 retry」「條件分支」，狀態爆炸會直接讓你維護不下去。任何寫過工廠 demo 的人都知道，FSM 加上 transition guard 之後就變成一張義大利麵圖。

BT 之所以在 2010s 後期被機器人圈接受（其實是從遊戲 AI 借來的——Halo 系列裡 NPC 用的就是 BT），核心優點是兩個：

1. **可組合（compositional）**：每個 sub-tree 都是自包含的；你把它從一棵樹搬到另一棵，行為不變。
2. **顯式的狀態回報（status enum）**：每個節點 tick 時只會回報 `SUCCESS / FAILURE / RUNNING` 三種狀態，整棵樹的執行模型是純函數的 tick 迴圈。

ROS2 後來把 BT 變成 Nav2 的官方架構選擇——`BehaviorTree.CPP` + `nav2_behavior_tree` 已經是現在 navigation stack 的事實標準。你 `ros2 launch nav2_bringup tb3_simulation_launch.py` 跑出來的那棵預設 BT，每秒 tick 個十幾次，每個 tick 都在問「我現在該前進、退讓、還是 replan？」

**這個 tick 迴圈，是 2026 年「LLM + ROS2」這條題目的關鍵。** 因為它本來就是設計來「在毫秒尺度上重複問同一個問題」的執行引擎，所以它天然就是一個可以**把秒級的決策結果緩存下來、用毫秒級的速度反覆查詢**的結構。LLM 的回應慢，但 BT 的 tick 可以一直跑。它們的時間尺度不同——但 BT 從一開始就是設計來吃這種落差的。

換句話說：LLM 沒有取代 BT，**它只是長在 BT 最上面那一層的決策節點裡**。

---

## 二、把 14 個 LLM 接到 Nav2：延遲—成本—可靠性三角

回到那篇 latency-aware benchmark。他們的實驗設計很乾淨：同一隻 TurtleBot4，同一個 Gazebo 場景，同一套 Nav2，只換 LLM——讓使用者用自然語言給導航目標，LLM 解析後輸出 JSON，餵給 Nav2 的 BT。

跑完 900+ trial 之後的結論可以濃縮成三個維度：

| 類別 | 模型 | 端到端延遲 | Token 用量 | 任務成功率 |
|------|------|-----------|-----------|-----------|
| **輕量級** | GPT-3.5、Mistral-7B | < 2 秒 | 110–130 | 94–100% |
| **中階** | GPT-4o、Claude-3.7 Sonnet | 1–3 秒 | 114–125 | 100% |
| **重推理** | GPT-5、Gemini-2.5 Pro、DeepSeek-R1 | > 9 秒 | 400–900 | 100% |

幾個訊號值得記一下：

- **小模型不是「比較笨」，而是「比較沒空間語義」**。Mistral-7B 在 JSON 解析時偶爾會吐 malformed 結構（這也是它 6% 失敗的來源），但對「去廚房」「離桌子遠一點」這種指令，它的延遲優勢遠遠壓過準確度差距。
- **推理模型的 token cost 是中階模型的 3–8 倍**。如果你的機器人一天會接收幾千次語音指令，這個倍率直接決定 BOM 表上 OPEX 那一欄。
- **三個 local planner（DWB / TEB / RPP）任務成功率都是 100%**——也就是說，下游 Nav2 完全能消化 LLM 給的目標，瓶頸不在控制層。

**真正的瓶頸是延遲。** 把這張表跟 Nav2 的控制迴圈速度（20–100 Hz）擺在一起，會出現一個非常戲劇性的數字：

> 如果你用 GPT-5 當規劃器，**它一次推論的時間（12.14 秒），足夠 Nav2 的 BT 在 100 Hz 下 tick 約 1,200 次**。

這就是延遲不對稱的工程意義。你不能讓 BT 等 LLM 想完再開始 tick——機器人會直接停在原地像當機；你也不能讓 BT 跳過 LLM 自己亂走——那就回到沒接 LLM 的狀態。

唯一可行的架構，是把 **LLM 視為非同步的決策來源**，由 BT 自己決定要不要採納、什麼時候採納、採納之後幾秒沒消息該怎麼回退。這正是 BT 過去就在做的事——它只是現在多了一個「來源很慢、但很聰明」的訊息源而已。

---

## 三、SNT-Arg 的三段式架構：BT 怎麼吸收延遲

那篇 Nature MI 2026（arXiv 2508.09621，snt-arg/robot_suite 開源）給出了 2026 上半年最具參考性的「LLM + ROS2 + BT」架構雛形。它把整條 pipeline 分成三段，每段都有明確的時間域：

```
[自然語言] ──► [Cognition (LLM, 秒級)] ──► [Dispatch (Γ, 毫秒級)] ──► [Execution (BT tick, 毫秒級)]
                       1.6–7.8s              < 1ms                       611–3154ms
```

- **Cognition**：LLM 把自然語言映射到語義意圖。論文用 GPT-4o + one-shot prompt，平均延遲 1.6–7.8 秒。**這段是整條 pipeline 的瓶頸，貢獻 90%+ 的總時間。**
- **Dispatch**：一個確定性的 dispatch 函數 Γ 把語義意圖路由到對應的 behavior 模組。延遲 < 1 ms，可以視為常數。
- **Execution**：BT 節點啟動對應的 driver，輸出到 ROS2 topic。611–3154 ms（依任務複雜度而定）。

整體端到端成功率 **94%**（Cognition 93%、Dispatch 92%、Execution 95%）。

這個架構幾個值得在心裡圈起來的設計選擇：

1. **LLM 不直接 publish topic。** 它只生成「中間意圖（intent）」，由確定性的 dispatch 翻成具體的 BT 行為。這樣即使 LLM 講錯話、講不清楚，也不會直接污染下游 control loop。
2. **BT 節點本身回報 `success / failure / running`，等 LLM 想完整段時間，BT 都還在持續 tick。** 也就是說，機器人不會「靜止等 LLM」——它持續在執行上一個決策。
3. **失敗回報走另一條路**：當 BT 的某個 driver 回 `failure`，論文用 LLM 生成一個 `failure explanation`，再走回 cognition 重新規劃。也就是 **整套架構是雙向的：LLM 規劃 → BT 執行 → BT 失敗 → LLM 解釋並重規劃。**

這就是為什麼說 BT 是真正的中介層。它同時扮演三個角色：**規劃結果的緩衝、執行迴圈的引擎、失敗訊號的彙整點**。LLM 沒有 BT 是不能用的——因為它的時間域跟控制堆疊根本對不上。

---

## 四、失敗回復率比延遲更值得焦慮：70%–89.6% 是個壞數字

論文 paper 都會用「success rate」當頭條。但對任何要把這套架構推到產線或公眾場域的工程師而言，那個數字只是「沒人推你」的成功率。真正應該盯的是 **failure recovery rate**——也就是當第一次規劃失敗時，系統能不能自己爬起來。

2026 上半年的學術數字大概落在這個區間：

- 純 neuro-symbolic 失敗偵測 + 回復：~70% 整體回復率。簡單任務 59% 完成，複雜任務 33%。
- Recover-style 框架接到 pick-and-place：開啟 failure recovery 後成功率從 72% 提升到 80%。
- BT 動態調整（modular planner + failure interpreter）：論文宣稱在家用場景 89.6%。

這些數字解讀方式是：

> **就算端到端成功率到 94%，每 16 次互動還是會壞掉一次。** 而 16 次互動在一台真正在用的服務機器人身上是多久？大概一個下班時段。

更慘的是，89.6% 已經是 2026 上半年最好的數字之一了，而且是在預先設定的家用任務集上。一旦你把任務集擴大、引入未見過的物件、或加入時間限制（「30 秒內走到充電站」），這個數字會更難看。

這就是為什麼我覺得「LLM + ROS2」這條題目目前還是研究階段——不是它不 work，是它的可靠性還沒辦法跨過大家對機器人的容忍底線（一般工業應用要求 99.x%）。

而真正的工程貢獻空間，恰恰就在這個缺口裡：**你要怎麼設計 BT 的 fallback 子樹、怎麼定義「LLM 講不通就退回 hard-coded 行為」的判定條件、怎麼避免 LLM 把同一個失敗指令連續嘗試 5 次。** 這些都是純粹的 ROS2 工程問題，跟 prompt engineering 沒有半點關係。

---

## 五、為什麼形式化驗證還在門外

這裡值得特別點一下：上面提到的所有架構，**沒有一個提供形式化的安全保證**。

SNT-Arg 的論文很誠實地寫了：他們的「safety」是 runtime state validation——也就是執行前檢查電量、感測器狀態之類的條件，但沒有對 LLM 的輸出做任何 specification 級別的驗證。Boston Dynamics、JPL、NVIDIA 的開源框架也都是同一套路。

研究界有 ROBOGUARD 這類嘗試——用 world model 把安全規則編譯成 specification，再把 LLM 的規劃跟 specification 做 synthesis——但這條路目前還是論文階段。Temporal Property Specification Language（TPTL）可以表達「收到低電量訊號後 30 秒內必須開始往充電站走」這種即時屬性，理論上能形式化驗證 BT 的行為，但接到 LLM 之後，整個系統的可被驗證範圍就大幅縮水。

對一個寫安全相關（safety-critical）系統的工程師（例如車規、醫療、工廠線），這代表一個很現實的判斷：**今天把 LLM 放進控制環路，等於把這個環路從「可驗證」降級到「可監控」。** 這是個工程降級，不是升級。

所以這條路線目前的合理應用場域，仍是：**人在旁邊監督的服務型機器人、家用助手、demo / 教學用平台。** 而不是任何「LLM 出錯，後果嚴重」的場合。

---

## 六、反方意見：可能根本不該用 LLM 接 ROS2

技術圈現在這波熱潮裡，幾個 ROS2 老兵的反對意見其實值得放在版面上：

> 「如果你只是想讓機器人聽『去廚房』『關燈』這類指令，你需要的是一個輕量的 intent classifier，**不是一個 70B 參數的 LLM。** 一個 BERT-class 模型 + 一張意圖映射表就解決，延遲 50 ms，成本接近零。」

這個論點不是 strawman，它在 2026 的 ROS2 社群討論串裡反覆出現。LLM 確實在開放式對話、組合式任務（「先去廚房拿杯水，再去書房放在桌上」）上有壓倒性優勢，但 **如果你的任務本質是封閉集（closed-set）**，用 LLM 是過度殺雞用牛刀，且承擔了它所有的延遲與可靠性問題。

另一個更尖銳的觀點：

> 「Behavior tree + LLM 看起來架構漂亮，但它把責任分散到太多層——LLM、dispatch、BT、driver——每一層都有自己的失敗模式。當機器人卡住時，你完全不知道是誰的鍋。debug 體驗會比純 BT 差一個數量級。」

這也是真的。我自己跑過幾次 ROSA 的 demo，當它說「我不懂你的指令」時，到底是 LLM 沒理解、還是 LangChain 的 tool routing 走錯、還是 ROS2 service 那邊掛掉——觀測性（observability）幾乎為零。要把這套東西做到 production-ready，前提是要先解決全鏈路的追蹤與可解釋。

這兩個反對意見的合理結論是：**不是所有機器人都該接 LLM。但接得上 LLM 的機器人，會有非常清楚的差異化價值。** 關鍵是分得清楚自己的應用屬於哪一類。

---

## 七、給點雲 / LiDAR / ROS2 工程師的入手建議

回到最開始的問題：作為一個本來就在 ROS2 生態裡寫感知 / 控制的工程師，這波 LLM 浪潮你該怎麼接？

我的建議分三層：

### 7.1 短期（一個週末就能做）：把 LLM 接到既有 BT 的最上層節點

ROSA 開源 + Auromix 的 ROS-LLM 框架都能在週末跑起來。把它跟你手邊任何一個 Nav2 demo 接起來，**但不要相信 demo 影片的成功率**。實際跑 50 次同樣的自然語言指令，數一下：

- 多少次 LLM 解析錯誤？
- 多少次延遲超過 5 秒？
- 多少次需要人類介入？

這 50 次跑完之後你會對「為什麼 BT 是中介層」這件事有非常直覺的理解。

### 7.2 中期（一兩個月）：設計一個 fallback 子樹

在你的 BT 裡放一個 `fallback` 節點，子節點順序是：

```
fallback
├── llm_planner_branch     ← LLM 規劃出來的行為
├── learned_policy_branch  ← 既有的 learned / hard-coded 策略
└── safe_idle_branch       ← 啥都沒成功時的停機策略
```

這個結構本身就是一個 production-grade 的設計：LLM 講得通就照講的做、講不通就退到既有策略、什麼都失敗就乾脆停下來。重點是 **LLM 在這個結構裡是「加分項」，不是「必要路徑」**。

這是我覺得 ROS2 工程師面對 LLM 最健康的姿勢：**讓 LLM 增強既有系統，而不是替換既有系統。**

### 7.3 長期（半年到一年）：把感知層的可解釋訊號上推給 LLM

這才是真正有差異化價值的地方。

目前主流的「LLM + ROS2」架構，LLM 對機器人感知到的世界基本上是瞎的——它只能看到 dispatch 給它的「intent slot」，看不到原始點雲、看不到 occupancy grid、看不到 obstacle 的速度估計。

但如果你寫 LiDAR / 感知，你最熟的就是「怎麼把點雲變成有意義的 scene description」。把這層輸出餵給 LLM，能讓它的規劃從「去廚房」這種抽象意圖，升級到「往左 1.5 公尺有一張椅子，繞過它再走 2 公尺到廚房」這種考慮真實環境的決策。

這是 NVIDIA Isaac GR00T N1.6 把 Cosmos Reason 接進來在做的事，也是接下來這條題目最大的工程紅利所在——而且它對「會寫感知演算法」的人有極高門檻。換句話說，**你的點雲 / LiDAR 經驗在 LLM 時代不只沒貶值，反而是 LLM 規劃層的關鍵 enabling layer。**

---

## 結語：別追 LLM，學 BT

過去一年我看到太多 ROS2 工程師看到 LLM 浪潮的第一反應是「我是不是該去學 prompt engineering」。我不覺得。

LLM 的能力增長是 1-2 個月一個 generation，2026 還是同樣的速度。你今年學的 prompt 技巧，明年大概率被新模型自動內化。但 behavior tree、ROS2 控制堆疊、感知層的設計——這些東西在過去十年穩定演進，未來十年也會穩定演進。它們是這個架構裡 **不會被 LLM 取代的部分**，而且恰恰是接住 LLM 的那個結構。

那篇 14 個 LLM 的 benchmark 給我最大的提醒是：當 GPT-5 用 12 秒回答一個導航問題、而 Nav2 的 BT 在那 12 秒裡 tick 了 1,200 次，你會清楚地看見**真正在做事的層是哪一層**。

LLM 是上游決策的補強。BT、感知、控制——這些才是機器人真正的生命線。

去把你的 Nav2 BT 拆開看一遍。看懂它怎麼處理 `RUNNING` 狀態、怎麼處理 retry、怎麼處理 fallback。那才是你能在這波 LLM 浪潮裡站得最穩的地方。

---

## 參考來源

- [Latency-Aware Benchmarking of Large Language Models for Natural-Language Robot Navigation in ROS 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12846292/) — 14 個 LLM 在 Nav2 上的延遲 benchmark
- [Interpretable Robot Control via Structured Behavior Trees and Large Language Models (arXiv 2508.09621)](https://arxiv.org/html/2508.09621v2) — Nature Machine Intelligence 2026，三段式架構
- [ROSA: NASA JPL's open-source ROS Agent](https://github.com/nasa-jpl/rosa) — LangChain-based, ROS1+ROS2 通用
- [Auromix ROS-LLM framework](https://github.com/Auromix/ROS-LLM) — 十分鐘整合的開源框架
- [snt-arg/robot_suite](https://github.com/snt-arg/robot_suite) — Nature MI 2026 論文的開源實作
- [NVIDIA Isaac ROS / GR00T N1.6 with Cosmos Reason](https://nvidia-isaac-ros.github.io/) — 工業級整合的代表
- [Recover: A Neuro-Symbolic Framework for Failure Detection and Recovery (arXiv 2404.00756)](https://arxiv.org/html/2404.00756v1) — 失敗回復率的學術參考
- [Behavior tree generation and adaptation for social robot control with LLMs (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0921889025002623) — 89.6% 成功率的家用場景
- [BehaviorTree.CPP / nav2_behavior_tree](https://github.com/BehaviorTree/BehaviorTree.CPP) — ROS2 Nav2 的事實標準 BT 引擎
