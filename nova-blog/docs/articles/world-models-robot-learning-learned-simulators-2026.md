---
title: "當模擬器自己學會做夢：World Models 如何重寫機器人學習的資料賬本"
slug: world-models-robot-learning-learned-simulators-2026
description: "2026 年，World Model（世界模型）從一個學術名詞變成機器人學習的主線敘事——一篇 43 頁的綜述把散落各處的研究收攏成體系，ICLR / ICRA 上 Ctrl-World、RIGVid 這類「靠生成影片學動作」的工作接連出現。但 World Model 真正改寫的不是模型能力，而是資料賬本：它把『手刻模擬器跑 sim-to-real』換成『學一個會做夢的生成式模擬器』。這篇拆解 World Model 的三種用途、它和傳統 sim2real 的根本差異，以及對做感知與自駕的工程師意味著什麼。"
date: 2026-05-24
tags: [World Model, 機器人學習, Sim-to-Real, 生成式模型, 影片預測, 自駕, 強化學習, Physical AI]
category: AI & Robotics
---

## 前言：資料賬本，而不是模型能力

過去兩年，機器人學習的主旋律是「規模」——更大的模型、更多的遙操作資料、更通用的策略。EgoScale 那類工作甚至證明了機器人基礎模型也吃同一套 scaling law。但規模路線有一個藏不住的帳要算：**真實機器人資料貴到離譜。** 一條人類遙操作的軌跡，要有人、有機器、有時間，採集速度被物理世界死死卡住。

2026 年，一個替代答案越來越響亮：**World Model（世界模型）。** 五月初出現的一篇 43 頁綜述〈World Model for Robot Learning: A Comprehensive Survey〉把這個原本散落在影片生成、強化學習、自駕各角落的題目收攏成一套體系；同期 ICLR / ICRA 上，Ctrl-World（可控生成式世界模型）、RIGVid（不靠真機示範、純靠生成影片學操作）這類工作接連登場。

但我想先把話講清楚，免得被「又一個 AI 變強了」的敘事帶走：**World Model 真正改寫的，不是模型有多聰明，而是資料賬本怎麼記。** 它把「手刻一個物理模擬器、再想辦法跨過 sim-to-real gap」這條老路，換成「直接學一個會做夢的生成式模擬器」。這篇文章就拆這本新賬怎麼記。

---

## 一、World Model 到底是什麼

綜述給的定義很乾淨：World Model 是**「在動作條件下，對環境如何演化的預測性表徵」**（predictive representations of how environments evolve under actions）。

把它拆成工程語言就是一個函數：

```
未來狀態 (或觀測) ≈ f(目前狀態, 接下來的動作)
```

關鍵在「動作條件下」這四個字。一個普通的影片生成模型只會續寫畫面；World Model 必須回答的是**反事實問題**——「如果機器手臂往左推 5 公分，桌上的杯子會怎樣？」這個 action-conditioned 的能力，正是它能拿來做規劃、做模擬、做資料生成的根本原因。

這個想法本身不新。早在 Dreamer 系列（一路到 DreamerV3）就示範過：在一個學出來的 latent 動力學模型裡跑 RL，agent 可以「在想像中」演練，再把策略搬回真實環境。新的地方在於**載體換了**——從低維 latent，換成了高維、可看見的**生成式影片**。當模型能直接「畫出」未來幾秒的像素，World Model 就從一個 RL 內部組件，升級成了一個通用的視覺模擬器。

---

## 二、三種用途：想像、可控展開、資料放大

綜述把 World Model 在機器人學習裡的角色歸成幾類，我把它整理成工程上最好理解的三條線：

### 1. 想像式規劃（Imagination / Planning）
策略不在真實環境裡試錯，而是在 World Model 內部「展開」（rollout）多條未來，挑回報最高的那條去執行。好處是試錯成本趨近於零，壞處是——**模型錯了，策略就跟著錯到底。** 想像的品質就是規劃的天花板。

### 2. 可控展開（Controllable Rollout）
這是 2026 年最熱的一塊，Ctrl-World 是代表。重點不只是「能生成未來影片」，而是**生成過程可被精準操控**：給定不同動作序列，模型要產出時序連貫、空間正確、且彼此可區分的影片。一個不可控的生成器只是個花俏的續寫機；可控，它才配當模擬器。

### 3. 資料放大（Data Amplification）
這條最戳痛點。RIGVid 那類工作的邏輯是：**用生成的影片當示範，省掉真機資料採集。** 你有少量真實資料，World Model 把它「擴增」成海量多樣的合成軌跡，policy 在這堆合成資料上學。資料賬本的革命就發生在這——真機採集從「主要來源」降級成「校準錨點」。

這三條線其實是同一件事的不同切面：**World Model 是一台能按需生產經驗的機器。** 你要規劃，它給你未來；你要訓練，它給你資料；你要評估，它給你一個不會摔壞真機的試煉場。

---

## 三、和傳統 sim-to-real 的根本差異

做過機器人或自駕的人會問：這跟我們用 Isaac Sim、CARLA 跑 sim2real 有什麼不一樣？這是好問題，差異也正是 World Model 的命門。

| 維度 | 傳統模擬器（手刻） | World Model（學出來的） |
|------|------------------|------------------------|
| 物理規則 | 工程師寫死（剛體、接觸、摩擦） | 從資料隱式學出 |
| 視覺擬真 | 靠 rendering pipeline，常有 domain gap | 直接生成像素，視覺逼近真實 |
| 覆蓋範圍 | 只有你建模過的場景 | 受訓練資料分佈限制 |
| 失敗模式 | 物理近似誤差、渲染落差 | 幻覺、違反物理、長程漂移 |
| 擴充成本 | 寫新資產 / 新規則（人力） | 餵更多資料（算力） |

一句話總結：**傳統模擬器把 gap 開在「視覺擬真」上，World Model 把 gap 挪到了「物理正確性」上。**

手刻模擬器的物理是對的（它就是按物理方程算的），痛在畫面假、建模窮舉不完。World Model 反過來——畫面可以以假亂真，但它沒有任何硬性物理約束，可能生出杯子穿過桌面、物體憑空消失、或長序列裡慢慢漂移失真的「夢」。對一個追求底層原理的工程師，這個 trade-off 必須看清楚：**你不是消除了 sim-to-real gap，你是把它換了個位置。** 而視覺 gap 我們有一堆成熟工具對付，物理幻覺的檢測與約束，目前還是開放問題。

---

## 四、自駕：World Model 最成熟的試驗場

對 Adam 這種做感知 / 感測融合的人，最值得盯的不是桌上機器手臂，而是**自駕**——這是 World Model 落地最深的領域之一。

自駕的 World Model（公開代表如 Wayve 的 GAIA 系列、NVIDIA Cosmos）做的事很直接：吃進多相機影像 +（自車）動作，生成「如果這樣開，接下來路況會怎麼演化」的影片。它解的正是自駕資料的死穴——**長尾場景。** 鬼探頭、逆光、罕見事故前兆，真實世界裡可能幾萬公里才遇一次，但你可以叫 World Model「生」出成千上萬個變體來練感知與規劃。

這和站內〈[Sim-to-Real Gap：Cadence 與 NVIDIA 的兩條路](sim-to-real-gap-cadence-nvidia-2026.md)〉講的高保真模擬是互補而非取代的關係：手刻模擬器保證物理可信、World Model 保證視覺多樣與長尾覆蓋。成熟的 pipeline 多半兩者並用。而它和〈[On-Sensor Perception](on-sensor-perception-lidar-edge-2026.md)〉的連結則在於——當感知前移到感測器、延遲被壓到極致，World Model 生成的訓練資料是否涵蓋那些 on-sensor 才看得到的原始訊號分佈，會直接決定模型在 edge 上的泛化能力。

---

## 五、開放問題：這本賬還沒記完

綜述自己坦承「文獻仍在架構、功能角色、應用領域之間碎片化」。把學術話翻成工程清單，World Model 還沒解決的硬骨頭是：

1. **物理一致性無保證。** 生成的未來好看不等於正確。長程 rollout 的累積誤差、物理違反的自動檢測，仍缺乏可靠機制。把違反物理當成「分佈外」訊號來剔除，是值得做的方向。
2. **評估標準缺失。** 影片「逼真」用 FVD 之類指標還能量，但「對機器人學習有沒有用」沒有公認 benchmark。一個視覺漂亮卻動力學錯誤的模型，可能比一個畫面粗糙但物理對的模型更危險。
3. **動作可控性 vs 生成品質的拉扯。** 越要精準受控，生成往往越僵；越追求逼真多樣，可控性越鬆。Ctrl-World 這類工作就卡在這條張力線上。
4. **計算成本。** 高解析度影片生成本身很貴，要當「即時規劃的內部模擬器」用，延遲預算是另一座山——這點和 World Model 想服務的即時控制場景天然矛盾。

---

## 結語：把資料問題，變成算力問題

如果要我用一句話講 World Model 對 2026 年的意義：**它試圖把機器人學習的「資料採集問題」，轉譯成「算力與生成品質問題」。** 真機資料受制於物理世界的採集速度，這是硬約束；而算力與模型，是過去十年我們最擅長往上堆的東西。誰能把資料瓶頸搬到算力這條我們有把握的賽道上，誰就改寫了規模競賽的規則。

但別把它當銀彈。World Model 沒有消滅 sim-to-real gap，它把 gap 從「視覺」搬到了「物理正確性」——而後者目前更難檢測、更難約束。對做感知與即時系統的工程師，真正該追的不是「生成的影片多逼真」，而是兩個更冷的問題：**它生成的經驗，物理上可信嗎？延遲上跑得動嗎？** 這兩題答好了，做夢的模擬器才算真的醒著。

---

### 延伸閱讀（站內）
- [EgoScale：機器人基礎模型的 Scaling Law](egoscale-robotics-scaling-law-2026.md)
- [Physical AI 基礎模型：通用機器人的底座之爭](physical-ai-foundation-models-robotics-2026.md)
- [Sim-to-Real Gap：Cadence 與 NVIDIA 的兩條路](sim-to-real-gap-cadence-nvidia-2026.md)
- [On-Sensor Perception：把感知推進感測器](on-sensor-perception-lidar-edge-2026.md)

### 參考來源
- World Model for Robot Learning: A Comprehensive Survey — arXiv:2605.00080（2026）
- Ctrl-World: A Controllable Generative World Model for Robots — ICLR 2026
- RIGVid: Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations — ICLR 2026
- Foundational World Models Accurately Detect Bimanual Manipulator Failures — ICRA 2026
- 背景：DreamerV3、NVIDIA Cosmos、Wayve GAIA 系列、Meta V-JEPA 2（公開資料）
