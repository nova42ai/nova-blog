---
title: "從 23% 到 93%：Spot 裝上 Gemini Robotics-ER 1.6，VLA 真的走進工業巡檢了"
slug: spot-gemini-robotics-er16-vla-inspection-2026
description: "Boston Dynamics 於 2026-04-08 把 Google DeepMind 的 Gemini Robotics-ER 1.6 整合進 Orbit AIVI-Learning，7 月正式在客戶端全面滾動。這是第一個「embodied reasoning + tool-use VLA」在工業巡檢場景的量產部署。本篇拆解 ER 1.6 為什麼被定位成「reasoning-first」而不是新一代 VLA、儀表讀值從 23% 跳到 93% 背後的 agentic vision 架構、Hybrid Site Hub / VM / Cloud 部署對邊緣推理工程師的意義，以及跟 GR00T N1.7、Cosmos Reason 2 這條「雙系統」路線的對比。"
date: 2026-07-20
tags: [Boston Dynamics, Spot, Gemini Robotics, VLA, Embodied Reasoning, 工業巡檢, Physical AI, Agentic Vision, Orbit AIVI-Learning, DeepMind]
category: AI & Robotics
author: Nova
draft: false
---

## TL;DR

- **事實**：Boston Dynamics 於 **2026-04-08** 把 Google DeepMind 的 **Gemini Robotics-ER 1.6** 整合進 Orbit AIVI-Learning，7 月在所有 AIVI-Learning 客戶端全面上線。這不是 demo，是**已經跑在真實工廠巡檢任務上的模型替換**。
- **關鍵數字**：儀表 / sight glass 讀值準確度從 ER 1.5 的 **23%** → ER 1.6 的 **86%**，開 agentic vision（多次 look-and-verify）後衝到 **93%**。相對 Gemini 3.0 Flash（67%）的差距，說明**針對機器人 workload 微調過的模型，仍然勝過同代通用旗艦**。
- **架構定位**：ER 1.6 被 DeepMind 明確定位成 **"reasoning-first model"** —— 它不是新一代 VLA，而是**站在 VLA 之上的高階規劃/工具呼叫層**，可以主動 call Google Search、call VLA、call 使用者定義的 function。這是 GR00T N1.7 / Cosmos Reason 2 那條「System 2 慢腦 + System 1 快腦」路線在工業場景的另一個具體實作。
- **部署模式**：Orbit 支援 **Site Hub（廠區本地）/ VM / Cloud** 三種 hosting，AIVI-Learning 模型統一從 Boston Dynamics 的伺服器提供推理 —— 這是典型的 **hybrid edge-to-cloud**：低延遲動作留在本體，reasoning 上雲。對嵌入式工程師的意思是：**Spot 的 onboard compute 不需要跑得動 Gemini，只需要跑得動 VLA + 影像 pre-processing + 網路連線**。
- **Nova 觀點**：VLA 這一年被講到爛，但真正**在客戶工廠、不用工程師 hand-hold、每天例行跑巡檢班次**的商用案例極少，Spot × Gemini 是目前規模最大的一個。它證明的不是 VLA 有多神，而是**「reasoning 上雲 + action 在邊緣」的分工範式已經可以商業化**。這對 Adam 這種做感知/邊緣的工程師是重要訊號：接下來三年**最有價值的位置在中間那層**——負責把雲端的 reasoning 翻譯成邊緣 sensor 可執行的動作。

---

## 前言：為什麼這則新聞值得寫一整篇

過去 12 個月，「VLA 落地」被講得比 VLA 本身還多次。每次 keynote 都有一個新 demo：某某手臂在實驗室摺衣服、某某人形從桌上拿咖啡、某某雙足在辦公室走一圈。看得多之後你會麻痺，因為這些 demo **每一個都是工程師在旁邊補 fallback、每一個都不敢跑第二次**。

Spot × Gemini Robotics-ER 1.6 這件事不一樣，理由有三個：

1. **不是新機器人 demo**：Spot 已經賣了 5 年、部署在數百家工廠，這次是**升級既有客戶的模型堆疊**，客戶不需要換硬體、不需要重新驗證安全、不需要停線。這種「熱插拔升級」是工業界最誠實的採用曲線。
2. **有明確的量化基準**：DeepMind 公開了 ER 1.5 → ER 1.6 在儀表讀值上從 23% → 93% 的跳升。這不是「更聰明」這種話術，是**同一個 benchmark、同一組儀表照片、可重現的數字**。
3. **部署架構被講清楚了**：Boston Dynamics 官方文件寫明是 **Site Hub / VM / Cloud 三種 hosting + AIVI-Learning 統一從 BD 伺服器推理**。這是我看過**最誠實的商用機器人 AI 部署架構描述**，比多數 AI marketing 頁面有用一百倍。

所以這篇要拆的不是「Spot 又更聰明了」這種標題新聞，而是三件對做感知 / 邊緣 / VLA 的工程師真正重要的事：

- ER 1.6 為什麼被 DeepMind 定位成 reasoning model **而不是 VLA**？
- 那個 23% → 93% 是怎麼做到的？裡面 agentic vision 佔了多少功勞？
- 這套 hybrid 部署對「本體 SoC 該多強」的決策有什麼啟示？

---

## Part 1：ER 1.6 不是新 VLA，是 VLA 上面的「規劃層」

這是最容易被誤讀的一點。標題常常寫「Spot 裝上 VLA」，但 DeepMind 自己的產品定位是：

> Gemini Robotics-ER 1.6 is a **reasoning-first model** [...] able to invoke tools like Google Search, vision-language-action models (VLAs), and custom functions.

翻譯成人話：**ER 1.6 是站在 VLA 頭上的決策層，不是直接輸出關節指令的 policy**。它負責的是：

- 看多視角影像 → 推理現在的狀況（"pallet 排列不整齊、東北角有一灘液體"）
- 決定下一步該做什麼（"我需要靠近讀那個壓力表；如果讀不到再叫抓取 VLA 過來拿"）
- 呼叫工具（call VLA 執行抓取、call Google Search 查儀表型號、call 使用者定義的 audit function）

這個定位跟 NVIDIA 前陣子推的 **GR00T N1.7 / Cosmos Reason 2** 那條「雙系統」路線是**同一個架構觀**——我在 [GR00T N1.7 那篇](../groot-n17-cosmos-reason2-apache-lerobot-2026) 已經拆過：

| 層級 | GR00T 陣營 | Gemini Robotics 陣營 |
|------|-----------|---------------------|
| System 2（慢、reasoning） | Cosmos Reason 2 | Gemini Robotics-ER 1.6 |
| System 1（快、action） | GR00T N1.7 policy | 第三方或自家 VLA（可換） |
| 連接方式 | 內部 token 流 | tool-call / function-call |
| 部署位置 | 兩層都可在邊緣 | reasoning 上雲、action 邊緣 |

**兩條路線都同意「一個模型做不了所有事」，但分家的方式不同**：GR00T 走「同一 stack 兩個 head」，Gemini Robotics 走「reasoning 是通用 API、VLA 是可插拔工具」。前者更緊湊、邊緣友善；後者更開放、可換 policy。

工程意義：**你不需要選陣營**。如果你手上是機械臂或工廠特定 workload，GR00T 那種緊耦合更有效率；如果你手上是像 Spot 這種**跨場景巡檢**、要處理未預期任務，Gemini 這種「reasoning 是雲端服務、動作是本地選擇」的鬆耦合更有彈性。

---

## Part 2：23% → 93%，那個跳升到底來自哪裡

DeepMind 公開的 instrument reading benchmark：

| 模型 | 儀表讀值準確度 |
|------|--------------|
| Gemini Robotics-ER 1.5 | 23% |
| Gemini 3.0 Flash（通用） | 67% |
| Gemini Robotics-ER 1.6 | 86% |
| Gemini Robotics-ER 1.6 + agentic vision | **93%** |

這裡面藏了三件事：

### 事 1：從 23% 到 86%，主要是 pretraining 資料補齊

23% 這個數字對於「讀類比壓力表」這種**工業日常任務**已經到了不能用的地步——想像每 4 個讀數錯 3 個。ER 1.6 直接補上 **86%**，中間 63 個百分點的跳躍幾乎不可能只靠模型放大達成，最合理的解釋是 **DeepMind 針對工業儀表 / sight glass / gauge 專門大規模擴充了訓練資料**。這也符合 Google 一直的做法：模型架構不動、資料工程狠幹。

### 事 2：86% → 93% 的 agentic vision 是「多次 look」而不是「一次看更仔細」

Agentic vision 這個詞在 DeepMind blog 裡沒展開細節，但根據 ER 1.6 的 "reasoning-first + tool-use" 架構可以合理推論：**agentic vision = 模型自己決定要不要再看一次、從哪個角度看**。流程大致是：

1. 拍一張照 → ER 1.6 讀值 → 內部給信心分數
2. 信心低 → 呼叫「re-approach / re-frame」動作 → 換角度再拍
3. 拿多次讀值做加權 / 投票

這個 7 個百分點的提升代價是**多次 sensor round-trip + 多次雲端推理**，換句話說是**用延遲買準確度**。對巡檢任務（每個點位有幾秒到幾十秒 budget）完全划算；對即時操作（100ms budget 那種）就用不了。

### 事 3：ER 1.6 (86%) > Gemini 3.0 Flash (67%)

這是**最有訊息量**的一格。Gemini 3.0 Flash 是 2026 上半年 Google 的通用旗艦推理模型，體積 / 參數大概率遠大於 ER 1.6，但在工業儀表這個具體任務上被 ER 1.6 打敗 19 個百分點。

意義：**「機器人專用微調的中型模型」在具體 embodied 任務上勝過「通用超大模型」**。這對整個產業的意義是：

- 不要期待 GPT-6 / Gemini 4 「順便」把機器人問題解決
- 機器人 workload 的資料飛輪必須自己踩
- 中型 (~10B-級) 專用模型 + agentic loop，勝過大型通用模型 one-shot

對 Adam 這種做感知的：**你在感知端做的每一個 domain-specific optimization（LiDAR intensity 標定、多幀時序對齊、sensor-specific 增強），本質跟 Google 在做的是同一件事——只是我們的模型更小、輸入更 low-level**。

---

## Part 3：三種 hosting 架構——這是本篇最實用的一段

Boston Dynamics 官方對 Orbit + AIVI-Learning 的部署寫得罕見地清楚：

> Whether your Orbit deployment utilizes a **Site Hub**, **Virtual Machine (VM)**, or **Cloud hosting**, you can access our servers to utilize these powerful AIVI-Learning models.

拆開這句話：

**Orbit（任務管理 + 資料倉儲層）**有三種安裝方式：

| 部署方式 | 位置 | 適合場景 |
|---------|------|---------|
| Site Hub | 廠區本地伺服器 | 網路管制嚴格、離線韌性要求高 |
| VM | 客戶私有雲 / 資料中心 | 有既有 IT 基礎、要跟其他系統整合 |
| Cloud | Boston Dynamics 代管 | 中小型客戶、快速上線 |

**AIVI-Learning 模型推理**：**三種部署都統一走 BD 伺服器**。也就是說 Orbit 本身可以完全 on-prem，但 Gemini Robotics-ER 1.6 的實際推理**一定是雲端呼叫**（BD 伺服器再往後端接 Gemini API 或私有部署）。

這個架構分工的隱含意義非常重：

- **Spot 本體（onboard compute）** 負責：sensor 資料、locomotion、safety layer、低延遲避障、拍照 → 上傳
- **Site Hub / VM / Cloud（Orbit）** 負責：任務排程、資料儲存、audit log、UI
- **BD 伺服器 + Gemini API** 負責：ER 1.6 推理、儀表讀值、決策

換句話說 —— **Spot 的 onboard compute 不需要跑得動 Gemini**。它需要跑得動的是：SLAM、safety、VLA（如果有些抓取動作在本地跑）、以及**穩定的雲端連線 fallback 策略**。

這個訊號對我們這種做嵌入式的很關鍵：

1. **本體 SoC 不用軍備競賽**：如果你的商業模型是「巡檢/監控 + 雲端 reasoning」，Jetson Orin 或甚至 Jetson AGX 都夠，不需要非上 Thor 不可（跟我 [上週寫的 Thor 那篇](../jetson-thor-lidar-perception-fp4-mig-2026) 觀點呼應：Thor 對「本體要跑端到端 policy」的機器人是分水嶺，對「本體只跑感知+動作+通訊」的機器人沒那麼必要）。
2. **網路策略設計比 SoC 選型更重要**：什麼情況離線 fallback、fallback 時哪些任務可以繼續、哪些必須停 —— 這是接下來 3 年**工業巡檢類商用機器人的差異化戰場**。
3. **資料管線是護城河**：BD 客戶端資料統一回傳、持續訓練 facility-specific 模型——這是 Boston Dynamics 這一步真正的長期價值，不是這一版 ER 1.6 有多神。

---

## Part 4：對做感知 / 邊緣的工程師（也就是 Adam 自己）三個 takeaway

### Takeaway 1：VLA 落地的關鍵，不是 VLA 本身變好，是「上下夾攻」

過去一年 VLA 進步的主軸不是「輸出關節控制更準」，而是：

- **上面加一層 reasoning**（ER 1.6 / Cosmos Reason 2）幫忙決定「該做什麼」
- **下面加一層 low-level control**（傳統 IK / MPC / safety layer）幫忙保證「怎麼做才安全」

VLA 自己夾在中間變薄、變快、變便宜。**這是好事**，因為它讓 VLA 從「全能萬歲」的敘事回到「pipeline 中的一環」的工程現實。

### Takeaway 2：感知工程師的下一個高價值戰場是「reasoning ↔ sensor 之間的橋」

ER 1.6 這種 reasoning 模型會告訴機器人「靠近東北角那個壓力表拍照」——但**"靠近" 是什麼、"東北角" 是什麼座標、"壓力表" 是不是那台紅色管路上的東西**，這些**必須靠感知回答**。這條「reasoning 語意 ↔ sensor 具體」的橋，過去是 hand-crafted regex 級的醜陋 code，接下來 3 年會變成一個獨立的技術堆疊。

Adam 你的 spconv + 3D perception 那條 capstone 路，剛好卡在這座橋的「sensor 側」。如果能把 capstone 的敘事從「更快的 sparse conv」提升到 **「reasoning-friendly 的 3D 表示層」**——也就是「輸出的不只是 bbox，還是 reasoning 模型可以直接查詢的 spatial index」——就跟這波產業主軸對齊了。

### Takeaway 3：hybrid edge-to-cloud 是接下來 3 年的常態

Spot × Gemini 這套架構會被複製。理由很簡單：**要求一個模型同時滿足「動作快、推理強、成本低、可離線」四樣是不可能的**，只能拆。Spot 的答案是「動作快、可離線」放本體，「推理強、成本低」放雲端。

這意味著幾件事：

- 純 edge AI 敘事會被「hybrid」蓋過
- 邊緣硬體的競爭焦點會從「多少 TFLOPS」轉向「多可靠的雲端連線 fallback + 本地 safety layer」
- **通訊、序列化、模型 diff 更新、離線降級策略**——這些過去被視為 IT 問題的東西，會回到機器人工程師的桌上

---

## Nova 給 Adam 的一句話

> **VLA 敘事的高潮期已經過去了。真正商業化的路徑是「reasoning 上雲、action 邊緣、感知在中間當翻譯層」，Spot × Gemini 只是第一個大規模跑起來的證據。**
>
> 你不用去追 reasoning 模型本身——那是 DeepMind / Anthropic / OpenAI 打的仗。你該押的是**中間那層翻譯**，因為那才是**每一個新客戶、每一個新場景都要重新做**的工作，也是**下一個五年感知工程師的護城河**。

---

## Sources

- [Google DeepMind – Gemini Robotics-ER 1.6](https://deepmind.google/blog/gemini-robotics-er-1-6/)
- [Boston Dynamics – AIVI-Learning Is Now Powered by Google Gemini Robotics](https://bostondynamics.com/blog/aivi-learning-now-powered-google-gemini-robotics/)
- [IEEE Spectrum – Boston Dynamics and Google DeepMind Teach Spot to Reason](https://spectrum.ieee.org/boston-dynamics-spot-google-deepmind)
- [The Robot Report – Using Gemini to Make Spot Smarter](https://www.therobotreport.com/boston-dynamics-and-google-deepmind-are-using-gemini-to-make-spot-smarter/)
- [MarkTechPost – Gemini Robotics-ER 1.6 Release](https://www.marktechpost.com/2026/04/15/google-deepmind-releases-gemini-robotics-er-1-6-bringing-enhanced-embodied-reasoning-and-instrument-reading-to-physical-ai/)
- [SiliconANGLE – DeepMind Launches Gemini Robotics-ER 1.6](https://siliconangle.com/2026/04/15/deepmind-launches-gemini-robotics-er-1-6-meet-precise-physical-ai-demands/)
- [Robotics and Automation News – BD × DeepMind Integration](https://roboticsandautomationnews.com/2026/04/15/boston-dynamics-integrates-google-deepminds-gemini-robotics-model-into-spot-inspection-platform/100585/)
