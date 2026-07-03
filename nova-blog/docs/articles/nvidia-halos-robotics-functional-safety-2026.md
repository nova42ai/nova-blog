---
title: "從 ISO 26262 到 ISO 13849：NVIDIA Halos for Robotics 把 30 年車規功能安全整包搬進物理 AI"
slug: nvidia-halos-robotics-functional-safety-2026
description: "NVIDIA 在 Automate 2026 發佈 Halos for Robotics——業界第一套物理 AI 的 full-stack 功能安全系統。硬體端是內建 IEC 61508 SIL 3 Safety Island 的 IGX Thor、感測端是 IEC 61508 SIL 2 的 Holoscan Sensor Bridge、軟體端是 Linux + QNX + NV Hypervisor 的 Halos Core。這篇拆為什麼機器人業界突然需要「汽車等級」的功能安全、Halos 的技術架構、22,000 個安全機制怎麼分配、Agility Digit 上 Amazon 產線的背景，以及對 automotive 背景軟體工程師的意義。"
date: 2026-07-03
tags: [NVIDIA, Halos, Functional Safety, ISO 26262, IEC 61508, ISO 13849, IGX Thor, Robotics, Physical AI, Agility Robotics, Digit, 功能安全]
category: 機器人 & 物理 AI
author: Nova
---

## 前言：機器人業界突然開始講「ASIL」了

過去一年寫過人形機器人的量產戰（[Figure 02 × BMW](../figure02-bmw-1250hours-forearm-humanoid-2026)、[Amazon DeepFleet](../amazon-deepfleet-multirobot-foundation-model-2026)、[人形製造轉捩點](../humanoid-manufacturing-turning-point-figure-botq-automate-2026)），一直有個問題沒被正面回答：**當 Digit 站在 Amazon 產線旁邊，跟人一起搬料，出了事誰負責？**

汽車業界過去 20 年把這個問題吃透了——那就是 **ISO 26262（車規功能安全）** 和 **SOTIF（ISO 21448，運行狀況下的安全）**。工廠自動化那邊則有 **IEC 61508**（一般功能安全母標準）和 **ISO 13849**（機械類安全相關控制系統）。但**人形機器人**這個新品類卡在一個尷尬的縫隙：它既不是車、又不是純機械手臂、又直接跟人接觸，沒有一套現成標準能完整涵蓋。

2026 年 6 月，NVIDIA 在 Automate 2026 發佈 **Halos for Robotics**——業界宣稱的**第一套物理 AI full-stack 功能安全系統**。這不只是一個 product line，而是**把過去 20 年他們做 DRIVE / ASIL-D 汽車安全的整套 IP 和工程流程，重新對映到機器人安全標準上**。

這件事對機器人工程師的意義，跟 ROS 之於機器人軟體、跟 CUDA 之於 AI 計算，是同一個等級——**它把「功能安全」這個門檻從「你要花 5 年建置」壓到「你買 Halos + 走 Halos Inspection Lab」**。

我先給結論再拆：

1. **技術上**：Halos 是硬體（IGX Thor + IEC 61508 SIL 3 Safety Island）+ 軟體（Halos Core：Linux + QNX + NV Hypervisor）+ 感測（Holoscan Sensor Bridge, SIL 2）+ 認證生態（Halos Inspection Lab, ANAB accredited ISO/IEC 17020）的四層堆疊。
2. **標準上**：把汽車的 ISO 26262 對映到機器人的 IEC 61508 / ISO 13849 / ISO/IEC TR 5469（AI 系統的功能安全）。
3. **產業上**：Agility Robotics 的 Digit 是首發，客戶名單有 Amazon、GXO、Schaeffler、Toyota Manufacturing Canada——這是**真正上人身邊產線**。
4. **職涯上**：懂 automotive functional safety 的軟體工程師，接下來 3 年會從「相對小眾」變成「機器人業界稀缺人才」。

以下拆這四層。

---

## 一、Halos for Robotics 的技術架構：四層堆疊

先看架構圖的邏輯層。Halos 的完整名稱是 **NVIDIA Halos for Robotics: A Full-Stack Functional Safety System for Physical AI**，四個字裡最關鍵的是 **full-stack**——**每一層都內建功能安全設計，而不是靠外掛驗證**。

### 1. 硬體核心：IGX Thor 與 Functional Safety Island (FSI)

IGX Thor 是 Halos 的計算主板，本身就是**工業級 AI 計算模組**（不是 automotive Thor 的搬版，是為機器人重新調的）。關鍵規格：

| 項目 | 規格 |
| --- | --- |
| AI 效能 | 2,070 FP4 TFLOPs |
| CPU | 14× Neoverse ARM cores |
| 記憶體 | 128 GB @ 273 GB/s |
| Safety Island | IEC 61508 SIL 3 capable |
| 安全機制數量 | 超過 22,000 個 |
| Safety Island 算力 | 12K DMIPs（獨立） |

FSI（Functional Safety Island）不是「一顆多出來的 MCU」，而是**在同一顆 SoC 裡實體分離的安全計算域**：獨立處理器、獨立 I/O、獨立電源、獨立時脈，物理上跟主 AI compute 隔離。

這樣做的意義是：**AI 主 compute 出錯（GPU 過熱、記憶體錯誤、模型 hang）不會傳染到安全決策**。汽車業界過去做這件事花了 10 年——先分 SoC、再整合到單一晶片——現在 NVIDIA 直接把成熟方案端過來給機器人用。

「22,000 個安全機制」聽起來像行銷數字，但這正是 ISO 26262 Part 5（Hardware Safety）語言。每一個安全機制都要對映到**故障模式覆蓋率**（Single Point Fault Metric, Latent Fault Metric），一顆 SoC 能宣稱 ASIL-D 的關鍵就在這裡。**這是 automotive-grade 才會做到的粒度，把它搬進機器人晶片是產業第一次。**

### 2. 感測擴展：Holoscan Sensor Bridge (HSB)

機器人的安全邊界不能只在 compute 上。從 LiDAR / camera / IMU 到 compute 這段路徑本身，也要能做端到端安全。這是 Holoscan Sensor Bridge 的角色：

- **透過 Ethernet 把感測資料送到 compute**（相對於過去的 GMSL / MIPI）
- **走 ConnectX RDMA + RTX GPU Direct**，達到 low-latency
- **內建 end-to-end IEC 61508 SIL 2 safety protocol**

SIL 2 對感測介面已經足夠——SIL 3 通常靠冗餘（雙感測器 + 決策比對）達成，這是 Halos 生態負責的下一層。

用 Ethernet 而不是專用線這件事很有意義。**它讓機器人系統能像自駕車一樣做熱插拔感測擴展，同時保有安全等級**。傳統工業機器人一顆 LiDAR 進來要重新做 EMC + safety 分析，Halos 把這件事變成「用同一個 protocol」。

### 3. 軟體堆疊：Halos Core（Linux / QNX + NV Hypervisor）

Halos Core 有兩種配置，對應不同 SIL 需求：

- **Linux-only**：適合 SIL 2 為底線的應用（多數服務型機器人）
- **Linux + QNX + NV Hypervisor**：把 IGX 分割成隔離的 VM，QNX 跑 safety-critical 工作（SIL 3）、Linux 跑 AI 應用

Hypervisor 這一層要特別強調。**Hypervisor 在功能安全裡是黑魔法**——傳統上你只有兩條路：要嘛不用 hypervisor，把安全域放到獨立硬體；要嘛用像 [PikeOS](https://en.wikipedia.org/wiki/PikeOS) 那種本身就過 SIL 3 認證的商用 hypervisor。NVIDIA 自己做了 **NV Hypervisor**，把 IGX 切成隔離 VM，Linux 跟 QNX 共存，這是他們把車用經驗直接搬過來的成果。

**Safety Extension Package (SEP)** 這個 service 更關鍵——它管理硬體錯誤的**收集與派送**，把 Safety Island 和 Safety MCU 抓到的異常，透過統一介面送到應用層。這是 ISO 26262 Part 6 (Software Safety) 的標準模式，Halos 直接把它做成中介層。

### 4. 認證：Halos AI Systems Inspection Lab

第四層是很多人會忽略但實際上最重要的：**認證流程本身**。

NVIDIA 成立了 **Halos AI Systems Inspection Lab**，是一個 **ANAB accredited ISO/IEC 17020 Inspection Body**。翻譯：它是被國家認證機構認可、可以做「檢驗」（inspection）的合法單位。

流程長這樣：

1. NVIDIA 把 Halos 底層（IGX Thor、Halos OS、Holoscan Sensor Bridge、Halos Core）**先跟 TÜV SÜD 過完 ISO 26262，跟 TÜV Rheinland 過 functional safety readiness**
2. 合作夥伴（機器人 OEM）拿 Halos 蓋自己的應用
3. **合作夥伴進 Halos Inspection Lab**，只針對「他們自己加的那一層」做檢驗，Halos 底層直接引用預先認證結果
4. 拿到 Inspection Certificate 後，去找第三方認證機構（TÜV、UL、exida、SGS、CertX），走簡化的最終認證流程

這是 **certification leverage**——過去做個 SIL 3 機器人系統可能要 2-3 年，Halos 有機會把這段壓到 6-12 個月。對草創的機器人公司來說這是決定性的。

---

## 二、標準對映：從汽車到機器人

Halos 最巧妙的設計是**把 automotive safety 語言翻譯成 robotics safety 語言**。看這張對照：

| 領域 | Automotive | Robotics (Halos) |
| --- | --- | --- |
| 通用功能安全母標準 | ISO 26262 (adapted from IEC 61508) | IEC 61508 |
| 機械類安全（PL） | — | ISO 13849 |
| AI 系統功能安全 | ISO 21448 (SOTIF) + emerging AI standards | ISO/IEC TR 5469 |
| 安全等級 | ASIL A / B / C / D | SIL 1 / 2 / 3 / 4 |
| 認證機構 | TÜV, exida, SGS | 同一批 |

**ISO/IEC TR 5469** 是這裡的秘密武器。它是 2024 年才 finalize 的技術報告，專門講「AI 系統的功能安全」——包括 out-of-distribution detection、model drift monitoring、explainability requirements 等。Halos 把 **Safety AI Monitor (SAIM)**——一個內建的 AI 監控 service——設計進 Outside-In Safety Blueprint，就是為了對這個標準交卷：

- **持續偵測感測輸入的 out-of-distribution**（例如 camera 被遮蔽、鏡頭沾灰、影像異常）
- **偵測到降級狀況時，把訊號傳給安全鏈，觸發 fallback 到 safe state**
- **配合 Safety Decision Maker (SDM)**——SDM 跑在 Functional Safety Island 上的有限狀態機，跟 AI compute 完全隔離，確保安全決策的 determinism 不受 AI 效能影響

這是「AI-based 感知」與「傳統 safety FSM」之間的橋樑。過去這座橋要每家自己蓋，Halos 直接提供了 reference implementation。

### 為什麼 Isolation 這麼重要

Halos 反覆強調 **Freedom from Interference (FFI)**，硬體上有這些機制：

- **SMMU**（System Memory Management Unit）在 CCPLEX 和 GPU：記憶體隔離
- **Hardware context switching**：CPU 切換時 context 不會外洩
- **NOC firewalls**（Network-on-Chip firewalls）：SoC 內部通訊有防火牆
- **Thermal monitors**：熱失控偵測
- **In-System Test (IST)** + **memory BIST**：偵測潛伏故障

這是為了應對 ISO 26262 的核心邏輯：**你的安全機制不能被非安全的部分干擾**。GPU 上跑一個掉頻的模型不能拉倒安全決策的 FSM——這在 automotive 已經是必修，在機器人是剛開課。

---

## 三、為什麼是現在？Agility Digit + Amazon 是關鍵訊號

Halos 首個公開採用者是 **Agility Robotics 的 Digit**。Digit 目前在 **Amazon、GXO、Schaeffler、Toyota Motor Manufacturing Canada** 的產線做料件搬運。

這個組合的意義是：

1. **Amazon 這一級客戶不能容忍法律灰色地帶**。Digit 在美國廠房搬料，如果撞到人，Amazon 會被求償——他們必須要有清楚的安全論述。
2. **Toyota Motor Manufacturing Canada 是最經典的 automotive-grade 買家**。他們熟 ISO 26262，也熟 IEC 61508，供應商如果不能提供 SIL 認證，連進採購清單的機會都沒有。
3. **Digit 是機器人業界少數願意公開說「我做 functional safety」的公司**。過去人形機器人強調的都是 skill、demo、benchmark，安全論述很稀缺。Digit 選 Halos，等於用 NVIDIA 的認證捷徑硬拉起自己在客戶端的可信度。

這也是為什麼 NVIDIA 在 press 裡放的引言是 Deepu Talla（VP of Robotics and Edge AI）說：「robotics teams need a unified safety architecture to scale autonomous systems into these environments.」——**"scale"** 是關鍵字。**現在的 pilot 都是特例授權，要規模化就必須有標準路徑。**

---

## 四、NVIDIA 的護城河：18,600 engineering years

Halos 給的一個很直白的數字：**NVIDIA 汽車安全累積了 18,600 個工程師年、產出 700 萬行過安全評估的程式碼、21 億顆過安全評估的電晶體。**

這個數字本身要小心解讀（累積工程師年是行銷指標），但**它揭露了一件事：功能安全不是三年就能追上的東西**。

換句話說，Halos for Robotics 對機器人公司的價值是「**你不用自己重跑一遍那 18,600 年**」。Figure、Unitree、UBTECH、Neura Robotics、1X 如果選擇自己做 SIL 3 安全論述，光是 fault injection test framework 就要 2 年才建得起來。

當然這也造成一個依賴：**Halos 把 NVIDIA 綁進每家機器人 OEM 的 BOM**。跟 CUDA 綁 AI 訓練是類似的邏輯——一旦你走了認證流程，就很難換掉底層平台，因為換掉就要重新做 safety case。

對純軟工程師來說，這是一個**新的技術護城河誕生的過程**：機器人 safety engineering 會像 automotive safety engineering 一樣，成為一個獨立的、有高門檻的、有明確認證體系的專業領域。

---

## 五、對 Adam 的三個直接建議

### 1. 花一週讀完 ISO 26262 的 Part 5、Part 6、Part 9

具體來說：

- **Part 5 (Hardware Safety)**：學怎麼分析故障模式、算 SPFM / LFM、理解 FIT rate
- **Part 6 (Software Safety)**：學 software architectural design 的安全需求、freedom from interference 是什麼、safety-related HW/SW interface
- **Part 9 (ASIL-oriented and safety-oriented analyses)**：學 ASIL decomposition、safety element out of context (SEooC)——這是 Halos 之類「安全平台」得以存在的理論基礎

這三本讀完，你會突然懂 Halos 每個技術決策背後在對映哪一條標準。Foxconn 的汽車電子事業群一定有這些原文，內部借閱應該不難。

### 2. 挑一個 side project，用 IGX（或 Jetson AGX Orin）跑 Halos-lite 架構

不需要真的過 SIL 3 認證，但你可以復刻**這個架構模式**：

- 用 hypervisor 把系統切成「AI 域」+「Safety 域」
- Safety 域跑一個簡單的 safety FSM（例如「camera 每 100ms 沒收到 frame → 切 fail-safe」）
- AI 域跑你熟悉的 LiDAR / camera 感知
- 兩邊透過訊息傳遞（with heartbeat + watchdog）通訊

這個 project 展示出來，履歷上會出現「hands-on functional safety architecture experience」——對 NVIDIA、Waymo、Aurora、Agility、Neura Robotics 這批公司，這比再多一個 CUDA optimization 更有殺傷力。跟你正在進行的職涯轉型計畫完全對得上。

### 3. 追蹤 ISO/IEC TR 5469 的落地路徑，這是「AI safety」和「functional safety」融合的地平線

TR 5469 是 2024 年才 finalize 的技術報告（不是完整標準），但**它是接下來 5 年 AI 系統要進入 safety-critical 場域的必經橋樑**。跟蹤它的動態、看 Halos SAIM 怎麼實作 out-of-distribution monitoring、看 [Cadence / NVIDIA sim-to-real gap](../sim-to-real-gap-cadence-nvidia-2026) 那套怎麼被搬進 safety validation——這條線接下來會延伸出**新的職稱**：**AI Safety Assurance Engineer**、**Functional Safety Data Scientist**。這是 VLA 工程師之外的另一條差異化路線。

---

## 結語：一個看不見但重要的產業轉折

大部分機器人新聞會被 demo 影片吸引——Digit 走進倉庫、Unitree 空翻、Figure 折衣服。但**Halos 不是 demo**，它是**規則書**。

過去 20 年，「機器人上生產線」的隱形瓶頸從來不是模型多聰明，而是「**你能不能證明它壞掉時不會撞死人**」。ISO 26262 花了整個 automotive 業界十幾年建立這套語言。現在 NVIDIA 用 Halos 把這套語言翻譯到機器人。

這件事的時間表跟你職涯下一個 5 年高度重疊。學它，寫它，用它——這是接下來的 delta。

有 automotive 背景的軟體工程師，會在接下來 3 年變得非常搶手。你剛好在這個交叉點。

---

## 參考來源

- [NVIDIA Newsroom — Halos for Robotics 發表新聞稿](https://nvidianews.nvidia.com/news/nvidia-announces-halos-for-robotics-the-industrys-first-full-stack-safety-system-for-physical-ai)
- [NVIDIA Technical Blog — Inside NVIDIA Halos for Robotics](https://developer.nvidia.com/blog/inside-nvidia-halos-for-robotics-a-full-stack-functional-safety-system-for-physical-ai/)
- [The Robot Report — NVIDIA releases Halos, a full-stack safety system for robotics](https://www.therobotreport.com/nvidia-releases-halos-a-full-stack-safety-system-for-robotics/)
- [Engineering.com — NVIDIA launches Halos safety system for robotics](https://www.engineering.com/nvidia-launches-halos-safety-system-for-robotics/)
- [Robotics 24/7 — Automate 2026: NVIDIA announces Halos for Robotics](https://www.robotics247.com/article/automate-2026-nvidia-announces-halos-for-robotics)
