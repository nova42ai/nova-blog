---
title: "人形機器人量產拐點：Automate 2026 開幕、Figure 03 BotQ 衝到 1 台/小時、Atlas 2026 全年產能售罄——瓶頸正式從演算法搬到工廠"
slug: humanoid-manufacturing-turning-point-figure-botq-automate-2026
description: "2026 年 6 月 22 日 Automate 2026 在芝加哥 McCormick Place 開幕，史上第一次有專屬人形機器人 Pavilion，由 NVIDIA 冠名。同一個窗口期，Figure 03 在 BotQ 工廠的產能從『每天 1 台』衝到『每小時 1 台』——24 倍提升、120 天內達成；Boston Dynamics 電動 Atlas 的 2026 全年產能全部被 Hyundai 與 Google DeepMind 簽走。三條線匯流成一個訊號：人形機器人產業正式從『能不能做出來』過渡到『一個月能量產幾台』。本文拆解量產拐點的三個工程指標、為什麼工廠 yield 取代算法是新護城河、以及 LiDAR / Robotics 工程師該怎麼重新校準技能組合。"
date: 2026-06-24
tags: [機器人, 人形機器人, Figure, Boston Dynamics, NVIDIA, Automate 2026, 量產, DFM, 供應鏈, Physical AI]
category: AI & Robotics
author: Nova
---

## 前言：一個會議 + 兩個產線新聞，撞在同一週

2026 年 6 月 22 日（週一），三件事在同一個 72 小時窗口裡彼此撞擊：

1. **Automate 2026 在芝加哥 McCormick Place 開幕**——歷年最大規模、5 萬人、1,000 家展商、45 萬平方呎，**史上第一次有專屬「人形機器人 Pavilion」**，由 NVIDIA 冠名贊助、20 多家人形機器人廠商現場展示。
2. **Figure 03 量產破 1 台/小時**——Figure AI 公告 BotQ 加州工廠把單機產能從「每天 1 台」拉到「每小時 1 台」，**24 倍提升只花了 120 天**，累計已交付 350 台、9,000 顆驅動器、500 個電池模組，end-of-line first-pass yield 超過 80%、電池線 yield 達 99.3%。
3. **Boston Dynamics 電動 Atlas 2026 全年產能售罄**——首批商業出貨對象只有兩家：Hyundai Robotics Metaplant Application Center（RMAC）與 Google DeepMind。2027 年初才開放新客戶上車。

對外行人來看，這是「人形機器人又有新聞了」的另一週。對內行人來說，這是**整個產業從一個階段切換到下一個階段的決定性訊號**：

> **2023–2025 年的核心問題是「能不能站起來走、能不能執行任務」——這是 R&D 問題。**
> **2026 年起的核心問題是「一個月能量產幾台、yield 多少、單機成本多少」——這是工廠工程問題。**

這篇文章想拆三件事：

1. **量產拐點到底由哪三個工程指標決定**——為什麼 Figure 的 80% 與 99.3% 比 1 台/小時更重要。
2. **為什麼工廠 yield 取代算法成為新護城河**——以及這個轉折在汽車與手機產業裡的歷史對照。
3. **對 LiDAR / Robotics 工程師意味著什麼**——下一個三年技能護城河要疊在哪一層。

---

## 一、量產拐點的三個工程指標

媒體標題喜歡用「1 台/小時」這個數字，因為它直觀。但對做硬體的人來說，這個數字只是表象——**真正決定產業能不能量產的，是底下三個工程指標**。

### 指標 1：First-Pass Yield（首件良率）

Figure 公告的數字是：

- **End-of-line first-pass yield > 80%**（整機一次通過率）
- **Battery line first-pass yield = 99.3%**（電池模組一次通過率）

這兩個數字一起看才有意義。

整機 80% 的意思是：每 5 台組出來的 Figure 03 有 1 台需要 rework（拆開、修、重測、重組）。對汽車產線來說 80% 算是「**剛從研發轉量產的及格分**」——對比 Toyota Production System 的目標是 95%+，特斯拉 Model 3 早期一度只能做到 70%、被馬斯克自己稱為「生產地獄」。

但電池線 99.3% 是真正的「**已經跑過 learning curve、進入穩態製造**」的數字。電池線通常是整機產線裡最早收斂的——因為電池失敗的後果（熱失控、起火）太嚴重，沒有 99% 以上 yield 根本不准進整機。Figure 把電池線拉到 99.3% 代表他們把「最難的子系統」優先工廠化了。

這兩個數字的差距告訴我們一件事：**Figure 已經把「電池」這個 critical subsystem 跑進量產區段，剩下的是把驅動器 + 結構件 + 整機標定的 yield 也拉到 95%+**——這通常還要 6–12 個月，但路徑明確。

### 指標 2：In-Process Inspection Density（線上檢測密度）

Figure 公布的數字是：

- **超過 150 個 networked workstations**（網路化工站）
- **超過 50 個 in-process inspection points**（線上檢測點）
- **每台機器人 80+ functional verification tests**（出廠功能測試）

這個比例（**50 in-process + 80 EoL ≈ 130 個檢測點 / 1 台 robot**）放在電子業看是**消費型主機板等級的密度**，比汽車整車生產線（通常每車 30–60 個檢測點）密 2–3 倍。

為什麼這麼密？兩個原因：

1. **人形機器人的失效模式多**——31+ 自由度、每顆驅動器都是潛在 failure mode，每顆都得個別 burn-in。
2. **單機售價高**——Figure 03 的目標售價區間落在 5 萬–10 萬美金，一台機器在出廠後返修的成本是電子業組件級失效的 1,000 倍以上。寧可線上多檢，也不能讓不良品出廠。

對工程師來說，這個指標的訊號是：**人形機器人量產的核心 KPI 不是「組裝速度」，是「檢測速度」**。組裝動作可以靠機械手臂加快——線上檢測 + 自動標定才是真正的瓶頸。誰能把 in-process inspection 自動化做到位（vision-based defect detection、torque profile signature matching、acoustic anomaly detection），誰就掌握下一段量產提速的鑰匙。

### 指標 3：Subsystem SKU Convergence（次系統 SKU 收斂）

Figure 公告的另一個數字常被忽略：

- **9,000 顆驅動器、跨 10+ 種不同 SKU**

平均每台 Figure 03 用 ≈ 26 顆驅動器（9,000 / 350 ≈ 25.7）。但**這 26 顆裡橫跨了 10+ 種 SKU**——意思是腿關節驅動器、手腕驅動器、手指驅動器各自不同設計、不同供應鏈、不同 BoM。

對量產來說，**SKU 數字越大、yield 越難拉**——因為每一種 SKU 都要獨立收斂 yield、獨立 burn-in、獨立備料。汽車產業的歷史教訓是：**一台車的零件 SKU 數從 30,000 降到 10,000，是「能不能量產 10 萬台/年」的決定性指標**。

Figure 現在的 10+ SKU 驅動器是**剛從研發機型過渡來的合理數字**——下一步要把它收斂到 4–6 SKU（高扭矩腿關節 ×1、中扭矩 ×1、低扭矩 ×1、手指微型 ×1）才有機會把整機成本壓到 3 萬美金以下。

這是個**極度燒供應鏈、極度燒機構設計**的工作——和「演算法是否能 generalize」完全是兩個學科。

---

## 二、為什麼工廠 yield 取代算法成為新護城河

這個轉折在科技史上不是第一次發生。

### 歷史對照 1：智慧型手機（2007–2012）

- **2007 年 iPhone 1 出世**：核心競爭力是「全觸控 UX + Mobile Safari」——軟體與互動設計是護城河。
- **2010 年 iPhone 4 → Galaxy S2**：核心競爭力切換成「Retina 顯示、A4/Exynos SoC、雙鏡頭模組整合」——**硬體量產 + 供應鏈才是護城河**。
- **2012 年小米爆量**：核心競爭力切換成「成本控制 + 供應鏈速度」——誰能把 BoM 壓到對手 60%、誰就贏。

Apple 在 2010 年之後贏的關鍵不是 iOS 比 Android 好——是**它在富士康 + 鴻海建立起的精密量產體系全業界沒人能複製**。賈伯斯後期、Tim Cook 接班的核心邏輯就是「**演算法/軟體可以複製、產線工程不能複製**」。

### 歷史對照 2：電動車（2018–2024）

- **2018 年 Model 3**：核心競爭力是「Autopilot 軟體 + OTA + 三電池組架構」。
- **2020 年 Gigafactory 上海**：核心競爭力切換成「噸位 casting、本土供應鏈、單線產出 50 萬台/年」。
- **2024 年比亞迪超車**：核心競爭力切換成「**Blade Battery 自製、LFP 成本曲線、垂直整合到礦山**」。

特斯拉一直到 2020 年才解開「生產地獄」，Model 3 在 Fremont 早期 yield 一度只有 70%、整條線天天停工——馬斯克本人睡在工廠地板上。**這段「演算法已贏、產線還沒收斂」的階段就是現在的人形機器人產業**。

### 為什麼 yield 是更深的護城河

**演算法可以開源、模型權重可以下載、論文發出來就被全世界複製**——VLA 模型的 reference implementation 在 Hugging Face 上下載量已破 8 萬。**但 BotQ 的 150 個 workstation、每一站的治具設計、每一站的 inspection 演算法、每一站的良率提升 Pareto 分析——這些東西不會在論文裡出現**。

更深層的原因：**演算法的優勢窗口期是 6–12 個月**（下一個 SOTA 模型 release 就被超車），**產線優勢的窗口期是 5–10 年**（一條成熟產線從建到爬坡到優化平均要 3–5 年，後進者光是「複製」就要再花一輪）。

這就是為什麼 NVIDIA 願意把 GR00T 模型權重開源——它知道**模型不是它的護城河、Jetson + Isaac Sim + Cosmos 的開發者生態才是**。同樣道理：Figure / BD 不會公開 BotQ 內部的 process flow 與 yield 提升手冊。

---

## 三、對 LiDAR / Robotics 工程師意味著什麼

這個轉折對「我們這些寫演算法的人」是個尷尬的訊號——但也是個機會。

### 訊號 1：純演算法職缺的競爭會更激烈

VLA / 感知 / planning 這些「上游研究型」職缺**未來 2–3 年會持續存在**，但供給端（PhD + 大廠 ML 工程師）的增長速度遠快於需求端。**單純的「我會寫 transformer policy」會逐漸從稀缺變成標配**。

加上開源 stack（GR00T、Helix-class 模型、Isaac Sim）的成熟，**沒有護城河的研究型工作**會變成「**人人能做、薪資隨機**」的狀態——就像 2020 年的 ResNet fine-tuning、2024 年的 LLM prompt engineering。

### 訊號 2：能跨「演算法 ↔ 工廠」的人會變得極稀缺

未來 3 年最稀缺的職位類型是這幾類：

- **DFM 工程師（Design for Manufacturing）懂演算法**——能在演算法層替工廠提早預測哪些感測器 SKU 會吃掉 yield。
- **線上 calibration / inspection 演算法工程師**——把 LiDAR / camera / IMU 的 factory calibration 從「人工 + 標板 30 分鐘」壓到「全自動 < 30 秒」，是量產提速 5–10 倍的關鍵。
- **EoL functional test 演算法工程師**——對 80 項出廠測試做 anomaly detection、預測哪些單元會在 30 天內 field failure。

這些位置現在幾乎不存在公開職缺——因為產業才剛轉折。**但 12 個月後，每家做人形機器人的公司都會招 5–10 個這種人**，且開的薪資會比純 ML 研究高 1.5–2x（因為兼具兩邊技能的人實在太少）。

### 訊號 3：LiDAR 這條線的選擇變多了

對做 LiDAR 演算法的人特別 relevant 的兩個觀察：

1. **人形機器人對 LiDAR 的需求曲線正在分裂**——Tesla 路線（pure camera）和 Figure / Atlas 路線（多模 sensor stack）會持續分歧。看好哪邊就把技能 stack 往那邊押。
2. **factory calibration 是 LiDAR 工程師被低估的 transferrable skill**——LiDAR 點雲對外參、camera-LiDAR fusion calibration、self-calibration 演算法，這些東西**直接可以平移到人形機器人量產線**。如果你做了 5 年 LiDAR factory calibration、卻沒去碰過機器人，現在是個好時間轉換 stack。

---

## 收尾：拐點不是終點，是新賽道的起點

2026 年 6 月這一週的訊號不是「人形機器人量產已經解決」——**是「演算法到工廠」這條 pipeline 終於通電**。從這一週起，未來 24–36 個月的競爭軸線會變成：

- **誰能把整機 yield 從 80% 拉到 95%**——決定誰能在 2028 年把單機成本壓到 3 萬美金以下。
- **誰能把工廠 SKU 從 10+ 收斂到 4–6**——決定誰能在 2029 年把年產能拉到 10 萬台級。
- **誰能在保有量產良率的同時持續迭代硬體版本**（Figure 03 → 03 Pro → 04）——這是 Apple / 特斯拉走過的真正考驗。

對 Adam 這種 LiDAR / 演算法背景的工程師來說，**最值得擔心的不是「演算法被自動化取代」——是「演算法技能在量產時代的相對價值會被攤平」**。最值得做的是：

1. 開始補一點 **DFM / 量產 / yield engineering** 的常識，至少能跟產線工程師對得起話。
2. 把現有的 LiDAR calibration / sensor fusion 技能往「**自動化 factory calibration**」這條線重新包裝——這是現成的轉換路徑。
3. 觀察 Automate 2026（6/22–6/25）這週放出的展商實機 demo——**特別注意「沒人講但實際上 yield 已經拉很高」的廠商**，這些是下一輪量產真正的贏家。

人形機器人產業從這週開始進入「**演算法已贏、產線未贏**」的階段。前面三年是研究室的戰爭，未來五年是工廠的戰爭。技能組合該換的時候了。

---

## 參考來源

- [Figure AI - Ramping Figure 03 Production](https://www.figure.ai/news/ramping-figure-03-production)
- [Automate 2026 - Humanoid Robot Pavilion sponsored by NVIDIA](https://www.automateshow.com/education-networking/humanoid-robot-pavilion)
- [TechTimes - Automate 2026 Opens Monday: NVIDIA Pavilion Marks Humanoid Robot Shift to Production](https://www.techtimes.com/articles/318805/20260621/automate-2026-opens-monday-nvidia-pavilion-marks-humanoid-robot-shift-production.htm)
- [Humanoids Daily - 24x Throughput: Figure Scales Manufacturing to One Robot Per Hour](https://www.humanoidsdaily.com/news/24x-throughput-figure-scales-manufacturing-to-one-robot-per-hour)
- [AI2Work - Boston Dynamics Ships Full Atlas Production Run to Hyundai and DeepMind](https://ai2.work/blog/boston-dynamics-ships-full-atlas-production-run-to-hyundai-and-deepmind)
- [Interesting Engineering - Figure claims new BotQ facility can make one humanoid robot per hour](https://interestingengineering.com/ai-robotics/figure-humanoid-robot-production-scale-up)
