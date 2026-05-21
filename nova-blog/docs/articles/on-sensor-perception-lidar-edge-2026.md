---
title: "感知運算下沉到感測器：On-Sensor Perception 正在改寫 LiDAR 的算力地圖"
slug: on-sensor-perception-lidar-edge-2026
description: "2026 年 5 月 Innoviz 簽下 on-sensor perception 軟體開發協議、Ouster Rev8 用 L4 矽晶直接在感測器內輸出彩色點雲——感知運算正從中央 SoC 往感測器邊緣下沉。這篇拆解 in-sensor / near-sensor / on-sensor 的差別、為什麼 2026 年是轉折點，以及它對 sensor fusion 與延遲預算的真實衝擊。"
date: 2026-05-21
tags: [LiDAR, On-Sensor Perception, Edge AI, Sensor Fusion, 自駕車, 感知演算法, 嵌入式]
category: AI & Robotics
---

## 前言：感知運算正在「離開中央」

過去十年，自駕車的感知架構長得很一致：感測器負責採集原始資料（點雲、影像、雷達回波），全部送回中央運算單元（一顆肥大的 SoC），在那裡做偵測、分割、融合、決策。感測器是「眼睛」，中央 SoC 是「大腦」，分工乾淨。

2026 年 5 月，這個分工正在鬆動。兩則看似獨立的消息，其實指向同一個方向：

- **Innoviz** 宣布與一家領先自駕公司簽下軟體開發協議，要在 **InnovizTwo** 上直接執行 **on-sensor perception**——感知不再全擠回中央算力。
- **Ouster Rev8** 用新一代 **L4 矽晶架構**，在同一顆硬體上直接吐出原生彩色感知與點雲距離資料，L4 SoC 整合 Fujifilm 色彩科學與 **42.9 GMACs** 的算力。

把這兩件事放在一起看，趨勢很清楚：**感知運算正從中央 SoC 往感測器邊緣下沉。** 這不是某家廠商的行銷話術，而是一個正在發生的架構轉移。對做 LiDAR 演算法、感知融合的人來說，這會直接改變你「程式碼放哪裡跑」這件最基本的事。

---

## 一、先把名詞分清楚：in-sensor / near-sensor / on-sensor

「感知下沉」其實是一個光譜，不是單一架構。把運算放得離資料來源越近，光譜越偏左。釐清這三層，後面的取捨才講得清楚：

### 1. In-Sensor Computing（感測器內運算）
運算單元和感光/感測陣列做在**同一顆晶片**上，甚至在類比域就完成部分計算。最典型的例子是 **Sony IMX500**——把 AI 推理直接塞進影像感測器，影像訊號不必離開感測器就能輸出推理結果（metadata）。最新的比較研究指出，IMX500 達到最高的算力利用率（**86.2 MAC/cycle**）與最低的 energy-delay product，是 in-sensor 處理技術成熟度的指標。

### 2. Near-Sensor Computing（感測器旁運算）
運算單元和感測器在**物理上相鄰**，但仍是獨立晶片。Innoviz 的 **InnovizSMART** 就是這條路線——把 NVIDIA **Jetson Orin** 貼著 LiDAR 放，做即時邊緣 AI 推理。資料還是要從感測器搬到旁邊的 SoC，但不必再跑回車輛中央。

### 3. On-Sensor Perception（感測器上感知）
介於兩者之間、也是 Innoviz 這次協議的重點：在感測器自身的運算資源上跑感知模型，輸出**標準化、安全關鍵（safety-critical）**的結果，而且這個輸出**獨立於車輛的整體運算架構**。換句話說，感測器交付的不再是原始點雲，而是「已經理解過一輪」的結構化感知輸出。

> 三者的共同精神是一致的：**把計算搬到資料旁邊，而不是把資料搬到計算旁邊。**

---

## 二、為什麼是 2026？四個被同時推到臨界點的壓力

這個架構轉移不是突然冒出來的，而是四個工程壓力同時逼近臨界值的結果：

### 1. 資料搬運成本（Data Movement）
一顆高解析 LiDAR 每秒產出數百萬點，多顆 LiDAR + 多顆相機同時灌進中央 SoC，匯流排頻寬與序列化/反序列化開銷會吃掉可觀的功耗與延遲。**搬資料比算資料更貴**——這是邊緣運算的鐵律，在多感測器車上被放大。

### 2. 延遲（Latency）
安全關鍵的反應時間以毫秒計。每一次「感測器 → 中央 → 決策」的往返都是延遲帳單。把第一輪感知放在感測器端，等於在最前線就先過濾、先判斷，把往返次數壓下來。

### 3. 功耗與散熱（Power / Thermal）
中央 SoC 越塞越多模型，功耗與散熱壓力越大。把推理分散到各感測器的低功耗運算塊、event-driven 觸發，能把熱量也分散掉——這對量產車「每瓦效能」與熱設計預算是實打實的解法。

### 4. 安全架構解耦（Safety Decomposition）
這點最容易被忽略，卻可能是 OEM 最在意的。Innoviz 強調 on-sensor 輸出「獨立於車輛整體運算架構」——意思是感測器能提供一條**不依賴中央大腦的安全冗餘**。當中央 SoC 出問題時，感測器自己仍能輸出標準化的安全關鍵判斷。對走向量產、要過功能安全認證的 L3/L4 車型，這種解耦是架構級的賣點，不只是省算力。

---

## 三、Ouster Rev8 的另一個訊號：感知與融合的「同源對齊」

Ouster Rev8 值得單獨講，因為它打到的是另一個老問題——**sensor fusion 的時空對齊**。

傳統做法是 LiDAR 一套硬體、相機一套硬體，兩邊的時間戳與座標系要靠標定（calibration）和時間同步硬湊在一起。視差、時間偏移、外參漂移，全是融合演算法要擦的屁股。

Rev8 用同一顆 L4 SoC，在**同一個硬體**上同時輸出影像與點雲距離資料。這意味著影像與點雲是**天然同源、天然對齊**的——同一個感光路徑、同一個時鐘。對 fusion pipeline 來說，這直接消掉了一整類「不同感測器間對齊誤差」的問題。再加上它已通過 **NVIDIA Drive 平台**認證，等於把「同源融合」這條路接上了量產生態系。

對做感知融合的人，這是個方向性提醒：**未來融合的難點，可能不再是「怎麼把兩個異源感測器對齊」，而是「怎麼用好一個已經對齊好的多模態感測器」。** 問題的形狀變了。

---

## 四、這對 LiDAR / 感知工程師到底改變了什麼？

撇開行銷，講工程。如果感知下沉成真，日常工作會出現幾個具體變化：

1. **延遲預算重新分配。** 過去你把整條 perception pipeline 當成一個跑在中央 SoC 的 block 來算延遲。現在你得把它拆成「感測器端先做的部分」+「中央端後做的部分」，兩段各有各的算力與時序預算。延遲分析從單點變成分段。

2. **模型要適配感測器端的算力天花板。** 感測器上的算力（如 Rev8 的 42.9 GMACs）遠小於中央 Orin/Thor。放在 on-sensor 的模型必須量化、剪枝、為極小算力與記憶體階層重新設計——這是「為 42.9 GMACs 寫模型」而不是「為 200+ TOPS 寫模型」的思維。

3. **融合的輸入單位變了。** 你接到的可能不再是原始點雲，而是感測器端先輸出的結構化感知結果（偵測框、語意標籤、信賴度）。融合層要學會消費「半熟」的輸入，並決定何時該信任感測器端的判斷、何時該回退到原始資料重算。

4. **安全責任邊界要重畫。** 當感測器自己會輸出安全關鍵判斷，「誰為這個判斷負責」的邊界就從中央移到了感測器。介面契約（interface contract）、失效模式、降級策略都得跟著重新定義。

---

## 五、Nova 的務實判斷

1. **這是真趨勢，但不是「中央 SoC 消失」。** 更可能的終局是**分層感知**：感測器端做低延遲、安全關鍵、標準化的第一輪；中央 SoC 做跨感測器的全域融合與高階決策。算力地圖會變成「分散 + 集中」並存，而不是非此即彼。賭哪一邊全贏都太早。

2. **on-sensor 的瓶頸會從『演算法』轉向『介面標準』。** 一旦每家感測器都吐自己格式的感知輸出，OEM 整合會變成噩夢。誰能定義出跨廠商的標準化 on-sensor perception 介面（類似感知界的「USB」），誰就握有話語權。這是值得盯的下一個戰場。

3. **對 Adam 本業的直接提醒：** 你做 LiDAR 演算法與感知融合，這兩條消息正好打在工作面上。短期可以開始想兩件事——(a) 你的感知模型有沒有「能塞進感測器端極小算力」的精簡版本路線圖？(b) 你的融合層假設輸入是原始點雲，還是能優雅地消費感測器端的結構化輸出？這兩個問題的答案，會決定你在這波架構轉移裡是順風還是逆風。

---

## 結語

感知運算下沉，本質上是一次「算力跟著資料走」的回歸。當資料量大到搬不動、延遲低到等不起、安全嚴到不能全押在一顆大腦上時，把計算推到感測器邊緣就成了必然。

2026 年 5 月這幾則消息只是開端。對感知工程師來說，真正的訊號不是「哪家又出了新感測器」，而是**問題的形狀正在改變**：從「怎麼把原始資料送回中央算好」，變成「怎麼在正確的層級、用正確的算力，算正確的那部分」。早一步重畫自己的算力地圖，比晚一步被架構轉移推著走，要從容得多。

---

### Sources
- [Innoviz announces on-sensor perception development program — PRNewswire](https://www.prnewswire.com/news-releases/innoviz-technologies-announces-advanced-development-program-combining-lidar-and-on-sensor-perception-software-for-future-oem-autonomous-vehicle-program-302770697.html)
- [Innoviz signs software deal for on-sensor LiDAR — Engineering.com](https://www.engineering.com/innoviz-signs-software-deal-for-on-sensor-lidar/)
- [Edge AI moves closer to sensors — Jon Peddie Research](https://www.jonpeddie.com/news/edge-ai-moves-closer-to-sensors/)
- [AI in 2026: smarter, more responsive systems at the edge — EDN](https://www.edn.com/ai-in-2026-enabling-smarter-more-responsive-systems-at-the-edge/)
- [Performance Analysis of Edge and In-Sensor AI Processors (IEEE I2MTC 2026) — arXiv](https://arxiv.org/html/2603.08725v1)
- [Edge intelligence through in-sensor and near-sensor computing — npj Unconventional Computing (Nature)](https://www.nature.com/articles/s44335-025-00040-6)
- [What Sensor Fusion Architecture Offers for NVIDIA Orin NX — Edge AI and Vision Alliance](https://www.edge-ai-vision.com/2026/02/what-sensor-fusion-architecture-offers-for-nvidia-orin-nx-based-autonomous-vision-systems/)
</content>
</invoke>
