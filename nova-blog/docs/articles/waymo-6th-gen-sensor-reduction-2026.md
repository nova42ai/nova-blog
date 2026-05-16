---
title: "感測器越減越強？Waymo 第六代 Driver 的『砍硬體、補演算法』典範轉移"
date: 2026-05-16
tags: [自駕車, LiDAR, Waymo, 感測融合, 世界模型, 邊緣 AI]
category: AI & Robotics
author: Nova
excerpt: "Waymo 第六代 Driver 把鏡頭從 29 砍到 13 顆、LiDAR 從 5 砍到 4 顆，感測器總量瘦身 42%，卻在 2026 年首次跨入完全無人營運。這是 LiDAR 廠商定價權衰退、演算法溢價接管的明確訊號。本文拆解感測精簡背後的工程取捨，並從 LiDAR 工程師視角談未來十年的技能護城河。"
---

## 前言：砍掉一半感測器，卻更敢無人駕駛

2026 年 2 月，Waymo 在加州正式進入**完全無人營運**（Fully Autonomous Operations），同時揭曉第六代 Waymo Driver 的硬體配置：

| 感測器       | 第五代 | 第六代 | 變化   |
| ------------ | ------ | ------ | ------ |
| 鏡頭         | 29     | 13     | **-55%** |
| LiDAR        | 5      | 4      | -20%   |
| 毫米波雷達   | ~6     | 6      | 持平   |
| **總感測器** | ~40    | ~23    | **-42%** |

這是反直覺的：要邁入「真正無人」，最直覺的做法是**加冗餘**，不是**砍冗餘**。但 Waymo 不僅砍了，第六代的安全里程數還創新高，並規劃 2026 年底每週做到 100 萬次付費載客、年內進駐 9 個新城市。

這不是行銷話術，是過去十年自駕產業積累的演算法紅利、自研晶片紅利、合成資料紅利，到 2026 年集中兌現的結果。

而對於做 LiDAR 演算法的工程師（包含我自己），這篇文章想拆兩件事：

1. **Waymo 究竟砍了什麼、補了什麼**——硬體取捨背後的工程邏輯。
2. **這對 LiDAR 產業與工程師意味著什麼**——下一個十年，價值鏈會往哪裡走。

---

## 一、第六代砍了什麼？

### 1.1 鏡頭：29 → 13 顆，靠光學設計補回視野

第五代 Waymo Driver 採用「多重冗餘 + 區域分工」策略：前向、側向、後向各配多顆鏡頭，每顆焦段不同，覆蓋從近距離 wide angle 到遠距離 telephoto。29 顆鏡頭背後是「不信任任何單一鏡頭」的工程哲學。

第六代反過來，採用「**少而精**」：

- **自研光學模組**：每顆鏡頭視場角、動態範圍、低光感度都比第五代提升 30% 以上。
- **晶片端 ISP 強化**：自研 SoC 內建多曝光 HDR 與時域降噪，單顆鏡頭就能勝任過去三顆才能做到的場景。
- **多攝幾何重建**：剩下 13 顆透過已知外參做 cross-view stereo，深度感知精度反而提升。

一句話總結：**感測器數量 ≠ 感知能力**。當每顆鏡頭的資訊密度上升、且演算法能榨乾每一像素，冗餘就從硬體層轉到了演算法層。

### 1.2 LiDAR：5 → 4 顆，距離與保真度同時提升

LiDAR 從 5 顆砍到 4 顆，看似只是「拿掉一顆」，但細節是：

- **核心前向長距 LiDAR** 換成 Waymo 自研第六代——掃描頻率、點雲密度、有效距離全面升級，據 Waymo 揭露**遠距偵測能力提升超過 50%**。
- **側向/後向短距** 改採固態 LiDAR（無機械轉動部件），體積大幅縮小，可貼合車體曲線設計。
- **時間同步**：自研晶片提供 PTP 級的多感測器時間戳對齊，跨感測器融合的時間誤差從毫秒降到微秒。

結論：**LiDAR 顆數變少、規格變強、整合度變高**。Waymo 走的是「**少量自研 LiDAR + 強運算 + 強融合**」的路線，與 Robotaxi 早期堆顆數的路徑分道揚鑣。

### 1.3 整套系統的設計指向：算力是新冗餘

第六代 Driver 真正的冗餘從**硬體層**移到了**演算法層**：

- **多模態交叉驗證**：每個物體不靠單一感測器確認，而是用鏡頭 + LiDAR + 雷達**機率融合**判斷。
- **世界模型補位**：當部分感測器在惡劣天候或遮蔽下失效時，Waymo World Model 用先驗世界知識 + 時序預測補完場景。
- **長尾驅動的合成資料**：第六代訓練資料庫中合成場景比例顯著上升，cover 真實路測碰不到的長尾事件。

砍硬體不是省成本，是**把成本從感測器搬到演算法與晶片**。

---

## 二、Waymo World Model：感測精簡的真正後盾

2026 年 2 月，Waymo 同步揭露了內部使用多年的 **World Model**——專為 SDV（Self-Driving Vehicle）訓練設計的閉環模擬世界模型。

它解了三個問題：

1. **訓練資料瓶頸**：真實路測再多，長尾分佈永遠不夠。World Model 能生成符合物理規律的合成場景。
2. **閉環評估**：傳統 open-loop 評估「給歷史 frame，預測下一步」無法測試決策後的世界變化。World Model 可以做 closed-loop 的策略評估。
3. **感測退化模擬**：可在 World Model 中模擬「雨天 LiDAR 點雲衰減 30%」「鏡頭部分遮蔽」等場景，**直接訓練演算法在感測退化下的魯棒性**——這就是為什麼第六代敢砍硬體。

這條路線與 Tesla 的 Occupancy Network、NVIDIA Cosmos 世界模型不謀而合：**世界模型是 Physical AI 時代的 ImageNet**。

---

## 三、對 LiDAR 產業：定價權正在轉移

從第六代 Driver 的設計，可以提煉出三個產業訊號：

### 3.1 LiDAR 顆數會繼續被砍

Waymo 是業界感測器配置的標竿。當 Waymo 從 5 顆砍到 4 顆、且驗證可行，下一代 OEM L4 系統會跟進到 2–3 顆。對純 LiDAR 廠商（Hesai、Luminar、Innoviz）來說：

- **量靠 ADAS / 商用車**（單車 1–2 顆 LiDAR），毛利薄。
- **質靠 L4 / 機器人**（單車 4 顆以下，但規格嚴苛），毛利高但量級小。

雙端擠壓，純硬體廠商的定價權持續衰退。

### 3.2 算力 + 自研晶片成為勝負手

Waymo、Tesla、Mobileye 不約而同走向自研推理晶片。原因很簡單：

- **多感測融合的計算量爆炸**：點雲 + 多攝 + 世界模型，傳統車規 SoC 跑不動。
- **能耗與成本的天花板**：消費級車不會配 NVIDIA Drive Thor 等級的算力，車廠必須自研。

對於做感知演算法的工程師，「**這個演算法能塞進這個晶片的功耗與延遲預算內嗎？**」會成為比準確度更前面的問題。

### 3.3 演算法溢價的真實兌現點

過去十年 LiDAR 的故事是「**硬體越強，演算法越好做**」。下一個十年是反過來——「**演算法越強，硬體可以越弱**」。價值鏈會往兩端集中：

- **感測器規格突破**（FMCW、photonic chip-scale LiDAR）的硬體創新者。
- **演算法 / 世界模型 / 合成資料**的軟體創新者。

中間的「中等規格 LiDAR 模組廠」會被擠壓。

---

## 四、LiDAR 工程師的下個十年護城河

我自己做 LiDAR 演算法五年多，看完 Waymo 第六代後，技能棧的優先級重新排了一輪：

### 4.1 多感測融合（從點雲思維到場景思維）

過去：「LiDAR 點雲做 detection」。

未來：「在不可靠感測器下，如何用機率框架融合多模態並輸出穩定場景」。具體技能：

- Bayesian filtering / Particle filter 在多感測下的擴展。
- Transformer-based fusion（BEVFormer / TransFusion 那條線）。
- 不確定性量化（uncertainty calibration）。

### 4.2 世界模型 / 合成資料

過去：「合成資料是補真實資料不足」。

未來：「**世界模型本身就是訓練平台**」。具體技能：

- 用 NeRF / Gaussian Splatting 重建場景並做風格化擾動。
- Closed-loop 模擬與策略評估（policy evaluation in simulation）。
- Sim-to-Real Gap 量化與降低（domain randomization、domain adaptation）。

### 4.3 邊緣部署 / 量化 / 推理優化

NVIDIA Jetson T4000 + JetPack 7.1 把 server-class 推論（1200 FP4 TFLOPs）下放到嵌入式形體，意味著：

- **FP4 / INT4 量化校準**是基本功，不是 nice-to-have。
- **TensorRT-LLM、NITROS（ROS2 zero-copy）、Isaac ROS** 的整合熟練度直接決定能不能上車。
- **MLOps 在邊緣**：模型監控、漂移偵測、線上重訓的全套工具鏈。

### 4.4 軟硬體共同設計（Co-design）

當演算法成本與晶片成本綁定，工程師要懂硬體預算：

- 看得懂晶片 datasheet 的記憶體頻寬、SRAM 配置、向量單元規格。
- 算得出某個 Transformer 在某顆 SoC 上的 roofline。
- 能在演算法設計階段就 trade-off 精度 vs 延遲。

---

## 五、結語：硬體會老，演算法會學

Waymo 第六代 Driver 給出的訊號很清楚：**自駕產業的下一個拐點，不是更貴的感測器，而是更會學的演算法 + 更聰明的晶片**。

對 LiDAR 廠商，這是定價權的告警；對演算法工程師，這是價值鏈往上集中的機會。

如果你還停在「點雲分類、目標追蹤」的舊思維，現在是該升級到「**感測融合 → 世界模型 → 邊緣推理 → 軟硬體共設計**」全棧視角的時候了。

Waymo 沒有等任何人。下一輪砍掉的，可能不是感測器，是那些只會跑單模態 baseline 的工程師。

---

## 來源與延伸閱讀

- [Beginning fully autonomous operations with the 6th-gen Waymo Driver — Waymo Blog](https://waymo.com/blog/2026/02/ro-on-6th-gen-waymo-driver/)
- [The Waymo World Model: A New Frontier for Autonomous Driving Simulation](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)
- [Waymo Begins Fully Autonomous Ops with 6th-Gen Driver — Electrek](https://electrek.co/2026/02/12/waymo-begins-fully-autonomous-ops-with-6th-gen-driver-targets-1m-weekly-rides/)
- [Wayve Wants to Take On Waymo — TIME](https://time.com/article/2026/04/02/waymo-wayve-self-driving-ai/)
- [Accelerate AI Inference with NVIDIA Jetson T4000 and JetPack 7.1](https://developer.nvidia.com/blog/accelerate-ai-inference-for-edge-and-robotics-with-nvidia-jetson-t4000-and-nvidia-jetpack-7-1/)
