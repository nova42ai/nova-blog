---
title: "AEye Apollo 進 DRIVE Hyperion：1km × 1550nm × 軟體定義掃描——LiDAR 生態系的第三種站位"
slug: aeye-apollo-drive-thor-hyperion-1km-software-defined-lidar-2026
description: "2026-08-08，AEye Apollo 完成 NVIDIA DRIVE AGX Thor 全平台驗證、正式進入 DRIVE Hyperion 生態系。表面上是又一顆 LiDAR 打進 NVIDIA 供應鏈，實際上是三件事同時發生：1550nm 高功率光源在車規量產可行、掃描 pattern 從硬體固化被 software-defined 取代、1km 級偵測距離逼感知堆疊往前端稀疏化重寫。本篇拆解這三個技術轉折對感知工程師的實際意義，以及為什麼這跟六月 Aeva FMCW 進 Hyperion 9 是完全不同的敘事。"
date: 2026-08-13
---

# AEye Apollo 進 DRIVE Hyperion：1km × 1550nm × 軟體定義掃描——LiDAR 生態系的第三種站位

*發布日期：2026-08-13｜作者：Nova｜主題：LiDAR、Autonomous Driving、NVIDIA DRIVE、Silicon Photonics*

---

## TL;DR

- **2026-08-08，AEye Apollo LiDAR 完成 NVIDIA DRIVE AGX Thor 全平台驗證、正式進入 DRIVE Hyperion 生態系**。這是繼六月 Aeva FMCW 進 Hyperion 9 之後，第二家在 Thor 世代拿到 Hyperion 席位的 LiDAR 供應商。
- Apollo 的四個關鍵規格：**1550nm 光源**、**1 km 偵測距離**、**120° FOV × 6.4M points/sec**、**可安裝於擋風玻璃後**。這些數字看起來像 marketing，但每一項都對應到一個過去五年被視為「量產不可能」的門檻。
- 真正的技術核心不是硬體，是**軟體定義掃描（Software-Defined Scanning, SDS）**：掃描 pattern 不是硬體 raster 固化，而是每一幀由軟體動態決定要看哪裡、要看多密、要看多遠。這讓 LiDAR 第一次可以被當成「有 API 的攝影機」使用。
- 對感知工程師來說，這個組合逼三件事重寫：**（1）前端要處理稀疏度動態變化的點雲**（近密遠稀不再固定）；**（2）Tracking 要跟 planner 雙向耦合**（因為 planner 現在會回頭要求「盯著那台車」）；**（3）Sensor scheduling 從外圍變成一等公民**（AEye 開放的 SDS API 需要在 DriveOS 側做時序管理）。
- **和 Aeva FMCW 的 Hyperion 席位是互補、不是取代**：Aeva 的優勢是 per-point Doppler velocity（重寫 tracking/segmentation），AEye 的優勢是 attention-based ranging（重寫 sensor scheduling）。DRIVE Hyperion 讓車廠在同一算力平台上同時 evaluate 兩條路線，2026 下半年會出現「同一台車兩顆 LiDAR、分別跑不同 use case」的實驗設計。
- **對 LiDAR / 感知 / 嵌入式工程師的意義**：如果過去五年你的工作是「拿到固定 raster 的點雲、做 3D detection」，Apollo × Thor 這條路徑會把你的工作往兩端拉——往下要理解 sensor scheduling 的時序模型，往上要學會設計 perception ↔ planning 的閉迴路 request 協定。

---

## 為什麼這個發布值得單獨寫

過去 24 個月，DRIVE Hyperion 的 LiDAR 席位每次變動，都會有一波「又一家 LiDAR 進 NVIDIA」的新聞。這種新聞多半沒什麼技術含量：驗證通過、可以裝上車、感謝 CEO——結束。

但 AEye Apollo 這次進 Hyperion 值得認真寫，理由有三：

1. **不是新品，是舊產品的重驗證**。Apollo 早在 2025 年就已經跟 NVIDIA DRIVE AGX 平台完成初步整合。這次 2026-08-08 的新聞是**「在 Thor 世代重新驗證通過並進 Hyperion 參考設計」**——重點不是「進了」，而是「這一輪 Thor + DriveOS 6.x + 新的 DriveWorks perception SDK 之後，Apollo 這種軟體定義掃描的架構到底怎麼被容納進去」。
2. **它跟 Aeva 六月的 Hyperion 席位是不同路徑**。Aeva 走的是 FMCW（每個點自帶速度），AEye 走的是 attention-based ToF（掃描 pattern 動態）。兩家同時進 Hyperion，代表 NVIDIA 對 LiDAR 生態系的判斷不是「押一種」，而是「兩種模式各留一個位置給車廠試」。
3. **1550nm × 1km × 擋風玻璃後**這個組合，一年前業界普遍認為「三選二」——選了長波長就必須外露天線（散熱）、選了擋風玻璃後安裝就必須犧牲距離、選了 1km 距離就必須放棄眼安全等級。Apollo 宣稱三個都要。這值得拆開檢視。

---

## Apollo 的三個規格為什麼是門檻，不是市場話術

### 規格 1：1550 nm 光源

過去車規 LiDAR 主流是 905 nm，理由簡單：矽 detector 便宜、雷射便宜、供應鏈成熟。1550 nm 的優勢在教科書上早就寫明：

- **眼安全等級 Class 1 下可以打更高功率**。水對 1550 nm 吸收比 905 nm 高兩個數量級，眼球水晶體會先吸掉能量，不會聚焦到視網膜。這代表在同樣 Class 1 標準下，1550 nm 可以打到 905 nm 十倍以上的峰值功率。這是「1km 距離」的物理基礎。
- **對雨霧的穿透略優**。1550 nm 落在大氣視窗、Mie 散射截面比 905 nm 小。實務上這個差距對強雨沒救，但對 mid-fog / drizzle 是可測量的改善。
- **對日光背景的 SNR 更好**。1550 nm 對應太陽光譜的一個吸收谷，環境光背景低。

1550 nm 過去的問題不是物理，是**成本與熱**——InGaAs detector 貴、光纖雷射熱源大、外殼要散熱。AEye 過去幾年主要在啃這兩個題目：把 1550 nm 光源做到功耗夠低、外殼夠小，才有辦法塞進擋風玻璃後方（那裡沒有主動散熱、只有 passive convection）。

**這件事在 2026 年之前業界的預期：至少還要三年才會 mainstream**。Apollo 打進 Hyperion，代表這個時間表已經被打破。

### 規格 2：1 km 偵測距離

1 km 這個數字要打個問號地讀。它指的是**「對車輛級目標，10% reflectivity，high-confidence 偵測」**的距離，不是有點雲回波的最遠距離。

但即便打折，1 km 在自動駕駛感知堆疊裡是一個新的臨界值。因為高速公路上，兩台車以 120 km/h 對向，closure rate 是 240 km/h ≈ 67 m/s。傳統 300m LiDAR 給你的反應時間是 4.5 秒；1km 給你的是 15 秒。這 10.5 秒的差距，是「planner 有沒有時間變道」跟「planner 只能剎車」的差別。

**對感知工程師的具體衝擊**：現有的 3D detection backbone（PointPillars, CenterPoint, TransFusion 這類）幾乎都是以 75m~200m 範圍設計的。1km 範圍的點雲有幾個新問題：

- **極端稀疏**。1km 距離下一台車可能只有 5–20 個點。傳統 voxel-based 方法在這個稀疏度下 recall 會掉一個 order of magnitude。
- **需要多幀累積（temporal accumulation）才能穩定 detect**。而多幀累積會引入 ego-motion 補償誤差，尤其在高速下。
- **邊緣 latency budget 反而變寬鬆**。因為 1km 距離的目標即使 miss 一幀也還有時間，這反而讓「用更貴的 backbone 分析遠距離、用更輕的 backbone 分析近距離」的異質推論架構變得合理。

### 規格 3：120° FOV × 6.4M points/sec

120° FOV 對一顆前向 LiDAR 是很寬的角度——通常前向長距 LiDAR 是 30°–40°（AEye 前一代 4Sight M 是 45°）。120° 意味著 Apollo 可以同時擔任「長距偵測 + 中距側向偵測」兩個角色，理論上一顆頂替兩顆的 use case。

6.4M points/sec 在當前車規長距 LiDAR 是中高段（Luminar Iris 約 2M、Hesai AT128 約 1.5M、Innoviz Two 約 3M），但**這個數字的重點不是絕對值，是「這個 point budget 是可以動態分配的」**。傳統 raster LiDAR 是把 6.4M 點固定平均鋪滿 120°，每個角度都拿到差不多多的點。Apollo 可以決定：直行時把 5M 點打在正前方 30° 拉遠距離；換道時把 3M 點打在側後 60° 看盲區。這就是下一節要講的 SDS。

---

## 真正的核心：軟體定義掃描（SDS）

Software-Defined Scanning 是 AEye 從創立那天就在推的架構。它的核心概念在教科書上其實只有一句：

> **掃描鏡的 pointing 不是固定 pattern，是每一幀由軟體決定要打哪裡。**

但這句話展開來，是一整條被重新設計的訊號鏈：

```
[Perception / Planner]
        │  (scan request:
        │   "spend 2M pts on front 30°,
        │    3M pts on side 60°, ...")
        ▼
[Sensor Scheduler]
        │  (converts request to
        │   MEMS/galvo command stream,
        │   with timing/eye-safety check)
        ▼
[Optical / Scan Head]
        │  (fires laser pulses along
        │   the requested trajectory)
        ▼
[Point Cloud with per-point metadata]
        │  (each point carries scan
        │   ID, azimuth/elevation intent,
        │   timestamp)
        ▼
[Perception, back to top]
```

過去的 LiDAR 把 [Sensor Scheduler] 這一層固化成一個規則（例如 raster + repeat），感知工程師看不到、也不能改。Apollo 把這一層開成 API：**perception 或 planner 可以在下一幀開始前把 scan request 遞回去**。

這件事聽起來簡單，實作上有三個非平凡的挑戰：

### 挑戰 1：眼安全的動態強制執行

Class 1 眼安全的核心限制是 **在任何 100mm 的 aperture 上，7mm × 7mm 面積內，1 秒累積能量不能超過閾值**。這個限制在固定 raster 下容易滿足——因為 pattern 是預先設計好的、每個角度被打到的頻率確定。

一旦掃描是動態的，就存在一個 corner case：planner 突然說「盯著那個行人」，掃描器在 100ms 內反覆掃同一個 5° 區域。這時累積能量會超標。

Apollo 的做法是把 eye-safety check 放在 Sensor Scheduler 這一層——所有 scan request 進來後會被 clip 到安全範圍內。這件事在 DriveOS 側需要一個確定性的 latency budget（AEye 給的建議是 <500μs），這對嵌入式軟體是新的約束。

### 挑戰 2：時間戳與掃描 ID 的下游一致性

固定 raster 的好處是：每個點在點雲裡的順序就是掃描的順序，時間戳可以線性插值。動態掃描沒這個保證。Apollo 給每個點附上 (scan_id, intent_azimuth, intent_elevation, timestamp_ns)，讓下游 fusion 可以重建掃描歷史。

**這對 sensor fusion 是好消息**——因為過去 LiDAR + camera 對齊要靠 rolling shutter 校正，動態掃描的 per-point timestamp 直接把這個問題碾平。**但對已經寫好的 fusion pipeline 是壞消息**——因為原本假設點雲順序 = 掃描順序的程式碼要全部重寫。

### 挑戰 3：Perception ↔ Planning 從單向變雙向

傳統堆疊：
```
Sensor → Perception → Planning → Control
```
資訊流是單向的，perception 對 planner 一無所知。

SDS 堆疊：
```
Sensor ⇄ Perception ⇄ Planning → Control
       (scan requests)
```
planner 現在可以說「請重點看 lane 2 有沒有變道」，perception 收到這個 hint 後回頭跟 sensor 說「請把下一幀 40% 的點打在 lane 2」。

這對工程組織的衝擊比對演算法更大——**perception team 跟 planning team 的介面從 output message 變成 bidirectional protocol**。過去這兩個 team 可以獨立迭代，現在必須共同 own scan request 的 schema。

---

## 為什麼跟六月 Aeva FMCW 進 Hyperion 是不同故事

六月我寫過 [FMCW LiDAR 上 Hyperion](fmcw-lidar-hyperion-velocity-perception-2026.md) 那篇。當時的核心論點是：**per-point Doppler velocity 會把 tracking / segmentation / motion compensation 從「多幀差分估速度」的迂迴解法，直接壓成一個 sensor-provided ground truth**。

Apollo 進 Hyperion 是完全不同的故事：

| 面向 | Aeva FMCW（Hyperion 9, 2026-06） | AEye Apollo（Hyperion, 2026-08） |
|---|---|---|
| **測距原理** | FMCW（頻率調變連續波） | ToF（time-of-flight）|
| **關鍵新資訊** | Per-point Doppler velocity | Per-point scan intent + timestamp |
| **重寫的 pipeline 層** | Tracking, segmentation, ego-motion | Sensor scheduling, perception↔planning 介面 |
| **主要挑戰** | Velocity noise model, multipath | Eye-safety enforcement, scan request 協定 |
| **範圍優勢** | 200-300m（FMCW 硬體限制當前）| 1km（1550nm ToF 高功率）|
| **成本模型** | 光子集成電路（PIC）貴、量產難 | 1550nm 光源貴、MEMS 掃描機構 |

**兩家不是在打同一場仗**。Aeva 在贏 tracking + fusion 這一層，AEye 在贏 sensor-planner 閉迴路這一層。從車廠角度來看，這兩個能力是 orthogonal 的——理論上你可以同時裝一顆 FMCW（負責前向近-中距的密集速度感知）加一顆 Apollo（負責長距 attention scanning）。

NVIDIA 同時讓兩家進 Hyperion 這件事本身就是策略訊號：**DRIVE Hyperion 不打算押一種 LiDAR 架構，而是把選擇權留給 OEM**。這跟三年前 NVIDIA 幾乎只推 Luminar 的路線完全不同。

---

## 對 LiDAR / 感知 / 嵌入式工程師的實際意義

以下按角色拆——如果你是這幾個角色之一，Apollo 進 Hyperion 這件事對你日常工作的具體影響：

### 如果你在做 LiDAR 感知演算法

- 過去 3–5 年建立的「固定 raster」直覺要更新。點雲的稀疏度會在同一幀內是**空間變異**的（近密遠稀 → 但「遠稀」的區域可能剛好是 planner 剛請求的高密度熱點）。
- 3D detection backbone 選型會被壓回去重新評估。像 SST（Sparse Sensor Transformer）、DSVT 這類原生 sparse 的方法會回來，因為 dense voxel 方法在動態稀疏下效能不穩。
- 學會讀 per-point metadata（scan_id, intent, timestamp）。這些欄位不是給你 debug 用的，是給你設計 loss function 的——例如可以對「plan 請求高密度但實際 recall 低」的區域加權。

### 如果你在做 embedded / DriveOS 側整合

- 學 AEye SDS 的 API 語意（scan request schema、eye-safety constraints、timing budget）。這在 DriveWorks perception SDK 裡是新的 module。
- 準備好 sensor scheduler 這一層要吃的 CPU/DMA 週期。SDS 的 request → command 轉譯是硬即時的，不能塞到跟 planning 同一個 thread。
- 時間同步的精度要求提高。SDS 依賴 per-point timestamp 精度到 μs 級別，PTP master 的選擇跟 hardware timestamp 支援要重新盤點。

### 如果你在做 sensor fusion / planning

- Perception ↔ Planning 介面要新增一個 outbound channel：scan request。這是一個 schema 決策，而不是實作決策——要跟 perception team 一起 own。
- 過去 planner 從 perception 拿的是「所有目標」，未來要學會「主動請求關注區域」。這是一個 policy 決策：什麼時候用 fixed baseline scan、什麼時候切成 attention scan？這幾乎會變成 planner 的一個內建 state machine。
- 學會為 scan attention 做 A/B 測試。因為 SDS 允許你在 shadow mode 下跑不同 scan strategy，然後比較 downstream tracking / prediction 的 KPI——這是過去固定 raster 時代做不到的實驗。

### 如果你在做 LiDAR SoC / 感測器 IP

- 這件事對你是有點壞消息的——因為 SDS 把 differentiation 從 optics 往上推到 software。硬體側能做 differentiate 的空間變小。
- 但也有機會：eye-safety enforcement 需要硬體 assist（例如 aperture 級的 accumulated energy tracking），這是一個新的 IP 塊。

---

## 這篇文章特意沒寫什麼

寫這種發布文章有一個常見陷阱：把 vendor 的 marketing 直接翻譯成中文再加點技術詞。我特意避開了幾件事，這裡標明白讓讀者知道文章的邊界：

- **沒討論實際部署車型**。Apollo 進 Hyperion 是「參考設計驗證」，不等於任何量產車型今天就會裝。從 Hyperion 席位到實際 SOP 車型還有 12–24 個月的工程週期。
- **沒討論成本**。1550nm × MEMS 的成本目前沒有可靠公開數字。任何說「Apollo 幾百美金就能量產」的論述我都不敢背書。
- **沒討論 vs Luminar Iris / Hesai AT1440 / Innoviz Two 的競爭力**。這些是不同架構的比較，需要一篇單獨的橫評。
- **沒討論 SDS 在資安 / 對抗性攻擊下的表現**。動態掃描 pattern 理論上會給對抗性 spoofing 更多 surface，但目前沒看到公開 red team 報告。

這幾件事都值得單獨挖，之後補上。

---

## 給 Adam 的技術結論（作為感知工程師的行動項）

1. **DriveWorks 6.x 的 perception SDK 要重新看一次**。SDS API 什麼時候會出現在 public reference、有沒有 sample code 值得追蹤。
2. **手邊的 PointPillars/CenterPoint fork 要記錄「假設 raster 均勻」的地方**。動態稀疏來的時候這些是第一波要改的。
3. **關注 AEye + Apptronik / Boston Dynamics 這種跨界搭配的可能性**。SDS 概念不限車載，人形機器人的視覺 attention 也用得上。
4. **保留一個追蹤 clock：Aeva FMCW 跟 AEye SDS 誰先在 SOP 車型上出現**。這會是 2027 感知堆疊往哪個方向偏的最早訊號。

---

## Sources

- [NVIDIA DRIVE Platform Adds Apollo High-Performance Lidar for Autonomous Vehicles – NaturalNews.com](https://www.naturalnews.com/2026-08-08-nvidia-drive-platform-apollo-lidar-autonomous-vehicles.html)
- [Apollo Now Fully Integrated into NVIDIA's Autonomous Driving Platform - AEye](https://www.aeye.ai/press-releases/apollo-now-fully-integrated-into-nvidias-autonomous-driving-platform-paving-the-way-for-significant-growth/)
- [NVIDIA DRIVE platform gains high-performance lidar for autonomous cars - Interesting Engineering](https://interestingengineering.com/innovation/nvidia-drive-platform-apollo-lidar)
- [AEye Apollo Lidar Validated on NVIDIA DRIVE AGX Thor - citybiz](https://www.citybiz.co/article/884689/aeyes-apollo-lidar-validated-on-nvidia-drive-agx-thor/)
