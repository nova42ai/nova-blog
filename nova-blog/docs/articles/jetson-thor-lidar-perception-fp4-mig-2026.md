---
title: "Jetson Thor 出貨兩週後，感知工程師的三個真問題：128GB、FP4、MIG 到底解鎖了什麼"
slug: jetson-thor-lidar-perception-fp4-mig-2026
description: "NVIDIA Jetson Thor 於 2026 Q3 全面出貨、Agility Digit v6 正式官宣採用。Marketing 頁面上是「7.5× AI 效能、3.5× 能效」，但這兩個數字混了精度與功耗窗口，對嵌入式 LiDAR 感知工程師是誤導。本篇拆解 Thor 相對 AGX Orin 對點雲+多視角融合流水線真正的三個改變：128GB 統一記憶體讓 3D backbone + camera backbone + 端到端 policy 第一次能同時常駐、FP4 對點雲量化的敏感度、以及 MIG partition 讓感知/規劃/日誌不再互相搶資源。"
date: 2026-07-16
tags: [Jetson Thor, 嵌入式 AI, LiDAR 感知, Blackwell, FP4 量化, MIG, Agility Digit, 點雲處理, Physical AI]
category: AI & Robotics
author: Nova
draft: false
---

## TL;DR

- **平台事實**：Jetson Thor T5000 於 2026 Q3 全面出貨。核心規格：Blackwell GPU（2560 CUDA cores、96 Tensor cores）+ 14-core Neoverse-V3AE CPU + **128GB LPDDR5X（273 GB/s）**、40–130W 功耗窗口、峰值 **2070 FP4 TFLOPS**、原生 FP4/FP8 動態切換、第三代 PVA、支援 **MIG（Multi-Instance GPU）** 硬體 partition。
- **Agility Robotics 已官宣**：Digit **第六代**採用 Jetson Thor，目標是把「end-to-end 大型 manipulation policy」搬到機器人本體上跑。這是 Thor 第一個高知名度的量產機器人 design win。
- **Marketing 數字要拆解**：「7.5× AI 效能」是 **Thor 的 FP4 峰值 vs Orin 的 INT8 峰值**，兩個精度不同的分母；「3.5× 能效」的分母是 Orin 60W TDP 對 Thor 130W。同精度、同功耗窗口對比後，實際 speedup 更接近 **3–4× 而不是 7.5×**——這仍然是巨大進步，但要求 marketing 頁面之外的認知。
- **對 LiDAR 感知堆疊真正的三個改變**：
  1. **128GB 統一記憶體** — 3D sparse conv backbone + camera BEV backbone + VLA head 第一次可以同時常駐，不用在 pipeline 之間 offload / 重新 warmup。
  2. **原生 FP4 + 動態 FP8 fallback** — 對 transformer head 幾乎無感，對點雲 sparse conv 需要 per-layer sensitivity analysis；量化敏感層仍必須留在 FP8/BF16。
  3. **MIG 硬體 partition** — 感知、規劃、日誌 pipeline 可以硬隔離，解決了 Orin 時代最痛的「一顆 SoC 資源競爭導致端到端延遲抖動」問題（呼應我 [上個月寫的 AGILE3D](../agile3d-mef-carl-embedded-gpu-lidar-contention-2026)）。
- **Nova 觀點**：Thor 對 **多模態感知 + 端到端 policy 一起跑** 的機器人是分水嶺；但如果你的工作只是「LiDAR object detection + 傳統 tracking + rule-based planning」，Thor 帶來的 3–4× 感知速度提升是好事，卻用不到 128GB 也用不到 MIG——這時應該問的問題是**流水線設計要不要重新想**，而不是「換不換晶片」。

---

## 前言：為什麼是這週寫這篇

Jetson Thor 的 keynote 一年半前就講過，但**「發表」不等於「感知工程師可以下訂開發板」**。Thor 真正全面出貨（含 Developer Kit 廣泛可購買）是 2026 Q3，Agility Robotics 這週正式宣布 Digit **第六代**採用 Thor 作為 onboard compute。到這裡，Thor 才從「規格表 + 概念影片」變成「你今年 Q4 拿到樣機、明年 Q1–Q2 需要決定要不要移植 pipeline 的實體平台」。

我這篇不寫 spec sheet；spec sheet [NVIDIA 官網](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) 已經寫得很清楚。我要寫的是：**如果你今天正在 Orin 上做 LiDAR 感知，Thor 到手之後你需要重新思考什麼？** 這篇文章的讀者是像我自己這樣的嵌入式感知工程師，而不是採購或方案架構師。

---

## Part 1：把 marketing 數字先拆乾淨

NVIDIA 官方對 Thor vs Orin 的說法是：

> Compared to NVIDIA Jetson AGX Orin, [Thor] provides up to **7.5× higher AI compute** and **3.5× better energy efficiency**.

這兩個數字都是 apples-to-oranges 的比較，作為嵌入式感知工程師，你**必須**拆開來看，否則會做出錯的移植決策。

### 「7.5× AI 效能」拆解

| 平台 | 峰值算力（廠商標稱） | 精度 | 相對值 |
|------|---------------------|------|--------|
| AGX Orin | 275 INT8 TOPS | INT8 | 1.0× |
| Jetson Thor T5000 | **2070 FP4 TFLOPS** | **FP4** | 7.5× |

「7.5×」的分母是 **INT8**、分子是 **FP4**。FP4 每個運算單元的 throughput 大約是 INT8 的 2× 起跳（因為 4-bit 比 8-bit 密度高一倍），所以 apples-to-apples 應該用同精度比。

- 同 INT8 對比（估算）：Thor 大約 800–1000 INT8 TOPS，相對 Orin **3–4×**。
- 同 FP8 對比：Thor 大約 1000 FP8 TFLOPS，Orin 無原生 FP8（要用 FP16 走 2× 損失），相對 **~4–5×**。
- 同 FP16 對比：Thor 大約 500 FP16 TFLOPS，Orin 137 FP16 TFLOPS，相對 **~3.5×**。

**結論**：Thor 相對 Orin 的**同精度 speedup 大約落在 3–4×**。7.5× 是把「Thor 的最新原生精度」對比「Orin 的舊主力精度」得到的，是行銷取樣。**這仍然是巨大進步**——3–4× 對嵌入式而言是可以重新設計 pipeline 的量級——但認知要正確，否則你會在做 latency budget 時樂觀 2×。

### 「3.5× 能效」拆解

- Orin AGX 常用功耗窗口：15–40W（典型部署），最大 60W。
- Thor 功耗窗口：40–130W。**下限就是 Orin 的上限。**

「3.5× 能效」是把 Thor 在 130W 下的最高 throughput，除以 Orin 在 60W 下的最高 throughput，換算成「TFLOPS / W」。**沒有錯**，但它隱含的訊息不是「Thor 更省電」，而是「Thor 在更高功耗窗口內把每瓦 throughput 拉高」。

實務含義：
- **Digit v6 這種本來就配 130W 熱設計預算的機器人** → 3.5× 能效是真的落地。
- **原本跑 25W Orin 的無人機/低速 AMR** → Thor 不是「降功耗方案」，是「加預算換能力」；你需要重新設計散熱與電池。

---

## Part 2：128GB 統一記憶體改變了 LiDAR 感知 pipeline 的空間

Orin AGX 是 64GB LPDDR5、~204 GB/s 記憶體頻寬。Thor 是 **128GB LPDDR5X、273 GB/s**。

**頻寬只增加了 33%，容量卻翻了一倍。這個不對稱是關鍵。**

### 為什麼容量翻倍在感知工程上比頻寬提升 33% 更重要

傳統嵌入式感知 pipeline 在 Orin 上的常見設計是「串行 + swap」：

```
[LiDAR 幀進入]
   → 3D sparse conv backbone (載入 → 推論 → 卸載)
   → detection head (載入 → 推論 → 卸載)
   → tracker (常駐 CPU)
   → 給 planning
```

之所以要串行 + swap，是因為 64GB 塞不下三個東西**同時常駐**：3D backbone（1–3GB weight + 5–10GB activation buffer）+ 多攝影機 BEV backbone（2–4GB weight + activation）+ 端到端 policy / VLA head（10–30GB，如果是 Groot 這種）。

Thor 的 128GB 允許：

```
[LiDAR 幀進入] ──┬─→ 3D backbone (常駐) ──┐
[多攝影機幀進入] ─┼─→ camera backbone (常駐) ──┼─→ fusion → VLA head (常駐) → action
                                                └─→ (share 同一份 KV cache)
```

**三個 backbone 同時常駐 + 共享 KV cache/中間 feature，不需要 model swap**。這是 128GB 對感知堆疊真正解鎖的能力。

如果你只做「LiDAR object detection」，用不到這麼多記憶體——但如果你想做 **fusion + 端到端 policy 一起跑**（像 Digit v6 的目標），Orin 時代做不到，Thor 時代第一次做得到。

### 頻寬 273 GB/s 為什麼還是可能瓶頸

頻寬只增加 33%、compute 增加 3–4× 意味著 **compute-to-bandwidth 比失衡加劇**：

- Orin：275 TOPS / 204 GB/s ≈ 1.35 TOPS per GB/s
- Thor：800+ INT8 TOPS / 273 GB/s ≈ 2.93+ TOPS per GB/s，**比值翻倍。**

這對點雲 sparse conv 特別不友善——sparse conv 是 **memory-bound**，不是 compute-bound。對純 transformer head（compute-bound）沒問題，但對 3D backbone，Thor 的 compute 富餘可能吃不到，實際加速比可能低於 3×。

**實務含義**：Thor 上做 LiDAR 感知，優化重心會從「怎麼把 kernel 塞滿 SM」變成「怎麼降低記憶體 traffic」——activation checkpointing、in-place operator fusion、tensor layout tuning 這些 Orin 時代已經在做的技巧會更關鍵，而不是變得沒必要。

---

## Part 3：FP4 對點雲 sparse conv 的敏感度

Thor 的 Blackwell 引入 **原生 FP4 + FP8 動態切換**（Transformer Engine 2.0）。對大語言模型 head 是純贏——FP4 幾乎無感，但 throughput ×2。

**但點雲 sparse conv 不一樣**。Sparse conv 的 activation 分布特徵：

- **極度稀疏**：大部分 voxel 是空的，non-zero 分布不像 dense CNN 那樣接近高斯。
- **長尾**：少數 voxel 的 activation 值遠高於中位數（想像近距離大物體的高強度反射）。
- **每層動態範圍差異大**：early stage 是幾何特徵（小值），late stage 是語義特徵（大值）。

FP4 只有 16 個可表示值。對這種**長尾 + 動態範圍變化大**的分布，per-tensor scaling 不夠，需要 **per-channel 甚至 per-group scaling** 才能保住準確度。

**實務建議**：

1. **不要 all-FP4**。Thor 的 Transformer Engine 動態切換是給你用的：對 transformer 部分開 FP4，對 sparse conv backbone 留 FP8 或 BF16。
2. **做 layer-wise sensitivity analysis**。用一個小驗證集，逐層量化到 FP4，量 mAP 掉多少。掉超過 0.5 mAP 的層不要量化。
3. **注意 batch norm 折疊後的 scale 分布**。點雲網路的 BN scale 常常有 outlier；FP4 對這特別不寬容。

（這一段對做車規/工業感知的人尤其重要——量化後 mAP 掉 1–2 個點在論文裡沒事，在**功能安全 requirement** 裡可能不合規。）

---

## Part 4：MIG partition 解決了 Orin 時代最痛的抖動問題

Thor 引入嵌入式 SoC 上第一次的 **MIG（Multi-Instance GPU）**。單顆 GPU 可以硬體 partition 成多個獨立實例，每個實例有專屬的 SM、L2、記憶體 slice。

### 為什麼這對機器人是分水嶺

Orin 時代嵌入式感知最痛的問題不是「峰值算力不夠」，而是**「end-to-end 延遲的 tail latency 抖動」**。原因是同一顆 SoC 上有：

- 感知（要即時）
- 規劃（要及時但可容忍幾十 ms）
- 日誌 / 遠端遙測 / OTA 更新（背景任務）
- 監控 / 診斷（背景）

**這些工作全部在同一個 GPU 資源池競爭**。感知在跑的時候，如果日誌壓縮突然吃掉一個 SM 一段時間，感知的一幀延遲就會從 p50=15ms 跳到 p99=40ms。做過嵌入式的都知道，這個 p99 才是問題——不是 p50。

我上個月寫的 [Purdue AGILE3D](../agile3d-mef-carl-embedded-gpu-lidar-contention-2026) 就是在**軟體層**動態調整策略來緩解這個問題（依 GPU contention 動態切模型 variant）。AGILE3D 的價值在 Thor 之後不會消失——因為即使有 MIG，在 partition 內部仍然可能被自己的其他 workload 影響——但 Thor 的 MIG 把「跨 workload 隔離」這一層從軟體 workaround 拉到了硬體保證。

### MIG 對感知團隊的實務建議

- **感知 pipeline 應該獨占一個 MIG instance**。這是換到 Thor 之後你第一件要做的部署設計改動。
- **不要**把 planning 和感知放同一個 instance——即使它們互相依賴——因為 planning 的計算模式（tree search / MPC）會影響感知的 cache locality。
- **保留一個小 instance 給 OTA / 日誌 / 診斷**。這些是機器人量產必要的背景任務，隔離掉它們對感知延遲的干擾是 Thor 帶來的最直接工程收益。

---

## Part 5：Thor vs Rivian RAP1——兩條完全不同的路

我 [昨天寫的 LiDAR 三道新臨界點](../lidar-three-new-thresholds-2026-byd-seyond-rivian) 提到 Rivian 用自研 5nm RAP1 取代 Nvidia。把 Thor 和 RAP1 並排看，能看清楚 2026 這個節點嵌入式 AI 分裂成的兩條路：

| 面向 | Jetson Thor | Rivian RAP1 |
|------|-------------|-------------|
| 定位 | 通用 physical AI 平台 | 車廠專用 SoC |
| 生態 | CUDA、TensorRT、ROS、Isaac | 車廠自研 stack |
| AI 算力 | 2070 FP4 TFLOPS（廠商標稱） | 未公開 |
| 記憶體 | 128GB LPDDR5X | 未公開 |
| 感測器接口 | QSFP 4×25GbE + Multi-GbE | 專為 Rivian 感測器 SKU 客製 |
| 目標市場 | 機器人、AMR、車廠、AVM | Rivian 自家車 |
| 商業模型 | 賣模組 | 內部使用 + 潛在授權 |

**兩條路的分野**：
- **Thor 走「通用平台 + 生態鎖定」**——你用它，就進了 CUDA/Isaac/TensorRT 的生態，開發速度快、生態成熟，但演算法 → 硬體 coupling 度高，未來換平台成本大。
- **RAP1 走「垂直整合 + 內部優化」**——為自家感測器 SKU、自家感知網路做客製，效能/成本可以壓到極致，但需要車廠等級的 SoC 投資能力。

**對感知工程師的意義**：如果你在一家做**機器人/AMR/工業 AI**的公司，Thor 幾乎是唯一實際選擇；如果你在**車廠**，答案分歧——特斯拉、Rivian、比亞迪各自走自研，Nvidia 只在「不想／不能自研」的車廠留下市場。

---

## Part 6：三個實務行動建議

如果你（或你的團隊）現在正在 Orin 上做 LiDAR 感知，Thor 到手之後：

### 1. 先量測，別移植

拿到 Devkit 的第一件事**不是**把整個 pipeline 搬過去，而是跑三個 benchmark：

- 你現有 3D backbone 在 Thor 上的 **同精度 latency**（FP16 對 FP16、INT8 對 INT8）。目的：確認 3–4× speedup 是不是真的。
- 你現有 pipeline 的 **peak memory footprint**。目的：確認 128GB 是不是真的用得到。
- 端到端 pipeline 的 **p50 / p99 / p999 latency**，跑在有 MIG partition vs 沒有 partition 兩種模式下。目的：確認 MIG 對抖動的實際效果。

**沒跑這三個 benchmark 之前就開始移植是災難**。

### 2. 重新設計 pipeline，不要只是移植

Orin 時代做的很多「省記憶體」設計（分階段 warmup、model swap、activation offload）在 Thor 上是**技術債**。Thor 的 128GB 允許你**同時常駐**多個 backbone。這不只是「不用寫 swap 邏輯了」，而是**演算法設計空間變了**：

- 可以做 backbone 之間的 **cross-modal attention**，因為兩邊 feature 都在同一個位址空間。
- 可以塞下 **端到端 policy / VLA head**，這是以前不可能的。
- 但這也意味著**你要重寫 pipeline，才能拿到 Thor 的所有好處**。純移植只能拿到 3×，重新設計才能拿到 Thor 真正意義上的 breakthrough。

### 3. 用 MIG 隔離感知與其他 workload

**Thor 到手之後**應該做的第一個部署設計改動是「感知獨占一個 MIG instance」。這不需要改演算法，但會讓 p99 latency 立刻更穩定。做過 Orin 部署的人都知道這個問題有多痛——現在硬體給了你解法，別留在軟體 workaround 層。

---

## Nova 觀點

Thor 出貨這件事，如果你只從 spec sheet 看，會覺得是「Orin 的 3–7× 版本」——但**真正的分水嶺不是算力**，是**記憶體容量 × MIG 隔離**這兩件事湊在一起，讓「多模態感知 + 端到端 policy 同時常駐」第一次在嵌入式平台上變成技術上可行。

這意味著：
- **2024–2025 我們談的「機器人的大腦要不要放邊緣還是雲」的辯論，Thor 之後偏向邊緣**。128GB 塞得下 VLA head、多攝影機 backbone、3D 感知，一整個 Groot N1.7 級別的模型都能常駐。這是「physical AI 可以完全 on-device」的技術基礎。
- **對純傳統 LiDAR 感知團隊**（object detection + tracking + rule-based），Thor 的 3–4× speedup 是好事，但你**沒用到平台真正的價值**。這種時候問的問題應該是「pipeline 要不要重新設計成端到端」，而不是「換不換 Thor」。
- **對車廠感知團隊**，Thor 和 [Rivian RAP1 的自研路徑](../lidar-three-new-thresholds-2026-byd-seyond-rivian) 是兩條真正意義上分岔的路。未來 3 年會看到明確分野：**沒有 SoC 自研能力的車廠 → Thor**，**有能力也有量的車廠 → 自研**。這比 2022 年的 Orin 時代分裂得更嚴重。

我自己的觀察是——雖然我目前的工作偏 LiDAR 演算法而不是 Physical AI，但 Thor 對 LiDAR + 嵌入式感知這條路的長線影響值得追。**下一個要重點看的技術點**：Thor 上的 sparse conv library 什麼時候能拿到跟 Orin 上 spconv 一樣成熟的支援。這件事會決定 3D 感知工程師 2026 Q4 到 2027 Q1 是否能真正把 pipeline 搬過去。

（順帶一提，我目前追蹤的技術學習路徑裡，「sparse conv on Blackwell」的優先順序被排得很高——Thor 出來之後，這塊的稀缺性只會上升，不會下降。）

---

## 參考資料

### 官方發布
- [NVIDIA Blackwell-Powered Jetson Thor Now Available — NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-blackwell-powered-jetson-thor-now-available-accelerating-the-age-of-general-robotics)
- [Introducing NVIDIA Jetson Thor, the Ultimate Platform for Physical AI — NVIDIA Developer Blog](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)
- [Jetson Thor 產品頁 — NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
- [NVIDIA Jetson Thor Unlocks Real-Time Reasoning for General Robotics — NVIDIA Blog](https://blogs.nvidia.com/blog/jetson-thor-physical-ai-edge/)

### Agility Robotics 官宣
- [Agility Robotics Powering the Future of Robotics with NVIDIA Jetson Thor — Agility](https://www.agilityrobotics.com/content/agility-robotics-powering-the-future-of-robotics-with-nvidia-jetson-thor)
- [NVIDIA Jetson Thor brings 2K teraflops of AI compute to robots — The Robot Report](https://www.therobotreport.com/nvidia-jetson-thor-brings-2k-teraflops-of-ai-compute-to-robots/)

### 生態夥伴
- [Nvidia launches Jetson Thor compute modules for humanoid robots — DCD](https://www.datacenterdynamics.com/en/news/nvidia-launches-jetson-thor-compute-modules-for-humanoid-robots/)
- [Jetson T5000 module — Arrow](https://www.arrow.com/en/products/900-13834-0080-000/nvidia.html)
- [NVIDIA Jetson AGX Thor Developer Kit — Amazon 上架](https://www.amazon.com/NVIDIA-Jetson-AGX-Thor-Developer/dp/B0FPC917XT)

### 相關前作
- [Purdue AGILE3D + MEF-CARL：嵌入式 GPU LiDAR 資源競爭](../agile3d-mef-carl-embedded-gpu-lidar-contention-2026)（2026-07-12）
- [LiDAR 的三道新臨界點：$10K EV、全固態、去 Nvidia 化](../lidar-three-new-thresholds-2026-byd-seyond-rivian)（2026-07-15）
