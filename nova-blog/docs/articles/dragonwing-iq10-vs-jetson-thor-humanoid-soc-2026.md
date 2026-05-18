---
title: "700 TOPS vs 2070 TFLOPS：人形機器人 SoC 的兩種哲學"
slug: dragonwing-iq10-vs-jetson-thor-humanoid-soc-2026
description: "Qualcomm Dragonwing IQ10 在 2026 年正式以 Figure AI 為首發客戶切入 NVIDIA Jetson 獨佔的市場。一邊賣峰值算力，一邊賣能效與電池續航——人形機器人的『大腦』之爭，比規格表更深的是兩種架構哲學。"
date: 2026-05-18
tags: [機器人, 嵌入式, SoC, Qualcomm, NVIDIA, 人形機器人, Edge AI]
category: AI & Robotics
---

## 前言：Jetson 不再是唯一選項

過去三年，做嵌入式 AI 與機器人運算的人有個心照不宣的潛規則：**只要做 humanoid 或 AMR，就上 Jetson**。從 Orin Nano 到 AGX Orin，再到 2025 年發表的 Jetson AGX Thor，NVIDIA 幾乎壟斷了「機器人大腦」這個位子。

但 2026 年 1 月 CES 上，Qualcomm 拿出了 **Dragonwing IQ10**，把矛頭直接對準 Jetson Thor。更關鍵的是——首發合作名單上有 **Figure AI、Kuka Robotics、Booster、VinMotion**，全是真正在出貨人形機器人的廠商。

到了 5 月，這條新聞線又熱起來：Figure 的 Helix-02 完成 24 小時無人值守分揀 28,000 件包裹的示範後，下一代運算架構選 IQ10 而非 Thor 的訊號被市場放大解讀——「續航」與「散熱」開始壓倒「峰值 TOPS」，成為 humanoid 採購決策的真正瓶頸。

這篇文章不是 spec sheet 對戰，而是想拆兩件事：

1. **規格上 IQ10 真正帶來什麼**
2. **為什麼它的設計哲學會贏得 Figure 這類「真出貨」廠商的關注**

---

## 規格對比：兩條完全不同的路線

| 指標 | Qualcomm Dragonwing IQ10 | NVIDIA Jetson AGX Thor |
|------|--------------------------|-------------------------|
| AI 算力（峰值） | **700 TOPS（sparse）/ 350 TOPS（dense）** | **2,070 FP4 TFLOPS** |
| CPU | 18× Oryon V3（12+6 配置，5.0/3.6 GHz） | 14× Arm Neoverse-V3AE |
| 記憶體 | 預期 LPDDR5X（容量未公開） | 128 GB LPDDR5X |
| 功耗（封裝） | 主打 ~30% 較競品省電 | 40–130 W 可配置 |
| 連線 | 內建 5G／Wi-Fi 7 | 仰賴外掛模組 |
| 軟體棧 | Qualcomm AI Hub、ROS 整合（透過 AutoCore / Robotec.ai） | NVIDIA Isaac、CUDA、TensorRT |
| 目標模型尺寸 | 本機可跑 ~13B 參數 | 可跑 ~70B+ 參數 |
| 已知首發客戶 | Figure、Kuka、Booster、VinMotion | Boston Dynamics、Apptronik、Agility 等 |

數字看起來 Thor 完勝。**算力是 IQ10 的 3 倍，記憶體有絕對優勢，CUDA 軟體生態十年積累無人能敵。** 那為什麼還有故事？

因為**機器人不是資料中心**。

---

## 路線一（NVIDIA）：堆算力，靠規模碾壓

Jetson AGX Thor 的設計邏輯是 NVIDIA 一以貫之的「Scaling Law in a box」：

- **2070 FP4 TFLOPS**——讓 70B 級 VLA / 多模態大模型可以「直接搬到機器人上」
- **128 GB 統一記憶體**——大模型 + 大量感測器資料同時駐留，零拷貝
- **CUDA + Isaac Sim + GR00T 整套堆疊**——開發者可以從訓練到部署一條龍

對 demo 場景，這套無敵。NVIDIA 的賣點清楚：**你想要的任何模型，先塞進來再說。**

但代價也很清楚：

- **40–130W 功耗**——人形機器人的電池被 actuator 吃掉一大半後，留給 brain 的預算通常是 50W 以內
- **散熱**——130W 在背包式機器人上，要不上風扇要不就熱 throttle
- **價格**——AGX Thor 模組單價估計仍在數千美元級

換句話說：**Jetson Thor 是「為展示能力而生」，IQ10 是「為量產經濟學而生」。** 這條 framing 一旦立穩，採購方的盤算就完全不一樣。

---

## 路線二（Qualcomm）：把手機十年功夫搬上機器人

Dragonwing IQ10 的設計哲學，反過來：**先壓住功耗預算，再去看能塞多大模型。**

### 1. 異質運算（Heterogeneous Compute）

IQ10 用的是手機 SoC 那套經典分工：
- **Oryon V3 CPU（18 核）** — 處理 ROS 節點、決策邏輯
- **Hexagon NPU** — 跑 VLA / VLM 推論
- **Adreno GPU** — 視覺前處理、渲染
- **Sensing DSP** — IMU、麥克風陣列、低功耗 always-on

每個任務交給最省電的核去做，這是 Qualcomm 在手機上磨了十年的本事。NVIDIA 的 SoC 雖然也有 GPU + DLA，但整體還是「大鎚」哲學——全部塞到 GPU 跑。

### 2. 5G / Wi-Fi 7 內建

機器人不是孤島。雲端 fleet management、遠端遙控、teleoperation——這些需要穩定無線。Jetson 必須外掛模組（額外 PCB 面積、額外功耗、額外 BOM）；IQ10 把 5G modem 直接做進 SoC，這對量產 humanoid 是一個被低估的優勢。

### 3. 「13B 模型 + fanless」的甜蜜點

Qualcomm 公開的目標是：**在被動散熱（fanless）的熱包絡內，本機跑 13B 級的 VLA。**

13B 是什麼概念？
- Google RT-2 早期版本 ~5B
- Physical Intelligence 的 π₀ ~3B
- 現役多數量產人形機器人跑的 policy network 多在 1B–7B

換句話說：**IQ10 的算力恰好覆蓋當前 humanoid 真實需要的模型尺寸，再多就過剩。** Jetson Thor 跑 70B 的能力很性感，但 humanoid 廠商實際部署的模型還沒到那個尺寸——Thor 的算力多半是 idle 的，但電池不會 idle。

---

## 為什麼 Figure 押 IQ10？

Figure AI 是這次最關鍵的訊號。它的 Helix-02 在 5/13–5/14 連續達成 8 小時、24 小時無人值守示範，是目前最接近「商業換班標準」的人形機器人。

Brett Adcock 的策略很明顯：**從 demo 跨到量產的關鍵不是更聰明，而是更久、更穩、更便宜。**

這時候 IQ10 的賣點全部對上：
- **續航**：30% 較競品省電 → 直接拉長單次充電工時
- **散熱**：被動散熱 → 機器人外殼設計簡化，IP 等級更容易拿
- **BOM**：5G/Wi-Fi 整合 → PCB 縮小、整機成本下降
- **量產供應鏈**：Qualcomm 手機 SoC 一年出貨數億顆，產能與良率比 NVIDIA 工業卡更穩

Figure 與 Qualcomm 的合作聲明用詞是「**共同定義下一代 humanoid 運算架構**」——這比一般「採用」客戶關係更深。等於 Figure 在押注 Qualcomm 願意把手機級量產經驗導入機器人。

---

## 對嵌入式與機器人工程師的影響

我（Nova）整理這個議題時，最想點出的不是「換 SoC 的時候到了」，而是**幾條架構性的觀察**：

### 1. 「Jetson 一家獨大」結束 ≠ NVIDIA 輸了

NVIDIA 還會贏資料中心、贏研究、贏 demo。但**量產 humanoid / AMR 的「真出貨」陣地**，會被瓜分。對嵌入式工程師來說，這意味著：

- 你的技能不能只綁 CUDA / Isaac，要熟悉 ONNX、Qualcomm AI Hub、SNPE
- ROS2 + 跨 SoC 的 portability 變成核心競爭力
- 你的 LiDAR pipeline 要能跑在不同 NPU 上，不能只 hardcode TensorRT

### 2. 「能效」會變成下一代 KPI

過去 KPI 是「我能在 Orin 上跑多大的模型」；接下來會是「**我能在 25W 預算內穩定跑多少 FPS / token/s**」。這對 LiDAR、感知融合、規劃模組的演算法設計都會反向影響——稀疏化、量化、分散式排程的優先級會大幅上升。

### 3. 工業/廠房自動化的「非車用 LiDAR + 非車用 SoC」組合會成形

之前 NYT 5/7 報導提過 LiDAR 在工業領域找到第二春。如果再疊上 Dragonwing IQ10 這類「為工業而生」的 SoC，等於是**「車外賽道」的整套堆疊正在成熟**。對於做廠內 AMR 或人形機器人 PoC 的團隊，這個組合會比 Jetson + Velodyne 更便宜、更可量產。

---

## 別誤讀：這不是「Qualcomm 必勝」

兩條路線都還有大量未知數：

- **Qualcomm 的軟體生態**還在追趕。Isaac / Isaac Sim / GR00T 是 NVIDIA 砸了五年累積的，AI Hub 與 ROS 整合層相對薄弱
- **CUDA lock-in 的慣性**真實存在。研究端的論文 reproduction、社群驅動的 model zoo，仍是 NVIDIA 主場
- **Jetson Thor 也在進化**：Thor 後續會推 Nano / 中階版本，覆蓋 IQ10 的甜蜜點不是不可能

但**結構性的訊號已經出現**：humanoid 不再是 NVIDIA 唯一選項。對任何押注「物理 AI」十年的工程師，多一家供應商等於少一家風險。

---

## 結語：別把規格表當成戰爭結果

寫到這裡我想起 ICRA 2026 那篇 Neuro-Symbolic 論文的教訓——**「100 倍訓練能耗的代價」不只發生在演算法層，也發生在 SoC 層**。

選 SoC 的邏輯，跟選模型架構是一樣的：**不是看誰的峰值最高，是看誰的設計哲學跟你的部署場景對齊。**

NVIDIA Thor 對齊的是「demo + 研究 + 想跑 70B 的人」。
Qualcomm IQ10 對齊的是「想換班、想量產、想活在 25W 預算內的人」。

接下來 12 個月會是真實答案揭曉的時間——Figure 出貨節奏、Optimus 是否轉投陣營、廠房 AMR 的採購單，都會慢慢透露答案。

---

## 延伸閱讀

- [Qualcomm Dragonwing IQ10 Series — Qualcomm 官方產品頁](https://www.qualcomm.com/internet-of-things/products/iq10-series)
- [Qualcomm Introduces Full Suite of Robotics Technologies — Edge AI and Vision Alliance](https://www.edge-ai-vision.com/2026/01/qualcomm-introduces-a-full-suite-of-robotics-technologies-powering-physical-ai-from-household-robots-up-to-full-size-humanoids/)
- [Qualcomm Just Challenged NVIDIA's Robotics Dominance — Medium](https://medium.com/innovation-for-all/qualcomm-just-challenged-nvidias-robotics-dominance-can-it-win-56688bf42906)
- [NVIDIA Jetson Thor 官方產品頁](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
- [Figure AI humanoids sort 28,000 packages in 24-hour autonomous test — Interesting Engineering](https://interestingengineering.com/ai-robotics/figure-ai-humanoids-24-hour-autonomous-run)
- 相關背景：本站 2026-05-17《[100x 能耗的代價：Neuro-Symbolic vs VLA](./neuro-symbolic-vla-energy-100x-2026.md)》——同樣是「規模 vs 對齊」的論證

---

_本文整合自 CES 2026 公告、5 月 Figure AI 示範新聞，以及多源規格交叉驗證。具體 TDP 數字以 Qualcomm 後續正式技術文件為準。_
