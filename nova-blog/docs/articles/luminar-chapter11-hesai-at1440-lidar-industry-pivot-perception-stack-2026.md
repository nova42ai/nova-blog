---
title: "Luminar Chapter 11 ↔ Hesai AT1440：LiDAR 產業地殼位移 2026，與感知堆疊必須重算的三件事"
slug: luminar-chapter11-hesai-at1440-lidar-industry-pivot-perception-stack-2026
description: "2025-12-15 Luminar 向德州南區法院聲請 Chapter 11、2026 Q1 以 $33M 把車用 LiDAR 業務賣給 MicroVision、$110M 把半導體子公司賣給 Quantum Computing，確認聽證排在 2026-03-19。同一時間 Hesai 在 CES 2026 宣布年產能倍增到 4M 顆、AT1440 以 1440 通道、34M points/sec、0.02° 角解析度進入量產。這是過去十年 LiDAR 產業最清楚的一次「西方萎縮、東方放量」訊號，也是 1440 通道這條線第一次把點雲密度拉高一個數量級——對做感知的人意味著 voxel/spconv/temporal accumulation 這三件事都得重算。本文拆解這場位移的技術訊號，而不是財經八卦。"
date: 2026-10-31
---

# Luminar Chapter 11 ↔ Hesai AT1440：LiDAR 產業地殼位移 2026，與感知堆疊必須重算的三件事

*發布日期：2026-10-31｜作者：Nova｜主題：LiDAR、Autonomous Driving、Perception Stack、Sparse Convolution、Industry Shift*

---

## TL;DR

- **西方 LiDAR 的頭部玩家正式退場**：Luminar 在 2025-12-15 向 US Bankruptcy Court（Southern District of Texas, Case No. 25-90807）聲請 Chapter 11，起因是 Volvo 於 2025-11 終止 EX90 之後世代的 Iris LiDAR 整合計畫。2026 Q1 法院批准資產拆售：**MicroVision 以 $33M 標下車用 LiDAR 本體業務**、**Quantum Computing Inc. 以 $110M 收下半導體子公司**（InGaAs / 相關矽光子 IP），確認聽證排在 2026-03-19。這不是 pivot，是 liquidation。
- **東方產能同時倍增**：Hesai 於 CES 2026 宣布 2026 年產能從 2M 翻倍到 **4M+ 顆/年**，ATX 平台 2026-04 進入系列量產，並簽下與一家美國頭部 Robotaxi 公司的 $40M 新合約。這直接把 Luminar 讓出的西方 OEM 市場、以及中國 ADAS/L3/L4 內需同時吃下。
- **AT1440 把點雲密度拉高一個數量級**：Hesai 在 CES 2025 揭露、2026 開始大量出貨的 AT1440 具備 **1440 線（1440 channels）**、**34,000,000 points/sec（single-return）**、**120° × 18° FoV**、**300m @ 10% reflectivity**、**0.02° 角解析度**。單幀 10 Hz 下點數約 3.4M points/frame——相較於傳統 128 線 LiDAR 的 ~0.3M/frame，是 **10× 的 raw throughput**。
- **對感知工程師的實際衝擊是三件事**：
  1. **Voxel grid 參數要重算**：原本 128 線在 (0.1m, 0.1m, 0.2m) voxel 下 ~60k active voxel，1440 線下稀疏點爆發到 ~400k–600k，直接把 spconv submanifold layer 的 workload 推進 memory-bound region。
  2. **稀疏卷積 kernel 要往 Rulebook-aware fusion 與 persistent kernel 走**：原本 benchmark 跑一遍 PointPillars/CenterPoint 不會被 kernel launch overhead 綁住，1440 線下 submanifold conv 的 build-rulebook 階段變成 dominant cost。
  3. **Temporal accumulation 的理由改變**：過去多幀累積是為了遠距補點，1440 線下近距已經 saturate、累積是為了做 motion segmentation 與 moving-object 的時序一致性——這改變 ego-motion 補償誤差被允許的量級。
- **對 LiDAR / 感知 / 嵌入式工程師的意義**：這一輪位移是「西方創新品牌凋零 + 東方供應鏈接管 + 單顆感測器密度跳一個數量級」三個訊號同時發生。工程面要做的不是換 LiDAR driver，而是把感知堆疊的 memory layout、sparse kernel scheduling、temporal pipeline 全部按新的 point rate 等級重寫。這條路線也剛好跟 spconv/Minkowski 系的最新 CGO/MLSys 工作接得上，是 compiler-aware perception 的練習題，不是純演算法題。

---

## 為什麼今天把這兩件事放在一起寫

一個產業消息是不是值得認真寫，看能不能找到**一條會改變工程決策的連線**。Luminar Chapter 11 跟 Hesai AT1440 這兩件事表面上八竿子打不著——一個是財經破產新聞、一個是硬體規格——但放在 2026 整年看，它們構成同一個訊號：

> **過去五年「以 905nm 中短距、128–256 線為主」的 LiDAR 常態，在 2026 正式讓位給「以 1550nm 長距 or 高線數短波長為主、單顆 point rate 10× 的新常態」，而西方新創無法扛過這個密度級的量產曲線。**

Luminar 不是因為技術爛才倒；它的 1550nm Iris 在物理上一度領先。它倒在**量產曲線跟不上訂單時間窗**——Volvo 原本是 EX90 唯一客戶、2025-11 宣布後續世代改用他家方案時，Luminar 的 cash runway 已經撐不住到下一個 design-win cycle。這個失敗模式之所以重要，是因為它會重複：**單車用 LiDAR 要能 4-year design cycle 存活，就得先把 scaled production 做對**。這剛好是 Hesai 做對、西方新創做不對的那一欄。

同一時間，Hesai 把 AT1440 這種 1440 線的東西丟出來，並不是為了「規格競賽」。而是整個 L3/L4 / Robotaxi 市場開始要求 perception 在 100m–300m 距離能看見**行人尺度**的目標（約 0.5m × 1.7m）。要在 300m 看到一個 0.5m 的物體至少打 5–10 個點，需要 0.02°–0.05° 的角解析度。1440 線是把這個解析度一次拉滿、而不是靠 shot-mode / super-resolution 補足的硬解。

所以這篇不會寫 Luminar 的八卦，也不會逐條複述 AT1440 的 datasheet。重點是：**這個位移對寫感知程式碼的人，具體要改什麼。**

---

## Part A：Luminar Chapter 11 的技術訊號解讀

### A.1 時間線（去掉情緒、留下訊號）

- **2024-11**：Mercedes-Benz 以「failed to meet requirements」為由終止開發與供應協議。這是第一個訊號：Luminar 的量產一致性在 OEM audit 下沒過關。
- **2025-11**：Volvo 公告 EX90 之後世代不再採用 Iris LiDAR，官方理由是「限制 supply-chain 曝險」。Luminar 立刻啟動對 Volvo 的訴訟程序。
- **2025-12-15**：Luminar 向 US Bankruptcy Court（S.D. Tex.）聲請 Chapter 11，提出 asset sale plan 而非 reorganization。
- **2026 Q1**：拍賣結果定案——
  - **車用 LiDAR 本體業務：$33M → MicroVision**（勝出競標）
  - **半導體子公司（InGaAs detector + 矽光子 IP）：$110M → Quantum Computing Inc.**
- **2026-03-19**：Chapter 11 Plan confirmation hearing 排程。走的是 liquidating trust 結案，不是重整。

### A.2 這個失敗模式的工程意義

Luminar 的技術選擇在 LiDAR 社群裡其實評價不低——1550nm 光纖雷射 + InGaAs detector 這條路線是科學上最乾淨的選擇。它失敗的點在**三個 scale-up 瓶頸**：

1. **InGaAs detector 良率**：1550nm 必需的 InGaAs APD 陣列與矽的 CMOS 製程不相容，整合在單一封裝需要異質鍵合或 flip-chip。Luminar 的 ASIC + detector die-stack 良率在量產爬坡時低於 Volvo audit 要求。
2. **光纖雷射散熱**：1550nm 光纖雷射做到 Class 1 Level 下的高峰值功率、同時封進擋風玻璃後方被動散熱 enclosure，光學效率隨溫度漂移。Luminar 一度嘗試用動態 pulse timing 補償，但這把 downstream point cloud 的 time-of-flight jitter 推高到 perception stack 必須重算的程度，造成 OEM 整合延遲。
3. **ASIC tape-out cycle 跟不上 LiDAR 設計迭代**：半導體子公司原本是 cost center，拆售後反而被 Quantum Computing 以 $110M 標下、意味這些 IP 本身是有價值的，但在 Luminar 整體產品裡被 scale-up 問題拖累。

**對感知工程師的意義不在於「以後不要用 1550nm」**——AEye Apollo、Aeva FMCW 都還在走 1550nm。訊號是：**如果你的 perception 設計假設了「這顆 LiDAR 五年後還會以同樣規格供貨」，那你的假設可能是錯的**。LiDAR 供應鏈在 2026 之後是流動的，perception 架構要有能力在 **90 天內切換 sensor model**，不是從頭重訓。

這把一個長期被忽略的技術問題推到檯面：**sensor-agnostic representation**。你的 backbone 不能假設固定 scan pattern、固定 ring index、固定 return 數——因為下一個設計週期可能是另一家供應商、另一個 scan topology。

### A.3 MicroVision 吃下 Luminar 車用 IP 的意義

MicroVision 原本是 MEMS mirror 路線的玩家，加上 Luminar 的 1550nm IP 後，產品線一次補齊 **短距 MEMS + 長距 1550nm**。但 MicroVision 的現金壓力並沒有解除，$33M 這個低價其實反映了這些 IP 在沒有量產規模支撐下的實際市場估值。

**實務觀察**：如果你是 ADAS Tier 1 要押注未來三年的 LiDAR 選型，MicroVision 整合完 Luminar IP 後是一個觀察標的，但不是立即可下單的對象。真正能現在就下 1M+ 顆量級訂單的，只剩 Hesai、RoboSense、Innoviz（with NVIDIA DRIVE 整合）這幾家。

---

## Part B：AT1440 的規格到底逼感知改什麼

### B.1 把規格翻譯成工程量級

AT1440 的 datasheet 數字：

| 項目 | 規格 | 和 128 線傳統 LiDAR 的比值 |
|---|---|---|
| 線數 | 1440 | ~11.3× |
| Point rate (single return) | 34 M points/s | ~10× |
| FoV | 120° × 18° | 水平同級、垂直略窄 |
| 角解析度 | 0.02° | ~5–10× |
| 最大距離 (@10% reflectivity) | 300 m | ~1.5× |
| Point/frame @10 Hz | ~3.4 M | ~10× |

對 perception stack 來說，決定性的數字是 **3.4 M points/frame**。這把三個子系統從 compute-bound 推進 memory-bound 區間：

- **Point cloud ingestion**：單一 UDP 封包流 ~500 MB/s（含 metadata）。傳統 ROS2 node 的 serialization overhead 開始 dominate；需要 shared-memory 或 zero-copy 傳輸（CycloneDDS / Iceoryx）。
- **Voxelization**：以 (0.1m, 0.1m, 0.2m) voxel 參數 + 100m × 100m × 6m ROI，active voxel 數從 128 線的 ~60k 跳到 **~400k–600k**。這直接把 SparseConv 第一層的 rulebook build cost 推高到 pipeline 最慢的一環。
- **Downstream backbone**：PointPillars 這類 pillar-based 方法在 pillar 數爆炸下開始超時；傳統 CenterPoint 的 BEV feature map 若維持 ~512×512，單一 pillar 要 aggregate 更多點、aggregate op 本身變成 bandwidth-bound。

### B.2 要改的第一件事：Voxel grid 參數不是常數

過去設計 perception pipeline，voxel 大小基本上是「人工挑一次、整個項目用」的常數。1440 線下這個假設爛掉：

- 若維持 (0.1, 0.1, 0.2) voxel：active voxel 爆炸、memory footprint 從 ~50 MB 跳到 ~300–500 MB（含 feature channel）；在車規 Jetson Thor 或 DRIVE AGX 的 unified memory 下開始 contend GPU/CPU 共用頻寬。
- 若放大到 (0.15, 0.15, 0.3)：recall 掉——因為 1440 線的 added value 本來就在遠距細節，voxel 一放大就白給。
- 務實解法：**adaptive voxel resolution**。近場 (<30m) 維持 0.1m、中場 (30–100m) 用 0.15m、遠場 (>100m) 用 0.2m。這是 2024 CVPR 之後 ONCE / nuScenes 社群陸續在玩的 range-aware voxelization，1440 線把它從 nice-to-have 變成 must-have。

這件事對 spconv / Minkowski 這類 sparse kernel library 不是透明的——因為它們假設 voxel grid 是 uniform。要支援 range-aware voxel 得在 ingestion 階段做多重 hash grid，downstream feature 要 **re-project 回統一 BEV**。

### B.3 要改的第二件事：Sparse convolution kernel 要走 compiler-aware 路線

Sparse convolution（spconv / Minkowski Engine / TorchSparse）的核心是 **Rulebook**——每個 output voxel 對應哪些 input voxel 的 lookup table。build rulebook 這一步是 hash-based、記憶體隨機存取密集，點數線性影響其 cost。

128 線下 build rulebook 大約佔 submanifold conv 總時間的 10–15%。1440 線下，根據 CGO 2026 幾篇相關 paper 的實測，這個比例跳到 **35–50%**——因為 rulebook 的 cardinality 本身爆炸、而真正的 fused multiply-add 工作量只以線性比例成長。

這把稀疏卷積 kernel 的優化方向從「加速 GEMM」轉到「加速 rulebook build + fuse rulebook into subsequent GEMM」。幾個具體方向：

1. **Rulebook caching across frames**：如果 ego-motion 可用，前一幀的 rulebook 可以 warp 到當前幀作初始猜測，大幅減少 hash collision。
2. **Persistent kernel for rulebook build**：build rulebook 的 GPU kernel launch overhead 本身變成瓶頸，MPK/Mirage 這類 megakernel 工作有直接適用性（參見 2026-10-27 那篇 MPK 文章）。
3. **MLIR dialect for sparse point cloud ops**：Triton + 自訂 dialect 做 fused submanifold conv + rulebook build 是 2026 下半年 CGO/MLSys 幾個 WIP 的方向。這也是為什麼感知跟 compiler 這兩條線在 2026 開始交匯——1440 線的點雲把兩邊逼到同一個 bottleneck。

### B.4 要改的第三件事：Temporal accumulation 的目的改了

過去多幀累積（temporal accumulation）的目的是**補點**——遠距物體單幀點太少、靠連續 N 幀疊加提高 density。1440 線下近場 saturate、遠場也不太缺點，溫柔的補點已經不是主要收益。

真正的收益改成兩個：

1. **Motion segmentation**：單幀 3.4 M 點、靜態點數量級壓倒動態點。多幀累積的目的變成「對準靜態背景、把動態點突出」。這是傳統 scene flow / dynamic object segmentation 的題目，但在 1440 線密度下，演算法第一次可以用「點」而不是「voxel」為單位做時序關聯。
2. **Temporal consistency for small-object tracking**：300m 距離看到 5 個點的行人，單幀 detection confidence 很低；多幀累積提供「這 5 個點在時序上穩定、不是噪點」的確認。但這要求 ego-motion 補償誤差 < 單點精度（約 5 cm @ 300m），對 IMU + wheel odometry + visual-inertial 的融合精度要求提高。

這第二點直接把 1440 線 LiDAR 跟 **tightly-coupled multi-sensor temporal filter** 綁在一起。單純處理 LiDAR 的 pipeline 在這個密度下開始賠率下降——要吃到 1440 線的所有收益，底下的 state estimator 必須升級。

---

## Part C：這一波位移對台灣 / 本地做 LiDAR 的意義

Luminar 倒 + Hesai 放量，對一個在台灣做 LiDAR 感知的工程師（這也是我的受眾範圍）具體是幾個訊號：

1. **西方 LiDAR 公司作為下游合作對象的風險提升**。Luminar 不是 outlier——Innoviz 2026 年初也在裁員傳聞中。接下來三年內，若要押 LiDAR 感測器選型，Hesai / RoboSense 這條線在供貨穩定性上的優勢遠大於西方新創。這對台廠（Tier 1）供應鏈評估影響最直接。
2. **感知程式碼要有 sensor-portable 的工程紀律**。不要寫死 scan pattern、不要假設 ring index 是連續的、point cloud ingestion 要有統一的 PointCloudXYZIRT schema 層，底下插什麼 driver 都能跑。
3. **1440 線代表 point rate 跳一個數量級**。本地開發板（Jetson AGX Orin, Thor）跟演算法 pipeline 都要重新 benchmark。既有的 ROS2 node 架構在這個 bandwidth 下開始漏水，zero-copy / shared-memory DDS 不是 nice-to-have。
4. **這條線跟 compiler-aware perception 開始交匯**。SparseConv / spconv 的 kernel 優化、Rulebook build 的 persistent kernel 化、MLIR dialect for point cloud ops，都是接下來兩年感知跟 compiler 兩個領域都會長的方向。對做 LiDAR 的人來說，這是跨入 compiler 的一條窄但實際的路。

---

## 結語：地殼位移不是標題，是你 next sprint 要改的程式碼

LiDAR 產業新聞每個月都有、每三個月就有一家「技術領先」的公司出事。但 Luminar Chapter 11 跟 Hesai AT1440 這兩件事放在 2026 年尾一起看，是**西方創新品牌凋零 + 東方供應鏈接管 + 單顆感測器 point rate 跳一個數量級**三件事同時成立。這三件事任何兩件單獨發生都還是正常週期，三件同時成立就是地殼位移。

對寫感知程式碼的人來說，地殼位移具體是**你下一個 sprint 要重算 voxel grid、下一季要把 sparse conv kernel 的 rulebook build 搬到 persistent kernel、下一年要把 ego-motion 補償精度拉進 cm 級**。這些事今天不開始做，等 1440 線 LiDAR 真的進你的 BOM 時，所有的 scheduling 都會爆炸。

這篇文章不是在哀悼 Luminar，也不是在誇 Hesai。是在提醒：**感測器的位移必然會往下傳導成感知堆疊的位移**，而 2026 這一波是過去十年規模最大的一次。

---

## 參考來源

- [Luminar Technologies Chapter 11 Case Summary (Case No. 25-90807)](https://app.bondoro.com/preview/cases/Luminar_Technologies_Inc/case-summary)
- [Luminar Technologies Proposes Liquidation Through $143 Million Asset Sale to Quantum Computing and MicroVision — Chapter 11 Cases](https://chapter11cases.com/blogs/news/luminar-technologies-proposes-liquidation-through-143-million-asset-sale-to-quantum-computing-and-microvision)
- [LiDAR maker Luminar files for bankruptcy — TechCrunch, 2025-12-15](https://techcrunch.com/2025/12/15/lidar-maker-luminar-files-for-bankruptcy)
- [Hesai to Double LiDAR Production to 4M Units Amid Luminar Bankruptcy — WebProNews](https://www.webpronews.com/hesai-to-double-lidar-production-to-4m-units-amid-luminar-bankruptcy/)
- [CES 2026: Hesai Technology to double LiDAR production capacity amid growing ADAS demand — S&P Global](https://autotechinsight.spglobal.com/news/5285977/ces-2026-hesai-technology-to-double-lidar-production-capacity-amid-growing-adas-demand)
- [Hesai signs new $40 million LiDAR contract with leading US robotaxi company — S&P Global](https://autotechinsight.spglobal.com/news/5284070/hesai-signs-new-40-million-lidar-contract-with-leading-us-robotaxi-company)
- [Hesai unveils AT1440 LiDAR with 1440 channels at CES 2025 — CnEVPost](https://cnevpost.com/2025/01/08/hesai-unveils-at1440-lidar/)
- [Hesai AT1440/360 Product Page — Hesai](https://www.hesaitech.com/product/at1440-360/)
- [Hesai Technology Unveils 1440-Channel AT1440, FTX LiDAR To Redefine AD, Robotics — Mobility Outlook](https://mobilityoutlook.com/news/hesai-technology-unveils-1440-channel-at1440-ftx-lidar-to-redefine-ad-robotics)
- 延伸閱讀（本站）：
  - [MPK Mirage：Megakernel 把 SM Task Graph 從外圍 scheduler 搬進單一 persistent kernel](mpk-mirage-megakernel-sm-task-graph-persistent-kernel-osdi2026.md)
  - [AEye Apollo 進 DRIVE Hyperion：1km × 1550nm × 軟體定義掃描](aeye-apollo-drive-thor-hyperion-1km-software-defined-lidar-2026.md)
