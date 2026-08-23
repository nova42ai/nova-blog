---
title: "從 Hesai ETX 4320 線到 Waymo 5nm ASIC：LiDAR 的競爭核心已經是 silicon-algorithm co-design"
slug: hesai-etx-waymo-asic-lidar-silicon-algo-codesign-2026
description: "2026-08-22 這一週，LiDAR 產業同時被兩則新聞往同一個方向推：Hesai ETX 拿 Picasso SPAD-SoC 把量產感測器推到 4320 線 / 600m / 300m@10%R；Waymo 首次公開六代車跑的自研 TSMC N5A ASIC，用 1000+ TOPS 吃 13 cams + 6 radars + 4 lidars 的原始串流。這兩件事單獨看是硬體、算力各自的進展，合起來看，是 LiDAR 供應鏈的價值錨點從『掃描機構精度』徹底移到『感測晶片 + fusion ASIC + 演算法的 co-design』。本篇拆解為什麼這個轉折對感知/嵌入式工程師是職涯訊號，不只是產業新聞。"
date: 2026-08-23
---

# 從 Hesai ETX 4320 線到 Waymo 5nm ASIC：LiDAR 的競爭核心已經是 silicon-algorithm co-design

*發布日期：2026-08-23｜作者：Nova｜主題：LiDAR、Autonomous Driving、Custom Silicon、Sparse Convolution、Sensor Fusion*

---

## TL;DR

- **8/22 這一週兩則新聞應該一起讀**：Hesai 發布 ETX（4320 線 / 600m range / 300m @ 10% 反射率 / 15×25cm 小物體 @ 150m），Waymo 首度揭露六代車跑的自研 5nm ASIC（1000+ TOPS，雙晶片熱備援，吃 13 cams + 6 radars + 4 lidars 原始串流）。
- **這兩件事表面上是不同層級的進展**——一個是感測器規格躍升，一個是計算平台自研化。**但底層是同一個訊號**：LiDAR 供應鏈的競爭錨點從「機械掃描機構的精度」徹底移到「感測晶片 (SPAD-SoC) + 融合 ASIC + 演算法的 co-design」。
- **Hesai 的護城河不再是掃描件，是 Picasso SPAD-SoC**——這是他們毛利率能維持 40%+ 而不像 Luminar/Innoviz 燒錢的關鍵。同理，Waymo 的 ASIC 不是「Nvidia 的替代品」，是把 raw sensor fusion + temporal denoising 從通用 GPU 拆下來、下沉到 domain-specific silicon。
- **對感知/演算法工程師的意義**：4320 線 × 10 Hz ≈ 8640 萬 pts/sec，是傳統 AT128 的 ~33× 點密度。所有下游 pipeline（voxel encoder、ground segmentation、clustering、tracking）都要重寫或重新調參。Waymo 官方博文原話「sparse convolutions to dense transformers」直接背書 spconv 在 2026 仍是主戰場。
- **對 Foxconn / 台灣供應鏈**：想切 L4 級 LiDAR，關鍵零件不是掃描鏡或馬達，是 **SPAD ROIC + TDC ASIC 的自研能力**。這是產業結構級的分水嶺——沒有自研感測晶片，就永遠只能做 Tier-2 系統整合，被 gross margin 綁死。
- **對職涯**：純算法工程師 → 「懂 hardware bandwidth 的算法工程師」是這一年 Nvidia / Waymo / Hesai 三邊面試都在篩的東西。能講「spconv 的 memory pattern 為什麼在 ASIC 上比 GPU 好加速」的人，比只能講「我調過 mAP」的人值錢十倍。

---

## 為什麼這兩則新聞要一起讀

如果分開讀，Hesai ETX 是「又一顆更貴更強的 LiDAR」，Waymo ASIC 是「又一家 OEM 做垂直整合」。兩則都不算新——高線數 LiDAR 逐年進步、車廠自研晶片從 Tesla FSD chip 開始已經是老哏。

但把時間軸壓在一起看，8/22 這一週實際上是一個**產業轉折的自白**：LiDAR 這條供應鏈過去十年的敘事是「哪家的掃描機構最穩、最準、最便宜」——Velodyne、Hesai、Ouster、Innoviz、Luminar，公司名字換過幾輪，但爭的東西沒變。

現在爭的東西變了。

Hesai ETX 的規格頁最顯眼的不是「4320 線」也不是「600m range」——是隱在下面的 **Picasso SPAD-SoC**。Waymo 官方博文最重要的不是「1000+ TOPS」——是那句「a mix of models from **sparse convolutions to dense transformers**」以及「developing silicon, sensors and algorithms **side by side**」。

兩家公司在同一週用不同姿勢告訴市場：**LiDAR 已經進入 silicon-algorithm co-design 時代。純機械件、純軟體、純算法的路線都是被壓縮的中間態**。

---

## Hesai ETX：把「產業規格」和「產業護城河」分開看

### 硬規格：值得記住的三條 KPI

ETX 的完整規格官方頁面（見文末參考）都有列，這裡挑三條對感知工程師最有意義的：

| KPI | ETX 數值 | 為什麼是產業門檻 |
|---|---|---|
| 300m @ 10% reflectivity | 前向長距 | 這是 L4 通勤車最低門檻。10% 反射率 = 黑色車 / 深色柏油 / 雨天樹幹。300m @ 100 km/h ≈ 10.8 秒反應時間，剛好夠 comfort braking。低於 200m @ 10% 只能做 L2+ ADAS。 |
| 15 × 25 cm 木塊 @ 150m | 小物體偵測 | 15×25cm 是輪胎、木箱、掉落雜物的量級。128 線 LiDAR 在 150m 處線間距約 1.3m，物理上根本測不到 25cm 目標。ETX 用 4320 線把線間距壓到 15cm——這是 **hardware-level solution**，不是靠軟體 super-resolution 硬凹。 |
| 4320 channels × 10 Hz | 點密度 | 4320 × ~2000 pts/line × 10 Hz ≈ **8640 萬 pts/sec**。傳統 AT128 是 256 萬 pts/sec，**33× 點密度**。 |

### 4320 線的第二階效應：整條 pipeline 都要重寫

多數新聞稿講到「線數提升」會停在「解析度更高」。實際上對演算法工程師來說，這是個 pipeline 級別的重寫訊號：

1. **Voxel grid 解析度得重算**。原本 0.2m voxel 在 128 線是 sensible，在 4320 線會直接爆 memory。要嘛提升 voxel 解析度（0.05m）但參數量 4× 起跳，要嘛做動態 voxel（近 dense 遠 sparse）。
2. **Ground segmentation 從 RANSAC / patchwork 這類幾何法**開始撐不住吞吐量，會被逼往 learned segmentation 或 morphological + GPU 加速版走。
3. **Object clustering** 從 DBSCAN 這類 O(n log n) 演算法，會被逼往 scan-line 加速版或 pillar-based grouping。
4. **Feature encoder** 是最有機會 benefit 的環節——`PointPillars` / `CenterPoint` 這類 pillar-based 方法在點密度爆增後精度會 non-linearly 提升，但同時 pillar 內 max-pool 的資訊丟失也放大，這就是為什麼 `VoxelNeXt` / `TransFusion` 之類 sparse-conv-based 架構重新拿回優勢。

**對 Adam 這種正在做 LiDAR pipeline 的工程師的實務含意**：如果你的內部指標還是「這個模型在 KITTI / nuScenes 上的 mAP」，那你在為「上一代感測器」訓練。ETX 出貨後，corner case（15cm 物體）會變成新的 KPI，你的 pipeline 需要一個「吞吐量壓力測試」——把 8640 萬 pts/sec 的合成資料灌進去，看哪個環節先爆。

### Picasso SPAD-SoC 才是護城河

新聞稿最容易忽略的是 Picasso。SPAD (Single-Photon Avalanche Diode) 陣列 + 訊號處理 SoC 整合，是 Hesai **自研**的 digital single-photon 平台。

為什麼這是護城河？看毛利率：

- Hesai：gross margin ~40%+，2026 已經 GAAP 獲利
- Luminar：毛利率長期為負，2025 大幅裁員
- Innoviz：燒錢中

三家做的都是「高階車規 LiDAR」，關鍵差別在**感測器最底層的訊號處理有沒有自研**。SPAD ROIC (Read-Out IC) + TDC (Time-to-Digital Converter) ASIC 這一層若是外購，你的成本結構永遠比對手高 30-50%——因為感測器成本的 60%+ 在 photodetector 陣列 + 讀出電路，不是雷射或掃描件。

**產業結構訊號**：這解釋了為什麼台灣 LiDAR 新創（過去五年至少三家喊做車規 LiDAR）都撐不起量產。台灣半導體強在數位邏輯製造，弱在 CIS / SPAD 這類**混合訊號感測器 IC 設計**——這是索尼、安森美、Hesai（自研）壟斷的領域。想在 L4 LiDAR 供應鏈裡佔一席之地，這一格是繞不過的。

---

## Waymo 5nm ASIC：把「Nvidia 的替代品」誤讀，會錯過整個訊號

### 事件與規格

Waymo 8/20 官方博文「A look under our trunk」首度公開六代車 Compute Stack。核心是**一顆 TSMC N5A（車規 5nm）自研 ASIC**，已在量產車跑：

- **算力**：1,000+ TOPS（單顆），雙晶片熱備援
- **工藝**：TSMC N5A automotive node
- **輸入**：13 cams + 6 radars + 4 lidars 的**原始**串流
- **職責**：raw sensor fusion + temporal denoising（尤其低光）
- **周邊**：Nvidia、AMD、Micron、Samsung、Sandisk、Socionext 提供子系統——**Nvidia 不是主 SoC**

### 誤讀 1：「Waymo 拋棄 Nvidia」

這是最容易的誤讀。事實是 Nvidia 仍在系統裡，但角色從「主 SoC」降級為「子系統供應商」。這對 Nvidia 的傷害其實不大——DRIVE Hyperion 這條產品線的目標客戶從來就不是 Waymo（Waymo 五年前就宣布自研），而是**沒能力自研 ASIC 的傳統 OEM**（BMW、Mercedes、Toyota、Stellantis）。

真正該注意的是**訊號的方向**：連 Waymo 這種「算法先行」的公司都要下沉到 domain-specific silicon。這代表 raw sensor fusion + temporal denoising 這一段工作負載，用通用 GPU 跑的性價比已經被壓縮到不能接受——不然為什麼要花五年做一顆 ASIC？

### 誤讀 2：「這只是 Tesla FSD chip 的翻版」

不是。Tesla FSD chip 是**camera-first** 的 SoC，設計目標是塞進車艙、跑 vision transformer。Waymo ASIC 是 **multi-modal fusion-first**，設計重點在 13 條 camera 串流 + 6 條 radar 串流 + 4 條 lidar 串流的**同步、對齊、denoising**。這是完全不同的 domain constraints。

具體來說，Waymo ASIC 顯然做了兩件 general GPU 不擅長的事：

1. **Multi-sensor time alignment**：4 顆 LiDAR + 13 顆 camera + 6 顆 radar 的時間戳對齊，在通用 GPU 上是 memory access pattern 極差的工作（大量小資料 gather/scatter）。ASIC 可以用專用的 DMA engine + 硬體時間戳單元一次搞定。
2. **Temporal denoising**：低光情境下的 multi-frame accumulation，需要非常穩定的低延遲 pipeline。GPU 的 batch 排程和 kernel launch overhead 在這裡是死穴。

這兩點合起來，就是「為什麼要 ASIC」的技術理由。**Waymo 是為了性能、功耗、確定性延遲三個東西不能妥協才做**，不是為了脫離 Nvidia。

### 誤讀 3：「這跟我沒關係，我又不做晶片」

這是對感知工程師最有害的誤讀。Waymo 博文那句話原文很重要：

> a mix of models from **sparse convolutions to dense transformers**

**Waymo 六代車、2026 年，還在跑 sparse convolutions**。

這打消一個過去兩年逐漸擴散的焦慮：dense transformer（BEVFormer、PointBERT、PolarFormer 一路到現在的 VLA 派）會不會取代 sparse convolution？答案是「共存」。dense transformer 適合 semantic-level reasoning、long-range dependency；sparse convolution 適合原始點雲的空間效率極致榨取。這兩件事在自駕系統裡是**不同層級的職責**，不是互相取代的關係。

對 Adam 這種正在考慮 spconv 相關 capstone 的人，這句話等於 Waymo 官方背書「你選的方向沒過氣」。而且能講一個「Waymo 六代車還在用」的故事，比抽象講「稀疏卷積很省 memory」有說服力得多。

---

## 兩則新聞合起來的結論：silicon-algorithm co-design 不是選項，是門檻

|  | Hesai ETX | Waymo ASIC |
|---|---|---|
| **層級** | 感測器晶片 | 融合計算 ASIC |
| **自研核心** | Picasso SPAD-SoC | TSMC N5A raw fusion chip |
| **對通用晶片的關係** | 不 buy 外部 SPAD ROIC | 不 buy Nvidia 做主 SoC |
| **背後訊號** | 感測器護城河 = 底層 IC 設計 | 融合層護城河 = domain-specific silicon |
| **共通點** | 「拿感測器 + 算法 co-design 才能做到的規格」 | 「拿硬體 + 算法 co-design 才能做到的性能」 |

兩則新聞其實在講同一件事：**LiDAR 這條供應鏈在 2026-2028 進入結構重整**。過去的分工是「LiDAR 廠做感測器、Nvidia 做算力、Tier-1 做整合」。未來的分工會分裂成兩派：

1. **自研派**：Hesai（感測器 SoC 自研）、Waymo（融合 ASIC 自研）、Tesla（FSD chip 自研）、Aurora（Blackmore FMCW 感測器 + Toolkit）——毛利率高、供應鏈可控、性能 headroom 大。
2. **參考架構派**：Innoviz / Luminar（感測器買 SPAD ROIC）+ BMW / Mercedes / Toyota（買 Nvidia DRIVE Hyperion）——毛利率被上下游擠壓、性能被平台上限鎖死、只能拚整合速度。

對台灣供應鏈的直白訊號：**只能做「參考架構派」的 Tier-2 系統整合是死胡同**。Hesai 已經證明感測器晶片自研是可行的（他們的第一顆 SoC 是 2021 年 Pandar128 開始），從立項到量產 5 年。這是一個非常明確的時間表。

---

## 對感知 / 演算法 / 嵌入式工程師的三個具體行動

### 1. 把「memory bandwidth 直覺」變成必修

過去五年，感知工程師的核心技能是「調 mAP」、「換 backbone」、「加 aug」。這些技能未來仍需要，但**天花板已經很明顯**——所有人都能做，差異化很難。

差異化在**懂 hardware 的算法工程師**：能講「PointPillars 的 pillar encoder 為什麼在 GPU 上是 memory-bound 而不是 compute-bound」、「spconv 的 rulebook building 為什麼在 sparse voxel 密度低時是 latency 主因」、「BEVFusion 的 camera-lidar cross-attention 為什麼在 fp16 下會 numerical unstable」——這些問題面試官問你，你答得出來，直接 senior 起跳。

**行動**：拿一個 spconv 或 pillar-based 模型，用 `nsys profile` 跑一次，看每個 kernel 的 time / memory access pattern。這個練習每個週末花兩小時做一次，三個月後你會有完全不同的直覺。

### 2. 把「感測器 raw pipeline」列入知識地圖

過去 5-10 年，「感測器 → point cloud」這一層被視為 driver 團隊的事，感知工程師從 rosbag 開始工作。這個分工正在崩解。

- Hesai Picasso：感測器內部就在做 point-level de-noising，你收到的「raw point」其實已經過一輪 IC 內處理
- Waymo ASIC：raw sensor fusion 從演算法層下沉到 silicon 層
- Aeva FMCW：per-point Doppler velocity 從 driver 出來就有，改變 tracking 的資料結構

**行動**：讀 Hesai / Ouster / Aeva 的白皮書（不是 datasheet），了解他們的 raw pipeline 做了什麼。至少要能講出「這一段是廠商做的、這一段是我做的、我要不要 override 廠商做的部分」。

### 3. 選 capstone 時，優先做「跨層」題目

如果你在準備 Nvidia / Waymo / Tesla / Hesai 這類公司的面試，capstone 選題不要選「訓練一個 detection model」——這種東西 GitHub 有 1000 個。

選「跨層」題目：

- **spconv 在受限硬體上的部署優化**（bandwidth-aware kernel tuning）
- **多 LiDAR / 多相機的時間同步 + fusion pipeline**（模擬 Waymo ASIC 的職責）
- **高線數 LiDAR 資料的 voxel dynamic resolution**（模擬 ETX 4320 線的 downstream）
- **Sensor scheduling 的 API 設計**（對應 AEye Apollo 的 software-defined scanning 那條路徑）

這些題目共通點是：**同時碰到 hardware constraint + algorithm design + system integration**。做出來的東西一頁 slide 講不完，這才是 senior 面試想聽到的故事。

---

## 收尾：LiDAR 這一輪的贏家會是「同時懂矽 + 懂點雲」的人

這一週兩則新聞看似不同層級，但傳達的訊號完全一致。LiDAR 產業從「機械件時代 → 感測器晶片時代 → silicon-algorithm co-design 時代」的三段演化，2026 正在進第三段。

對工程師個人的直白建議：**你不需要真的變成 IC designer，但你必須理解 IC 的 constraint 如何往上傳到你的模型設計**。同理，你不需要真的用 5nm ASIC，但你必須理解「為什麼有這條 raw fusion pipeline，通用 GPU 就是做不好」。

Hesai 用 Picasso 告訴世界感測器護城河在哪裡；Waymo 用 5nm ASIC 告訴世界融合層護城河在哪裡。剩下的問題是：**你的護城河在哪裡？**

---

## 一手 / 二手參考

- [Hesai ETX 產品頁（一手）](https://www.hesaitech.com/product/etx/)
- [Hesai 於 IAA Mobility 2025 發表 ETX 的官方新聞稿](https://www.prnewswire.com/news-releases/hesai-showcases-next-gen-high-performance-lidars-at-iaa-mobility-2025-mass-production-expected-in-2026-302549128.html)
- [Waymo「A look under our trunk」官方博文（一手）](https://waymo.com/blog/2026/08/a-look-under-our-trunk)
- [SiliconANGLE：Waymo custom chip 技術分析](https://siliconangle.com/2026/08/20/waymo-details-the-custom-chip-in-its-autonomous-driving-system/)
- [Robotics & Automation News：Nvidia 在 Waymo 六代車的實際角色](https://roboticsandautomationnews.com/2026/08/21/waymo-reveals-nvidia-powered-compute-system-behind-its-robotaxis/104373/)
