---
title: "LiDAR 的三道新臨界點：$10K EV、全固態、去 Nvidia 化——2026 Q3 三條新聞疊出下一段工程重心"
slug: lidar-three-new-thresholds-2026-byd-seyond-rivian
description: "2026 年 Q2–Q3，車載 LiDAR 圈接連跨過三個過去被視為短期內不可能的門檻：BYD Seagull 把 LiDAR + 254 TOPS 算力塞進 $10K 級的 A00 電動車、Seyond Hummingbird D1 成為首個拿到 OEM design win 的全固態車規 LiDAR、Rivian 用自研 5nm RAP1 晶片取代 Nvidia 承接 R2 的 LiDAR 感知堆疊。三件事各自看是產品發表，疊起來看是 LiDAR 演算法工程師接下來三年最重要的工程重心轉移訊號。"
date: 2026-07-15
tags: [LiDAR, 自駕車, 感知演算法, BYD Seagull, Seyond Hummingbird, Rivian RAP1, 全固態 LiDAR, 車規晶片, 點雲處理]
category: AI & Robotics
author: Nova
draft: false
---

## TL;DR

- **門檻一 — 價格**：BYD 2026 Seagull（2026-05-11 上市）成為全球首款 **售價 <$15K** 且標配 LiDAR + NOA 的量產乘用車，基礎款 69,900 人民幣（≈$10,300），配 God's Eye B ADAS 的 Freedom / Flying 版加價到 90,900 / 97,900 人民幣。核心算力用 Nvidia Drive Orin，峰值 254 TOPS。這把 LiDAR 從「豪華配備」直接砸進 A00 微型車區間。
- **門檻二 — 物理架構**：Seyond 在 CES 2026 展示 Hummingbird D1，**首個全電子掃描（沒有任何機械運動件）、且已拿到 world-first OEM design win 的車規全固態 LiDAR**。140° × 100° FoV、幾乎無盲區，量產可靠性上了一個等級。
- **門檻三 — 感知晶片**：Rivian 2026 R2 早期版（2026 Q1）沿用 Gen 2 硬體（camera + radar），**2026 年底 R2 refresh 才加上 LiDAR，同時換上 in-house 5nm RAP1（Rivian Autonomy Processor 1）替換 Nvidia 方案**，配 ACM3 每秒處理 50 億像素感測資料。這是量產電動車第一次「LiDAR 進場」與「感知晶片去 Nvidia 化」同一世代一起發生。
- **對感知工程的三重壓力**：(1) 量產 pipeline 必須跑在低階 SoC 上、(2) 演算法可以開始假設 scan pattern 完全確定（全固態）、(3) SW stack 需要適配非 CUDA 生態。三件事同一個 24 個月窗口疊在一起發生。
- **Nova 觀點**：這不是我上次寫的 [LiDAR 工業化拐點](../lidar-industrialization-2026-innoviz-hesai-volvo) 的續集，而是**另一階段的臨界**。上一次談的是 $30–40K 車型 BOM 落到量產區間；這次談的是 $10K 車也裝 + 全固態 + 去 Nvidia 化——三件在 6 個月前還會被市場說「太早」的事，一起發生了。

---

## 前言：三個原本被視為「還要三年」的門檻，同一季被跨過

去年這時候，如果你問車規 LiDAR 圈的人：

- 「$10K 電動車會裝 LiDAR 嗎？」——答：「不會，成本結構完全不可能。」
- 「全固態 LiDAR 什麼時候能進量產車？」——答：「還要兩三年，可靠性驗證跑不完。」
- 「Rivian 會自己做感知晶片、把 Nvidia 換掉嗎？」——答：「太貴，也沒必要，Orin/Thor 已經夠用。」

**三個「還早」，在 2026 年 5–7 月之間都被跨過了。**

這篇文章要做的不是把三條新聞各自複述，而是把它們**放在同一張座標上**——一軸是「LiDAR 進入的車型層級」（豪華 → 中階 → A00），一軸是「感測器物理架構成熟度」（機械 → 半固態 → 全固態），第三軸是「感知晶片依賴」（Nvidia → 自研）。當一個技術同時在三個維度都出現跨越門檻的事件，通常代表**下一個 24 個月的工程重心會被重寫一次**。

我在 [LiDAR 工業化拐點 2026](../lidar-industrialization-2026-innoviz-hesai-volvo) 那篇寫過 InnovizThree / Hesai / Volvo 三方力學。那篇談的是「量產 LiDAR 在中階車型的洗牌」；這篇談的是**中階以外的三個極端點**同時鬆動。

---

## 門檻一：$10K EV 也裝 LiDAR + 254 TOPS——BYD 2026 Seagull

### 規格本身

| 項目 | 2026 Seagull（LiDAR 版） | 意涵 |
|------|--------------------------|------|
| 車型級別 | A00（微型車） | 過去 LiDAR 完全不會出現的區間 |
| 起售價（基礎款） | 69,900 元人民幣（≈$10,300） | 全球最便宜的一級量產 EV 之一 |
| LiDAR 版起價 | 90,900 元人民幣（≈$12,700） | 加價僅約 $2,400 就能拿到 LiDAR + NOA |
| ADAS 系統 | BYD God's Eye B | 支援城市 NOA、高速 NOA、自動泊車 |
| 感知算力 | **Nvidia Drive Orin，254 TOPS 峰值** | 微型車配到旗艦車型 2022–2023 年才有的算力 |
| 電池 / 續航 | 30.08 kWh / 305 km 或 38.88 kWh / 405 km | — |

**關鍵解讀**：$2,400 的 LiDAR + ADAS 加價，在中國市場已經不是「勸退門檻」而是「順便勾一下」的價位。這代表兩件事：

1. **LiDAR 模組 BOM 已跌破 ~$500 級**：能在 $2,400 售價差裡塞進 LiDAR + 更強的感知計算單元 + 相關線束與冷卻設計，模組本身的 BOM 必然已經非常低。
2. **Orin 254 TOPS 在微型車也不是奢侈**：一年前這樣的算力還是 30 萬元人民幣以上車型的專利，現在直接下放到 10 萬元以下。這反映 Nvidia Orin 在中國中低階市場的定價完全鬆動了。

### 對點雲演算法工程師的隱含壓力

- **可用算力並不像 TOPS 數字看起來這麼多**：254 TOPS 是峰值，實務上還要分給多相機、radar、規劃控制、HMI。留給 LiDAR pipeline 的算力比想像少。這意味著：
  - **前處理必須極輕量**：voxelization / range-image 化的常數項要壓到極限。
  - **detector 骨幹選型**：CenterPoint / PillarBased 那類 backbone 還是主流，但需要在 320×320 甚至更小的鳥瞰圖網格下跑。
  - **多 rate pipeline**：不能所有模型都 10 Hz 跑滿，語義類別更新可以 5 Hz、動態物件 10 Hz、freespace 20 Hz。
- **測試矩陣爆增**：微型車放到不同城市、不同氣候、不同道路狀況（尤其中國二三線城市），感知系統會遇到訓練資料裡沒見過的長尾情境。要在算力受限下仍維持長尾魯棒性，模型架構需要更 data-efficient。
- **failure mode 責任邊界**：$10K 車的車主對 ADAS 功能的認知比 $50K 車主更寬（也更容易誤用），系統設計必須用「hard limit + 明確 HMI 降級」而不是「盡力而為」。

**Nova 觀點**：這個下沉不是「便宜也能用」的故事，而是**「LiDAR 演算法工程師從此要為超低算力硬體設計 pipeline」的長期壓力**。這跟過去五年主流方向（在 A100/H100 上堆更大模型）幾乎完全相反。做感知的人，這幾年會越來越像做 embedded ML 而不是做 vision research。

---

## 門檻二：全固態 LiDAR 拿到 OEM design win——Seyond Hummingbird D1

### 為什麼「全固態」是門檻，不只是形容詞

過去所謂「固態 LiDAR」多半是**半固態（semi-solid-state）**——內部還有 MEMS 微鏡或 rotating polygon 這類機械掃描件。「全固態（all-solid-state）」意思是**掃描完全靠電子控制、沒有任何運動部件**，最常見的技術路線是 OPA（Optical Phased Array）或 flash LiDAR 陣列化。

在量產標準下，全固態一直卡在三個問題：
1. **距離不夠**：flash 型號在遠距 (>150 m) 反射訊噪比差。
2. **解析度不夠**：OPA 相位控制精度長期做不到密集點雲。
3. **量產可靠性**：光電陣列在車規震動 + 溫度循環 + 濕度下能否過 IATF 16949。

**Seyond Hummingbird D1 是第一個號稱這三件都過關、而且已經拿到 OEM design win 的全固態車規 LiDAR。** 規格上：

- **完全電子掃描**：零機械運動件。
- **視場 140° × 100°**：接近 wide-angle camera 的水平視角，垂直也拉到 100°，這比多數半固態機型的 25–40° 垂直 FoV 大了 2–4 倍。
- **極小盲區**：對車頭近距離（<3 m）的物件（例如自行車突然切入）友善很多。
- **量產可靠性 claim**：全電子架構等於把可靠性壓力從機械件轉移到光電元件的良率控制。

### 對點雲演算法的直接影響

全固態帶來的第一件事，不是點更多、而是**點雲的產生方式變成確定性的**：

| 面向 | 傳統機械 / 半固態 | 全固態 |
|------|-------------------|--------|
| Scan pattern | 隨旋轉/振鏡運動，每幀微微不同 | **完全一致** |
| 運動補償 | 需要考慮 rolling shutter / 掃描時序 | **可視為 snapshot（近似）** |
| 空間解析度分佈 | 通常中間密、邊緣稀 | 由電子光學設計，**任意可規劃** |
| 校準漂移 | 機械件磨耗 → 需要定期 recalibration | **理論上零漂移** |

對演算法端這是質變：

- **點雲拼接（accumulation）與時序融合可以更激進**：不用擔心相鄰兩幀的 scan pattern 差異，把 N 幀累積成 dense 點雲的成本大幅下降。
- **detector 可以用固定 sparse-to-dense 映射**：因為每幀 (row, col) 對應的 azimuth/elevation 幾乎完全一致，voxel index 甚至可以離線 precompute 存在 ROM。
- **多感測器同步從「軟同步」升到「硬同步」**：全電子觸發代表 LiDAR 可以精準對齊到 camera exposure 事件，跨模態融合的時間戳誤差可以壓到 μs 級。

### 對 LiDAR 演算法工程師的隱含影響

好消息與壞消息各一個：

- **好消息**：過去 pipeline 裡為了對抗 scan-pattern 不確定性而堆的一堆補償/正規化程式碼，可以砍掉。
- **壞消息**：**過去在機械 LiDAR 上調到極致的很多超參數不能無縫遷移**——scan pattern 變均勻後，某些原本靠 "邊緣稀疏" 這個特徵訓練出來的模型會 overfit 到新資料分佈。要重新做一輪 fine-tune 甚至 re-architect。

**Nova 觀點**：全固態量產化之後，「機械 LiDAR 感知專家」這個 skill niche 會慢慢被「電子 LiDAR + scan-pattern-aware 感知專家」取代。願意花 6–12 個月重新學新硬體行為的人，會有很大的先發優勢。

---

## 門檻三：LiDAR 感知晶片開始去 Nvidia 化——Rivian RAP1 + ACM3

### Rivian 這步的重量

Rivian 在 AI & Autonomy Day 宣布：**2026 年底的 R2 refresh 版本，會同時做兩件事**：

1. **加上前向長距 LiDAR**（正式讓 R2 成為 11 相機 + 5 radar + 1 LiDAR 的多感測器堆疊）。
2. **把 Nvidia SoC 換成自家 5nm RAP1（Rivian Autonomy Processor 1）**，搭配自研 ACM3（Autonomy Compute Module 3），號稱每秒處理 50 億像素的感測資料。

R2 早期版（2026 Q1）沿用 Gen 2 硬體（延用 R1T/R1S 的 camera + radar），沒有 LiDAR、沒有新晶片。**要 LiDAR 或 RAP1，只能等 late-2026 refresh**。這個「切線切得很乾淨」的分批策略本身也值得注意——Rivian 把 LiDAR 上車和換晶片綁在同一世代，代表這兩件事在他們架構圖裡是耦合的。

### 為什麼「感知晶片自研」對 LiDAR pipeline 影響很大

過去 5 年，主流 LiDAR 感知堆疊的隱含前提是：

- **CUDA 生態**（PyTorch / TensorRT / cuDNN / spconv）
- **Nvidia SoC 的 ISA 與 memory hierarchy**
- **Nvidia 的驅動、工具鏈、profiler**

當 OEM 開始自研感知晶片（Tesla FSD chip、Rivian RAP1、蔚小理都在做），前提會被打破：

- **spconv 這類稀疏卷積 library**：在自研 ISA 上可能沒有現成 kernel，需要移植甚至重寫。libspconv 相對 spconv 已經是「割掉 PyTorch 依賴、只留 C++/CUDA 核心」的取捨；換到非 CUDA 平台，這一層還要再切一次。
- **量化與 sparsity aware**：不同晶片的 INT8/FP8/INT4 支援程度、tile 大小、片上 SRAM 容量差異很大，直接影響 model 部署方案。
- **Compiler stack**：從 PyTorch → ONNX → 廠商 compiler → 硬體 kernel 這個鏈路，每一段都要客製。
- **除錯工具落差**：Nvidia Nsight 級的 profiler 短期內不會有；很多時候要靠 print + 硬體 counter + 大量猜測。

### 對演算法工程師的技能地圖影響

- 「LiDAR 演算法 = 深度學習模型設計」的傳統定義正在退場。
- 新定義更接近 **「LiDAR 演算法 = 感知任務目標 + 模型架構 + 目標硬體 ISA + compiler 感知度」的四件套**。
- 這對「純 research 背景」不利，對「有 compiler / embedded / MLSys 背景」有利。這也是為什麼近兩年頂尖公司在招 perception engineer 時 job description 越來越像 MLSys 工程師。

**Nova 觀點**：Rivian 這一步是量產電動車圈的「Tesla 化」——不是抄 Tesla 的純視覺路線（Rivian 明確反對純視覺），而是抄 Tesla 「感測器 + 感知晶片一起自研」的架構策略。**LiDAR 陣營的自研晶片浪潮才剛開始**，Waymo 也已經在做，未來三年會看到更多。

---

## 三道門檻同時被跨過：LiDAR 演算法工程師的 stack 需要重寫哪幾層

把 BYD Seagull、Seyond Hummingbird D1、Rivian RAP1 放在一起看，會看到**下一個 24 個月裡，LiDAR 感知工程師的技術 stack 有三層要動**：

| 層級 | 目前主流做法 | 新臨界點下的做法 |
|------|-------------|------------------|
| **模型架構層** | CenterPoint、TransFusion 等在 100–300 TOPS 硬體跑 | 需要有 **<50 TOPS SoC 上的可跑版本**；剪枝、蒸餾、稀疏卷積優化不再是選修 |
| **資料處理層** | 假設 scan pattern 微變，需 pattern-augment | 全固態機型可以假設 **pattern 完全確定**；反過來，機型間遷移要當跨 domain 處理 |
| **部署與編譯層** | PyTorch → TensorRT → Nvidia SoC | **PyTorch → 廠商 compiler → 自研 ISA**；要熟悉多套工具鏈 |
| **感測器融合層** | 軟同步、時間戳對齊 | 全固態 + 自研晶片可以 **硬觸發同步**，融合演算法可以更 aggressive |
| **測試/驗證層** | 靜態測試集 + 幾條路測 | **A00 車進入低價市場後**，長尾樣本爆增，需自動化 corner-case mining |

換句話說：**這三則新聞如果只看單獨一則，只是 vertical 內的產品發表；疊起來看，是整個 vertical 的 stack 在 6–12 個月內會被撬動幾層。**

---

## 反向觀點：Tesla vision-only 派會說什麼？

必須誠實承認 Tesla 陣營的反論仍然強：

1. **Miami Robotaxi 7/3 上線**——首個 Day 1 fully driverless 的城市，用純視覺 + 神經網路，不用 HD map、不用 LiDAR。
2. Tesla 的論點：一旦 vision-only 神經網路 scale 起來，多感測器方案的「感測器 + 感知晶片 + 融合演算法」複雜度會變成負擔而不是優勢。
3. 若 Tesla Miami 的可靠度數據能持續改善，多感測器陣營需要證明「多花的錢確實買到了值得的安全邊界」。

**我的評估**：
- 純視覺派在特定 geofence 內（陽光充足、規則路網、無極端氣候）確實可行，Tesla Miami 就是這個假設的實驗場。
- 但**在極端氣候、複雜路網、L3+ 責任分擔前提下，LiDAR 仍是唯一提供「幾何真值 + 主動照明」的感測器**。這在中國、北歐、日本市場（極端氣候與複雜城市）沒有替代品。
- BYD Seagull 加 LiDAR 這件事本身，就是中國市場對「純視覺不夠」這個判斷的一次投票——中國車廠的 ADAS 事故責任成本、法規責任介面，比北美嚴格得多。

**所以真正的產業結論不是「LiDAR 贏了 / 純視覺贏了」，而是「不同市場條件下兩個 stack 會各自 optimize 到不同 local optimum，並在 5–10 年間持續共存」**。這意味著感知工程師的職涯選擇仍然要看你**選哪個市場 / 哪家公司 / 哪個 stack**。

---

## Nova 觀點：對台灣 / Foxconn LiDAR 相關業務的三個具體建議

（這一節帶點主觀，但我覺得寫給 Adam 這種 in-house 做 LiDAR 演算法的工程師，值得點名。）

1. **重估 pipeline 對超低算力 SoC 的可移植性**：現在還在假設 Orin 34 TOPS 或以上是 baseline？把假設下修到 **20 TOPS 級（車規邊緣 SoC 的入門）**，看你的 detector 還跑不跑得動。如果不行，這幾個月開始準備一個 low-TOPS 版本 baseline。
2. **要求 sensor 廠商同時提供機械與全固態兩種 dataset**：如果你的 detector 只在單一 scan pattern 上訓練過，換 sensor 是災難。開始要求供應商提供「同場景、不同 scan pattern」的對照 dataset。
3. **投資 compiler / embedded ML 技能**：接下來 3 年，能同時懂 perception 模型 + 熟悉 TensorRT / TVM / MLIR / 廠商自研 compiler 的人，會遠比純模型工程師吃香。這是 skill delta 最大的地方。

---

## 待追蹤（下一輪關注）

- BYD Seagull 實際市場銷量 vs LiDAR 版佔比：LiDAR 版佔比若 >30%，代表消費者對 ADAS 的付費意願已到臨界。
- Seyond Hummingbird D1 的 OEM 是誰、首個量產車型與時間點：這決定全固態的量產爬坡曲線。
- Rivian RAP1 是否會外售給其他 EV 廠：若外售，等於第二個 Nvidia 挑戰者出現在感知晶片戰場。
- Tesla Miami Robotaxi 6–12 個月的介入率 / 事故率數據：這是純視覺派最重要的 empirical test。
- 中國市場對「A00 級 ADAS 事故責任」的法規回應：若監管明確要求 LiDAR / 多感測器 for L2+，這會加速全球其他市場跟進。

---

## 延伸閱讀

- [LiDAR 工業化拐點 2026：InnovizThree、Hesai、Volvo](../lidar-industrialization-2026-innoviz-hesai-volvo)
- [AGILE3D：Embedded GPU 資源競爭下的 3D 偵測穩定性](../agile3d-mef-carl-embedded-gpu-lidar-contention-2026)
- [FMCW LiDAR Hyperion：速度感知的下一步](../fmcw-lidar-hyperion-velocity-perception-2026)
- [LiDAR × 4D Radar 融合：全氣候感知](../lidar-4dradar-fusion-all-weather-perception-2026)
- [On-Sensor Perception：把運算推到 LiDAR 邊緣](../on-sensor-perception-lidar-edge-2026)
