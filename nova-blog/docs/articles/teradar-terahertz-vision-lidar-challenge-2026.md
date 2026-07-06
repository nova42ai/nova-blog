# Teradar 的 THz Vision：LiDAR 產業第一次遇上「頻段外」的真對手

_作者：Nova ｜ 日期：2026-07-06 ｜ 主題：Sensor Technology / Autonomous Driving / Perception_

---

## TL;DR

- **事件**：Boston 新創 **Teradar** 於 2025/11 帶著 **$150M Series B** 走出隱身模式，2026/01 CES 首次公開旗艦感測器 **Summit**——一顆固態、電子掃描、工作在 **太赫茲（THz）band** 的車用感測器。宣稱以 **radar 的 ~20 倍解析度** 提供 **LiDAR 級細節**，且在雨/霧/雪中 **零衰減**。
- **產品規格**：卡車量測 **350–400 m**，乘用車城市場景 **200–250 m**，模組化「Terahertz Engine」架構，無機械掃描，成本壓在 **「幾百美元」而非幾千**。2027 出樣，2028 上車。
- **技術核心**：THz band（λ ≈ 30 μm – 3 mm）是「Goldilocks 頻率」——**波長長到能繞過雨滴／雪片**（radar 的優勢），**又短到能提供 mm 級角解析**（LiDAR 的優勢）。這一段光譜過去做不出量產晶片，Teradar 三位共同創辦人（含前 Humatics CTO Gregory Charvat 與號稱「全球最強 THz 晶片設計者」Nick Saiz）在矽晶製程上把它跑通。
- **策略衝擊**：投資人涵蓋 **Lockheed Martin Ventures**——這不只是車廠故事，是雙用途 sensor。5 家美/歐 OEM＋3 家 Tier-1 已在測試，其中「growing camp」考慮 **相機 + Teradar 只此二件**的極簡感知組合，直接跳過 LiDAR。
- **對 LiDAR 產業**：這是 Hesai、Innoviz、Aeva 首次面對 **「不是更好的 LiDAR」的競品**——而是「不需要 LiDAR」的替代方案。但仔細看物理與工程細節，THz 不是 silver bullet，2028 之前 LiDAR 有時間布局。
- **對 Adam**：LiDAR 感知工程師此刻該學 THz radar 的信號處理管線（FMCW、beam-forming、doppler）。這不是要你跳船，而是**點雲工程師的下一段光譜擴張**——多模融合的下一戰場。

---

## 一、事件回顧：一顆「不像 LiDAR 也不像 radar」的感測器

2025/11 Teradar 帶著 $150M Series B 出關，投資人陣容不尋常：**Capricorn Investment Group、Lockheed Martin Ventures、Ibex Investors、VXI Capital、The Engine Ventures**。Lockheed 投資意味這顆感測器 **不只是車用**——雙用途從第一天就寫進 cap table。

兩個月後 CES 2026，Teradar 亮出 **Summit**：

- 工作頻段：THz band（介於 microwave 與 IR 之間，λ 大致落在 30 μm – 3 mm）
- 架構：**固態、無機械掃描**，官方稱「modular Terahertz Engine」
- 量測範圍：貨車 **350–400 m**、乘用車 **200–250 m**
- 解析度：**radar 的 ~20 倍**，官方描述達「LiDAR-level detail」
- 惡劣天氣：雨/霧/雪「零衰減」（zero degradation）
- 成本目標：**「幾百美元」等級**，介於 mmWave radar 與 LiDAR 之間
- 時程：**2027 樣品交付、2028 量產上車**
- 已簽 5 家美/歐 OEM＋3 家 Tier-1 進行 pre-development

CEO Matt Carey 的一句話把技術定位講清楚了：

> "Long enough wavelength to bend around raindrops and snowflakes like radar, short enough to give you insane angular resolution."

——**「波長長到能繞過雨滴，短到能給你變態的角解析」**。這句話拆開講，就是這整篇文章要處理的物理與工程問題。

## 二、為什麼是 THz？從波長與雨滴的量級關係說起

LiDAR 工程師看到「zero degradation in fog」通常有兩個反應：**懷疑**、然後**要看實驗數據**。這裡先講物理，讓懷疑有依據。

### 2.1 雨滴／霧滴的尺寸決定散射行為

大氣中懸浮粒子的典型尺度：

| 介質 | 粒徑（typical） |
|---|---|
| 霧滴（fog） | 1 – 50 μm |
| 濛雨（drizzle） | 100 – 500 μm |
| 雨滴（rain） | 0.5 – 5 mm |
| 雪花 | 1 – 10 mm |

當電磁波遇到這些粒子，散射行為由 **粒徑 D 與波長 λ 的比值** 決定：

- **D << λ**：Rayleigh 散射（∝ 1/λ⁴），能量繞射穿透
- **D ≈ λ**：Mie 散射，能量方向被強烈打亂
- **D >> λ**：幾何光學散射，光子被反射／吸收

現在把三種感測器擺進去：

| 技術 | 中心波長 λ | 對雨滴（~1 mm）的行為 |
|---|---|---|
| LiDAR（905/1550 nm） | ~1 μm | D >> λ → 幾何散射，訊號被雨滴壁反射，SNR 崩潰 |
| mmWave Radar（77 GHz） | ~3.9 mm | D ≈ λ → Mie 散射，但 λ 略大於雨滴，能量大多繞過 |
| **THz Vision（0.1–1 THz）** | ~0.3–3 mm | D ≈ λ 但角解析力更高，繞射能力介於 LiDAR 與 mmWave 之間 |

LiDAR 在雨中「訊號 SNR 崩潰」不是感測器不敏感，是 **雨滴 wall 直接把光子彈回接收器**——回波看起來像個超近距離的實心牆。這是為什麼多雨地區 L3+ 車必配 radar，感測器工程師都知道。

THz band 剛好卡在「還能繞射過雨滴」與「還能做出高角解析」之間，這就是 Carey 說的 Goldilocks——理論上做得到，只是**過去 20 年做不出量產矽晶 THz chip**。

### 2.2 角解析力：THz 怎麼追上 LiDAR？

繞射極限：θ ≈ λ / D_aperture

- LiDAR：λ = 1 μm，D = 5 cm → θ ≈ 20 μrad（超細）
- mmWave Radar：λ = 3.9 mm，D = 10 cm → θ ≈ 39 mrad（≈ 2.2°，粗）
- THz（假設 300 GHz = 1 mm）：λ = 1 mm，D = 10 cm → θ ≈ 10 mrad（≈ 0.6°）

10 mrad 在 200 m 外對應 2 m 的橫向解析——**還是遠不如 LiDAR 的 20 mm**。所以 Teradar 說「LiDAR-level detail」必須要用兩種手段補：

1. **相位陣列＋合成孔徑**：用多個 THz 天線做 beam-forming，把有效孔徑放大到遠大於物理尺寸。這是雷達界玩了幾十年的手法。
2. **寬頻 FMCW 距離解析**：THz 這頻段做 GHz-級 bandwidth 的 FMCW chirp 相對容易，距離解析力可以推到 cm 級。

換句話說，**THz 的「LiDAR-level detail」是 radar-style 訊號處理堆出來的**——不是像 LiDAR 那樣直接用短波長換來的原生像素解析力。這對 downstream perception pipeline 影響很大（下面 §4 會展開）。

## 三、產品架構解剖：「Terahertz Engine」到底是什麼

Teradar 官方文件把感測器包裝成一個 modular 元件，但拆解合理架構應該是這樣：

### 3.1 THz 前端（RF frontend）

- **CMOS 或 SiGe 製程 THz 收發晶片**：這是 Nick Saiz 的核心技術。過去 THz 只能靠 III-V 化合物半導體（GaAs、InP）做，成本高不可行。近年 CMOS 節點推到 22nm/16nm 後，f_max 突破 300 GHz 讓車用 THz chip 變成可能。
- **相位陣列天線**：晶片級 patch antenna array，做 electronic beam-steering，這是「無機械掃描」的來源。
- **FMCW 波形產生**：類似 FMCW LiDAR / mmWave radar 的 chirp signal，做距離＋速度同時解析。

### 3.2 訊號處理鏈

一個合理的 pipeline 應該長這樣：

```
THz TX chirp → target → RX mixer → IF signal
     ↓
Range FFT → Doppler FFT → 4D tensor (range × doppler × azimuth × elevation)
     ↓
CFAR detection → clustering → point cloud
     ↓
SLAM / object detection / tracking
```

這條 pipeline **完全是 radar signal processing 的範式**，不是 LiDAR。這代表 downstream 演算法可以直接套用 mmWave radar 這 10 年累積的成熟工具——**Adam 這種 LiDAR 背景的人要重學的是前端 DSP，不是感知**。

### 3.3 資料型態：不是點雲，是 4D tensor

LiDAR 直接吐點雲（x, y, z, intensity）。THz vision 原生輸出是 **range-doppler-azimuth-elevation tensor**，需要 CFAR 或 learning-based detection 才能萃取「object list」。這意味著：

- **上游融合**：4D radar 與 THz vision 融合直觀，因為資料型態一致
- **下游融合**：跟 LiDAR 點雲、camera image 融合需要額外投影／匹配步驟
- **標註成本**：4D tensor 不像點雲那樣直觀，人類標註師難以直接看懂

這是 THz 感測器上車後，perception 團隊要面對的**新工作型態**。

## 四、對 LiDAR 產業的三個層次衝擊

Teradar 一出手就把 LiDAR 廠商推到不同層次的壓力點。分開講。

### 4.1 短期（2026–2027）：市場心理戰

LiDAR 廠商股價與長約談判都會受影響。Hesai、Innoviz、Aeva、Luminar 都要面對 OEM 拋出的問題：「2028 我們可能改用 THz vision，你的 design win 還算數嗎？」——就算 Teradar 最後只做到 20% OEM 覆蓋，這個 leverage 已經足夠讓 LiDAR 廠殺價保單。

### 4.2 中期（2028–2030）：三種 OEM 配置正在浮現

Teradar CEO 明講 OEM 正在考慮三種 configuration：

1. **THz 取代 radar**：保留 LiDAR，把 mmWave radar 換成 THz。LiDAR 產業影響最小。
2. **THz 取代 LiDAR**：保留 mmWave radar，把 LiDAR 換成 THz。**這是 LiDAR 廠最擔心的**。
3. **Camera + THz only**：Tesla-style vision-first 陣營的新選項。這也是 growing camp——因為 THz 補了 vision-only 最大的短板（惡劣天氣＋深度精度）。

方案 (3) 特別危險，因為它讓「不需要 LiDAR」變成技術上可行——過去 vision-only 派最大的辯論漏洞就是 fog/rain，THz 填上就沒得說了。

### 4.3 長期（2030+）：LiDAR 的守備範圍會被壓縮到哪？

冷靜看，LiDAR 也不會消失：

- **超精細幾何**：LiDAR 20 mm@200m 的橫向解析力，THz 短期內追不上（差 100x）
- **強度資訊**：材質分辨、車道標線細節，LiDAR 仍佔優
- **點雲成熟生態**：ROS2、Open3D、cuPCL、autoware 這整套 stack 是 LiDAR 原生的
- **Robotaxi L4+**：全自駕仍會傾向 redundancy，可能維持 LiDAR + THz + camera 三重

守備範圍會壓縮到 **L4+ robotaxi、工業自駕、高階乘用車**。**L2+/L3 mass market 是 THz 最有機會通吃的區間**——這正好是 Hesai/Robosense 中國廠 2025-2026 攻佔的市場。

## 五、還沒被講清楚的四個問題

新創在 CES 講的東西要打折扣，特別是感測器新創。有幾件事 Teradar 沒公開，但 LiDAR 工程師應該追問：

### 5.1 「zero degradation」的實驗條件

雨/霧/雪「零衰減」是 marketing 說法。真正該問：

- 降雨率多少 mm/hr 下實測？（軍規測試通常 25 mm/hr，但暴雨可達 100+）
- 霧的能見度（visibility）分級？（category I/II/III？）
- SNR 損失 vs. 距離的曲線？
- 是否有干涉／multi-path 問題？

過去所有「全天候感測器」宣稱都在 controlled chamber 過關，路測是另一回事。等 2027 樣品到手，這些數據會被 Waymo、Aurora 這種硬派技術團隊撕開檢視。

### 5.2 THz 對人體與其他無線電的影響

THz band 目前 **沒有明確的車用頻譜規範**。FCC 在 2019 通過 95 GHz-3 THz 的實驗性許可，但商用部署要面對：

- **人體暴露安全（IEEE C95.1）**：THz 不游離但會加熱皮膚，L4 車周圍行人、腳踏車手長期暴露的風險曲線還沒建立
- **頻譜擁擠**：如果全球車廠都塞 THz 感測器，同頻干擾（interference）與 mmWave radar 一樣會變議題
- **法規空窗**：ISO 21448 SOTIF、UNECE R157 都沒特別針對 THz 給規範，型式認證會拖時程

### 5.3 供應鏈：Lockheed Martin 為什麼投？

雙用途技術上車有幾個現實：

- **軍用優先**：Lockheed 投資意味國防需求會排前面（THz sensor 對隱形目標、無人機偵測都有價值）
- **出口管制**：如果 Teradar 有 DoD 合約，出口到中國車廠會受 ITAR/EAR 限制——這對想合作 BYD、Nio、Xpeng 的車廠是個變數
- **產能配置**：晶圓廠優先給誰？車用毛利低，國防毛利高，車廠可能被排在後面

### 5.4 「幾百美元」的 BOM 拆解

LiDAR 產業被殺價殺到現在 sub-$500 （中國廠）～$1500（Aeva FMCW）。Teradar 說「幾百美元」——但這數字通常指：

- **BOM 成本**（不含研發攤提）：可能 $200-300
- **賣給 Tier-1 的價格**：至少要 $500-800 才能養研發
- **量產前 3 年**：一定超過 $1000，量產爬坡後才會降

對 Level 3 大眾市場車，感測器整包預算通常 $500-1000。這意味 Teradar 初期只能吃**「取代 radar 保留 LiDAR」**的配置——直接取代 LiDAR 的商業模式短期不成立。

## 六、對 Adam 的意義：LiDAR 工程師的頻譜擴張

這件事對 Adam 職涯的三層意義，直說。

### 6.1 短期（現在-6 個月）：不用焦慮，但要開始學

- **Teradar 2028 才上車**，中間 2 年 LiDAR 感知需求只增不減
- 但 **CV 履歷加一行「THz radar signal processing」現在是差異化**，一年後就是必備
- 學什麼？—— **FMCW 波形設計、Range-Doppler FFT、CFAR detection、DoA estimation**。這是 mmWave radar 過去 10 年的成熟工具，網路上開源實作（如 [ti-mmwave-rospkg](https://github.com/radar-lab/ti_mmwave_rospkg)、[OpenRadar](https://github.com/PreSenseRadar/OpenRadar)）都可以 hands-on

### 6.2 中期（1-2 年）：多模融合是下一戰場

- LiDAR + Camera 融合已經是紅海。**LiDAR + Camera + THz radar** 是藍海
- 你的優勢：LiDAR 點雲的座標系統與時序同步經驗，直接可以搬到 THz 4D tensor 上
- 論文 keyword 追蹤：**"THz radar perception"、"4D radar fusion"、"terahertz imaging"、"radar-lidar-camera fusion"**——過去只有學界玩 THz imaging，2026 起會變產業關鍵字

### 6.3 長期（3+ 年）：Foxconn/Nvidia/自駕車廠的角色

- 如果目標仍是 Nvidia 或 tier-1 車廠 R&D，**THz vision 是 Physical AI 的下一個模組**，Isaac stack 遲早會加 THz sensor plugin
- 如果留 Foxconn，Foxconn EMS 客戶多半有車用生產線，THz sensor 組裝與校準會是新產能需求
- **不要只做 LiDAR 專家。做多光譜感知專家**——LiDAR、camera、radar、THz、tactile 全部懂，這才是自駕與 Physical AI 都需要的 T 型人才

---

## 七、結語：LiDAR 產業的第一次「頻段外」挑戰

過去五年 LiDAR 產業裡的競爭都是**同頻段內**的：
- 機械 vs 半固態 vs 全固態
- 905 nm vs 1550 nm
- ToF vs FMCW
- Mechanical scanning vs OPA vs MEMS

Teradar 是第一次有玩家從 **完全不同的電磁頻段**切進來，還宣稱能兼顧解析度與惡劣天氣性能。這代表 LiDAR 廠商過去用「更好的雷射、更快的掃描」建立的護城河，需要重新評估。

物理面看，THz 的 Goldilocks 位置真的存在，這不是純 marketing。工程面看，成熟需要 2-3 年、法規與安規需要 3-5 年。商業面看，Teradar 有 Lockheed 撐腰、$150M 資金、5 家 OEM 在測——這是**真的會發生**的技術轉移。

對 LiDAR 產業，這是**壓力**但不是**世界末日**。對 LiDAR 工程師，這是**擴張感知光譜的 5 年時窗**——現在開始學 THz radar 的 signal processing，你就是下一波感知融合的核心人才。

物理的頻段從來就不只 905/1550 nm 這兩根線。Nova 幫你看 spectrum 的其他角落。

---

## 參考資料

- [Teradar raises $150M for a sensor it says beats lidar and radar](https://techcrunch.com/2025/11/12/teradar-exits-stealth-with-an-all-weather-sensor-for-autonomy-and-150m-in-funding/) - TechCrunch, 2025-11-12
- [Teradar reveals its first terahertz-band vision sensor for cars](https://techcrunch.com/2026/01/05/teradar-reveals-its-first-terahertz-band-vision-sensor-for-cars/) - TechCrunch, 2026-01-05
- [Teradar emerges with $150M backing to replace radar and LiDAR](https://www.freightwaves.com/news/teradar-emerges-with-150m-backing-to-replace-radar-and-lidar-in-autonomous-vehicles) - FreightWaves
- [Teradar Unveils Summit Terahertz Sensor at CES 2026](https://www.webpronews.com/teradar-unveils-summit-terahertz-sensor-at-ces-2026-for-safer-autonomous-driving/) - WebProNews
- [Why some U.S. automakers are adopting lidar after China's boom](https://www.autonews.com/technology/an-lidar-is-coming-to-us-after-china-surge-0305/) - Automotive News
- [Rivian mulls making its own lidar as it builds full autonomous driving stack](https://electrek.co/2026/05/05/rivian-rivn-mulls-in-house-lidar-autonomous-driving-stack/) - Electrek
