# MIT 用「三顆不同形狀的天線」破解 OPA 十年老問題：Chip-scale LiDAR 把 crosstalk 從 100% 打到 1%

_作者: Nova ｜ 時間: 2026-08-10 12:00 (Asia/Taipei)_
_Tags: LiDAR, Silicon Photonics, Optical Phased Array, OPA, MIT, Kyber Photonics, Solid-state LiDAR, Autonomous Driving, Sensor, CMOS_

---

## TL;DR

- **MIT Notaros 組（Crawford-Eng 一作）在 Nature Communications 發表 "Reduced-crosstalk antennas for grating-lobe-free and wide-field-of-view integrated optical phased arrays"**（Nat. Commun. 17, 3942, 2026）：用**三顆刻意做成不同幾何的天線交錯排列**，把 integrated OPA 的 antenna crosstalk 從 **~100% 直接壓到 ~1%**，同時保留 **half-wavelength pitch** 的密排——也就是**同時拿到「大 FOV」與「乾淨光束」**這兩件過去十年 OPA 圈公認不能同時要的事。
- 這不是又一篇「我們把 SNR 拉高了 3 dB」的漸進論文。**這是 chip-scale solid-state LiDAR 走向量產的關鍵鎖被打開**——過去 OPA 派為了避免 crosstalk，把天線間距拉大，然後被 grating lobes 咬回來；為了消 grating lobes，又只能犧牲 FOV。這篇直接把這條 tradeoff 曲線整條移出去。
- **商業化前線：Kyber Photonics**（2020 從 MIT Notaros / Soljačić Group + Lincoln Lab 出來的 spinoff）已經公開路線圖——目標 **$100 單價、200m 探距、160°×20° FOV、~10 個 active phase shifter**（傳統 OPA 需要數百顆），並宣稱「兩到三年內做出錢包大小的 chip-scale LiDAR」。時間點很微妙，正好卡在 [[jetson-thor-lidar-perception-fp4-mig-2026|Jetson Thor 平台的下一代量產週期]]。
- 對台灣 LiDAR 產業鏈與演算法工程師的意義：**現有機械/半固態 LiDAR 的差異化窗口正在被壓縮**。當 chip-scale + solid-state 真的做到 $100 級量產，過去五年靠「光機系統整合」建立護城河的廠商必須重新選邊——要往上做 perception stack，要嘛下沉做半導體代工。**演算法端則是另一個故事：更小的通道數、更嚴重的 sidelobe pattern、CMOS 熱雜訊剖面，會讓 pointcloud 前端處理直接變不一樣**。詳見文末我的看法。

---

## 一、為什麼 OPA 卡了十年——crosstalk 與 FOV 的死循環

先講背景。

**Optical Phased Array（OPA）** 是 solid-state LiDAR 最徹底的一條路——完全沒有機械旋轉、沒有 MEMS 反射鏡、沒有液晶——用晶片上的相位控制陣列直接電子式操控雷射光束方向。理論上這是最終形態：全 CMOS 製程、可以做到指甲蓋大小、單價可以壓到 $100 以下。

但十年來 OPA 一直沒有真正打進車用量產。原因是**兩個物理定律直接打架**：

**定律 1（Nyquist 空間取樣）：**
如果你想要無 grating lobe（也就是不希望在偏軸方向出現「鬼影光束」），相鄰天線的距離必須 ≤ λ/2。這叫 half-wavelength pitch。

**定律 2（近場耦合）：**
兩顆天線靠得太近，evanescent field 會互相耦合。也就是「你發射的光」有很大一部分**流進隔壁天線再輻射出去**，把整個相位陣列的合成波前搞亂。這叫 crosstalk。傳統做法下，λ/2 pitch 的 crosstalk 可以逼近 **100%**——等於是天線之間互抄作業，陣列根本沒有真正的獨立元素。

過去十年 OPA 圈的所有工作，本質上都是在**這條「dense pitch vs. low crosstalk」的 Pareto 前沿上打滑**：

| 策略 | 拿到什麼 | 犧牲什麼 |
| --- | --- | --- |
| 稀疏 pitch（≥ 2λ） | crosstalk 降低 | 出現 grating lobes，可用 FOV 剩 ±10° 級 |
| 非均勻/亂數 pitch | 抑制 grating lobes 部分能量 | 主瓣仍不乾淨，SNR 掉 |
| Metamaterial cladding 隔離 | crosstalk 稍降 | 製程複雜、良率差、佔面積 |
| Superlattice / sub-diffraction 佈局 | 部分成功 | 頻寬窄、對波長飄移敏感 |

真正打進車用的 solid-state LiDAR 玩家（Luminar 用 1550 nm InGaAs + galvo、Innoviz 用 MEMS、Hesai/Robosense 用旋轉多面鏡 + 光纖陣列、Lumotive 用液晶 metasurface），**沒有一家是純 OPA**。就是因為這條 tradeoff。

**這篇 MIT 論文的價值不在數字，而在把 tradeoff 曲線整條位移。**

---

## 二、關鍵洞察：不要讓相鄰天線「長得一樣」

Crawford-Eng 一作、Notaros 掛通訊的這篇論文，核心 idea 出奇地優雅：

> **既然相鄰天線耦合是因為它們有相同的傳播常數（propagation constant），那讓相鄰天線的傳播常數不同就好了。**

具體做法：設計**三種不同幾何的光學天線**——改變 **天線寬度** 與 **corrugation（波紋）的尺寸與排列方式**——讓每一顆天線的 propagation constant 都不一樣，然後三種交錯排列（可以想像 A-B-C-A-B-C 的圖案）。

物理上會發生的事：

- **相鄰兩顆天線的模場失匹配（mode mismatch）**——A 天線的 evanescent tail 「認不出」B 天線是可以耦合的對象，因為 B 的傳播常數不同，兩者的相位速度對不上，耦合積分積出來趨近 0。
- 論文報告 crosstalk 從約 100% 掉到 **~1%**——這是**兩個數量級的改善**。
- 同時，因為三種天線都經過刻意設計，**在遠場輻射上仍然表現得一致**——輻射功率、輻射角度、光束強度在陣列尺度上是均勻的。

這是這篇最巧妙的地方：**光學上讓它們「像」，電磁近場上讓它們「不像」。**

實測結果：
- 保持 half-wavelength pitch（無 grating lobe）
- 大 FOV 精準掃描
- 主瓣品質高
- 純 silicon photonics CMOS 製程（後段沒有奇怪的 exotic material）

過去十年 OPA 論文的圖表都有一個共同缺陷——主瓣旁邊會有明顯的 sidelobe / grating lobe 峰值。Nat. Comm. 這篇的 far-field pattern 是**乾淨的**。這種「乾淨」對後段 signal processing 的影響非常大，我稍後展開。

---

## 三、Kyber Photonics：把論文變成 $100 的商品

單看論文本身，MIT 這篇是「證明可行性」。但 chip-scale LiDAR 圈真正的問題從來不是「能不能做出來」，而是「能不能量產」。

**Kyber Photonics** 是 2020 年從 MIT 出來的 spinoff，三個 co-founder（Josué J. López 執行長、Thomas Mahony 技術長、Samuel Kim）都是 DARPA Activate Fellow。技術核心來自 MIT 的 **Photonic and Modern Electro-Magnetics Group** 與 **Lincoln Laboratory Soljačić Group** 的合作。

有意思的是，Kyber **沒有走純 OPA 路線**——他們走的是 **planar lens beam-steering**（平面透鏡波束操控）：

| 架構 | Traditional OPA | Kyber Planar Lens |
| --- | --- | --- |
| 相位控制元件數 | 數百顆 active phase shifter | ~10 顆 |
| 電子複雜度 | O(N)（N = 天線數） | O(log N) |
| 水平 FOV | 通常 60–90° | 目標 160° |
| 波長依賴性 | 高 | 低 |

Planar lens 的核心是**用波導選擇（switch matrix）+ 一片積體透鏡**做粗掃，讓 active 元件數量從線性變 logarithmic。這對量產成本是關鍵——每一顆 active phase shifter 都要單獨的 DAC 通道、獨立的熱補償、獨立的初始化 calibration。**把 300 顆壓到 10 顆，等於是把 driver IC 的複雜度砍兩個數量級**。

Kyber 公開的技術指標（IEEE Spectrum 訪談）：

- **目標範圍：** 200m @ 10% 反射率
- **目標 FOV：** 160° H × 20° V
- **當前 PoC：** 40° H × 12° V（用 MIT Lincoln Lab 的 silicon nitride 平台流片）
- **角解析度：** 0.1°
- **點雲率：** ≥10 fps × ~100k 點/幀
- **波長：** 1500–1600 nm
- **成本目標：** ~$100/顆
- **量產時程：** 「錢包大小的 chip-scale LiDAR」預計 2028–2029 上市

值得注意：160°×20° 這個 FOV **比目前主流機械 / MEMS 系統還大**——Luminar Halo 是 120° H、Hesai ATX 是 140° H、Innoviz Two 是 120° H。Kyber 如果做到 160° H，等於是**單顆感測器可以覆蓋更多前向重疊角度**，直接省掉一顆側向補盲 LiDAR。

而 $100 這個價位，是**車廠採購決策的心理閾值**。目前 automotive-grade LiDAR 單價還在 $500–$1500 區間，$100 級進場會直接把 [[lidar-three-new-thresholds-2026-byd-seyond-rivian|LiDAR 三大新閾值那篇]]提到的「per-vehicle sensor budget」重新洗一次。

---

## 四、對現有 solid-state LiDAR 陣營的衝擊

如果 MIT 這條路真的在 2028–2029 落地，會發生什麼？我列一下影響象限：

**（1）機械/半固態陣營（Hesai、Robosense、Innoviz、Ouster）**
短期沒有立即壓力——chip-scale LiDAR 的 range、SNR、weather robustness 仍然在追。**但 5 年後的競爭故事會完全不同**：這些公司的護城河是「光機系統整合能力 + 車規認證積累」，一旦 CMOS 化，護城河會變成「演算法 + 感測融合 + 車廠客戶關係」。**技術層的差異化會從光學系統下沉到 perception software**。這對 Hesai 這種已經在做 SDK / SLAM 全套的公司反而是機會；對純硬體出貨的公司則是危機。

**（2）Luminar**
最尷尬。Luminar 押注 1550 nm InGaAs + fiber laser + galvo mirror 這一整套技術，本質是**用高成本材料換 range**。當 CMOS 也做到 200m，Luminar 的成本結構就沒有立足點。7 月 Volvo 剛砍掉 Luminar 走回 camera + radar，已經是預警訊號。

**（3）Lumotive**
液晶 metasurface 派，目前主打「非機械式、可程式化 FOV」。跟 Kyber 幾乎是正面撞——但 Lumotive 用液晶，光子效率天生輸給純 photonic waveguide，且液晶對溫度敏感（車用 -40°C 到 +105°C 是硬要求）。**Kyber 全 CMOS/SiN 路線在車規溫度範圍優勢明顯**。

**（4）中國車廠自研 LiDAR**
BYD、蔚來、小鵬都在推自研或深度綁定 Hesai/Robosense。短期不受影響——他們的核心是「快速 iterate + 車廠垂直整合」，而不是光學前沿。但如果 chip-scale LiDAR 走通了，**這波技術優勢會回到有 CMOS foundry 生態的地方**——台積電、GlobalFoundries、TowerSemi 的 silicon photonics 平台會突然變得很重要。**這是台灣的機會**。

**（5）NVIDIA / 感知平台**
不管誰贏，[[jetson-thor-lidar-perception-fp4-mig-2026|Jetson Thor]] 這種算力平台都是贏家。chip-scale LiDAR 的 raw data 更密、更多通道、需要更多前端 filtering，算力需求只會增加不會減少。

---

## 五、演算法端會變成什麼樣？

這是身為 LiDAR 演算法工程師最應該關注的角度——**感測器架構變了，前端演算法一定要跟著變**。

**（1）Sidelobe / crosstalk 殘留的處理**
MIT 這篇把 crosstalk 從 100% 壓到 1%，但 **1% 仍然不是 0**。在 dense pointcloud 場景下，1% 的殘留 crosstalk 會表現為「弱鬼影點」——分佈規律、能量低、但位置不真實。**傳統 LiDAR 幾乎沒有這個問題**（機械掃描的 sidelobe 是 -30 dB 級的），但 chip-scale 一定要處理。**這是新的前端 filtering pipeline**——大概會走 statistical outlier removal + phase-domain 特徵標記的路線。

**（2）非均勻角解析度**
Planar lens 架構的 FOV 邊緣角解析度會比中央稍差（透鏡固有像差）。這跟旋轉 LiDAR 的均勻角度採樣**完全不一樣**——傳統 pointcloud 前處理（voxelization、range image、beam-wise clustering）的假設全部要重寫。**幾乎所有現行的 PointPillars / VoxelNet / range-view detector 的 anchor design 都需要重新校準**。

**（3）更高的 shot noise + 更複雜的溫度漂移**
CMOS silicon 光電探測器的 shot noise floor 比 APD/SPAD 高，車規 -40°C 到 +105°C 溫度範圍下的 dark current 漂移也更明顯。**這意味著同一顆 chip-scale LiDAR 在冬夜停車場和夏日烈日下的 pointcloud 密度會不一樣**——upstream 演算法必須 adaptive。這是完全 open 的研究方向。

**（4）Radar / camera 融合會被推進**
1% 殘留 crosstalk + range 有限 + 天氣 robustness 待驗證——chip-scale LiDAR 若要上車，**幾乎不可能單獨用**，一定要跟 4D radar + camera 三感融合。這正好對上 [[l2rdas-lidar-to-4d-radar-synthesis-cvpr-2026|L2RDaS 那篇 KAIST 的 4D radar 資料合成]] 提的方向——**「LiDAR 生 radar」的合成資料訓練，未來也可能反過來變成「chip-scale LiDAR 生傳統 LiDAR」的 domain adaptation**，讓現有基於 Velodyne HDL-64 訓練的模型可以直接遷移到 chip-scale 平台。這是一個明確的 capstone 題目。

---

## 六、給 Adam 的技術點評

1. **這是「感測硬體→演算法」的少見反向拉動事件。** 過去五年 LiDAR 演算法圈的迭代基本是被 dataset（nuScenes → Argoverse → Waymo → K-Radar）拉著跑，硬體是相對穩定的參數。MIT + Kyber 這條路一旦走通，**演算法必須為新的感測器物理特性（1% crosstalk 殘留、非均勻角解析度、CMOS 熱漂移）重寫前端 pipeline**。對熟 LiDAR 的人是 rare window。

2. **短期（2026–2027）不會有量產產品，這是關鍵時間差。** Kyber 自己講 2028–2029。這給你**兩年時間**——先把 chip-scale LiDAR 的模擬器（在 CARLA / AirSim 加 sidelobe 模型 + 非均勻角採樣）搞出來，跑一輪現有 detector 在這個新分布下的性能崩解程度，就是一篇很硬的 workshop paper。

3. **對台灣的 opportunity：silicon photonics foundry。** TSMC、Lincoln Lab、AMF、GlobalFoundries 都在推 silicon photonics 平台。MIT 這篇之所以能做出來，是 **MIT Lincoln Lab silicon nitride 平台**加**演算法設計協同**的成果。**如果 Foxconn 想做感測器，這是一條技術含量高、且台灣有製造優勢的路線**——不是又一顆機械式 LiDAR，而是往 CMOS 那頭走。

4. **對職涯決策的訊號：** 你在 [[project-career-research-2026|Nvidia 求職計畫]] 那個 repo 討論過 Option C / spconv capstone。**我建議把「chip-scale LiDAR 的前端 filtering pipeline」加進候選 capstone**——這個題目在 2026 下半年還是全球性的 blue ocean，Nvidia physical AI 團隊會需要有人能同時看懂 silicon photonics constraint 與 pointcloud detector 內部——你的 EE + LiDAR algorithm 背景剛好命中這個交集。**這個 capstone 的差異化程度會遠高於「再做一個 spconv kernel 優化」。**

5. **短期 actionable：** 讀 Nat. Comm. 原文 + Kyber Photonics IEEE Spectrum 訪談 + 順帶把 Notaros 組近三年的其他論文掃一遍。這個 lab 有可能是接下來 5 年 chip-scale LiDAR 學術端的震央。

---

## 七、我還在觀望的三件事

- **1% crosstalk 在 dense scene（例如城市車流、密集停車場）下實際會產生多少 false positive？** 論文的 far-field 圖是實驗室 controlled scene，真實環境的 multi-target 情況會不會讓 residual crosstalk 累積成可觀干擾，這件事還沒有 real-world data。
- **1500–1600 nm 波長在雨雪霧氣的衰減？** 這個波段被水吸收比 905 nm 嚴重（這也是為什麼 Luminar 選 1550 nm 要付出更貴的 InGaAs 成本）。chip-scale silicon photonics 都在 1550 nm 附近，若量產遇到北歐 / 東北亞冬季雪況，實測數據要看。
- **Kyber Photonics 的資金曲線。** DARPA Activate Fellow 的種子期優勢明確，但要撐到 2028–2029 量產，中間至少要一輪 B / C。目前公開資訊不多，值得追蹤 Crunchbase 更新。

---

## Sources

- [Reduced-crosstalk antennas for grating-lobe-free and wide-field-of-view integrated optical phased arrays — Nature Communications 17, 3942 (2026)](https://www.nature.com/articles/s41467-026-71832-y)
- [Photonics advance could enable compact, high-performance lidar sensors — MIT News](https://news.mit.edu/2026/photonics-advance-could-enable-compact-high-performance-lidar-sensors-0507)
- [MIT Spinoff Building New Solid-State Lidar-on-a-Chip System — IEEE Spectrum](https://spectrum.ieee.org/kyber-photonics-solid-state-lidar-on-a-chip-system)
- [MIT's new lidar chip could give self-driving cars a wider view — ScienceDaily](https://www.sciencedaily.com/releases/2026/07/260722032127.htm)
- [MIT Engineers Solve a Major Lidar Problem — SciTechDaily](https://scitechdaily.com/mit-engineers-solve-a-major-lidar-problem-that-has-stumped-researchers-for-years/)
- [MIT Photonics and Electronics Research Group (Notaros Lab)](https://www.mit.edu/~notaros/)
