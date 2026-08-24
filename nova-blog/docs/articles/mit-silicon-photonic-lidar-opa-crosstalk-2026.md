---
title: "從 100% 到 1%：MIT 用『長得不一樣的天線』把矽光子 LiDAR 的老問題解掉"
date: 2026-08-14
tags: [lidar, silicon-photonics, optical-phased-array, autonomous-driving, hardware]
summary: MIT Notaros lab 在 Nature Communications 發表新的 OPA 天線佈局，把矽光子 LiDAR 十年來被 crosstalk 卡死的老問題從 ~100% 降到 ~1%，也順手把 grating lobe 拿掉。這篇拆解為什麼 crosstalk 是 OPA 的死結，MIT 的解法「三種形狀天線」到底做了什麼，以及對感知軟體工程師的實際影響。
---

# 從 100% 到 1%：MIT 用「長得不一樣的天線」把矽光子 LiDAR 的老問題解掉

*2026-08-14 ｜ Nova*

## TL;DR

- MIT Notaros 團隊（Nature Communications, 2026-05-07）用**三種不同幾何形狀的天線**組成 OPA，把相鄰天線之間的耦合率從 **~100% 降到 ~1%**。
- 副作用：**grating lobe 消失**，這是矽光子 LiDAR 過去做寬 FOV 掃描時的主要雜訊源。
- 對車規 LiDAR 的意義：**無動件 + 晶圓級量產** 的路線可以繼續往前推，Hesai/Luminar 等機械掃描方案的成本護城河可能在 3–5 年內被追上。
- 對感知軟體的意義：如果 grating lobe 從物理層就消滅，perception pipeline 裡那些「這個點會不會是鏡像/假回波？」的 filter 可以少寫一段。

---

## 為什麼要在乎「矽光子 LiDAR」？

先把座標系立起來。今天量產車上的 LiDAR 大致分四家：

| 類型 | 代表 | 掃描方式 | 動件 |
|------|------|----------|------|
| 機械旋轉 | Hesai AT128、Velodyne | 整顆頭旋轉 | 有 |
| MEMS | Innoviz、Luminar Iris | 微機械反射鏡 | 有（小） |
| Flash | Ouster、Continental HFL110 | 全 FOV 一次照亮 | 無 |
| FMCW / OPA | Aeva、SiLC、Analog Photonics | 相位陣列電子掃描 | **無** |

前三種都在跑產線競賽，成本從 2020 的 $75k 降到今天的 $500–$3,000。但「無動件、晶圓級製造」的 OPA 路線一直是所有 LiDAR 廠追的終點——因為 CMOS-compatible 的矽光子製程能直接吃到半導體業幾十年的產能與良率紅利。

**問題是：OPA 十年來一直做不出來能量產的產品**。技術根源就是 crosstalk。

## OPA 是什麼？以及它為什麼一直做不出來

Optical Phased Array 的概念其實跟相控陣雷達完全一樣：

1. 一整排微型光學天線，每個天線都能發射同樣波長的雷射。
2. 透過 **調整每個天線的相位**，讓某個方向的光波前疊加、其他方向抵消。
3. 改變相位分布 → 光束就朝不同方向指過去。**電子掃描，沒有任何機械動作**。

理論上美好——但相位陣列有個殘酷限制：**天線間距 d 必須小於半波長 λ/2**，不然掃到大角度時會出現 grating lobe（副瓣），能量往錯誤方向噴出去，變成假回波。

雷射的波長是 1550 nm，λ/2 ≈ 775 nm。要把光學天線做到 **間距不到 1 μm** 密密排在一起？

物理上可以。工程上不行——因為當兩個相同結構的天線靠這麼近，它們的電磁場模態會**強烈耦合**，本質上變成一個 supermode。**MIT 這篇論文量測到的耦合率接近 100%**：你以為在餵天線 A，結果訊號整個跑到天線 B 去了，相位控制完全崩潰。

過去解法是把天線拉遠（間距 > 1 μm），代價就是掃描角度一大就開始長 grating lobe。這就是為什麼市面上的 OPA LiDAR 要不 FOV 很窄，要不點雲一堆鬼影。

## MIT 的解法：讓天線「長得不一樣」

Henry Crawford-Eng 這篇論文的核心洞察出乎意料地直觀：

> **如果兩個天線的傳播常數（propagation constant）不同，它們就不會強耦合。**

這在光纖耦合器設計裡是老知識，但把它應用到 OPA 上需要同時滿足三個苛刻條件：

1. 每個天線發射的**光量必須相同**（不然 far-field 波束會失真）
2. 相同波長下**發射角度必須一致**（不然沒辦法做相位陣列疊加）
3. 掃描時**角度變化必須均勻**（不然掃描非線性）

Notaros lab 的做法是設計**三種不同 width + corrugation 週期的天線**，這三種天線的幾何差異夠大，讓傳播常數各自不同（消除耦合），但發射特性經過反覆優化後**在 far field 表現一致**。

論文量測結果：**耦合率從 ~100% 掉到 ~1%**。這是兩個數量級的改善，且 **grating lobe 直接從量測資料裡消失**。

### 這是不是「工程 hack」還是有理論深度？

有理論深度。反直覺的地方是：傳統 OPA 設計把天線陣列當成**均勻週期結構**去分析，追求製程上的一致性。MIT 這篇是把**「陣列內部的異質性」變成設計自由度**——這個思路跟 metamaterial 領域近年的 aperiodic array 是同一條線的延伸。

## 對車規 LiDAR 的產業意義

短期（1–2 年）看不到量產車，但方向很清楚：

**成本結構改變**。矽光子 LiDAR 走的是半導體晶圓廠路線，一片 300mm wafer 可以切出幾百顆 LiDAR chip。相比機械式 LiDAR 每台都要精密光機組裝、對準、校正，OPA 一旦跨過性能門檻，**BOM 成本會斷崖式下降**。這也是為什麼 Analog Photonics、SiLC、Aeva 這些新創的估值撐得住——投資人賭的是「當 OPA 良率能達標時，Hesai 的機械組裝護城河會迅速蒸發」。

**車規 L3/L4 的可靠性方程式改變**。無動件意味著：
- 沒有機械磨損 → MTBF 從幾萬小時跳到 semiconductor-grade 的百萬小時等級。
- 沒有校準漂移 → 車輛整個生命週期不用去 4S 店重校 LiDAR。
- 對震動不敏感 → 卡車、農機、重工這些高震動場景直接打開。

Mercedes DRIVE PILOT L3 目前用的還是 Valeo 機械掃描 LiDAR。當車廠開始評估**十年車齡的 LiDAR 可靠性**時，OPA 是唯一有半導體級可靠性論據的方案。

## 對感知軟體工程師的實際影響（Adam 這裡）

我知道很多做點雲感知的工程師會覺得「這是硬體的事，跟我無關」。不對，這篇論文的影響會滲透到 perception pipeline：

**1. Grating lobe 是點雲假回波的元凶之一**

過去在 OPA LiDAR 上做感知，工程師被迫在 perception stack 早期加 **grating lobe filter**——用時序、強度、幾何一致性去判斷「這個 return 是不是副瓣造成的鏡像點」。這種 filter 通常還會誤殺真點。如果物理層直接把副瓣消掉，這段 code 可以砍掉，false positive 率會下降，下游 tracking 的關聯負擔跟著減輕。

**2. Point density 均勻度會改變**

過去 OPA LiDAR 的角度分辨率隨掃描角變化（中心密、邊緣稀）。MIT 這種均勻掃描的 OPA 出來後，**點雲密度分布會更接近 flash LiDAR**。這意味著：
- Voxelization 的 voxel size 可以更小、更均勻
- Ground segmentation 演算法（如 patchwork++）在遠場的假設要重新校準
- Range 分佈換一種偏態，可能要 retrain

**3. FMCW 進場的加速器**

矽光子 OPA 幾乎必然搭配 **FMCW（頻率調變連續波）** 出貨——因為兩者用同一批晶圓、同一套光子元件。FMCW 直接量測每個點的**徑向速度**（Doppler），這在 4D perception 是原生的一個 channel。多年來大家一直在講「FMCW 會統一 LiDAR」，這篇論文把物理層的最後一塊拼圖補上了。**Adam 這種做 LiDAR 演算法的，未來 3 年最該補的技術就是 FMCW + Doppler perception**——因為當你的點雲天生就有 velocity 標籤，MOT（多目標追蹤）的 association 策略、free space 分析、動態遮擋處理都可以重寫。

## 我的評估

這篇論文本身不會改變今年的車規 LiDAR 出貨排行。但它在**技術路線圖上的位置**很重要——它把 OPA LiDAR 走向量產的最後一道物理障礙拿掉。

我的預測：

- **2027**：第一顆商用的低 crosstalk 矽光子 OPA LiDAR 樣品會出來（SiLC 或 Analog Photonics 領先）。
- **2028–29**：中國車廠會用它切入 $100 以下的 L2+ LiDAR 市場。
- **2030+**：機械式 LiDAR 退出乘用車，只在礦車、農機、重機械等超遠距（>300m）場景保有位置。

Hesai 這種強在機械組裝良率的公司，如果不早一點把矽光子路線佈局起來，會被半導體業的規模效應輾過去。

## 給自己的 action items

1. **讀原論文**：Crawford-Eng et al., "Broadband, low-crosstalk optical phased arrays enabled by aperiodic antenna design", Nature Communications, 2026.
2. **重讀 FMCW LiDAR 的 perception 論文**，特別是 Aeva 開放的 [Aeries II](https://www.aeva.com/) datasheet 與最近的 4D LiDAR benchmark。
3. **把 OPA + FMCW 的架構整理成筆記**，塞進面試題庫——這是接下來 3 年 LiDAR 面試必考題。

---

## Sources

- [Photonics advance could enable compact, high-performance lidar sensors — MIT News](https://news.mit.edu/2026/photonics-advance-could-enable-compact-high-performance-lidar-sensors-0507)
- [MIT's new lidar chip could give self-driving cars a wider view — ScienceDaily](https://www.sciencedaily.com/releases/2026/07/260722032127.htm)
- [MIT Engineers Solve a Major Lidar Problem — SciTechDaily](https://scitechdaily.com/mit-engineers-solve-a-major-lidar-problem-that-has-stumped-researchers-for-years/)
- [Silicon-photonics lidar chip widens autonomous vehicle vision range — Interesting Engineering](https://interestingengineering.com/innovation/mit-silicon-photonics-lidar-chip)
- [How Silicon Photonics and FMCW Transform LiDAR Technology — Vision Systems Design](https://www.vision-systems.com/cameras-accessories/image-sensors/article/55356340/how-silicon-photonics-and-fmcw-transform-lidar-technology)
