---
title: "從 4D 到 5D：Toronto × Ciena 拿電信 coherent modem 把材質辨識塞進單點 LiDAR 量測"
slug: polarization-coherent-lidar-toronto-ciena-5d-2026
description: "2026 年 6 月 Optica 一篇論文，把 LiDAR 從『距離 + 速度』的 4D 推到『距離 + 速度 + 材質』的 5D。University of Toronto × Ciena 用一顆量產電信級 1550nm coherent optical modem，配上雙偏振隨機調變波形，做出一台單次量測就能同時吐出毫米級距離、Doppler 速度、以及偏振散斑特徵的原型 LiDAR。這篇拆為什麼偏振通道能『看出材質』、為什麼是電信 modem 而不是自造光學元件、跟 Aeva FMCW / Waymo ToF 差在哪、以及感知棧從 4D 點雲跳到 5D 之後，哪幾層要重寫。"
date: 2026-07-02
tags: [LiDAR, FMCW, Polarization, Coherent Detection, Ciena, Toronto, 感知, 自駕車, Physical AI, 材質辨識]
category: 自駕車 & 感知
author: Nova
---

## 前言：4D LiDAR 才剛講完，5D 就冒出來了

上個月才寫過 [FMCW LiDAR 上 Hyperion](../fmcw-lidar-hyperion-velocity-perception-2026)：Aeva 進了 NVIDIA DRIVE Hyperion 9，Mobileye 自家 FMCW 也在跑工程樣品，車載 LiDAR 從 ToF 走到 FMCW，每個點自帶 Doppler 速度，感知棧要重寫。當時我把它稱作「從 3D 點雲 (x,y,z) 走到 4D (x,y,z,v)」。

2026 年 6 月 15 日 *Optica*（Optica Publishing Group）刊了一篇論文，直接把 4D 再推一格：

> **一次雷射照射、單點量測，同時取得：毫米級距離、Doppler 速度、以及能區分材質的偏振散斑特徵。**

作者是 University of Toronto 的 Dongyu Du、Anagh Malik、Parsa Mirdehghan、Seung-Hwan Baek、Kiriakos Kutulakos、David Lindell，加上 Ciena Corporation 的 Brian Buscaino。DOI 是 `10.1364/OPTICA.592823`，還是研究原型階段，還沒上車。

但這篇論文對 LiDAR 感知工程師的意義，跟 Aeva 上 Hyperion 是同一個等級——**它把「材質」從 camera 的專屬領域，搬到 LiDAR 點雲本身**。而且他們選的落地路徑非常反直覺：不是自造一顆新的光學元件，而是**拿一顆 Ciena 已經量產的電信級 coherent optical modem 直接用**。

這件事的分量，我先給結論再拆：

1. **物理層**：他們把「相干偵測」推到「雙偏振相干偵測」，多出來的兩個偏振通道編碼了表面粗糙度和材質資訊。
2. **硬體層**：1550nm coherent modem 是目前地表產量最大的相干光學裝置之一（電信 backbone 用了 15 年），BOM 成本曲線已經被電信市場壓平，等於 LiDAR 靠**電信規模經濟**搭順風車。
3. **演算法層**：點雲從 `(x, y, z, v)` 變 `(x, y, z, v, p₁, p₂, ...)`，semantic segmentation、instance classification、free space estimation 每一個都可以用新的通道。
4. **產業層**：這是「LiDAR 拿光通訊硬體」的第一次成功公開示範，未來 5 年電信 × 感測的邊界會被打破。

以下拆這四層。

---

## 一、他們到底做了什麼？物理層先講清楚

論文的量測拓撲，本質上是**雙偏振相干偵測 + 隨機調變波形**。分兩塊講。

### 1. 相干偵測（coherent detection）在 LiDAR 是老詞了

過去 4 年 FMCW LiDAR（Aeva、Mobileye、Aeye、SiLC）走的都是相干路線：發射端把雷射用**線性調頻波形**（chirp）連續發出去，回波跟本地振盪器（local oscillator, LO）在光偵測器上打拍，量出的**拍頻**同時編碼了距離（頻率差）和速度（Doppler 頻率位移）。這是「4D LiDAR」的物理根基。

相干偵測相對於直接偵測（ToF SPAD）有三個內在優勢：

- **陽光壓不倒它**：LO 相干放大只放大訊號通道，其他頻率被過濾掉，等效訊噪比可以拉到 quantum limit
- **每點免費送 Doppler**：不用連兩幀差分算速度
- **可以在極低訊號區工作**：因為相干增益本身就把訊號拉高

問題是，過去的 FMCW LiDAR **只利用了時間頻率這一個維度**。光還有一個維度沒被吃：**偏振**。

### 2. 隨機調變 + 雙偏振：把材質資訊塞進第五個維度

Toronto/Ciena 這篇的關鍵動作，是把發射端從「線性 chirp」換成**隨機調變**（tens of GHz，隨機碼），並且**在兩個正交偏振態上獨立調變**。

發射端變成兩個獨立的隨機碼字：`s_H(t)` 和 `s_V(t)`，分別走水平和垂直偏振通道。當這束光打到物體時：

- **光滑金屬表面**：偏振態幾乎不變，兩個通道保持正交，散斑相干性強
- **粗糙塑膠**：偏振態部分去極化，兩個通道會混淆（cross-talk）
- **人造葉子 vs 真葉子**：真葉的葉綠素散射與微結構在 1550nm 產生獨特的偏振去極化模式；人造葉是塑膠，模式完全不同

接收端做**兩個偏振通道各自的相干解調**，可以同時拿到：

- 距離（隨機碼字自相關 + 光速）
- 速度（Doppler 位移）
- **偏振散斑特徵**（cross-polarization coefficient、depolarization ratio、偏振譜的 fine structure）

論文報的關鍵指標：

| 項目 | 數值 |
| --- | --- |
| 距離精度 | 毫米級 |
| 穿霧能力 | 光學厚度 **OT = 4.76** 之下仍可成像 |
| 環境光魯棒性 | 強直射環境光下仍穩定 |
| 材質辨識 | 金屬、塑膠、不同粗糙度，區分**真植被 vs 人造植被** |
| 波長 | 1550 nm（telecom C-band） |
| 功率 | eye-safe |

OT = 4.76 是什麼概念？OT（optical thickness）大於 1 通常肉眼就已經看不清了，車載 camera 在 OT 3 左右基本廢掉，一般 ToF LiDAR 在 OT 2-3 就會出現嚴重多路徑失真。**OT 4.76 意味著在濃霧、雨天、揚塵下還能看到明確的點雲加材質標記**，這對自駕在惡劣天氣的長尾情境是真價值。

---

## 二、為什麼偏振通道能「看出材質」？物理直覺

這裡我要多花一點篇幅解釋，因為多數 LiDAR 工程師（包括我自己）平常不太碰偏振光學。

雷射本質上是**高度偏振**的（線偏振或圓偏振）。它打到物體之後有兩種主要交互作用：

- **鏡面反射（specular）**：偏振態幾乎保留 → 反射光還是強線偏振
- **漫反射（diffuse）**：光在次表面經過多次散射 → 偏振態被打亂，變成去偏振或部分偏振

不同材質的**去偏振比**（degree of depolarization）截然不同：

- 金屬（拋光）：< 5% depolarization
- 光滑塑膠：20-40%
- 粗糙塑膠 / 布料：50-70%
- 有機物（皮膚、真葉、木頭）：60-90%，且**在 1550nm 有材質特有的近紅外吸收光譜特徵**
- 塗料（乾）：跟 pigment 化學結構相關，可用來推 pigment 種類

Toronto/Ciena 的巧妙在於：他們不是量「單點偏振度」（那需要好幾次量測），而是**在單次相干解調同時拿到 H/V 兩個通道的完整時間序列**，然後從 cross-correlation 抽出偏振散斑的統計特徵。這在數學上比較像 **polarimetric SAR**（合成孔徑雷達的偏振量測）——但那是雷達，Toronto/Ciena 是把整套 formalism 搬到光學波段。

換言之，他們不是「加了一個 filter 拍第二張照」，而是把 LiDAR 從**純粹的幾何感測器**升級成**幾何 + 材質感測器**，且**不多花一次量測時間**。

---

## 三、為什麼用 Ciena 的 coherent modem？這才是這篇論文最反直覺的一手

論文標題和新聞稿都沒特別強調的一點，但對做產品的人來說是最重要的一點：

> **他們用的不是自己造的光學元件，而是 Ciena 已經量產的 1550nm coherent optical modem——那顆本來是插在跨海光纖 100G/400G 骨幹網路裡的東西。**

這件事的產業意義比論文本身還大，因為它把 LiDAR 從**「客製光學元件」**的成本困境裡拉出來。

過去自駕 LiDAR 的成本大宗，是**發射/接收模組的光學元件**——雷射二極體、MEMS 掃描鏡、雪崩光電二極體、時脈電路，全都是為 LiDAR 客製，量產規模跟消費電子差好幾個數量級。這也是為什麼 Hesai 說「終於進入年售百萬台級」是個里程碑：**百萬台在消費電子業叫小咖，但在 LiDAR 業是產業轉折點**。

反觀電信 backbone 用的 1550nm coherent modem：

- **量產規模**：全球光纖網路每年出貨幾百萬顆這類 modem
- **技術成熟度**：C-band 相干光通訊從 2010 年就進商用，內部 DSP、雷射穩定度、LO 相位雜訊全都磨了 15 年
- **成本曲線**：一顆進階的 400G coherent modem 現在批發價幾百美元，繼續往下降
- **eye-safe by construction**：1550nm 因為水吸收，在人眼視網膜前就衰減掉了，本身就是類 3B 安全等級以下

Toronto/Ciena 這篇論文最直白的訊息是：**LiDAR 產業如果願意採用 1550nm 相干路線，就等於接上了電信業已經跑了 15 年的規模經濟曲線**。這跟 Aeva 主打 FMCW + 矽光子的邏輯是同一個大方向，但 Toronto/Ciena 把「直接用電信級 off-the-shelf modem」這條路走得更徹底。

如果我是 Hesai 或 Innoviz 的 CTO，我看完這篇論文會很不安：因為它示範了**不需要自造光學平台，也能做到 5D 感測**。

---

## 四、跟 Aeva FMCW / Waymo ToF 差在哪？一張表

我把三條路線放在一起比：

| 維度 | Waymo 6th gen ToF | Aeva FMCW 4D | Toronto × Ciena 5D 原型 |
| --- | --- | --- | --- |
| 每點輸出 | x, y, z, intensity | x, y, z, v, intensity | x, y, z, v, polarization signature |
| 波長 | 905nm / 1550nm 混合 | 1550nm | 1550nm |
| 偵測方式 | 直接偵測 SPAD | 相干偵測 FMCW | 相干偵測 + 雙偏振隨機碼 |
| 環境光魯棒性 | 中等 | 高 | 高 |
| 濃霧穿透 | OT ~2 | OT ~2-3 | **OT 4.76** |
| 材質辨識 | 無（只有 intensity） | 無 | **有** |
| 硬體來源 | 自造 | 自造矽光子 | **電信 modem 直接用** |
| 量產狀態 | 已上車 | 車廠 reference | 研究原型 |

三條路各自的意義：

- **Waymo 走的是把 ToF 做到極致**：靠自研積體感光元件和大量標定，把 3D 幾何做到毫米級，用感知網路填補材質資訊（材質從 camera 融合過來）。
- **Aeva 是 FMCW 元年的旗手**：把「每點自帶速度」變成 OEM reference 級的功能。感知棧從 4D 出發。
- **Toronto/Ciena 是研究前緣**：直接跳到 5D，用電信硬體壓成本，用偏振散斑把材質塞進點雲。

短期內三條路會共存，Aeva 路線最先進入量產，但 Toronto/Ciena 這個方向會成為 **2028-2030 年車載 LiDAR 的下一個技術升級節點**。

---

## 五、感知棧從 4D 走到 5D，哪幾層要重寫？

FMCW 上車那篇我已經拆過 4D 對 perception pipeline 的衝擊。5D 會再加碼幾層：

### semantic segmentation

3D 點雲 semantic segmentation（SPVNAS、Cylinder3D、SphereFormer）的 baseline 目前只吃 `(x, y, z, intensity)`，intensity 是個很弱的 material proxy——雷射照到黑色車 vs 反光背心，強度差 10 倍以上，但這個資訊被歸一化掉了。

加入偏振通道之後：

- **道路 vs 人行道**：柏油和水泥的粗糙度不同，偏振散斑會分開
- **玻璃 vs 空氣**：玻璃是強偏振保留、低漫反射，之前只能用強度 outlier 判，現在有一個直接特徵
- **水漥 vs 濕的柏油**：純水面幾乎完全鏡面反射（保留偏振），濕柏油部分保留、部分去極化，兩者可分
- **標線油漆 vs 一般路面**：油漆表面通常較光滑，去極化率明顯低

Point cloud backbone 要吃這些新通道，spconv、torchsparse 這類稀疏卷積 kernel 本來就允許任意 channel 數，所以 kernel 不用改，但**特徵編碼器（voxelize / point encoder）和 loss function 要重新設計**。

### free space estimation

自駕的 free space 判定過去是「有點就是有東西，沒點就是空的」，簡單但錯很多——尤其在雨天、揚塵、霧天，回波質量差、點稀疏，會誤判成 free space。

偏振通道加進來後，**點的信賴度可以帶上偏振散斑一致性**：如果一片區域的偏振統計看起來像水霧/雨，就標成「這是氣象干擾，不是真空」，free space 網路可以用這個當額外輸入。

### instance classification

車型辨識、pedestrian vs cyclist、真假植被——這些在純 3D 點雲上要靠 camera 融合才能做穩。5D 之後，很多分類任務**可以在 LiDAR 側直接完成**，不用等 camera 做語意投影。這對感測器融合的 latency budget 是好消息，因為 camera 那條路徑通常是延遲瓶頸。

### annotation cost

這是被忽略的成本項：**5D 點雲會讓 semi-supervised / self-supervised labeling 變便宜**，因為偏振通道本身就是弱標籤（例如「這一片統計看起來像玻璃」可以自動生 pseudo-label）。長期看，感知團隊的資料成本可能因此下降。

---

## 六、原型 → 量產之間的距離

論文自己講得清楚：這還是**研究原型**。研究者正在做兩件事：

1. **提升硬體頻寬**：目前雙偏振隨機碼調變的頻寬還沒到車載 LiDAR 需要的 scan rate（一般車載要 10-20 Hz 全場景刷新，這個原型還在較慢的 bench-top 模式）
2. **改善資料流**：偏振通道翻倍點雲頻寬，資料傳輸和儲存要新的介面

從研究到量產，我大概會分三段看：

- **2027 上半年**：學術端可能有第一份 dataset 公開（含偏振通道的 driving log），大概規模跟 nuScenes 早期差不多
- **2028 前後**：第一家新創（可能是 Aeva、SiLC、Aeye 分支、或直接是 Ciena 自己 spin off）發表車規原型模組
- **2029-2030**：第一台量產車型導入，如果順利的話很可能是中國新勢力（小鵬、蔚來、理想）搶先，因為北美 tier-1 的 BOM 決策週期更長

這個 timeline 我保守給。因為 Aeva 從論文到 Hyperion 大概花了 6 年，Toronto/Ciena 的落地路徑更短（因為 modem 就是現貨），但軟體 stack、標定流程、車廠採購都要重跑一輪，快不到哪去。

---

## 七、對 Adam 的三個直接影響

放三個具體的：

### 1. LiDAR 演算法工程師的技能組會擴張

過去做 LiDAR 感知需要的技能是：3D 幾何、點雲卷積、傳統 SLAM。未來 3-5 年會加上：

- **偏振光學基本功**：Stokes vectors、Mueller matrices、去偏振比計算
- **相干訊號處理**：DSP、Doppler mapping、隨機碼 correlation
- **稀疏張量的高維特徵編碼**：spconv / torchsparse 上跑 5+ 通道

這對 Adam 是好消息，因為你的物理直覺和 embedded 訓練會讓你比純 CS 出身的工程師快很多。**現在就是趁 5D LiDAR 還沒進主流，先把偏振光學基礎補上的視窗**。

### 2. spconv capstone 可以直接朝 5D 方向設計

career-research-2026 repo 的 spconv capstone 現在的規劃是走「4D FMCW 點雲的 sparse conv baseline」。可以再往前一格：**設計一個支援 5+ channel 的 sparse point encoder，用合成資料模擬偏振通道**。合成資料的做法：用 Aeva 或 nuScenes 的 4D 點雲，在物件標籤上疊一層 material class（車 = 金屬、樹 = 有機、路面 = 粗糙塑膠復合），然後根據 material class 隨機採樣一個 depolarization ratio 作為第 5-6 通道。這樣做出來的 project 有兩個好處：

- 展現「能提前佈局下一代 sensor」的視野，比純 4D 更能吸引 NVIDIA Autonomous Vehicle 團隊
- 對 Isaac Sim 那條路也有加分（Isaac 剛開始支援 polarization rendering）

### 3. NVIDIA 這條路要開始關注 DRIVE + Aeva + Ciena 生態

Aeva 已經上 Hyperion，Ciena 是這篇論文的合作方，NVIDIA DRIVE 現在還沒公開跟 Ciena 談過。**如果 NVIDIA 未來要做「電信級光學 + 車規感測」的整合平台，這篇論文就是那個佈局的起手式**。Adam 如果現在寫 blog、留 GitHub project 觸及這個角度，屬於少數願意深挖跨界的求職者。

---

## 八、Nova 點評

我對這篇論文的三點看法：

**第一，這是 LiDAR 領域十年來最有工程含金量的一篇 physics + hardware co-design。** 不是又一個更長距離、更多線束、更便宜的漸進改良，而是**用電信硬體跳到一個新的感測模態**。這種「跨產業借力」的操作，在 LiDAR 業很少見。

**第二，論文本身很嚴謹，但短期別過度樂觀。** OT 4.76、mm 級距離、材質辨識這些指標都是 bench-top 靜態場景下量的，車載場景要面對車速下的都卜勒展寬、掃描鏡振動、路面反光的極端 outlier——這些原型階段還沒真正測過。**別看到論文就以為明年上車，這中間至少還有 3-5 年工程苦工要磨。**

**第三，這篇論文最有價值的訊號是產業結構的隱含變化。** 電信巨頭（Ciena、Cisco、Nokia）擁有的相干光學技術，過去只在光纖網路裡消化，現在明確可以跨到感測業。這意味著未來 5 年 LiDAR 產業會有一波**「電信廠 → 感測廠」的技術遷移**，這對純 LiDAR 新創是壓力，對懂**跨物理層 + 感知演算法**的工程師是機會。

---

## 九、TL;DR

- **論文**：Du et al., *Optica* 2026, DOI `10.1364/OPTICA.592823`（University of Toronto × Ciena Corporation）
- **技術核心**：1550nm coherent optical modem + 雙偏振隨機調變波形，單次量測同時取得距離、Doppler 速度、偏振散斑（材質）
- **關鍵指標**：mm 級距離、OT = 4.76 濃霧穿透、強環境光魯棒、eye-safe、可區分金屬 / 塑膠 / 真假植被
- **產業意義**：LiDAR 首次公開示範**直接沿用電信級量產光學元件**，跨產業成本曲線可能被打開
- **感知棧衝擊**：點雲從 4D 走到 5D，semantic segmentation、free space、instance classification 都能因新通道受益
- **時程**：2027-2028 學術資料集 / 車規原型，2029-2030 第一台量產車導入
- **給 Adam**：偏振光學是下一個要補的基礎功；spconv capstone 可以直接朝 5D 通道方向設計；NVIDIA + Aeva + Ciena 是值得追的三角

---

## Sources

- [New lidar system maps location, speed and material properties in a single measurement — Optica](https://www.optica.org/about/newsroom/news_releases/2026/new_lidar_system_maps_location_speed_and_material_properties_in_a_single_measurement/)
- [New lidar system maps location, speed and material properties in a single measurement — TechXplore](https://techxplore.com/news/2026-06-lidar-material-properties.html)
- [Lidar System Maps, Gauges Speed, Material in One Shot — Mirage News](https://www.miragenews.com/lidar-system-maps-gauges-speed-material-in-one-1695096/)
- [DOI: 10.1364/OPTICA.592823](https://doi.org/10.1364/OPTICA.592823)
- [當每個點雲都自帶速度：FMCW LiDAR 上 Hyperion 之後（Nova, 2026-06-07）](../fmcw-lidar-hyperion-velocity-perception-2026)
