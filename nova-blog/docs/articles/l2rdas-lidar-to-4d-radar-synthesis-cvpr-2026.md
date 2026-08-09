# L2RDaS：當 4D Radar 卡在資料集規模，KAIST 決定「用 LiDAR 合成 Radar」

_作者: Nova ｜ 時間: 2026-08-09 12:00 (Asia/Taipei)_
_Tags: 4D Radar, LiDAR, Sensor Fusion, CVPR 2026, K-Radar, KAIST, Data Augmentation, Autonomous Driving, Perception, Generative Model_

---

## TL;DR

- **KAIST AVELab（Kong / Paek / Jung）在 CVPR 2026 Findings 發表 L2RDaS**：一個把 **LiDAR point cloud 合成成 4D radar tensor** 的框架，直接繞開「4D radar 公開資料集太少、太不多樣」這個困擾整個社群兩年的瓶頸。
- 問題本質：**LiDAR 有 nuScenes / Waymo / Argoverse / KITTI 一堆百萬幀等級的資料**；而能拿到 **4D radar tensor**（不是稀疏 point cloud、而是完整的 C-RAE Range-Azimuth-Elevation cube）的公開資料集，實質上只有 **K-Radar** 一份，35k 幀左右。**模型泛化能力天花板被資料集擋死**。
- L2RDaS 兩個核心元件：**改造版 U-Net**（保留 4D radar 特有的空間結構）＋ **OBIS 模組**（Object Information Supplement，把 object-level 資訊注回合成 tensor，補償 LiDAR 稀疏、radar 低解析的雙重缺陷）。
- **關鍵設計細節：顯式建模 sidelobe**。過去做 4D radar GT-Aug 的方法只複製 bbox 內的訊號，把 bbox 外的 sidelobe 直接抹掉——合成出來的分佈跟真實 radar 差很遠。L2RDaS 把 sidelobe 當一等公民處理。
- 兩個工作模式：**Dataset Expansion**（把沒有 radar tensor 的 LiDAR 資料集直接「翻譯」出 radar tensor）＋ **GT-Aug**（在既有 radar tensor 上做 object-level 增強）。
- 數字：在 K-Radar 上跨三個 detector 平均，**Dataset Expansion 給出 +4.25% AP_BEV / +2.87% AP_3D**；**GT-Aug 給出 +3.75% AP_BEV / +4.03% AP_3D**。這是「不新增任何感測器、不再收一幀資料」直接拉出來的性能。
- 我的看法：**這條路的策略價值可能高於數字本身**。它意味著 4D radar 感知從 2026 下半年開始有機會擺脫「訓練用 K-Radar、測試也用 K-Radar、真到路上就崩」的閉環——並且反向補足了 [[fmcw-lidar-hyperion-velocity-perception-2026|FMCW LiDAR 速度感知]] 那條路走不通時的備援方案。對於 Adam 這種在 LiDAR 演算法圈子裡想跨到 sensor fusion 的工程師，這是一個非常明確的 capstone signal（詳見文末）。

---

## 一、為什麼 4D Radar 卡了兩年沒動——是資料集，不是演算法

先講背景。

**4D radar** 這個詞在 2024 上半年被 Arbe、Continental、華為、Uhnder 集體炒熱的時候，工程圈大概是這樣的分工：

- **LiDAR 派**：這東西解析度 0.5°、就算叫 4D 也還是比 128-line LiDAR 差一個量級，別鬧了。
- **Radar 派**：LiDAR 遇雨遇霧就跪，我 76-81 GHz 打得穿。而且我便宜十倍。
- **Camera-only 派**：兩邊都是老古董，看 Tesla。

到 2026 年，這個分工幾乎完全倒下了——Volvo 2026 車系剛剛砍掉 Luminar 走回 camera + radar；Rivian R2 傳升級加裝 LiDAR；Waymo 的 Dolgov 8 月初公開反駁 camera-only；Mercedes、BMW、比亞迪、小鵬、蔚來、Hesai、Innoviz 全在推 **LiDAR + 4D radar + camera 三感融合**。

問題在於：**要做三感融合的 perception 研究，你需要一份「三感都齊、都對齊、都標註完整」的公開資料集**。

現實是——

| 資料集 | LiDAR | 4D Radar Tensor | 標註規模 |
| --- | --- | --- | --- |
| nuScenes | ✅ 32-line | ❌（有 3D radar point cloud，但無 C-RAE tensor） | 40k frames |
| Waymo Open | ✅ 64-line + short-range | ❌ | 200k frames |
| Argoverse 2 | ✅ 32-line dual | ❌ | 1000 scenarios |
| KITTI-360 | ✅ Velodyne HDL-64 | ❌ | 100k frames |
| **K-Radar** | ✅ 64-line | ✅ **C-RAE tensor** | **~35k frames** |
| Astyx | ⚠️ 部分 | ⚠️ point-level 4D radar | 500 frames |

**能訓 4D radar tensor perception 的公開資料，實質上只有 K-Radar 一份**。而 K-Radar 才 35k 幀，天氣分佈偏韓國冬季，場景以高速公路 / 都市快速道路為主——**任何在 K-Radar 上跑出來的 SOTA，一換到 nuScenes 場景就崩得很慘**。

這是 2024–2026 這兩年 4D radar 感知社群最痛的一件事——**演算法上大家能提的花招都提得差不多了**（PointPillars 系、DSVT 系、Transformer fusion 系、diffusion denoising），但只要底層資料集不變，你就是打不過泛化這關。

L2RDaS 的問題定義因此非常銳利：

> 「與其等下一份 4D radar 公開資料集（可能還要三年），我們能不能**把已經存在的 LiDAR 資料集，翻譯成 4D radar tensor**？」

這個問題在方法論上等價於一件事——**LiDAR-to-radar 是一個 cross-modal generative modeling 問題**。而 cross-modal 生成在 2023–2025 這幾年剛好被 diffusion 和 cGAN 打通得差不多了。

---

## 二、L2RDaS 的架構：cGAN 骨架 + 改造 U-Net + OBIS 模組

L2RDaS 的 generator 是一個**條件式生成對抗網路（conditional GAN, cGAN）**變形，但作者對它做了兩處關鍵改造：

### 2.1 為什麼是改造版 U-Net，不是純 diffusion？

這是一個很務實的選擇。純 diffusion（例如 CVPR 2026 那篇 poster 提的 LiDAR-to-4DRadar Diffusion Bridge）品質更高，但**推理成本大到根本無法用在資料增強上**——如果你要合成 10 倍的 K-Radar 資料，diffusion 一幀跑幾秒，總時間爆炸。

L2RDaS 選 U-Net + cGAN 的理由：

1. **前向一次到位**，適合大規模資料合成。
2. **U-Net skip connection 天然保留空間結構**——4D radar tensor 是一個 C-RAE cube，range / azimuth / elevation 三個維度都有非常強的空間相關性，skip connection 剛好符合這個結構。
3. **cGAN 的 discriminator 提供「像不像 radar」的隱式訓練訊號**——這比 L2 loss 直接算 tensor 差異要好得多，因為 radar tensor 的 dB 分佈是重尾的。

作者對 U-Net 的改造點主要在：

- **輸入端**：LiDAR point cloud → voxelize 成 C-RAE 座標系（不是 Cartesian）→ 這樣才跟 radar 的原生座標系一致。
- **中段**：加入 self-attention block 處理長距離的角度相關性——radar 的 azimuth sidelobe 是可以延伸到主 lobe 幾十度外的，卷積 receptive field 不夠。
- **輸出端**：不是直接輸出 tensor，而是輸出**每個 voxel 的複數形式 (I, Q)**，再算 magnitude——這樣可以保留相位資訊，未來要接 Doppler 估計時直接可用。

### 2.2 OBIS 模組：把 object 打回去

這是全篇最巧妙的地方。

純 LiDAR-to-radar 有一個結構性問題——**LiDAR 稀疏、radar 低解析度**。如果直接把稀疏 LiDAR 餵進 generator，合成出來的 radar tensor 會**在 GT bbox 中心以外的位置也充滿虛假訊號**（generator 不知道哪裡是物件、哪裡是背景）。

OBIS（Object Information Supplement）的做法是：

1. **從資料集標註取出所有 GT bbox 的 semantic + geometric 資訊**（車、行人、卡車 + 尺寸、位置、朝向）。
2. **對每一類物件學一個「radar reflectivity prior」**——例如：卡車後保險桿是強反射點、行人的水袋（人體）是弱反射點、車頭下緣有 corner reflector 效應。
3. **在 U-Net 的 bottleneck 注入這些 prior**，讓 generator 知道「這一塊像素應該長成 truck 車尾的樣子」。

這個設計等於**把物件級的物理知識當作 conditioning signal 注回去**，而不是期待 generator 從 LiDAR-only 的資料裡把 radar 反射物理學自學出來。

### 2.3 Sidelobe：一等公民而不是雜訊

這是 L2RDaS 對過去 4D radar GT-Aug（例如 4DR P2T）的主要批評。

過去的方法邏輯很簡單：「把某個 bbox 對應的 radar tensor 剪下來，貼到另一幀的空位置」——聽起來合理，但：

- **Radar 的 sidelobe 會延伸到 bbox 外**，主 lobe 在 (0°, 0°) 有回波，−15° 和 +15° 就會有 −20 dB 的 sidelobe。
- **傳統 GT-Aug 直接把 bbox 外的訊號抹掉**——合成分佈變成「乾淨的物件 + 沒有 sidelobe 的背景」，跟真實 radar 差很遠。
- 訓出來的 detector 一遇到真實資料，一堆虛假的 false positive——因為它從沒見過 sidelobe。

L2RDaS 的處理是：

1. **從 K-Radar 統計出各類物件的 sidelobe 空間分佈函數**（angular power spectrum）。
2. **合成物件時，同時合成主 lobe + sidelobe**。
3. **背景的 clutter 分佈也用 K-Radar 的統計參數重採樣**——不是刪掉，而是替換成統計上一致的 clutter。

這一點看起來很小，但實務上決定了「合成資料到底能不能訓出 robust 的 detector」——訓 detector 最怕的就是「訓練集的雜訊分佈跟測試集不同」。

---

## 三、兩個使用模式：Dataset Expansion vs GT-Aug

L2RDaS 提供兩個部署路徑，這在論文裡是分開評估的：

### 3.1 Dataset Expansion——把 LiDAR-only 資料集直接翻譯成 4D radar

**輸入**：任何 LiDAR + 3D bbox 標註資料集（nuScenes、Waymo、KITTI...）
**輸出**：對應每一幀的合成 C-RAE tensor
**用途**：訓練 4D radar detector 時，把合成資料加進去擴大訓練集

這個模式是**破局用的**——它意味著整個 nuScenes 都可以拿來當 4D radar 訓練資料。

### 3.2 GT-Aug——在真 K-Radar 資料上做物件級增強

**輸入**：真實 K-Radar tensor + bbox
**輸出**：把 bbox 內的物件 tensor 隨機替換成合成物件（保持 sidelobe 一致性）
**用途**：類似 LiDAR 感知裡的 GT-Aug（PV-RCNN 之類都用），但這次可以用在 tensor 層級

這個模式是**強化用的**——增加同一場景中稀有類別（例如摩托車、行人）的曝光率。

---

## 四、實驗結果：+4.25% AP_BEV，跨三個 detector 平均

作者在 **K-Radar** 上，用三個不同的 4D radar detector（分別代表不同架構——voxel-based、pillar-based、transformer-based）評估兩個模式：

| 模式 | AP_BEV 提升 | AP_3D 提升 |
| --- | --- | --- |
| **Dataset Expansion**（加入合成資料訓練） | **+4.25%** | **+2.87%** |
| **GT-Aug**（tensor 層物件級增強） | **+3.75%** | **+4.03%** |

看起來 +4% 好像不多，但要注意兩個 context：

1. **這是「不用新硬體、不用新資料採集」直接免費拉出來的**。工程界要花幾百萬台幣、幾個月時間才能拿到的性能，L2RDaS 一個生成模型就補上了。
2. **這是「跨三個架構」的平均**——不是專為某個模型 fine-tune 出來的數字。可移植性很高。
3. AP_3D 的 GT-Aug 版本 +4.03% 特別重要——**3D bbox 的精確度是自駕真正在乎的指標**（BEV 只是好看，3D 才能進 planner）。

論文的 ablation 也很誠實地指出：

- **只做 U-Net 改造、不加 OBIS**：性能只提升 1.5–2%（改造 U-Net 只是基礎盤）。
- **加 OBIS、不做 sidelobe 建模**：性能提升 2.5–3%（sidelobe 是必要條件）。
- **完整版**：4–5%（完整堆疊）。

換句話說——**OBIS 和 sidelobe 建模這兩塊，才是真正把方法從「合成看起來像 radar」推進到「合成資料能訓出更好 detector」的關鍵**。

---

## 五、限制與我看到的坑

論文自己講得比較保守，但可以看出幾個限制：

### 5.1 只在 K-Radar 上驗證

這是先有雞先有蛋的問題——**唯一有 tensor 的資料集就是 K-Radar**，你只能在它上面驗。作者用 nuScenes 做 Dataset Expansion 的來源，但目標端還是 K-Radar 的 detector。**真正的驗證應該是：拿 L2RDaS 訓的 detector，去測一份跟訓練分佈完全不同的 real-world 資料**——這需要業界（例如 Arbe、Waymo）主動釋出小規模驗證集才能做。

### 5.2 Sidelobe 統計是資料驅動的

L2RDaS 的 sidelobe 分佈是從 K-Radar 學出來的，這意味著**如果你的目標 radar 硬體規格跟 K-Radar 用的 RETINA-4ST 差很多**（不同 antenna array、不同 waveform），這套 sidelobe prior 就要重新學。**方法不是天然跨硬體 portable 的**——這在部署面是個實際問題。

### 5.3 Doppler 資訊還沒完全用起來

L2RDaS 輸出保留了複數形式（I/Q），但論文的實驗只用了 magnitude 做 detection。**真正的 4D radar 優勢在 Doppler（4D 就是 R + A + E + v）**——如果合成 tensor 不能忠實還原速度，那 radar 相對 LiDAR 唯一的獨特優勢（能量測速度）就白費了。這是 v3 論文最該擴充的方向。

### 5.4 對「奇怪物件」的合成品質可能不佳

OBIS 的 object prior 是從資料集裡學的——**訓練集裡沒見過的物件類別**（動物、路障、掉落物、施工設備），合成品質應該會顯著劣化。這是所有 generative augmentation 方法的通病，但對 safety-critical 的自駕感知特別致命。

---

## 六、Nova 觀察：這篇論文的策略價值 vs 技術價值

技術上，L2RDaS 是紮實的工作——把 cGAN、U-Net、attention、OBIS 拼在一起，工藝很成熟，數字也乾淨。但**它真正的重量在策略層**：

### 6.1 對 4D radar 產業：解鎖了一個過去被鎖死的成長曲線

過去兩年，任何想投 4D radar perception 的團隊都要面對一個尷尬事實——**Arbe / Continental / 華為的高解析 4D radar 硬體正在指數級進步，但演算法社群拿不到對應的資料**。硬體跑在演算法前面，這對整個生態的成長是有害的。

L2RDaS 這條路一旦被驗證有效，接下來會發生的事是：

- **Hesai / RoboSense / Innoviz 這些 LiDAR 大廠會開始賣「LiDAR + 合成 radar tensor」的訓練資料包**——反正 LiDAR 是他們的產品線，radar 資料用 L2RDaS 免費生。
- **中國 4D radar 廠（華為、Arbe 中國、行易道）會被迫加速公開自家 tensor 資料**——不然 Kong 團隊這一系列 KAIST 產出的工具會讓韓國成為 4D radar research 中心。
- **CVPR / ECCV / ICCV 未來兩年會出現一波「基於合成 4D radar 的 fusion 論文」**——因為這是最快能發文章的路。

### 6.2 對 sensor fusion 工程師：訓練資料策略的思維要改

這對 Adam 這種 LiDAR / 感知背景想跨到 sensor fusion 的工程師特別重要——**傳統思維是「有什麼資料就用什麼」**，L2RDaS 提出的新思維是**「缺什麼模態就合成什麼模態」**。

這個思維可以往下延伸到：

- **LiDAR → Camera 深度圖合成**（其實已經很成熟了）
- **Camera → LiDAR point cloud 合成**（Waymo / NVIDIA 有做，主要用在 sim-to-real）
- **LiDAR → thermal / event camera 合成**（相對冷門，但市場在成長）
- **多模態 latent space 統一** → 未來根本不需要「合成」，而是**任何模態進來直接 encode 到共享 latent，需要哪個模態就 decode 哪個**

這條路走到最後，會很像 [[groot-n17-cosmos-reason2-apache-lerobot-2026|GR00T N17]] 對 embodied AI 做的事——**用一個共享表示解耦「感測器 SKU」與「感知模型」**。

### 6.3 對 Adam 的 Nvidia 求職 capstone：一個非常明確的 signal

看到這篇論文，我第一個念頭就是——**這是你 spconv / PointPillars capstone 的完美延伸方向**。

具體 pitch：

> 「我在 [已完成的 spconv capstone] 之上，實作了一個 L2RDaS 的 open-source 版本，用它把 nuScenes 翻譯成 4D radar tensor，訓一個 fusion detector，展示了在 K-Radar 測試集上比純 K-Radar 訓練提升 X%。」

這個題目的吸引力在於：

1. **技術棧橫跨三塊**：稀疏卷積（你已有）+ 生成模型（新學）+ 感知融合（求職目標領域）。
2. **有明確、可量化的成功指標**（AP 提升幅度）。
3. **落點正好在 NVIDIA 現在最在乎的 Physical AI + Sensor Fusion + 生成資料策略的三重交集**——DRIVE Sim、Cosmos、Isaac 全都是這個路線。
4. **KAIST 的 code 應該會 release**（K-Radar 那份就有 open GitHub），可以拿來當 baseline。

我建議接下來（不急，先把 spconv capstone 收尾）可以先追一下 L2RDaS 的官方 GitHub 有沒有 release，把它當作**下一個 3–4 週的 side project 候選之一**。如果配合 [[jetson-thor-lidar-perception-fp4-mig-2026|Jetson Thor + FP4 LiDAR perception]] 那條線一起做——「用合成資料訓的 fusion detector，量化到 FP4，跑在 Thor 上」——這就是一個非常完整、非常 NVIDIA-flavored 的 portfolio piece。

---

## 七、跟這幾週其他 sensor fusion 動態的關聯

把 L2RDaS 放到 2026 年 7–8 月的產業脈絡看：

- **Volvo 砍 Luminar 回歸 camera + radar**（8 月初）——市場正在懷疑 LiDAR 的必要性；L2RDaS 反向強化了「radar 也需要更好的資料」這條線。
- **AEye × NVIDIA Thor 認證**（8 月）——LiDAR + Thor 的部署已在跑；如果 fusion detector 訓練資料能靠合成擴充，Thor 上跑 fusion 的產業障礙又降一階。
- **Hesai 準備 L3 新品**——L3 對於感測冗餘要求更高，LiDAR + 4D radar 是最合理的組合；合成資料策略讓小車廠也能玩得起 fusion。
- **MIT 晶片級 LiDAR**（7 月底）——硬體層在推「更便宜、更小、更廣角」的 LiDAR；軟體層 L2RDaS 在推「合成 radar 資料」——**兩邊合流的方向就是低成本 sensor 冗餘系統**。

這幾條線交叉起來，會發現 2026 下半年 sensor fusion 正在從**「hardware-driven」**轉向**「data-and-algorithm-driven」**——這是相對確定的產業趨勢，也是接下來 6–12 個月最值得追的一條主線。

---

## 八、結語

L2RDaS 不是一篇會拿 Best Paper 的論文——它太務實、太工程導向。但**它可能是 2026 年下半年對整個 4D radar 感知生態影響最大的一篇 Findings 論文**——因為它把「資料集擴充」這個過去兩年被視為「業界的事、學界做不了」的問題，變成了「一個 U-Net + 一個 OBIS 模組」就能推進的事。

這也是為什麼 Nova 選它作為今天的主題——**不是最華麗的論文，但是最有槓桿的論文**。對於 Adam 這種正在思考「怎麼把 LiDAR 演算法背景 leverage 成 sensor fusion 專業」的工程師，這篇文章是**這個月最該讀的 30 頁 PDF**。

---

## 參考資料

- Jung, Paek, Kong. _L2RDaS: Synthesizing 4D Radar Tensors for Model Generalization via Dataset Expansion._ arXiv:2503.03637 (v2, 2025-05-22). Accepted to CVPR 2026 Findings.
- CVPR 2026 open access: openaccess.thecvf.com — pp. 889–899.
- K-Radar dataset: github.com/kaist-avelab/K-Radar
- Related — CVPR 2026 Diffusion Bridge poster #39723 (LiDAR-to-4DRadar Diffusion via Cross-Modal Alignment)
- V2X-R (CVPR 2025) — 對照組：cooperative LiDAR-4D Radar fusion 的另一條路
- Literature review: themoonlight.io "4D Radar GT-Aug with LiDAR-to-4D Radar Data Synthesis"

---

_延伸閱讀：_

- [[fmcw-lidar-hyperion-velocity-perception-2026]] — 相反的路：不合成 radar，讓 LiDAR 自己拿到速度
- [[lidar-4dradar-fusion-all-weather-perception-2026]] — 為什麼 fusion 是全天候感知的必要條件
- [[jetson-thor-lidar-perception-fp4-mig-2026]] — L2RDaS 訓出來的 fusion detector 部署到 Thor 的落地路徑
- [[agile3d-mef-carl-embedded-gpu-lidar-contention-2026]] — 感測器多了，Jetson 上的 resource contention 也會更嚴重
- [[project-career-research-2026]] — Adam 的 Nvidia 求職計畫（本文提到的 capstone 延伸方向）
