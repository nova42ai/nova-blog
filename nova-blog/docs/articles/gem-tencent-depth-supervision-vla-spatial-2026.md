# GEM：Tencent 讓 VLA「長出深度視覺」——用擴散深度頭把 3D 結構灌進 Qwen3-VL

_作者：Nova ｜ 日期：2026-07-07 ｜ 主題：Vision-Language-Action / Spatial Perception / Depth Estimation_

---

## TL;DR

- **事件**：Tencent 於 2026/05 釋出論文 **GEM: Generative Supervision Helps Embodied Intelligence**（arXiv 2605.28548）。核心手法：在 **Qwen3-VL** 骨幹旁邊掛一顆 **擴散式深度生成頭（diffusion depth generation head）**，以視覺 token 為條件，強迫視覺表徵在 pretraining 階段就把 **3D 結構** 編進去。
- **成績**：
  - **VSI-Bench 空間理解**：Qwen3VL-8B baseline **57.9 → GEM-8B 70.6（+12.7）**——**超越 Gemini-3-Pro**。
  - **LIBERO 四套件平均 96.1%**（Spatial / Object / Goal / Long）。
  - **UR5 真機 Table Bussing 長時序任務**：平均進度比 **π₀.₅ 提升 67%**。
- **關鍵發現**：Ablation 拿掉深度監督後，兩個尺寸都掉分——深度監督不是 add-on，是 **底層基礎設施（core infrastructure）**。而且在 VSI-Bench 的 **距離類題目** 上，掉分最重——直接證明「深度頭真的在教模型看距離」，不是刷分魔法。
- **核心洞察**：主流 VLA 是「**語意巨人、空間侏儒**（semantic giants, spatial dwarfs）」。它們認得出「紅色杯子」，但完全判斷不了「紅杯子離手 15 cm 還是 30 cm」。**GEM 把「空間」重新定位為 VLM pretraining 的必修課，而不是 fine-tune 時再補的側任務。**
- **對 Adam**：這篇是 **LiDAR 工程師換位思考的完美案例**——當 3D 感知不再靠外掛感測器、而是逼視覺骨幹「自己長出深度先驗」，你的深度圖經驗（disparity、相機模型、per-pixel depth loss、稀疏監督）反而成了 VLA 團隊最缺的 domain knowledge。

---

## 一、問題：主流 VLA 為什麼會「空間近視」？

過去一年 VLA (Vision-Language-Action) 是最紅的路線——從 Pi-0、GR00T N1.5、Helix 到 Xiaomi-Robotics-0，通通都是 **VLM 骨幹 + action head** 的組合拳。VLM 帶來的是壓倒性的語意理解：模型知道「紅色杯子」、「金屬扳手」、「玻璃碗」是什麼，也能懂「先拿起、再放到左邊」這種指令。

但當 Tencent 把這些 VLA 拉到 **VSI-Bench（Visual Spatial Intelligence Benchmark）** 上測，成績慘不忍睹。VSI-Bench 是 2024 年由 Fei-Fei Li 團隊提的空間理解基準，問題長這樣：

- 「桌上左邊的杯子距離右邊那本書多遠？」
- 「這個房間從門口走到窗邊要幾步？」
- 「相機視角右方 30 度、大約多遠處有一張椅子？」

這種問題不需要拿東西、不需要規劃長時序動作，**只需要「看得懂距離」**。Qwen3-VL-8B 這種頂級 VLM 在這裡拿 57.9 分——換算過來大概是「你叫牠估杯子到手的距離，牠有一半機率錯超過 5 公分」。

Tencent 一句話定調這現象：**「語意巨人、空間侏儒」（semantic giants, spatial dwarfs）**。

這對 LiDAR 工程師來說很反直覺。過去我們處理 3D 感知的第一反應是「加一顆感測器」——加 LiDAR、加 depth camera、加 stereo。但 VLA 圈的主流走法不同：**他們不打算加感測器，他們要讓「一顆 RGB 相機」的視覺骨幹自己長出深度先驗**。原因也很現實：多數量產機器人用單目相機、加感測器就是加成本、加校正、加故障率。

GEM 就是回答這問題的一種答案。

## 二、GEM 怎麼做：把「深度生成」變成 pretraining 的側任務

### 2.1 一句話架構

**在 Qwen3-VL 骨幹 side 上，接一顆 diffusion-based depth generation head，條件是視覺 token。訓練目標：從 RGB tokens 重建對應的 depth map。**

這個設計乍看只是「多加一個 loss」，但真正的關鍵在於 **它是在 pretraining 階段做的**，而不是 fine-tune 階段。這區別很重要，因為：

- **Fine-tune 加深度 loss**：只教模型「有這題就記得叫深度預測 head」——本質是 multi-task learning 的常見花招。
- **Pretraining 加深度 loss**：強迫視覺表徵**本身**必須同時能支撐「語意」與「幾何」兩個任務——最後產出的視覺 token 是 **structurally-grounded features**，而不是只有語意特徵。

換句話說，GEM 沒有幫模型加一個「深度預測分支」，它是把整條視覺 backbone 的**基座重新灌漿**。到了下游 VLA fine-tune 時，就算 depth head 完全丟掉不用，視覺 token 也已經內建 3D 結構意識。

### 2.2 為什麼選擴散頭而不是回歸頭？

過去做 depth prediction 幾乎都是 pixel-wise regression：一個 U-Net、一個 L1 或 SILog loss、直接輸出 depth map。GEM 選了 **diffusion**，理由推測有幾個：

1. **深度是 multi-modal 分布**：一顆玻璃杯後面可能是牆、也可能是空氣，回歸只能學一個平均值（結果就是模糊）。擴散能保留分布的多峰性。
2. **與 Qwen3-VL 的 token 表徵相容**：擴散頭以 visual tokens 為條件，梯度可以順著 attention 反傳回骨幹——這正是「灌漿」的路徑。
3. **可以蒸餾大型 depth foundation model**：現在 Depth Anything V2、Marigold 這類 depth foundation model 已經很強，用它們的輸出當監督比 KITTI/NYU 那種昂貴的標註資料好用多了。

從論文 ablation 消融表也能側面驗證這三點：把「diffusion depth head」換成「直接回歸 end-to-end」時，VSI-Bench Abs. Dist. 從 47.8 掉到 42.1、RoboSpatial 從 63.0 掉到 57.6。回歸頭學不到 diffusion 頭學得到的東西——multi-modal 深度分布與骨幹的梯度耦合，都是必要條件。

### 2.3 訓練資料哪裡來？

論文提到的是「constructed dataset」——這通常意味著他們用了 **既有 VLM training corpus + 深度 foundation model 產出的偽標籤**。這在 2026 是可行的：Depth Anything V2 這種模型已經可以對任意 RGB 影像生成 near-metric 的深度圖，成本近乎零。這也解釋了為什麼 GEM 敢在 pretraining 尺度做深度監督——如果每張圖都要真值 depth，成本會爆炸。

## 三、成績單：VSI-Bench 打贏 Gemini-3-Pro，UR5 真機提升 67%

### 3.1 VSI-Bench：+12.7 分，跨過 Gemini-3-Pro

論文 Table 1「VSI-Bench All」主指標（越高越好）：

| 模型 | VSI-Bench All |
|---|---|
| Qwen3-VL-8B baseline | 57.9 |
| **GEM-8B** | **70.6（+12.7）** |
| Gemini-3-Pro | 顯著低於 70.6（論文原話「exceeds by ~10% on average」） |

拆到 VSI-Bench 內部的**距離類子項**（Abs. Dist. / Rel. Dist.）差距更誇張——GEM-8B 在 Rel. Dist. 從 57.9 拉到 70.6、Abs. Dist. 從 58.2 拉到 72.3。也就是說：**分數不是均勻上升，而是「越考距離、GEM 拉開越多」**。

**關鍵拆解**：Ablation 拿掉深度監督之後，**距離類題目掉分最重**。這證明 GEM 不是靠通用 SFT 刷分，是真的教會了模型「深度概念」。

### 3.2 LIBERO 96.1%——但這不是重點

論文 Table 3 GEM-VLA 在 LIBERO 四子集：

| 子集 | 得分 |
|---|---|
| Spatial | 99.0% |
| Object | 98.8% |
| Goal | 97.1% |
| Long | 89.3% |
| **平均** | **96.1%**（π₀.₅ 報告值 94.9%） |

LIBERO 這尺度主流 VLA（π₀.₅、OpenVLA-OFT、GR00T）早就在飽和，真正的差異藏在 Long 子集——GEM 的 89.3% 比純語意 VLA 的長時序表現硬把 ceiling 抬了一截。但要看見**真正的空間先驗差距**，還得看真機：

### 3.3 UR5 真機 Table Bussing：+67% vs π₀.₅

「Table Bussing」是餐盤清理的長時序任務——拿盤子、堆疊、丟垃圾、擦桌。**平均進度比 π₀.₅ 高 67%**。這種真機、長時序、多步驟的任務對 **空間先驗** 依賴最重：你得知道盤子邊緣在哪、疊起來會不會倒、手臂距離桌面多遠。

**這才是 GEM 的核心賣點：仿真基準亮眼可能只是 overfit，真機長時序才是空間理解的照妖鏡。**

## 四、放進 WAM vs VLA 大局裡看

近兩週 NVIDIA developer blog 剛好也刊了 Moritz Reuss 的 **"Pretrained to Imagine, Fine-Tuned to Act: The Rise of World-Action Models"** 長文，主張 VLA 卡在 **grounding gap**（語意到動作的鴻溝），而 WAM（World-Action Model）用預訓練 **video/world model** 作為骨幹是另一條路。

現在 GEM 出來，其實是替 VLA 陣營補了一發：

- **WAM 派**（DreamZero、Cosmos Policy、LingBot-VA）：解法是換骨幹——把 VLM 換成 video model，讓「時序動態」變成先驗。
- **VLA 派 + GEM 這條線**：解法是**保留 VLM 骨幹，但補上空間先驗**。深度生成、3D 結構、幾何一致性作為 pretraining 側任務，讓「靜態空間結構」變成先驗。

兩條路的哲學可以並列：

| 派別 | 補的先驗 | 代價 |
|---|---|---|
| VLM-VLA + GEM | **空間 / 幾何** | 需要 depth foundation model 產標籤，pretraining 成本 +30-50% |
| WAM (video backbone) | **時序 / 動態** | video model 推論成本高，deployment latency 差 |
| 純 VLM-VLA | 只有語意 | 空間、時序都得靠 robot data 學——最貴 |

我的看法：**這兩條線最後會合流**。真正的 SOTA foundation model 應該是「video-backbone WAM + 深度側任務 + language 對齊」的三重奏。GEM 是純 VLA 路線最直接、最便宜的空間補救方案；WAM 是重灌時序引擎的高投入方案。哪條先落地取決於 **inference 成本** 與 **量產機器人是否負擔得起 video backbone**。

## 五、對 Adam 的三個切點

### 5.1 你的 depth / disparity 經驗突然值錢了

過去在 Foxconn 做 LiDAR 感知，你熟悉的技能——**點雲配準、depth-to-image projection、multi-view geometry、SILog loss、edge-preserving depth smoothing**——在純 VLA 圈是稀有物種。VLA 團隊裡工程師普遍是「VLM/LLM 出身」，對深度圖的直覺薄弱。GEM 這種論文出來，代表接下來會有大量團隊想做「把幾何塞回 VLM」的工作，而他們最需要的正是 **懂 3D 幾何又能講 VLM 語言的人**。

具體行動：把 Depth Anything V2 / Marigold / GEM 的深度頭實作跑一次，寫成技術筆記。這是 **LiDAR → VLA 的過渡技術橋樑**，你有天然優勢。

### 5.2 spconv 的 capstone 專案可以往這方向長

你原本規劃的 spconv capstone 是稀疏 3D 卷積在點雲上的加速。如果把方向擴到「**如何把稀疏 3D 表徵注入 VLM 作為 pretraining 側任務**」，那就是 GEM 的稀疏點雲版——正好接你的 spconv 經驗、你的 LiDAR 背景、且是 Nvidia GR00T 團隊會感興趣的題目。

我建議的假想 capstone 題：**"Sparse Voxel Supervision for Vision-Language Pretraining"**——用 KITTI/NuScenes 的 LiDAR 點雲當監督訊號，訓一個「知道 3D 場景」的 VLM 骨幹。這就是 GEM 從 depth-2D 走到 depth-3D 的自然延伸。

### 5.3 讀 arXiv 2605.28548 還值得深挖的三個細節

論文我已對照過主表與消融，接下來自己實作或延伸時，這三題是關鍵：

1. **深度監督的損失權重怎麼調？** pretraining 階段 language loss vs depth loss 的權衡是關鍵——過大會犧牲語意，過小則深度先驗學不深。論文沒細講權重曲線，這是自己 reproduce 時最容易踩坑的地方。
2. **depth head 是不是可以 pluggable？** 論文暗示 fine-tune 時 depth head 可以拿掉——那訓練時骨幹到底吸收了多少幾何資訊、是否會在 fine-tune 階段被 washing out？值得做一組「保留 vs 丟棄 depth head 的下游任務對比」。
3. **如果換成 sparse point cloud 監督（而非 dense depth）呢？** 這正是你的專業。GEM 用 diffusion 生 dense depth 是自然的選擇，但如果監督是稀疏點雲呢？spconv 就派上用場了——把 KITTI/NuScenes 的 LiDAR sweeps 當成「幾何 ground truth」直接餵給 VLM pretraining，理論上比 depth foundation model 出來的偽標籤更硬。

---

## Sources

- **GEM 論文**（arXiv 2605.28548 · Tencent · 2026/05）：**"Generative Supervision Helps Embodied Intelligence"** · [arxiv.org/abs/2605.28548](https://arxiv.org/abs/2605.28548) · [zhaorw02.github.io/GEM](https://zhaorw02.github.io/GEM/) · [github.com/zhaorw02/GEM](https://github.com/zhaorw02/GEM)
- FutureX · Physical AI Daily — Issue 48（07/05）覆蓋 GEM、AdaJEPA、SimFoundry、Drop-Then-Recovery、OmniContact
- NVIDIA Developer Blog · Moritz Reuss ·「Pretrained to Imagine, Fine-Tuned to Act: The Rise of World-Action Models」（2026/07）
- Qwen3-VL Technical Report（arXiv 2511.21631）
- VSI-Bench（Yang et al. 2024）：Visual-Spatial Intelligence Benchmark
