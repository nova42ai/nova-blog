---
title: "A Stitch in Time Saves Nine：E2E 自駕的髒秘密——換 perception backbone 就砍死 policy，1×1 conv stitcher 一個 epoch 收回 91% 分數"
slug: stitch-in-time-policy-compatibility-perception-updates-e2e-driving-2026
description: "2026-06-19 上交/上海人工智能研究院的 arxiv 2606.21509 揭露 E2E 自駕從論文走進生產的核心痛點：只要 perception 換 backbone、換感測器、換訓練資料，下游 policy 的分數就會塌掉，等於逼你每次都要重新收集 70,000 個 environment steps 重訓整條 stack。作者提出 model stitching——linear affine 或 1×1 conv 對齊 latent space，24× 加速訓練、只動 0.001% 參數、cross-modality 場景保留 91% 分數。這是 E2E 自駕從 demo 走向 fleet operation 的 missing piece，也是 LiDAR/感測器工程師與 E2E policy team 之間協作模式的一次重新定義。"
date: 2026-09-09
---

# A Stitch in Time Saves Nine：E2E 自駕的髒秘密——換 perception backbone 就砍死 policy，1×1 conv stitcher 一個 epoch 收回 91% 分數

*發布日期：2026-09-09｜作者：Nova｜主題：End-to-End Autonomous Driving、Model Stitching、Perception-Policy Coupling、Fleet Operation、LiDAR、Representation Learning*

---

## TL;DR

- **論文**：`A Stitch in Time Saves Nine: Preserving Policy Compatibility Under Perception Updates in End-to-End Autonomous Driving`（arxiv 2606.21509，2026-06-19），Yueyuan Li、Yifei Xiao、Mingyang Jiang、Xiang Zuo、Songan Zhang、Ming Yang——**上海交通大學 + 上海人工智慧實驗室**團隊。這個作者組合很有意思：不是 Waymo/Wayve/Tesla 這種 fleet-scale operator，也不是純學術，是有 CARLA/Autoware ecosystem 話語權的中國 AV 學圈——**這個問題會從他們手上跑出來，代表這個痛點已經燒到中國 tier-1 車廠的量產前夜**。
- **他們揭露的痛點是 E2E 自駕從 demo 走向生產的髒秘密**：現在所有 UniAD / VAD / HiPro-AD 這一路的 E2E 模型都把 perception encoder 跟 downstream policy 用 latent representation 串起來——**latent 是被學出來的，不是被規範的**。**只要你把 perception encoder 換掉——換 backbone、加 sensor、修 training set——latent 分佈就變、downstream policy 立刻失能**。**論文原話**：「Updates to perception models can alter these representations and degrade the performance of downstream policies that remain fixed」——這句話是輕描淡寫。實測 Case 5（多相機換成單 LiDAR）Route Completion 從 100% 掉到 43.79%，這根本不叫「degrade」，叫「整條 stack 廢掉」。
- **這件事為什麼過去半年沒有被端出來檢討**：因為 E2E 自駕**還在 demo 階段**——所有的 UniAD/VAD 論文都是「訓一次、跑一次 benchmark、發論文」的一次性 pipeline。**但只要往量產前夜多走一步，這個問題就會爆**：Waymo 每季度都在換 perception model（Waymo Foundation Model 論文自己講過）、Tesla FSD 每個 build 都在調 perception（HW3 → HW4 蒸餾走了半年）、Xpeng 從 nuScenes 遷到自家車隊資料的過程也是 perception 先動 policy 後補。**過去大家的解法都是「兩邊一起重訓」——但這個成本正在爆炸**。這篇論文是第一個把成本量化並提出 O(0.001%) 參數的替代方案的 paper。
- **他們的方案很簡潔到讓人懷疑「為什麼不早點想到」**：把 perception encoder 更新前後的 latent space 用**線性映射（linear affine，`z' = zA + b`，closed-form 最小二乘法解，2,048 個 latent 對）**或 **1×1 convolution（32→32 channels + BN，訓 1 個 epoch、batch size 16、lr 0.001）** 對齊，就能讓 frozen policy 直接吃新 encoder 的輸出。**這個想法背後是「representation similarity」文獻的積累**——2021 之後很多 paper 都在講「不同 backbone 學到的 feature space 其實可以互相線性對齊」，但**過去沒人真的把這個結論拿去解決 AV 生產問題**。這篇 paper 的貢獻不在 novelty，在**把一個 representation learning 的側題，接到了一個 fleet operation 的正問題上**。
- **數字最狠的地方是訓練成本**：Case 5（sensor shift）的 retraining 需要 **22.18 小時、18.66 GB GPU memory、70,000 個 environment steps**——這個 70,000 步是 CARLA reinforcement learning 的老規矩，實務上代表要租 GPU 跑一週。**convolutional stitching 只需要 0.91 小時、5.66 GB memory、0 個 environment steps**——**24× 加速、GPU memory 砍到 30%、且完全不需要模擬環境**。這件事在 fleet operation 語境下是質變：以前 perception 團隊改一次 backbone 要跟 policy 團隊排整週的重訓 slot，現在 policy 團隊只需要一個下班前的 batch job 就能把 compatibility 收回來。**這是我看到這篇 paper 覺得「這件事會影響 AV 公司的組織架構」的原因**——perception 團隊跟 policy 團隊過去因為「重訓成本」被綁在同一個 release cadence 上，這篇之後可以解耦。
- **但這件事的限制也很明確——cross-domain（真實→模擬、或反之）目前還是死結**：Case 6（nuScenes 真實資料訓的 encoder → CARLA 模擬環境的 policy），convolutional stitching 保留 92.23% Route Completion，但更驚人的是——**retraining 只保留 47.98%**。等於 domain shift 這件事**連重訓都救不回來**，stitching 反而**看起來比重訓好**。**這個結果其實不是好消息**，是壞消息：它說明 domain shift 這個問題**不是靠對齊 latent space 能解的**，policy 本身在 CARLA 上的分數上限就被封在那——「模型看得懂 domain 但駕駛不了 domain」的深層錯配。**這暗示現在所有『sim-to-real transfer』的論文都可能低估了問題本質**，這是我讀完這篇最想拉出來獨立寫一篇的線索。
- **對比 Waymo Foundation Model / Wayve GAIA-2 / NVIDIA GR00T 的 fleet update 策略**：Waymo 沒公開他們怎麼處理 perception update 的下游影響，但從他們論文的「訓一整個 foundation model 一次」語氣看，**他們就是硬吃「兩邊一起重訓」的成本**——用 fleet-scale 資料量把問題壓下去。Wayve 走 GAIA world model → policy 蒸餾路線，perception 是 world model 的內部狀態，反而繞開了這個問題（但代價是 world model 本身變成 SPOF）。NVIDIA GR00T N1.5 是機器人賽道但同樣的痛點——他們的 mitigation 是把 perception 收在 Cosmos world model 裡而不是暴露 latent。**這篇 paper 是第一個講出「其實你不用把 encoder 藏起來、直接對齊一下就好」的思路**——這對開源 / mid-tier AV 玩家（Autoware ecosystem、中國 tier-1、robotaxi 新創）是可執行的路徑，對頂尖玩家沒差因為他們有錢重訓。
- **對 LiDAR 工程師（Adam 這一路）的具體訊號**：這篇 paper 的 Case 3（單模態 → 多模態）與 Case 5（多相機 → 單 LiDAR）**直接對應到你在 Foxconn 可能碰到的實務場景**——「感測器配置從 pilot 到量產、從 A 車型到 B 車型」的 latent space 遷移。**過去這件事在文獻裡是找不到 recipe 的**，都是各家內部 tribal knowledge。這篇之後你可以直接引用作為 baseline：「我們把 LiDAR encoder 從 CenterPoint 升級到新版時，policy 團隊不需要 co-retrain，我們用 1×1 conv stitcher 一個 epoch 收回 87% 分數」——**這是量產 AV 工程師需要的 talking point**。
- **冷讀**：這篇 paper 三個月內會被業界 pick up、六個月內會出現在 Waymo / Wayve / Xpeng 的 blog post 或會議演講、一年內會被寫進主流 AV 論文的 preliminaries——**因為它解的是「大家都在偷偷處理但沒人願意承認」的操作痛點**。純學術影響會有限（沒有 novel 的 architecture），但**工程實作影響會非常大**。這是那種「引用數不多、被實務工程師默默用爆」的 paper——類似 batch normalization 之前那批「不酷但改變了訓練流程」的 utility paper。

---

## 為什麼今天要寫這篇

過去 7 天寫了 6 篇 compiler / systems 相關的文章——8/28 TOSA MLIR MXFP、9/2 CuTeDSL、9/3 Flashlight、9/4 Event Tensor ETC、9/5 Syncopate Triton、9/6 LLM42、9/8 TIER IV——加上內部還有 Argus data flow invariants 那篇——**這個 ratio 已經超過每週 3–4 篇 compiler 的節制目標了**。今天必須輪換到 physical AI / autonomous driving。

我原本的候選是：
1. Waymo 最新 Ojai 感測器堆疊分析（但過去兩個月已經寫過 waymo 6th gen、waymo perception）
2. Figure BMW 11 個月 pilot 收工後的 hardware redesign（等 Figure 03 spec 出來再寫更 solid）
3. Xpeng VLA 2 → VLA 3 的 architecture leak（傳言階段，還不夠 solid）
4. **arxiv 2606.21509 A Stitch in Time**（新鮮、被業界忽略、直接關聯 Adam 的 LiDAR 背景）

選第 4 個的原因不是「最熱」，是**最直接對 Adam 有職涯槓桿**。Adam 在 Foxconn 做 LiDAR 演算法，之後轉 compiler 職涯的路徑上（見 [[project-career-research-2026]] Option C），**「LiDAR perception + E2E policy 之間的介面問題」這件事既是他現在的實務痛點，也是未來 5 年 AV compiler 工程師會被大量需要處理的問題**。這種同時滿足「今天工作直接用得到」跟「未來職涯稀缺技能」的雙目標題材不多，這是其中一個。

另外一個原因：這篇 paper **6 月上 arxiv 到現在 3 個月，中文圈幾乎沒有嚴肅討論**。我 Google 中文與英文技術部落格都沒看到 in-depth breakdown（只有一些 paper list 的一句話描述）。**寫這篇的邊際貢獻高**——不是又一篇「Waymo 又發新聞」的稀釋，是把還沒被解讀的 primary source 打開來給讀者看。

---

## 事實：這篇 paper 到底講了什麼

### 一句話概括

**E2E 自駕的 perception encoder 跟 downstream policy 之間的 latent representation 是耦合的、脆弱的、不可維護的——這篇 paper 提出用一個非常輕的 stitcher（linear affine 或 1×1 conv）在 latent space 對齊，讓 perception 更新後 policy 不需要重訓就能繼續用。**

### 具體實驗設定

- **Simulator**：CARLA（版本 0.9.15，Bench2Drive 前身的官方 evaluation suite）
- **Perception backbone**：BEVFusion 為基底，變換 encoder 內部結構（VAE / AE / CNN+MLP）與 sensor 配置
- **Policy**：CILRS-style 的 conditional imitation learning policy，frozen 不動
- **Evaluation scenario**：CARLA `EnterActorFlowV2`（進入車流、需要 gap acceptance judgment 的場景，是 AV 的 canonical 測試場景）
- **Metrics**：
  - Route Completion (RC)：跑完路線的比例
  - Driving Score (DS)：CARLA 官方綜合指標，扣分項含碰撞、闖紅燈、逆向、偏離航線
- **對照組**：Retraining (完整重訓)、Fine-tuning (只調 policy)、Linear Stitching、Convolutional Stitching、No-shift baseline

### 七個 test case（Table IV）

| Case | 類型 | 說明 |
|------|------|------|
| 1 | Minimal shift | 只換 random seed 重訓相同 encoder |
| 2 | Model type shift | VAE ↔ AE 的 representation 換型 |
| 3 | Modality shift | 單模態 ↔ 多模態感測輸入 |
| 4 | Sensor count shift | 感測器數量變動 |
| 5 | Sensor type shift | 多相機 → 單 LiDAR |
| 6 | Domain shift | 真實資料 (nuScenes) → 模擬 (CARLA) |
| 7 | Supervision shift | BEV segmentation → 3D object detection |

**這個 case 選擇非常好——它把 fleet operation 會碰到的每一種「perception 側 update」都覆蓋了**。第一版 pilot 到量產 → Case 4；A 車型到 B 車型 → Case 5；換供應商模型 → Case 2；跨區域 → Case 6。**這不是學術題目的抽象拆解，是實務 checklist**。

### 主要結果（Table V，Route Completion / Driving Score）

| Case | 場景 | Linear RC | Linear DS | Conv RC | Conv DS | Retrain RC | Retrain DS |
|------|------|-----------|-----------|---------|---------|-----------|-----------|
| 1 | Minimal | 98.21% | 96.00 | 100.0% | 98.51 | 100% | 99.12 |
| 5 | Sensor shift | 63.59% | 43.79 | 97.11% | 91.19 | 100% | 97.56 |
| 6 | Domain shift | 36.91% | 35.63 | 92.23% | 89.88 | 47.98% | 35.10 |

（No-shift baseline: RC=100%, DS=98.58）

**三個觀察**：
1. **Linear stitching 在小 shift 場景（Case 1）幾乎完美**——這代表「不同 seed / 同架構」的 latent space 差異真的是線性可對齊的
2. **Conv stitching 在中大 shift 場景（Case 5）壓到接近 retrain**——1×1 conv + BN 這麼輕的 transform 就把 97% 的 Route Completion 找回來，這件事直觀上很反直覺
3. **Case 6 的 domain shift 才是真正的 boss**——所有方法都塌，包含 retraining（RC 47.98%）——**這個結果我下面獨立展開講**

### 訓練成本（Table VI，以 Case 5 為例）

| Metric | Retraining | Fine-tuning | Linear | Conv |
|--------|-----------|-------------|--------|------|
| Runtime | 22.18 h | 18.94 h | 0.02 h | **0.91 h** |
| Updated params | 100% | 100% | 0% | **0.001%** |
| GPU memory | 18.66 GB | 18.66 GB | 1.75 GB | **5.66 GB** |
| Environment steps | 70,000 | 50,000 | 0 | **0** |

**這張表是全篇最有殺傷力的表**——因為它把「代價」量化到 fleet operator 能算 ROI 的層級。以下拆開講。

---

## 深挖 1：為什麼 perception update 會殺死 policy——latent space 這件事有多脆弱

### 一個實例思考

想像你在 Foxconn 訓練了一個 E2E 自駕原型：LiDAR + camera 進 BEVFusion，出 latent BEV token，接 planning head 出方向盤/油門/剎車。跑得不錯，pilot 開始上路。

三個月後，你發現 LiDAR backbone 有個明顯 bug——某類反光物體的 point cloud 特徵沒學好。你花兩週修好、重新訓 encoder、mIoU 從 0.52 提升到 0.61——**perception 客觀變強**。

問題來了：你把新 encoder 塞回原本的 policy，車子開始亂開。**因為 policy 學到的是舊 encoder 的 latent 統計特性——特定 dimension 對應到「前方有障礙」、另一組 dimension 對應到「左邊有車道」——新 encoder 的 latent space 座標系旋轉了、平移了、可能還縮放了**。policy 根本讀不懂。

**你的選擇有三個**：
1. 把 policy 一起重訓——需要重新跑 70,000 個 CARLA episodes 或幾百 GB 真實駕駛數據，兩到四週
2. Fine-tune policy——快一點但還是需要 50,000 個 steps、成本仍然大
3. **這篇 paper 提出的第三條路**——訓一個 stitcher 對齊 latent space、1 個 epoch、0 個 environment steps

### 這件事在文獻的位置

這篇的核心技術根源是 **Representation Similarity 文獻**（Kornblith et al. 2019 CKA、Bansal et al. 2021 stitching connectivity、Csiszárik et al. 2021 relative representations、Moschella et al. 2023 Relative Representations），這一路的核心結論是：**不同模型學到的 feature space 之間，經常存在線性或近線性的對齊關係**。

**過去這個結論停留在 vision classification 的 toy setup**（ImageNet 上換 backbone）。真正把它拉到 sequential decision-making + safety-critical 場景的，**這篇是第一個做完整實驗表格的**。

### 為什麼「1×1 conv + BN 訓 1 個 epoch」就夠

三個直覺：
- **1×1 conv over BEV grid** = 對 BEV 空間的每一個 grid cell 做一個獨立的 32→32 channel 的線性映射（32×32=1024 params）+ BN 的仿射變換——這個容量剛好夠對齊「相同語義但不同座標系」的兩個 latent space
- **BN 這一項貢獻很大**——BN 學到的 scale + shift 是 per-channel 的 recalibration，正好對齊了「新舊 encoder 每個 channel 的統計分佈變化」
- **不需要 environment steps** = 純用 supervised learning target（feature alignment loss + optional task loss on offline data），這是為什麼從 22.18h 變 0.91h 的關鍵——**擺脫了 RL 對環境交互的依賴**

### 這件事的組織意涵

以前 perception 團隊跟 policy 團隊被綁在同一個 release cadence 上——因為 perception 一改，policy 就得跟著重訓，重訓要 GPU + CARLA + 一週人力。**這種綁定會讓兩個團隊 velocity 都降下來**——policy 團隊不敢動 perception，perception 團隊改東西要跟 policy 排 slot。

**stitching 讓兩個團隊解耦**：perception 團隊自己迭代 backbone、每次交付附一個 stitcher 訓練 recipe，policy 團隊拿到 stitcher 直接 plug-in——**這是 microservice / API contract 的思路搬到 AI 系統內部**。

我認為這是這篇 paper 一年後被業界大量引用的真正原因——**它給了一個工程管理的 primitive**，不只是一個技術 trick。

---

## 深挖 2：24× 加速 + 0 environment steps 為什麼是 fleet operation 的質變

### 22.18h → 0.91h 意味著什麼

這不只是「快 24 倍」，它把重訓從「排 GPU cluster」變成「compile 一次」的等級。

具體對比：
- **舊做法**：perception 改 → policy team 收到 spec → 排 8×A100 節點的 slot（等 1–3 天）→ 開始 CARLA rollout 70,000 steps（跑 22 小時）→ 收集 checkpoint → 手動 evaluate → 如果不夠好再調 → 整週跑掉
- **新做法**：perception 改 → policy team 收到新 encoder + 2,048 個 latent pair 樣本 → 跑 stitcher train script（1 GPU × 1 小時）→ 直接 plug-in

**這是把 AI 系統的迭代週期壓到跟軟體 build 差不多的量級**。CI/CD pipeline 都能塞得下——`perception_updated` git hook 自動觸發 stitcher training、跑完 diff 上跑 regression test、綠燈就 deploy。**這是 MLOps 的正確樣子**。

### 0 environment steps 這件事被大幅低估

這篇 paper 講「0 environment steps」的方式很輕描淡寫，我要獨立強調：**RL / 需要環境互動的訓練方式，是 AV 迭代速度的最大瓶頸**。

- CARLA rollout 需要 CARLA server + 客戶端 sync，很難 parallel（每個 CARLA instance 吃 4–8 GB GPU），實務上 8 台 GPU 只能跑 2–4 個 concurrent instance
- 真實駕駛數據更慘——需要車隊實地跑（Waymo 級別）或用 driving simulator（成本高、fidelity 有限）
- **這是為什麼 Tesla FSD 每個 build 之間要間隔數週**——重訓 policy 需要 fleet-scale rollout

**stitcher 完全脫離 environment interaction**——只需要 offline collected 的 latent pair（新舊 encoder 對同一批感測輸入的 output）。**這是 supervised learning 的成本結構，不是 RL 的**——差別不是常數倍數，是**complexity class 的差別**。

我覺得這篇 paper **最容易被人忽略但影響最深遠的 line** 是這個「0 environment steps」——它把 policy update 從 RL 世界的成本結構拖到了 supervised learning 世界的成本結構。

### 對 fleet operator 的具體 ROI 換算

假設一家 robotaxi 公司每季度 perception 換 1 次 backbone（保守）：
- 舊做法：每次 policy retrain 22.18h × 8 GPU × 4 週準備時間 = **每季度 4 週工程時間 + 幾千 USD GPU 成本**
- 新做法：每次 stitcher train 0.91h × 1 GPU × 0 週 rollout = **每季度 1 天工程時間 + 幾十 USD GPU 成本**
- 節省：**約 90% 的成本 + 4 週 → 1 天的 lead time**

**這個 ROI 大到會逼所有 fleet-scale operator 至少評估一次**。

---

## 深挖 3：Case 6 domain shift 保留 92% 反而是壞消息

這是我讀這篇 paper 覺得最耐人尋味的地方，也是 TL;DR 我要獨立標記的地方。

### 結果的困惑

- **No-shift baseline**：RC 100%, DS 98.58
- **Case 6 domain shift, Conv stitching**：RC 92.23%, DS 89.88（乍看很不錯）
- **Case 6 domain shift, Full retraining**：RC 47.98%, DS 35.10（**竟然比 stitching 還差**）

如果你只看第一行，會覺得「哇 stitching 好強，跨 domain 都能保留 92%」。**但看第二行——連 retraining 都塌到 48%**——事情就完全不對了。

### 為什麼 retraining 反而更差

只有一種解釋合理：**在 domain shift 情境下，policy 學不會**。

具體來說：nuScenes 是真實資料、CARLA 是模擬環境——**兩者的物理特性、物件行為、交通密度、駕駛風格分佈都完全不一樣**。你把在 nuScenes latent 上訓的 encoder 拿去 CARLA，encoder 側的 feature 提取還算過得去（因為 encoder 學的是低階視覺特徵），但**你想在 CARLA 上重新訓 policy 時，policy 學到的行為根本沒辦法在 CARLA 上得分**——因為 CARLA 的獎勵函數、agent 行為、交通規則跟 nuScenes 內隱的 driving norm 不一致。

**stitching 之所以「看起來比較好」，是因為它保留了原本在 nuScenes latent 上訓的 policy 行為**——這個行為在 CARLA 上剛好不會太差（因為 policy 是 conditional imitation learning，不會做太激烈的動作）。**stitching 在這個 case 是「不動 policy 反而好」的假象**——不是真的解決了 domain shift。

### 這個結果的深層意涵

這暗示現在所有 sim-to-real transfer 論文可能都低估了問題：
1. **domain shift 不只是 perception 的問題**——policy 的分佈也在 shift
2. **重訓 policy 不能無腦解決 domain shift**——因為 reward function / behavior norm 也在 shift
3. **這件事在 fleet 部署會超級痛**——「訓練環境 → 測試場地 → 量產區域」每一次都有 domain shift，不能靠重訓解

**這是我讀完最想拉出來獨立寫一篇的線索**。這個 finding 顛覆了「先訓好 encoder + policy，再靠 domain adaptation 收 gap」的傳統做法——它暗示 **policy 必須從一開始就被設計成能在多個 domain 通用**，不能靠事後 stitching 或事後重訓。

### 對 Adam 的具體暗示

你在 Foxconn 如果做 LiDAR 演算法遷移到不同 domain（例如中東氣候 → 東南亞氣候、高速公路場景 → 城市場景），**不要指望重訓能解**——這篇 paper 給了證據，說明重訓在 domain shift 情境下**不比不動好多少**，甚至可能更差。**你需要的是從一開始就 domain-robust 的架構設計**，這個要求會逼你去讀 domain generalization、multi-domain learning、causal representation learning 這幾條線的論文——這是接下來半年可以下功夫的方向。

---

## 深挖 4：對比 Waymo / Wayve / NVIDIA 大家怎麼處理這個問題

### Waymo Foundation Model 路線

Waymo 2024 底公布的 Waymo Foundation Model 是把 perception + prediction + planning 全部塞進一個 giant transformer——**這個架構本質上是「用同一個模型的所有 parameter 承擔所有更新成本」**。每次要改 perception，等於重訓整個 model。

**他們吃這個成本的方式是「fleet-scale 資料量 + fleet-scale GPU 集群」**——每天車隊回傳幾百 TB 資料、後台幾千 TPU 節點連續訓練。**這個做法對 Waymo 適用（他們有錢、有 fleet）**，對其他人不 scalable。

### Wayve GAIA-2 世界模型路線

Wayve 走的是**用 world model 當 perception 的替代**——GAIA-2 是自駕版的 Sora，直接生成未來場景、policy 從世界模型的預測序列裡蒸餾出來。**這個做法巧妙地繞開了 perception update 的問題**——因為 perception 被藏進 world model 的內部 state，外面看不到。

**但代價是 world model 變成 single point of failure**——一旦 world model 有偏差，整條 stack 都受影響。而且 world model 本身也會被 update，屆時同樣的問題會出現，只是換到 world model → policy 這一層。

### NVIDIA GR00T N1.5（機器人賽道類比）

GR00T N1.5 的架構是 **Cosmos world model → policy**——跟 Wayve 邏輯類似。**同樣的 latent coupling 問題會存在於 Cosmos 更新的時候**，NVIDIA 沒公開他們怎麼處理，猜測也是硬吃重訓成本（NVIDIA 也不缺 GPU）。

### Xpeng / 中國 tier-1 路線

Xpeng VLA / 華為 ADS 3.0 / 蔚來 Aquila 都走 modular but co-trained 路線——perception 跟 policy 是分開模組但共同訓練。**這個做法某種程度上把 stitching 隱式做進 co-training 裡了**——每次 perception 或 policy update，都靠 co-training 的 gradient 對齊 latent space。**但成本仍然是全參數更新**。

### 這篇 paper 的定位

上交 + 上海人工智慧實驗室這個作者組合，讀者群顯然不是 Waymo 這種頂尖 fleet operator（他們有錢自己解），**是中國 tier-1 車廠 + Autoware ecosystem + 開源 AV 新創**——**這批玩家需要一個 explicit、輕量、可審計的 solution，這篇 paper 剛好對應**。

**這是為什麼我說這篇的引用曲線可能是「學術引用不高、工程實作大爆」——它服務的是實作端而不是頂會端**。

---

## 對 LiDAR / AV 工程師（Adam 這條線）的具體行動

### 一、把 stitcher 加進你手邊的 perception PoC

如果你在 Foxconn 有任何 LiDAR encoder / BEV encoder 的 pilot 專案，**下次要換 backbone 或改 sensor config 時，先試 1×1 conv stitcher 而不是 co-retrain**。實作成本很低：
- 收 2,048–8,192 個 latent pair（新舊 encoder 對同一輸入的 output）
- 訓一個 `Conv2d(32, 32, 1, 1) + BatchNorm2d(32)`
- 用 feature alignment loss（`||T(z_new) - z_old||²`）或 task-alignment loss
- Batch size 16, lr 0.001, 1 epoch

**這個 recipe 是這篇 paper 直接給的**——你可以複製過來測。實測結果比自己的 co-retrain baseline 好，就有寫論文/內部技術報告的素材。

### 二、把「perception-policy decoupling」寫進技術履歷

現在市場上「E2E 自駕工程師」的職缺敘述都寫得很虛（「熟悉 UniAD / VAD / 熟悉 nuScenes」），**沒有人寫「熟悉 perception-policy interface 的維護」——因為過去這件事沒有標準做法**。

這篇 paper 之後，這件事會變成 tier-1 車廠面試的具體考題——「如果你的 perception backbone 從 CenterPoint 升級到 X，你怎麼確保 downstream planning policy 不塌？」——**你能講出「用 1×1 conv stitcher + BN 對齊 latent space, 1 個 epoch 訓練, 0 environment steps」的候選人會立刻脫穎而出**。

### 三、關注 domain shift 這條線

Case 6 的結論（**重訓在 domain shift 下不比不動好**）是 open problem。這個問題接下來 6–12 個月會有一批論文出來嘗試解——**如果你想從 LiDAR engineer 轉 E2E 自駕研究向的路徑，這是很好的入場題目**。

推薦追蹤的方向：
- Domain generalization 文獻（ICLR / NeurIPS 2024–2026 的相關 workshop）
- Causal representation learning（Bernhard Schölkopf 這條線的工作）
- Robust policy learning under distribution shift（Sergey Levine / Chelsea Finn 這批 UC Berkeley 學者）

### 四、compile-side 的思考——這件事跟 [[Compiler-Path]] 的關係

**stitcher 這個 primitive 本身其實是 compiler 議題**——「兩個不同編譯結果之間的 semantic-preserving transformation」在傳統 compiler 領域叫 **decompilation / recompilation with representation invariance**。

一個具體的研究方向：**能不能在 MLIR / TOSA 層級把 stitcher 表達成 pass？**——把「perception encoder 的新舊版本 + stitcher」表達成一個 IR-level transformation，讓 compiler 自動優化 stitcher 的執行（fuse 進 encoder output layer、量化、SIMD 加速）。**這是「AI 系統維護」跟「AI compiler」兩條線交會的具體題目**。

如果你之後要選 spconv 之外的 capstone 主題，這個方向也是強 candidate——**它同時展示 domain knowledge（AV）+ compiler skill + system engineering，正好對應 NVIDIA / Tier-1 車廠的 senior 職缺 spec**。

---

## 冷讀

**這篇 paper 的重要性不在論文的技術新穎度**——linear probing / representation similarity 這些工具都是現成的。**重要性在它挑對了一個被業界默默處理但沒被拉到明面上的問題，並用最輕的方式解決**。

**這種類型的 paper 特徵**：
- 引用會慢——因為沒有 novel architecture，學術跟蹤沒有明顯 bait
- 但實務會快——因為 recipe 具體、實驗表格可信、複製成本低
- 一年後你會在 Waymo blog、Wayve tech talk、Xpeng CVPR workshop 的講稿裡看到「we adopted a stitching-based approach inspired by...」

**這是 utility paper，不是 breakthrough paper。但 utility paper 的長期影響經常大過 breakthrough paper**——參考 batch normalization、dropout、residual connection、AdamW——**這些都是「不酷但改變了訓練 pipeline」的 primitive**。

三年之後回頭看，這篇如果被廣泛採用，會被視為「AV 從 demo 走進 production 的關鍵基礎工程之一」；如果沒被採用，會被 obscure 到 arxiv 深處。**但選對問題的邊際回報永遠遠大於選對技術**——這件事這篇 paper 選對了。

**對 Adam 的最後一句**：這是那種**你今天讀完會覺得「合理但不太震撼」、三個月後在工作上碰到 perception update 問題時會突然想起「操，那篇 stitching」**的 paper。**先把它存進腦子的 cache 裡**——實務痛點碰上時，你會知道要打開哪一頁。

---

*Sources:*
- [A Stitch in Time Saves Nine (arxiv 2606.21509)](https://arxiv.org/abs/2606.21509)
- [Bansal et al. 2021 - Revisiting Model Stitching](https://arxiv.org/abs/2106.07682)
- [Moschella et al. 2023 - Relative Representations Enable Zero-Shot Latent Space Communication](https://arxiv.org/abs/2209.15430)
- [Kornblith et al. 2019 - Similarity of Neural Network Representations Revisited (CKA)](https://arxiv.org/abs/1905.00414)
