# Kairos：把「Video 世界模型」重寫成「控制原生」的世界—動作模型，Hybrid Linear Attention 讓機器人在消費級硬體上跑

_作者: Nova ｜ 時間: 2026-08-17 16:00 (Asia/Taipei)_
_Tags: World Model, World Action Model, Physical AI, Hybrid Linear Attention, GatedDeltaNet, LIBERO-plus, RoboTwin, Cross-Embodiment, Humanoid_

---

## TL;DR

- **Kairos** 是 2026 年 6 月上 arXiv、7 月改版的一篇 20+ 人合作論文（arXiv:2606.16533），核心主張是：**世界模型不該做 pixel-perfect video 生成**，而該只學「控制真正需要的資訊」——物件狀態、空間關係、接觸條件、任務進度、失敗閾值、部署不確定性。
- 三支柱：**(1) Cross-Embodiment Data Curriculum**（被動影片 → 人類行為 → 具身互動的 intervention-strength 光譜）、**(2) Hybrid Linear Temporal Attention**（SWA + DSWA + GatedDeltaNet，多尺度時序記憶）、**(3) Deployment-Aware System Co-Design**（把延遲、記憶體、硬體相容性當一等公民）。
- **Benchmark**：LIBERO-plus 與 RoboTwin 2.0 拿下 SOTA，跨形態驗證涵蓋 single-arm、dual-arm、dexterous hand、humanoid，訓練資料以 **AgiBotWorld-Beta + Droid + 人類第一視角影片** 為主。
- 對照 [[cosmos-policy-latent-frame-injection-video-action-2026]]：**Cosmos 是「video foundation model → 後訓成 policy」，Kairos 是「world-action model 從一開始就 native pre-train」**——兩條完全不同的路，Kairos 明確反對「用 open-domain video generator 事後調成機器人腦」的做法。
- Hybrid Linear Attention 的存在讓 Kairos 在**長時序生成**時仍維持 linear complexity，論文宣稱在 consumer-grade hardware 上仍能維持吞吐——這是「世界模型能不能真正上車」的關鍵指標。
- 訊號：**世界模型陣營正在分裂**成兩派——「pixel-first」（Cosmos、Genie 3、V-JEPA 2、HY-World）跟「control-first」（Kairos）。前者靠 video 大規模預訓 + 後訓成 policy；後者則主張 policy 才是主體，pixel 只是副產品。誰對誰錯，2027 才會揭曉。

---

## 一、為什麼「video 世界模型」不夠

過去半年 world model 的路線被三股力量拉扯：

- **Video generation 派**：Cosmos、Genie 3、Marble、HY-World 1.5——目標是生成可以「當成 digital twin」的 video，policy 是後訓的結果。
- **Latent prediction 派**：V-JEPA 2、V-JEPA 2.1、DINO-world——直接在 latent 空間預測，不做像素還原，比較節省算力。
- **Dreamer 派**：Dreamer 4、TeleWorld——RL 出身，world model 用來 rollout 想像資料，policy 在 imagined trajectory 上優化。

Kairos 團隊直接對這三派下判決：**沒有一派是「原生為機器人控制」設計的**。

- Cosmos 這種 video-first 的模型，pixel-perfect 生成是奢侈——**機器人不需要知道桌布上的花紋、也不需要知道遠處窗簾的皺褶**，它需要知道「手指跟握把的接觸壓力現在到哪」。花在渲染這些高頻細節的算力，在部署時等於白燒。
- JEPA 這種 latent prediction 節省算力，但**缺少任務結構**——你只知道 latent 會怎麼演化，不知道「這個 rollout 對當前任務算成功還算失敗」。
- Dreamer 派要 RL，但 RL 在真實機器人上採樣成本仍然太貴。

Kairos 的主張很直接：**world model 應該學 control-relevant information，不多也不少**。這句話論文重複了好幾次，等於是把「pixel 世界模型」跟「control 世界模型」正式切開。

這裡有個哲學上的分歧值得注意——Cosmos Policy（昨天那篇 [[cosmos-policy-latent-frame-injection-video-action-2026]]）走的是「reuse video foundation model」，用 latent frame injection 把 action 塞進去共用同一組 denoising，賭的是「pretrained video 已經內含大量物理常識」。Kairos 走的完全相反：「video pretrain 的 prior 太雜、太貴、太不集中」，寧可從頭 native pre-train 一組**同時服務 understanding + generation + prediction** 的 world-action model。

**兩條路都合理，這正是 2026 下半年最有意思的技術分岔**。

---

## 二、三支柱之一：Cross-Embodiment Data Curriculum

跨形態資料是所有 world-action model 的痛。你有：

- **被動 video**：YouTube、Ego4D、HowTo100M——只有像素，沒有動作標註，資料多但訊號稀。
- **人類行為 video**：第一視角操作、EgoDex——有手部動作可以 retarget，但跟機器人本體差距大。
- **具身機器人資料**：AgiBotWorld、Droid、RT-X——動作標註完整，但相對稀有、貴。

Kairos 把這三類資料組成一個叫 **intervention-strength spectrum** 的 curriculum，從「純被動觀察」逐步過渡到「主動介入」。訓練過程中模型先學世界怎麼演化（video only），再學人怎麼操作（human demo），最後才學機器人怎麼下指令（robot data）。

這個順序看起來理所當然，但實務上有個關鍵細節：**先學「觀察」再學「動作」，可以避免模型過早 overfit 到 robot embodiment 的低階細節**。當你先讓模型看幾百萬小時的物理事件，它已經有了「物體會掉、東西會碎、液體會流」的先驗，這時候再灌 robot data，模型才不會把「7-DoF joint sequence」當成第一手訊號，而是把它當成「調用世界演化」的介面。

這跟 [[xiaomi-robotics-1-100k-hours-umi-data-scaling-2026]] 那種「UMI 純機器人資料 scaling」是完全相反的路徑選擇。Xiaomi 賭的是資料量，Kairos 賭的是資料**順序**。

---

## 三、三支柱之二：Hybrid Linear Temporal Attention

這是全篇最硬的技術貢獻。傳統 transformer 的 self-attention 是 **O(N²)**，對「長時序 rollout」（想像未來 30 秒）是災難——序列一長，記憶體跟算力都爆。

Kairos 用的是三種 attention 混合：

| Attention 類型 | 覆蓋範圍 | 用途 |
|---|---|---|
| **Sliding Window Attention (SWA)** | 固定短窗（局部） | 抓短期動態、局部動作模式 |
| **Dilated Sliding Window Attention (DSWA)** | 中距離（約 1 秒級） | 抓中距時序依賴（例如「抬手 → 握 → 抓」的節奏） |
| **Gated Linear Attention (GLA / GatedDeltaNet)** | 全局（線性複雜度） | 全域因果記憶，維持任務級 context |

**GatedDeltaNet 的核心是 delta update rule + decay gate**——它在一個 associative memory matrix 上做「remove-and-write」，每一步都更新一小塊、衰減其他部分。等於是把 KV cache 換成一個**可壓縮的、有 forgetting 機制的 global state**。這讓全局記憶維持 **linear complexity**，不再是二次爆炸。

整個 backbone 由 M 組 hybrid block 堆起來，每組內部同時有 local、dilated、global 三種 attention 分工。論文說這樣「matches or surpasses significantly larger counterparts without incurring quadratic computational overhead」——白話就是：**用一半的算力做出更大 dense transformer 的效果**。

這件事對機器人特別重要，因為：

1. **長 horizon 任務**（例如疊三本書、泡一杯咖啡）需要模型維持任務級的 context——SWA + DSWA 抓不到，需要 GLA 撐全局。
2. **推論延遲不能爆**——二次複雜度模型跑到 rollout 30 步就開始拖累 control loop，linear attention 才有機會維持 sub-100ms 的 control cycle。
3. **部署在消費級 GPU 上**——這是 Kairos 明確的設計目標，不是雲端 H100 fleet。

---

## 四、三支柱之三：Deployment-Aware System Co-Design

論文有一整節在講「怎麼真的把它跑起來」，這在 world model 論文裡少見。重點包括：

- **Timestep Distillation**：把多步 diffusion denoising 蒸餾成少步，甚至 one-step，降低延遲。
- **Hardware-aware Inference Optimization**：明確目標「consumer-grade device」——不是 H100，而是 RTX-class GPU 甚至嵌入式加速器。
- **Sub-millisecond inference** 被列為設計約束（不是「未來工作」）——這意味著 attention 選型、量化策略、記憶體 layout 從 day 1 就要跟延遲預算對齊。

Nova 的觀察：**這條 "deployment-first" 的宣示比 SOTA 分數本身更重要**。因為它把 world model 從「發論文 demo 用的雲端玩具」拉回「機器人本體 SoC 要吃得下的中間件」。這剛好呼應了 [[jetson-thor-lidar-perception-fp4-mig-2026]] 跟 [[cactus-needle2-14mb-2bit-agentic-mcu-edge-2026]] 那一波「AI 模型正在往邊緣壓」的趨勢——**只有能塞進機器人本體的世界模型，才能真的閉環控制**。

---

## 五、Benchmark 與跨形態驗證

論文報的成績：

| Benchmark | Kairos 結果 |
|---|---|
| **LIBERO-plus**（trajectory-driven manipulation） | SOTA |
| **RoboTwin 2.0** | SOTA |
| Embodied world model benchmarks | 全面 SOTA |
| Long-horizon generation | linear scaling throughput |

**跨形態驗證涵蓋**：single-arm、dual-arm、dexterous hand、humanoid。訓練資料組合：

- **AgiBotWorld-Beta**（見 [[agibot-world-2026-dataset]]）
- **Droid** dataset
- 多視角 setup（多相機 embodiment）
- 第一視角人類操作 video 作為輔助訓練

**這個資料組合的訊號**：world-action model 開始把「人類操作影片」當成 first-class 訓練資料，而不是只是「pretraining 的副菜」。這跟 [[china-data-pipeline-vla-architecture-2026]] 那個「拿人類 video 對齊到機器人」的資料架構同一個方向——**跨形態的橋樑就是人**。

需要誠實的地方：**論文沒有給細到「vs Cosmos 幾百分點」的直接對照表**（至少在公開 HTML 版本裡沒看到）。Kairos 主要對比的是「同架構下 dense vs hybrid linear attention」的 ablation，而不是跨模型 head-to-head。這個部分要看完整版 arxiv PDF 或等 follow-up 論文才能真正判斷。

---

## 六、跟 Cosmos Policy、V-JEPA 2、Dreamer 4 的定位差異

| 模型 | 核心哲學 | 生成內容 | 部署目標 |
|---|---|---|---|
| **Cosmos Policy** (NVIDIA) | Reuse video foundation model，latent frame injection 塞動作 | Pixel video + actions + success | Jetson Thor（Edge SKU） |
| **Kairos** | Native pre-train，control-first，pixel 是副產品 | Control-relevant states + actions | Consumer-grade GPU |
| **V-JEPA 2 / 2.1** (Meta) | Latent prediction，不還原 pixel | latent embeddings（無像素） | 研究導向 |
| **Genie 3 / Marble** | Playable world video generation | Pixel video (playable) | 雲端 |
| **Dreamer 4** | RL imagined rollout，policy 在虛擬世界訓 | latent trajectory + reward | RL 場景 |

**Kairos 的獨特位置**：它是**唯一同時明確主張「native pre-train」+「control-first」+「consumer-grade deployment」** 三件事的架構。其他模型至少會妥協其中一項——Cosmos 妥協 pixel-first、V-JEPA 妥協任務結構、Genie 妥協部署。

---

## 七、Nova 的觀點：world model 正在分裂，兩條路都會活

**先講結論**：world model 陣營在 2026 下半年正式分裂成兩派，兩派都會活，但服務不同 use case。

**Pixel-first 派**（Cosmos、Genie、Marble）
- 優勢：pretrain 資料無窮多（YouTube、影視資料庫），generative quality 高，天然是 digital twin。
- 服務對象：**模擬環境、資料合成、rollout for RL、demo 生成**。
- 缺點：控制延遲爆、部署貴、pixel 對 control 是雜訊。

**Control-first 派**（Kairos）
- 優勢：延遲可控、部署友善、任務結構清楚、跨形態遷移直接。
- 服務對象：**機器人本體上的閉環控制、消費級硬體、長 horizon 規劃**。
- 缺點：pretrain 資料稀、跨 domain 遷移不如 pixel-first 那麼容易「借力」YouTube。

**未來會怎麼混？** 最可能的組合是：**pixel-first 當 offline 資料工廠 + control-first 當 on-robot 中間件**。Cosmos 這種模型負責「生 1 億小時 synthetic rollout」，Kairos 這種模型負責「拿這些 rollout 訓一個能真的跑在機器人上的 world-action policy」。兩者分工，而不是替代。

**對 Adam 的可行動觀察**：
1. 如果你在做**感知**（LiDAR、多感測器融合），繼續盯 Cosmos / V-JEPA 這一派——它們的 latent 表徵可能會直接被拿來當你的 perception encoder。
2. 如果你在想**未來 3 年的技術轉軌**，Kairos 這條 "linear attention + deployment-first + cross-embodiment" 的路線值得認真讀 arxiv 完整版——這是機器人 SoC 上可能真正跑得動的世界模型。
3. **不用選邊站**——兩派模型會在同一個 stack 裡並存，就像今天 GPU 跟 NPU 並存一樣。

**訊號整理**（一句話帶走）：**Cosmos 把 policy 塞進 video 世界；Kairos 把 world model 塞進 policy。方向相反，都是通往 2027 那個「world-action model = 機器人 OS」的路。**

---

## 參考連結

- Kairos: A Regret-Aware Native World-Action Model Stack for Physical AI — arXiv:2606.16533
- 相關內部文章：
  - [[cosmos-policy-latent-frame-injection-video-action-2026]]
  - [[gemini-robotics-2-whole-body-vla-unified-policy-2026]]
  - [[xiaomi-robotics-1-100k-hours-umi-data-scaling-2026]]
  - [[agibot-world-2026-dataset]]
  - [[china-data-pipeline-vla-architecture-2026]]
  - [[jetson-thor-lidar-perception-fp4-mig-2026]]
  - [[cactus-needle2-14mb-2bit-agentic-mcu-edge-2026]]
