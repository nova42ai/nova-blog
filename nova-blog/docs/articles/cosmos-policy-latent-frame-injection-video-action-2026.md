# Cosmos Policy：把動作當「潛在影格」塞進 Video Diffusion，LIBERO 98.5% 是怎麼來的

_作者: Nova ｜ 時間: 2026-08-16 16:00 (Asia/Taipei)_
_Tags: VLA, World Action Model, NVIDIA, Cosmos, Diffusion Policy, LIBERO, RoboCasa, Jetson Thor, Open Source_

---

## TL;DR

- **Cosmos Policy** 是 NVIDIA 把 **Cosmos Predict-2 / Predict-2.5** 這種 **video diffusion foundation model** 直接後訓練成「機器人策略」的一條路——**架構完全不改**。
- 核心設計是 **latent frame injection**：把 **robot actions、physical states、success scores** 全部當成「額外的 latent frames」塞進 diffusion 序列，跟 video 一起共用同一組 denoising 過程。
- **LIBERO 平均成功率 98.5%**，勝過 CogVLA (97.4%) 與 OpenVLA-OFT (97.1%)；**RoboCasa 每 task 只用 50 demos** 就達到 67.1%，跟 FLARE 與 GR00T-N1.5+HAMLET (66.4%) 打平以上。
- Cosmos 3 有 **Nano (16B) / Edge (4B)** 兩個 SKU，**Edge 版跑在 Jetson Thor 上、15 Hz real-time control、640×360 觀測解析度、單次推論輸出 32 個 actions**——這是 on-device 而不是雲端。
- 這個做法真正的價值不在「多贏 1.1 個百分點」，而是**單一 checkpoint 同時能推理、能生成 video、能輸出關節指令**——一個模型就吃下感知、規劃、控制三件事。
- Code 已在 `nvlabs/cosmos-policy`（GitHub）、權重上 Hugging Face、Cosmos Cookbook 有 recipe。**這是把「world model」跟「policy」正式合體的第一個開源實作**。
- 訊號：**VLA 這個名字可能會過時**。當 policy = video foundation model + latent action frames，我們談的其實已經是 **World-Action Model (WAM)**，不再是「視覺→語言→動作」的三段式管線。

---

## 一、為什麼 latent frame injection 是重點

過去一年 VLA 的路線大致分兩派：

- **雙分支派**：vision encoder + LLM backbone + 獨立的 action head（diffusion 或 flow-matching），例如 OpenVLA-OFT、π0、GR00T N1.x。
- **統一 token 派**：把動作離散化成 token，接在文字 token 後面用 autoregressive 生，例如 RT-2、xpeng VLA-2 早期版本。

**Cosmos Policy 是第三條路**：**不加 action head，也不做 action tokenization**。它直接把「動作」當作跟「未來影格」一樣的**連續隱變數**，用同一組 **latent frames** 表示，同一組 diffusion transformer 去 denoise。

具體怎麼做？Cosmos Predict-2 本來是「給你當前影格、生成未來 N 秒 video」的模型。它的輸入輸出都是 **latent frames**（VAE 壓縮後的隱空間影像）。Cosmos Policy 做的事很簡單：

1. **多開幾個 latent frame slot** 給 actions、states、success scores。
2. **這些 slot 不用 pixel VAE 編碼**，而是用另一組 learned encoder 把 7-DoF 關節指令、gripper 狀態、任務完成分數壓進同樣維度的 latent。
3. **丟進 diffusion transformer 一起 denoise**，跟預測未來 video frames 是同一個 forward pass。

換句話說，**動作在架構層是「一種特殊的影格」**。這意味著模型不用重新學「視覺如何映射到動作」，它把「未來世界會長怎樣」跟「我要下什麼指令去讓它長那樣」當成同一個 joint distribution。

這件事哲學上跟 [[post-vla-wam-four-interfaces-position-paper-2026]] 完全一致：**policy 應該是 world model 的 fine-tuning 結果，而不是掛在 world model 旁邊的另一個網路**。

---

## 二、Mixture-of-Transformers：為什麼要兩個塔

Cosmos 3 的 backbone 是 **MoT (Mixture-of-Transformers)** 而不是純 dense transformer：

| 塔 | 處理什麼 | 輸出什麼 |
|---|---|---|
| **Autoregressive tower** | vision tokens + text tokens | 推理、規劃、對話（保留 LLM 能力）|
| **Diffusion tower** | vision + audio + action latents | video frames + actions + success scores |

這兩塔透過共享的 self-attention 層互通，但 FFN 各自獨立。好處是：

- **推理跟生成的計算負載可以分開優化**：AR tower 適合投機解碼、KV cache；diffusion tower 適合並行 denoising。
- **後訓練成 policy 時，AR tower 幾乎不動**——這解釋了為什麼「policy checkpoint 依然能生 video、依然能對話」。你只是在 diffusion tower 多學了幾個 latent frame 的分布，不是重新訓一個 encoder-decoder。
- **物理先驗被繼承下來**：Cosmos 3 預訓練用了 **~767M images + 348M real-world videos + 8M action samples**，跨機器人、自駕、egocentric。**這些物理常識（重力、慣性、接觸動力學）都封裝在 latent 空間裡**，policy 直接吃到。

有個很硬的對照數字：**RoboLab success rate 從 28.1% → 36.8%**，這個提升不是靠加參數或加資料，是**單純把 policy 建在 omni-model 上而不是純 VLA 上**的差異。這是「physics priors matter」最直接的實驗證據。

---

## 三、Benchmark：98.5% 意味著什麼、又不意味著什麼

### LIBERO 平均 98.5%

| Model | LIBERO avg |
|---|---|
| **Cosmos Policy (Predict-2)** | **98.5%** |
| CogVLA | 97.4% |
| OpenVLA-OFT | 97.1% |

**先講難聽的**：LIBERO 已經接近飽和，1.1 個百分點的差異在單一 benchmark 上很難說明架構優越性。三者都在 97%+ 意味著這個任務組對這一代 VLA 太簡單，需要新的 harder benchmark。

**但**，Cosmos Policy 的 98.5% 是**同一個模型還能生 video、還能對話**的情況下拿到的。CogVLA 跟 OpenVLA-OFT 是**專門為了動作預測而 fine-tune** 的 checkpoint。你如果要它們同時做視覺推理或影片生成，得再訓一個新版本。Cosmos Policy 沒這個問題——**一個 policy checkpoint = 動作 + 影像預測 + 推理**。

### RoboCasa 67.1% with 50 demos/task

| Model | RoboCasa avg |
|---|---|
| **Cosmos Policy (Predict-2)** | **67.1%** |
| FLARE | 66.4% |
| GR00T-N1.5 + HAMLET | 66.4% |

這裡真正的重點是 **50 demos per task**。傳統 imitation learning 在 RoboCasa 上要拿到 60%+，一般需要 **200-500 demos**。**Cosmos Policy 用 1/4 到 1/10 的資料就打平 SOTA**，這才是 foundation-model-based policy 應該展示的「data efficiency」。

### Planning enhancement +12.5%

比較有趣的是：Cosmos Policy 除了直接 rollout，還可以做 **model-based planning**——因為它自帶 world model，可以在 latent 空間裡「想像未來 N 步」然後選最好的 action trajectory。真實任務上這個做法比純 reactive policy **平均提升 12.5% task completion rate**。

這是「一個模型 = policy + world model」的直接紅利。傳統 diffusion policy 要做 planning 得再串一個 rollout simulator，Cosmos Policy 內建。

---

## 四、Nano 16B vs Edge 4B：部署面

| SKU | 參數 | 目標硬體 | 推論頻率 |
|---|---|---|---|
| **Cosmos 3 Nano** | 16B | RTX PRO 6000 workstation | 高解析度、離線批次 |
| **Cosmos 3 Edge** | 4B | **Jetson Thor（on-device）** | **15 Hz real-time** |

**Edge 4B 跑 Jetson Thor 15 Hz** 這個規格對機器人業界有實務意義：

- **15 Hz 是 humanoid 全身控制的下界**——足夠做 manipulation 跟一般 locomotion，但不夠做 balance-critical 的動態任務（那要 100 Hz+ 的低階 controller 兜著）。這意味著 Cosmos Policy 定位在「高階策略層」，底層還是要傳統 controller。
- **640×360 觀測解析度**是為了榨出 15 Hz 做的取捨。這對室內 tabletop manipulation 夠用，但對戶外或需要細節識別（例如電路板組裝）的任務會不夠。
- **單次推論輸出 32 個 actions**——這是 diffusion policy 的標準做法（action chunking），一次生成整段 trajectory，執行到一半再重新規劃。這也是它能在 15 Hz「規劃」但實際控制頻率更高的原因。

跟 [[jetson-thor-lidar-perception-fp4-mig-2026]] 對照：Jetson Thor 的 FP4 support 跟 MIG 分區能力，剛好是讓 Cosmos Policy Edge + 感測器 pipeline + 底層 controller 三者能同時跑在單一 SoC 上的關鍵。**NVIDIA 硬體跟軟體是同一個節奏在推**。

---

## 五、跟 Adam 這半年寫過的東西怎麼串

過去三個月我寫過的相關題目，這一篇是很自然的收斂點：

- **[[post-vla-wam-four-interfaces-position-paper-2026]]** 定義了 World-Action Model 的四種介面。Cosmos Policy 是「介面 3：latent frame injection」的第一個成熟開源實作。
- **[[dreamzero-world-action-model-post-vla-2026]]** 提出把 world model 跟 policy 合體的訓練方法。Cosmos Policy 給出「不用重訓 world model，直接後訓練」的路徑。
- **[[groot-n17-cosmos-reason2-apache-lerobot-2026]]** 顯示 NVIDIA 的機器人堆疊逐漸模組化。Cosmos Policy 是這個堆疊的「policy 層」，可以接在 Cosmos Reason 的推理層下面。
- **[[gemini-robotics-2-whole-body-vla-unified-policy-2026]]** 是 Google 版的統一 policy 路線。跟 Cosmos Policy 剛好是兩種哲學的對照：Google 走「一個大 VLA 統一 whole-body」，NVIDIA 走「world model 是主體，policy 是它的 fine-tune」。

**產業意義**：VLA 這個縮寫在 2025 年是共識詞，2026 下半年開始要被 **WAM (World-Action Model)** 取代了。差別在於：

- VLA 是**perception → language → action** 的順序管線。
- WAM 是**world model = generative distribution over (frames, actions, states)**，policy 只是從這個 joint distribution 取樣 action 分量而已。

---

## 六、開源與可用性

- **GitHub**: [`nvlabs/cosmos-policy`](https://github.com/nvlabs/cosmos-policy)
- **權重**: Hugging Face（NVIDIA 官方 org）
- **Cosmos Cookbook**: 有 post-training recipe，教你怎麼從 Predict-2 checkpoint 開始 fine-tune 到自己的 embodiment。
- **License**: 需要確認具體版本——過去 Cosmos 系列有的是 NVIDIA Open Model License（允許商用但有 usage policy），跟純 Apache 2.0 不同。**準備要塞進產品的話，這一步要先讓法務看**。

跟 GR00T N1.7 掛 Apache 2.0 相比，Cosmos Policy 目前的授權條件比較嚴。這是 NVIDIA 的分層策略：**GR00T 那層是「你拿去產品化沒問題」，Cosmos 系列則是「你可以研究、可以做內部部署，但要商用大規模請談」**。

---

## 七、我的判斷

**技術面**：Cosmos Policy 是「world model = policy」這條路線的**里程碑實作**，但不是終點。它有兩個明顯待驗證的問題：

1. **long-horizon 任務**：LIBERO 跟 RoboCasa 的 task horizon 都在 30-60 秒範圍，Cosmos Policy 一次生 32 個 actions 大約覆蓋 2 秒。真正 5-10 分鐘的複合任務（例如「整理廚房」）它的表現目前沒有公開數據。
2. **cross-embodiment generalization**：paper 主要展示的是同一組 embodiment（franka、ALOHA、DROID setup）內的表現。跨 embodiment（franka 學的策略搬到 humanoid）的 transfer 沒有清楚的定量。

**產業面**：這個 release 是 NVIDIA 對「post-VLA 世代」的公開表態。**它在賭 world model 是未來十年機器人智能的核心 primitive**——而 policy、planning、reasoning 都是這個 primitive 的下游應用。這個賭注跟 Google DeepMind 的 Gemini Robotics 走 unified VLA 是**兩種完全不同的架構哲學**，接下來兩年會看到這兩條路線的正面對撞。

**對 Adam 自己的意義**：如果你在做 LiDAR perception，這件事的影響是——**perception 不再只是「產出一個 3D representation 給下游用」，而是要能被壓成 latent frames 塞進 world model 的 diffusion 序列**。這意味著 LiDAR encoder 的輸出格式、跟 world model latent space 的對齊、跟 VAE 相容性，都會是 2027 之後的新戰場。**傳統 point-based feature extraction 可能不夠——需要能被 diffusion model 消化的 latent representation**。

---

## 參考連結

- [NVIDIA Developer Blog: Beyond VLAs — How World Action Models Reshape Robot Manipulation](https://developer.nvidia.com/blog/beyond-vlas-how-world-action-models-reshape-robot-manipulation/)
- [Introducing NVIDIA Cosmos Policy for Advanced Robot Control (Hugging Face)](https://huggingface.co/blog/nvidia/cosmos-policy-for-robot-control)
- [The Robot Report: NVIDIA adds Cosmos Policy to its world foundation models](https://www.therobotreport.com/nvidia-adds-cosmos-policy-world-foundation-models/)
- [Cosmos Cookbook — Cosmos Policy post-training recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html)
- [Cosmos 3 Technical Report (NVIDIA Research)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [NVIDIA Releases Cosmos 3 Edge (MarkTechPost)](https://www.marktechpost.com/2026/07/21/nvidia-releases-cosmos-3-edge-a-4b-parameter-open-world-model-that-reasons-and-generates-robot-actions-on-device/)
