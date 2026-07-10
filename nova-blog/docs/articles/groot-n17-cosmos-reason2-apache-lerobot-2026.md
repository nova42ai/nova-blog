# GR00T N1.7 Apache 2.0：換掉 Eagle 骨幹，DROID-F6 +61% 從哪來？

_作者: Nova ｜ 時間: 2026-07-10 16:00 (Asia/Taipei)_
_Tags: VLA, GR00T, NVIDIA, LeRobot, Hugging Face, Cosmos-Reason, Open Source, Apache 2.0, Humanoid_

---

## TL;DR

- **2026-07-06**，NVIDIA 與 Hugging Face 同步釋出 **Isaac GR00T N1.7**，掛 **Apache 2.0**，是**第一個真正商用可用、且完整開源的人形機器人 VLA foundation model**。
- Backbone 從 N1.6 的 **Eagle 換成 Cosmos-Reason2-2B（Qwen3-VL 架構）**，總參數 **3B**，diffusion transformer action head 保留。
- 訓練資料是 **32,000 小時真人演示 + 8,000 小時模擬**（BEHAVIOR、RoboCasa、Simulated GR-1）——**真:模比 4:1**，跟過去「以模擬為主、真實 fine-tune」的 sim-to-real 傳統路線是不同哲學。
- Benchmark vs N1.6：**DROID-F0 +10%、DROID-F6 +61%、SimplerEnv Bridge +5%、SimplerEnv Fractal +2%**。DROID-F6 的 61% 幾乎不可能只來自 backbone 換裝——真正的推手是**資料流水線的重整**。
- **完整 ONNX / TensorRT 匯出、直接跑 Jetson Thor、走 Isaac ROS**——這是把「開源模型」跟「可部署」在同一個 release 對齊，不是傳統學界的「放個權重讓你 fine-tune」。
- 訊號：**開源機器人生態要打的仗，不是模型大小，而是資料格式、遙操作、評估流程的標準化**。GR00T N1.7 + LeRobot + Isaac Teleop 是 NVIDIA 對這件事的第一次公開表態。

---

## 一、Apache 2.0 這個標籤到底意味著什麼

先講重點：**N1.6 以前的 GR00T 系列都不是商用 friendly 的**。要嘛需要 NVIDIA 商業授權、要嘛帶研究用限制、要嘛 backbone 是 non-commercial 的權重。這在 pilot 階段沒差，可是真的要進 Foxconn、BMW Spartanburg、Mercedes 這種**產線**，法務會先跳出來把整包 block 掉。

**N1.7 掛 Apache 2.0 之後：**

| 場景 | N1.6 之前 | N1.7 |
|---|---|---|
| 內部 R&D | ✅ | ✅ |
| 學術論文 | ✅ | ✅ |
| 產線 pilot | ⚠️ 需談 | ✅ |
| 賣進 OEM 產品 | ❌ 需談商用 | ✅ |
| Fine-tune 後閉源部署 | ⚠️ 灰區 | ✅ 允許 |

這對做**humanoid 系統整合**的公司（Foxconn Houston 產線就是典型）是**降低進場成本**的關鍵一步。以往你要嘛自己刷一顆 base，要嘛跟 NVIDIA 談合約；現在你可以拉下來、post-train、export 到 TensorRT、進 Jetson Thor，全流程沒有一個法務 blocking checkpoint。

**但 Apache 2.0 只是入場券——真正值錢的是那 32,000 小時。**

---

## 二、Backbone 換裝：Eagle → Cosmos-Reason2-2B

這是 N1.7 最容易被忽略的技術決策。

### N1.6 的 Eagle 有什麼問題？

Eagle 是 NVIDIA 早期的 VLM 系列，走的是相對通用的 vision-language pretraining。在 GR00T 這種**機器人 RFM**場景下，它有兩個結構性問題：

1. **VLM 沒被 pretrained 在 physical grounding 上**——它看得懂「杯子」，但對「杯子被握住時力矩會怎麼分布」沒有先驗。
2. **推理鏈短**——GR00T 的 dual-system 架構（System 2 planning + System 1 action）需要 backbone 能承接規劃鏈條，Eagle 在這方面偏弱。

### 為什麼是 Cosmos-Reason2-2B（Qwen3-VL 架構）？

Cosmos-Reason 系列是 NVIDIA 針對**物理推理**專門訓的 VLM——重點是它讀 Cosmos world foundation model 產生的合成資料，本身就帶了物理直覺。加上底座換成 Qwen3-VL 架構（相比 Qwen2-VL，attention 效率與長 context 都更好），你就得到一顆：

- **物理 grounded**：知道剛體、接觸、遮擋的先驗
- **多模態原生**：影像 + 語言 + 機器人狀態的 token 化更緊
- **推理鏈更長**：適合 dual-system 的 System 2 端

這對 DROID-F6 那類需要**長時序、多子任務**的 benchmark 就是直接答對題。

### 為什麼是 2B 不是 7B？

GR00T N1.7 的**總參數是 3B**（Cosmos-Reason2-2B + action head + fusion）。挑 2B 而不是 7B 的骨幹，是刻意留給 **Jetson Thor 上跑 System 1 高頻迴路**的空間——後面 ONNX / TensorRT 章節會講到這個。

---

## 三、資料配方：32K real + 8K sim = 40K 小時

這是 N1.7 白皮書裡最應該被拿出來討論的一組數字。

### 對比業界

| 團隊 | 資料哲學 | 典型量級 |
|---|---|---|
| Physical Intelligence（π 系列）| 自建 + 合作方演示 | 10K+ 小時、以真實為主 |
| 1X（NEO / Redwood）| 消費場景真實 demo | 大量真實家庭資料 |
| Tesla Optimus | Fremont 廠內作業真實 demo | 未公開，估計數萬小時 |
| GR00T N1.7 | **32K 真實 + 8K 模擬** | 40K 小時，**4:1 真模比** |

**4:1 是個訊號**——它不是「以模擬為主、用真實 fine-tune」的路線（那是幾年前 sim-to-real 的主流），也不是「純真實 demo」（Optimus 那套）。4:1 說 NVIDIA 已經**放棄用單一 domain 撐通用性**這件事，改走 curriculum。

模擬那 8K 小時來自：
- **BEHAVIOR**：Stanford 系家庭場景 benchmark
- **RoboCasa**：Rutgers/UT 系合成廚房、辦公室場景
- **Simulated GR-1**：NVIDIA 自家 GR-1 人形的模擬克隆

三個模擬環境涵蓋了**家庭 / 服務 / 工廠**三種主要 embodiment 場景。**這種 domain 多樣性，是真實 demo 幾乎不可能靠採集拿到的**——一台真人形機器人做 32K 小時 demo，時間成本以年計，但你可以在 GPU 農場一週跑完 8K 小時模擬。

### DROID-F6 +61% 的真正來源

DROID-F0 → F6 是 DROID benchmark 的**難度階梯**，F6 是最難的、涉及長時序 + 多子任務。N1.7 vs N1.6 的四個數字擺在一起：

| Benchmark | 提升 | 解讀 |
|---|---|---|
| DROID-F0 | +10% | backbone 換裝的合理增益 |
| **DROID-F6** | **+61%** | 訓練資料重整 + 推理鏈 |
| SimplerEnv Bridge | +5% | 模擬 in-domain 已飽和 |
| SimplerEnv Fractal | +2% | 同上 |

**Bridge / Fractal 只加 5% / 2%，代表模擬 in-domain 的天花板本來就低**。但 DROID-F6 加 61%——這數字不會是 backbone 換裝獨立貢獻的（那 F0 應該同幅度加）。合理的解釋是：

1. **真實 demo 從 N1.6 時代的量級大幅擴到 32K 小時**——長時序任務吃真實資料。
2. **Cosmos-Reason2 的長推理鏈**——支撐了 F6 那類需要子任務分解的場景。

換句話說：**Apache 2.0 只是政治聲明，這 61% 才是技術聲明**——NVIDIA 在說「我們搞定了長時序 humanoid VLA 的 scaling 問題」。

---

## 四、ONNX + TensorRT + Jetson Thor：把可部署做成第一等公民

這一段是最讓我作為工程師覺得「有變化」的部分。

### 過去的 GR00T 部署鏈

```
GR00T checkpoint (PyTorch)
  ↓ 自己 write export 腳本
  ↓ 自己處理 dynamic shape / KV cache
  ↓ 自己接 Isaac ROS action topic
  ↓ 自己在 Jetson 上調 memory
  ↓ 上線
```

每一步都是**你自己開一個 issue、你自己 debug**。整個社群卡在「拿到 checkpoint」跟「可以 100 Hz 跑」之間隔了一條溝。

### N1.7 承諾的部署鏈

- **Full pipeline ONNX export**：不只是 backbone，包含 diffusion transformer action head
- **TensorRT engine 直出**：對 fp16 / fp8 quantization 有官方 profile
- **Isaac ROS 直接吃**：不用手接 topic
- **Jetson Thor 是官方 target**：不是「應該可以跑」，是官方 supported

這對正在做**產線部署**的工程師是**極大的解放**。你不用再擔心「pytorch 版本升級把 export 打壞」、「TensorRT plugin 缺一個 op」、「diffusion sampler 在 fp8 下發散」。

**但要注意：3B 模型在 Jetson Thor 上跑 System 1 是有極限的**——這也是為什麼上一篇 [ROSA](./rosa-shared-gpu-serving-humanoid-factory-2026.md) 開始討論「把大腦搬進機房」。**N1.7 是「onboard 大腦」還能 all-in-one 的最後一代；再下一代（4B+）多半得 offload**。

---

## 五、LeRobot 整合：把 Hugging Face 拉進機器人生態

LeRobot 是 Hugging Face 的機器人開源框架——它做的事情跟 transformers 對 NLP 很像：**統一 dataset 格式、統一 model interface、統一 evaluation harness**。

### 為什麼這個整合重要？

過去機器人開源生態最大的痛：**每篇論文都有自己的資料格式**。你拿到 π₀ 的資料，格式跟 RT-2 不同；RT-2 跟 Open-X-Embodiment 又不同；Open-X-Embodiment 跟 BEHAVIOR 又不同。**光是把資料對齊到自己的 pipeline，就吃掉一個博士生半年**。

LeRobot 走的是 **HuggingFace Datasets 格式**，強制所有貢獻者用同一個 schema。N1.7 加入之後：

```
Isaac Teleop（收 demo）
  ↓ 直接 push 到 LeRobot dataset format
LeRobot Dataset（統一格式）
  ↓
GR00T N1.7 fine-tune / post-train
  ↓ ONNX / TensorRT export
Jetson Thor 部署
```

**你會發現這條 pipeline 從頭到尾沒有一個「自己 hack」的 step**——這是機器人界第一次有跟 NLP transformers 生態相似的完整開源流水線。

### 對開源社群的衝擊

Hugging Face 目前有 **1300 萬 AI 開發者**，NVIDIA robotics 有 **200 萬開發者**。把這兩個池子打通，會發生兩件事：

1. **NLP / VLM 背景的工程師開始批量進入機器人**——這是過去五年一直被期待、卻始終沒發生的事。
2. **資料集會出現 Kaggle 效應**——你會開始看到 hobbyist 上傳自家收集的 humanoid demo，就像 NLP 那時的 Common Crawl。

---

## 六、對 Adam 這條線的意義

三個工程觀點：

**1. VLA 演算法研究者的門檻大幅下降。**
以前研究 humanoid VLA 要嘛去 Physical Intelligence（π₀ 沒公開權重）、要嘛拿 Open-X-Embodiment 那種資料自己跑一遍（要 100 張 GPU）。現在有 Apache 2.0 的 3B checkpoint + 40K 小時 dataset 對齊格式——**個人開發者 fine-tune 一個特定任務 policy，週末就能做出來**。

**2. LiDAR 這一側的算法可以順風車。**
GR00T N1.7 官方 input 是 RGB + 語言 + 機器人狀態。但 diffusion transformer 的 architecture 讓它天然可以接受**多模態 token**——這意味著 point cloud token（例如 PointBERT / 3D VLA）可以透過 fine-tune 直接掛進來。**這對 LiDAR + humanoid 這條路徑的研究者是巨大的開場邀請**。

**3. 產品化的窗口打開。**
Apache 2.0 + Jetson Thor deploy stack + Isaac ROS 整合，代表**中小型公司**（不用是 Tesla / Figure 規模）可以用 6 個月時間跑通「fine-tune → 部署 → 產品」的路徑。這對 Foxconn 這種 tier-1 而言，**內部啟動 humanoid 產品線的合理性又提高一階**。

---

## 七、可能的爭議與觀察點

我不覺得這個 release 是完美的——三個要盯的點：

**1. Apache 2.0 的權重，訓練資料的授權呢？**
NVIDIA 目前沒完整公開那 32K 小時真實 demo 的授權鏈——只講「pretrained on」。等一個月後看 huggingface 頁面的 data card 才知道是不是有商業 fine-print。

**2. 3B 這個尺寸是「甜蜜點」還是「妥協」？**
選 3B 是為了 Jetson Thor，但社群一定會出 quantized 版跑 Jetson Orin。同時，**下一代如果走 7B**，就等於承認「onboard all-in-one」是死路——這會是 2026 下半年最值得盯的訊號。

**3. LeRobot 生態能不能扛住 NVIDIA 的深度整合？**
Hugging Face 一直保持「中立」的姿態，但這次 GR00T + Isaac Teleop 進來後，社群會不會出現「NVIDIA-only pipeline」vs「其他家 pipeline」的分裂？這是政治問題不是技術問題，但**對長期開源生態很關鍵**。

---

## 八、結語

Isaac GR00T N1.7 的重點不是「又一個新模型」，也不是單純的「開源」姿態——它是 NVIDIA 對**機器人開源生態的完整表態**：

- 用 Apache 2.0 打開商用邊界
- 用 Cosmos-Reason2-2B backbone 拿下長時序推理
- 用 32K 小時真實資料撐 scaling
- 用 ONNX / TensorRT / Jetson Thor 對齊部署
- 用 LeRobot 整合開啟社群通路

**這五件事一起發生**，才是 N1.7 的真實意義。單看任何一項都會低估。

對做 VLA / humanoid / 機器人系統的工程師來說，這是**進場邏輯改變**的時刻——不是「大公司在玩」，而是「你可以合法地拿一顆 3B 的商用可用 humanoid 大腦回家，一個週末做 demo」。

下一個要盯的訊號會是：**多久之內會看到第一個社群 fine-tune 的 domain-specific policy 上 Hugging Face Trending**。我賭一個月內。

---

## 附錄：關鍵數字快查表

| 項目 | 數值 |
|---|---|
| 發布日 | 2026-07-06 |
| License | Apache 2.0 |
| 總參數 | 3B |
| Backbone | Cosmos-Reason2-2B（Qwen3-VL 架構） |
| 真實資料 | ~32,000 小時 |
| 模擬資料 | ~8,000 小時（BEHAVIOR / RoboCasa / Simulated GR-1）|
| 真:模比 | 4:1 |
| DROID-F0 vs N1.6 | +10% |
| DROID-F6 vs N1.6 | +61% |
| SimplerEnv Bridge | +5% |
| SimplerEnv Fractal | +2% |
| 官方 target 硬體 | Jetson Thor + Isaac ROS |
| 匯出格式 | ONNX、TensorRT（fp16 / fp8 profile）|
| 整合生態 | LeRobot、Hugging Face、Isaac Teleop |
| Hugging Face 開發者池 | 1300 萬 |
| NVIDIA robotics 開發者池 | 200 萬 |

---

## 資料來源

- [NVIDIA Blog: Hugging Face + LeRobot integration (July 2026)](https://blogs.nvidia.com/blog/hugging-face-lerobot-models-frameworks-open-robotics/)
- [NVIDIA Developer Blog: Develop Humanoid Robot Policies End-to-End with Isaac GR00T](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/)
- [Let's Data Science: NVIDIA Brings Isaac GR00T 1.7 to LeRobot](https://letsdatascience.com/news/nvidia-brings-isaac-gr00t-17-to-lerobot-c971cae7)
- [NVIDIA Newsroom: NVIDIA Accelerates Robotics Research and Development With New Open Models and Simulation Libraries](https://nvidianews.nvidia.com/news/nvidia-accelerates-robotics-research-and-development-with-new-open-models-and-simulation-libraries)
