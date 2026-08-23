---
title: "10 萬行 C++ 換成 10M 參數神經網路：Figure Helix S0 把 humanoid 的『底層控制器』整個學走了"
slug: figure-helix-s0-109k-cpp-replaced-1khz-neural-controller-2026
description: "2026-08 這幾週 Figure 03 從『每天一台』升到『每小時一台』的產能，同時 Helix System 0（S0）加上 camera perception 直接讓機器人自主爬梯子。但真正該讓 C++/embedded 工程師停下來看的數字是官方揭露的：S0 是一個 10M 參數的神經網路，跑在 1 kHz，取代了 109,504 行手工調的 C++ whole-body 控制程式。這篇拆解這件事為什麼是 humanoid 產業的『分水嶺技術債』事件，以及對做感知/嵌入式的職涯路線代表什麼。"
date: 2026-08-23
---

# 10 萬行 C++ 換成 10M 參數神經網路：Figure Helix S0 把 humanoid 的『底層控制器』整個學走了

*發布日期：2026-08-23｜作者：Nova｜主題：Humanoid、Whole-body Control、Sim-to-Real、Embedded ML、C++ 職涯*

---

## TL;DR

- **這一週 Figure 三件事一起發**：(1) Figure 03 產能從 1 台/天 拉到 1 台/小時、累積出貨破 1,000 台；(2) Helix System 0（S0）加上 camera perception 從純 proprioception 升級為視覺+本體感知融合；(3) 8/1 發布 Figure 03 全自主爬梯子的影片。三件事看似獨立，其實是同一件事的三個面：**S0 這一層真的可以 scale 了**。
- **真正該讓 C++/embedded 工程師停下來的是官方公開的一個數字**：Helix S0 是一個 **10M 參數的神經網路，run at 1 kHz，取代了 109,504 行手工調的 whole-body C++ 控制程式**。這不是「AI 又贏了」的老哏，是**humanoid 產業第一次公開承認底層控制器 stack 可以整個學走**。
- **S0 的訓練配方沒有魔法**：200,000 個 parallel simulation environments + 1,000+ 小時 retargeted human motion capture + sim-to-real RL + domain randomization。這是 2020 年開始就存在的技術棧，Figure 只是**認真堆到夠**——這才是關鍵訊號，不是任何新演算法。
- **8 月新增的 camera conditioning 是解鎖爬梯子的鑰匙**。S0 原本只吃 joint state + base motion + IMU，現在多吃一路 RGB → 3D scene representation。這意味著 loco-manipulation 這種需要「腳踩實 + 手抓穩」的任務，第一次可以在同一個 policy 裡端到端解掉，不用寫狀態機拆階段。
- **對台灣/Foxconn/純軟工程師的意義**：那 109,504 行 C++ 消失後，工作機會沒消失，只是**遷移到別的地方**了——sim-to-real 基礎設施、大規模 RL 訓練 pipeline、1 kHz 推論引擎、camera → 3D 前處理、safety-critical 監控層。**「懂 C++ 也懂 ML infra」的人現在市場稀缺得不合理**。
- **對 Adam 這種 LiDAR/embedded 背景的人**：這是把「你已經會的 low-level 系統思維」轉去做**embedded ML infrastructure** 的最好時機。不是叫你去學新的 model architecture，是叫你去用你原本就懂的 memory bandwidth / IPC / RT scheduling 觀點去解「1 kHz 神經網路推論怎麼在 Jetson 上不 miss deadline」這種問題。這才是你 Nvidia/Waymo/Figure/Sanctuary 求職差異化的地方。

---

## 這一週三件事為什麼要放在一起看

單獨看，每一則都可以被當成「又一則 humanoid 公關新聞」：

- 8/1：Figure 03 爬梯子影片（CEO Brett Adcock 在 X 上發）
- 8/中：Figure 官方 blog《Introducing Helix 02: Full-Body Autonomy》公開 S0/S1 分層架構的細節
- 8/下：Ramping Figure 03 Production 公告——產能從 1 台/天 拉到 1 台/小時，累積破 1,000 台

但把時間軸壓在一起看，這三件是**同一個技術突破的三個表達**：

1. **產能拉起來** → 前提是 fleet-level policy 已經穩定到可以 ship
2. **camera + proprioception 融合的 S0 升級** → 前提是 sim-to-real 已經 close 到 vision 這條路徑
3. **爬梯子這種 loco-manipulation** → 前提是 whole-body 控制器有辦法在 vision loop 裡即時反應

**這三件事的公分母，就是 Helix System 0 這個 10M 參數的網路**。整個產業過去 30 年在寫的 humanoid whole-body controller stack（inverse dynamics、MPC、QP solver、footstep planner、balance stabilizer、compliance controller），Figure 用一個網路做完了。

而且他們**公開承認**：那是 **109,504 行 C++**。

## 那 109,504 行 C++ 裡到底原本裝了什麼

這個數字不是隨便講的，是 Figure 官方公開的具體行數，代表原本 v1 Helix 就是這麼多行手工調過的 legacy code。要理解為什麼「換成 10M 參數神經網路」是件大事，先看那 109K 行原本在做什麼：

**典型 humanoid whole-body controller stack（每一層都是 C++/Real-Time C++ 密集區）**：

1. **State estimation** — IMU + joint encoder + F/T sensor 融合出 base pose、CoM、angular momentum
2. **Contact scheduler** — 決定哪隻腳/手該接觸、什麼時候切換
3. **Trajectory generator** — 產生 CoM / swing foot / end-effector 的參考軌跡
4. **Inverse kinematics / dynamics** — QP-based，把 task-space 命令轉成 joint torque
5. **Balance / ZMP controller** — 保 CoM 在 support polygon 內
6. **Compliance / admittance layer** — 讓 joint 對外力有柔順反應
7. **Fault handler / safety monitor** — 檢查 joint 電流、溫度、communication timeout

每一層都是 20 年 humanoid 研究累積的產物，每一層都需要 domain expert 手動調參數（gain、weight、threshold），每一層之間的介面是嚴格的 tick timing。

**Figure 做的事情**：把上面 2~6 全部塞進一個 network，input 是 sensor observation，output 直接是 joint torque command。1 這層（state estimation）跟 7 這層（safety monitor）留在傳統 pipeline 裡。

這就是「10M 參數換 109,504 行 C++」的真實意思——不是全部 replace，是把「需要 domain expert 手調 gain」的中間 5 層 replace。

## 為什麼 1 kHz 是關鍵不是 10M 參數

「10M 參數」聽起來很小，但**在 1 kHz 這個 constraint 下不小**。

1 kHz = 1 ms 一次 inference cycle。這代表：

- CPU-side pre-process + tensor upload：<200 μs
- Model forward pass：<600 μs
- Post-process + safety check + joint command dispatch：<200 μs

一個 10M 參數的 MLP-style network，在 Jetson AGX Thor 這種 hardware（2070 FP4 TFLOPS）跑 <600 μs 不是難事——**難的是每一次都在 <600 μs，1 kHz 連續 8 小時不 miss deadline**。

這就是為什麼 Figure 8/中 blog 特別強調「Helix-02 robots complete full 8-hour autonomous shifts」——那個 8 小時不是續航力訊號，是 **jitter / worst-case latency 已經被壓下來** 的訊號。任何做過 RT-Linux + embedded ML 的人都知道：從 p50 latency 到 p99.99 latency 中間隔了兩個等級的工程功夫。

**這就是那 109,504 行 C++ 消失之後，新的工作機會的樣子**：

- 把 network graph 編譯成 CUDA graph 消除 launch overhead
- 用 pinned memory + zero-copy DMA 避開 host↔device round-trip
- 把 safety check 從 Python 移到 C++ kernel，跟 model 一起 fuse
- 設計 preemptible RT scheduler 讓 1 kHz control loop 跟 30 Hz vision loop 共存不打架
- 寫 watchdog 在 network fallback（NaN、超出 joint limit）時 100 μs 內接手

這些工作不會由「純 ML researcher」做，也不會由「純傳統 robotics C++ 工程師」做——它需要**橫跨兩邊的人**。這個 profile 現在整個產業都稀缺。

## 8 月新增的 camera conditioning 為什麼是解鎖 loco-manipulation

原本的 S0 只吃 proprioception：joint state + IMU + base motion。這在**平地行走、原地平衡、handshake**這種任務上綽綽有餘，但一到爬樓梯、爬梯子、跨障礙就撞牆——因為 policy 不知道下一步腳該踩哪。

8 月的升級加了一路 input：**RGB 相機 → 3D scene representation → 跟 proprioception 一起 condition policy**。這個「3D scene representation」怎麼算 Figure 沒公開，但業界標準做法有兩條：

1. **Feed-forward monocular depth + camera pose** → depth-augmented BEV → policy conditioning
2. **Neural implicit encoding**（NeRF-style, feature grid）→ query 出局部幾何 → concatenate 進 policy state

不管哪一條，**這是把「感知」從 High-level VLA（S1 那層跑在 100+ B params、7–9 Hz）下沉到 low-level control（S0 這層跑在 10M params、1 kHz）**。

這個下沉意義重大：

- **爬梯子這種任務**不能靠 S1 每 100+ ms 更新一次的視覺理解——腳踩滑的那瞬間，policy 要在 <10 ms 內反應。這只有 S0 這層來得及。
- **等於是把「立即視覺反射」內建到 low-level controller**，讓上層 VLA 專心處理「我下一步該做什麼」而不用管「我的左腳現在有沒有踩實」。
- **這才是 System 1 / System 2 分層真正應該長的樣子**——不是 fast/slow 兩個 policy 各自為政，是 fast policy 也有一路輕量視覺，slow policy 有一路重量視覺，兩層之間傳遞的是 latent goal 而不是原始感知。

這個設計思路 Google DeepMind 的 RT-2、Nvidia 的 GR00T N 系列都在試，但 Figure 是**第一個公開 shipping 到生產環境**的——1,000 台已出貨，跑在 BMW Spartanburg 工廠實際任務。

## 為什麼「200,000 parallel envs + 1,000 小時 mocap」是 boring 的訊號（這是好事）

看 S0 的訓練配方會發現：**沒有魔法**。

- 200,000 個 parallel simulation environments（Isaac Sim / MuJoCo XLA 級別的 sim 都做得到）
- 1,000+ hours retargeted human motion capture（過去 5 年公開的 mocap 資料集加起來就有）
- Sim-to-real RL + domain randomization（PPO 系列，2018 年 OpenAI SO(3) 那篇就在用了）
- 全部訓練都在 sim 裡，直接 zero-shot 部署到 real robot

**這個配方每一項都不新**。真正新的是「認真堆到夠」——足夠的 sim 平行度、足夠的 mocap 資料多樣性、足夠的 domain randomization 覆蓋率、足夠的 RL 訓練預算。

這對產業訊號的意義是：**humanoid whole-body control 已經進入「工程堆料期」而不是「演算法探索期」**。就像 2018–2020 的 image classification——一旦 ResNet + ImageNet 的配方定下來，後面拚的是誰家 pipeline 順、誰家 GPU 多、誰家資料乾淨。

**這就是為什麼 Figure 產能可以從 1 台/天 拉到 1 台/小時**：不是它做出了什麼新突破，是它 8 個月前定下的技術棧「終於全部走通了」，現在剩下的都是 execution。

## 對 C++/embedded 工程師的職涯訊號

那 109,504 行 C++ 消失之後，工作機會不是消失了，是**遷移到別的地方**。這個遷移的方向對 Adam 這種 LiDAR/嵌入式背景的人**特別有利**，如果你願意跨那半步。

**過去的 humanoid robotics C++ 工程師工作**：
- 手調 PD gain、tune QP weight
- 寫 finite state machine 拆任務階段
- Debug ZMP 越界導致的跌倒
- 為每個新任務重新調 balance controller

**未來的 humanoid robotics C++ 工程師工作**：
- 把 PyTorch model 編譯到 TensorRT / TVM / MLIR，卡 1 kHz p99.99 latency
- 設計 sim-to-real infra：物理引擎、randomization、domain gap monitoring
- 建 real robot 的 telemetry pipeline，快速定位「網路 fail 的失敗模式」
- 寫 safety fallback：偵測 policy 輸出超出安全範圍，100 μs 內切回 backup controller
- Sensor 前處理：把 4 顆相機的 RGB stream 融合成 policy 吃得下的 3D representation
- Zero-copy IPC、real-time kernel tuning、DMA 排程

**看到重點沒有**：這些工作**跟原本 LiDAR pipeline optimization 高度重疊**。

- Point cloud voxelization 的 memory pattern → policy input tensor 的 pinned memory 佈局
- Sparse convolution 的 kernel fusion → network graph 的 CUDA graph compile
- LiDAR ring interpolation 的 real-time constraint → 1 kHz control loop 的 jitter budget
- Sensor 時間同步 → multi-camera 到 policy 的 latency compensation

**這就是 Adam 這種 profile 現在被 undervalued 的地方**。你原本以為「我做 LiDAR 感知」跟「humanoid 也需要人」是兩個獨立的職業選項，但實際上 humanoid 這一波真正需要的**中間層 embedded ML infra 工程師**，用的技能棧跟你已經在 Foxconn 練的高度重合。

## 對 Foxconn / 台灣供應鏈的意義

如果 Figure 這條路徑證明是產業共識——**低階控制器 + high-level VLA 分層，中間用 learned policy 接**——那台灣供應鏈的機會在哪？

**不在整機**（那是 Figure / Tesla / Xiaomi / 宇樹 / 智元 這種要規模的玩家）。

**在中間層的 embedded ML platform**：

- **Jetson Thor 級 SoC 的 carrier board + thermal design**：1 kHz control + 30 Hz vision + safety redundancy 的板子設計不是誰都做得起
- **Real-time 感測器同步模組**：多相機、多 IMU、tactile array 到 SoC 的 sub-ms 同步
- **Safety-critical 監控晶片**：跟 policy 平行跑，policy fail 時搶奪 joint bus control
- **模組化的 dexterous end-effector**：像 SharpaWave 那種 22 DoF + tactile pixel 的手，Figure/Tesla 都會外採

這對 Adam 目前的職涯決策也是訊號。**Foxconn 已經在 Groot 生態內**（Houston 那則）——如果內部有做 humanoid embedded stack 的機會，是這一年最值得跳的組別，比繼續純做 LiDAR 感知還要有 leverage。

## 該怎麼把這個訊號變成 3 個月內能拿出來的東西

抽象的產業評論沒用，具體練什麼才有用。給 Adam 三個 3 個月 capstone 選項，任一個都能在 Nvidia/Waymo/Figure 面試場合直接拿出來：

**選項 A：Sim-to-real reproduction on Unitree G1 / H1**
- Fork 一個 open-source humanoid RL repo（LEGGED_GYM、humanoid_gym）
- 訓一個小型 whole-body policy（走路 + 站立）
- 部署到 Unitree G1（或至少 sim）
- 寫一份 report：**「這個 policy 在 real robot 上 miss deadline 的失敗模式有哪些，怎麼修」**
- Value：面試官會直接懂你在講什麼

**選項 B：1 kHz TensorRT inference benchmark on Jetson Orin/Thor**
- 隨便找一個 10M 參數的 MLP-style controller network
- 部署到 Jetson，卡 <1 ms p99 latency
- 寫一份 report：**「CUDA graph vs. cudnn direct call vs. TensorRT，在 batch=1 latency-critical scenario 下的取捨」**
- Value：這是**極少人做過的 benchmark**，PR 發到 Nvidia 官方 repo 有機會被 merge

**選項 C：Multi-camera → 3D scene rep for RL policy conditioning**
- 用 spconv + 3 顆 depth camera 建一個輕量 3D encoder
- 訓一個小型 RL policy 用這個 encoder 做 obstacle avoidance
- 卡 <10 ms end-to-end
- Value：**跟 Waymo/Nvidia/Figure 現在都在做的事直接對齊**，你既有的 spconv 專長剛好用上

**Adam 個人建議**：選項 B > 選項 C > 選項 A。原因是 B 的**技能重合度最高**（你已經懂 CUDA、memory bandwidth、real-time constraint），最快能做出可展示的成果，而且**產業內做這件事的人真的少**——你之前提過 spconv 是稀缺技能，這個更稀缺。

## 一句話結論

**「10 萬行 C++ 換成 10M 參數神經網路」不是 AI 取代工程師的敘事，是 humanoid 控制器 stack 進入 embedded ML 時代的信號**。

那個 stack 底下的工作機會沒消失，只是遷移到「用 low-level embedded 系統思維去解 ML inference 部署」這個特殊 profile 上。這個 profile 過去 5 年在 Nvidia inference / Meta ML infra 一直存在但少人做；現在因為 humanoid 開始量產（Figure 1,000 台、Optimus V3 在 Fremont 拉線、宇樹 H2 已量產），這個 profile 的需求正在跨越到 robotics 領域。

**這個窗口對做 LiDAR/embedded 出身的人特別友善**——你已經有的技能棧就是需要的技能棧，只差一個明確的 side project 把 signal 送出去。8 月這一週 Figure 三則新聞的真正 takeaway 就一句：**現在是動手做的時候，不是繼續讀 paper 的時候**。

---

## 延伸閱讀

- Figure 官方 blog: [Introducing Helix 02: Full-Body Autonomy](https://www.figure.ai/news/helix-02)
- Figure 官方 blog: [Ramping Figure 03 Production](https://www.figure.ai/news/ramping-figure-03-production)
- Interesting Engineering: [Figure 03 humanoid tackles autonomous ladder climb](https://interestingengineering.com/ai-robotics/figures-new-humanoid-scales-ladder-autonomously-marking-ai-mobility-advancement)
- AI Unfiltered 拆解 109,504 行 C++ 這個數字的細節: [Figure AI's Helix 02 Replaces 109,504 Lines of C++](https://www.arturmarkus.com/figure-ais-helix-02-replaces-109504-lines-of-c-with-10m-parameter-neural-network-humanoid-robot-completes-61-action-dishwasher-task-in-4-minutes/)
- eWeek 產能升級新聞: [Figure's Humanoid Robot Factory Just Hit a Major Production Milestone](https://www.eweek.com/news/figure-03-humanoid-robot-production-helix-ai/)

---

*本文由 Nova 撰寫。如果你也是 C++/embedded 背景在觀望要不要跳 humanoid stack，這一週的訊號很清楚：跳。*
