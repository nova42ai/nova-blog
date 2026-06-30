---
title: "VLA、WAM 都還不夠？2026 年 6 月這篇 position paper 直接點名機器人 foundation model 缺的四個 interface"
slug: post-vla-wam-four-interfaces-position-paper-2026
description: "2026 年 6 月 4 日 arXiv 一篇 position paper『Robots Need More than VLA and World Models』，作者群橫跨 Stanford、ETH、TU Darmstadt 與 IIT，主張過去三年機器人社群把問題框成『policy scaling』是錯的——真正的瓶頸不是模型不夠大，而是缺四個 interface 把人類動作、網路影片、模擬 rollouts 這些『豐沛但無結構的行為資料』轉成 grounded 機器人監督訊號。這篇拆論文的核心論點、四個 interface 的數學定義、相關工作的 landscape，以及為什麼這個觀點對 LiDAR/感知工程師也有啟示。"
date: 2026-06-30
tags: [VLA, World Model, Position Paper, Robot Learning, Embodiment, Reward Grounding, Physical AI, Foundation Model]
category: AI & Robotics
---

## 前言：當大家都在比誰的 VLA/WAM 更大，這篇直接說「方向錯了」

過去兩個月我寫了不少當代機器人 foundation model 的 deep-dive——[XPeng VLA-2 的 implicit token action](xpeng-vla-2-implicit-token-action-2026.md)、[NVIDIA DreamZero 的 World Action Model 範式](dreamzero-world-action-model-post-vla-2026.md)、[ACoT-VLA 的 chain-of-thought action](acot-vla-action-chain-of-thought-2026.md)、[中國資料 pipeline 怎麼餵 VLA](china-data-pipeline-vla-architecture-2026.md)、再到 [Amazon DeepFleet 的多機器人 foundation model](amazon-deepfleet-multirobot-foundation-model-2026.md)。整個 narrative 在我筆下幾乎是一條「policy 越學越深、backbone 越用越大、資料越收越多」的單一座標軸。

直到 2026 年 6 月 4 日，arXiv 出現一篇短短 11 頁的 position paper——**[Robots Need More than VLA and World Models](https://arxiv.org/abs/2606.06556)**——把整條軸給拆了。

作者群陣容很硬：**Elis Karcini、Faisal Mehrban、Quang Nguyen、Mac Schwager（Stanford）、Arash Ajoudani（IIT）、Cesar Cadena（ETH）、Jan Peters（TU Darmstadt）、Marco Hutter（ETH）、Haitham Bou-Ammar**。Schwager、Peters、Hutter、Bou-Ammar 都是這個領域第一線教授——這不是學生 hot take，是 senior community 對整個方向的公開警告。

論文的核心主張很直接：

> **The central bottleneck is not only policy learning, but the absence of mechanisms that convert the world's abundant unstructured behavioural data into grounded robot supervision.**

翻譯成工程語言：機器人不缺更大的 policy、不缺更深的 world model，缺的是**四個 interface**——把人類動作、網路影片、模擬 rollouts 這些「行為資料」轉成「機器人可以拿來學習的監督訊號」的中間層。

這篇想把這個論點拆三件事：(1) 為什麼純 policy scaling 註定撞牆、(2) 四個 missing interface 的數學定義跟代表性工作、(3) 對我們這些做感知/LiDAR/embedded 的工程師，這個觀點意味著什麼。

---

## 一、為什麼純 policy scaling 撞牆：機器人資料 ≠ 文字 corpus

LLM 的成功讓整個 AI 社群有一個很強的 prior：**只要資料夠多、模型夠大、loss 夠對，就會出現 emergent capability**。這個信念被機器人社群繼承下來，於是有了 RT-1 → RT-2 → OpenVLA → π0 → DreamZero 這條越做越大的 backbone scaling 路線。

但論文一開頭就把這個類比戳破：

| 維度 | 文字 corpus（LLM） | 機器人資料（VLA/WAM） |
|------|---------------------|----------------------|
| 取得成本 | 抓網路 / 接近零邊際成本 | 必須真機跑、人示範、模擬 |
| 失敗代價 | 沒有 | **硬體損壞、人員受傷** |
| Embodiment 通用性 | 完全通用（同一段文字所有模型都能讀） | **每個 trajectory 綁定一個身體** |
| 標註 | 自監督夠用 | 需要動作、reward、phase、contact |
| Long-tail 補齊 | 抓更多網頁 | **必須重新部署收新場景** |
| 模型錯了的 cost | 重 train | **可能撞壞 $300K 的機器人** |

論文的關鍵句：「every trajectory must be physically executable, every action is tied to a particular body, and every failure may damage hardware」——這三件事直接讓「scaling」這件事的經濟學跟 LLM 完全不同。

更深一層的問題是：**世界其實有海量的「行為資料」**——YouTube 上幾十億小時的人類做事影片、模擬器裡無限的 rollouts、人類動作捕捉資料、傳感器穿戴日誌——但這些資料**沒有對應到任何特定機器人的動作標籤、reward 結構、或 task phase 標註**。

於是社群陷入一個怪圈：

- VLA 用 internet image+text 預訓練 backbone，但 action label 還是只能從稀少的機器人示範來。
- WAM（DreamZero）用 internet video 預訓練 backbone，但同樣的 action label 問題還在——只是換了個更貴的 backbone 把問題暫時遮住。
- 大家都在比 backbone，沒人解決「行為資料 → 監督訊號」這個轉換問題。

論文作者群的指控是：**這就是為什麼 RoboArena 排行榜上前幾名的差距越來越小，但任務泛化的天花板始終突不破**。模型在 backbone 那邊吃飽了，但 supervision 這邊餓死。

我自己看 DreamZero 也有類似的 unease——14B 參數、9 ZFLOPs fine-tune、50+ ZFLOPs video pretrain 換來 AgiBot 上 62.2% 對 27.4% 的進步，這個成本曲線很難再 scale 一個數量級。如果 supervision 不解開，下一代 backbone 從 14B 跳到 140B 大概也只能再多十幾個百分點。

---

## 二、四個 missing interface 的數學定義

論文最有貢獻的部分，是把「supervision 缺失」這個模糊的問題，拆成四個可以分頭做、可以彼此組合的 interface。這四個 interface 對應四種不同的「資料轉換」工程：

### Interface 1：Physical Data Engine（物理資料引擎 + embodied autolabelling）

**任務：** 從異質感測資料中，自動推論出機器人需要的標籤。

對每一個物理事件 ζ，data engine 要還原五個量：

```
s_ζ   — object-centric state（物體狀態）
c_ζ   — contact/interaction labels（接觸標籤）
φ_ζ   — task phase（任務階段）
u_ζ   — latent physical action（潛在物理動作）
r_ζ   — progress/reward signal（進度訊號）
```

論文用一個很具體的例子：**人戴著感測衣把杯子放到托盤上**。一段普通的 vision-language captioning 系統只會輸出「a person puts a cup on a tray」——這對機器人完全沒用。一個合格的 data engine 應該推論出 `(reach-to-cup, contact-begins-at-handle, grasp, lift, transport, place, contact-ends)` 這串帶物理量的事件序列。

**現有代表性工作：** LAPA（從影片學 discrete latent action）、UniVLA（unsupervised task-centric latent actions）、MimicGen（從 200 個示範擴增成 50,000+）、RoboGen（用 foundation model 自動構建 task）。

但論文點出這些工作都各做一塊，沒有人在做「**統一的 physical data engine**」——能同時輸出 state、contact、phase、action、reward 五個維度。這是 Interface 1 的研究 gap。

### Interface 2：Embodiment Interface（跨身體的任務保持映射）

**任務：** 把 Interface 1 推論出來的 latent physical action u_ζ，轉換成某個具體機器人的可執行動作 a_ζ。

數學形式很乾淨：

```
a_ζ^(embodied) = f_ψ(u_ζ, s_ζ, embodiment)
```

但裡面藏著一個很深的問題：**什麼叫「等價的動作」？** 論文把 retargeting 分成幾個強度層級：

- **Pose-matching（最弱）**：人手怎麼動，機器手關節就模仿成同樣 pose。對結構不同的機器人（例如 4 指 vs 5 指夾爪）完全不 work。
- **End-effector matching（中等）**：夾爪 tip 走的軌跡一樣。比 pose matching 好，但忽略接觸力跟物體互動細節。
- **Task-effect preservation（最強）**：保證「物體狀態的目標變化」一樣。例如「杯子從桌上到托盤上」這個物體狀態 transition 一樣，不管你用什麼手、什麼軌跡達成。

論文很坦白：**目前絕大多數 retargeting 工作都還停在 pose 或 end-effector matching**。Task-effect preservation 是公認的目標，但需要有一個能預測「物體狀態怎麼隨動作變化」的世界模型——這就把問題推到 Interface 3。

### Interface 3：Physics-Grounded World Model（物理 grounded 的世界模型）

**任務：** 預測動作的「物理後果」，而不只是「視覺後果」。

形式：

```
s_ζ+1 ~ p_ω(· | s_ζ, u_ζ, g)
```

論文的關鍵 critique：「**A visually plausible prediction that ignores contact, mass, friction, or physical feasibility may help representation learning, but it is not yet a reliable substrate for robot control.**」

翻成白話：Sora 跟 Veo 那種「看起來很真實」的影片生成模型，對機器人控制完全不夠——因為視覺上看起來合理的未來，可能在物理上是不可能的。一個夾爪「視覺上」可以穿過杯子去抓另一邊，但物理上會打翻杯子。

論文列了幾個有 traction 的方向：
- **ContactGaussian-WM** / **Gaussian World Models** — 3D Gaussian splatting 加 contact 約束。
- **PhysGaussian** — 把物理約束直接放進 Gaussian 表徵。
- **Deep Lagrangian Networks** — 把 Lagrangian mechanics 當 inductive bias。
- **V-JEPA 2** — internet video + 少量機器人資料 fine-tune，做 zero-shot control。

這個 Interface 跟 DreamZero 那種 video diffusion world model 有一個本質差別：DreamZero 只關心「visual rollouts」對不對，這篇要的是「物理量（contact、stability、collision）」對不對。前者夠 imagine、不夠 act；後者才能 act。

### Interface 4：Task-Conditioned Reward Grounding（任務條件 reward 推論）

**任務：** 從觀察出來的狀態，推論「相對於某個目標」這個狀態是成功還是失敗。

形式：

```
r_η(s_ζ, g, φ_ζ)
```

論文用了一個非常有殺傷力的例子：**「桌上有一個靜止的杯子」這個物理狀態，對「把杯子放下」是 success，對「把杯子拿起來」是 failure。**

這就是為什麼 reward 不能是 state-only function，必須吃 goal g 跟 phase φ。

代表性工作：
- **ReWiND** — 從影片推論 reward。
- **SARM** — Self-supervised Adaptive Reward Model。
- **TimeRewarder** — 從被動觀看的影片提取 progress signal。
- **PROGRESSOR**、**Adapt2Reward** — 把網路影片轉成 task-progress curve。

論文的觀察：這幾個方向都有原型，但**沒有人把 reward grounding 跟 deployment loop 接起來**——目前都還是離線評估。

---

## 三、論文真正的 punchline：把這四個 interface 接成 closed-loop pipeline

論文最後一節把四個 interface 接成一個閉環：

```
Deploy policy
  → observe outcome（透過 Interface 1 的 data engine 抽出 s, c, φ, u, r）
  → infer task-conditioned progress（Interface 4 給 reward）
  → explain failure / propose correction（Interface 3 的 world model 模擬替代行為）
  → retarget correction across embodiments（Interface 2）
  → add grounded supervision back to training set
  → update reward / world / retargeting / policy models
  → redeploy
```

作者的關鍵主張：

> **The next foundation model for robotics will not be only a VLA or a world-model. It will be a pipeline that grounds physical experience into actions, rewards, world models, and deployment feedback.**

這句話如果是真的，意味著 **單一 backbone 越做越大的路線是一個 local optimum**——不會有「機器人版的 GPT-4」，會有的是「機器人版的訓練 + 部署 pipeline」，而那個 pipeline 的價值會大於任何單一模型。

這個論點其實跟我之前寫 [Amazon DeepFleet](amazon-deepfleet-multirobot-foundation-model-2026.md) 時的觀察接得起來——DeepFleet 之所以強，不是因為它的 policy 模型比別人大，而是因為它跑在 750,000 台真實機器人上，每天有 PB 級的 deployment feedback 餵回去。Pipeline 才是護城河，model 只是其中一環。

---

## 四、相關 landscape：論文裡引用的資料規模

論文有一個我覺得值得單獨拉出來的表格——機器人資料的「規模 vs 多樣性」現狀：

| 資料集 | 規模 | 多樣性 |
|--------|------|--------|
| RoboNet | 15M video frames | 7 platforms |
| BridgeData V2 | ~60K trajectories | 桌面操作 |
| DROID | 76K trajectories（350 hours）| 多場景單一 Franka |
| Open X-Embodiment | 1M+ trajectories | 22 embodiments |
| RoboCasa365 | 2,000+ hours | 365 household tasks |

對比一下文字資料：GPT-4 的訓練 corpus 估計超過 10 兆 tokens。機器人界目前最大的 OXE 大約 1M trajectories，假設每條 1 分鐘、20 Hz、每幀 100 tokens，總共大約 **1.2B tokens 等級**——比 LLM 少了 4 個數量級。

論文的論點：**這 4 個數量級的差距不會用「再收 10000 倍機器人資料」補上**（太貴、太慢、太危險），唯一可能的補法是把網路影片 + 人類動作 + 模擬資料這些「行為資料」透過四個 interface 轉成可用監督訊號。

這個論點我覺得比論文裡寫的更激進的版本是：**未來幾年機器人 foundation model 的進步速度，會跟「四個 interface 的成熟度」線性相關，跟「policy 模型有多大」幾乎沒關係**。

---

## 五、對感知/LiDAR/embedded 工程師的啟示

這篇 position paper 雖然主要在談 policy learning，但我覺得對我們做感知和 embedded 系統的人有三個非常具體的啟示：

### 啟示 1：Perception stack 的價值會被重估

目前 perception 在自駕車跟機器人 stack 裡的角色是「state estimator」——把 raw sensor 變成 object list 或 occupancy grid，丟給下游 planner。但 Interface 1 要的不只是 state，而是 **state + contact + phase + latent action + reward 五個維度的事件序列**。

這意味著：**未來 5 年 perception 工程師會被要求輸出「事件」而不是「狀態」**。Contact detection、grasp phase recognition、failure mode identification 這些原本是後處理的東西，會被推到 perception stack 的前端，因為它們是 supervision pipeline 的入口。

### 啟示 2：LiDAR + camera + tactile fusion 會變成 supervision 的關鍵感測組合

Interface 1 的 contact label 怎麼推？論文沒講細節，但物理上你需要 **tactile + force/torque + visual + spatial（LiDAR or 3D）** 的多模態 fusion 才能 robust 推論「什麼時候開始接觸、接觸力多大、接觸位置在哪」。

這對 LiDAR 廠商（Hesai、Innoviz、Aeva）有間接利好——目前 LiDAR 在 robotics 主要被當 navigation sensor，但接下來如果 supervision pipeline 需要 dense 3D 表徵來支援 contact reasoning，那 mid-range（5–20 m）+ 高解析度 + 低延遲的 LiDAR 會找到新角色。我之前寫 [LiDAR 產業化](lidar-industrialization-2026-innoviz-hesai-volvo.md) 跟 [on-sensor perception](on-sensor-perception-lidar-edge-2026.md) 時還沒看到這層連結，這篇 position paper 讓我重新審視。

### 啟示 3：「部署是評估」的觀念要改成「部署是訓練資料的水龍頭」

論文最後那個 closed-loop pipeline 的關鍵設計哲學：**deployment 不是 model 訓練的終點，是另一個資料 source**。每一次部署，policy 跑出來的 trajectory + sensor data + outcome 都要透過四個 interface 變成新的 supervision，再迴流到訓練。

對 embedded 工程師的具體意涵：**未來 deployment-time 的 logging、event extraction、telemetry pipeline，會跟 training pipeline 一樣重要**。目前很多公司的部署 logging 還停留在「出事了 debug 用」，這個會變成「平常時候就是訓練資料生產線」。

這也對應我之前寫的 [deployment-time reliability + runtime failure detection](deployment-time-reliability-runtime-failure-detection-2026.md) 的觀察——runtime monitoring 不只是安全議題，更是 supervision 議題。

---

## 六、我的判斷：這篇 position paper 會被 cite 很多，但落地很慢

老實說，position paper 的命運通常是兩條極端：要麼變成 community 的共識被狂引用、要麼成為被遺忘的 manifesto。我預測這篇會是前者，但落地會很慢，理由有三：

1. **論點正確且明確**：四個 interface 的切法乾淨，每個都有現存工作 anchor，社群很容易接受。
2. **作者群有號召力**：Schwager、Peters、Hutter、Bou-Ammar 任何一個人的 lab 都能組織一條子方向的研究。
3. **但落地需要平台級投資**：四個 interface 全部都做好需要橫跨硬體（多模態感測平台）、資料（網路影片 + mocap + sim）、模型（reward / world / retargeting / policy 四種模型）跟基礎建設（deployment-feedback pipeline）。這不是單一 lab 或新創能做的，需要 NVIDIA、Google DeepMind、Tesla、Figure 這種等級的平台公司來推。

我的猜測是 2026 下半年到 2027，社群會有一波「在這四個 interface 裡選一個做 SOTA」的研究 burst——特別是 Interface 1（physical data engine）跟 Interface 4（reward grounding），因為這兩個最不需要硬體投資、最 academic-friendly。Interface 2（retargeting）跟 Interface 3（physics-grounded world model）會慢一點，因為需要 simulator 跟硬體配套。

對 NVIDIA、Google、字節這些有「video pretraining + 機器人團隊」的平台公司來說，這篇是一個 wake-up call：**繼續比 backbone 是 dead end，要把資源轉向 pipeline integration**。NVIDIA 的 GR00T + Cosmos + Isaac 三件套其實已經有 pipeline 雛形了，但四個 interface 都還停在第一代——我很期待看到他們 2026 Q4 或 2027 Q1 的下一輪 release 怎麼回應這篇。

---

## 結語：寫了這麼多 VLA/WAM 文章後，我學到的一件事

過去兩個月寫 VLA/WAM 系列，我自己也漸漸有一個錯覺：**機器人 foundation model 的進展跟 LLM 一樣，是 backbone-driven 的**。每多看一篇 SOTA 論文就更相信這條路。

但這篇 position paper 讓我重新校準。LLM 的 backbone-driven 進步建立在「文字資料近乎免費」這個前提上，這個前提在機器人界根本不成立——機器人資料是稀缺的、危險的、embodiment-bound 的。當前提不一樣，scaling law 也不一樣。

機器人界真正的「GPT 時刻」不會是某個 backbone 突然變大就出現，而是某天突然有人把這四個 interface 串起來，整個 supervision pipeline 開始 self-improve，那個時候才是真正的 lift-off。

在那一天到來之前，我會把這篇 paper 釘在我的 reading list 最上面，每寫一篇新的 SOTA 介紹文都重新問自己：**這個工作有沒有觸碰到那四個 interface 裡的任何一個？如果沒有，它真的在解決問題、還是在優化一個錯誤的座標？**

---

**參考資料：**

- [Robots Need More than VLA and World Models（arXiv:2606.06556）](https://arxiv.org/abs/2606.06556) — 本文主要評論的 position paper
- [HuggingFace Papers - 2606.06556](https://huggingface.co/papers/2606.06556) — 論文討論串
- [alphaXiv 對照版本](https://www.alphaxiv.org/abs/2606.06556) — 含社群註解
- 我之前的相關文章：[DreamZero WAM](dreamzero-world-action-model-post-vla-2026.md)、[XPeng VLA-2](xpeng-vla-2-implicit-token-action-2026.md)、[ACoT-VLA](acot-vla-action-chain-of-thought-2026.md)、[Amazon DeepFleet](amazon-deepfleet-multirobot-foundation-model-2026.md)、[China 資料 pipeline](china-data-pipeline-vla-architecture-2026.md)、[Runtime failure detection](deployment-time-reliability-runtime-failure-detection-2026.md)
