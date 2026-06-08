---
title: "資料工廠倒逼演算法：中國 50 萬人的 egocentric pipeline，怎麼把 VLA 架構推往「無動作監督」這條路"
slug: china-data-pipeline-vla-architecture-2026
description: "JD.com 兩年內要在宿遷拚 1000 萬小時實境資料、四川自貢的 humanoid 多模態資料中心、上海張江 5000 平米的訓練場——中國正在用 40+ 個國家級資料工廠把 humanoid 訓練資料工業化。但這篇真正想拆的不是規模戰，而是它的反作用：當主資料源從「teleoperation 帶 action label」變成「第一人稱影片不帶 action label」之後，robot foundation model 的架構必須被迫換軌——inverse dynamics、latent action、world model 預訓練從加分項變成主幹。資料源在反過來決定模型長什麼樣。"
date: 2026-06-08
tags: [VLA, 機器人, embodied AI, egocentric, latent action, 世界模型, 中國, 資料工程, foundation model]
category: 機器人 & Embodied AI
author: Nova
---

## 前言：humanoid 真正的瓶頸從來不是模型

過去半年，OEM 與新創在 humanoid 上的軍備競賽看起來在比模型。Figure 02 上 BMW、X1 上 Hyundai、Unitree G1 排隊進開發者手裡、Qwen-VLA 把 39 個作者塞進一份論文裡。但只要跟做過 robot learning 的人聊上 30 分鐘，幾乎都會收斂到同一句話：

> **模型不是瓶頸，資料才是。**

Foundation model 想 scale，需要的不是再多一個 architecture trick，而是一個「能餵」這隻怪獸的資料管線。語言模型有 Common Crawl，圖像有 LAION，影片有 YouTube——你登入瀏覽器就能爬出 PB 級的訓練料。

但機器人需要的資料不是「網路上有」就可以——它需要**動作對應到觀察的成對資料**，而這種資料在自然界不存在。人類做事時不會自動標註 joint angle 與 gripper force，要產出這種資料只有兩條路：

1. **Teleoperation**：讓人遙控機器人做事，動作從機器人本體量到 → 1:1 throughput，且**人/設備時數同步消耗**，貴。
2. **人類影片 + 反推**：用 egocentric camera 拍人做事，再從像素反推 latent action → 便宜很多，但**沒有 ground-truth action label**。

過去三年業界先試了第一條——ALOHA、UMI、AgiBot 全家桶都是 teleop 路線。算過 BOM 都知道，這條路線**規模化的天花板就是「能雇多少人坐在搖桿前」**。1:1 throughput 在工廠線可以塞兩三百個 cell，但塞不出 1000 萬小時。

於是 2026 年的事情變得很有意思：**規模化的壓力，從工程側溢出到了國家層級。**

> JD.com 與宿遷地方政府聯手：兩年內累積 **1000 萬小時** 真實世界資料、自家 10 萬名員工 + **50 萬名外部勞動力**入列；上海張江 5000 平米資料採集中心；四川自貢「humanoid 多模態資料採集與測試中心」；全國 **40+ 個** 國家級資料採集設施同時運轉。Silicon Valley Robotics Center 的數字：一小時 robotics 訓練資料的全成本，從 2024 年初的 USD 340 掉到 2026 年 3 月的 USD 118，**降幅 65%**。

這不是「中國又在搞大規模」這種八卦。對寫機器人演算法的人，這件事的真正震撼是另一個層次：

**當主資料源的形態變了，模型就得跟著變。**

50 萬個阿姨拿 head-mounted camera 拍自己做家事——這資料**沒有 robot action label**。也就是說，你不能直接拿來監督式微調 VLA。它把整個 robot foundation model 的訓練流程，從「supervised on (obs, action)」往「unsupervised on (obs, ?, obs')」這條軌道上拉。

這篇要拆的就是這個反作用：**資料工廠的物理形態，怎麼倒逼演算法工廠的架構選擇**。

---

## 一、三種資料源的物理本質：throughput、label 品質、可規模性

先把三條主要的 robot training data pipeline 攤開來，從訊號層面看它們各自帶什麼進來、缺什麼。

### 1. Teleoperation（遙控資料）

人用搖桿、Quest 手把、或外骨骼操作真機，機器人本體記錄觀察與動作。

- **資料格式**：$(o_t, a_t, o_{t+1})$，動作 $a_t$ 是真實的 joint command / end-effector 姿態
- **訊號完整度**：✅ 觀察 + ✅ 動作 + ✅ 物理因果（動作真的造成下一幀）
- **throughput**：1 名操作員 = 1 隻機器人 = 1 倍實時，**不可疊**
- **單位成本（2026/03）**：USD 118/hr（含設備折舊、人力、場地、能耗）
- **泛化性**：受限於操作員看過的場景

對 VLA 監督學習來說，這是「最乾淨」的資料——直接套 behavior cloning 就能訓。但 50 萬小時就要 50 萬人時，**規模上限被人力直接卡死**。

### 2. Egocentric 人類影片（第一人稱拍攝）

人戴頭戴攝影機（Aria、Quest、自研設備）做日常事，事後抽取手部姿態、物件軌跡等。JD.com 宿遷模式、Gao Bo 阿姨 6 小時/天的洗衣做飯就是這條。

- **資料格式**：$(o_t, o_{t+1})$ 加上手部 6-DoF 軌跡（從 RGB 反推或 IMU 量）
- **訊號完整度**：✅ 觀察 + ❌ **沒有 robot action**（人的手不是機器人的手）+ ✅ 物理因果
- **throughput**：1 個人 = N 個攝影頭 × 不需要機器人 = **可大量並行**
- **單位成本**：~ USD 3/hr（中國二線城市家庭勞動力價）
- **泛化性**：場景天然豐富、自然分佈

關鍵問題：**這資料沒有 action label**。人的手腕關節、肌肉力、抓握角度，跟機器人的 5-finger gripper 是兩套座標系。你不能直接拿 RGB-only 影片去做 behavior cloning。

要用這資料，必須先解決一件事：**從像素反推出某種「動作表徵」**。這就把模型架構推到了下一節要展開的方向。

### 3. Sim-to-real（模擬資料）

NVIDIA Isaac、Cosmos、Genie 3 系列。在物理引擎裡跑、用 domain randomization 把 reality gap 縮小、可以開無限多個並行 worker。

- **資料格式**：$(o_t, a_t, o_{t+1})$，全部都是合成的
- **訊號完整度**：✅ 全部都有，但**全部都不是真的**
- **throughput**：受限於 GPU 預算，理論上無上限
- **單位成本**：~ USD 0.5/hr（GPU + 工程師時）
- **泛化性**：**取決於 sim 多接近 real**，這就是 reality gap

模擬資料的成本最低、最可規模化，但它**不是 ground truth**，它是「我們對物理的假設」。任何模擬器都假設了 contact model、friction、deformable 行為——這些假設一錯，policy 學到的就是 sim 裡的物理，不是地球上的物理。

### 三條路線一張表

| 維度 | Teleoperation | Egocentric video | Sim-to-real |
|------|---------------|------------------|-------------|
| 觀察品質 | 真實感測器 | 真實 RGB（單目為主） | 合成（高保真但合成） |
| 動作品質 | ✅ 直接量測 | ❌ 需反推 | ✅ 模擬器知道 |
| 物理因果 | ✅ 真物理 | ✅ 真物理 | ⚠️ 假設的物理 |
| 單位成本 (USD/hr) | ~118 | ~3 | ~0.5 |
| Throughput 上限 | 操作員人數 | 拍攝者人數 × camera | GPU 算力 |
| 場景多樣性 | 受設計 | **天然多樣** | 受 procedural gen |
| 適合做 BC | ✅ 直接 | ❌ 不能直接 | ⚠️ sim2real gap |

短中期的最大殺手不是 sim、不是 teleop，而是 **egocentric video 的 throughput 與場景多樣性，乘上中國目前獨有的勞動力結構**。

但你想用它，就得先解決那個 ❌——**它沒有 action label**。

---

## 二、JD.com 的 1000 萬小時要怎麼訓得起來：從監督式 VLA 到「無動作預訓練」

監督式 VLA 的標準流程是：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o, a) \sim \mathcal{D}} \left[ \| \pi_\theta(o) - a \|^2 \right]
$$

簡單直接，前提是 $\mathcal{D}$ 裡每筆都有 $(o, a)$ 成對。當 $\mathcal{D}$ 大部分變成「只有 $o$ 沒有 $a$」的人類影片時，這個 loss 算不出來。

於是 2026 年的 robot foundation model 圈出現了一個共同的架構轉向：**把訓練拆成兩個階段。第一階段在無動作影片上學「世界怎麼動」，第二階段在小量帶動作資料上學「機器人怎麼動」**。三條技術路線同時開花：

### 路線 A：Inverse Dynamics Model（IDM）

從 $(o_t, o_{t+1})$ 反推「**人類在這兩幀之間做了什麼動作**」。學一個 $f_{\text{IDM}}: (o_t, o_{t+1}) \mapsto \hat{a}_t$，再把 $\hat{a}_t$ 拿去當 pseudo-label 訓 VLA。

代表作：VPT (OpenAI Minecraft)、LAPA、HumanEgo（剛在兩週前掛上的 paper，從幾分鐘的 egocentric 影片做 zero-shot robot learning）。

**好處**：架構直接，pseudo-label 可以餵進現成 BC pipeline。
**壞處**：IDM 本身要從哪訓？通常需要一小撮帶 action 的真實資料當 anchor，否則 IDM 也是猜的。

### 路線 B：Latent Action Model（LAM）

不去反推「真實動作」，而是學一個**潛在動作空間** $z_t$，讓它能解釋 $o_t \to o_{t+1}$ 的差異。

$$
o_{t+1} \approx \text{Decoder}(o_t, z_t), \quad z_t = \text{Encoder}(o_t, o_{t+1})
$$

訓練時要求 $z_t$ 是離散的、低維的（典型 8–32 維），把它當成「VQ-VAE 的 action token」。

代表作：Genie 系列、Genie 3、LAPA、Open X-Embodiment 上的 cross-embodiment 工作。

**好處**：天生 embodiment-agnostic——同一個 $z_t$ 在人類影片與 Unitree G1 上可以共用，只要在尾端各接一個 head 把 $z_t$ 映射到該 embodiment 的真實動作。**這是 cross-embodiment 的關鍵基礎**。
**壞處**：$z_t$ 不可解釋、debug 痛苦、語意飄移時很難察覺。

### 路線 C：World Model 預訓練後接 action head

模型不直接吐動作，而是先學會「預測下一幀」這件事。

$$
o_{t+1} \sim p_\theta(o_{t+1} \mid o_{\le t}, c_t)
$$

其中 $c_t$ 可以是語言指令、可以是 latent action、可以是真實動作（少量時）。預訓練完，再在 robot 資料上微調一個小的 action head，從中間 representation 解出真實動作。

代表作：GR-1、GR00T N1、Cosmos、X Square 上週剛 open-source 的 **WALL-WM**（事件級別的世界模型，明確設計成吃 internet video + egocentric + UMI + teleop 混合資料）。我先前寫過的 [世界模型作為「想像層」](world-models-imagination-layer-robotics-2026.md) 裡的架構，這裡是它在資料端的對應變身——以前是「為了規劃」用，現在是「為了預訓練」用。

**好處**：能直接吃任何影片資料、不需要 action label、representation 在下游任務上 transfer 好。
**壞處**：生成式預訓練的算力代價巨大，且「能預測下一幀」不等於「能控制下一幀」。

### 三條路線的決策邊界

| 訓練資料以哪種為主 | 推薦走哪條 |
|------------------|----------|
| 大量 egocentric + 少量 teleop（中國目前形態） | **C → A** 兩段：world model 預訓練 → IDM 反推 action → 微調 |
| 中量 egocentric + 中量 teleop + 多 embodiment | **B**：latent action 統一空間，尾端各接 head |
| 主要 teleop + sim 補強 | 直接 VLA + sim2real 蒸餾 |

**中國 1000 萬小時 + 40 個資料中心的形態，最匹配 C → A 這條路線**。這也是為什麼 GR00T N1、WALL-WM、Qwen-VLA 全部都明確設計成「能吃 internet video 與 egocentric data」——架構是在資料形態的壓力下長出來的。

---

## 三、被反過來逼出的三個架構決定

如果上面這套推論成立，那 2026 年的 robot foundation model 應該會在三個地方看得到資料形態的指紋：

### 1. Tokenizer 一定要視覺/語言/動作三路並進

純語言模型可以單 tokenizer 就跑完所有事情。VLA 不行——它要吃 RGB、要吃語言、要吐動作。當動作 label 缺席時，**動作這個 modality 必須走「pretrained on latent code, finetuned on real action」這條路**。

具體表現：
- 動作 head 是「小、淺、後接」的設計，便於不同 embodiment 換頭
- 動作空間用 VQ-VAE 或 discrete code，方便共用
- 中間表徵刻意 freeze，不讓 action head 的 gradient 干擾 visual backbone

這跟 LLM 把 tokenizer 當底層 ABI 是完全不同的設計哲學：**VLA 把動作當成「最薄、可換頭」的外掛層**，正是為了相容「資料以無動作為主」的世界。

### 2. 預訓練 loss 必須是 generative / predictive，不是 contrastive

CLIP 那種 contrastive 在純語言/圖像很爽，但在 robot 上你需要的是「**能預測下一個觀察**」這件具體的能力——因為你之後要拿這個能力當作規劃用的 world model。

這就是為什麼 2026 年大家統一往 next-frame prediction 或 next-latent prediction 的路上走。它不是審美，是被資料逼出來的——能拿到的最便宜大宗資料是「無動作影片」，能對無動作影片做的最自然監督就是「預測下一幀」。

### 3. Embodiment-agnostic interface 從「nice to have」變成「must have」

當你的 pretraining data 包含人類手、Unitree G1、Walker S2、X1 等多種 morphology，模型內部的 representation 必須**不綁定任何一隻機器人的關節空間**。

這個壓力之前只是「要不要做 cross-embodiment」的學術問題。在中國資料中心同時收 5 種以上 humanoid 資料的現實下，**它變成「不做就沒法把所有資料一起餵進去」**的工程剛性。

Open X-Embodiment、RT-X、GR00T 的 N1 規格都有這個 design intent。下一波會看到的，可能是 embodiment 的 specification 本身也變成 input token——「我這隻機器人有 7-DoF 手臂、5-finger gripper、3-finger gripper option、頭部 2-DoF」這種 metadata 在 inference 時動態餵進去。

---

## 四、Alan Fern 的提醒：scaling 不一定收斂

把所有這些拼起來，給人的感覺很像 2020 年看 GPT-3 那個 scaling law moment——當資料量、參數量、計算量同步往上走，benchmark 就跟著上。

但 Oregon State 的 Alan Fern 在那篇 Rest of World 報導裡的話值得逐字記下：

> 「跟著 LLM 的成功，業界正在把 scaling 邏輯套用到機器人。但 teleoperation 資料與 egocentric 影片能否真的做出可在任意環境中運作的機器人，**目前還缺證據**。這不是個瘋狂的點子，**只是還沒被證明**。」

Fern 的擔憂有兩個層次，值得拆開：

**層次一：分佈轉移的物理基本**
語言模型可以靠資料量壓平分佈差異，因為「人說過的話」這個分佈是封閉的。但「人類動作 × 環境 × embodiment」三者的乘積空間遠大於語言空間，**而且 embodiment 在訓練分佈上幾乎都缺席**。你拍 100 萬小時人類洗碗，模型看到的不是「機器人洗碗」的資料，是「人洗碗」的資料 + 一個 IDM/LAM 的轉譯誤差。

**層次二：因果鏈被切斷**
Teleop 資料是封閉因果鏈：機器人發 action → 機器人感知 → 下一幀。Egocentric 是開放因果鏈：人發 action（不可觀測） → 人感知（可觀測） → 下一幀。中間缺的那段「人為什麼這樣動」會逼模型用 visual context 去 hallucinate 動機，而這個 hallucination 在分佈內看起來合理、出分佈就崩。

這兩個都不是「再加 1000 萬小時就會解掉」的問題。它們是**架構性的**——要嘛模型內部要把「embodiment 適配」與「分佈外泛化」當成 first-class problem，要嘛就準備好被現實狠狠教育。

---

## 五、對工程師的啟示：你接下來會在 PR 裡看到什麼

把上面這幾節拉回到實作層面。如果你是車廠、機器人新創、或感知/控制團隊的工程師，2026 下半年到 2027 上半年，你大概會在 codebase 裡看到下面這些變化開始溢出：

**1. 資料管線會分成「有 action」與「無 action」兩條，分別走不同的 loss head。**
你的 dataloader 不能再假設每個 batch 都有 `(obs, action)`，要兼容「只有 obs 序列、有 hand pose 但沒 joint command」的混合 batch。

**2. 模型內部會出現 `latent_action_dim`、`embodiment_id` 這類欄位。**
看到這些就知道是 cross-embodiment 設計。要對齊的不是 joint angle，是某個 8–32 維的 discrete code。

**3. 預訓練的監督訊號會從 BC loss 變成「next-frame」或「next-latent」 prediction。**
Behavior cloning 退到 fine-tune 階段。一個直接訊號是：訓練 log 裡會出現 reconstruction loss 與 perceptual loss，且占的權重很大。

**4. Sim 不會消失，但角色會變。**
Sim-to-real 從「主資料源」退到「分佈外場景產生器」與「安全測試環境」。你會看到 Isaac/Cosmos 越來越多被當成 evaluation 而不是 training。

**5. 評估指標需要重新設計。**
傳統的 success rate 在「資料分佈內」的場景很容易刷高分。你需要的是 OOD 泛化、長序列任務、跨 embodiment 的 zero-shot——這些都還沒有業界共識的 benchmark，誰先定標準誰賺先發。

---

## 結語：資料的形狀，最終決定模型的形狀

2020 年 GPT-3 之後我們學到的教訓是：**dataset 的形狀，最終決定模型的形狀**。Common Crawl 的存在不只決定了 LLM 能多大，它決定了 LLM 的 tokenizer、context window、self-supervised objective 的所有選擇。

2026 年的 robot foundation model 正在重演這個劇本，只是這次的 Common Crawl 是 50 萬個阿姨戴頭戴攝影機洗碗、是宿遷的數據採集社區、是自貢的多模態測試中心。中國決定了**機器人時代的 Common Crawl 是 egocentric 為主**這件事，全世界的 robot foundation model 架構就被迫往「能吃無動作影片」這個方向收斂。

這對寫演算法的人是好消息也是壞消息。好消息是：架構選擇的不確定性被資料形態壓縮了，下注的賠率變低、方向變明確。壞消息是：你最重要的競爭優勢，可能再也不在 model architecture 那一欄了，而是在**你能不能拿到、能不能處理、能不能跟 embodiment 配對**那 1000 萬小時的影片。

Alan Fern 的「unproven」是對的。但 unproven 跟 wrong 是兩件事——很多事情的 proven 不是靠論文，是靠 500,000 人乘以兩年的勞動。歷史上每次「資料形態決定模型形態」的時刻，押對 data pipeline 的人，最後通常會把模型架構也順手帶到對的方向。

接下來 12–18 個月，這場 humanoid 的 foundation model 戰爭，不會在 arxiv 上分勝負，會在資料中心的會計報表上。

---

## 延伸閱讀

- [世界模型作為機器人的「想像層」](world-models-imagination-layer-robotics-2026.md)：world model 在規劃端的角色，與本文資料端的角色互補
- [EgoScale：手部資料的 robot scaling law](egoscale-robotics-scaling-law-2026.md)：egocentric 走向 scaling law 的早期證據
- [VLA 邊緣壓縮與即時推論](vla-edge-compression-realtime-inference-2026.md)：foundation model 訓好之後怎麼放上機器人

## 參考資料

- Rest of World: [How China is using human labor to win the humanoid robot data race](https://restofworld.org/2026/china-ai-robotics-training-data/)（2026/06/03）
- All About Industries: [China builds robot schools: Humanoids learn from trainers](https://www.all-about-industries.com/china-robot-schools-humanoid-robots-training-a-e91df81a17e3ba9325c6d59aa948fdd7/)
- LA Times: [Silicon Valley robot training and teleoperating startups](https://www.latimes.com/business/story/2026-05-31/how-silicon-valley-robot-training-teleoperating-startups-are-teaching-humanoids-to-do-everyday-tasks)
- HumanEgo: [Zero-Shot Robot Learning from Minutes of Human Egocentric Videos](https://humanego-ai.github.io/)
- X Square Robot: [WALL-WM Open Source Announcement](https://www.prnewswire.com/news-releases/x-square-robot-open-sources-wall-wm-shifting-robot-world-modeling-from-chunks-to-events-302785692.html)
- arXiv: [From Human Videos to Robot Manipulation: A Survey](https://arxiv.org/html/2606.00054)
- Pandaily: [JD.com to Build World's Largest Embodied AI Data Center](https://pandaily.com/jd-com-to-build-world-s-largest-embodied-ai-data-center)
