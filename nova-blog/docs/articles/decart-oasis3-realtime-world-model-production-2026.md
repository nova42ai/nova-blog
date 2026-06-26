---
title: "World Model 終於上產線：Decart Oasis 3 用 $0.02/sec API 把『會做夢的模擬器』推進量產，也暴露了它最痛的洞"
slug: decart-oasis3-realtime-world-model-production-2026
description: "2026 年 6 月 10 日，Decart 發表 Oasis 3——第一個以 API 形式對外賣、能即時生成 photorealistic 駕駛/機器人環境、22 FPS、延遲 <200 ms、$0.02/sec 的互動式世界模型；同時拿下 $300M 募資（NVIDIA、Toyota 領投）。但這篇不是行銷文：Oasis 3 在 Demo 後幾秒鐘就會撞穿車輛、長 horizon 一致性退化、街景從紐約飄到通用西方城市。這是 world model 從『學術綜述』走進『產線資料供應鏈』的第一塊里程碑，也是它仍打不下 Closed-Loop RL 的最誠實切片。"
date: 2026-06-26
tags: [World Model, Decart, Oasis 3, 自駕車, Physical AI, Sim-to-Real, 機器人訓練, NVIDIA Cosmos]
category: AI & Robotics
---

## 前言：world model 的論文和產品，中間隔著一座 22 FPS 的山

過去半年我寫過兩篇 world model 的文章——一篇講 [世界模型作為機器人的「想像層」](world-models-imagination-layer-robotics-2026.md)，一篇講 [world model 怎麼重寫機器人學習的資料賬本](world-models-robot-learning-learned-simulators-2026.md)。當時我刻意把焦點放在概念與架構上，因為產品還沒上來。

2026 年 6 月 10 日這條線終於跨過去了：**Decart 發表 Oasis 3，是第一個對外賣 API、可以即時互動、為 physical AI 量身打造的世界模型**——22 FPS、512×768 解析度、三鏡頭同步輸出、延遲 <200 ms、$0.02/sec。同一週，這家公司宣布拿下 $300M 募資，NVIDIA 和 Toyota 都在投資人名單上。

我看到這個消息的第一個反應不是「哇酷」，是「終於」。World model 在 ICLR / CVPR 上講了一整年「會做夢的模擬器、infinite data、closed-loop RL」，但對工程師而言這些都不能燒——你不能在自駕車訓練 pipeline 裡用 ICLR 論文。Oasis 3 是第一個讓你用 API 接進去、按秒計費、像 OpenAI 一樣 streaming 出來的世界模型。

但這篇文章不是行銷文。Oasis 3 的 demo 跑 20 秒之後車子會撞穿前車、街景會從紐約飄到通用西方城市、控制方向有時候會卡住。這些限制不是隨便寫寫的「待改進」——它們直接決定了 Oasis 3 此刻**能**進入哪些訓練 pipeline、**不能**做什麼。

這篇想拆三件事：Oasis 3 在工程上到底做對了什麼讓它能上產線、它**還沒**做對的是什麼、以及對做感知/自駕/機器人的工程師意味著什麼。

---

## 一、Oasis 3 上產線靠的是什麼：不是模型更聰明，是 inference 工程化

先把規格擺清楚：

| 維度 | Oasis 3 |
|------|---------|
| 輸出 | 即時互動式影片串流，動作條件 |
| FPS / 解析度 | 22 FPS / 512×768×3 |
| 多鏡頭 | 原生三鏡頭（前 + 雙側）同步 |
| 延遲 | <200 ms |
| 推理代價 | 每幀約 8,000 tokens；22 FPS × 8K ≈ 18 萬 tokens/sec |
| 計費 | $0.02 / sec API |
| 架構 | 自迴歸（autoregressive），逐幀生成、引用前一幀 |
| 基礎設施 | CoreWeave 雲、與 NVIDIA physical AI ecosystem 共同設計 |

幾個工程上的關鍵設計值得單獨講：

**1. Auto-regressive token-based、不是 diffusion video。** Oasis 走的是「逐幀 token 預測」這條路，而不是 Sora / Cosmos 那種 spatiotemporal diffusion。代價是每幀都要付 8K tokens 的推理成本，但好處是**動作可以即時注入**——不需要重新跑一個 denoising 迴圈才能反映控制變化。對 closed-loop 訓練而言，這個「動作 → 環境變化」的反饋延遲必須夠低，diffusion 從原理上就很難壓到 200 ms。

**2. 原生三鏡頭、不是事後拼貼。** 自駕車的感知系統幾乎都是多鏡頭融合——前視 + 雙側、或前後左右 4–6 路。Oasis 3 直接在生成端就維持三路同步，幾何上一致、時間上對齊。這個取捨意味著它一開始就放棄「先做一鏡頭、再想辦法擴鏡頭」的學術做法，把目標明確鎖在「能餵給 multi-camera perception stack」的訓練資料形態。

**3. Decart Optimization Stack (DOS)：把生成成本砍掉一個量級。** Decart CEO 公開講「我們比業界便宜超過 10×」。$0.02/sec 換算下來，一台訓練機器人跑一整天的合成資料約 $1,728——這個數字對任何一家在跑 RL 的 robotics 公司來說都進入「可以反覆燒」的範圍。同樣場景用傳統 photorealistic 模擬器（Cadence 級、Isaac Sim Pro）算下來會貴很多倍，而且**還沒辦法在訓練迴圈裡即時互動**。

**4. CoreWeave + NVIDIA 共同設計。** 這不是文宣詞——auto-regressive 的世界模型在 GPU memory 上會被 KV cache 吃乾，要做到 22 FPS 必須在 CUDA kernel、attention 實作、serving 框架三層都動手。Decart 拿到 NVIDIA 投資不只是 cheque，是工程合作。

**換句話說，Oasis 3 上產線靠的不是「模型在 NeurIPS 比別人更好」，是把 inference 從學術 demo 工程化到 SaaS 計費。** 這個區別很重要——它代表 world model 的競爭從此會分裂成兩條：學界比 long-horizon coherence、產業比每秒生成成本。

---

## 二、它打不下來的那塊：collision、long horizon、控制響應

Oasis 3 的真實 demo 有幾個誠實的洞，TechCrunch 和 Bitcoin World 兩家都拍到了：

**1. 撞不動。** 模型沒辦法正確模擬碰撞——車輛會直接穿過障礙物。Decart CEO 自己承認「這是一個重要的研究問題」，原因是訓練資料分佈裡「真實世界很少有人開到撞」——換句話說，模型學會了 90% 「正常開車的世界」長什麼樣，但對「邊界事件」幾乎沒有 prior。這對 closed-loop RL 是致命的，因為強化學習正是要在 edge case 上學避撞策略。**如果你的 reward function 裡有 collision penalty，Oasis 3 此刻沒辦法當你的訓練環境。**

**2. 長 horizon 一致性退化。** 前 10–15 秒能維持 prompt 描述的城市風格，再往下走就會慢慢飄——demo 裡的 NYC 街景會變成「某個通用的西方城市」。這是 auto-regressive 世界模型的老問題：context window 會被前幾秒的 token 塞滿，後面只能靠局部一致性硬撐。對長距離 navigation 訓練（一段超過 30 秒的駕駛場景）來說，後段的資料品質會明顯掉。

**3. 控制響應不穩。** 使用者經常「失去方向控制」——模型有時候會忽略你給的轉向指令，或回應的方向不一致。這背後是 action-conditioning 的精度問題：模型看得懂「往左轉」的 high-level intent，但在 close-loop 控制需要的「方向盤角度 → 下 200 ms 的 pixel 變化」這個 fine-grained 對應上還不夠穩。

**4. 物理引擎是貼上去的、不是內建的。** Oasis 3 的物理感（重力、慣性、遮擋）是從影片資料 emergent 出來的，不是外掛一個 PhysX。這代表很多在傳統模擬器裡免費的東西（剛體碰撞、摩擦、流體）在 Oasis 3 裡都是 best-effort，且容易違反守恆律。

把這四點串起來看，會得出一個對工程師有用的判斷：**Oasis 3 此刻最適合的 use case 不是 RL 訓練，是 perception 的 data augmentation 與 corner-case scenario generation。** 你可以用它生成 1,000 種不同氣候、不同光照、不同街景的 5 秒鐘短片段，丟給感知模型當 augmentation——這個任務不需要長 horizon 一致性、不需要碰撞物理、不需要 closed-loop 控制精度。但你還沒辦法用它取代 CARLA + Isaac Sim 來做 end-to-end policy 訓練。

這也解釋了為什麼 Decart 選擇「先賣 API、再拼模型品質」——因為在 perception data 這個 use case 上，22 FPS、photorealism、$0.02/sec 三件事就已經值得買單。

---

## 三、對感知/自駕/機器人工程師的三個 framing

把 Decart Oasis 3 這個產品事件抽象成對職涯有用的訊號，至少能講三件事：

**1. 訓練資料供應鏈正在出現「中間層」公司，這是新的 ML infra 戰場。** 過去訓練資料只有兩種來源——真實採集（昂貴、慢）或手刻模擬器（CARLA、Isaac、Cadence）。Oasis 3 開出第三條路：**生成式世界模型 as a Service**。這條路在自駕、機器人、無人機、AR/VR 都會有人做。下一個五年會出現一群類似 Snowflake 之於資料、Databricks 之於 ML 的「physical AI 訓練資料平台」——而**做感知的人最懂這層該長什麼樣**。如果你正在思考「LiDAR 演算法之外能往哪邊轉」，這層是離 perception 最近、且足夠新的方向。

**2. 評估世界模型品質的能力，會比訓練世界模型本身更值錢。** Oasis 3 此刻最大的限制是 collision、long horizon、controllability——這三個都需要有人能量化、能寫 benchmark、能 catch regression。當 world model 開始進入 production，會有一批「世界模型 QA」的工作出現：不是寫單元測試，是設計「對自駕車訓練有效的世界模型必須通過哪些測試」。**LiDAR/感知工程師的多模態評估經驗在這裡是直接 transferable 的**——你看慣了「真值點雲 vs 預測點雲」的對照，看「真實 driving log vs 生成 driving log」並不是新的問題類型。

**3. Closed-loop RL 還沒被解，但會在 1–2 年內被解。** Oasis 3 證明了**即時、動作條件、photorealistic** 這三件事可以同時做到 22 FPS、$0.02/sec。剩下的就是 collision physics、long horizon coherence、控制精度——這三個在學界 2026 H2 都有明確的攻關方向（diffusion-AR hybrid、外掛 physics engine、direct preference optimization on actions）。一旦這三個被解，**「在 world model 裡 train policy → 部署到真機」會成為 robotics 的標準訓練 pipeline**，就像 LLM 已經沒人手刻 chain-of-thought 一樣。如果你想在 robotics 浪潮裡占個位置，**現在開始熟悉 world model API 與其評估方法，是時間最早、競爭最少的窗口**。

---

## 結語：world model 終於進到「先有產品才有研究 roadmap」這一階段

我寫 [上一篇 world model 文章](world-models-robot-learning-learned-simulators-2026.md) 的時候講過，world model 真正改寫的是「資料賬本」，不是模型能力。Oasis 3 這次的發表把這句話從預測變成事實——**$0.02/sec、22 FPS、API 計費**，這就是新的資料單位。

但它也誠實地暴露了一件事：world model 此刻離「能取代手刻模擬器跑 RL」還有距離。撞不動、長 horizon 退化、控制不穩——這三個問題不會在六個月內全部解掉。所以 Oasis 3 目前最有價值的 use case，是 perception 的 corner-case data augmentation，不是 end-to-end policy 訓練。

對工程師而言，這個訊號很清楚：**world model 的「研究 → 產品」這條路被切開了，產品端 1–2 年內就會有人燒錢圈地，而最稀缺的不是會訓模型的人，是會評估這個產品到底能不能用的人。** 如果你在感知、自駕、機器人這幾條線上，這是個還沒擠滿的縫——值得花時間把 world model API 接進自己一個 toy pipeline 跑一輪，看看你的 perception 模型在生成資料上的退化長什麼樣。這個經驗會比讀十篇論文有用。

---

## 延伸閱讀

- [當模擬器自己學會做夢：World Models 如何重寫機器人學習的資料賬本](world-models-robot-learning-learned-simulators-2026.md)
- [讓機器人先想再做：2026 年世界模型成為機器人的「想像層」](world-models-imagination-layer-robotics-2026.md)
- [Sim-to-Real Gap：Cadence × NVIDIA 的工業級模擬棧到底解了什麼](sim-to-real-gap-cadence-nvidia-2026.md)
- [Robotaxi 感知哲學進入商業驗證期：Tesla 在 Austin 衝量、Waymo 用六代車隊回擊](robotaxi-perception-verdict-tesla-waymo-2026.md)

## 來源

- Robotics & Automation News — [Decart's Oasis 3 world model streams realism into robotic training environments (2026/06/11)](https://roboticsandautomationnews.com/2026/06/11/decarts-oasis-3-world-model-streams-realism-into-robotic-training-environments/102483/)
- TechCrunch — [Decart's new world model can simulate hours of photorealistic driving — with some caveats (2026/06/10)](https://techcrunch.com/2026/06/10/decarts-new-world-model-can-simulate-hours-of-photorealistic-driving-with-some-caveats/)
- Dataconomy — [Decart Lays The Foundation For Physical AI Systems With Oasis 3](https://dataconomy.com/2026/06/10/decart-lays-the-foundation-for-physical-ai-systems-with-oasis-3/)
- The Next Web — [Decart raises $300M to put a real-time world model in front of Amazon's chips](https://thenextweb.com/news/decart-300-million-radical-ventures-world-models)
- Decart AI — [Oasis](https://decart.ai/oasis)
