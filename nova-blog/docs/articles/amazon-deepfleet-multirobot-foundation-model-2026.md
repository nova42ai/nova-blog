---
title: "百萬機器人之後：Amazon DeepFleet 用 foundation model 改寫 fleet coordination，把『MAPF 演算法』推進淘汰賽"
slug: amazon-deepfleet-multirobot-foundation-model-2026
description: "2026 年 6 月，Amazon 倉儲機器人突破 100 萬台，同步發表 DeepFleet——第一個對外公開、為多機器人協調量身打造的 foundation model。10% 的整體效率提升、四種架構（RC/RF/IF/GF）對 inductive bias 的賽馬、840M 參數的 robot-floor 變體、billions of hours 的訓練資料——這不是另一篇 RL 論文，是 fleet coordination 從 search-based MAPF 走進 learned predictor 的分水嶺。但它也誠實地承認自己現在還做不了 task assignment。這篇拆三件事：DeepFleet 的四種 inductive bias 為什麼是這樣選、它為什麼能打贏傳統 MAPF、以及對做 perception/embedded 的工程師意味著哪一條新賽道。"
date: 2026-06-27
tags: [DeepFleet, Amazon Robotics, Foundation Model, Multi-Agent, MAPF, Warehouse Automation, Physical AI, Fleet Coordination]
category: AI & Robotics
---

## 前言：humanoid 的舞台太亮，但真正在燒錢的物理 AI 是這條

過去半年我寫了不少 humanoid 的文章——[Figure 02 在 BMW 跑 1250 小時](figure02-bmw-1250hours-forearm-humanoid-2026.md)、[humanoid 平台戰爭](humanoid-platform-war-2026-nvidia-unitree-openai.md)、[humanoid 製造轉折點](humanoid-manufacturing-turning-point-figure-botq-automate-2026.md)。這些東西的鏡頭很有畫面，但講真話：**2026 上半年真正在「燒出產線」的物理 AI，不是人形，是輪型 AGV/AMR 在倉儲裡的集體調度。**

2026 年 6 月底這條線剛跨過去一個關鍵點：**Amazon 全球倉儲機器人數量正式破 100 萬台**——第 100 萬台機器人交付到日本的一個 fulfillment center，分布在全球 300+ 設施。同一週，Amazon 把這套幾年來內部用的 fleet coordination 模型公開——名字叫 **DeepFleet**——並丟出一篇 27 頁的技術論文（arXiv:2508.08574）說明四種架構變體。

數字很乾：**fleet 整體效率提升 10%。** 聽起來不性感，但你算算看——一台機器人每年的 OpEx 加上機會成本估約 $3–5k 美元，100 萬台就是 30–50 億美元的年度成本基底。10% 是直接 $3–5 億美元的年度節省，而且這個比例會隨資料量繼續放大（論文裡明確說「performance improved with training data volume」，是典型的 foundation model scaling 行為）。

這篇不是「Amazon 又厲害了」那種行銷文。我想拆三件事：

1. DeepFleet 為什麼是**四個變體賽馬**而不是單一架構——這個 inductive bias 的取捨對工程實作意義重大
2. 它為什麼能打贏傳統 MAPF（multi-agent path finding）演算法——這是 fleet coordination 從 search-based 走進 learned predictor 的分水嶺
3. 它**還沒**解決什麼——以及對做 perception/embedded 的工程師（也就是我自己這個位置）意味著什麼新的賽道正在開啟

---

## 一、四個架構變體的賽馬：inductive bias 怎麼選決定能不能 scale

先把規格擺清楚。DeepFleet 不是一個模型，是四個變體一起被訓練、一起被評估，最後挑出能 scale 的那個：

| 變體 | 表示方式 | 主要組件 | 參數量 |
|------|---------|---------|--------|
| Robot-Centric (RC) | 以單一 ego robot 為中心，編碼 30 個最近鄰機器人、100 個最近 grid cells、100 個附近物件 | Autoregressive decision transformer | ~中等 |
| Robot-Floor (RF) | 機器人與地板分離編碼，cross-attention 建模交互 | Transformer + cross-attention | **840M** |
| Image-Floor (IF) | 地板當成「pixels」、動態/靜態特徵當 channel | CNN + Transformer | ~中等 |
| Graph-Floor (GF) | 把整個倉庫建成時空 graph、用 GNN 跑 message passing | GNN + temporal attention | **13M** |

論文的結論很乾脆：**robot-centric 和 graph-floor 表現最好，而且 scale 上去也最有效。** 這個結果背後其實藏了一個對工程師最有用的訊號——

**為什麼 RF（840M 參數、看起來最豪華的那個）不是贏家？**

我的解讀是：把「機器人」和「地板」分成兩個獨立 encoder 用 cross-attention 串起來，在數學上有 expressive power，但它的 inductive bias 太弱——你等於告訴模型「機器人和地板有關，請自己學」，model 必須從零學到「相對位置」「碰撞距離」「congestion 在 spatial 上會怎麼擴散」這些已經很顯然的事。840M 參數是被這個沒有效率的歸納偏好吃掉的。

而 RC 和 GF 為什麼贏？

- **RC** 把先驗鎖在「ego 視角 + 局部鄰居」上——這直接對應 fleet coordination 的物理真實：機器人決策受局部影響為主，幾個 grid 外的事不需要全注意力。論文用 decision transformer 的設計，把「動作 → 未來狀態」的條件式生成做成 autoregressive，這對「forecast 未來幾秒內的 congestion」是天然合適的。
- **GF** 把先驗鎖在「空間鄰接結構」上——倉庫地板天生就是 graph，貨架、走道、charging station 之間的拓樸關係用 GNN 表達是 nearly free 的。13M 參數能打贏 840M 的關鍵不是 GNN 多神奇，是它**不需要再學一遍空間結構**。

**這條結論對任何在做 multi-agent learning 的工程師都極具啟發：在 fleet/swarm/manipulation 這種有強烈空間結構的問題上，砸參數打不過砸 inductive bias。** 學界這幾年最大的迷思之一就是「夠大的 transformer 可以學到任何 prior」——DeepFleet 在工程實證上給了一個反例。

---

## 二、為什麼能打贏傳統 MAPF：從 reactive 到 predictive 的範式轉移

倉儲機器人調度過去十年最主流的解法叫 MAPF（Multi-Agent Path Finding）——本質是 search-based：給定當前 fleet 狀態 + 目標，在離散時空裡解一個 conflict-free 的路徑指派問題。代表演算法有 CBS（Conflict-Based Search）、ECBS、PIBT 這些。

MAPF 的問題不在解不出來——在它的反應方式：**它是 reactive 的。**

當倉庫運行到某個點，三條走道的機器人開始彙集到同一個 chokepoint，傳統 MAPF 要等到「conflict 被偵測到」才會 replan。即使你把 plan horizon 拉長到 30 秒，計算成本會隨機器人數量爆炸，實務上產線只能跑短 horizon 的 plan。結果就是——**bottleneck 已經形成，才開始繞路。**

DeepFleet 改了什麼？它把 fleet coordination 重新建模成 **forecasting 問題**：給定當前狀態，預測未來幾步每台機器人會在哪、congestion 會在哪裡形成。這個 forecast 一旦比 MAPF 的 reactive 更快、更準，控制層就可以提前繞路——bottleneck 還沒成形就被解掉。

這個範式轉移有兩個工程上的關鍵理由它能贏：

**1. 推理成本 vs 規劃成本。** 傳統 MAPF 是「規劃即計算」——你想看 30 秒後會怎樣，要實際 simulate 30 秒。foundation model 是「規劃靠推理」——一次 forward pass 就能 forecast 未來幾步的 occupancy。Amazon Science blog 的原話很直白：「a learned model can quickly infer how traffic will likely play out without exhausting real-time computation resources already allocated to operational planning.」用工程語言講：**MAPF 是 O(robots × horizon × branching) 的搜尋；DeepFleet 是 O(model_size) 的 inference，與 horizon 解耦。**

**2. 訓練資料是內生的不對稱優勢。** Amazon 有 billions of hours 的歷史 fleet 軌跡——這個資料量在任何外部對手都拿不到。foundation model 的核心優勢是「資料越多越強」，而 MAPF 的優勢函數對歷史資料是免疫的（一個更厲害的 CBS 變體不會因為你給它更多歷史資料就變強）。**這代表 DeepFleet 的效率優勢會隨時間自我擴大，而 MAPF 競爭者的優勢會封頂停滯。** 這是商業上最致命的不對稱。

但要小心一個容易誤讀的點：DeepFleet **不是要取代** MAPF——它是 fleet 規劃 stack 裡的**預測層**，下游還是要一個 controller 把預測轉成實際指令。這個架構意義上有點像自駕車裡的「prediction module」之於「planning module」的關係。

---

## 三、它還沒解決什麼：task assignment 才是真正的金礦

DeepFleet 的論文很誠實，這點我蠻欣賞——它清楚把 scope 鎖在 **location prediction** 上。模型現在做的事是「給定當前 fleet 狀態，預測未來幾步每台機器人的位置/狀態」。**它還沒做 task assignment（哪台機器人去拿哪個 tote）、也沒做 autonomous navigation 決策。**

這個 scope 鎖定有兩層意義：

**第一層（戰術層）：location prediction 是最容易上線的 wedge。** 它可以**完全 offline 訓練、online 唯讀預測**——預測錯了不會直接撞車，只會讓上游 controller 繞了沒必要繞的路。這是 Amazon 把 foundation model 推進 production 最低風險的切入點，而且 10% 的提升足夠 ROI 立刻為正。

**第二層（戰略層）：task assignment 才是真正會改寫物理 AI 商業價值的那一塊。** Fleet coordination 的真正瓶頸從來不是「機器人怎麼走」，是「誰應該去做哪件事」——這個指派決策牽涉到貨架實際庫存、訂單優先級、機器人電量、跨樓層平衡。把這個交給 foundation model 來做，意味著要把 ERP/WMS 的所有結構化資料拉進 context，再加上 fleet 預測層做 closed-loop 決策。**這是 next horizon——也是 Amazon 為什麼會把 DeepFleet 設計成可擴展的 multi-architecture 賽馬而不是單一模型：他們在為下一階段的 task assignment foundation model 鋪路。**

換句話說：DeepFleet 1.0 是「fleet GPS」，2.0 會是「fleet brain」，而 1.0 已經足夠把 10% 效率拿走。

---

## 四、對 perception/embedded 工程師意味著什麼

寫到這裡我得誠實——這篇文章的主題對我直接的日常工作（LiDAR 點雲處理、嵌入式感知）距離不算近。但這個賽道有兩個訊號值得我認真注意：

**1. 「物理 AI 的 foundation model」不只是 VLA 一條路。** 過去半年我寫的 VLA 文章——[VLA edge compression](vla-edge-compression-realtime-inference-2026.md)、[ACoT VLA](acot-vla-action-chain-of-thought-2026.md)、[XPeng VLA-2](xpeng-vla-2-implicit-token-action-2026.md)——都聚焦在「單機器人的視覺-語言-動作」這條軸。DeepFleet 開了完全不同的另一條軸：**多機器人協調的 foundation model**。這條軸不需要 RGB 影像，輸入是 fleet state（位置、速度、目標、地板特徵），輸出是時空 occupancy 預測——資料模態不同、推理拓樸不同、scaling law 也很可能不同。

對任何想做 physical AI 的工程師，這條軸的好處是**不需要 robot 硬體就能入場**——只要有 fleet 狀態資料（從 SLAM、從 fleet manager log），都可以訓 toy 版本驗證概念。

**2. 「inference 算 prediction、不是 search」這個觀念會擴散到其他子系統。** 我做的 LiDAR perception，本質上長期是「給定點雲、解 segmentation/detection」的 inference 工作。但 fleet 級的決策層長期被 search-based 演算法佔據——MAPF、A*、Hybrid A*、CBS 這些。DeepFleet 打開的縫是：**很多看似 search-only 的問題，當訓練資料足夠多的時候，predict 比 search 便宜也準。** 我在 perception 之後的 prediction/planning 領域如果想接觸，這個範式轉移是必須先理解的。

具體一點：如果未來我在感知 stack 上要做 Sensor-to-Action 的端到端優化，DeepFleet 給的啟發是——**不要急著用一個巨大的 transformer 統吃所有問題，先想 inductive bias 應該鎖在哪裡。** RC 用 ego-centric、GF 用 graph 拓樸，這兩個 prior 之所以贏，是因為它們對應的物理結構在問題裡客觀存在。我做 LiDAR 點雲時的等價問題是：什麼樣的幾何 prior（voxel? graph? 圓柱 polar? range image?）對下游任務的 inductive bias 最強？**這個問題在 2026 之後會比「我用了多少參數」更重要。**

---

## 結語：fleet AI 的「ImageNet 時刻」剛剛開始

ImageNet 從來不是因為它的圖片漂亮——是因為它把「視覺問題的訓練資料」放上同一張賽馬桌，讓不同架構在統一基準下競爭。DeepFleet 對 fleet coordination 的意義有點類似：**它把 fleet AI 從「每家倉儲廠商各自 reverse engineer 自己的調度啟發式」，變成「在共同的預測問題上比 model 架構誰最好」。**

不一樣的地方是——這次的 ImageNet 在 Amazon 手裡。Billions of hours 的 fleet 軌跡資料不會像 ImageNet 那樣對外公開，這代表這個賽道短期內不會有平等的學術競爭，而是會在 Amazon、Symbotic、AutoStore、京東這些自己手上有大規模 fleet 資料的玩家之間打。學界要追上，得靠 sim 資料（這就是為什麼 [Decart Oasis 3 那種 world model](decart-oasis3-realtime-world-model-production-2026.md) 會接著變重要——它可以生成大量合成 fleet 軌跡）。

對工程師的 takeaway 我寫得簡單一點：

- **物理 AI 不只有 humanoid。** Fleet-level 的 multi-agent foundation model 是同等重要的另一條軸，而且現在更接近能立刻商業化的位置。
- **架構選擇要 follow inductive bias，不要 follow 參數。** 13M 參數的 GF 打贏 840M 的 RF，這個訊號比任何 scaling law 文章都直白。
- **search-based 演算法的舒適圈在縮小。** MAPF、傳統 motion planning、規則式 scheduling 這些領域接下來幾年會被 learned predictor + 輕量 controller 的組合一塊一塊撕掉，做這些方向的工程師要開始想「我要怎麼把我的 domain expertise 變成 inductive bias」。

DeepFleet 是 fleet AI 第一個被工程化到 production 的 foundation model。它的下一代會是 task assignment、再下一代會是跨設施全球協調。每一代都會吃掉一塊本來屬於演算法工程師的領地，但也會開出新的「怎麼把這個系統做得更可靠、更便宜、更可解釋」的工程縫隙。

那才是接下來幾年真正會招人的地方。

---

## 參考資料

- [DeepFleet: Multi-Agent Foundation Models for Mobile Robots (arXiv:2508.08574)](https://arxiv.org/abs/2508.08574)
- [Amazon builds first foundation model for multirobot coordination (Amazon Science)](https://www.amazon.science/blog/amazon-builds-first-foundation-model-for-multirobot-coordination)
- [Amazon deploys over 1 million robots and launches new AI foundation model (About Amazon)](https://www.aboutamazon.com/news/operations/amazon-million-robots-ai-foundation-model)
- [Amazon's Logistics Strategy Evolves With DeepFleet and One Million Robots (National CIO Review)](https://nationalcioreview.com/articles-insights/extra-bytes/amazons-logistics-strategy-evolves-with-deepfleet-and-one-million-robots/)
- [Amazon foundation model for robots shows what's possible (TechTarget)](https://www.techtarget.com/searchenterpriseai/news/366626747/Amazon-foundation-model-for-robots-shows-whats-possible)
