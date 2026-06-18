---
title: "XPENG VLA 2.0：把語言層整個拿掉，自動駕駛 VLA 走進「隱式 token」時代"
slug: xpeng-vla-2-implicit-token-action-2026
description: "XPENG VLA 2.0 已在 2026 Q1 量產上 Ultra 車型。它做了一件具爭議性的事：把 VLA 經典 Vision→Language→Action 管線裡的「語言」層整個拿掉，改走 Vision-Implicit Token-Action。從 ACoT-VLA、Alpamayo R1 的對照來看，自動駕駛 VLA 已正式分裂成兩條路線：留下語言當鷹架，或徹底拋棄它換來控制頻率與一致性。"
date: 2026-06-18
tags: [AI, 自動駕駛, VLA, XPENG, Physical AI, 端到端]
category: AI & Robotics
---

## 前言：當「語言」變成 VLA 的負債

過去兩年，自動駕駛圈最熱的話題是 VLA（Vision-Language-Action）。

公式看起來很乾淨：相機看到的東西進來、LLM 在中間推理一段、最後輸出方向盤角度與油門踏板。Tesla 的 FSD v13、Li Auto 的 MindVLA、NVIDIA 的 Alpamayo——大家都在這個範式裡卷規模、卷推理、卷世界模型。

但 2026 年 Q1 量產、4 月起在 Ultra 車型上密集鋪開的 XPENG VLA 2.0，做了一件業內爭議極大的事：

> **它把「Language」層整個拿掉。**

XPENG 在新聞稿裡稱這個架構為 **Vision-Implicit Token-Action**：視覺進來，過一層「隱式 token」就直接出動作。沒有自然語言、沒有人類可讀的 reasoning trace、沒有 CoT。

何小鵬在 CVPR 2026 的訪談裡用了一個很重的詞——「**language is poison**」（語言是毒藥）來形容傳統 VLA 在自動駕駛任務上的這條中介路線。

這篇文章想把 XPENG VLA 2.0 的技術抉擇拆乾淨：他們到底拿掉了什麼？為什麼能拿掉？跟我前幾天寫的 [ACoT-VLA](/articles/acot-vla-action-chain-of-thought-2026) 路線對比起來，自動駕駛 VLA 的兩條岔路其實已經非常明確了。

---

## 為什麼 VLA 一開始要長「語言」這條尾巴？

要理解「拿掉語言」的決定，得先回答另一個問題——它為什麼一開始會在那？

### 語言層的三個原始功能

當 RT-2 / OpenVLA 那一代把 VLM（Vision-Language Model）改造成 VLA 時，自然語言其實扛了三件事：

1. **預訓練的能力遷移**：CLIP、Qwen-VL 那種預訓練的 vision-language 對齊，已經把世界知識壓進了 LLM 的權重裡。直接砍掉這條尾巴等於放棄這些先驗。
2. **任務描述的入口**：「靠右停在便利商店前面」這種指令本來就是自然語言，需要一個語言層當入口。
3. **可解釋性鷹架**：CoT 推理產出的文字推理 trace 對 debug、安全審查、責任歸屬都有用，量產車輛特別需要。

這三個功能在「VLA 從 0 走到 1」的階段是必要的——沒有它們，模型沒辦法在合理的資料量下泛化。

### 但量產一上路，三件事都成了拖油瓶

當你要把 VLA 真的灌進量產車、跑在邊緣晶片、控制週期要進入 20–50 Hz 區間時：

| 原本的優勢                    | 量產時的代價                                                 |
| ----------------------------- | ------------------------------------------------------------ |
| 預訓練 VLM 知識遷移           | LLM 解碼 token-by-token，造成 latency floor，動輒 300–500 ms |
| 自然語言任務入口              | 駕駛任務的指令空間其實非常封閉（停車、變道、過彎），語言泛化能力的代價遠大於收益 |
| 文字 reasoning trace 可解釋   | 進入毫秒級控制後，沒人讀得完那串文字；且 reasoning 與實際動作常出現脫節（hallucinated CoT） |

XPENG 的觀點是：當 VLA 從研究 demo 跨進量產車這道門檻之後，**語言這條尾巴的邊際成本，已經超過它的邊際效益**。

---

## Vision-Implicit Token-Action：到底拿掉了什麼，又留下了什麼？

「拿掉語言」不等於「沒有中介」。重點是中介的形式變了。

### 傳統 VLA 管線

```
camera frames
      │
      ▼
[Vision Encoder]
      │  visual tokens
      ▼
[LLM Decoder]  ←──── instruction tokens
      │  text tokens（reasoning + action description）
      ▼
[Action Detokenizer]
      │
      ▼
steering / throttle / brake
```

問題：每一個 text token 都要過 LLM 的自迴歸解碼，每一次解碼都是一次 forward pass。

### XPENG VLA 2.0 的 Vision-Implicit Token-Action

```
camera frames
      │
      ▼
[Vision Encoder]
      │  visual tokens
      ▼
[Implicit Reasoning Transformer]
      │  latent tokens（不解碼成文字！）
      ▼
[Action Head]
      │
      ▼
trajectory / control commands
```

關鍵在 **Implicit Reasoning Transformer** 這層——它跟 LLM decoder 的本質差別有三：

1. **不做 token detokenization**：中間 latent 永遠停留在連續向量空間，不過 softmax、不選詞、不會 hallucinate 出與動作脫鉤的句子。
2. **單次 forward 而非自迴歸**：傳統 LLM CoT 要跑 N 次 forward 才湊出一段推理；隱式 token 只要一次 forward 就出 K 個 latent step。延遲從 O(N) 降到 O(1)。
3. **與 action head 端到端訓練**：latent 的「語意」是被動作梯度雕刻出來的，不是被人類語料雕刻出來的。所以這些 token 對控制任務本身更 grounded。

換句話說，XPENG 沒有放棄「推理」，他們放棄的是「**把推理產出成人類可讀文字**」這個產物。

---

## 數字背後：規模、資料、晶片

技術路線聽起來再漂亮，VLA 終究是個堆出來的活兒。XPENG 揭露的 VLA 2.0 規格非常重：

- **參數量**：72B 等級（部分早期版本是 30B，但量產版上 Ultra 的是 72B）
- **訓練資料**：1 億段駕駛 clips，官方對等換算為「人類駕駛 65,000 年」的累積經驗
- **每次迭代消耗 token**：超過 4 兆
- **車端算力**：自研 Turing AI 晶片，三顆冗餘配置共 2,250 TOPS
- **訓練投入**：何小鵬透露年度 AI 訓練支出約 5 億美元

這組數字之所以重要，不是「XPENG 比 Tesla 大」這種無聊的比較，而是它揭露了一個工程現實：

> **走「隱式 token」這條路，是要用模型規模和資料規模去買回原本語言層帶來的世界先驗。**

語言層帶來的 VLM 先驗是「免費的午餐」——只要對齊好，模型就能用 CLIP / Qwen-VL 那一兩兆 token 的世界知識。一旦把它拿掉，這部分的能力就得用更大的駕駛專屬資料集從頭訓出來。

XPENG 之所以能做這個取捨，前提是他們已經有規模化的真實駕駛資料採集管線（中國市場的車隊回傳）+ X-World 世界模型補資料。對於資料採集規模小一兩個數量級的對手而言，這條路是封死的。

---

## X-World：把世界模型當「資料工廠」

跟 VLA 2.0 一起發佈的還有一份 **X-World Technical Report**——一個可控、多視角的駕駛世界模型。它在 VLA 2.0 訓練流水線裡扮演的角色非常微妙：

- **資料倍增器**：真實採集的 1 億 clips 不夠？用 X-World 在條件控制下生成 corner case（雨夜超車、相同路口不同方位的閃車、施工錐桶突然倒下）。
- **離線驗證器**：模型訓練完先在 X-World 模擬器裡跑十萬次 rollout，找出 long-tail 失敗模式再回灌訓練集。
- **CVPR 2026 投稿**：X-World 已經是學術曝光點。

這跟一年前大家對「世界模型」的想像不太一樣。最早的 Sora / Genie 路線是想用世界模型直接當 planner，但工程上太貴。XPENG 把它退回到**資料工廠**的位置，反而走通了。

---

## FastDriveVLA：把視覺 token 砍掉 75%

VLA 2.0 量產的另一個工程關鍵是 XPENG 跟北大合作、入選 AAAI 2026 的論文：**FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning**。

核心觀察很單純：人類駕駛者看路時，注意力其實只放在前景幾個關鍵物件（車道線、前車、行人、號誌）上，背景（天空、遠處建築、路邊圍籬）對「下一秒要怎麼開」貢獻趨近 0。

FastDriveVLA 的做法：

1. 在 vision encoder 後面接一個 reconstruction-based importance scorer
2. 對每個 visual token 算「砍掉之後重建誤差多大」
3. 保留高重要度的 token，砍掉其餘的

實測結果：把 visual token 從 3,249 個壓到 812 個——**保留率僅 25%，計算量大幅下降，nuScenes planning 精度幾乎沒掉**。

放到生產環境的意義：

- 同樣的 Turing 晶片可以塞下更大的 reasoning transformer
- 控制頻率有空間從 20 Hz 推到 50 Hz
- 多餘的算力可以挪去跑安全冗餘模型

這篇論文跟 VLA 2.0 不是「兩個專案」，它就是 XPENG 為了把 72B 模型塞進量產車而做的優化工程的學術產出。

---

## 對照組：自動駕駛 VLA 已經分裂成兩條路線

把這篇放在最近三個月的 VLA 進展裡看，路線分裂已經非常明確：

### 路線 A：保留語言當鷹架（Alpamayo R1、Li Auto MindVLA）

- **代表**：NVIDIA Alpamayo R1（首款 open reasoning VLA，與 Mercedes-Benz CLA 合作 2026 Q1 上市）、Li Auto MindVLA
- **理由**：語言層的 reasoning trace 對監管、責任歸屬、可解釋性都有價值；歐美主管機關特別吃這套
- **代價**：必須在控制頻率上妥協，或者大幅縮短 reasoning 長度

### 路線 B：徹底拋棄語言中介（XPENG VLA 2.0）

- **代表**：XPENG VLA 2.0
- **理由**：賭量產車的控制頻率需求遠大於可解釋性需求；用模型規模和資料規模買回語言層的世界先驗
- **代價**：可解釋性大幅下降，需要在 X-World 模擬器與大量 rollout 上補回安全驗證

### 第三條路：在動作空間裡推理（ACoT-VLA）

- 這是我前幾天寫過的 [ACoT-VLA](/articles/acot-vla-action-chain-of-thought-2026)：不在語言、也不直接砍掉中介，而是把「思考」搬到動作空間
- 機器人領域的取徑，跟 XPENG 的問題定義不同（機器人任務空間更開放，動作維度更高）

XPENG 的路線可以視為「自動駕駛任務空間夠封閉，可以用更激進的端到端架構吃掉中介」的賭注。賭得對不對，要看 2027 全球部署後的安全數據。

---

## 對感知工程師的工程啟示

我自己做 LiDAR 演算法跟感知這塊，看 VLA 2.0 這個架構，最直接的感受是：

### 1. 感知模組會從「輸出可讀目標」往「輸出可學表徵」遷移

傳統感知模組的輸出是 bounding box、語義分割、追蹤 ID——這些是給下游 planner 用的「人類可讀中介」。VLA 2.0 這種架構不需要這層人類可讀中介，它要的是**對 action head 有用的 latent feature**。

LiDAR 點雲處理可能會出現新的 paradigm：BEV encoder 直接吐 latent token，跟視覺 token 一起進 implicit reasoning transformer。bbox / track 變成「次要產出」，純為 logging 跟人類驗證用。

### 2. Sensor fusion 的對齊邏輯會改變

傳統 sensor fusion 是在「目標物層」對齊：相機看到一台車、雷達看到一個目標、LiDAR 看到一團點雲，融合成「同一台車」。

如果中介層變成 implicit token，融合可能會在 token 級別發生——也就是 cross-attention over multi-modal tokens。這對 LiDAR 的點雲表徵設計提出新需求：要能跟視覺 token 共享 transformer 計算圖、要能在 token 級別上對齊時空。

### 3. 可解釋性會被推到「事後重建」

模型即時運作時不再產出文字推理，但量產車仍然需要可解釋性（事故調查、保險、監管）。可能會出現「事後 reasoning 重建」這個新工程任務：用一個離線、慢速的大模型把當時 latent 序列 + 感知輸入重新解碼成自然語言報告。

這個方向對 LiDAR 工程師意味著：點雲資料的儲存規格、時間對齊 metadata、跟 latent token 的索引關係，會變得跟感知本身一樣重要。

---

## 風險與懸念

VLA 2.0 不是一份完美方案。三個我會持續觀察的問題：

### 懸念一：long-tail 安全性

實測影片裡 VLA 2.0 在中國複雜路況表現亮眼，但「拿掉語言中介」這個決定在 long-tail 場景（極端天氣、罕見施工、道路異常標誌）下的表現尚未經過大規模驗證。語言層雖然慢，但它確實提供了一層「semantic sanity check」。

### 懸念二：跨地理泛化

72B 模型 + 中國車隊資料，這個組合在中國市場可能表現極好，但跟 Volkswagen 的全球合作要打入歐洲、北美時，能不能在資料分布外維持安全紅線？這也是為什麼大量產車計畫是「2027 全球部署」而不是「今年」。

### 懸念三：監管路徑

歐盟的 AI Act、美國 NHTSA、中國工信部對 L3 / L4 系統的審查趨勢都在朝「可解釋性」走。一個完全沒有人類可讀 reasoning trace 的系統，要拿到認證的工程難度可能比技術本身還高。XPENG 賭的是 X-World 模擬器 + 大量 rollout 證據能夠替代傳統的 CoT 解釋路徑。

---

## 結語

XPENG VLA 2.0 不一定是「正確答案」，但它是 2026 年自動駕駛 VLA 路線最強硬的一個技術表態：

> **「語言不是 VLA 的本質，它只是早期借來用的鷹架。當你有足夠的駕駛資料和模型規模時，這道鷹架應該拆掉。」**

對於同樣在做感知、控制、嵌入式系統的工程師——這不是一個可以等到「塵埃落定」再學的潮流。隱式 token 架構帶來的設計轉變，會反向影響感測器的輸出規格、融合的對齊邏輯、整個 stack 的可觀測性設計。

接下來半年值得追蹤三件事：

1. Volkswagen 在歐洲的實際部署數據
2. Alpamayo R1 跟 Mercedes-Benz CLA 上市後的對照表現（語言派 vs 隱式派的正面對決）
3. AAAI 2026 Spotlight 之後，FastDriveVLA 那套 token pruning 會不會被其他玩家複製

L4 量產車的最後一哩路，正在這場「語言要不要留」的爭論裡走完。

---

**參考資料：**

- [XPENG Unveils VLA 2.0: Latest Breakthrough in Autonomous Driving Architecture for 2026 — Blockchain.News](https://blockchain.news/ainews/xpeng-unveils-vla-2-0-latest-breakthrough-in-autonomous-driving-architecture-for-2026)
- [XPENG Releases World Model Technical Report, Powering VLA 2.0 Model R&D and Verification](https://www.xpeng.com/news/019dd72da86c9dd703de8a0282290002)
- [Xpeng spends $500M/year on AI training to beat Tesla FSD — Electrek](https://electrek.co/2026/06/05/xpeng-vla-interview-cvpr-language-poison-tesla-fsd/)
- [Xpeng VLA 2.0 test drive: Tesla is not alone with 'Full Self-Driving' anymore — Electrek](https://electrek.co/2026/04/29/xpeng-vla-2-test-drive-tesla-not-alone-full-self-driving/)
- [XPENG-Peking University Collaborative Research Accepted by AAAI 2026: FastDriveVLA — PRNewswire](https://www.prnewswire.com/news-releases/xpeng-peking-university-collaborative-research-accepted-by-aaai-2026-introducing-a-novel-visual-token-pruning-framework-for-autonomous-driving-302650038.html)
- [Breakdown of the "FastDriveVLA" — AI-Led L4 Autonomous Driving from XPENG & Peking University — CleanTechnica](https://cleantechnica.com/2025/12/29/breakdown-of-the-fastdrivevla-ai-led-l4-autonomous-driving-from-xpeng-peking-university/)
- [NVIDIA Announces Alpamayo Family of Open-Source AI Models — NVIDIA Newsroom](https://nvidianews.nvidia.com/news/alpamayo-autonomous-vehicle-development)
- [XPENG Accelerates Global Deployment of VLA 2.0 With Public Road Testing and 2027 Delivery Plan](https://www.xpeng.com/pressroom/news/019cae5e67b99c0960ee8a028129016a)
