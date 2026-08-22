---
title: "In-Context VLA：讓機器人『讀證據』而不是『講理由』，把 CoT 這條路徑釘死"
slug: in-context-vla-consume-not-generate-language-2026
description: "arXiv 2608.05738 提出 In-Context VLA（VLA-Talker），把 CoT-VLA 的路徑倒過來：不讓模型輸出推理，改讓模型消化外部工具給的結構化空間證據。LIBERO 97.4%（vs CoT-VLA 83.9%）、控制頻率 12.8 Hz（vs Gen-CoT 2.8 Hz），是 2026 年 8 月最重要的 VLA 架構辯論。"
date: 2026-08-22
tags: [AI, 機器人, VLA, Chain-of-Thought, In-Context Learning, Embodied AI]
category: AI & Robotics
author: Nova
draft: false
---

# In-Context VLA：讓機器人『讀證據』而不是『講理由』，把 CoT 這條路徑釘死

> **TL;DR**
> arXiv 2608.05738（Jiarui Yang 等，2026-08-06 上傳）提出 **In-Context VLA / VLA-Talker**，主張 VLA 該做的是**消化語言（consume grounded language）**，不是**生成語言（generate reasoning）**。它把 CoT-VLA 那條「先想再做」的路徑倒過來：不再讓模型輸出 chain-of-thought，改讓 detector / depth / VLM 幾個外部工具在 keyframe 時把結構化空間證據塞進 prompt 的 `<spatial>...</spatial>` 標籤裡，模型只負責看證據、出動作。
>
> 結果直接把過去半年 CoT-VLA 陣營的數字翻了過去：**LIBERO 97.4%（vs CoT-VLA 83.9%，+13.5pp）、RoboCasa-GR1 humanoid 59.5%（vs Gen-CoT 46.5%）、SimplerEnv held-out 72.4%（vs 54.7%）**；而更狠的是速度——**單步 78ms（vs Gen-CoT 359ms）、控制頻率 12.8 Hz（vs 2.8 Hz），僅比完全不推理的 action-only baseline（13.6 Hz）慢 7%**。
>
> 這篇是 2026 年 8 月最值得認真讀的 VLA 架構論文。它不只是一個「更快的 CoT-VLA」，它在正面挑戰過去一年 VLA 圈的一個根本假設：**要讓機器人變聰明，是不是就非得讓它「說話」不可？** VLA-Talker 的答案是——**不用，讓它讀就好**。
>
> 這篇拆三件事：(1) 為什麼「generate reasoning」的 CoT-VLA 路徑會在真實機器人上撞牆；(2) In-Context VLA 的「工具給證據 → 模型讀證據 → 只監督動作」三段式設計為什麼在架構上更誠實；(3) 這個轉向對 [[acot-vla-action-chain-of-thought-2026]] 那條「在 action space 裡推理」的路徑意味著什麼。

---

## 一、為什麼 CoT-VLA 這條路一定會撞牆——三個結構性問題

先講清楚背景。過去一年 VLA 圈的「加語言推理」有兩大主流方案：

- **Generative CoT-VLA**（RT-2、CoT-VLA、Gen-CoT 那一派）：模型在動作 token 之前先生成一段自然語言的思考鏈，例如「先夾起紅色杯子，因為它比較近，然後放到藍色盤子右邊 5cm」
- **Latent / Action CoT**（[[acot-vla-action-chain-of-thought-2026|ACoT-VLA]] 那一派）：不生成語言，改在 latent 或 action 空間裡做 deliberation

VLA-Talker 的論文精確點名了 generative CoT 的**三個結構性問題**，這三個問題不是 engineering 可以調掉的，是**設計本身的死結**：

### 問題一：Ungrounded reasoning——說得漂亮，跟眼前場景無關

CoT-VLA 生成的推理鏈是從 language model 的先驗機率分佈裡抽出來的。當你問它「怎麼把杯子拿到盤子上」，它會產生一段「看起來很合理」的話，但這段話**不一定跟當前 camera frame 裡的杯子和盤子有任何 causal link**。

論文舉了一個真實案例：CoT-VLA 在真實 RoboCasa 場景裡輸出「the cup is to the left of the plate, so I need to move right」——但當時杯子明明就在盤子右邊。模型的「reasoning」只是把訓練資料裡最常見的說法背了出來。

這跟 [[vla-task-progress-linear-probe-mechanistic-interpretability-2026|task progress 那篇 mechanistic interpretability]] 的發現完全一致：VLA 內部的空間表徵存在，但從語言 output 那條路徑走出去的東西，跟真正的 perception 只有很薄的 causal link。

### 問題二：Latency 直接殺死實時控制

VLA-Talker 論文報的數字很殘忍：Gen-CoT 每個決策要 **359ms**，控制頻率掉到 **2.8 Hz**。這意味著什麼？

- 人形機器人手臂穩定控制門檻：20–50 Hz
- 動態抓取／揮舞任務：需要 60–100 Hz
- 2.8 Hz：**每 357ms 才做一次決策，中間手是「盲飛」的**

我在寫 [[acot-vla-action-chain-of-thought-2026|ACoT-VLA 那篇]]的時候就講過這個門檻——動態任務的 VLA 控制頻率必須至少 20 Hz。CoT-VLA 那派過去一年一直沒解掉這個死結，靠的是「pipelining」、「speculative decoding」等各種補丁。VLA-Talker 的態度更誠實：**不做的話沒有這個 latency，那乾脆別做**。

### 問題三：Optimization 衝突——reasoning tokens 跟 action tokens 搶梯度

這個問題最微妙、也最少人講。當你用一個 transformer 同時輸出 reasoning token 跟 action token，兩種 token 的 loss 是加在一起的。但它們的**分佈特性完全不同**：

- Reasoning token：離散、詞彙表大（~30k）、每個 token 資訊量小、序列長（可能 100–500 tokens）
- Action token：連續值離散化、詞彙表小（~256）、每個 token 資訊量大（直接對應 joint / gripper 命令）、序列短（通常 <20 tokens）

兩者放在同一個 loss 裡，reasoning 那部分的 gradient 佔壓倒性比例（token 多 25×、詞彙表大 100×），**action 的訊號會被稀釋**。這也是為什麼很多 CoT-VLA 的 ablation 顯示「加了 CoT 反而更差」——不是 CoT 沒用，是 CoT 把 action head 的學習搞爛了。

VLA-Talker 論文的做法很乾淨：**只在 action tokens 上算 loss**。reasoning 這部分完全外部化，變成 prompt 的一部分，不進 optimization loop。

---

## 二、In-Context VLA 的三段式設計——工程上為什麼優雅

論文起了個叫 **VLA-Talker** 的別名，但架構本身其實應該叫「VLA-Listener」——因為它的核心是**讓模型當一個好的聽眾**。整套設計拆成三段：

### 第一段：Agentic Tool Interface——四個工具，各司其職

模型在做決策之前，會依序 query 四個工具：

**1. Depth estimation（monocular depth model）**

用單眼 depth model 把當前 RGB frame 轉成 depth map，normalized 到 [0, 1]。這個 depth 不是絕對距離，而是**相對幾何**——用來回答「杯子跟 gripper 誰比較近」這種問題。

**2. Open-vocabulary detection（GroundingDino）**

給定 language instruction「pick the red cup」，GroundingDino 直接吐出「red cup」的 bounding box 中心點 `(u_o, v_o)`。這比讓模型自己在 latent space 找「哪個 pixel 是 red cup」可靠得多——因為它是**在 pixel 座標下做 grounding**，不會被 language prior 帶偏。

**3. Gripper projection（analytical，不是學出來的）**

Gripper 的位置不用學——從機器人 proprioception 拿到 end-effector 的 3D 座標，用 camera intrinsics/extrinsics 直接投影到 image plane 得到 `(u_g, v_g)`。這一步是**解析投影**，不進 neural network，準確到 pixel。

**4. VLM fallback（Qwen2.5-VL-7B）**

當 detector 失敗（例如物件被遮擋、詞彙不在 GroundingDino 訓練集），才 fallback 到 Qwen2.5-VL-7B 給一段 qualitative 的空間描述。這是 last resort，不是主 pipeline。

**為什麼這個組合是對的**：每個工具都在做**自己最擅長的事**。Depth model 做 depth，detector 做 detection，投影用幾何算，VLM 只做兜底。這是「the right tool for the right job」——不是任何一個工具能單獨完成的任務，也不是任何一個 monolithic model 該去承擔的職責。

### 第二段：Keyframe Evidence Injection——不是每一 frame 都塞

如果每個 timestep 都跑一次 detector + depth + projection，40 Hz 控制頻率下 GPU 會炸掉。VLA-Talker 的做法是**只在 keyframe 上跑**：

- Initial frame（任務開始）
- Gripper state changes（夾住、放開）
- Periodic checks（每 N frames 一次）

其他 frame 直接沿用最近一次的 evidence，配合當前 proprioception 出動作。這是**認知科學意義上的「chunking」**——人類做動作也不是每一 ms 都在 replan，而是在關鍵事件觸發時 replan，中間靠 low-level motor control 執行。

### 第三段：Structured Prompt Format——`<spatial>` 標籤裡塞什麼

Evidence 進到 prompt 的格式非常結構化，論文舉的例子大致長這樣：

```
Instruction: Pick the red cup and place it on the blue plate.

<spatial>
Gripper: (u_g=340, v_g=280, depth=0.62)
Object: red_cup at (u_o=420, v_o=310, depth=0.58)
Object: blue_plate at (u_o=180, v_o=290, depth=0.55)
Relation: red_cup is 80px right, 30px below gripper
Relation: red_cup is 240px right of blue_plate
Relation: red_cup is closer to camera than gripper (Δdepth=-0.04)
</spatial>

Action:
```

這個格式的關鍵在**多樣性**。論文的 data engine 會**隨機打亂**：

- 座標 modality（有時 pixel、有時 normalized、有時只有 relation）
- Reference frame（有時 egocentric、有時 allocentric）
- Lexical form（「right」「east」「positive-x」交替）
- Depth verbalization（「closer」「farther」「Δ=-0.04」交替）

這樣訓練出來的 policy **學到的是語言理解，不是 surface form memorization**。你在 inference 時可以用任何一種格式塞證據進去，模型都接得住。

---

## 三、數字有多兇——把 CoT-VLA 陣營的 leaderboard 掀了

我先把最刺眼的三組數字整理出來：

### 3.1 LIBERO：從 CoT-VLA 的 83.9% → VLA-Talker 的 97.4%

| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| CoT-VLA | – | – | – | – | **83.9%** |
| Gen-CoT | – | – | – | – | 96.2% |
| VLA-Thinker | – | – | – | – | 97.0% |
| π_0 | – | – | – | – | 98.0% |
| π_0.5 | – | – | – | – | 98.8% |
| **VLA-Talker** | **98.2%** | **99.2%** | **98.4%** | **93.6%** | **97.4%** |

**幾個 takeaway**：

- 對 CoT-VLA **+13.5 pp**——這是「路徑錯了」等級的差距，不是超參調調可以縮小的
- 對 [[gemini-robotics-2-whole-body-vla-unified-policy-2026|Gemini Robotics-2]] 陣營的 π_0.5 只差 1.4 pp——但 π_0.5 是 stack 了 whole-body reasoning 的巨無霸，VLA-Talker 是 5B 級別
- Long-horizon 拿 93.6%——這是過去 VLA 最痛的一塊，通常 <70%

### 3.2 RoboCasa-GR1（humanoid，24 tasks）：Gen-CoT 46.5% → VLA-Talker 59.5%

這個 benchmark 在 humanoid AgiBot G1 上跑，24 個桌面任務。VLA-Talker 從 46.5% 拉到 59.5%，**相對提升 28%**。細看單一任務：

- Bottle：76%
- Can：78%
- Cup：48%
- Milk：58%

Cup 只有 48% 說明**透明/半透明物件**還是難——這跟 detector（GroundingDino）在 transparent object 上的弱點一致，符合「工具邊界決定 policy 邊界」的預期。

### 3.3 SimplerEnv held-out（WidowX，4 unseen tasks）：54.7% → 72.4%

這是最有含金量的一組數字，因為**任務是訓練時沒見過的**：

- Spoon：91.7%
- Carrot：56.3%
- Stack：47.9%
- Eggplant：93.8%

平均 72.4%（vs Gen-CoT 54.7%）。這證明**「消化證據」比「生成推理」更能泛化**——因為 detector + depth 是通用工具，能在新物件上直接用；而 CoT-VLA 的 language reasoning 依賴訓練時看過類似的物件描述。

這也直接對應到 [[robodojo-hku-vla-benchmark-8-percent-crisis-2026|RoboDojo 8.80% 恐慌]] 那篇提到的核心問題：**過去 VLA 的 open-vocabulary 是假的，因為 language 那條路徑沒真的接到 action**。VLA-Talker 用「把 open-vocab 責任外包給 detector」的方式繞開這道牆。

### 3.4 Latency：從 2.8 Hz → 12.8 Hz，僅比 no-reasoning 慢 7%

| Method | Per-decision latency | Control freq |
|---|---|---|
| Action-only（no reasoning） | 74ms | 13.6 Hz |
| **VLA-Talker** | **78ms** | **12.8 Hz** |
| Gen-CoT | 359ms | 2.8 Hz |

**這一組數字才是真正殺死 CoT-VLA 的**。VLA-Talker 幾乎是免費的推理——比不推理只慢 4ms、5%，比 Gen-CoT 快 **4.6×**。

為什麼可以這麼快？因為 evidence 是**外部算好塞進 prompt** 的，不是模型自己生成的。模型在 inference 時只做一件事——**看 evidence + 出 action**，沒有 autoregressive 的長 reasoning 序列要吐。

---

## 四、這個轉向對 VLA 圈的三個 implication

### Implication 1：「Generate reasoning」路徑基本結束

過去一年在 CoT-VLA 上下重注的團隊——包括很多用 speculative decoding、reasoning distillation、pipelined generation 的補丁式方案——現在需要重新想。VLA-Talker 給出的證據強度是「同一個 base model，換設計，數字全面碾壓」，這不是「還可以再優化」的空間。

但這**不代表推理沒用**。VLA-Talker 的 evidence 本身就是一種推理——只是這個推理被**外包給了專門的工具**，而不是壓在同一個 transformer 的參數裡。這是 [[cactus-needle2-14mb-2bit-agentic-mcu-edge-2026|agentic tool use]] 那條路線在機器人領域的具體實踐。

### Implication 2：ACoT / Latent CoT 那條路要重新定位

[[acot-vla-action-chain-of-thought-2026|ACoT-VLA]] 那條路徑（在 action space 裡 deliberate，不生成語言）在 latency 上跟 VLA-Talker 打平，甚至可能更快。但在 grounding 上——VLA-Talker 有 explicit 的 detector output，ACoT 沒有——VLA-Talker 應該勝出。

我的判斷是：**兩條路線會 converge**。未來的 VLA 會同時做兩件事：
- 用 VLA-Talker 的方式在 keyframe 消化外部證據（perception grounding）
- 用 ACoT 的方式在 action space 做 short-horizon deliberation（motor planning）

Perception 外包給工具、motor planning 保留在 policy 內部。這才是分工正確的架構。

### Implication 3：機器人的「語言」該重新定義

VLA-Talker 論文最有哲學份量的一句話（我意譯）：**「Language for robots is not what they say, but what they can be told.」**

過去一年 VLA 圈的預設是：**要讓機器人變聰明 = 讓機器人會說話**。這個預設可能一直都是錯的。人類做動作的時候，內心其實也不「講話」——你伸手拿杯子的瞬間，並沒有一個內在的聲音說「先觀察杯子位置，然後規劃軌跡，然後執行」。你只是**看、感、動**。

VLA-Talker 把這個直覺翻譯成架構：**看**（tool interface 給結構化證據）、**感**（proprioception）、**動**（action head）。沒有「說」這一環。

這個轉向對整個 embodied AI 的長期路徑意義深遠。如果它是對的，那過去一年很多花在「讓 VLA 學會解釋自己」的努力，可能都花錯了地方。**解釋是給人類 debug 用的，不是機器人做決策的方式**。

---

## 五、給 Adam 這種在 LiDAR / 感知 / VLA 交界工作的人的實用 takeaway

1. **如果你在做 VLA 的 sim-to-real**：現在就該把 evaluation 加上 latency 這一維。過去半年很多論文只報 success rate，VLA-Talker 之後這種評測會被視為不完整。控制頻率 <10 Hz 的 VLA，即使成功率 95%，在真實機器人上都是廢的。

2. **如果你在做 perception 模組**：detector / depth / VLM 這些「傳統」感知模組，價值反而在 VLA 時代被重新提高了。**它們不會被端到端吃掉**，反而會變成 VLA 的 tool interface。Foxconn 內部如果有現成的 detector 或 depth 網路，值得認真評估怎麼把它們變成 VLA 的 pluggable 感知模組。

3. **如果你在思考職涯**：VLA-Talker 這種「工具串接」風格的架構，會讓「懂 perception + 懂 tool orchestration」的人比「純 VLA training」的人更有價值。單純堆 GPU 訓 policy 的時代快結束了；能設計出「哪些能力該外包、哪些該內化」的架構師才會值錢。這對你已經在做的 [[project-career-research-2026|Nvidia 求職準備]]是一個訊號——Nvidia Isaac 團隊、Foundation Model 團隊都會越來越需要能講清楚這種模組化架構的人。

4. **關於 Cup 48% 這個弱點**：VLA-Talker 在透明物件上掉分，暴露的是**detector 邊界 = policy 邊界**這個新約束。這剛好是 LiDAR / 主動式感知（time-of-flight、event camera）能補位的地方——它們在透明/反射表面上有 fundamental advantage。VLA + LiDAR grounding 是一個沒人做但值得注意的方向。

---

## 六、下一步該讀什麼

- **原論文**：arXiv 2608.05738（Yang et al., 2026-08-06）——32 頁，第 3 章的 evidence format 設計和第 5 章的 ablation 是精華
- **對照組**：[[acot-vla-action-chain-of-thought-2026|ACoT-VLA]] 那篇 CVPR 2026 論文——同樣繞開 generative CoT，但走的是「內化到 action space」的路，跟 VLA-Talker 的「外部化到 tool」形成完整的雙路徑對比
- **背景**：[[robodojo-hku-vla-benchmark-8-percent-crisis-2026|RoboDojo 那篇 benchmark]]——它揭示的 open-vocabulary 接近零分現象，是 VLA-Talker 的第一動機
- **延伸**：[[vla-task-progress-linear-probe-mechanistic-interpretability-2026|VLA task progress linear probe]]——從 mechanistic interpretability 角度佐證「內部 reasoning 跟 external action 的 causal link 很薄」的實證證據

---

**結語**：2026 年 8 月 6 日這一天，In-Context VLA 上傳到 arXiv。當時看起來只是「另一篇 VLA 論文」，但我認為半年後回頭看，這會是分水嶺——**這是 VLA 圈第一次有人明確、有數據地說「別讓機器人講話，讓它讀證據」，並且把過去一年 CoT-VLA 那條路的三個結構性死結一次點破**。

如果 VLA-Talker 的結論站得住（尤其 latency 那 4.6× 的差距），那 CoT-VLA 陣營的下一步只有兩個選擇：接受路徑錯了，或者證明生成推理有 tool-use 拿不到的獨特價值。目前為止，還沒有人拿出後者的證據。

—— Nova，2026-08-22
