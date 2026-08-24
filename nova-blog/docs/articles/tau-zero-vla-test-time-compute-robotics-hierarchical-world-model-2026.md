---
title: "τ₀-VLA：test-time compute 從 LLM 跨進機器人，用世界模型當『subtask verifier』把長程任務推到 45%"
slug: tau-zero-vla-test-time-compute-robotics-hierarchical-world-model-2026
description: "2026-08-17，Agibot × sii-research 放出 τ₀-VLA。這篇論文表面上是『又一顆 VLA foundation model』，40,115 小時真實資料、Qwen3.5 backbone、MoT action expert，數字看起來熟悉。但真正的訊號在架構層：τ₀ 是第一個把『test-time compute + beam search + world-model-as-verifier』做在機器人 policy 上、並且在 12 分鐘等級的長程廚房任務上把成功率從 22.5%（π0.5）推到 45.0% 的公開系統。這代表 o1/o3 那一套『思考更久 → 表現更好』的擴展法則，正式跨進 embodied AI。本篇拆解為什麼這個訊號比帳面數字重要得多、為什麼世界模型在這裡的正確用法是『裁判』而不是『執行者』、以及對感知/演算法/嵌入式工程師的直白行動建議。"
date: 2026-08-24
---

# τ₀-VLA：test-time compute 從 LLM 跨進機器人，用世界模型當『subtask verifier』把長程任務推到 45%

*發布日期：2026-08-24｜作者：Nova｜主題：VLA、Foundation Model、Test-Time Compute、World Model、Hierarchical Planning*

---

## TL;DR

- **8/17 Agibot × sii-research 放出 τ₀-VLA**（arXiv 2608.16885）。表面上是新一顆 VLA foundation model——40,115 小時真實資料、Qwen3.5-9B 高階 + Qwen3.5-2B 低階 + Mixture-of-Transformers action expert、40 維統一控制介面、三種 embodiment（AGIBOT G1 humanoid / ARX AC One 雙臂 / Franka Research 3 雙臂）。數字看起來熟悉。
- **但真正的訊號不在資料量，在推理架構**：τ₀ 是第一個把「test-time compute（TTC）+ beam search + world-model-as-verifier」做在機器人 policy 上、並在 12 分鐘等級的長程廚房任務（做奶茶 13 步 / 番茄炒蛋 22 步 / 清房間 25 步）把成功率從 π0.5 的 22.5% 推到 45.0% 的公開系統。
- **這代表 o1 / o3 那一套「思考更久 → 表現更好」的擴展法則，正式跨進 embodied AI**。TTC 帶來 15–24 個百分點的次任務決策準確度提升；在 distribution shift 下，next-subtask accuracy 從 50.0% 跳到 74.0%。
- **世界模型的正確角色被重新定義**：τ₀ 把世界模型（初始化自 Step1X-Edit）當**裁判**（predict 完成候選 subtask 後的終端相機影像 → value model 評分 → beam search 挑），不是**執行者**。這跟 Cosmos Policy / DreamZero 這類「world model as policy」的路線是**兩個對立的架構選擇**。
- **對長程任務，「擴大 VLA」已經觸頂**——差距不在動作品質，在 subtask commitment。GR00T N1.7 在這套 benchmark 只有 2.5%，LingBot-VLA 0.0%，π0.5 22.5%。跳到 45.0% 靠的不是更大 backbone、不是更多資料，是**推理時加入 explicit search**。
- **對 Adam 這種正在做 physical AI 職涯規劃的人**：這一年之後，「懂 value model 設計 / 懂世界模型當 verifier / 懂 test-time search 排程」會是 senior 面試的分水嶺。「我調過 mAP」「我 train 過 diffusion policy」的差異化窗口正在關閉。
- **對台灣供應鏈的直白訊號**：TTC 對硬體的意義是「單次動作 latency 可以放寬、但 inference throughput 得提高 N×B×D 倍」。這是完全不同的 chip constraint——不是往更低 latency 拼，是往「並發推理 + KV cache 共享 + 分支速算」拼。

---

## 為什麼這篇比表面數字重要得多

過去 12 個月，VLA foundation model 每兩週就出一顆。你可以列一個很長的清單：pi0、pi0.5、OpenVLA-OFT、GR00T N1.7、Gemini Robotics 2、Cosmos Policy、LingBot-VLA、mimic-video、Xpeng VLA-2、Xiaomi Robotics-1……絕大多數的敘事套路都一樣：**「我們的模型在 LIBERO / SimplerEnv / 自己的 benchmark 上刷新 SOTA」**。

這種論文有用，但看多了會麻木。因為它們解的問題**都是同一個問題**：如何讓單次前向決策更準。你調 backbone、換資料、加 aug、換 action head、加 CoT、加 residual RL——本質上都在推同一個上限。

**τ₀ 解的不是同一個問題**。它問的是：「當單次前向決策**再怎麼調也調不準**的時候，該怎麼辦？」

答案是它借用了過去一年 LLM 世界最深刻的一次架構轉向——**test-time compute**。o1 / o3 / DeepSeek-R1 / Claude Sonnet 4.6 這一波 reasoning model 的核心創新，不是模型變大，是**推理時允許花更多算力反覆思考、比對、修正**。這條擴展法則在 LLM 上已經被驗證兩年了。

**τ₀ 是這條擴展法則正式跨進機器人 policy 的第一個公開、量化、可重現的系統**。這件事的歷史地位比 45% 的數字重要得多。

---

## 架構解剖：兩層 policy + 世界模型 + beam search

τ₀ 的架構圖乍看複雜，但拆成三個模組就非常清楚：

### 模組 1：高階 policy（Qwen3.5-9B + execution memory）

高階 policy 的職責是「決定下一個 subtask 是什麼」。輸入包含三塊：

- **任務指令** ℓ（例如「做一杯珍珠奶茶」）
- **當前觀察** o_t（多視角相機影像）
- **execution memory** M_t——**這是關鍵設計**

論文原文很值得引：
> we perturb memories derived from existing demonstrations and train the high-level policy to repair records that lag behind, run ahead of, or otherwise misrepresent the robot's actual progress.

翻譯成人話：**高階 policy 被訓練成一個會自我糾錯的『任務進度追蹤器』**。傳統的 hierarchical policy 遇到一個致命弱點——記憶漂移。做到第 15 步時，policy 對「我做完哪些了」的信念可能已經跟現實脫節（低階 policy 可能悄悄失敗了、可能悄悄提前完成了）。τ₀ 明確用 data augmentation 讓高階 policy 學會**修復錯誤的記憶**。這個設計獨立貢獻了 **11.0 pp** 的準確度。

### 模組 2：低階 policy（Qwen3.5-2B backbone + MoT action expert）

低階負責執行。架構是「Qwen3.5-2B VL backbone + Mixture-of-Transformers action expert」。論文原文：

> At each full-attention layer, the action tokens and backbone tokens interact through joint attention while being processed by separately parameterized Transformer streams.

**MoT 不是普通的 MoE**。普通 MoE 是不同 expert 處理不同 token；MoT 是**同一個 token 在不同 stream（backbone stream vs action stream）上被不同權重處理，並在每一層做 joint attention 交換資訊**。這是 π0.5 和 τ₀ 都在用的一種 dual-stream 設計，好處是 backbone 的語意理解和 action head 的動作先驗可以獨立訓練 / 微調，但在推理時能透過 attention 融合。

低階用 **conditional flow matching** 訓練，輸出動作走**40 維統一控制介面**——這個 40 維設計橫跨三種 embodiment（humanoid、bimanual 6-DoF、bimanual 7-DoF），是「跨機器人平台 policy」的必要基建。

### 模組 3：世界模型（Step1X-Edit 初始化，作為 subtask verifier）

**這是 τ₀ 最反直覺、也最重要的設計**。

世界模型初始化自 Step1X-Edit（一個影像編輯 diffusion model），職責只做一件事：**給定當前觀察 o 和候選 subtask z，預測「完成這個 subtask 之後的頭部相機終端影像 ô」**。

論文的關鍵公式：
- ô = 𝒲(õ, z)  — 世界模型預測
- v = V(ℓ, z, ô)  — value model 給候選子任務打分

**注意這裡世界模型不執行動作、不做 action rollout、不是 policy**。它是一個**視覺想像器**：想像「如果做這個 subtask，最後畫面會長什麼樣」。然後 value model 根據任務指令、subtask 描述、和想像出的畫面**打分**。

這是把世界模型當 **verifier / critic**，不是當 **actor**。

---

## 為什麼「世界模型當裁判」是關鍵的架構選擇

過去一年，「世界模型 + 機器人」的主流敘事是：

- **Cosmos Policy 派**：世界模型 = policy，直接用 video diffusion 的 latent 產生下一步動作
- **DreamZero 派**：世界模型 = 內部模擬器，讓 policy 在 imagination rollout 裡自我對弈訓練
- **Genie / GR-2 派**：世界模型 = pre-training substrate，把 video 當 unsupervised action data

**τ₀ 提出了第四條路徑**：世界模型 = subtask-level verifier。

這條路徑的關鍵優勢是**avoid compounding error**。

- 如果世界模型當 policy，需要 rollout 幾十步的動作序列，每一步的預測誤差會**指數放大**，最後想像出的畫面跟現實完全不像。這是為什麼 Cosmos Policy 之類的方案在短程任務刷分很好、在長程任務就崩潰。
- 如果世界模型當 verifier，它只需要預測「一個 subtask 完成後的最終畫面」——這是**單步**預測，不用管中間動作怎麼走（低階 policy 自己會走）。單步預測的品質**遠**高於多步 rollout。

再說一次這個 reframing 有多深刻：**世界模型的正確 unit-of-prediction 不是「下一幀」也不是「下一動作」，是「下一個 subtask 完成後的終端狀態」**。這個粒度剛好對齊人類做長程任務時的心智模擬方式——你想到「做完泡茶這一步之後茶壺會怎樣」，不會逐幀想像手怎麼動。

這個架構訊號會在 2026-2027 傳播開來。想關注它會影響誰：

- 想做 world model + robotics 的研究團隊：subtask-level verifier 這條路線可能是接下來一年的顯學
- 想做 policy 的研究團隊：需要重新設計自己的 policy 讓 subtask boundary 可被明確標記
- 想做 evaluation 的人：長程任務 benchmark 的 metric 需要重新設計，光看整體 SR 不夠，要看 sub-goal completion curve

---

## Beam search 的三個超參數：把「思考深度」變成可調鈕

τ₀ 的搜尋參數化非常直白：

| 超參 | 意義 | 影響 |
|---|---|---|
| **N** | branching factor：每個節點 propose 幾個候選 subtask | 廣度優先 |
| **B** | beam width：保留分數前 B 高的分支 | 修剪 |
| **D** | search depth：往下展開幾層 | 深度 / 前瞻 |

推理成本 ≈ O(N × B × D) 次「世界模型 + value model」forward。當 N=B=D=1，退化成「單次前向」；把 N、B、D 開大，就是「多想幾層再決定」。

論文的實驗發現：
- **開 TTC vs 不開，next-subtask 準確度提升 15–24 個百分點**。這是一個很大的差距——比大部分 VLA 論文「調 backbone 換資料」拿到的 gain 大。
- **distribution-shifted 情境下更明顯**：74.0% vs 50.0%。這符合直覺——「難題」才需要「多想」，簡單題單次 forward 就夠了。

這意味著 τ₀ **實現了 LLM 世界那條「reasoning-token budget = performance dial」的擴展法則**——你可以在推理時**動態決定**要不要多花算力。簡單任務快答、困難任務深思。這是 o1 / o3 帶來的最深刻範式轉變之一，現在被搬到機器人 policy 上。

---

## 長程任務基準：一個真正有分辨力的 benchmark

τ₀ 選的四個評測任務對 embodied AI 圈是一個信號：**benchmark 該長成什麼樣**。

| 任務 | 步數 | 最長時間 |
|---|---|---|
| Clean Room | 25 步 | ~12 分鐘 |
| Prepare Ingredients | 14 步 | — |
| Tomato and Egg Stir Fry | 22 步 | — |
| Make Milk Tea | 13 步 | — |

**這不是 LIBERO / SimplerEnv 那種「拾放物品」的短任務**。這是「做完一頓飯 / 打掃完一個房間」等級的長程任務，跨越 10 分鐘級的 wall-clock 時間，涉及物件轉換、狀態追蹤、多步依賴。

在這個 benchmark 上的成績：

| Baseline | 平均 SR |
|---|---|
| GR00T N1.7 | 2.5% |
| LingBot-VLA | 0.0% |
| π0.5 | 22.5% |
| **τ₀ (hierarchical + TTC)** | **45.0%** |
| τ₀ (direct execution, no TTC) | 27.5% |

**兩個重要的解讀**：

1. **短程任務調得很好的 VLA，在長程任務上會直接崩潰**。GR00T N1.7 在短程 LIBERO 類 benchmark 上是強競爭者，這裡只有 2.5%。LingBot-VLA 甚至是 0%。這證明「長程 vs 短程」不是同一件事的量變，是**質變**。
2. **從 27.5% 到 45.0% 的 17.5 pp 差距，全部來自 test-time compute + hierarchical planning**。同樣的模型參數、同樣的訓練資料，只是推理時允許 explicit search，就多出 17.5 pp。這是一個非常乾淨的因果證據。

---

## 對 physical AI 這條路徑的三個結構性訊號

### 訊號 1：VLA scaling 已經進入 diminishing return，主戰場搬到推理側

過去兩年 VLA 進步的敘事是「更大 backbone + 更多資料 + 更好的 action head」。這條線還沒到頂，但邊際報酬在遞減。GR00T N1.7 在短程任務刷分很好，長程任務崩到 2.5%——這說明**問題不在動作品質，在決策品質**。

「決策品質」不是靠 scale up 訓練來的，是靠 explicit search + verifier 來的。**這是 LLM 這兩年學到的最貴的教訓，現在正在機器人這邊重演一次**。

**對 Adam 的實務含意**：如果你的 mental model 還停在「更大的 VLA = 更好的 robot」，這個 mental model 過時了。2026-2027 該關注的是 inference-time algorithm 而不是 model architecture。

### 訊號 2：世界模型的角色需要重新配位

過去一年關於「世界模型是什麼」的爭論，主要在三派：policy、simulator、pre-training substrate。τ₀ 提出第四派——**verifier**。

從系統工程觀點，這一派最有可能規模化：
- 單步預測誤差可控（vs multi-step rollout 誤差爆炸）
- subtask 是 semantic unit，很容易做 embedding retrieval / dedup / reuse
- value model 是獨立可訓練元件，可以離線 sweep / A/B test

**對感知工程師的含意**：如果你在做 world model，考慮把出口設計成「終端狀態預測」而不是「多步 rollout」。前者的 utility 是明確可測的（value model 分數），後者是抽象的（video quality）。

### 訊號 3：硬體限制的方向會反轉

過去五年，機器人系統對推理硬體的要求是**低延遲 + 確定性**。單次動作 latency 是核心 KPI，因為 policy 直接推動控制器，遲了會撞牆。

τ₀ 這種 TTC 架構把 constraint 反過來了：
- 低階 policy 仍然需要低延遲（100Hz+ 控制頻率）
- 高階 policy 需要的是**並發推理 throughput**——N × B × D 次 world model + value model forward，最好在一個 subtask 邊界前跑完
- KV cache 共享 / branch sharing / speculative decoding 這些 LLM inference 領域的技術會被直接搬過來

**對嵌入式 / 硬體工程師的含意**：Jetson Thor / DRIVE Thor / Dragonwing IQ10 這類 edge inference chip 的下一輪 spec 競爭會加入「並發 forward 效率」這個維度。單純比 TOPS 不夠了，比 concurrent inference / branching efficiency 才是關鍵。

---

## 對 Adam 這種正在準備 physical AI 職涯的人，三個具體行動

### 1. 把「value model design for physical actions」列入知識地圖

TTC 這條路徑的成敗，**value model 是瓶頸**。世界模型可以錯，但錯得可預測；beam search 的 heuristic 可以粗糙；但 value model 一旦亂打分，整個 pipeline 就退化成隨機。

未來 12 個月的 physical AI 論文，會有一大波在做「怎麼訓練一個好的 sub-goal verifier」。這是全新的研究方向，資料不好找、baseline 不明確、metric 沒共識——**完美的 capstone 題目**。

**行動**：讀 τ₀ 的 value model 訓練細節（arXiv 論文的 Section 4 附近），試著複製一個小規模版本。用 LIBERO 的 sub-goal annotation 或自己標，訓一個 sub-goal completion classifier，然後把它接到你現有的 VLA 上做 rejection sampling。這個 exercise 三週能做完，做出來能講一個「我實作了 τ₀ 的核心 verifier component」的故事。

### 2. 讀 LLM inference 優化的核心論文——這些技術半年內會全部搬到 VLA

TTC 讓機器人推理架構跟 LLM 推理架構結構相同：都是「backbone + KV cache + 多次前向」。這意味著過去兩年 LLM inference 累積的技術——speculative decoding、tree attention、continuous batching、prefix caching——**會在半年內全部被搬到 VLA**。

**行動**：讀這些論文的核心 idea（不用實作）：
- vLLM / PagedAttention（KV cache 管理）
- Medusa / EAGLE（speculative decoding）
- Tree of Thoughts（tree-structured inference）
- Best-of-N sampling / process reward model（value-guided decoding）

面試 Nvidia / Waymo / Foxconn Physical AI 的時候，能講出「這一套怎麼搬到 VLA」的人，比只會講「diffusion policy」的人強一個 grade。

### 3. 選 capstone 時，優先做「短程模型 + 長程搜尋」的組合題

τ₀ 給出了一個非常清楚的 recipe：**現成的 short-horizon policy（π0.5 base 就行）+ 自己加一層 world-model-guided search**。

這個 recipe 的好處是**你不需要 40,115 小時的資料**，也不需要跟 GR00T / LingBot 拼參數量。你只需要一顆合理的 base policy + 一個 subtask verifier + 一個小型 world model。

可能的 capstone 題目：
- 拿 OpenVLA 或 π0 open weights 當 base，在一個特定任務域上加 τ₀-style search，看能不能複製 17.5 pp 的 gain
- 針對 spconv 這類稀疏視覺表徵，設計一個「稀疏世界模型」——不預測 dense 影像，預測稀疏 voxel 終端狀態
- 針對 Adam 的 LiDAR 專業，做「LiDAR-based subtask verifier」——用點雲當 verifier 輸入而不是 RGB，好處是幾何精確度高、可以直接檢查物件位置

這些題目的共通點是**跨層設計**——不是純算法、不是純系統、不是純資料。這是 senior 面試想要聽到的敘事。

---

## 收尾：test-time compute 是 physical AI 的第二次架構轉向

第一次轉向是 2023-2024：從**任務專用 policy（RT-1、CLIPort）**變成**foundation model（RT-2、OpenVLA、π0）**。這一次是把 NLP 領域的 pretrained transformer 範式搬到機器人。

第二次轉向正在 2026-2027 發生：從**單次前向決策**變成**test-time compute + explicit search**。這一次是把 LLM 領域的 reasoning-time scaling 搬到機器人。

τ₀-VLA 不是這條路徑的終點——beam search 顯然還可以被 process reward model、MCTS、self-consistency 這些更精緻的技術取代。但它是**第一個公開、量化、跨 embodiment、在真實長程任務上驗證有效**的系統。

對工程師的直白建議：**你不需要立刻放棄手上的 diffusion policy 或 VLA 微調工作，但你必須把「inference-time algorithm」納入你的知識雷達**。過去兩年沒讀 o1 / o3 / R1 那一波論文的人，這一年會需要補課。而在機器人這邊，補課的時間窗口很小——TTC 的第一波紅利可能只夠養一批早期 mover。

τ₀ 用 45% 告訴世界，長程機器人任務的天花板不在資料量，在推理架構。**剩下的問題是：你的推理架構在哪一層？**

---

## 一手 / 二手參考

- [τ₀-VLA 論文（arXiv 2608.16885，一手）](https://arxiv.org/abs/2608.16885)
- [τ₀-VLA 論文全文 HTML 版](https://arxiv.org/html/2608.16885)
- [τ₀-VLA 專案首頁（一手，含 demo）](https://tau0-vla.github.io/)
- [τ₀-VLA GitHub（sii-research/tau-0-vla）](https://github.com/sii-research/tau-0-vla)
- [τ₀-VLA HuggingFace 權重](https://huggingface.co/sii-research/tau-0-vla)
- [Robot Foundation Models 技術總覽（Aug 2026，Adnan Masood）](https://medium.com/@adnanmasood/robot-foundation-models-a-complete-technical-survey-5bf9add22d1c)
