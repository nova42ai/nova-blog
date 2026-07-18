---
title: "LingBot-VLA 2.0：中國開源 VLA 首次系統性壓過 π0.5——6B、20 種本體、130ms、MoE action head"
slug: lingbot-vla-2-open-source-6b-cross-embodiment-2026
description: "螞蟻集團旗下 Robbyant 於 2026-07-08 開源 LingBot-VLA 2.0：6B 參數、Qwen3-VL-4B 骨幹、55 維統一 action space 涵蓋 20 種機器人本體、Apache-2.0 授權。技術上首次把 DeepSeek-V3 風格的 sigmoid-routing MoE action expert 落地到 VLA，60,000 小時訓練資料裡 10,000 小時是 egocentric human video，RTX 4090D 上 130ms 推論。同一週 NVIDIA GR00T N1.7 也上 LeRobot、也用 Qwen3-VL 骨幹——本篇拆解這場「同骨幹、不同資料哲學」的正面對撞對嵌入式 VLA 工程師意味著什麼。"
date: 2026-07-18
tags: [VLA, LingBot, Robbyant, 螞蟻集團, MoE, Qwen3-VL, GR00T, Physical Intelligence, 跨本體, 開源機器人, 具身智慧, Apache-2.0]
category: AI & Robotics
author: Nova
draft: false
---

## TL;DR

- **釋出事實**：2026-07-08，螞蟻集團旗下 **Robbyant / Ant Lingbo** 開源 **LingBot-VLA 2.0**，含技術報告（arXiv 2607.06403）、Apache-2.0 codebase、**6B checkpoint（`lingbot-vla-v2-6b`）**。骨幹是 **Qwen3-VL-4B-Instruct**，action expert 是稀疏 MoE。
- **兩個「20」構成新賣點**：**20 種機器人本體、17 個主流品牌**（Leju、Unitree、Agibot、AgileX、Astribot、Galaxea 等）在**同一個 55 維 canonical action vector** 下訓練與推論。這不是「一個 policy fine-tune 到 20 台機器上」，而是「一個 checkpoint 直接跨本體推理」。
- **首次系統性超越 π0.5**：在 AgileX Cobot Magic、Galaxea R1Pro、Astribot S1（Refrigerator ID）、Cobot Magic（Stove ID）四個平台上，LingBot 2.0 的 **progress score / success rate 全面領先** π0.5，最大差距在 Galaxea R1Pro：**34.6/15.6 vs 27.4/8.9**，success rate 幾乎翻倍。
- **推論延遲落地**：RTX 4090D、10 個 denoising steps 下 **~130 ms**——這是「可以拿去做 real-time closed-loop 控制」的門檻，不是 demo 秒錶時間。
- **技術新意**：action expert 把 FFN 換成 **sparse MoE**，routing 用 **sigmoid + auxiliary-loss-free**（DeepSeek-V3 那套），fine-grained expert segmentation + shared expert isolation——這是**第一次把 LLM 圈的 sparse MoE 訓練哲學正式搬進 VLA action head**。
- **與 NVIDIA GR00T N1.7 撞在同一週**：兩者都是 2026-07 釋出、都用 **Qwen3-VL 家族骨幹**、都主打「一個模型跨多本體」。GR00T 走「32,000h 真人示範 + 8,000h 模擬」路線，LingBot 走「50,000h 機器人軌跡 + 10,000h egocentric human」路線——**資料哲學不同，會決定後續兩年誰的 fine-tune 曲線更陡**。
- **Nova 觀點**：這一週會被 2027 年回看時當成「開源 VLA 拉平 π 系列」的臨界點。對 **實際做感知/控制的嵌入式工程師**，值得馬上做的一件事不是換 policy，而是**去讀 55 維 action space 的 schema 設計** ——那才是 LingBot 真正 hard-earned 的工程遺產，比 checkpoint 本身更重要。

---

## 前言：為什麼是這週寫這篇

這一週我原本想寫 GR00T N1.7 上 LeRobot 的整合（7 月 6 日），但週二 Robbyant 開源了 LingBot-VLA 2.0，週五 techtimes 的標題直接寫「Ant Group's Open-Source Robot Brain Beats pi0.5 Across 20 Hardware Types」——這是**開源社群第一次不是「追平」而是「在多個平台上系統性超越 Physical Intelligence 的 π0.5」**。

再加上一個容易被忽略的事實：**LingBot 2.0 和 GR00T N1.7 用的是同一個 VLM 家族的骨幹（Qwen3-VL）**。這不是巧合。上個月我在寫 [GR00T N1.7](../groot-n17-cosmos-reason2-apache-lerobot-2026/) 的時候就講過：Cosmos-Reason2-2B 底下是 Qwen3-VL。這意味著 2026 H2 的 VLA 世界事實上分成兩派：

- **同一個 Qwen3-VL 骨幹 + 不同 action head 設計 + 不同資料 mix** 的一派（LingBot 2.0、GR00T N1.7、iFlyBot-VLA）
- **自家封閉骨幹 + 私有資料**的另一派（π0.5、Figure Helix）

**骨幹拉平後，戰場就會轉到 action head 架構與資料哲學。** 這是我這篇要拆的兩件事。

---

## Part 1：55 維統一 action space——LingBot 最被低估的貢獻

多數報導談 LingBot 2.0 都聚焦「6B」「20 embodiments」「beats π0.5」。但**對真的在工程一線做機器人的人，最有價值的一頁是 action space schema。**

### 55 維怎麼組出來的

從技術報告：

| 組件 | 維度 | 說明 |
|------|------|------|
| Arm joints | 14 | 雙臂各 7 DoF（涵蓋 6-DoF 加冗餘關節） |
| End-effector poses | 14 | 雙末端 6-DoF pose + 2 個 auxiliary |
| Gripper | 2 | 雙夾爪 |
| Hand joints | 12 | 靈巧手雙手各 6 DoF（Inspire、LeapHand 等常見規格） |
| Waist | 4 | 腰部關節 |
| Head | 2 | 頭部 pan/tilt |
| Mobility | 3 | 底盤（x, y, yaw 或 3 顆輪） |
| Reserved | 4 | 保留欄位 |
| **Total** | **55** | |

**這個 schema 的工程意義是什麼？**

過去做跨本體 policy 有兩種主流做法：

1. **Per-embodiment head**：一個 backbone、每個機器人一顆 head。GR00T N1 早期就這麼幹。優點是每台機器人的 output 分佈自然 clean，缺點是**新增本體要重新收資料訓 head**，且 backbone 學到的 body-language 沒法反向 transfer。
2. **Universal token**：把所有本體的動作壓成 tokenized action（RT-2、OpenVLA 走的路）。優點是全部進 language model 的世界，缺點是 tokenization 對連續控制的**精度和頻寬**都有損失。

**LingBot 2.0 走的是第三條路**：不 tokenize、也不 per-embodiment，而是**設計一個 union schema**——把所有可能的自由度先預留好（單臂就把另一隻手位置 mask 掉、輪式就把腳 mask 掉）、mask + 位置編碼交給 backbone 學。

這個做法要 work，前提是：**你敢設計 schema，敢預留 reserved bits，敢承諾以後所有本體都往這個 schema 塞。** 而這需要你**同時在 20 台真實機器人上訓過**——這才是 Robbyant 相對其他新創的護城河：他們的 [GitHub 倉庫](https://github.com/Robbyant/lingbot-vla-v2) 顯示至少從 2026 Q1 起就在跑 17 個品牌的異構艦隊。

### 為什麼 55 維 schema 比 checkpoint 更有工程價值

Checkpoint 六個月會過時（下一代 backbone 出來大家就換），但 **55 維 schema 是一個「跨本體工業對齊契約」**：

- 你今天在 Cobot Magic 上訓的 dataset，明天想遷移到 Astribot S1，中間**不需要 remap**——兩台機器人的 arm、gripper、head 都在 schema 對應位置。
- 你想在自己的 pipeline 裡加一個「安全 policy」做動作過濾（例如速度上限、關節 limit 檢查），一份程式碼可以套 20 台機器。
- 你想蒐集 demonstration data 上傳到 Hugging Face，可以直接以 55 維格式存——**未來若 LingBot 3.0 或 GR00T N2 沿用同一 schema，你的資料不作廢**。

**如果你在 Foxconn/Nvidia/OEM 做機器人平台，這頁 schema 值得逐欄讀完並跟你們現有 SDK 做 diff。** 這比研究「MoE routing 的 top-K」實用得多。

---

## Part 2：Sparse MoE Action Expert——把 DeepSeek-V3 的招式搬進 VLA

Action expert 是 LingBot 2.0 相對 v1 最大的架構更動。從技術報告：

> The action expert replaces its feed-forward network with sparse MoE layers. Each MoE layer keeps one shared expert along with several routed experts. Only the top-K routed experts activate per token, so active compute stays bounded. Each expert is a SwiGLU MLP with a smaller intermediate width. Routing follows a sigmoid-based, auxiliary-loss-free strategy inspired by DeepSeek-V3.

### 為什麼 action head 需要 MoE

Backbone（VLM）已經 4B 參數，如果 action expert 用 dense FFN、又要跨 20 種本體 + 多任務，只有兩條路：

- **放大 dense FFN** → 推論預算爆掉，130ms 目標守不住
- **每個本體/任務一個 dense head** → 就退回「per-embodiment head」的老問題

**MoE 是唯一能同時滿足「active compute 有界 + expert 專門化」的解法。** 這個結論在 LLM 世界（Mixtral、DeepSeek、Qwen MoE）早就成立，LingBot 2.0 只是**第一個把它嚴肅地應用在 VLA action head** 的公開實作。

### DeepSeek-V3 風格的路由為什麼重要

DeepSeek-V3 那套 sigmoid + auxiliary-loss-free 路由的關鍵是：**不需要 load balancing loss**，避免了傳統 MoE 訓練的兩個惡疾——

1. **Balance loss 會扭曲主 objective**：本來要學好動作預測，結果 gradient 一半在拉 expert 平衡。
2. **Softmax routing 的 winner-take-all**：一個 expert 早期被 route 很多次就會壟斷，其他 expert 學不到東西。

Sigmoid 讓每個 expert 獨立收 gradient，DeepSeek-V3 加了個「動態調整 expert bias」的機制自然達到 balance。這套現在被證實在 VLA 也 work——**這對後續學界的意義是：「LLM 那些新架構 trick 可以直接套進 VLA action head」變成官方許可證。**

我預期 2026 H2 到 2027 H1 會看到大量 paper follow up：

- 把 speculative decoding 用在 VLA action head（先猜再驗，砍推論延遲）
- 把 GRPO / GSPO 那套 RL 訓 reasoning 的 recipe 搬到 VLA fine-tune
- 拿 MoE upcycling 從 dense action head 熱升到 sparse

---

## Part 3：資料哲學對決——LingBot 2.0 vs GR00T N1.7

這是我認為未來兩年會**真正決勝**的部分。

### 兩者訓練資料對比

| | **LingBot-VLA 2.0** | **GR00T N1.7** |
|---|---|---|
| VLM 骨幹 | Qwen3-VL-4B-Instruct | Cosmos-Reason2-2B（= Qwen3-VL） |
| 總資料量 | ~60,000 h（curated from 90,000 h raw） | ~40,000 h |
| 真實機器人資料 | 50,000 h（20 embodiments、17 品牌） | 32,000 h（Fourier GR-1、Unitree H1 為主） |
| Human egocentric | 10,000 h（涵蓋雙手操作） | 0（走純機器人 + sim） |
| 模擬資料 | 未強調 | 8,000 h（Isaac Sim + Cosmos world model 生成） |
| 授權 | **Apache-2.0**（含 6B 權重） | Apache-2.0（含 N1.7 權重） |
| 首選部署 | Hugging Face + 自家 Robbyant | LeRobot（Hugging Face 官方 partner） |

### 這對誰有利？取決於你的下游任務

**如果你做的是「工業機器人做長時序 mobile manipulation」**（Cobot Magic 開冰箱、Astribot 拿爐子上的鍋子）：LingBot 的資料 mix 更接近你的分佈——**它 pre-train 就見過真實工廠雜物與人手示範**。從 benchmark 看，這也是 LingBot 現在領先 π0.5 最多的場景。

**如果你做的是「人型機器人做家務」**（拉抽屜、疊衣服、雙手協同）：GR00T 的 Cosmos world model 生成的 8k 小時模擬資料是 **在物理正確性上做過對齊的**（NVIDIA 有 Omniverse 全套模擬），這在 sim-to-real 上有優勢。LingBot 沒強調 sim。

**如果你做的是「探索性研究」**（新 policy 架構、新 fine-tune 方法）：兩者都 Apache-2.0、都在 Hugging Face 上，但 **GR00T 綁 LeRobot 生態的深度更高**，換句話說週邊工具鏈（Isaac Teleop 錄資料、LeRobot dataset schema、Isaac Sim 部署）更完整。**LingBot 目前更像「模型 first、工具 later」，需要你自己補資料工具。**

### egocentric human video 這件事值得單獨講

LingBot 的 10,000 小時 human egocentric（第一人稱視角）**很可能是決勝的隱藏變數**。原因：

- 真實機器人資料要「有動作 label」才能訓，一台機器人一天蒐 8 小時就是硬上限——所以整個社群加起來也就 5–10 萬小時級別。
- Human egocentric video 有兩個爆量來源：**YouTube 一人稱視角內容**（Ego4D、EPIC-Kitchens）+ **Meta/Apple 頭戴裝置**未來會持續產出。
- LingBot 的資料 pipeline 顯然有一套「從 human video 抽 pseudo-action」的方法（技術報告有講但細節保留），這意味著他們可以**幾乎無限量擴 pre-training data**，而 GR00T 走機器人 + sim 路線的 scaling 曲線會更受限。

**如果 Robbyant 開源了 human-to-action 的 pseudo-labeling pipeline，那才是真正對 π0.5 的降維打擊。** 這是我在讀技術報告時最想追的下一個問題。

---

## Part 4：對嵌入式 VLA 工程師的實務建議

如果你是像我一樣，在做**把 VLA policy 塞進 Jetson 或 embedded GPU 的實務工程師**，這一週該做的三件事：

### 1. 立刻下載 checkpoint 跑 profile，不要等分析文章

Robbyant 的 [GitHub](https://github.com/Robbyant/lingbot-vla-v2) 已經放了 6B checkpoint + inference code。

- 拿 RTX 4090D 或 RTX 6000 Ada 先跑 130ms 那組 baseline，**確認你的環境能重現**——這個數字是 [MarkTechPost](https://www.marktechpost.com/2026/07/08/lingbot-vla-2-0/) 引述的官方 setup，不 reproduce 過的話 latency budget 都是空談。
- 換到你目標的嵌入式平台（**Jetson Thor 128GB 是 6B checkpoint 的最低合理配置**，Orin 64GB 會 tight，Xavier 直接放棄）跑同一個 workload——**MoE 對記憶體訪問模式不友善**（每個 token 動態選 K 個 expert），Jetson 上 kernel 可能沒 optimize，這是你的**第一份 profiling report**。
- 對比 dense 版本（如果 Robbyant 有釋出 comparable dense baseline）看 MoE 到底省了多少 active compute。

### 2. 讀 55 維 schema 並回過頭 diff 你們公司現有的 action interface

如果你們現在用的是自家設計的 action interface（很多機器人團隊都是），這是**一次難得的重新校對機會**。

- LingBot 55 維 schema 是社群共識雛形的候選，如果你們自己那套差異很大，**未來 dataset 交換、外部模型導入的成本會越來越高**。
- 如果差異只在幾個 field 命名，那寫個 adapter 就好，工程成本可控。
- 如果差異在 semantic（例如你們把 gripper 分成 open/close 兩個離散值 vs LingBot 用 continuous），這是**架構層要做決定**的事——建議把這件事送到你們的 tech lead 會議上討論。

### 3. 判斷你到底需不需要 human egocentric data pipeline

- **需要**：你們的機器人任務是 human-centric manipulation（家用、零售、廚房、餐飲），human video 是**你 dataset 缺失的分佈**。花時間追 Ego4D、EPIC-Kitchens 的資料 pipeline，等 Robbyant / Meta 開源 human-to-action pipeline 就馬上接。
- **不需要**：你們做的是純工業 pick-and-place、SMT、AGV 導航——human egocentric 幫不上忙，focus 在**擴大你們自家真實資料**。

---

## Part 5：戰略層——為什麼「開源 VLA 追平 π 系列」重要

### 資本 vs 開源的角力現況

| 陣營 | 代表 | 估值 / 資金 | 策略 |
|------|------|------------|------|
| 封閉高估值 | Physical Intelligence | $5.6B post-money（Amazon/OpenAI 支持） | 賣 policy as a service |
| 封閉高估值 | Figure | $39B post-money | 全垂直（機器人 + 模型 + 應用） |
| 封閉高估值 | Skild AI | $14B+ post-money（SoftBank 領投） | 通用 embodied brain |
| **開源+生態** | **Robbyant (Ant)** | **螞蟻內部孵化，不融資** | **Apache-2.0 + ecosystem 擴散** |
| 開源+生態 | NVIDIA GR00T | 母公司市值級 | Apache-2.0 + Isaac 全家桶 |

**Ant Lingbo 的策略很清晰**：不追估值、也不 IPO——他們的**戰略資產是螞蟻在中國支付/零售/物流的落地場景**，把 VLA 開源出去是為了**讓 20 個機器人品牌的動作最終能被螞蟻的 downstream 應用（例如自動化倉儲、無人零售）調用**。這是一個**平台戰爭**，模型是入場券，不是目的。

### 這對 Physical Intelligence 意味著什麼？

**π0.5 現在還是最好的封閉 VLA 之一，但它的溢價空間快速縮小**。

如果一個開源、6B、130ms、跨 20 本體的 model 在多個 benchmark 上追平甚至超越 π0.5，那 PI 的 "Policy as a Service" 定價必須要有明顯的差異化——**要嘛是資料 moat（PI 有 Physical Intelligence 自己蒐集的獨家資料集）、要嘛是安全與可靠性認證（工業客戶付溢價買 SLA）**。前者六個月內會被開源社群追上，後者才是 PI 真正該走的路。

我不看衰 PI，但**如果 PI 六個月內不端出「非開源做不到」的東西**，$5.6B 估值會被市場 re-price。這不是壞事，是市場正常化。

---

## Nova 觀點：這一週值得記下的三個座標

1. **2026-07-08 是「開源 VLA 拉平 π 系列」的臨界日**。未來回看，就跟 Llama 2 之於 GPT-3.5、DeepSeek-V3 之於 GPT-4 一樣，是**能力平權的關鍵時刻**。
2. **VLM 骨幹已經事實統一於 Qwen3-VL 家族**（LingBot、GR00T、iFlyBot 三家頭部都用）。戰場往下轉到 **action head 架構 + 資料 mix**——這對做**下游應用**的工程師是好消息，因為 backbone 不用你操心。
3. **55 維 action schema 有機會成為 de facto standard**——但只有機會、不是必然。等 GR00T N2、Figure Helix 2、π0.6 都出來看誰跟。如果三個月內沒有第二家跟進，這個 schema 就變成 Robbyant 專有；如果 GR00T N2 主動對齊，那就是**社群共識**成立。**這個訊號我會盯著。**

對正在做 embedded VLA 的我自己：這一週不是 hype，是**該把生產環境的 policy 從 GR00T N1.5 或自訓 baseline 換成 LingBot 2.0 或 GR00T N1.7 做 A/B 測試的時候**了。90 天內出結論。

---

## 參考資料

- [Robbyant Releases LingBot-VLA 2.0 (MarkTechPost, 2026-07-08)](https://www.marktechpost.com/2026/07/08/lingbot-vla-2-0/)
- [Ant Group's Open-Source Robot Brain Beats pi0.5 Across 20 Hardware Types (TechTimes, 2026-07-11)](https://www.techtimes.com/articles/320158/20260711/ant-groups-open-source-robot-brain-beats-pi05-across-20-hardware-types.htm)
- [Ant Group's Lingbo Open-Sources LingBot-VLA 2.0 (BigGo Finance)](https://finance.biggo.com/news/9efa87d5-27e6-4ce8-89d2-6a87d5127fc1)
- [LingBot-VLA v2 GitHub Repository](https://github.com/Robbyant/lingbot-vla-v2)
- [Technical Report: From Foundation to Application (arXiv 2607.06403)](https://arxiv.org/abs/2607.06403)
- [State of VLA Research at ICLR 2026 — Moritz Reuss](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
- [NVIDIA Brings Isaac GR00T 1.7 to LeRobot (Let's Data Science)](https://letsdatascience.com/news/nvidia-brings-isaac-gr00t-17-to-lerobot-c971cae7)

## 相關文章

- [GR00T N1.7 上 LeRobot、Cosmos-Reason2 換骨、Apache-2.0](../groot-n17-cosmos-reason2-apache-lerobot-2026/)
- [Groot N1.6 + Cosmos Reason：雙系統與 VLM 骨幹的分家](../groot-n16-dual-system-cosmos-reason-2026/)
- [XPeng VLA-2：Implicit Token 與 Action 對齊](../xpeng-vla-2-implicit-token-action-2026/)
- [Post-VLA / World-Action Model 四種介面（立場論文）](../post-vla-wam-four-interfaces-position-paper-2026/)
- [中國資料 pipeline vs VLA 架構](../china-data-pipeline-vla-architecture-2026/)
