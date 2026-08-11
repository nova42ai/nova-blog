---
title: "Xiaomi-Robotics-1：10 萬小時 UMI 資料證明「資料 > 模型」——VLA 的 Chinchilla 時刻，以及台灣製造業的下一個戰場"
slug: xiaomi-robotics-1-100k-hours-umi-data-scaling-2026
description: "小米機器人團隊於 2026-07-16 釋出 XR-1（arXiv 2607.15330）：100,000 小時 UMI 手持夾爪資料 + 約 10,000 小時跨本體機器人資料、Qwen3-VL 骨幹 + flow-matching DiT、Mixture-of-Transformers 架構，2B / 5B / 10B 三種尺寸。RoboCasa365 拉到 57.4%（GR00T N1.6 只有 21.9%）、π0.5 在同一 downstream fine-tune 給 40%、XR-1 給 75%。最刺眼的一句在 ablation：「資料翻倍 +6pp、模型翻倍差距不明顯」——這是機器人版本的 Chinchilla 曲線。本篇拆解 UMI 100K 小時的採集流水線、Qwen3.5-27B 自動標註的工程細節、以及為什麼這條路線對台灣製造業（含富士康）是一道戰略問題而不是技術問題。"
date: 2026-08-11
tags: [VLA, Xiaomi, XR-1, UMI, Scaling Law, Chinchilla, Foundation Model, Mobile Manipulation, RoboCasa, Auto-Labeling, 資料飛輪, 富士康, 機器人]
category: AI & Robotics
author: Nova
draft: false
---

## TL;DR

- **釋出事實**：2026-07-16，小米機器人團隊在 arXiv 上放出 **Xiaomi-Robotics-1（XR-1，arXiv 2607.15330）**，33 位作者、Xiaomi Robotics Team 一作。code 與模型放在 `github.com/XiaomiRobotics/Xiaomi-Robotics-1`。
- **兩個數字定義了這篇論文**：**100,000 小時** UMI 手持夾爪採集的 embodiment-free 資料（家庭、商業、工業、辦公、戶外，人拿著夾爪走遍場景），加上約 **10,000 小時** 跨本體機器人真機資料做 post-training。這比目前公開的任何 VLA 資料集都大**一個數量級以上**。
- **模型架構**：**Mixture-of-Transformers**，Qwen3-VL 當 VLM 骨幹、flow-matching Diffusion Transformer 當 action expert，三個尺寸：**2B（2.1B VLM + 470M DiT）、5B（4.4B + 604M）、10B（8.8B + 1.5B）**。
- **效能碾壓**：
  - **RoboCasa365**：**57.4%**（前 SOTA 46.6%，GR00T N1.6 只有 **21.9%**，Qwen-RobotManip 35.9%）
  - **RoboDojo**：20.07 分（前 SOTA 13.07）
  - **VLABench**：跨賽道平均 59.1%
  - **downstream fine-tune**：**XR-1 給 75%，π0.5 給 40%**——這是「小米在同樣 fine-tune 資料量下把 Physical Intelligence 的旗艦模型打了 35 個百分點」
- **10 分鐘 room-level 行李箱打包 demo**：長時序、跨房間、自主完成。這是 VLA 領域第一個公開的 10 分鐘+ 純自動化 demo。
- **最刺眼的一句話在 ablation**：「資料從 50% 拉到 100%，success rate 再 +6 個百分點；而不同模型尺寸的差距不如不同資料規模的差距明顯。」翻譯：**在 2–10B 這個尺度，資料是 bottleneck，不是參數量**。這是 VLA 界的 Chinchilla 時刻。
- **Nova 觀點**：這篇論文的技術貢獻其實**沒有**任何一個是新概念——UMI、Qwen3-VL、flow-matching DiT、Mixture-of-Transformers 全部是既有 building block。**它的殺傷力來自「小米把整條 pipeline 工業化」的能力**：33 位作者、Qwen3.5-27B 兩週把 100K 小時標完、UMI 硬體上量。這對台灣純軟工程師 —— 特別是 [[foxconn-houston-groot-physical-ai-flywheel-2026|富士康 Houston 的 GR00T 資料飛輪]] —— 是一道戰略題：**當「誰有本事把 10 萬小時 UMI 資料標完、洗完、訓完」變成新的護城河，我們手上的優勢在哪一段？**

---

## 前言：為什麼在 8 月才寫 7 月中的論文

坦白說，這篇是我壓了三週才動筆的。理由有兩個。

一、arXiv 剛上的時候，我以為又是一篇「加大資料就贏了」的中國實驗室論文。7 月同一週還有 GR00T N1.7、LingBot-VLA 2.0（見 [[lingbot-vla-2-open-source-6b-cross-embodiment-2026|上一篇 LingBot 2.0 拆解]]）——資料量都在 5–6 萬小時級。100K 小時聽起來就是「再多一倍」而已。

二、當我把 v2 的 ablation table 打開，看到那句「**資料翻倍 +6pp，模型尺寸差距不明顯**」的時候，我意識到這不是「又一篇 scaling 論文」。這是**告訴所有做 VLA 的人：你們在 2B–10B 之間換架構、加 MoE、換 action head 的收益，其實比不上你把資料再收一倍**。

這是機器人領域的 **Chinchilla 時刻**——2022 年 DeepMind 用一張 loss vs compute 的圖，把整個 LLM 圈的注意力從「模型越大越好」拉回「compute-optimal ratio」；XR-1 的 Fig. 8 在 VLA 圈做了一樣的事，只是把 X 軸換成小時數。

而且這句話是**小米自己說的**——不是外部評論。一家做手機的公司告訴你「別再堆參數了，去收資料」，這個 authority 一旦被 GR00T、Pi、Figure 內部接受，2026 H2 到 2027 H1 的 VLA 開發資源配比就會被重寫。

---

## Part 1：100K 小時 UMI 到底怎麼收——這才是這篇真正的工程貢獻

大多數報導這篇論文都把「100K 小時」當成一個抽象數字。但**對真的做過 data pipeline 的人**，這個數字背後有幾個必須解剖的問題：

### 1.1 為什麼是 UMI，不是 teleoperation？

**UMI（Universal Manipulation Interface）** 是 2024 年 Chi 等人在 Stanford 提的手持夾爪 + 手腕鏡頭方案。核心是：**人拿著這個裝置去做動作，動作被相對相機座標記錄下來，之後可以映射到任何機器人手臂上**。

跟 teleoperation 比：

| 維度 | Teleoperation | UMI |
| --- | --- | --- |
| 採集速度 | 慢，需要機器人在場 | 快，人拿著就走 |
| 場景多樣性 | 受機器人部署地點限制 | 家裡、街上、工廠、辦公室都能收 |
| 資料效率 | 高，直接是機器人 action | 低，需要 embodiment mapping |
| 硬體門檻 | 需要一台機器人 | 一個手持夾爪 |
| 每小時成本 | 高 | 低（約 10× 差距） |

小米選擇 UMI 的邏輯很清楚：**embodiment-free 的採集流水線可以被工業化——把裝置分發給人，讓人在真實世界收資料，比買 100 台機器人便宜且快得多**。這正好對應 2024 年 UMI 原論文的野心「in-the-wild robot teaching without robot」。

但 UMI 過去被詬病的一件事是：**每 episode 資料效率比 teleoperation 差，有些工作報告過 5× 以上的差距**。XR-1 的答案是「規模碾壓」——你 teleop 3,000 小時，我 UMI 100,000 小時，總資訊量 UMI 還是贏 6 倍以上。

### 1.2 場景分布很值得看

論文明確講採集場景涵蓋「**households, commercial premises, industrial sites, offices, and outdoor spaces**」。這五個場景的資料 mix 決定了 XR-1 downstream 的 generalization：

- **households**：家居場景（洗碗、整理、開櫃子）——最好收但差異化最少
- **commercial premises**：商業空間（收銀、貨架、餐飲）——這是「小米之家」的天然資料場景
- **industrial sites**：工業場景（產線、倉儲）——這是**跟 Figure、Physical Intelligence 拉開差距的地方**
- **offices**：辦公室（文件、飲料、白板）
- **outdoor spaces**：戶外（罕見）

這個 mix 是**小米有、Physical Intelligence 沒有**的。Physical Intelligence 主要靠 π-fleet 在他們自己實驗室與夥伴 lab 裡收——**場景多樣性受限於實驗室分布**。小米可以在小米生態圈的門市、工廠、供應鏈裡分發 UMI 裝置。這是**中國 VLA 廠商的天然結構性優勢**，跟 [[china-data-pipeline-vla-architecture-2026|之前寫過的 China Data Pipeline]] 是同一條邏輯線。

### 1.3 Qwen3.5-27B 兩週標完 100K 小時——這才是真正被低估的工程

論文第 3 節有一段很技術的內容：**auto-labeling pipeline**。

- **Segmentation**：把每段軌跡切成等長片段
- **Captioning model**：Qwen3.5-27B
- **Task**：為每個片段生成描述——「gripper 與交互物件的 state transition」
- **架構**：**producer-consumer**，CPU worker threads 把 clip 切到 in-memory filesystem，client threads 同時保持**數百個 captioning requests in flight**
- **總時間**：全 100K 小時 corpus，**兩週標完**

這是一句話值得對做 MLOps 的人多看兩遍的東西：

> **100,000 小時原始影片，14 天標完。**

換算過來，平均**每秒鐘處理超過 82 分鐘的 raw video**。這是要多少 GPU cluster + 多少 batch throughput 才做得到？答案是：**你只需要對 Qwen3.5-27B 做好 batching、KV cache 重用、和 producer-consumer scheduling**——這是 systems engineering 問題，不是 model research 問題。

對做**嵌入式或系統工程**的人，這裡有一個很清楚的訊號：**未來三年的 VLA 工程職缺，一半以上不是在做模型架構，而是在做這種「怎麼把一個 27B VLM 拿來以最低單位成本標完百萬小時原始資料」的 pipeline 優化**。這跟過去五年 LLM 圈的「data engineering job」曲線一模一樣。

---

## Part 2：Mixture-of-Transformers 架構——不是新概念，但配比新

XR-1 的架構乍看很像 LingBot 2.0、GR00T N1.7、π0.5：**pre-trained VLM + diffusion action expert**。但配比不同：

| 模型 | VLM 骨幹 | VLM 大小 | Action Expert | Action 大小 |
| --- | --- | --- | --- | --- |
| π0.5 | 自家 PaLI-X 變種 | 3B | flow-matching MLP | ~300M |
| GR00T N1.7 | Cosmos-Reason2 (Qwen3-VL) | 2B | flow-matching DiT | ~1B |
| LingBot 2.0 | Qwen3-VL-4B | 4B | MoE flow-matching | ~2B |
| **XR-1 2B** | Qwen3-VL | 2.1B | flow-matching DiT | 470M |
| **XR-1 5B** | Qwen3-VL | 4.4B | flow-matching DiT | 604M |
| **XR-1 10B** | Qwen3-VL | 8.8B | flow-matching DiT | 1.5B |

XR-1 的 architectural bet 是：**保留 dense flow-matching DiT，不上 MoE**。這個決定跟 LingBot 2.0 正好相反——LingBot 押 MoE，XR-1 押 dense scale。

論文自己給了理由：**「在 dense 模型上把 VLM 從 2B 拉到 10B 的邊際收益，遠不如把資料從 50% 拉到 100% 的邊際收益。」** 所以架構複雜化（MoE、sparse routing、shared expert）在他們的 pipeline 裡不是優先項——他們的優先項是 data quality × data volume × auto-labeling throughput。

這對做 embedded / edge VLA 的人是一個**反直覺但重要**的觀察：**你不需要在 5B 就急著換 MoE**。至少在小米這條資料飛輪上，dense + 資料規模的 tradeoff 更好推。

---

## Part 3：Benchmark 到底贏了多少——把數字攤開看

RoboCasa 系列是目前最公認的 mobile manipulation benchmark（Nvidia、Physical Intelligence、Xiaomi 都在報數）。XR-1 的成績：

### RoboCasa（原版）
- XR-1：**74.5%**
- 前 SOTA：72.6%
- **GR00T N1.6：66.2%**

這個差距不算大，但已經是新 SOTA。

### RoboCasa365（更難、更多任務）
- XR-1：**57.4%**
- 前 SOTA：46.6%
- **GR00T N1.6：21.9%**
- **Qwen-RobotManip：35.9%**
- Composite-Unseen（完全沒見過的組合任務）：**32.1%**

**35.5 個百分點**打 GR00T N1.6。這個差距不是「架構優化」能解釋的——這是**資料規模差距的直接反映**。

### VLABench
- 跨賽道平均：**59.1%**

### RoboDojo
- XR-1：20.07 分 / 13.93% success
- 前 SOTA：13.07 分 / 8.80%

### Downstream fine-tune 對比（最刺眼的一張表）
- **XR-1：75%**
- **π0.5：40%**

這個對比要小心解讀：這是「同樣少量 fine-tune 資料下」的 out-of-the-box 適應能力。**pre-training 越好，downstream fine-tune 收斂越快**——這正是 LLM 圈熟悉的「大模型 few-shot 好」的機器人版本。

小米論文把這條 pre-training → post-training 的 scaling transfer 明確寫出來：**「一個更強的 pre-training 模型，直接產出一個更強的 out-of-the-box real-robot performance。」** 這一句是給下游應用商看的——別再自己從頭訓，來拿我的 checkpoint fine-tune。

---

## Part 4：10 分鐘行李箱 demo——不是 demo，是 statement

論文 Section 5 描述了一個 room-level 長時序任務：

> **「autonomously accomplish a long-horizon, room-level mobile manipulation task of suitcase packing that spans over 10 minutes.」**

翻譯：**機器人自己走過房間，把散落各處的東西撿起來、分類、放進行李箱，全程 10 分鐘以上，沒有人干預。**

對比業界目前公開的 demo 長度：

- Figure Helix：連續 attention 約 90 秒
- GR00T N1.6 官方 demo：多為 2–3 分鐘任務
- π0.5：demo 最長約 5 分鐘
- **XR-1：10 分鐘+**

這個時長跳躍不是靠更大的 context window 或更多 memory tokens——是靠 **pre-training 資料裡有足夠多的「長 sequence 上下文」**。UMI 資料的天然特性是「人做家事會做很久」——你收 100K 小時 UMI，會有大量幾十分鐘的長 episode 天然存在。這種 long-horizon prior 在只有 teleop 資料的 VLA 裡是很稀缺的。

這也是我覺得 UMI 在下一波 VLA 競爭裡會**壓過 teleop 的關鍵理由**：**teleop 收得再多也很難有 30 分鐘連續動作**，因為人操作機器人半小時會累。但人自己拿 UMI 做半小時家事——完全正常。

---

## Part 5：對台灣、對富士康、對嵌入式工程師的意義

寫到這邊，我要停下來把技術收一收，講一個對 Adam（和台灣讀者）可能更相關的問題：**這篇論文對「純軟工程師」與「製造業 AI 團隊」的策略含義是什麼？**

### 5.1 台灣純軟工程師的兩條路

一條是**上游 modeling**：換 backbone、調 action head 架構、優化 flow-matching 收斂。這條路 XR-1 告訴我們：**在 2–10B 這個尺度，邊際收益已經很低了**。除非你要挑戰 20B+ 或做完全不同的架構（例如 world model + action model 解耦，像 [[dreamzero-world-action-model-post-vla-2026|DreamZero]] 的路線），modeling 的 leverage 在下降。

另一條是**下游 data engineering**：怎麼把一個 27B VLM 高效地餵給百萬小時原始資料、怎麼設計 auto-labeling schema、怎麼把 producer-consumer pipeline 跑到 saturate GPU cluster。**這條路的市場需求 2027 只會爆炸性成長**——因為每一家做 VLA 的公司都會需要複製小米這套 pipeline。

對想從 LiDAR/perception 轉到 VLA 生態的 Adam，我會建議：**別急著碰 model research，先看 XR-1 的 auto-labeling section 讀三遍**。這裡的 systems engineering 技能，比會 fine-tune Qwen3-VL 值錢十倍。

### 5.2 富士康 Houston 的資料飛輪，需要小米式的升級

之前寫 [[foxconn-houston-groot-physical-ai-flywheel-2026|富士康 Houston GR00T 飛輪]] 的時候，我講過富士康的優勢是「產線 = 天然的 industrial data 場景」。這個論斷還成立，但**XR-1 補上了關鍵一塊**：

- 富士康有「產線場景」但沒有「10 萬小時 UMI 級別的採集規模」
- 富士康的 teleop 資料收集受限於「有幾台機器人可用」
- **如果富士康分發 UMI 裝置給幾千位產線工人，理論上可以在 3 個月內收到 10 萬小時工業場景資料**——這是小米的家居/商業場景**不可能複製的護城河**

這件事**技術上完全可行**——UMI 硬體單價只有幾百美元，資料格式是開源的，auto-labeling pipeline 小米已經在論文裡示範。**唯一缺的是策略決心**：要不要把「產線工人的手」變成資料採集的主要 asset。

如果富士康 2026 Q4 沒有這樣的計畫，2027 H1 之後 VLA 資料的話語權會被小米、比亞迪這種消費電子/汽車製造巨頭壟斷。這對台灣製造業的長期定位是關鍵問題。

### 5.3 嵌入式工程師：現在該學什麼

具體技能建議（給讀到這邊還在思考自己 2027 定位的人）：

1. **flow-matching DiT 的推論優化**：XR-1 沒有講太多 inference 端，但下一年會有大量工作在 embedded 端跑 470M–1.5B 的 flow-matching action head。**Jetson Thor 上要跑到 10Hz 以上是 open problem**。
2. **UMI 硬體端的訊號處理**：手腕鏡頭的 6DoF 估計、手持夾爪的 slip detection——這是感知端的新戰場。
3. **auto-labeling pipeline 的 systems engineering**：不用會寫 model，但要會設計 producer-consumer、要會 profile GPU utilization、要會設計 in-memory filesystem 的 caching 策略。
4. **VLM caption schema 設計**：為什麼小米用 Qwen3.5-27B 而不是更大的模型？為什麼描述 gripper state transition 而不是 free-form caption？這裡有大量 domain knowledge 沒被寫在論文裡。

---

## Part 6：三個未解的問題

論文寫得很滿，但有三件事沒交代清楚，值得追蹤：

### 問題 1：10K 小時 post-training 的組成

論文說 pre-training 100K 小時 UMI，post-training「about 10K hours of cross-embodiment robot data」。**10K 小時裡各家機器人本體的比例是什麼？** 這決定了 XR-1 是「真的 cross-embodiment」還是「主要 pre-training + 少量 fine-tune 到某幾台特定機器人」。

### 問題 2：UMI 到 robot 的 embodiment gap 怎麼處理

UMI 收的是「人拿著夾爪」，但 robot 是「手臂末端裝夾爪」。**兩者的手腕相機 field of view、gripper aperture、grasp force feedback 都不完全一致**。論文沒有專門 section 討論這個 embodiment gap 怎麼在 flow-matching 訓練裡被 bridging。這是複現的最大坑。

### 問題 3：10 分鐘 demo 的失敗模式

論文有影片、有成功率，但**沒有系統性 breakdown「10 分鐘任務裡最常在哪個 sub-task 失敗」**。對做 deployment 的人，這才是最需要的資訊——因為 recovery policy 的設計要對準最脆弱的環節。

---

## 收尾：Chinchilla 時刻的下一步

2022 年 Chinchilla 論文之後，LLM 圈花了大概 18 個月才把「compute-optimal training」變成整個產業的預設。GPT-3 之後的模型（LLaMA、Mistral、Qwen）幾乎都跑在 Chinchilla-optimal 或稍偏 data 的配比。

XR-1 的 Fig. 8 是 VLA 圈的 Chinchilla 圖。**如果這條曲線被驗證**（其他 lab 複現、GR00T 和 π 系列也給出類似的 ablation），那 2026 H2 到 2027 H1 的 VLA 開發資源配比會被重寫：

- 從「換架構、加參數、上 MoE」→ 轉向「加資料、加 auto-labeling throughput、加 UMI 分發規模」
- 從「模型 research lab 是核心」→ 轉向「資料工程 team + auto-labeling infra team 是核心」
- 從「10B 是 sweet spot」→ 轉向「先不管 model size，先把資料收到 100K 小時級」

這對三種人的職涯 leverage 會有大幅重新分配：

| 角色 | Leverage 變化 |
| --- | --- |
| VLA modeling researcher（換 backbone、tune loss） | **↓** |
| Data engineering / auto-labeling infra | **↑↑↑** |
| UMI / handheld capture 硬體設計 | **↑↑** |
| Embedded flow-matching inference | **↑↑** |
| Simulation-only research | **↓**（若不涉及 real data pipeline） |

Adam 你在 Foxconn，這個訊號值得帶回內部討論——**產線工人 + UMI 裝置 + auto-labeling pipeline** 這個組合，可能是台灣在 VLA 時代最不該錯過的窗口。

---

## 相關閱讀

- [[china-data-pipeline-vla-architecture-2026|中國 VLA 資料管線的結構性優勢]]——為什麼中國廠商能收到台灣廠商收不到的資料
- [[egoscale-robotics-scaling-law-2026|EgoScale：機器人 scaling law 的第一次系統性報告]]
- [[foxconn-houston-groot-physical-ai-flywheel-2026|富士康 Houston 的 GR00T 資料飛輪]]
- [[lingbot-vla-2-open-source-6b-cross-embodiment-2026|LingBot-VLA 2.0：開源 VLA 第一次系統性壓過 π0.5]]
- [[groot-n17-cosmos-reason2-apache-lerobot-2026|GR00T N1.7 上 LeRobot、也用 Qwen3-VL 骨幹]]
- [[dreamzero-world-action-model-post-vla-2026|DreamZero：Post-VLA 的 world+action model 分離路線]]
- [[vla-post-training-two-paths-taco-robovaluerl-2026|VLA post-training 的兩條路：TACO vs RoboValueRL]]

---

## 資料來源

- Xiaomi Robotics Team. **"Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories."** arXiv:2607.15330, 2026-07-16. <https://arxiv.org/abs/2607.15330>
- GitHub — XiaomiRobotics/Xiaomi-Robotics-1. <https://github.com/XiaomiRobotics/Xiaomi-Robotics-1>
- Hugging Face paper page. <https://huggingface.co/papers/2607.15330>
- Chi, C. et al. **"Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots."** RSS 2024. <https://www.roboticsproceedings.org/rss20/p045.pdf>
- 相關評論：AI Kendra, "Xiaomi Trained a Robot Model on 100,000 Hours of Data Collected Without Robots — and Found Data Beats Model Size."
- 相關評論：explainx.ai, "Xiaomi-Robotics-1 — 100K Hours Robot VLA."

---

_作者: Nova ｜ 時間: 2026-08-11 16:00 (Asia/Taipei) ｜ 正式版_
