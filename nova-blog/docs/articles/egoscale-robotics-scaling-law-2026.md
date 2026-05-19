---
title: "EgoScale：20K 小時人類影片，把 Scaling Law 搬進機器手"
slug: egoscale-robotics-scaling-law-2026
description: "NVIDIA GEAR 在 2026 年 2 月丟出 EgoScale 論文，用 20,000 小時 egocentric 人類影片預訓練，首次在機器人領域畫出像 LLM 一樣乾淨的 log-linear scaling law（R²=0.9983）。22-DoF 機械手的成功率從 30% 拉到 71%——這可能是 2026 年最重要的模型層發現。"
date: 2026-05-19
tags: [AI, 機器人, VLA, Scaling Law, Dexterous Manipulation, NVIDIA, 2026]
category: AI & Robotics
---

## 前言：機器人領域終於有了「自己的 GPT-3 時刻」

過去三年，每當有人問「機器人什麼時候會有 ChatGPT 那樣的爆發點？」答案大多是同一句話：**「等 scaling law 出現再說。」**

LLM 的故事大家都熟：Kaplan 2020、Chinchilla 2022，把「資料量 × 參數量 × 算力」與 loss 的關係畫成乾淨的直線，整個產業才敢把幾十億美金壓在預訓練上。但機器人這邊，多年來的共識是「資料是瓶頸、但是不是 scaling 的形狀沒人能保證」。Real-world 機器人資料貴到離譜（一小時遙操示教動輒上千美元），就算砸錢做了，能不能換成可預測的能力提升？沒人敢拍胸脯。

直到 2026 年 2 月，**NVIDIA GEAR** 把這篇丟上 arXiv：

> **EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data**

一句話總結：**用 20,000 小時的 egocentric 人類影片預訓練 VLA，loss 與資料量呈完美 log-linear 關係（R² = 0.9983），而 loss 又能直接預測真機成功率。**

這是機器人領域第一次拿到乾淨、可外推、且與真機表現強相關的 scaling law。它意味著——錢可以開始放心地往「人類影片資料」這條路上倒了。

---

## 核心發現：那條漂亮到不像真的的直線

EgoScale 把 human pretraining data 從 1k 小時逐步擴到 20k 小時，量出來的 validation loss 滿足：

$$
L = 0.024 - 0.003 \cdot \ln(D)
$$

其中 D 是人類預訓練資料的小時數，R² = **0.9983**。

這個 R² 數字看起來像在唬人，但對熟悉 LLM scaling law 的人來說很眼熟——Kaplan 那篇 power-law 擬合的 R² 大概就是 0.99 等級。換句話說，**這不是「大致呈現某種趨勢」，而是「教科書級別的直線」**。

更關鍵的是下游表現的同步：

| 預訓練資料量 | Validation Loss | 真機平均成功率 |
|---|---|---|
| 1k 小時 | （baseline）| 0.30 |
| 20k 小時 | log-linear 下降 | **0.71** |

loss 不只是 loss——它在 5 個 dexterous task 平均下來能線性預測真機成功率。**這才是 scaling law 真正的意義**：你不用每次都跑真機 evaluation，看 pretraining loss 就能知道值不值得繼續砸資料。

---

## 模型架構：Flow-based VLA + DiT Action Expert

EgoScale 不是純粹靠資料量堆出來的，架構選擇本身也很有訊息量：

- **Backbone**：VLM（vision-language model）做感知與語意理解
- **Action Expert**：DiT（Diffusion Transformer）負責生成連續動作分佈
- **整合方式**：flow-based VLA policy，連續軌跡用 flow matching 採樣
- **跨 embodiment 對齊**：人類與機器人用統一的 **wrist-level action representation**，proprioceptive 與手部動作各自有 embodiment-specific adapters

這個設計回答了一個老問題：**人類的手跟機器人的夾爪 / 五指手根本不一樣，憑什麼可以共享預訓練？**

EgoScale 的答案是「在 wrist（手腕）這一層對齊」。手腕的 6-DoF 位姿在人類與機器人之間幾乎可以一對一對應，於是把手指動作交給 embodiment adapter 各自處理。這個切分相當聰明——抽象到夠高的層次共享，但不過度抽象到失去物理意義。

---

## 訓練設定：算力沒在客氣

來看看 NVIDIA 自家是怎麼跑這個的：

| 階段 | GPU | Steps | Batch Size |
|---|---|---|---|
| Stage I：Human Pretraining | **256× GB200** | 100K | 8192 |
| Stage II：Human-Robot Mid-training | （未明示）| 50K | 2048 |
| Stage III：Robot Post-training | （未明示）| 10K | 512 |

256 顆 GB200 跑 100K steps 是什麼概念？大概是中型 LLM 預訓練的算力等級。**機器人領域過去從來沒有人在預訓練階段砸這個量級的算力**——因為過去沒有 scaling law 能保證 ROI。EgoScale 反過來說：先放心砸，曲線會告訴你停在哪。

三階段流程也很值得注意：
1. **Stage I（廣度）**：純人類 egocentric 影片，學感知 + 動作 prior
2. **Stage II（橋接）**：少量「人類-機器人對齊」資料做 mid-training
3. **Stage III（特化）**：真機 demonstration 做 post-training

關鍵啟示：**Stage II 是 secret sauce**。沒有這個對齊步驟，Stage I 的人類預訓練就會卡在「看得懂、做不到」的尷尬狀態。

---

## 真機表現：22-DoF 機械手做 5 件事

實驗用的是一隻 **22-DoF 機械手**（接近人手自由度），測試五個任務：

1. **Shirt Rolling**（捲衣服）
2. **Card Sorting with Tongs**（用夾子分牌）
3. **Bottle Uncorking**（開瓶蓋）
4. **Syringe Liquid Transfer**（針筒抽取液體）
5. **Pen / Dish Placement**（筆與碗的擺放）

這幾個任務的共通點是 **fine motor control + 長時序**——傳統 VLA 的死穴。

**結果**：相對「無預訓練 baseline」平均提升 **54%** 成功率。

更猛的是 **one-shot 適應**：

- **Fold Shirt**（摺衣服，訓練集沒出現過）：0.88 成功率
- **Water Bottle Unscrewing**（旋瓶蓋）：0.55 成功率

注意「one-shot」的意思——只給一次示教，模型就要直接會做。0.88 在這個設定下已經是讓人睜大眼睛的數字。

---

## 為什麼這篇重要：三個產業層級的衝擊

### 1. 資料策略徹底翻轉

過去機器人公司的標準操作：**砸錢遙操、收 demonstration、訓 BC policy**。一小時資料 800-1500 美元，10,000 小時就是千萬等級的純資料成本。

EgoScale 之後，路線改寫成：

```
1. 大規模 ego-centric 人類影片預訓練（資料便宜、可從 YouTube / 公開資料集挖）
   ↓
2. 少量人類-機器人對齊資料（mid-training）
   ↓
3. 極少量真機 demonstration（post-training / one-shot adaptation）
```

**遙操資料從「主菜」變成「最後一道調味」**。這對 Tesla Optimus、Figure、Unitree 這些大規模收 demonstration 的廠商是個架構性威脅——也是機會（看誰先把 pipeline 改過來）。

### 2. 評估指標跟著重組

loss 既然能線性預測成功率，那 **pretraining loss 本身就是商業 KPI**。研發團隊不用每改一版就排隊上真機，看 loss 曲線就能判斷模型有沒有變強。這跟 LLM 從「跑 MMLU」變成「看 loss 曲線」是同一條路。

### 3. Hand pose 標註技術鏈被瞬間放大

EgoScale 的人類動作標註靠 **SLAM + hand-pose estimation + retargeting** 的 pipeline。這條鏈條過去是學術圈的小眾領域，現在突然變成「能放大 100 倍效益」的 critical path。

MANUS Meta（那家做 mocap 手套的公司）已經跟 NVIDIA 合作了。**未來 12 個月內，hand-tracking 與 ego-video 標註會從周邊技術躍升為機器人預訓練的核心 infrastructure**——而且窗口期不長。

---

## 不要太興奮：值得潑冷水的三個點

身為一篇要好好被讀的部落格文章，這節必須有。

### 1. 標註鏈條本身有 noise

SLAM 漂移 + hand pose 估計誤差 + retargeting 失真，這三層 noise 累積起來不是小事。論文沒有完整 ablate「如果手部標註再準 10%，scaling law 的斜率會不會更陡？」這意味著 **目前的曲線可能還沒到天花板，也可能離天花板很近——我們不知道**。

### 2. 飽和點未知

20k 小時是論文的上界。下一段是 50k？100k？還是早就在 25k 飽和？沒有人知道。LLM 領域至少 Chinchilla 給了「資料與參數的最佳比例」，EgoScale 還沒走到這一步。

### 3. Mid-training 資料的依賴沒被解決

Stage II 的「人類-機器人對齊資料」依然得人工收集。如果這份對齊資料的成本跟原本的純遙操差不多，那 scaling law 帶來的省錢效益就要打折。論文承認這是限制之一。

---

## 個人觀察：這是 2026 機器人最值得關注的論文

我看過今年很多「驚人 demo」——Figure 03、Unitree H1、AgiBot 的數據集發佈，每一個都熱鬧。但 **真正會改變產業 5 年走向的，是 EgoScale 這種「打開 scaling 之門」的論文，不是那些 demo**。

理由很簡單：demo 告訴你「現在能做到什麼」，scaling law 告訴你「未來能做到什麼、需要花多少錢」。一個是現況快照，一個是 roadmap。

如果你在做機器人相關研發，**這篇值得整篇精讀**（不是看摘要、不是看 X 上的線程，是真的 print 出來在桌上劃線那種精讀）。它對你未來兩年的資料策略、預訓練架構、評估流程都有具體影響。

如果你在投資 / 觀察這個產業，**重點觀察以下幾件事在接下來 6-12 個月會不會發生**：

- [ ] Tesla / Figure / Unitree 是否公開承認「人類影片預訓練是主菜」
- [ ] 第二、第三家獨立團隊重現 EgoScale 的 scaling 曲線
- [ ] 出現 100k 小時等級的開源 ego-centric 資料集
- [ ] Mid-training 對齊資料的成本下降技術（synthetic alignment？）

這四件事任何一件發生，就是趨勢確認；如果半年內都沒動靜，那 EgoScale 可能只是漂亮的單點論文，不是時代轉折。

---

## 參考資料

- [EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data (arXiv 2602.16710)](https://arxiv.org/pdf/2602.16710v1)
- [NVIDIA GEAR EgoScale Project Page](https://research.nvidia.com/labs/gear/egoscale/)
- [Paper Notes by Lixin Xu](https://davidlxu.github.io/posts/2026/02/egoscale-paper-notes/)
- [MANUS × NVIDIA Use Case](https://www.manus-meta.com/use-cases/nvidia-egoscale-scaling-dexterous-robot-manipulation-with-manus-gloves)
- [Neural Scaling Laws in Robotics (arXiv 2405.14005)](https://arxiv.org/pdf/2405.14005) — 較早的鋪墊論文，可對照閱讀
