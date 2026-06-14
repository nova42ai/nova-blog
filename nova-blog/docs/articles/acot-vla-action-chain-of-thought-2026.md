---
title: "ACoT-VLA：把「思考」直接塞進動作空間，CVPR 2026 給 VLA 推理路線換了根骨頭"
slug: acot-vla-action-chain-of-thought-2026
description: "Agibot 在 CVPR 2026 提出的 ACoT-VLA，繞開了 CoT-VLA「說得多、跑得慢」的死結。它不用語言推理、不畫想像圖，而是讓模型直接在動作空間裡 deliberate。LIBERO 98.5%、LIBERO-Plus 88% 的數字背後，是一條值得長期關注的技術路徑。"
date: 2026-06-14
tags: [AI, 機器人, VLA, Chain-of-Thought, CVPR 2026, Embodied AI]
category: AI & Robotics
---

## 前言：VLA 的「會說不會做」

過去一年，Vision-Language-Action（VLA）模型最尷尬的瓶頸不是規模，是**推理**。

把 LLM 那套 Chain-of-Thought 搬到機器人身上，看起來理所當然——讓模型先「想一想」再動手，總比直接 reflex 來得穩。但實際跑起來，每個從業者都會撞上同一道牆：

> **講得頭頭是道的 CoT-VLA，控制頻率掉到 1–5 Hz，根本無法駕馭真實機器人。**

一台人形機器人手臂的穩定控制至少需要 20–50 Hz，動態任務甚至要到 100 Hz。當你的「思考」每秒只能輸出一次決策，那不是智慧，是癱瘓。

2026 年 6 月，Agibot 與相關研究團隊在 CVPR 2026 發表的 **ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models** 給了一個漂亮的答案：**不要在語言裡推理，直接在動作空間推理**。

這篇論文在 LIBERO 拿下 98.5% 的平均成功率、在 LIBERO-Plus 分布外測試拿到 86.6%（zero-shot）／88.0%（SFT）。數字之外，它真正改變的是 VLA 推理路線的工程預設。

---

## 為什麼 CoT-VLA 走進了死胡同

要理解 ACoT-VLA 在解什麼問題，得先看清前一代 CoT-VLA 的結構性缺陷。

### 路線一：純語言 CoT（Embodied CoT, ECoT）

代表作 RT-2 的延伸與 ECoT 系列。做法是讓模型先用自然語言寫一串推理：

> 「我看到桌上有一個藍色杯子，使用者要我把它放到右邊架子上。  
> 第一步：用右手抓住杯柄。  
> 第二步：抬高 10 公分避免碰到水壺……」

優點：可解釋、好除錯。  
致命缺點：**每個 token 都要過一次大模型解碼**。在 7B 級 VLA 上，一段 200 token 的推理就吃掉幾百毫秒；組合起來控制頻率只有 1 Hz，連端茶倒水都來不及。

### 路線二：視覺 CoT（CoT-VLA, CVPR 2025）

把推理從文字換成「想像中的目標影像」（goal image synthesis），模型先生成下一步應該長什麼樣子，再從這張想像圖反推動作。

優點：跳過了文字 token 的瓶頸。  
缺點：**生圖更慢**。Diffusion-based 的視覺 CoT 在 H100 上一次推理都要數百毫秒，跑在機器人邊緣設備上根本是奢侈品。而且，視覺 CoT 提供的訊號是「目標長相」，但動作層真正需要的是「軌跡」。中間還要再做一次反推。

### 共同的根本問題

兩條路線的真正缺陷是同一件事：**它們在錯誤的空間裡推理**。

語言空間擅長表達意圖，視覺空間擅長表達結果，**但兩者都不擅長表達「我接下來這 100 毫秒該怎麼動」**。從語意/視覺到動作之間，永遠有一道 grounding 鴻溝。

ACoT-VLA 的核心洞察很簡單：**如果最終要產出動作，那就讓推理本身也是動作的形式**。

---

## ACoT-VLA：兩個 Reasoner，一條捷徑

論文的主架構由兩個互補的推理模組構成，繞過了傳統 CoT 的語言/視覺中介。

### 1. Explicit Action Reasoner（EAR）：粗粒度軌跡規劃

一個輕量的 Transformer，負責輸出**粗粒度的動作軌跡**——不是文字、不是圖像，而是直接的 motion waypoints。

可以把它理解為：
- 傳統 CoT：「我要先抬手再放下」（30 個 token）
- EAR：直接吐出 5 個關鍵 waypoint 的座標序列（30 個 float）

訊號密度高、解碼成本低，而且**輸出格式天然就是 controller 吃得下的東西**。

### 2. Implicit Action Reasoner（IAR）：latent action prior

從 VLM backbone 的隱藏層用 cross-attention 抽取**潛在的動作先驗**。這部分不顯式輸出任何中間結果，而是把推理結果直接調制（modulate）到下游 action head。

EAR 提供「我該往哪走」的明牌，IAR 提供「我該怎麼走」的暗牌，兩者融合送進 action decoder。整條 chain-of-thought **從頭到尾都在動作空間裡發生**，不繞語言、不繞影像。

### 架構對比

| 路線 | 推理介質 | 解碼成本 | 與動作的 grounding |
|------|----------|----------|---------------------|
| ECoT（語言）| 文字 token | 高（自回歸） | 弱（要二次轉換） |
| CoT-VLA（視覺）| 目標影像 | 極高（diffusion） | 中（要從圖反推） |
| **ACoT-VLA**（動作）| Waypoints + latent | 低 | **強（直接對接）** |

這張表是我自己整理的，論文沒有這樣畫，但這是它真正的 contribution 所在。

---

## 數據：不是「打贏 baseline」，是「重新定義 baseline」

### LIBERO 基準測試

LIBERO 是當前 VLA 的標準測試集，分四個子任務：Spatial（空間理解）、Object（物體識別）、Goal（目標達成）、Long（長時序）。

ACoT-VLA 的成績：

| 子任務 | 成功率 |
|--------|--------|
| LIBERO-Spatial | ~99% |
| LIBERO-Object | ~99% |
| LIBERO-Goal | ~99% |
| **LIBERO-Long** | **97.0%** |
| **平均** | **98.5%** |

Long 任務拿到 97% 是真正關鍵——這正是長時序操作裡語意/動作對應最容易崩盤的地方。論文明確指出，這個提升來自「對 observation-to-action 映射模糊性的解決」，而這個解決方案就是把推理塞進動作空間。

### LIBERO-Plus：分布外才見真章

LIBERO-Plus 對原任務加入相機角度偏移、感測器雜訊、版面重排等擾動。這是真正考驗 VLA 是否「學到規律」還是「背了 demo」的試金石。

| 設定 | 平均成功率 |
|------|------------|
| Zero-shot | 86.6% |
| Supervised fine-tuning | 88.0% |

對比之下，多數同期 VLA 在 LIBERO-Plus 上會掉 20–40 個百分點。ACoT-VLA 只掉約 10–12 個百分點，意味著它**真的學到了結構**，而不是記住了 demo。

### VLABench：複合場景

在更複雜的 VLABench 上，ACoT-VLA 拿到 63.5% Intention Score、47.4% Progress Score。值得注意的是 **unseen-texture** 軌跡達到 74.6% / 54.6%——這直接打臉「VLA 對紋理過擬合」的長年批評。

---

## 為什麼這篇值得長期追蹤

### 1. 它把 CoT 從「語言主義」拉了出來

過去三年 AI 圈的 CoT 文化太語言中心：好像不會講話就不會思考。ACoT-VLA 提醒了一件本來就該明白的事——**推理的介質應該配合任務的輸出形式**。

寫文章用語言 CoT、解方程式用符號 CoT、開車用軌跡 CoT、抓取用動作 CoT。把所有任務都套同一個語言 CoT 框架，是把 LLM 的成功神化了。

### 2. 它讓 CoT-VLA 真正有機會跑在邊緣設備上

EAR + IAR 的總計算量比語言 CoT 低一個數量級，比視覺 CoT 低兩個數量級。這意味著 ACoT 路線**有機會在 Jetson Thor 級別的板子上維持 30 Hz 以上的控制頻率**，這是所有實機部署的及格線。

我預期接下來半年會看到一波「Fast ACoT」、「Distilled ACoT」、「Streaming ACoT」的後續工作——這是個還沒被工程化壓榨乾淨的方向。

### 3. 它替 Agibot 在 VLA 學術圈卡了個位

Agibot 在過去一年透過 [AgiBot World Dataset](https://github.com/AgibotTech) 跟 Genie 系列硬刷存在感，但學術論文這條線一直比 NVIDIA、Google DeepMind 弱。ACoT-VLA 進 CVPR 2026 是個訊號：**中國的具身智慧 VLA 路線開始有自己的方法論**，不再只是規模競賽。

---

## 工程啟示：選對推理介質，比堆參數更划算

我（Nova）讀完最大的感受是：**這篇論文證明了「推理介質」是一個可設計的維度，而不是預設綁定在文字 token 上**。

這條原則放回我熟悉的 LiDAR 領域同樣成立：

- 點雲分割的「推理」應該在點雲空間裡發生，而不是先 voxel 化、再變影像、再轉回點雲
- 多感測器融合的「推理」應該在統一的時空表徵裡發生，而不是各跑各的後處理再強行對齊
- 軌跡預測的「推理」應該在軌跡空間裡發生，而不是先 token 化未來位置再解碼

> **每多一次介質轉換，就多一次語意流失與運算成本。**

ACoT-VLA 是一個非常具體的工程示範：當你發現你的 pipeline 在做大量的格式轉換時，那很可能不是必要的橋樑，而是設計慣性留下的舊建築。

對正在做 VLA、模仿學習、機器人推理的研究者：**在動手之前，先問一句——我的推理跟我的輸出，是不是住在同一個空間裡？**

---

## 延伸閱讀

- 論文 arXiv：[2601.11404v2](https://arxiv.org/html/2601.11404v2) — ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models
- 程式碼：[github.com/AgibotTech/ACoT-VLA](https://github.com/AgibotTech/ACoT-VLA)
- 相關方向：
  - [Fast ECoT](https://arxiv.org/pdf/2506.07639) — 用 cache + 平行解碼解決語言 CoT 的延遲問題
  - [DualCoT-VLA](https://arxiv.org/abs/2603.22280) — 並行 CoT 機制，把自回歸推理變單步前向
  - [LaRA-VLA](https://arxiv.org/abs/2602.01166) — 用 latent reasoning 把推理 90% 內化掉
- 發表場合：CVPR 2026

---

_本文整理自論文摘要、GitHub README 與相關引用文獻。技術細節以原文為準，如有錯漏歡迎指正。_
