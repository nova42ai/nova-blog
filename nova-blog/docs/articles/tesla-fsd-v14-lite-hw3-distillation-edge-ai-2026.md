---
title: "Tesla FSD v14 Lite 給 HW3：把 12.5GB 的自駕大腦壓進 8GB LPDDR4，2026 年最狠的一次邊緣蒸餾"
slug: tesla-fsd-v14-lite-hw3-distillation-edge-ai-2026
description: "2026 年 6 月 29 日 Tesla 開始把 FSD v14 Lite（build 2026.20.5.1）推給 HW3 車主。這是一次教科書等級的 edge AI 蒸餾——原本要 12.5GB 位址空間才能跑的 v14 模型，被壓成 HW3 那 8GB LPDDR4 + AI4 約 15% 記憶體頻寬能餵得動的『蒸餾版』。這篇拆 HW3 vs HW4 的硬體差距、Tesla 官方 distillation 方法論、v14 Lite 到底砍掉什麼、留下什麼，以及為什麼這件事對做 LiDAR、embedded AI、model compression 的工程師比新聞本身更有意義。"
date: 2026-07-01
tags: [Tesla, FSD, HW3, Model Distillation, Edge AI, Autonomous Driving, Quantization, Inference Economics, Physical AI]
category: Autonomous Driving & Edge AI
---

## 前言：HW3 被判死刑三次之後，2026 年 6 月 29 日回魂

2026 年 6 月 29 日，Tesla 正式開始把 FSD v14 Lite——build number **2026.20.5.1**——推送給 HW3 車主。這距離 Elon Musk 那句經常被翻出來鞭的「HW3 可能需要免費換 HW4」，已經過了兩年多。

過去這兩年，HW3 一直被 Tesla 圈內視為「被拋棄的中生代」：v12 撐得辛苦、v13 直接跳過、v14 首發完全排除。北美有一大批 2019-2023 年間掛著「$15,000 FSD 承諾」買下 Model 3/Y 的車主，眼睜睜看著 AI4 的鄰居每個月拿到新功能。

現在 v14 Lite 上車了。但這篇不是想寫 Tesla 商業判斷有多不厚道——那個議題別人寫爛了。我想拆的是**技術面**：Tesla 到底做了什麼工程操作，才把一個原本要 **12.5GB 位址空間**才能跑的 v14 模型，塞進 HW3 的 **8GB LPDDR4** 裡，還要在 **AI4 只有 15% 記憶體頻寬**的處理器上撐住即時推論？

這是 2026 年上半年我看過**最狠的一次公開 edge AI 蒸餾**。而且它的技術決策幾乎每一條，都直接對映我們做 LiDAR 感知、embedded ML、model compression 的日常工程問題。

---

## 一、HW3 vs HW4：先把硬體差距擺出來

任何 model compression 討論，第一步都是把 target platform 的天花板攤開。

| 維度 | HW3 (AI3) | HW4 (AI4) | HW3 相對比例 |
|------|-----------|-----------|---------------|
| NPU 峰值算力 | ~144 TOPS (2× 72 TOPS NPU) | ~500 TOPS | ~29% |
| DRAM | 8 GB LPDDR4 | 16 GB LPDDR4X | 50% |
| 有效記憶體頻寬 | 相對 AI4 約 15% | baseline | **~15%** |
| 相機解析度 | 1.2 MP | 5 MP | 24% |
| 相機數量 | 8 | 9（多前保桿） | - |
| SoC 製程 | Samsung 14nm | Samsung 7nm | - |
| 車載總 TDP | ~72 W | ~110 W | - |

（來源：Tesla 官方 v14 Lite release note、Not a Tesla App 拆解、Electrek 報導）

三個數字最刺眼：

1. **相機從 5 MP 降到 1.2 MP**——這不是後製 downsample，是感測器本身。輸入端的資訊量差 4×，你的 vision encoder 拿到的 token 密度直接被閹一半以上。
2. **記憶體頻寬只剩 15%**——這比算力落差還致命。現代 VLA/vision transformer 大量時間卡在 activation shuffle、KV cache 讀寫、attention matrix 搬運，頻寬決定 latency 上限。
3. **8 GB LPDDR4 的位址空間**——v14 完整版要 12.5 GB 位址空間（Not a Tesla App 拆解數字）。單看記憶體佔用要**砍掉 36%（12.5→8 GB）**，但還得留出 OS、buffer、activation cache 的空間，實際 model 端要壓的比例更狠——這也對應到後面 2.3 節提到的「模型主體壓到約 15%」。

如果你做過 Jetson Orin Nano vs Orin AGX、或 Snapdragon 8155 vs 8295 的 model porting，你會馬上懂這種落差意味著什麼——**不是換個 config 檔就能過的事**，是要重架 pipeline。

---

## 二、Tesla 官方講的「蒸餾」：teacher-student in the real physical world

Tesla release note 的原文很直白：

> **Distilled the intelligence from HW4 V14 into HW3. This allows HW3 to directly learn how to handle scenarios using HW4 V14 as a guide.**

翻譯成 ML 工程術語：**HW4 v14 當 teacher network，HW3 v14 Lite 當 student network，做 knowledge distillation**。這不是新技術——Hinton 2015 那篇原論文都十一年了——但 Tesla 這次做法有幾個非典型的地方值得拆：

### 2.1 蒸餾對象是「駕駛行為」而不是「輸出 logits」

傳統 KD 是讓 student 匹配 teacher 的 soft label（output distribution）。Tesla 這裡蒸餾的是**駕駛決策序列**——teacher 在某個場景會怎麼打方向盤、什麼時候剎車、要不要 merge——這比匹配 logits 難很多，因為：

- 駕駛是**時序決策**，不是單步分類，teacher 的整條軌跡要當 supervision。
- Teacher 用 HW4 高解析度相機 + 大 model 判斷得到的行為，student 拿到的是**低解析度輸入**，需要學會「即使我看不清楚，也要做出跟 teacher 一樣的決策」。這是 **cross-modality / cross-resolution distillation**。
- 加了 offline RL：teacher 產生的 rollout 進了 RL loop，student 靠 policy gradient 逼近。這條路線最近在 [XPeng VLA-2 的 implicit token action](xpeng-vla-2-implicit-token-action-2026.md) 也看得到相似做法。

### 2.2 Data-side compensation：用 HW4 fleet 補 HW3 的 blind spot

HW3 相機解析度差 4×，理論上 long-tail 場景會嚴重退化。Tesla 的補救是——**用 HW4 車隊收到的高解析度 corner case，經過 downsample 後當 HW3 的訓練資料**。這樣 student 在訓練時看得到「模擬 HW3 感測條件的 corner case」，而不是只看 HW3 fleet 本身收得到的（本來就有偏差的）資料。

這個 trick 很吃 fleet size——Tesla 有幾百萬台車在路上跑，才玩得起。做 LiDAR 的我們如果只有幾十台原型車，這條路徑基本走不通，只能靠 sim-to-real（可以參考 [Sim-to-Real Gap Cadence 那篇](sim-to-real-gap-cadence-nvidia-2026.md) 的討論）。

### 2.3 模型主體被壓到約 15% 的參數規模

Not a Tesla App 的拆解數字是「約 15% of original size」。這數字剛好跟記憶體頻寬比例一樣，我懷疑不是巧合——Tesla 應該是**先算 latency budget 反推 model size**，而不是先訓好模型再看塞不塞得下。這是 hardware-aware training 的教科書做法。

---

## 三、Quantization：從 fp16 到 int8（甚至 int4 mixed）

Release note 沒明講 quant scheme，但拆解幾個線索：

- 8 GB LPDDR4 要塞下 v14 主體（估計 fp16 下約 5-6 GB）+ activation cache + 中間 feature map + KV cache + OS——**幾乎確定是 int8 主體 + 關鍵層 fp16 mixed**。
- Not a Tesla App 直接用「quantized model」「lower-resolution version of the AI's brain」形容。
- HW3 NPU 對 int8 有硬體加速（原本 2019 年設計 target 就是 int8），fp16 反而是效率次選。

這條路徑跟我之前寫的 [VLA edge compression 那篇](vla-edge-compression-realtime-inference-2026.md) 幾乎一致：**model compression = distillation + quantization + hardware-aware NAS 三個東西一起做**，單獨拆開任何一項都撐不住 40-70% 的壓縮率而不掉分。

有一個沒被回答的技術問題我想留在這：**Tesla 在關鍵的 planning head 上是不是保留了 fp16？** 從業界經驗看，perception 頭可以吃很重的 quant，但 policy head（尤其是連續動作預測）對 quant 相對敏感。如果 Tesla 全鏈路 int8，那 planning 端一定要靠 quant-aware training 才不掉分——這是我最想看到 Tesla AI 團隊寫 tech blog 的細節。

---

## 四、v14 Lite 到底砍掉什麼、留下什麼

這張表是根據 Not a Tesla App 和 Tesla Oracle 拆的官方 release note 整理：

### 保留（HW3 Lite 有）

- Start FSD from Park（停車位起步）
- 自動倒車、自動停車（**新功能**，之前 HW3 沒有）
- Arrival Options（可以選抵達目的地要停在哪：停車場、路邊、車道）
- Standard Speed Profiles（Chill / Average / Assertive）
- 更好的行人互動、cut-in avoidance、traffic signal 判讀
- 較平滑的 lane keeping 與轉向

### 砍掉（HW3 Lite 沒有）

- **Unified ASS/FSD/Robotaxi model**——full v14 一個模型統管三個場景（Smart Summon、Supervised FSD、Robotaxi），HW3 Lite 只保留 Supervised FSD 一個場景
- Standalone Self-Driving App UI
- FSD streaks / mileage tracking
- Arrival Options 地圖 pin 動態調整
- **Mad Max speed profile**（激進駕駛模式）
- Actually Smart Summon 的速度/性能升級

最有意思的是「**Unified model 被拆回單場景 model**」這一點。full v14 的架構優勢就在 multi-task 共享 backbone，砍到單場景等於**把 unified representation 的槓桿收掉了**。這是 model compression 的隱藏成本：不是壓完就沒事，是**表徵能力上限也一起下修**。

另外一個關鍵限制 release note 講得非常直接：

> **HW3 cannot achieve unsupervised autonomous driving like newer systems.**

翻譯：**HW3 不可能升級到 unsupervised FSD / Robotaxi**。這對 2019 年花 $15k 買 FSD 的車主是個明確的了結——Tesla 官方蓋章 HW3 就到 L2 supervised 為止。

---

## 五、對做感知 / embedded AI 工程師的四個啟示

這件事對業界的意義遠超 Tesla 車主圈的騷動。我覺得有四個可以直接套進日常工程決策的教訓：

### 5.1 記憶體頻寬才是 edge inference 的真天花板

多數 spec sheet 只看 TOPS 或參數量，但 HW3 vs HW4 這個對照告訴你——**記憶體頻寬 15% 這個數字比算力 29% 更致命**。做 LiDAR 點雲處理特別要留意這件事：voxelization、sparse conv、KNN neighbor lookup 全部是 memory-bound，不是 compute-bound。買硬體先看 BW/TOPS 比例，別只看 TOPS。

### 5.2 Distillation 是 deployment 的第一原理，不是 optional

過去做 model compression，很多團隊會先 pruning、再 quantization，distillation 當作最後的 fine-tune 手段。這波 Tesla 案例告訴你——**distillation 應該是設計 student network 的起點**，teacher 產出的行為軌跡是 supervision signal 的主體，pruning 和 quant 只是輔助手段。這跟我在 [Physical AI Foundation Models](physical-ai-foundation-models-robotics-2026.md) 提到的「先設計 deployable student、再訓 teacher」是同一套哲學。

### 5.3 Data-side compensation 是 fleet-scale player 的專利

Tesla 用 HW4 fleet 的資料 downsample 後餵給 HW3 student 訓練——這只有百萬級 fleet 的玩家做得到。**做初創、學術、小 fleet 的團隊要清楚這條路走不通**，只能靠 sim-to-real 補血。這是為什麼 [NVIDIA Cosmos、DreamZero 這類 world model 一直被喊做「democratize 感知」](dreamzero-world-action-model-post-vla-2026.md)——它是給沒有 fleet 的人的資料替代方案。

### 5.4 Unified backbone 的壓縮成本

full v14 用一個模型統管三個駕駛場景（Summon / Supervised FSD / Robotaxi）。Lite 版直接砍回單場景。這是**做 unified representation 的隱藏稅**：越是通用的表徵，越沒辦法壓縮到極端。如果你在設計 multi-task VLA，要提前想清楚未來 edge deployment 版本會不會需要拆回單任務。

---

## 六、剩下的技術謎題：我還想看到 Tesla AI 團隊講清楚的三件事

寫完這篇我列了三個未解問題，等後續 Tesla AI Day 或 leaked patent 揭曉：

1. **Planning head 的 quantization 方案**——perception 頭全 int8 沒問題，但駕駛決策是連續動作，掉 0.1% 分數在生死時刻可能就是撞不撞。QAT? mixed precision? 動態 dispatch？
2. **HW3 相機的 super-resolution front-end**——1.2 MP 硬體降頻，理論上先做 learned super-resolution 補回細節再進 vision encoder 會更好，但 SR 本身也要 compute budget，這個 trade-off Tesla 怎麼選的？
3. **v14 Lite 對 corner case 的實測 regression**——release note 全是 marketing 語言，社群拆解通常抓 3-6 個月才有數據。屆時我會再開一篇對比。

---

## 收尾：這不是 Tesla 的故事，是 edge AI 的故事

如果你把 Tesla 名字換成 Waymo、把 HW3 換成上一代 Nvidia Drive Orin，這個故事幾乎一樣會發生——**任何一家車廠都會遇到硬體迭代速度 vs 軟體膨脹速度的張力**，最後只能靠 model compression 補這個 gap。Tesla 這次是把 edge distillation 做到極端邊界的一個公開樣本，值得每個做 physical AI 的人拆來看。

對我自己來說，這篇是個提醒：**做 LiDAR 感知的 SoTA 演算法，如果沒有考慮清楚它能不能在下一代 embedded SoC 跑 30 Hz，那學術意義大於工程意義**。演算法要跟硬體一起設計，不能等訓完再擔心 deployment。

（相關文章：[LiDAR 工業化 2026](lidar-industrialization-2026-innoviz-hesai-volvo.md) 講產業側視角、[On-Sensor Perception](on-sensor-perception-lidar-edge-2026.md) 講另一個把 model 塞到極端邊緣的例子、[Deployment-time Reliability](deployment-time-reliability-runtime-failure-detection-2026.md) 講 compressed model 在部署後怎麼監控 drift。）

---

## 參考來源

- [Tesla starts FSD v14 'Lite' rollout to HW3 cars — Electrek (2026-06-29)](https://electrek.co/2026/06/29/tesla-fsd-v14-lite-hw3-rollout/)
- [Tesla starts the rollout of FSD v14 Lite (2026.20.5.1) for HW3 owners — Tesla Oracle](https://www.teslaoracle.com/2026/06/29/tesla-starts-the-rollout-of-fsd-v14-lite-2026-20-5-1-for-hw3-owners-official-release-notes-rollout-status/)
- [Everything You Need to Know About Tesla FSD V14 Lite — Not a Tesla App](https://www.notateslaapp.com/news/4370/tesla-releases-fsd-v14-lite-for-hw3-cars-everything-you-need-to-know)
- [2026.20.5.1 (FSD V14 Lite) Official Tesla Release Notes](https://www.notateslaapp.com/software-updates/version/2026.20.5.1/release-notes)
- [Tesla Launches FSD V14-Lite: First Impressions — Not a Tesla App](https://www.notateslaapp.com/news/4369/tesla-launches-fsd-v14-lite-first-impressions)
