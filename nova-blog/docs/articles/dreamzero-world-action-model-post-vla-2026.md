---
title: "後 VLA 時代來了？NVIDIA DreamZero 用『先學做夢、再學動作』的 World Action Model，把新任務泛化做到 VLA 兩倍——代價是 9 ZFLOPs"
slug: dreamzero-world-action-model-post-vla-2026
description: "2026 年 NVIDIA 發表 DreamZero——一個 14B 參數的 World Action Model（WAM），把 video diffusion backbone 和 action decoder 用 flow matching 縫進同一顆 transformer，AgiBot 上 62.2% vs VLA baseline 27.4%，新任務泛化超過兩倍。但代價也很誠實：每個 action chunk 590–800 ms（VLA 是 190 ms）、訓練要 9 ZFLOPs、每幀 8000 tokens。這篇拆 WAM 跟 VLA 的本質差別、DreamZero 的 KV cache 不漂移技巧、以及為什麼 WAM+VLA 混血才是下一代機器人 foundation model。"
date: 2026-06-28
tags: [VLA, World Action Model, DreamZero, NVIDIA, Physical AI, Video Diffusion, Flow Matching, 機器人 Foundation Model]
category: AI & Robotics
---

## 前言：VLA 撞到的牆，是「語言到動作」的物理 grounding

過去半年我寫了很多 VLA 的文章——從 [XPeng VLA-2 的 implicit token action](xpeng-vla-2-implicit-token-action-2026.md)、[ACoT-VLA 的 action chain-of-thought](acot-vla-action-chain-of-thought-2026.md)、到 [中國資料 pipeline 怎麼餵 VLA 架構](china-data-pipeline-vla-architecture-2026.md)。一個共同的觀察是：VLA 走到 2026 年中後段，**新任務、新環境的泛化**已經是天花板。

理由其實很單純——VLA 的 backbone 是 vision-language model（VLM），它在「圖像 ↔ 文字」這個維度被預訓練得很深，但對「物體會怎麼動、互動會產生什麼後果」幾乎一無所知。當你只給 VLA 看靜態圖像和語言指令，它必須**從機器人示範資料**裡學會所有的物理 dynamics——而機器人資料是稀缺的、昂貴的、長尾的。

這個瓶頸被 NVIDIA 的研究團隊稱為「語言到動作的 grounding wall」。他們在 2026 年發表 DreamZero，提出一個新的範式回答：**如果機器人 backbone 改成從 video model 開始，而不是從 VLM 開始呢？**

這就是 World Action Model（WAM）的核心命題：**Pretrained to Imagine, Fine-Tuned to Act**——先讓模型看遍網路影片學會「世界會怎麼變」，再學「我要怎麼動才能讓世界這樣變」。DreamZero 是這個範式的第一個高品質實作，也是目前 RoboArena 跟 MolmoSpaces 上的雙料第一。

但這篇不是吹捧文。WAM 的代價也很誠實——9 ZFLOPs 的訓練 budget、每個 action chunk 590–800 ms 的延遲、每幀 8000 tokens 的記憶體開銷。這篇想拆三件事：WAM 跟 VLA 的本質差別在哪、DreamZero 的工程亮點（特別是那個 KV cache 不漂移的技巧）、以及為什麼最終的答案可能不是 WAM 取代 VLA，而是兩者混血。

---

## 一、WAM 跟 VLA 的本質差別：不是換個 backbone，是換個 prior

先把兩個範式的差別擺清楚：

| 維度 | VLA (Vision-Language-Action) | WAM (World Action Model) |
|------|------------------------------|--------------------------|
| Backbone 來源 | VLM（圖像-文字預訓練） | 影片 diffusion 模型（影片-文字預訓練） |
| Backbone 規模 | 1B–7B 為主 | 14B（DreamZero）甚至更大 |
| Prior 學到的東西 | 視覺概念 + 語言對應 | 視覺概念 + 時序 dynamics + 語言條件變化 |
| 機器人資料需求 | 大量、多 embodiment | 少量、可 30 分鐘新 embodiment 適配 |
| Token 開銷（每幀） | ~500–700 | ~8,000 |
| 推理延遲（每 action chunk） | ~190 ms | 590–800 ms（DreamZero-Flash 可壓到 150 ms） |
| 訓練 FLOPs（fine-tune） | 0.5–1 ZFLOPs | ~9 ZFLOPs（DreamZero） |
| 影片預訓練成本 | N/A | 50+ ZFLOPs |
| AgiBot 任務進度（seen tasks） | 27.4%（best baseline） | 62.2%（DreamZero） |
| AgiBot 任務進度（unseen tasks） | 不到 20% | 39.5% |
| DROID 新動詞泛化 | 25–32% | 49% |

最關鍵的差別不在「模型大兩倍」，而在 **prior 的維度**。

**VLM 學的是空間（一張圖）。** 你給它一張「廚房裡的杯子」圖，它知道那是杯子、知道在廚房，但它沒看過「杯子被推會傾倒、傾倒會灑水、水會流到桌邊」這條時序鏈。所有的「動作 → 後果」必須從機器人示範裡學。

**Video model 學的是時空（一段影片）。** 它在預訓練的時候就看過幾億段「東西被推、被拿、被放」的影片，它已經有「世界會怎麼變」的 prior。當你用機器人資料 fine-tune 它去輸出動作，它只需要學「我這個關節輸出怎麼對應到我預期的 visual change」——這比從零學整個 dynamics 容易得多。

NVIDIA 給這個對比一個很漂亮的口號：**Pretrained to Imagine, Fine-Tuned to Act**。Imagine 是免費的——影片無限、自監督；Act 是昂貴的——需要實機資料。把昂貴的部分壓縮到最小，是 WAM 對機器人資料稀缺問題的回應。

這也解釋了一個 DreamZero 論文裡的關鍵數字：**新 embodiment 只需要 30 分鐘的 play data 就能適配**。30 分鐘對學 dynamics 完全不夠，但對學「我的關節 → 視覺變化」的對應關係夠了——因為 dynamics 已經在 video pretraining 階段學完了。

---

## 二、DreamZero 的工程亮點：joint denoising、flow matching、KV cache 不漂移

DreamZero 不是第一個 WAM——之前的工作（LingBot-VA、Fast-WAM）已經探索過用 video backbone 做機器人 policy。但 DreamZero 是把這條路走得最深、效果最強的一個。它的工程設計有四個關鍵點值得單獨拆：

### 1. 單一 transformer 同時 denoise video 和 action

WAM 的早期做法有兩條：
- **Inverse dynamics**：先用 video model 生成「想看到的未來幀」，再用一個 inverse model 反推「我要做什麼動作才能達到這個未來」。
- **Joint prediction**：在同一個 forward pass 裡同時輸出 video tokens 和 action tokens。

DreamZero 選 joint prediction，而且用 **flow matching** 在同一顆 14B transformer 裡 denoise video 和 action。這個選擇的代價是訓練比較難（兩個 modality 的 loss 必須平衡），但好處很大——video 和 action 的表徵在每一層 attention 都互相影響，模型學到的是「我這個動作會產生什麼視覺結果」而不是「先想像視覺結果再 hack 出動作」。

論文的 ablation 顯示，joint prediction 比 inverse dynamics 在 unseen tasks 上強約 8–12 個百分點。

### 2. Wan 14B video diffusion 當 backbone

DreamZero 是從 **Wan 14B**（NVIDIA 內部的影片 diffusion 模型）做 continual pretraining，不是從零訓練。這代表 50+ ZFLOPs 的影片預訓練 budget 是「沿用」的，DreamZero 自己只多花 9 ZFLOPs 在機器人資料的 fine-tune。

對工程師的啟示是：**WAM 的時代，影片 backbone 是新的「ImageNet」**。誰擁有最強的影片預訓練（Wan、Sora、Cosmos、Veo），誰就有最好的機器人 foundation model 起點。這個競爭其實已經悄悄地讓 NVIDIA、Google、字節在影片生成軍備競賽裡有了「下游有去處」的合理性。

### 3. KV cache 替換：消除自迴歸漂移

這個技巧我覺得最巧妙，也是 DreamZero 能達到 7Hz closed-loop 的關鍵。

WAM 是 autoregressive 的——每次預測一個 action chunk + 對應的 future frames，然後把這些 frames 加進 context、預測下一個 chunk。問題是預測的 frames 跟真實 frames 永遠會有差距，這個誤差會在 context 裡累積，幾秒鐘後模型會「飄」進它自己幻想出來的世界。

DreamZero 的解法：**每執行完一個 action chunk，就把 KV cache 裡的『預測 frame』替換成『真實 frame』**。具體做法是——
- 推理時，模型先生成未來 N 幀的 video tokens + 對應的 action tokens
- 機器人執行 action chunk，過程中真實 camera 拍到 N 幀真實畫面
- 下一次預測時，把這 N 幀真實畫面的 token 寫回 KV cache，覆蓋之前的預測 token
- 模型繼續從這個「校正後的 context」往前推

這個設計**同時解決了兩個問題**：
1. **漂移消失**——context 裡永遠是真實觀測，不是幻想。
2. **推理效率**——KV cache 可以重用，每次只需要 denoise 新的 N 幀，不需要重跑整個 context。

論文裡的數字是 **150ms / action chunk**（DreamZero-Flash 版本），對應到 **7Hz 控制頻率**——對 manipulation 任務勉強夠用，對需要 100Hz+ 控制的 locomotion 還不行，但已經不再是「實驗室才能跑」的水準。

### 4. DreamZero-Flash：38× 推理加速

標準 DreamZero 用 16 步 diffusion，DreamZero-Flash 把它壓到 **1 步 diffusion** + 一系列 system 層級的優化（KV cache 量化、attention kernel 改寫、輸出 head 並行化），總共做到 **38× 加速**。

這跟之前 Decart Oasis 3 走的路類似——[world model 從學術上線量產](decart-oasis3-realtime-world-model-production-2026.md) 的核心戰場已經不在「模型誰更聰明」，而在「推理工程能不能榨乾 GPU」。NVIDIA 在這條路上有母公司優勢，因為他們能同時動 model、CUDA kernel、TensorRT serving 三層。

---

## 三、為什麼最終答案是 WAM + VLA 混血，不是 WAM 取代 VLA

DreamZero 的成績很漂亮，但有幾個 WAM 範式仍然沒打下來的點，這些點正好是 VLA 比較強的地方：

**1. 推理延遲。** 即使 DreamZero-Flash 壓到 150ms，仍比 Pi-0.5 的 190ms（一般 VLA）多了一個影片 denoise 的 overhead。對需要高頻反饋的任務（contact-rich manipulation、locomotion）來說，每多 100ms 都是傷害。

**2. 記憶體成本。** WAM 每幀 8,000 tokens，VLA 才 500–700。在邊緣硬體（Jetson Thor 級）上跑 WAM 還是很吃緊，[VLA edge compression](vla-edge-compression-realtime-inference-2026.md) 那條路 WAM 還沒走通。

**3. 語言精準度。** WAM 的語言條件來自影片預訓練裡的「caption → video」對齊，這比 VLM 的「指令 → 動作」對齊弱。複雜的條件指令（「如果杯子是空的就拿到水槽，不然放回桌上」）VLA 跑得更穩。

**4. 訓練成本門檻。** 9 ZFLOPs 的 fine-tune 已經不是小團隊玩得起的——這是 50–100 張 H100 跑數週的等級。VLA 的 0.5–1 ZFLOPs 大概是 8 張 H100 跑一週，差距一個量級。

所以業界已經開始往「混血」走：
- **Pi-0.7**（Physical Intelligence）把 world model 的 subgoal 整合進 VLA 框架——主 policy 還是 VLM 出身，但加了一個小型 video predictor 當輔助。
- **Being-H0.7**（中國團隊）用 latent planning + VLA pretraining——把 WAM 的優勢（dynamics prior）壓進 latent space，避免在推理時跑全套 video generation。
- **DreamZero-Lite**（NVIDIA 後續工作的早期 leak）用 representation-only 策略——video backbone 只當 feature extractor，不真的生成影片，把推理成本拉回 VLA 等級。

NVIDIA Technical Blog 自己在文章結尾就承認：「下一代機器人 foundation model 很可能是 WAM+VLA hybrid」。這個立場其實很誠實——NVIDIA 是 WAM 的最大推手，但他們知道單靠 WAM 拿不下整個產業。

---

## 四、對做感知/機器人/嵌入式的工程師意味著什麼

DreamZero 不只是又一個 SOTA 論文，它有三個對日常工作的影響值得注意：

### 1. 影片預訓練變成新的「免費資料」入口

過去機器人團隊買資料只能買「機器人 teleop demo」——昂貴、稀缺、長尾。WAM 範式打開的可能是：**你可以用 YouTube、Ego4D、Open X-Embodiment 的影片資料當第一階段的『dynamics 教材』**，然後只用少量機器人資料做最後的動作對齊。這對資料預算有限的中型公司（不是 OpenAI、不是 Figure）是真的解。

實作層面：如果你在帶機器人團隊，現在可能要開始評估「我們的下一代 backbone 要不要從 Wan / Cosmos / Sora 衍生出來」，而不是繼續從 LLaVA、Qwen-VL 開始。

### 2. KV cache 工程化會變成 robotics infra 的標配

DreamZero 的 KV cache 替換不是新發明（LLM serving 領域早就有 KV cache 壓縮、量化、reuse 的研究），但把它用到「機器人 closed-loop 控制」是新的。對 robotics infra 工程師，這意味著：
- vLLM、SGLang 這類 LLM serving 框架的 KV cache 管理機制，可以直接借鑑到 robotics serving。
- 自家的 inference stack 如果還是「每次 forward pass 從頭跑」，未來 2 年會被淘汰。

我自己在做 ROS2 + LiDAR 的時候，能想像的應用是——多模態 perception pipeline（LiDAR + camera + depth）也可以用類似的 KV cache 設計避免重複編碼前幾幀的 sensor input。這條路還沒有人深做。

### 3. 「Imagination layer」可能成為下一代 perception stack 的標準元件

更長期的影響：當機器人有了 video 預測能力，**它就有了 short-horizon 的「想像層」**。這層東西可以用來：
- 預判碰撞——在動作執行前先 imagine 一遍未來 0.5 秒，看會不會撞。
- 多模態檢驗——LiDAR 看到的東西跟 imagine 出來的影像對不上？這是 [out-of-distribution 偵測](deployment-time-reliability-runtime-failure-detection-2026.md) 的新路。
- 主動探索——RL agent 用 imagination 評估「往哪個方向走 information gain 最大」，不用真的跑過去試。

這條路 NVIDIA 在 [GR00T N1.6 dual-system](groot-n16-dual-system-cosmos-reason-2026.md) 那邊已經有雛形，但 DreamZero 把它的核心元件「能即時 imagine 的 video model」做到了量產等級。

---

## 結語：WAM 不是 VLA 的取代，是 VLA 的童年缺的那塊

我個人對 WAM 的態度是——它不會是「VLA 的下一代」，它會是「機器人 foundation model 一直缺少的 dynamics prior」。

打個比方：VLA 像是一個會說多國語言、認得所有物體的小孩，但他從來沒看過東西怎麼動——你叫他「把杯子拿起來」，他必須現場一步步試錯學會。WAM 則是先讓他看了幾億小時的影片，知道「拿杯子大概是怎麼回事」，再讓他開始試手——這個小孩學東西明顯比較快。

但 WAM 自己也不是萬靈丹。他想像力豐富，但反應慢、耗能高、複雜指令還不如純語言派的同學。最後的答案大概率是兩個小孩一起組隊——一個負責 dynamics imagination、一個負責 language reasoning、用 latent space 串起來。

對工程師而言，2026 下半年到 2027 是關鍵窗口——這段時間市場會湧現一堆「WAM-flavored」的開源模型（Hugging Face 上已經看到一些跟 DreamZero 同期的小型 WAM），然後業界會快速收斂到 hybrid 架構。如果你做的是 robotics foundation model、或下游應用層（VLA fine-tune、感知融合），現在就應該開始評估自己的技術棧能不能接 video backbone。

WAM 的時代不會立刻到，但它已經開始了。DreamZero 是這個轉折點的第一個高品質實作，值得記在心裡。

---

## 參考來源

- [DreamZero 官方專案頁](https://dreamzero0.github.io/)
- [DreamZero 論文 (arXiv 2602.15922)](https://arxiv.org/pdf/2602.15922)
- [NVIDIA Technical Blog: Pretrained to Imagine, Fine-Tuned to Act — The Rise of World-Action Models](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)
- [NVIDIA Glossary: What is a World Action Model?](https://www.nvidia.com/en-us/glossary/world-action-model/)
- [NVIDIA Releases New Physical AI Models as Global Partners Unveil Next-Generation Robots](https://nvidianews.nvidia.com/news/nvidia-releases-new-physical-ai-models-as-global-partners-unveil-next-generation-robots)
- [36kr: NVIDIA 兩篇論文揭示後 VLA 時代的具身智能新範式](https://eu.36kr.com/en/p/3678616267874953)
