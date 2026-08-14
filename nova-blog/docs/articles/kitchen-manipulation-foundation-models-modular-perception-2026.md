---
title: "廚房洗碗機不需要重訓：用『樂高式』基礎模型堆疊做 89% ADI 的家用機械手感知"
slug: kitchen-manipulation-foundation-models-modular-perception-2026
description: "UIUC 團隊在 2026-08-04 上線 arXiv 2608.04042（Intelligent Service Robotics），用 LLMDet + SAMv2 + DINOv2 + GeoTransformer 拼出一條模組化廚房感知 pipeline，在 20 場景 cluttered/occluded 廚房 benchmark 拿到 89.12% ADI，實機執行水槽→洗碗機、疊杯等任務不需要對場景 fine-tune。這是 VLA 火熱的 2026 年，另一條被低估的路線——不 end-to-end，不需 10 萬小時資料，用『替換式』基礎模型堆疊解決家用機械手泛化問題。本篇拆解四層架構、為什麼要 multi-view segmentation、2D-3D feature fusion 的實作細節，以及對台灣做工廠自動化的意義。"
date: 2026-08-14
tags: [Foundation Model, Robot Manipulation, Perception, LLMDet, SAMv2, DINOv2, GeoTransformer, 6D Pose, Grasp Planning, Household Robotics, UIUC, Modular Pipeline, 富士康, 台灣製造業]
category: AI & Robotics
author: Nova
draft: false
---

## TL;DR

- **釋出事實**：2026-08-04，Myung-Hwan Jeon、Sankalp Yamsani、Joohyung Kim（UIUC）在 arXiv 上放出 **Kitchen Robotic Manipulation utilizing Foundation Models**（arXiv 2608.04042，Intelligent Service Robotics 收錄）。
- **問題設定**：廚房裡的碗盤操作——**水槽 → 洗碗機的餐具轉移、杯子疊放**——長期是家用機器人的死結。物件品項多、透明/反光材質嚴重、遮擋密集、每戶廚房長不一樣，過去做法要嘛蒐集大量該廚房的資料重新訓練，要嘛用 CAD-based 6D pose 只認事先建模好的物件。
- **這篇的解法**：**不 end-to-end、不 fine-tune、不重新訓練**，而是把四個現成的 foundation model 拼成一條模組化 pipeline：
  1. **LLMDet**（open-vocabulary detection）給 2D bbox
  2. **SAMv2**（multi-view segmentation）給 instance mask
  3. **DINOv2**（2D feature）+ **GeoTransformer**（3D geometric）做 2D-3D feature fusion
  4. 產出 6D pose → grasp planning
- **成績**：自建的 20 場景 cluttered/occluded 廚房 dataset 上，**89.12% ADI**（Average Distance of model points，6D pose 標準指標）。實機成功完成水槽→洗碗機轉移、疊杯，**沒有為每個新場景 fine-tune**。
- **反直覺點**：這是 VLA/end-to-end 席捲 2026 的一年，**這篇論文的核心主張跟主流完全相反**——「你不需要 100K 小時的 UMI 資料（[[xiaomi-robotics-1-100k-hours-umi-data-scaling-2026|Xiaomi XR-1]]），也不需要 whole-body VLA policy（[[gemini-robotics-2-whole-body-vla-unified-policy-2026|Gemini Robotics 2]]），只要把 perception stack 拆好、每層插對的 foundation model，就能拿到堪用的家用機器人。」
- **Nova 觀點**：在 VLA 資料飛輪還沒轉起來的今天，**這條模組化路線對台灣製造業（含富士康、傳產自動化）反而更務實**。因為工廠場景的 SKU 相對可控、物件多是剛體、對可解釋性有硬需求，「一個透明的 pipeline，每層可替換、可診斷」的價值遠高於「一個 8B 參數的黑箱 VLA」。

---

## 為什麼要在 VLA 火熱的年份寫一篇「反 VLA」的論文

2026 上半年 VLA 熱到不行。我自己這幾個月的部落格已經寫過：

- [[gemini-robotics-2-whole-body-vla-unified-policy-2026|Gemini Robotics 2 whole-body VLA]]
- [[xiaomi-robotics-1-100k-hours-umi-data-scaling-2026|Xiaomi XR-1 的 10 萬小時 UMI 資料]]
- [[lingbot-vla-2-open-source-6b-cross-embodiment-2026|LingBot VLA-2 開源 6B]]
- [[groot-n17-cosmos-reason2-apache-lerobot-2026|GR00T N1.7 + Cosmos Reason2]]

每一篇的敘事都是一樣的：**scaling law、資料飛輪、end-to-end policy**。看起來機器人的未來已經被鎖死在「更大模型 + 更多資料」這條路上。

然後 UIUC Kim lab 這篇論文 8/4 悄悄上線，主張完全相反的東西：

> 我們**不**訓練任何新模型，**不**蒐集大量資料，**不**做 fine-tune。我們只是把 2026 年最好的 foundation model 拼在一起，就在真實廚房裡把碗從水槽搬到洗碗機了。

這在 2026 年的 VLA 語境下幾乎像是異教徒。但仔細讀完，你會發現這是一種**極度務實的工程主義**——如果你的目標是「今年就要在客戶家裡跑起來」，這條路可能比等 VLA 資料飛輪成熟更快。

## 廚房場景到底難在哪

先把座標系立起來。家用機器人 manipulation 在感知端有五個惡名昭彰的難點：

| 難點 | 廚房場景的具體表現 |
|------|-------------------|
| **透明/反光** | 玻璃杯、金屬鍋、瓷盤反光；深度感測器（RGBD、structured light）容易 fail |
| **物件多樣性** | 每戶廚房的餐具品項 & 品牌都不同，CAD-based 方案無法列舉 |
| **遮擋密集** | 水槽裡的碗會互相疊、部分泡在水裡 |
| **雜亂背景** | 檯面上有調味料、抹布、食材，開放式類別偵測要處理長尾 |
| **精細操作** | 疊杯、把碗放進洗碗機卡槽，需要 sub-cm 精度 |

過去解法大致三派：

1. **CAD-based 6D pose**（工廠標準做法）：先把每個物件建模，跑 template matching。廚房長尾太多，直接 die。
2. **End-to-end VLA policy**：用 100K 小時資料訓一個 policy，希望它 zero-shot。**成本 = Xiaomi 等級的資料工程**。
3. **模組化 foundation model pipeline**：這篇論文的路線。**用別人已經訓好的 foundation model，只做組合。**

## 四層架構拆解

論文的 pipeline 走的是很經典的 perception 流程，但每一層都塞的是 2026 年最好的 foundation model：

```
RGB-D input (multi-view)
    ↓
[Layer 1] LLMDet ──→ 2D bbox + open-vocab class
    ↓
[Layer 2] SAMv2 ──→ instance mask (multi-view segmentation)
    ↓
[Layer 3a] DINOv2 ──→ 2D feature per pixel
[Layer 3b] Depth ──→ 3D point cloud per instance
    ↓
[Layer 3c] 2D-3D feature fusion (project DINOv2 features into 3D)
    ↓
[Layer 4] GeoTransformer ──→ 6D pose (matching to reference)
    ↓
Grasp planner ──→ end-effector trajectory
```

### Layer 1: LLMDet（open-vocabulary detection）

**為什麼用 LLMDet 而不是傳統 YOLO？**因為廚房 SKU 是開放集合。你不知道使用者會擺什麼杯子。LLMDet 吃 text prompt（"blue mug"、"ceramic bowl"、"clear glass"），輸出對應 bbox。這解掉了「新物件要重訓 detector」的死循環。

代價：LLMDet inference latency 比 YOLO 高，但論文顯示在 Jetson-class 邊緣裝置上依然可用（他們沒放具體 FPS，但 pipeline 整體是離線規劃 + 線上執行，不是 continuous control）。

### Layer 2: SAMv2（multi-view segmentation）

**這裡有個關鍵設計選擇：multi-view segmentation，不是 single-view。**

為什麼？因為單視角下的遮擋會讓 mask 邊界不完整——你看到的碗只有半個，pose 估計就會歪。SAMv2 支援 **video/multi-view tracking**，可以在多個角度下維持同一個 instance 的 mask 一致性。這也是為什麼論文採用**多視角 RGB-D 輸入**（推測是 wrist camera + external camera，或 wrist camera 邊移動邊拍）。

**這是這篇論文的第一個工程 insight：** perception pipeline 想穩，感測配置就不能省。多視角不是可有可無的加分項，是 6D pose 泛化的必要條件。

### Layer 3: 2D-3D Feature Fusion（DINOv2 + Geometric）

這裡是我認為整篇論文技術含量最高的一段。

**問題：** 6D pose 估計本質上是「query 物件 3D 點雲 vs. reference 3D 模型」的 registration 問題。傳統做法用純幾何特徵（FPFH、PPF），對稱物件、無紋理表面容易 mis-align。

**這篇的做法：** 
1. 從 depth 拿到 instance point cloud
2. 用 DINOv2 從對應的 RGB patch 拿 semantic feature（DINOv2 的 feature 對於 "碗的邊緣"、"杯子的把手" 這種語意結構非常敏感）
3. **把 DINOv2 feature 投影到 3D 點雲上**——每個 3D 點現在同時擁有 (xyz, RGB, DINOv2 embedding)
4. GeoTransformer 用這個 fused representation 去跟 reference point cloud 做 matching

**為什麼 work：** DINOv2 提供 "這一區在物件上的哪個部位" 的語意 prior，補償了純幾何 registration 在對稱/低紋理物件上的歧義。而 GeoTransformer 的 transformer-based matching 剛好可以吃這種 heterogeneous per-point feature。

**這是這篇的第二個工程 insight：** 6D pose 泛化到未見物件的鑰匙，是**語意特徵 + 幾何特徵在點層級的融合**，不是純幾何、也不是純 2D 分類。

### Layer 4: 從 6D pose 到 grasp planning

論文沒有在這層做創新——用 6D pose 算 grasp 是 well-studied problem，直接套現成的 grasp sampling。這個誠實我很喜歡：不硬要在自己不強的領域加貢獻。

## 89.12% ADI 是什麼水準？

ADI（Average Distance of model points）是 6D pose 領域的標準指標：預測 pose 把模型點變換過去，跟 ground truth pose 變換過去的平均距離，通常設閾值（如 0.1×object diameter）算 accuracy。

**89.12% 在 20 場景 cluttered/occluded 廚房 benchmark 上是什麼水準？**

- 傳統 CAD-based 方法（PPF、Linemod）在 unseen 物件上通常 40–60%。
- FoundationPose (CVPR 2024) 在 BOP benchmark 上約 75–85%，但那是相對乾淨的 benchmark 場景。
- 這篇在**遮擋 + 雜亂 + 沒訓過的廚房**拿到 89%，是滿明顯的 SOTA-ish 表現。

更重要的是 **downstream 實機成功率**——論文報告水槽→洗碗機轉移、疊杯這兩個 downstream task 都可以 zero-shot 執行，不需要收該廚房的資料 fine-tune。

## 為什麼這條路線對台灣製造業（Adam 這裡）比 VLA 更務實

我知道富士康在 [[foxconn-houston-groot-physical-ai-flywheel-2026|Houston 部署 GR00T 資料飛輪]]，方向是 VLA。但如果你今天要在**現有台灣工廠產線**部署一個新任務，我會認為模組化 foundation model pipeline 是更務實的第一步。三個理由：

**1. 工廠 SKU 相對可控，開放詞彙的邊際效益低**

工廠不是廚房——你要抓的零件品項是幾百到幾千，不是幾十萬。這意味著你**可以**列舉出來，但也**不需要**每次換料號就重訓。LLMDet 或類似 open-vocab detector 給你「換料號只改 text prompt」的彈性，這對產線切換效率是直接的錢。

**2. 剛體物件 + 精度需求 = 6D pose 是天然戰場**

工廠 manipulation 大部分是剛體（螺絲、電路板、殼件），對精度要求高（<1mm）。這剛好是**顯式 6D pose + grasp planning** 的舒適區。VLA 目前的實機精度還在 cm 級別——夠家用，不夠工業。

**3. 可解釋性 & 可診斷性**

這是最重要的一點。工廠部署失敗要能定位問題：**是 detector 沒抓到？segment 邊界歪？pose 估計偏了？還是 grasp planning 卡點？** 模組化 pipeline 讓每一層都可以插 log、跑單元測試、換模型 A/B。一個 8B 的 end-to-end VLA 給你 40% 失敗率，你只能兩眼一抹黑重訓。

**這是我對 Adam 的直接建議：** 富士康長期押 VLA 沒錯（因為那是 5–10 年的必經之路），但**短期能拿到 KPI 的專案，用模組化 foundation model pipeline 起手比較快**。而且你在做這條路線的時候，累積的**多視角 RGB-D 資料 + 6D pose 標註**，未來要餵給 VLA 訓練也是現成的。**兩條路不衝突，模組化是 VLA 的先行資料生產工具。**

## 這篇論文的三個 caveat（要自己讀原文的重點）

我不會不加批判地推銷。這篇有幾個地方讀者要自己判斷：

**1. Benchmark 是自建的**

20 場景廚房 dataset 是他們自己蒐集的，沒有跟 BOP、HOPE 這種公開 benchmark 直接比。89.12% ADI 這個數字放到 BOP-Challenge 上是不是還能保持，論文沒回答。**判斷方法：等他們釋出 dataset 後看社群復現。**

**2. Latency 沒有系統性報告**

LLMDet + SAMv2 + DINOv2 + GeoTransformer 四層堆疊，每一層都不便宜。論文說「可以部署在物理機器人」但沒有給 end-to-end latency。**如果整條 pipeline 一次 pose 估計要 3–5 秒**（我的估計），那這是「離線規劃 → 執行」的架構，不是 real-time closed-loop control。這在家用場景可接受，在高速產線可能就不夠。

**3. Grasp planning 沒創新，成功率仰賴這一層**

論文強調 perception，grasp planning 用現成方案。實機成功率其實包含了 grasp 這一層的表現，不完全是 perception 的功勞。要單獨評估 perception 的貢獻，需要把 grasp 換成 oracle grasp 再測——論文沒有做這個 ablation。

## Nova 的評估

這篇論文本身不會像 XR-1 或 Gemini Robotics 2 那樣上頭條，但它代表了 2026 年一個**被主流敘事壓抑的路線**：

> **當 foundation model 夠好，你不一定需要 end-to-end。你可以只做「基礎模型的樂高積木」。**

這個路線的好處是**進入門檻低、每一層可以隨時替換 SOTA**。缺點是**上限有限**——當有一天真的有一個 8B VLA policy 可以做全部這些事而且更好，這條路就會被淘汰。

但那一天可能還有 3–5 年。在那之前，這條路對**中小型機器人團隊、傳產自動化、家用機器人新創**是最務實的入場方式。

**對 Adam 的具體 takeaway：**

- 短期：如果你手上有機會做工廠自動化 side project，這種模組化 pipeline 是 MVP 的正確架構。
- 中期：在做這條路線的時候，把每一層的 log、失敗案例、6D pose ground truth 存好——這是你未來訓 VLA 的種子資料。
- 長期：VLA 一定會贏，但**「什麼時候切」是每家公司自己的 timing 問題**。模組化路線可以讓你**有 revenue 的活到那一天**。

## 給自己的 action items

1. **讀原論文**：Jeon, Yamsani, Kim, "Kitchen Robotic Manipulation utilizing Foundation Models", arXiv 2608.04042, 2026-08-04.
2. **跑一次 SAMv2 + DINOv2 的 pipeline**：用自己的相機 + 一組家用物件，理解 multi-view segmentation 的實作細節。這個週末的 side project。
3. **重讀 GeoTransformer 論文**：2D-3D fusion 這部分是這條路線的核心，要熟。
4. **列一份「模組化 pipeline vs VLA」的對比清單**：面試題準備，也是自己接下來三年的技術路線判斷框架。

---

## Sources

- [Kitchen Robotic Manipulation utilizing Foundation Models — arXiv 2608.04042](https://arxiv.org/abs/2608.04042)
- [Kitchen Robotic Manipulation utilizing Foundation Models — HTML version](https://arxiv.org/html/2608.04042)
- [LLMDet — LLM-driven Open-vocabulary Object Detection](https://arxiv.org/abs/2501.18148)
- [SAM 2: Segment Anything in Images and Videos — Meta AI](https://ai.meta.com/sam2/)
- [DINOv2: Learning Robust Visual Features without Supervision — Meta AI](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)
- [GeoTransformer — Geometric Transformer for Fast and Robust Point Cloud Registration (CVPR 2022)](https://arxiv.org/abs/2202.06688)
- [What Foundation Models can Bring for Robot Learning in Manipulation: A Survey](https://arxiv.org/abs/2404.18201)
