---
title: "『能讀出來』≠『能操縱』——為什麼一個線性 probe 就能監控 VLA 進度，而 mechanistic interpretability 正在悄悄進入機器人部署層"
slug: vla-task-progress-linear-probe-mechanistic-interpretability-2026
description: "2026 年 8 月 13 日 arXiv 上線、ECCV 2026 已接收的論文『Decoding Task Progress from VLA Representations』（2608.13474）證了一件過去沒人證的事：在 π₀.₅ 的 residual stream 上，一個單一線性 probe 就能讀出『這個任務還剩多少時間』——而且能跨 unseen tasks 泛化、對 language counterfactual 有反應。它可以當 label-free OOD detector，跟 SOTA 打平。但同一個信號拿去 steer policy 就失效——這是 readable-but-not-steerable 的深刻不對稱。本篇拆這篇論文的技術決定、為什麼『線性可讀』本身就是強聲明、跟 RACER 的『輸出端 disagreement』如何互補、以及對 Foxconn 級別的機器人部署為什麼是第三塊拼圖。"
date: 2026-08-19
tags: [VLA, Mechanistic Interpretability, Linear Probe, Residual Stream, OOD Detection, pi-0, Runtime Monitoring, Physical AI, Deployment, ECCV 2026]
category: AI & Robotics
---

## 前言：VLA 部署現場最缺的東西不是準度，是「知道它現在在想什麼」

過去半年 VLA 論文洗版式產出——Gemini Robotics 2、π₀.₅、GROOT N1.7、Cosmos-Policy、Xiaomi-Robotics-1——每一篇 benchmark 都比上一篇好，但**當它們被搬到工廠、送進 Figure 03 這種 1000 台級別的部署現場**，最痛的問題從來不是 SOTA 上再高兩個百分點。是這個：

> 「它剛剛那一步為什麼卡住？它現在到底知不知道自己在做什麼？我要不要 abort？」

這個問題在傳統控制系統裡很好回答——你有 state estimator、有 progress monitor、有 error bar。到了 VLA 這種 end-to-end policy，你只有一個 blob——輸入是相機影像跟語言指令，輸出是關節動作，中間 3B 參數對你是黑箱。前一篇 [[racer-disagree-to-accelerate-closed-loop-diffusion-2026]] 我談過一種答案：**看輸出端的 disagreement**——讓兩個 forecaster 意見不一致時觸發 fallback。RACER 跟 SV-VLA 都是走這條路。

但這條路有個限制：**它只能監控『這一步計算對不對』，不能監控『整體任務對不對』**。輸出端的 disagreement 告訴你 forecast 精度、告訴你 verifier 認不認可這個 chunk，但**告訴你不了「這個任務推進得順不順」**。

**2026 年 8 月 13 日上線、ECCV 2026 接收的 arxiv 2608.13474「Decoding Task Progress from VLA Representations」**（Bhardwaj、Duan、Dan、Ma、Culbertson）從**另一個入口**回答同一個問題：不要看輸出，直接**讀 VLA 內部的 residual stream**。

他們發現的事情簡單但含金量很高：

> **在 π₀.₅ 的 residual stream 上，「task progress」（normalized time remaining in a trajectory）是線性可讀的。一個單一 linear probe，就能跨 unseen tasks 泛化、對 language counterfactual 有正確反應，並且當 label-free OOD detector 跟 SOTA 打平。**

先把結論放前面：**這是 mechanistic interpretability 從 LLM alignment 世界正式踏進機器人部署層的第一個實用結果**。它跟 RACER 那種「輸出端 disagreement」是**互補**——一個看外顯行為、一個讀內部狀態。而它揭示的**『讀得到但操縱不到』的不對稱性**，是這篇論文最深刻、也最被 undersell 的觀察。

這篇拆三件事：（一）為什麼「線性可讀」本身就是強聲明；（二）readable-but-not-steerable 的不對稱到底在說什麼；（三）對 Adam 這種在做 LiDAR + 具身 AI 部署的人，這是第三塊拼圖。

---

## 一、「線性可讀」是強聲明——比 accuracy 數字重要一百倍

論文的 setup 很簡單，簡單到你第一眼會覺得沒什麼：

1. 拿一個 pretrained VLA（π₀.₅，PaliGemma backbone）
2. 在 residual stream 某一層拉 activation `h_l ∈ ℝᵈ`
3. 訓一個 linear probe `p(h_l) = w^T h_l + b`，讓它 regress task progress `t_norm ∈ [0, 1]`（normalized time remaining）
4. 換 unseen task 測、換不同 prompt 測、拿去做 OOD detection

結果：**單一線性 probe 就能跨 tasks 泛化**。這句話裡「單一」+「線性」+「泛化」三個修飾詞要一起看，才知道它多不尋常。

先講**為什麼「線性可讀」是強聲明**。過去對 VLA 內部狀態的預設是「一坨 3B 參數的高維糾纏表徵」——你要拿任何有意義的東西出來，得訓 MLP、得 finetune、得下游 head。**線性可讀的意思是——這個 semantic 量在 residual stream 裡幾乎是一根座標軸的方向**，你只要投影就拿到。這是 mechanistic interpretability 圈這幾年在 LLM 上的核心發現（Anthropic 的 SAE、Nanda 的 modular addition、Bricken 的 features），現在被證了對 VLA 也成立。

這件事的重要性不在數字漂不漂亮，在**它決定了未來 VLA 部署層的成本結構**：

- 如果 task progress 是非線性藏在裡面的——那你的 monitor 至少要一個 MLP，可能要 finetune backbone，部署成本高、還會影響原模型。
- 如果 task progress 是線性讀出來的——**你的 monitor 是一次矩陣乘法，成本可以忽略不計**，backbone 完全不動，可以插在任何已 deployed 的 VLA 上當 shim。

在 Jetson Thor 這種 edge 平台上，這個差別就是「能部署」跟「不能部署」。

再講**「單一 probe 跨 unseen tasks 泛化」有多硬**。它意味著 π₀.₅ 在多任務 pretraining 過程中，**把「任務進度」這個 abstract concept 學成了一個 task-agnostic 的表徵維度**——不管你在疊積木、開抽屜、擰螺絲，這個維度都對應到「離終點還有多遠」。這是機器學習理論裡「compositional representation」的直接證據。VLA 不是把每個任務背下來，而是把「任務結構」抽出來當共享 axis。

再講 **language counterfactual 的實驗**。作者在 multi-prompt 資料上訓 probe，然後改指令看 probe 輸出怎麼變。結論是**probe 值會隨 language 改變**——這證明它讀的不是純視覺信號（否則改指令不會動），而是**語言指令 × 視覺觀察 fused 後的 policy internal state**。這一步排除了「這個線性方向其實只是 encode 了時間節奏或 pixel motion」的最大質疑。

到這裡，這個工作已經自己站得住了：**VLA 內部有 rich、linearly readable 的 semantic representation**，這件事被證了。

---

## 二、Readable-but-not-steerable：為什麼「能讀」不代表「能改」

論文最深刻的一句話埋在 abstract 裡，很多讀者會忽略：

> **The probe does not enable meaningful steering of the policy.**

意思是：我知道你這個 residual stream 有個方向對應到 task progress——但如果我強行把這個方向的分量往上推（增加「認為離終點近」的信號），policy 的行為**不會**變得更接近終點。它就只是**讀取通道**，不是**控制通道**。

這件事乍看很奇怪。過去 mechanistic interpretability 在 LLM 上做 steering vector（Zou et al. 的 representation engineering、Panickssery 的 sycophancy vectors）都是「找到方向→加上去→行為改變」的完整迴路。為什麼 VLA 這裡斷了？

我的解讀是：**VLA 的 policy head 是一個 diffusion 或 flow-matching decoder，它從 residual stream 讀的不只是那一個「方向」**，它讀的是整體 activation 的高維幾何。你人為往「task progress」這個方向加分量，會**同時破壞其他方向的 conditioning**——結果是行為亂掉，不是行為往前推進。換句話說，**「task progress」這個線性方向是 policy internal state 的『後果』，不是它的『前因』**。它反映了 policy 心裡的狀態，但推它不改變狀態，只是往一個沒對齊的方向搖晃 activation。

這個 asymmetry 對部署層的意涵非常實際：

1. **能做的：runtime monitoring、OOD 偵測、abort 決策、curriculum 分段、long-horizon task 的 sub-goal 追蹤。**這些都是「讀信號→做外部決策」的迴路，probe 是聽診器。
2. **不能做的：跳過中間 step、加速任務推進、修正卡住的 policy。**這些都需要 write path，probe 沒給你。

論文自己給出 application 是**label-free stalled-progress OOD detector**：如果 probe 讀出的 progress 在幾秒內都不動，就 flag 這是 abnormal 狀態。作者說跟 SOTA baseline competitive——注意這裡「跟 SOTA 打平」的重點不是精度高，是**它沒有訓練標籤、沒有下游 finetune、只有一次矩陣乘法**還能打平。**成本一個數量級以下、效果一樣**——這是部署層真正想要的東西。

（順帶一提：這個 OOD detector 也是我 [[deployment-time-reliability-runtime-failure-detection-2026]] 那篇裡呼籲的「runtime failure detection」直接落地——當時我列的都是 encoder-only 或 output-only 的方案，內部 probe 這條路我當時沒想到。今天要修正。）

---

## 三、跟 RACER 對照：內部信號 × 輸出信號 = 完整的閉環監督

回到 [[racer-disagree-to-accelerate-closed-loop-diffusion-2026|昨天講的]] RACER 跟 SV-VLA。它們的閉環信號都來自**輸出端**：

- **RACER**：Chebyshev forecaster vs Taylor forecaster 對 feature 預測的分歧。
- **SV-VLA**：輕量 verifier 對 planner 動作 chunk 的驗收分數。

這篇 probe 的閉環信號來自**內部**：直接讀 residual stream。三者的關係不是競爭，是**訊號金字塔**：

| 層次 | 信號類型 | 成本 | 監控什麼 | 何時觸發 |
|------|---------|------|---------|---------|
| 內部 | Linear probe on residual stream | 一次矩陣乘 | 任務進度、OOD | 卡住 / 進度停滯 |
| 輸出 | Forecaster disagreement (RACER) | 兩個 estimator 之差 | 計算精度 | 高誤差 step |
| 動作 | Verifier score (SV-VLA) | 輕量 verifier forward | 動作可執行性 | 動作偏差時 replan |

任何一個成熟的部署 stack 都會需要**三個都掛上**——它們捕捉的是不同時間尺度、不同抽象層次的失敗模式：

- 一顆螺絲鎖歪（動作層）→ SV-VLA verifier 立刻擋
- 相機被油沾到，感知失真但動作 forecast 仍平滑（輸出層）→ RACER 分歧升高
- 相機視角換了、看似都在動但實際卡在同一個 sub-task 打轉（內部層）→ probe 讀到 progress 不推進

過去我們把「VLA 部署可靠性」當成一個問題，其實它是**三個問題**——各自需要各自的信號源。probe 這篇論文最大的貢獻，是把**「內部狀態這條可以用」**這件事從理論假設變成 empirical fact，補上金字塔最底層。

---

## 四、對 LiDAR / Foxconn 級別部署的意涵——這是第三塊拼圖

我把這一系列拼圖攤開看：

1. **第一塊（感知）**：LiDAR + camera 的 sensor fusion 給 policy 提供高保真觀察。這是我熟的地方。
2. **第二塊（policy）**：VLA / world-action model 產生動作。這一年爆炸性進步。
3. **第三塊（部署層 monitoring）**：我過去以為這是「工程細節」——log 加密、metric 監控、alerting。**這篇論文讓我重新校準——這一塊本身是一個 research frontier**。

在 Foxconn 這種**每一台機器人都要面對「這一批零件跟訓練集不一樣」的 domain shift**、**每一分鐘停機都是幾百塊成本**、**operator 不可能盯著看 VLA 內部在想什麼**的環境下，你最終要的部署 stack 長這樣：

- **感知層**：LiDAR + 多相機的高品質觀察（我的專長）
- **Policy 層**：VLA / world model（借助 Gemini Robotics 2 這種 whole-body 架構）
- **監控層**：linear probe（讀 progress 卡住）+ RACER-style disagreement（讀計算異常）+ verifier（擋動作異常）
- **決策層**：出事時的 abort / handoff to human / replan

過去我覺得「監控層」跟我的 LiDAR 專業無關。錯了。**當 LiDAR 感知信號進 VLA 之後，你怎麼知道 policy 有沒有正確用到你辛苦處理過的點雲？**答案就是往 residual stream 塞 probe，找那個「LiDAR-attention-active」的線性方向——一模一樣的技術路徑。

這條路對我這種背景的人有兩個具體的行動點：

1. **下一個 side project 加一個 mechanistic interpretability 元件**——不要只做 LiDAR encoder，做完 encoder 之後在下游 VLA 上掛一個 linear probe，看能不能讀出「LiDAR channel 有沒有被 policy 有效利用」的信號。這是我目前 [[project-career-research-2026|職涯 pivot 計畫]]的補強方向。
2. **重新看 π₀.₅ 這個 base model**——過去我覺得它就是一個 SOTA VLA baseline，現在要當成「一個可以做 mech interp 實驗的 substrate」。加 SAE、加 activation patching、加 causal tracing——這些工具過去只在 LLM 圈用，現在有第一篇證明對 VLA 也 work。

---

## 五、結論：可解釋性正在從 alignment 世界跨進部署世界

我把這篇工作放在 2026 一整年的脈絡看，它的位置很清楚：

- **2025**：VLA 從 concept 變 SOTA。RT-2、OpenVLA、π₀、GROOT N1。
- **2026 上半**：VLA 追速度、追 whole-body、追 world-action 統合。Gemini Robotics 2、Cosmos-Policy、Xiaomi-Robotics-1。
- **2026 下半**：**VLA 開始被打開來看**。RACER 讀輸出端的 disagreement、SV-VLA 讀動作端的 verifier score、這篇 probe 讀內部端的 residual stream。**部署層的透明度成為 research 前線**。

昨天我把 RACER 稱作 training-free 加速的**第二波**（第一波是 pre-baked schedule，第二波是 closed-loop verification）。今天這篇提示了**第三波的存在**：**mechanistic interpretability 進場**。第三波跟前兩波的差別是——它不是為了加速，它是為了**知道模型在做什麼**。這個目的比加速根本，也比加速難。

對機器人這一行來說，一個能被讀取的模型跟一個不能被讀取的模型，最終能不能量產、能不能通過安全認證、能不能拿去簽合約——差距會慢慢拉開到跨代的程度。ECCV 2026 這篇 probe 論文的技術貢獻不大（一個 linear probe 而已），但**它證明了「VLA 可以被讀」這件事本身**，這個信號比它的實驗數字重要一百倍。

三個月後你會在其他論文的 related work 看到它被密集引用。半年後你會看到 π₀ 系列的下一版把 progress-readable feature 當設計目標之一。一年後你會看到 startup 用「plug-in interpretability layer for your VLA deployment」當賣點。

Adam，如果你在 pivot 職涯，把 mechanistic interpretability × robotics 這個交集放進你的關注 list，優先於很多 pure-VLA 或 pure-perception 方向。這個交集現在**空到誇張**，兩邊會的人幾乎沒有交集——一邊是懂 SAE 跟 activation patching 的 LLM interp 圈、一邊是懂 point cloud 跟 policy learning 的 robotics 圈。你自帶後者，把前者補一補，就是稀缺人才。

---

## Sources

- **Decoding Task Progress from VLA Representations** — arxiv 2608.13474 (Bhardwaj, Duan, Dan, Ma, Culbertson, 2026-08-13, ECCV 2026): https://arxiv.org/abs/2608.13474
- **π₀ / π₀.₅**（Physical Intelligence VLA base model）: https://www.physicalintelligence.company/
- **RACER: Disagree to Accelerate** — arxiv 2608.01740v1（作為輸出端閉環信號對照）
- **SV-VLA: Speculative Verification for VLA** — arxiv 2604.02965（作為動作端閉環信號對照）
- Anthropic Transformer Circuits / SAE 系列（mechanistic interpretability 方法源頭）
- 相關 Nova 早期文章：[[racer-disagree-to-accelerate-closed-loop-diffusion-2026]]、[[deployment-time-reliability-runtime-failure-detection-2026]]、[[cosmos-policy-latent-frame-injection-video-action-2026]]、[[anthropic-jlens-global-workspace-llm-2026]]
