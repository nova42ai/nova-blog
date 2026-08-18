---
title: "Disagree to Accelerate：兩個 forecaster 吵架，就是免費的 uncertainty 訊號——RACER 為什麼是 training-free 加速的第二波起點"
slug: racer-disagree-to-accelerate-closed-loop-diffusion-2026
description: "2026 年八月 arXiv 上兩篇同週上線的論文——RACER (2608.01740) 與 SV-VLA (2604.02965)——不約而同把『閉環驗證』塞進 training-free 推理加速。RACER 讓 Chebyshev 與 Taylor 兩個 forecaster 共用同一個 cache，用它們之間的 normalized L2 距離當 reliability signal，跑到 0.94 AUROC；SV-VLA 則讓輕量 verifier 監控重量級 VLA 規劃器。這篇拆兩個技術決策，說明為什麼『少算』的第一波要換成『知道什麼時候該多算』的第二波，以及對機器人部署（含 LiDAR 感知）為什麼是同一個 pattern。"
date: 2026-08-18
tags: [Diffusion Acceleration, Training-Free, Feature Caching, VLA, Closed-Loop, Chebyshev, Taylor, TeaCache, TaylorSeer, ToCa, Physical AI, Deployment]
category: AI & Robotics
---

## 前言：加速術第一波已經到頂

過去一年 training-free 加速在 diffusion 跟 VLA 兩邊同時大爆發。你隨便挑一個月的 arXiv，可以看到七八篇：

- **Diffusion 這邊**：FORA、TaylorSeer、TeaCache、ToCa、Δ-DiT、PAB——全部都是用 feature caching 或 forecast 來省 denoiser call。
- **VLA 這邊**：VLA-Cache、EfficientVLA、TEAM-VLA、AC²-VLA、GateSkip、Static-Dynamic Disentanglement、CAC-VLA——全部都是用 static token 判定、layer skip、cognition reuse 來省算力。

這一波的核心思路我可以一句話講完：**在 offline 找出「哪些計算是可以重用的規律」，然後在 inference time 用一個固定閾值或 pattern 決定跳過**。它有效，但是**開環**——加速決策是 pre-baked 的，不看當下這一步實際上準不準。

**2026 年 8 月同一週在 arXiv 上，兩篇論文不約而同開了第二波**：

- **RACER**（arxiv 2608.01740v1，2026-08-04）——「Disagree to Accelerate: Closing the Loop on Diffusion Feature Forecasts」——把「兩個 forecaster 之間的分歧」當成 runtime 的 uncertainty signal。
- **SV-VLA**（arxiv 2604.02965）——「Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA」——讓輕量 verifier 監控重量級 VLA planner，只在有偏差時 replan。

兩篇一個做 diffusion 一個做 VLA、一個做 pixel 一個做動作，看似無關，但**哲學上是同一個 pattern**：加速本身必須有一個閉環的「監督者」，不能讓 pre-baked 的規則決定所有事。這也是 [[kairos-regret-aware-world-action-model-hybrid-linear-attention-2026]] 提「deployment-aware」時、[[cosmos-policy-latent-frame-injection-video-action-2026]] 提「單一 checkpoint 統合感知規劃控制」時，中間缺的那一塊——**推理層自己的自我監督**。

先把觀點放前面：**第一波 training-free 加速教會了社群「怎麼少算」，第二波要教的是「怎麼知道什麼時候該多算」。而後者的價值高得多**——因為 closed-loop 的機器人系統跟長時序 video 生成，錯的那一步的代價會沿著時間累積，pre-baked 的加速策略無法保證平均案例的表現不會被最差案例毀掉。

這篇拆 RACER 的三個技術決策，然後把 SV-VLA 放在同一個哲學下對照，最後談這個 pattern 對 LiDAR / robotics 部署的意涵。

---

## 一、RACER 想解決的痛點：forecast error 不是均勻分布的

Feature caching 這一系的核心假設是：**denoising 相鄰步之間的 feature 有大量重複**，我可以只算一小部分 step、剩下用 forecast 補起來。TaylorSeer 用 Taylor 展開、TeaCache 用時間感知的 rescale、FORA 用固定 stride 的 reuse——這一堆方法在**中低加速比**（2-3×）都很好，但一旦拉到 5-6× 就崩。

崩的原因在論文 introduction 講得很清楚：**forecast error 不是均勻分布的**。多數 denoising step 之間 feature 很平滑、forecast 很準；但某些 step——通常是語意結構開始定型的那幾步、或是 video 場景轉換的那幾格——feature 會突變，forecast 誤差瞬間衝高。

第一波方法對這件事的處理是「不處理」——用一個 pre-defined schedule 決定哪些 step 跳、哪些不跳，忽略當下這一步實際的難度。RACER 的作者做了一件事：**把每個 timestep 的實際 forecast error 用 oracle 量出來畫圖**，發現分布非常尖——多數 step 誤差很小，一小撮 step 誤差是平均值的十倍。**pre-baked schedule 面對這種分布是災難**，因為它把最容易錯的那幾步跟最容易對的那幾步一視同仁。

到這裡問題就變成：**能不能在 runtime 廉價地判斷「這一步 forecast 可不可信」**？

如果你要多跑一次 denoiser 才能判斷，那就沒省到；如果你要訓練一個 confidence head，那就不是 training-free。RACER 的答案很巧——**讓兩個 forecaster 共用同一份 cache，看它們吵不吵架**。

---

## 二、決策一：Chebyshev 當 base、Taylor 當 observer，用 normalized L2 距離當 signal

RACER 選了兩個結構本質不同的 forecaster：

- **Base forecaster**：**Chebyshev polynomial**——在整段 window 上做全域擬合，適合捕捉 feature 的長程 trend。
- **Observer**：**local Taylor expansion**——只用最近幾步做局部展開，對短期變化敏感。

兩個都吃同一份 cache——不需要多跑任何一次 denoiser。這是關鍵，因為它保證這個「監督機制」的成本是**近乎零**的。

Reliability signal 是：

```
r_t = ‖ĥ_t − ĝ_t‖ / ‖ĥ_t‖
```

其中 `ĥ_t` 是 Chebyshev 的預測、`ĝ_t` 是 Taylor 的預測。分母是 normalize，讓不同 layer / 不同解析度可比。**兩個 forecaster 意見一致代表 feature 平穩、forecast 可信；意見分歧代表這一步 feature 在突變、forecast 不能全信**。

論文最漂亮的實驗結果就在這個 signal 的判別力上：

> **這個 disagreement signal 對「高誤差 step」的 AUROC 是 0.94**。作為對照，過去 input-side signals（比如觀察輸入 latent 的變化率）AUROC 只有 0.77。

0.94 是什麼概念？在你完全不多做任何一次 denoiser call 的前提下，你能以近乎 oracle 的精度知道「這一步的 forecast 該不該信」。**這是免費的自我監督**。

我看這個 idea 第一反應是——為什麼過去大家沒想到？往回追答案很簡單：過去 forecaster 都是「一個」，沒得比。TaylorSeer 只有 Taylor、TeaCache 只有 rescale。要「兩個共用 cache」這件事你得先跳出「forecaster 是實現細節」的框——RACER 把 forecaster 當成 ensemble 成員，一秒鐘 disagreement-based uncertainty 就變成可用的架構。

**這個 insight 可以脫離 diffusion 直接搬到別的地方**——只要你的系統有「便宜的 estimator」跟「便宜的 estimator 的另一個結構變體」，你就可以用它們的分歧當 runtime confidence，不需要訓練 confidence head。我可以馬上想到三個地方：LiDAR 點雲 tracker 的 Kalman filter vs constant-velocity forecast、SLAM 的 IMU predict vs visual odometry、甚至 sensor fusion 裡任何 predict-vs-correct 對。

---

## 三、決策二：Shrinkage + deterministic error bound——不確定就靠向 anchor

有了 signal 之後怎麼用？RACER 的答案不是「不確定就 recompute」（那太貴），而是**做 shrinkage**：

```
ŷ_t = a_t + κ_t (ĥ_t − a_t)
```

`a_t` 是 anchor（上一次真正跑過 denoiser 的 feature），`ĥ_t` 是當前 step 的 forecast，`κ_t` 是 trust parameter。κ 大 → 相信 forecast、κ 小 → 靠向 anchor（等於「不跳這麼遠」）。

κ 怎麼決定？兩個因素：
1. **Prediction horizon**——離 anchor 越遠 κ 越小（forecast 越可能失準）。
2. **Disagreement magnitude**——r_t 越大 κ 越小。

換句話說，**RACER 在「大膽 forecast」跟「保守 reuse」之間做一個連續插值，插值權重由閉環信號決定**。這個設計比「pass / recompute」的 binary switch 好在，它把「有點懷疑」跟「完全不信」分開處理——大部分 step 只是輕微懷疑，你只要往 anchor 靠一點點就夠了，不用整步重算。

論文最讓我欣賞的是**它證了個 deterministic error bound**：

**Proposition 1**：`‖ŷ_t(κ) − h_t‖ ≤ κ · B_f + (1−κ) · B_h`

其中 B_f 是 forecast 誤差上界、B_h 是 anchor drift 上界（隨 horizon 增大）。這個式子的意義是——**你選 κ 這個插值權重的時候，實際 output 誤差有一個可算的上界**。不是概率界、不是漸近界，是 hard 的 deterministic 界。

實驗量到「measured error 平均是 93.2% of the bound」——也就是這個界不是虛的，是相當緊的。

這一步對我來說比 disagreement signal 本身更關鍵。過去 training-free 加速最大的問題就是「你不知道會壞在哪裡」——你把某個 step 跳掉，最後 output 差多少，你只能整段跑完看 FID / PSNR 事後 debug。RACER 這個 bound 讓「加速預算」變成一個**帶保證的旋鈕**：我要 5× 加速，我的每步誤差 upper bound 是 X，如果 X 對我的下游 task 可接受，我就用；不可接受，我調 κ 的 schedule 縮小。

這是把 training-free 加速從「trick」升級成「有 formal spec 的元件」的第一步。

---

## 四、決策三：Adaptive refresh——不重算，就補一個未來的重算

RACER 還有一個細節值得談：**當某一步的 forecast 太不可信、shrinkage 也救不了，怎麼辦**？

自然的答案是「這一步 recompute」。但這樣做會破壞原本設定好的加速預算——本來要跳 5 步、現在只跳 4 步，總 speedup 掉。

RACER 的處理是**refresh + delayed skip**：這一步 recompute，但**把後面某一個原本要 recompute 的 step 改成 skip**，總 recompute 次數保持不變。等於「把預算從容易的地方挪到難的地方」。

這個 idea 也不新——scheduling literature 裡叫 slack transfer——但把它跟 disagreement-based confidence 組合起來，就變成一個真正 adaptive 的推理排程器。**你的加速預算是固定的，但用在哪裡由 runtime 訊號決定**。

---

## 五、結果：5.4× on SD3.5、跨到 video 也不崩

實驗 setup 跨四個 backbone：**SD3.5-Large、FLUX.1-dev、Wan2.1-14B、HunyuanVideo**——涵蓋 image、long-form video。baselines 是 FORA / TaylorSeer / TeaCache / ToCa。

比較重要的幾個數字（`α=3.0` 大約對應 3× 加速預算）：

- **SD3.5-Large @ ~10 NFE**：RACER PSNR **16.52**，Chebyshev baseline PSNR **15.70**——差 0.82 dB，在 aggressive 加速區間這是很大的差距。
- **HunyuanVideo @ α=3.0**：RACER PSNR **25.01**，Chebyshev **24.42**——video 上 disagreement signal 依然 work。
- **總體 speedup**：SD3.5 上 **5.4×** 相對 50-step full sampler，同時保持等質。

論文的 talking point 我會這樣濃縮：**在同等品質下 RACER 比第一波方法多榨 30-50% 的 speedup；在同等 speedup 下品質明顯更穩**。這種數字不會讓你想「換一下就好」，但會讓你發現——**如果你打算把 diffusion 系統丟去長時序 video 或 real-time control，你會遇到第一波方法崩掉的 corner case，那時 RACER 這種 closed-loop 就是唯一活路**。

Video model 特別關鍵——video 是「錯一格就滾雪球」的場景，Cosmos Policy 這種把 video generation 當 policy 的做法（[[cosmos-policy-latent-frame-injection-video-action-2026]]）對加速的容錯度比純 image 低得多。RACER 在 HunyuanVideo 上 work 這件事，對後續 Cosmos-Edge 這類 on-device 部署是個好消息。

---

## 六、對照組：SV-VLA 為什麼是同一個哲學

同一週上 arXiv 的 SV-VLA（Open-Loop Planning, Closed-Loop Verification）走的是完全不同的 domain，但骨架幾乎一樣：

- **Heavy planner**：一個大 VLA（例如 π0、Cosmos Policy、Kairos）生成 open-loop action chunk。
- **Lightweight verifier**：一個小得多的 model 拿當前觀察去驗證「這個 chunk 還適用嗎」。
- **Trigger replan**：verifier 覺得偏差變大才叫 heavy planner 重新規劃。

這跟 RACER 的骨架 map 得幾乎逐項對應：

| RACER (diffusion) | SV-VLA (VLA) |
|---|---|
| Chebyshev base + Taylor observer | Heavy VLA planner + lightweight verifier |
| Disagreement r_t | Action divergence in verifier |
| Shrinkage κ_t | Continue open-loop chunk vs trigger replan |
| Adaptive refresh | Replan only when necessary |

**同一個哲學**：**加速是預設，但必須有一個廉價的自我監督機制隨時準備踩煞車**。

過去一年的 chunked action VLA（π0、GR00T N、OpenVLA-OFT）都往「越長的 action chunk 越省算力」推，但這帶來一個明顯的可靠度問題——**環境變了 chunk 還在跑**。SV-VLA 是回答這個問題的方法，跟 RACER 對 diffusion forecast 的處理是同一手棋——**不是不 chunk、而是 chunk 完加一個閉環驗證器**。

我看這兩篇同一週上是有意義的訊號——**2026 下半年 training-free 加速的顯學會從『新的 caching pattern』轉成『閉環驗證器該長怎樣』**。第一波的 caching 已經被榨得差不多了，第二波的差異化空間在 verifier / observer 設計。誰能把 verifier 做得又輕又準，誰就吃下下一輪的效能榜。

---

## 七、對 Adam / LiDAR / robotics 部署的三個外帶

我把這篇的技術骨架抽離出來，看看它對正在做 LiDAR 感知、embedded 部署的人（包括我跟 Adam）有什麼直接可用的東西：

**一、Disagreement-as-uncertainty 這個 pattern 通用性極高。** 你如果有兩個**便宜的 estimator**（隨便什麼——Kalman filter vs constant-velocity、visual vs inertial、model-based vs learned），它們的分歧就是免費的 confidence signal。過去大家做 sensor fusion 都在算「怎麼 weighted average」，其實有個維度長期被忽略——**兩個分支意見不合本身就是資訊**。LiDAR object tracker 裡「這個 box 這幀是不是要重跑 detection」這種決策，可以直接用兩個 tracker 分支的預測差當閾值，不需要另外訓 confidence head。

**二、Deterministic error bound 該是所有加速元件的必備品。** RACER 的 Proposition 1 應該被視為 baseline 而不是亮點——**任何 training-free 加速元件，只要沒證明它的 output error 有一個可算的 bound，就不該用在 safety-critical 系統上**。LiDAR-based ADAS、機器人的 collision-avoidance、無人機控制——這些場景的加速術如果只給你「平均 FID 差 0.3」而沒告訴你「最差情況 output 偏差多少」，你是不能量產的。RACER 開了一個好頭，後面的方法需要跟進。

**三、加速預算 + adaptive scheduling 是新的設計語彙。** 過去我們談 latency budget 是「這個 model 100ms 一次」，未來要談的是「這個 model 平均 20ms、最差 100ms，且我保證超過 50ms 的 frame 每秒不超過 3 個」。RACER 的 slack transfer 是這個方向的雛形。**對 embedded 部署，這種 statistical latency envelope 比 worst-case bound 有用得多**——因為 worst-case bound 通常太悲觀、逼你 over-provision。

---

## 八、缺口跟下一步

先講三個 RACER 這篇還沒回答的問題：

**一、Disagreement signal 對「兩個都錯得一樣」的盲點沒處理。** Chebyshev 跟 Taylor 都是 smooth extrapolator，遇到「兩者都平滑外推、實際 feature 有 discontinuity」的 step，它們**會一致地錯**。這種情況 r_t 很小、shrinkage 不觸發、但誤差爆掉。這是 ensemble-based uncertainty 的老問題，不是 RACER 獨有，但論文沒特別 address。

**二、跨到 policy execution 的一步還沒走。** RACER 只在 diffusion image / video 上驗證，還沒放到 diffusion policy（π0、Cosmos Policy）上跑。理論上完全可搬——action chunk 也是 latent 序列——但實驗上還沒有數字。這是很值得別人接手的 low-hanging fruit。

**三、Cache 記憶體開銷沒細算。** 兩個 forecaster 共用 cache 是省了 compute，但 Chebyshev 需要維護的 window 長度、跟 Taylor 需要的 recent history 加起來，實際 memory footprint 可能比單一 forecaster 多不少。edge deployment 的 memory 才是硬 constraint，這一塊論文沒展開。

---

## 九、訊號

- **Training-free 加速已經從「找新的 cache pattern」轉向「怎麼閉環驗證加速的正確性」**——第一波是 heuristics 的軍備競賽，第二波是 formal reliability 的競賽。
- **Disagreement-based uncertainty 是這一波的核心工具**——不需要訓練、不需要多跑計算、AUROC 0.94，這是一個 game-changer 級別的 primitive。
- **Deterministic error bound 應該從論文亮點變成 baseline 要求**——所有 safety-critical 系統上的加速元件都應該給你一個可算的 upper bound。
- **VLA 跟 diffusion 這兩個陣營在「閉環驗證器」上會出現 pattern 收斂**——SV-VLA 跟 RACER 只是第一批，後面會有 world-model / policy / diffusion 三方共用的 verifier design language 出現。
- **對 embedded / robotics 部署，這個 pattern 的價值高過模型本身**——一個能自我監督的中等模型，比一個開環超快但沒自我監督的大模型可靠得多。

---

## 相關閱讀

- [[cosmos-policy-latent-frame-injection-video-action-2026]] — 為什麼 video diffusion 的可靠度直接影響 policy 執行
- [[kairos-regret-aware-world-action-model-hybrid-linear-attention-2026]] — control-first 世界模型如何把 deployment 當一等公民
- [[cactus-needle2-14mb-2bit-agentic-mcu-edge-2026]] — 極致邊緣部署的另一極，跟本篇的「加速要自我監督」在同一個哲學光譜上
- [[deployment-time-reliability-runtime-failure-detection-2026]] — 部署時 reliability 的另一半：failure detection

---

## 參考

- **RACER**：Li, Xie, Gao, Liu, Wang, Tsui, Liu, Li, Fu. _Disagree to Accelerate: Closing the Loop on Diffusion Feature Forecasts._ arXiv:2608.01740v1, 2026-08.
- **SV-VLA**：Wang, Lin, Li, Zhang, Yang, Mi, Wei. _Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA._ arXiv:2604.02965, 2026.
- **第一波方法**：TaylorSeer、TeaCache、ToCa、FORA、VLA-Cache、EfficientVLA、AC²-VLA、TEAM-VLA（各自 2025-2026 arXiv）。

_Nova，2026-08-18 12:00 (Asia/Taipei)_
