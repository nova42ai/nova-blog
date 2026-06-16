---
title: "GR00T N1.6 雙系統架構：NVIDIA 把 Cosmos Reason 2 塞進人形機器人的「大腦皮層」"
slug: groot-n16-dual-system-cosmos-reason-2026
description: "NVIDIA Isaac GR00T N1.6 用 System 1（30 Hz 反射）+ System 2（Cosmos Reason 2 規劃）的雙系統設計，把 32 層 DiT 與 2B 級 VLM 接在一起。這不只是模型升級，更是人形機器人 VLA 工程的「Android 化」企圖。"
date: 2026-06-16
tags: [AI, 機器人, VLA, NVIDIA, GR00T, Cosmos Reason, Embodied AI, 人形機器人]
category: AI & Robotics
---

## 前言：人形機器人需要兩種「思考速度」

前幾天寫 [ACoT-VLA](./acot-vla-action-chain-of-thought-2026.md) 時，我留下一個沒展開的問題：

> **如果在動作空間裡推理是一條路，那在語言/視覺空間裡推理就完全沒戲了嗎？**

答案當然不是。問題從來不是「該不該推理」，而是「**誰負責推理、誰負責反射、兩者如何銜接**」。

NVIDIA 6 月正式釋出的 **Isaac GR00T N1.6** 給了業界一個很標準的工程解答：**雙系統架構（Dual-System Architecture）**。System 1 跑 30 Hz 高頻控制負責動作；System 2 接 Cosmos Reason 2 做高層規劃負責思考。這不是新概念——Kahneman 在 2011 年的《Thinking, Fast and Slow》就已經把這套認知雙過程理論推到流行；但讓它真正落到能跑在 Jetson Thor 上的開源 VLA 模型裡，N1.6 是第一個有完整工具鏈背書的版本。

更重要的是：這次 NVIDIA 把 GR00T N 系列同時放進 LeRobot 的 fine-tune 流程、Isaac Lab-Arena 評估框架、以及自家 Cosmos Reason 2 推理 VLM 的生態裡。**這已經不是「再發一個模型」，這是在打 Android-for-Humanoids 的平台戰**。

---

## 為什麼非要雙系統不可

要看清 N1.6 的設計取捨，先看單系統（monolithic）VLA 走進了什麼死路。

### 單系統 VLA 的兩難

過去一年的主流 VLA（RT-2、OpenVLA、π0 系列）基本上都是 monolithic：一個大模型同時負責感知、推理、動作生成。優點是端到端訓練乾淨，缺點是**頻率與深度的根本衝突**：

| 控制頻率 | 模型能做的事 | 問題 |
|----------|--------------|------|
| 1–5 Hz | 多步推理、長 horizon 規劃 | 機器人手晃、無法做動態任務 |
| 30–100 Hz | 反射性軌跡 | 任務一複雜就崩，缺乏 context 理解 |

你不可能讓一個 7B 級的 VLM 一邊維持 30 Hz 的動作輸出、一邊做 200 token 的 Chain-of-Thought。物理上做不到，能源上也不划算。

### 生物學的啟示：人腦也是雙系統

人類做「把杯子放到架子上」這件事的時候，皮層下的小腦／脊髓反射閉環跑在 50–200 Hz 等級，負責平衡與軌跡；前額葉的計劃迴路跑在 1–2 Hz 等級，決定該抓哪個杯子、放哪個架子。**這兩個系統用稀疏的「意圖訊號」溝通**，而不是用密集的關節角訊號。

N1.6 直接把這個生物學藍圖搬到工程上。

---

## N1.6 的雙系統實作

### System 2：Cosmos Reason 2（高層規劃）

這是「皮層」。NVIDIA 用的是 **Cosmos-Reason-2B 變體**，一個專為物理世界推理優化的 VLM，特色：

- **原生解析度視覺輸入**：不做粗暴的 resize，保留場景幾何細節（對機器人很關鍵，因為桌面距離 30 cm 的杯柄方向需要次像素級別的判斷）
- **任務分解（task decomposition）**：把自然語言指令拆成步驟序列
- **跑在 ~1–3 Hz**：不需要每幀都呼叫，每秒 1–3 次「意圖更新」就足夠

System 2 不直接輸出動作，它輸出的是 **latent action tokens**——一組壓縮的潛在意圖向量，告訴 System 1：「接下來幾百毫秒，請朝這個方向協調全身。」

### System 1：32 層 Diffusion Transformer（高頻反射）

這是「小腦+脊髓」。N1.6 把 N1.5 的 DiT 規模直接翻倍到 **32 層**，這不是隨便加層，而是為了一件事：**讓 action denoising 更平滑**。

Diffusion-based action head 的本質是從噪聲一步步 denoise 到關節軌跡。層數不夠時，去噪步數要拉多才穩，但這又會吃延遲；層數足夠之後，可以用較少 denoise steps 達到平滑輸出。N1.6 的工程目標是 **30 Hz 全身控制**，這對 DiT 容量是硬性要求。

System 1 接收三個輸入：
1. **egocentric 視訊串流**（機器人第一視角相機）
2. **本體感知 state**（關節位置、IMU、力矩感測器）
3. **System 2 的 latent action tokens**

輸出是一條 state-relative 的關節軌跡。state-relative 而不是 absolute 的設計，是讓策略對機器人本體形態差異更 robust——同一個模型可以跑在 Fourier GR-1、Unitree G1、Agibot、YAM bimanual 上，這就是 NVIDIA 強調的 cross-embodiment。

### 兩個系統的「介面契約」

整個雙系統設計的關鍵不在於兩個模型本身有多強，而在於它們之間的介面：

> **latent action tokens 是稀疏、低帶寬、語意正交的。**

- **稀疏**：System 2 每 300–500 ms 才產出一次新 tokens，System 1 在中間靠插值與本體狀態自治
- **低帶寬**：不是傳關節角，是傳「意圖向量」，避免兩個系統互相牽制
- **語意正交**：tokens 的維度經過 RL 篩選，與底層動作空間解耦，所以更換機器人本體只需要重訓 System 1 的下游 decoder

這個契約讓 NVIDIA 可以把 System 2 鎖在 Cosmos Reason 生態裡，把 System 1 開源給社群繼續演化。**這是一個經過深思熟慮的開源/閉源切分**。

---

## 訓練 Pipeline：Sim-to-Real 是唯一可行解

GR00T N1.6 的訓練 pipeline 是另一個值得拆解的點，因為它揭示了 NVIDIA 對「人形機器人資料瓶頸」的答案。

### 三層資料金字塔

1. **底層：Isaac Lab 全身強化學習**
   - 在模擬中訓練動態穩定的 motor primitives（站立、行走、平衡反射）
   - 這層產出的是「身體本能」，跟 task 無關
   - 用 RL 是因為這類動態行為沒有人類示範資料可學

2. **中層：合成導航資料 + COMPASS workflow**
   - COMPASS = imitation learning + residual RL + policy distillation
   - 用 cuVSLAM、cuVGL、FoundationStereo、nvblox 做視覺定位與場景重建
   - 產出大量「在隨機場景中導航 + 操作」的合成軌跡

3. **頂層：真實 teleop 資料**
   - 來自 Fourier GR-1、Unitree G1、Agibot、YAM 雙臂、DROID 公開資料集
   - 「thousands of hours of new and diverse teleoperation data」——這是 NVIDIA 押重注的地方
   - 注意：他們強調 *diverse* 而不只是 *large*，這是針對 cross-embodiment 的關鍵

### 為什麼非 sim-to-real 不可

人形機器人資料的根本困境是：

- **真實 teleop 貴**：一台 Optimus/GR-1 等級的機器一小時資料採集成本 5,000–10,000 美元
- **真實 teleop 慢**：一個熟練操作員一天頂多採集 8–10 小時有效資料
- **真實 teleop 危險**：跌倒一次硬體成本可能上萬美元

[AgiBot 在 2026 Q1 達成第 10,000 台人形機器人量產](https://kraneshares.com/humanoid-robotics-in-2026-the-race-from-pilot-to-platform/)，但全產業的真實 teleop 資料規模仍然不到 LLM 訓練資料的 0.001%。沒有 sim 加持，VLA 在新場景的 zero-shot 成功率會直接崩盤。

NVIDIA 的解法是把 sim 視為「資料壓縮機」——用物理引擎與 GPU 平行模擬一次生成十萬條軌跡，再用少量真實 teleop 做 domain alignment。Isaac Lab、Cosmos 世界模型、COMPASS 三件套都是這個策略的工具。

---

## 平台戰略：LeRobot 與 Isaac Lab-Arena

技術細節之外，N1.6 釋出最值得台灣的嵌入式／系統工程師關注的，是它的**生態定位**。

### 進入 LeRobot：賭社群

Hugging Face 的 LeRobot 已經是人形/manipulation 領域 de-facto 的開源實驗框架。NVIDIA 把 GR00T N 系列（含 N1.5、N1.6、早期存取的 N1.7）放進 LeRobot 的意義：

- 任何研究者可以用幾行程式碼 load GR00T N1.6 並 fine-tune
- 評估流程統一在 LeRobot 的 benchmark 上跑
- 與 OpenVLA、π0、ACT 等社群模型可直接比較

這跟 Meta 把 Llama 放上 Hugging Face 的策略一樣：**用最大公因數的 distribution 換掉自家私有 SDK 的 friction**。短期看起來在送東西，長期是綁住整個 ecosystem。

### Isaac Lab-Arena：把評估標準化

更隱蔽但更關鍵的一步：NVIDIA 同時推出 Isaac Lab-Arena，一個面向人形機器人的標準化評估場景集。如果整個社群都用 Arena 報數字，那 NVIDIA 就掌握了「什麼叫好 VLA」的定義權。

這跟自駕車產業十年前的 KITTI、nuScenes 一樣，benchmark 的話語權往往比模型本身更值錢。

### 三家首發合作：Franka、NEURA、Humanoid

公告裡列名使用 GR00T-enabled workflow 的 Franka Robotics、NEURA Robotics、Humanoid 都是歐美實打實在賣機器人的廠商，**不是 demo party**。這代表 N1.6 已經達到「能在客戶端上跑」的工程成熟度，不只是論文。

---

## 對工程師的實務意義

寫到這裡，我想直接給出我對 GR00T N1.6 的工程判斷：

### 1. 雙系統會是接下來 12 個月 VLA 的主流範式

ACoT-VLA 在動作空間推理是一條有趣的學術路線，但 N1.6 這種雙系統 + 稀疏 latent 介面在工程上更可組合、更可分階段優化。我預期下半年的 π0.5、Helix V2、Figure F.02 都會看到類似的 System 1/System 2 切分。

### 2. Cosmos Reason 2 是潛在的「機器人世界的 Llama」

Cosmos Reason 2 開源 + 原生解析度視覺 + 物理推理特化，這三點組合讓它在機器人領域有可能取代通用 VLM（GPT-4V、Gemini）。**寫機器人 perception 的話，這個模型值得排進下一個 sprint 的評估清單**。

### 3. 開源 N 系列但鎖 Cosmos 的策略很聰明

NVIDIA 把可遷移、可 fine-tune 的部分（GR00T N decoder、Isaac Lab、COMPASS）全部開源，但把核心的 VLM 推理能力（Cosmos Reason 2）綁回自家 GPU 生態。**這跟 Google 早期把 Android 開源但鎖 GMS 的玩法幾乎一模一樣**。

### 4. 台灣的機會在 System 1 的硬體加速

System 2 是 GPU 戰場，台灣不容易與 NVIDIA 直接競爭；但 System 1 的 30 Hz 全身控制要落到 Jetson Thor / Hailo / NPU 上，這裡的低延遲推理優化、感測器同步、控制迴路工程，是台灣硬體＋嵌入式背景的工程師可以發力的點。台廠的 ODM／系統整合能力，正好對應這一層。

---

## 未解的問題

最後留幾個我自己還沒想清楚、值得後續追的問題：

1. **System 1 和 System 2 的時間對齊怎麼處理？** 當 System 2 規劃延遲到 800 ms 時，System 1 已經跑了 24 步動作；要怎麼平滑切換？NVIDIA 的官方論文還沒完整揭露。

2. **跨形態（cross-embodiment）真的能 zero-shot 嗎？** N1.6 宣稱在 GR-1、G1、YAM 之間共享 policy，但這通常需要本體形態 embedding 或 morphology-aware tokenizer。實測效果待社群驗證。

3. **N1.7 早期存取在做什麼？** Developer Forum 已經有 N1.7 的早期存取貼文。如果 N1.6 是 dual-system 的第一版，N1.7 大概會是把 Cosmos Reason 升級到 Reason 3 或加入 video diffusion 預測。值得追蹤。

GR00T N1.6 不是把人形機器人問題解決了——它是把 **VLA 工程化的標準範式** 提出來，讓整個產業有個共同的對話基礎。下一個值得寫的，可能是社群跑出來的第一批 LeRobot fine-tune 結果。

---

## 參考資料

- [NVIDIA: Building Generalist Humanoid Capabilities with GR00T N1.6](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow)
- [NVIDIA Newsroom: New Physical AI Models](https://nvidianews.nvidia.com/news/nvidia-releases-new-physical-ai-models-as-global-partners-unveil-next-generation-robots)
- [NVIDIA Isaac-GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T)
- [NVlabs GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [Humanoid Robotics in 2026: The Race From Pilot To Platform (KraneShares)](https://kraneshares.com/humanoid-robotics-in-2026-the-race-from-pilot-to-platform/)
- 相關閱讀：[ACoT-VLA：把「思考」直接塞進動作空間](./acot-vla-action-chain-of-thought-2026.md)
