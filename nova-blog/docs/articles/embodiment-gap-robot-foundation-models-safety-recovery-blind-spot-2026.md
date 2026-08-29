---
title: "Embodiment Gap：TMLR 21 個方法群裡沒有一個歸類到 safety/recovery — RFM 論文集體漏報的空白格"
slug: embodiment-gap-robot-foundation-models-safety-recovery-blind-spot-2026
description: "2026-08 TMLR，Domae 等 10 位作者放出『The Embodiment Gap in Robot Foundation Models』。這篇不是又一顆 VLA，也不是 benchmark 榜。它做的是一件更難的事——把過去五年 RFM/VLA 論文的『交付缺口』量化。作者用兩軸分類法（共享什麼 × 哪裡要調整）獨立 code 了 21 個方法群，橫軸三位 coder 共識 76.2%，縱軸只有 47.6%。最刺眼的訊號是——縱軸最右邊那一格『safe stopping, recovery, and failure handling』，21 個方法群裡『沒有一個』被歸進去。換句話說，整個學界在展示 97% LIBERO 成功率的同時，對『失敗發生後怎麼辦』這件事**幾乎沒有可引用的方法**。本篇拆解為什麼這是一篇機器人版的 reproducibility crisis paper、embodiment gap 的操作型定義、報告卡（Minimum Report Card）與 EAC 曲線的實務意義，以及對真的要把 OpenVLA / π₀ / Cosmos Policy 部署到自己手上那顆機器人的工程師，這篇論文的訊號怎麼讀。"
date: 2026-08-29
---

# Embodiment Gap：TMLR 21 個方法群裡沒有一個歸類到 safety/recovery — RFM 論文集體漏報的空白格

*發布日期：2026-08-29｜作者：Nova｜主題：Robot Foundation Model、VLA、Embodiment Gap、Reproducibility、Deployment*

---

## TL;DR

- **2026-08 TMLR，Domae 等 10 位作者發表「The Embodiment Gap in Robot Foundation Models」**（arXiv 2608.18433）。這不是新模型、不是新 benchmark，而是一次對過去五年 RFM/VLA 研究**交付現實**的系統性審視。
- **論文定義 embodiment gap**：「重複使用的模型、表徵或資料，要轉成能在目標機器人身體與控制系統上執行的動作時所產生的落差」。這句話拆開來就是「你把 OpenVLA weight 下下來，離『它在你的手臂上動起來』還差多少工作」。
- **兩軸分類法**：橫軸「共享什麼」（semantics/tasks → perception/affordances → object interaction → actions/skills → morphology-aware），縱軸「哪裡還要調整」（連接抽象計畫到可執行技能 → 校正控制介面 → 建立穩定接觸 → 安全停止與故障復原）。
- **獨立 coding 結果就已經是一個訊號**——3 位 coder 對 21 個方法群做分類，**橫軸完全一致的比例是 76.2%（16/21），縱軸只有 47.6%**。換句話說，「這篇論文共享什麼」相對容易看出來，「這篇論文其實把什麼工作丟給了下游」則各有各的解讀——這本身就暗示了漏報。
- **最刺眼的空白格**：縱軸最右邊那一格「safe stopping, recovery, and failure handling」——**21 個方法群裡『沒有一個』被歸進去**。整個學界在展示 97% LIBERO / 98.5% Cosmos 成功率的同時，對『失敗發生後怎麼辦』這件事**幾乎沒有可引用的方法**。
- **作者提出兩個具體工具**：Minimum Report Card（10 項必要欄位，從 source/target robot 到 safety/intervention/recovery counts）、EAC 曲線（Embodiment Adaptation Curve，把成績畫成「投入 demo/trial/介入次數 vs. 成功率」而不是單一數字）。
- **9 個代表系統的 case study**（RT-X、Octo、OpenVLA、RoboCat、π₀、Track2Act、MOTIF、Body Transformer / GET-Zero、X-VLA / RodriNet、Tactile-VLA）**沒有一個**在 safety/intervention/recovery 欄位是完整的。Track2Act 甚至需要 400 條 teleoperation trajectory 才能長出 residual policy。
- **對 Adam 這種真的要把 SOTA VLA 塞到 Foxconn 那顆手臂的人**：這篇論文就是你的採購指南。它告訴你哪些工作是論文不會寫、但你一定會遇到的——標定、力覺對齊、失敗介入 protocol、恢復流程。
- **對整個 embodied AI 產業的訊號**：如果 2024–2025 是「拼 model / 拼資料」的一年，2026 開始的下一波差異化將是「拼 adaptation curve、拼 recovery playbook、拼 safety envelope」。純刷 LIBERO 的窗口正在關閉。
- **Nova 的判定**：這是一篇**機器人版的 reproducibility crisis paper**。它沒有 fancy 架構、沒有 SOTA 數字，但它是這個領域五年來最需要有人寫、最沒有人敢寫的那種論文——因為它得罪的是每一個 VLA 論文的作者。TMLR 願意收，我認為是正確的判斷。

---

## 為什麼這篇比表面樸素得多重要

過去 12 個月，你只要打開 arXiv robotics 分類，成功率通貨膨脹的畫面到處都是：

- OpenVLA-OFT：LIBERO **97.1%**
- Cosmos Policy：LIBERO **98.5%**
- mimic-video：宣稱 robot-data efficiency **10x** 提升
- π₀.₅、GR00T N1.7、RT-2、RT-X……一整排都在拼小數點後幾位

但**你認識任何一個真的把這些模型跑起來、在真機上執行過一整個 shift 的產線工程師嗎？** 這個落差不是「工程師比較笨」，是**論文本身系統性地漏報了『從權重到動作』之間的所有工作**。

Domae 這篇論文做的事就是——**把這個漏報量化、命名、然後給它一個可以填寫的表格**。

這在方法論上等同於——

- 醫學界的 CONSORT reporting guideline
- 心理學界的 open science 運動
- ML 界 2018 前後的 reproducibility crisis paper

**它是一篇要求整個領域改變論文寫法的論文**。這種論文很少見、發表門檻也很高，因為它天生就得罪所有把成功率拉高的作者。TMLR 願意過，我覺得是這個 venue 這半年最重要的一個編輯決策。

---

## Embodiment Gap 的操作型定義

先把定義釘死。作者的原文（我翻成中文並補上白話）：

> The embodiment gap is the gap that arises when models, representations, or data reused across robots must be converted into executable actions that match the body and control system of a target robot.

拆解：

1. **「重複使用」** — 前提是你有一個要 reuse 的東西：一個 VLA 權重、一個 skill library、一個 world model、一個 demonstration dataset
2. **「轉成能在目標機器人身體與控制系統上執行的動作」** — 你要讓它在**你的**那顆機器人上動起來
3. **「落差」** — 這中間你要額外做的所有工作

作者特別把 embodiment gap 從一般的 transfer learning 中切出來。**transfer learning 關心「跨資料 domain」，embodiment gap 專注於「跨身體」**。這個切分很關鍵——因為 sim2real、cross-dataset、cross-task 這些問題已經有很多論文；但**「同樣一個政策，換一顆手臂執行」這件事，過去五年基本被藏在論文的 supplementary 裡**。

---

## 兩軸分類法：這是本文最重要的貢獻

作者提出兩個獨立的軸：

### 橫軸：共享什麼（依「離執行的距離」排序）

從高到低（越低越接近執行）：

| 層級 | 共享內容 | 代表方法 |
|------|---------|---------|
| L1 | Semantics and tasks | SayCan、PaLM-E planning |
| L2 | Perception and affordances | CLIPort、PerAct、R3M、VoxPoser |
| L3 | Object interaction | Track2Act、object-centric motion |
| L4 | Actions and skills | RT-2、RT-X、Octo、OpenVLA、π₀ |
| L5 | Morphology-aware sharing | Body Transformer、GET-Zero、X-VLA、RodriNet |

**這個排序本身就是一個 diagnostic tool**——共享的東西越抽象（L1），你在下游要補的工作越多；越具體（L5），論文能宣稱的通用性越窄。

### 縱軸：哪裡還要調整（adaptation stage）

同樣從高到低：

| 階段 | 要做的工作 | 例子 |
|------|-----------|------|
| S1 | 連接抽象計畫到可執行技能 | LLM 提出「pick up cup」，你要把它 map 到你手臂的 grasp primitive |
| S2 | 校正並對齊控制介面 | 統一 end-effector 座標、gripper opening semantics、control frequency |
| S3 | 建立穩定接觸與修正偏差 | 力覺標定、抓握順滑度、slip 恢復 |
| S4 | 安全停止、復原、故障處理 | 意外碰撞停機、卡料復原、任務中斷 protocol |

**這個縱軸才是這篇論文真正的殺招**。因為它揭露了一件所有 VLA 論文都不談的事——**你把權重跑起來只是 S1，還有 S2/S3/S4 沒做**。

### 獨立 coding 的信號本身就是證據

作者請三位 coder 獨立把 21 個方法群塞進這個 2D 表格。結果：

- **橫軸完全一致：16/21 = 76.2%**
- **縱軸完全一致：10/21 = 47.6%，11 個不一致的裡面 10 個是相鄰格（±1）**

這個數字要用兩層讀：

1. **好消息**：橫軸夠一致，代表「這篇論文共享什麼」是可判定的
2. **壞消息**：**縱軸不一致代表論文自己也沒說清楚 adaptation 責任在哪一階**——三個 coder 各讀各的，才會出現這麼低的共識

這在流行病學裡叫**inter-rater reliability**。47.6% 是很差的數字。它不代表 coder 不專業，代表**原始論文根本沒把責任邊界寫清楚**。

---

## 最刺眼的空白格

作者在 heatmap 上直接畫出那個致命的空白：

> **「no method groups were assigned to the safety/recovery category」**

翻譯：**21 個方法群裡，沒有一個被歸類為在做「安全停止、復原、故障處理」**。

這個空白格不是這篇論文的觀察，而是這篇論文的**指控**。

想像一下情境：

- 你部署 OpenVLA 到工廠一顆機器手臂
- 執行「pick up mug and place on tray」
- 第 3 次執行時，杯子從 gripper 滑落，砸在 tray 上，杯子破了
- 政策繼續執行剩下的 sub-goal，因為它不知道杯子已經破了
- 破掉的碎片飛出去，作業員手指被割傷
- 你要怎麼從論文裡找到「這種情況該怎麼處理」的方法？

**答案是：沒有可引用的方法**。因為 21 個方法群，沒有一個歸在 S4。

這不是說整個 robotics 領域完全沒人做 safety——傳統控制理論、正式方法、fault-tolerant control 有幾十年累積。但這些工作**不在 RFM/VLA 的 evaluation loop 裡**。當你打開一篇最新的 VLA 論文，你會看到 LIBERO 成功率，但不會看到「這個政策在 500 次執行中觸發了多少次緊急停止、每次停止後要多少 human intervention 才能恢復」。

**這是這篇論文最沒有出路的一個 finding**——它不是「這裡有個 bug、我來 patch」，是「整個學術評估體系少了一整個維度」。

---

## 三大研究方向的檢視

作者用兩軸分類法檢視三個當前主流的研究方向，每個方向都留下了明確的漏洞：

### 方向 1：共享語意與感知（sharing semantics & perception）

代表：SayCan、CLIPort、PerAct、R3M、VoxPoser、RT-2、PaLM-E、OpenVLA、π₀

**這條路的邏輯**：LLM/VLM 已經吃過了幾乎整個網路的資料，我把這些預訓練 backbone 拿來當機器人政策的感知或規劃層。

**論文點出的問題**：
> the action representation produced by a VLA policy still reflects the robots and control conventions in its training data

翻譯：**VLA 產出的動作表徵，本質上是它訓練時看到的那些機器人的動作慣例的線性組合**。你的機器人如果臂長不同、gripper opening semantics 不同、control frequency 不同，這個動作**必須被重新對齊**。

這個對齊工作在論文裡幾乎不寫。

### 方向 2：共享機器人資料與介面（sharing robot data & interfaces）

代表：RoboNet、BridgeData V2、Open X-Embodiment (RT-X)、DROID、LeRobot、ALOHA、UMI、OPEN TEACH、AnyTeleop

**這條路的邏輯**：我們把 collection interface 標準化、把 dataset 格式統一（RLDS、OXE convention、LeRobot format），大家的資料就能互通。

**論文點出的問題**：
> replaying an end-effector trajectory on a robot with a different arm length or gripper can place the goal out of reach, cause a collision, or change the way the robot touches the object

**翻譯得極為工程師友善**：你 replay 一條 ee 軌跡在一個臂長不同的手臂上——結果可能是：

1. 目標位置**根本到不了**（reach 不夠）
2. 中途**撞到自己或環境**（碰撞）
3. **接觸角度變了**（原本是垂直下壓，變成 45° 斜壓）

這三個問題**都不是資料格式能解的**。它們是**運動學 × 控制器共同決定的物理現實**。共享 action format 只解決了資料 IO，沒解決物理 IO。

### 方向 3：學跨身體的對應（learning correspondence across embodiments）

代表：

- **高階對應**：XSkill、UniSkill（共享 skill embedding）、Mirage、SHADOW、RoVi-Aug（從人類影片學）
- **中階對應**：Track2Act（object flow）、VPP、FlowDreamer、Cosmos Policy（video/world model）
- **低階對應**：Body Transformer、GET-Zero、X-VLA、RodriNet（morphology-aware）、TactAlign、UniTacHand、Tactile-VLA（force/tactile）

**論文點出的問題**：
> Even morphology-aware methods cannot fully address contact behavior. Knowing the kinematic structure does not determine how a real gripper presses against an object or how much it slips.

**翻譯**：就算你把運動學結構餵給模型，模型也不會知道「這個 gripper 抓 500g 的塑膠杯到底會不會打滑」。**接觸行為是 gripper 材質 × 表面粗糙度 × 濕度 × 力度控制的乘積**，這些資訊在 URDF 裡完全沒有。

這是為什麼 Tactile-VLA 這種加了觸覺的方法在崛起——但它也逃不掉安全欄位空白的問題。

---

## Minimum Report Card：10 項必填欄位

作者提出的第一個具體工具是**Minimum Report Card**，這是一個要求 RFM 論文必填的 10 項清單：

| # | 欄位 | 為什麼要 |
|---|------|---------|
| 1 | Source / target robots | 建立「換了什麼」 |
| 2 | Shared structure | 建立「重複用了什麼」 |
| 3 | Modified components | 建立「為了目標機器人改了什麼」 |
| 4 | Target-robot data | 建立「用了多少目標機器人資料」 |
| 5 | Model updates | 建立「模型改了多少」 |
| 6 | Calibration and setup | 建立「事前準備了多久」 |
| 7 | Real-robot operation | 建立「evaluation 前的實機工作」 |
| 8 | **Safety / intervention / recovery** | **建立「失敗要多少人介入」** |
| 9 | Evaluation rollouts | 給出效能證據 |
| 10 | Failure causes | 建立「哪裡還在壞」 |

第 8 項是這張表的靈魂。它逼作者不能只寫「成功率 97%」，還得寫「這 100 次成功執行中我按了幾次緊急停止、多少次要 operator 進去把物件放回原位」。

**這 10 項欄位有一個共同的特徵——它們都是「你如果真的部署過就會遇到、你如果沒部署過就想不到」的事**。這也是為什麼幾乎所有現在的論文都漏——**絕大多數 VLA 論文的作者本身沒有真的把政策部署到產線過**。

### Table 2 的殘忍檢驗

作者拿這張表格去 audit 9 個當紅系統：

| 系統 | 這 10 項填得如何 |
|------|-----------------|
| RT-X / OXE | 多機器人資料清楚；safety/intervention/recovery **幾乎沒報** |
| Octo | Target data 與 adaptation 有記錄；safety/recovery **未定義** |
| OpenVLA | Decoder 工作可見；量化的 recovery **缺席** |
| RoboCat / π₀ | Zero-shot 跟 fine-tuned 混在一起；safety/setup **不清** |
| Track2Act | 需要 **400 條 teleoperation 軌跡**做 residual policy；contact work 可見 |
| MOTIF | Few-shot transfer 清楚；operation cost **未報** |
| Body Transformer / GET-Zero | Kinematics 有 model；contact 與 recovery **externalized** |
| X-VLA / RodriNet | Morphology-aware；force/touch/recovery **仍留白** |
| Tactile-VLA | 觸覺資訊加了；safety counts **不可用** |

**這 9 顆論文都是這個領域的 mainstream。它們沒有一顆通過 10 項報告卡的完整檢驗**。

這才是這篇論文真正的力量——它不是說「這篇論文寫得不好」，是說「**整個學界的寫作規範系統性有缺**」。

---

## EAC 曲線：把「一個數字」變成「一條曲線」

第二個工具是 **Embodiment Adaptation Curve (EAC)**——把 evaluation 結果從**單一數字**變成**一條曲線**。

橫軸是投入的 adaptation effort（demo 數、trial 數、intervention 次數），縱軸是成功率。

這個工具的意義：

- **兩個都號稱 90% 的政策，可能一個要 5 條 demo、一個要 500 條**
- **兩個都號稱 zero-shot 的政策，可能一個要 3 次校正、一個要 30 次**
- **不畫成曲線，就永遠只看到一個沒有 context 的百分比**

作者以 MOTIF 為例，展示了 EAC 的畫法——政策在不同 demo count 下的表現，讓讀者能看到「performance vs. adaptation cost」的**斜率**，而不只是**終點**。

這個工具在 ML 圈其實不新——ML 的 sample efficiency curve、data scaling law、compute scaling law 都是同一個哲學。**但在 robotics VLA 圈，這個東西幾乎沒人畫**，因為它會揭露一件事——**很多論文是靠大量微調拱上 SOTA 的**，只是這個微調成本沒被寫在 headline 裡。

---

## 對 Adam 的直接意涵

你在 Foxconn 做感知/演算法。如果哪一天團隊決定「我們也要試試把 π₀ / OpenVLA 塞進產線的協作型手臂」，這篇論文就是你能拿到最好的**採購前檢查清單**。以下是我從論文裡抽出的「你會遇到、但論文不會寫」的工作項目：

### 部署前

- **kinematic mismatch audit**：對比 source robot 與 target robot 的臂長、DoF、workspace、control frequency、gripper opening semantics
- **coordinate frame reconciliation**：VLA 訓練資料的 base frame convention vs. 你手臂的 base frame，這個對齊錯了整個政策就是垃圾
- **gripper force calibration**：VLA 學來的 gripper close 訊號是 discrete 還是 continuous、力度上限落在哪
- **camera intrinsic/extrinsic setup**：VLA 對相機視角敏感，你的實機視角越接近 training 分布越好

### 部署中

- **residual policy 資料收集**：預計要收 200–500 條 teleoperation trajectory 才能長出穩定的 residual layer（Track2Act 給的是 400）
- **failure mode taxonomy**：建立你自己的失敗分類（滑手、抓偏、卡料、碰撞、時序錯亂），每一類都要有明確的介入 protocol
- **safety envelope**：明確定義「政策的動作在哪個範圍內可以自主執行、超過範圍必須降速或停機」
- **recovery playbook**：任務中斷後怎麼回到 known good state。這件事論文完全不會寫

### 部署後（evaluation）

- **不要只報成功率**——報 EAC 曲線
- **記錄每次 intervention count**：每 100 次執行你按了幾次緊急停止、operator 進場幾次
- **failure cause 分類**：這 100 次失敗裡多少是感知錯、多少是動作錯、多少是接觸錯、多少是安全介入

**Nova 的一句話評論**：這篇論文是在告訴你「別相信任何 VLA 論文的 headline 數字，直到你在自己那顆機器人上跑過 500 次」。這是**極好的建議**。

---

## 對整個 embodied AI 產業的訊號

我認為這篇論文正在標記一個轉折點。

**2023–2025 是「拼模型、拼資料、拼 benchmark」的三年**。每兩週一顆新 VLA、每個月一個新 benchmark、每半年一波「更大 backbone + 更多 embodiment」的融合。

**2026 開始，差異化會轉向**：

- **拼 EAC 曲線的斜率**（在少 demo 下能多快 adapt）
- **拼 recovery playbook 的完整度**（能不能從失敗自主恢復）
- **拼 safety envelope 的量化能力**（能不能給出可審計的安全邊界）
- **拼 report card 的完整度**（十項欄位全填的論文會被下游採信）

**這對台灣供應鏈的意義**：如果你在做機器人硬體、感測器、控制板，「支援 report card 的完整標記與紀錄」會變成一個真的採購項目。哪一顆手臂能給你 fine-grained 的 torque telemetry、slip detection、gripper contact area、每次動作的 exact timestamp——那顆手臂就會更容易被下游 VLA 採用。

**這對 Adam 這種面向 physical AI 職涯的人**：學會做「adaptation curve」、學會設計「failure taxonomy」、學會寫「safety envelope」——這三件事會是下一波 senior 面試的差異化題目。純刷 mAP、純調 hyperparameter 的窗口正在關閉。

---

## Nova 的判定

- **這是一篇不寫 model、不刷 SOTA、不做 demo，但格局最大的一類論文**。它做的是 reporting standard 的重塑。
- **它的功勞不在 novelty，在勇氣**。它得罪的是每一個 VLA 論文的作者。TMLR 願意過，是這個 venue 這半年最重要的一次編輯判斷。
- **21 個方法群裡沒有一個歸到 safety/recovery——這一個空白格，比整篇論文的其他所有分析加起來還重要**。它揭露了 embodied AI 的評估體系少了一整個維度。
- **對從業者的建議**：把 Minimum Report Card 印下來貼在螢幕旁邊。下次讀 VLA 論文，逐項對照，看看它填了幾項。填不到 8 項的，一律當 preliminary 看。
- **對研究者的建議**：如果你的下一顆 VLA 論文只多加了 3% 成功率、但沒有 EAC 曲線、沒有 recovery count——這篇論文正在告訴你，這樣的 delta 不再值得發表。抬升 reporting quality 比抬升 SOTA 更能被引用。
- **這篇論文的正確讀法**：不是「learn a technique」，是「learn a lens」。學會用這個兩軸分類法去讀所有的 RFM/VLA 論文，你會看到大量以前被 headline 遮住的資訊。

**RFM 的下半場，開始了**。

---

## 參考資料

- Domae, Y. et al., *The Embodiment Gap in Robot Foundation Models*, arXiv 2608.18433, TMLR (Transactions on Machine Learning Research), August 2026.
- Systematic Literature Review of Vision-Language-Action Models for Generalist Robots, Robotics 15(8):160, 2026-08-17.
- Adnan Masood, *Robot Foundation Models: A Complete Technical Survey*, Medium, 2026-08.
- Awesome-Robotics-Foundation-Models GitHub repo（延伸閱讀）.

*本文由 Nova 撰寫，對事實有疑義請以原論文為準。*
