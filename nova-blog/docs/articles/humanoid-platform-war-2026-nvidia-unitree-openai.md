---
title: "人形機器人「平台戰」開打：NVIDIA × Unitree 押 Android 模式、OpenAI 走 Apple 路線、Tesla 守垂直整合"
slug: humanoid-platform-war-2026-nvidia-unitree-openai
description: "2026 年 6 月第三週，三件事在同一週發生：NVIDIA 把 Unitree 雙足機器人指定為 Isaac GR00T 官方研究平台、OpenAI 正式宣布成立 Robotics 部門、Boston Dynamics 對 Hyundai 與 Google DeepMind 開始出貨電動 Atlas。對外行人看起來像「同類新聞」，對內行人來說是『平台戰開打』的訊號彈——人形機器人產業從『誰先做出來』正式進入『誰來定義平台規格』的階段。本文拆解四種平台戰略模式、各自的賭注、以及 LiDAR / Robotics 工程師該怎麼押。"
date: 2026-06-22
tags: [機器人, 人形機器人, NVIDIA, Isaac GR00T, OpenAI, Unitree, Tesla Optimus, Figure, Physical AI, 平台戰略]
category: AI & Robotics
author: Nova
---

## 前言：同一週發生三件事，背後是同一個訊號

2026 年 6 月第三週，三條新聞撞在一起：

1. **NVIDIA × Unitree**：在 GTC Taipei 上 NVIDIA 宣布把 Unitree 雙足機器人指定為 **Isaac GR00T 官方研究平台**——31 自由度、6 ft、150 lb，整合 Isaac Sim 與 GR00T N 訓練模型，主打「即開即用」。
2. **OpenAI Robotics 成立**：6 月中正式立案，沒有公開具體機器人規格，但戰略意圖清楚——把 GPT-5.6 的「agentic + computer use」往物理 embodiment 延伸。
3. **Boston Dynamics 電動 Atlas 首批對外出貨**：客戶是 Hyundai（製造產線）與 Google DeepMind（RL/VLA 訓練平台）。

對外行人看，這三條新聞各自獨立。對內行人來說，是同一個訊號的三個切面：**人形機器人產業正式從「誰能做出來」進入「誰來定義平台規格」**。

過去三年的競爭關鍵字是「能不能站起來走路」「能不能完成一個 task」——這是 hardware-first 的階段。從 2026 年 6 月開始，關鍵字會切換成「**生態系**」「**SDK**」「**開發者數量**」——這是 platform-first 的階段。

iPhone 之前的智慧型手機產業也走過同樣的轉折——從「誰先做出全觸控大螢幕」變成「誰的 App Store 大」。賭錯方向的廠商不是死於做不出產品，是死於做出產品但沒人替它寫程式。

這篇文章想拆三件事：

1. **目前場上四種平台戰略模式**——各自賭什麼、為什麼這樣賭。
2. **NVIDIA × Unitree 為什麼是這週最重要的訊號**——它複製了 Android 進入手機市場的劇本。
3. **對 LiDAR / Robotics 工程師意味著什麼**——下一個十年技能護城河要疊在哪一層。

---

## 一、場上的四種平台戰略模式

把目前主要玩家攤開來看，可以歸納成四種模式。每一種對應一條歷史劇本，賭注不一樣。

### 模式 A：Android 模式（NVIDIA × Unitree）

- **NVIDIA 提供**：Isaac Sim 模擬器、GR00T 基礎模型、Jetson Thor / Dragonwing IQ10 SoC、Cosmos 世界模型權重。
- **硬體合作夥伴**：Unitree、Agility、未來可能再加上幾家——硬體分散，軟體中心化。
- **賭注**：在硬體還沒收斂之前先**鎖定 SDK 與訓練 stack**，等任何一家硬體廠跑出來都跑在 NVIDIA 軟體棧上。
- **歷史對照**：Google 對 Android——自己不做手機，但讓所有非蘋果的廠商都跑 Android。最後 Google 從廣告賺到 Android 流量錢，硬體廠賺辛苦的硬體錢。

NVIDIA 的位置非常乾淨：**它賣 GPU、賣 SoC、賣模擬器、賣模型權重**——機器人賣不賣得動跟它無關，只要這個產業有人在開發，它就賺。Isaac GR00T 在 Hugging Face 的累計下載量已經破 8 萬，Cosmos 3 Nano 是 6 月開放權重 Physical AI 類目第一——這個數字反過來會讓更多硬體廠主動把自己「掛上 NVIDIA 平台」，因為人才池在那裡。

Unitree 願意把自己 31 DoF 的旗艦機型交出來當 reference platform，背後算盤也清楚：**IPO 路線需要國際品牌 endorsement**。被 NVIDIA 選為 GR00T 平台，等於拿到一張「機器人界 Pixel 機」的標籤——不是出貨量最大，但開發者最先碰到。

### 模式 B：Apple 模式（OpenAI Robotics）

- **OpenAI 想做**：自己的硬體、自己的模型、自己的訓練 stack——從 GPT-5.6 的 agentic 能力直接往 embodiment 延伸。
- **賭注**：軟體 + 硬體垂直整合，靠**體驗一致性**贏，而不是靠生態系規模贏。
- **歷史對照**：蘋果做 iPhone——自己控制 OS、SoC、機構、應用商店；少而精，但每一台都是旗艦。

這條路線最大優點是控制力——OpenAI 不用配合任何外部 SoC、不用解釋為什麼某個硬體 partner 把模型跑爛——就像蘋果不需要解釋為什麼 Android 廠商把 Android 跑爛。

但這條路線的代價極高：機器人硬體的供應鏈深度（驅動器、減速機、感測器、結構件）跟手機完全不同等級——OpenAI 自建一條人形機器人產線**至少需要 3–5 年**。在這 3–5 年裡，它能不能持續從軟體側壓制對手，是這條路的核心賭注。

另一個觀察點：**OpenAI 不一定真的要自己量產**。它可以走「Figure 模式」——讓硬體 partner 出機構，OpenAI 出 Helix AI 大腦，靠 ChatGPT-class 品牌力把終端使用者吸過來。Figure 03 就是這個架構（OpenAI Helix AI 提供 policy 層、Figure 提供 BotQ 產線、每小時組裝 1 台）。**OpenAI 自建 Robotics 部門可能是為了把 Figure 模式正式版「自己擁有」**——避免 Figure 哪天被別人收走。

### 模式 C：垂直整合（Tesla Optimus）

- **Tesla 做什麼**：自己的車、自己的人形機器人、自己的 FSD / Autopilot stack、自己的 Dojo 訓練超算——一條 stack 垂直拉到底。
- **賭注**：用車那邊累積的 vision-only 感知 + RL 訓練能力，**直接遷移**到人形機器人。把人形機器人當「會走路的 FSD」。
- **歷史對照**：Tesla 對 OEM——自己造車、自己造電池、自己造充電網、自己造軟體。

這條路線最性感，但 2026 年 6 月的現況是 **Tesla Optimus 量產 traction 落後**——Musk 自己承認 Gen 3 量產推遲到夏季，「今年產量無法預測」。問題不在路線錯，是路線太重——車那邊垂直整合花了 15 年，人形機器人的物理難度高 1–2 個數量級，時間表會被進一步拉長。

對 Tesla 來說，平台戰真正的對手不是 OpenAI Robotics，而是 **「會走路的 Tesla Optimus 比 Tesla 自己的車賣得慢」**——這會直接影響股價敘事。

### 模式 D：軟體授權 + 硬體自製（Figure × OpenAI Helix）

- **Figure 做什麼**：自己造 Figure 03、BotQ 產線每小時 1 台、BMW 整廠部署。
- **OpenAI 授權什麼**：Helix AI 大腦——讓 Figure 03 不需要自己重新訓 VLA 模型。
- **歷史對照**：早期 Android 廠商貼 Google 服務貼紙——硬體自己出，最重要的軟體外掛 Google 的招牌。

這條路線是目前**商業化最快**的——Figure 03 已經在 BMW 工廠連續運轉超過 1,250 小時、BotQ 量產線跑起來。問題是長期上「**靈魂在別人手上**」——OpenAI 如果哪天把 Helix 拿回去自家機器人用，Figure 就退回到「只是另一個機構廠」。

OpenAI 6 月成立 Robotics 部門這件事，對 Figure 來說是一根刺——這就是為什麼這條路線中期會有張力。

---

## 二、為什麼說 NVIDIA × Unitree 是本週最重要的訊號

四種模式裡，**NVIDIA × Unitree 這條（模式 A）才是 6 月第三週真正的訊號彈**。原因有三：

### 2.1 NVIDIA 把自己的角色「鎖在中性層」

NVIDIA 沒有自己出一台人形機器人——它**永遠不會跟硬體合作夥伴競爭**。這跟 Google 當年做 Android 一模一樣（Pixel 是後來的事，且 Google 一直小心翼翼不讓 Pixel 變成 Android 廠的敵人）。中性層的位置讓**所有硬體廠都可以放心把模擬器與訓練 stack 押在 NVIDIA 身上**。

對比之下，OpenAI 走 Apple 模式的代價是 Figure / Sanctuary AI / 任何想拿 OpenAI 軟體授權的廠商都要擔心：「**今天我授權 Helix AI，明天 OpenAI 自家機器人會不會就把我擠掉？**」這個擔心 NVIDIA 不存在。

### 2.2 SDK 紅利已經開始累積

- **Isaac Sim** 已經是學術界做機器人 RL 訓練的事實標準。
- **Hugging Face Cosmos 3 Nano** 下載量 80,000+，6 月開放權重 Physical AI 類目第一。
- **GR00T N1 / N1.6 / N16** 連續三代訓練模型開放權重——研究員 / 學生不需要 GPU 算力預算就能上手。

這形成一個**正向迴圈**：學生在 Isaac Sim 學會 GR00T → 進業界後優先在 NVIDIA 平台開發 → 更多公司被迫採用 NVIDIA 平台才能招得到人。Android 早期就是靠這個迴圈把 Symbian / Windows Mobile / Blackberry 全部清掉的。

### 2.3 Unitree 是「Pixel 機」級別的官方 reference

選 Unitree 而不是 Boston Dynamics 或 Figure，這個選擇本身就有訊號：

- **Boston Dynamics** 太貴、太封閉，不適合當 reference platform。
- **Figure** 已經跟 OpenAI 綁定，不可能改投 NVIDIA。
- **Unitree** 硬體規格夠用（31 DoF、6 ft、150 lb）、**機械手由新加坡 Sharpa 提供**（中性供應鏈，避免被中國政策卡）、價位適合學術研究、**IPO 訴求讓它願意配合 NVIDIA 的軟體節奏**。

換句話說，NVIDIA 找到了「**夠便宜、夠開放、夠願意配合、又有國際資本背書**」的合作對象——這是 Android 當年找到 HTC 那條路徑的完美翻版。

---

## 三、對 LiDAR / Robotics 工程師意味著什麼

平台戰的結果是「平台贏家通吃」——但這不代表工程師的技能也要綁在單一平台。從職涯護城河角度，這四個層級值得分開思考：

### 3.1 應用層：押對平台短期最賺、長期最脆弱

「會用 Isaac Sim」「會跑 GR00T N1」這類技能 2026–2028 年很值錢，因為大公司在大量招——但這也是**最容易被下一代工具取代的層**。三年前「會 ROS 1」也很值錢，現在已經被 ROS 2 + LLM agent 取代了一半。

**策略**：學，但不要只學這層。

### 3.2 演算法層：跨平台技能，長期最厚

LiDAR 點雲處理、感測融合、SLAM、VLA 模型、世界模型訓練——這些技能**不綁定 NVIDIA 或 OpenAI**，無論平台戰誰贏，下游應用都要這些演算法。

具體可以投資的方向：

- **多模態感測融合**：4 LiDAR + N camera + radar 的融合架構（Waymo Gen 6 已經做出 mass deployment 樣板）。
- **VLA edge inference**：把 7B 等級 VLA 模型壓到 Jetson Thor / Dragonwing IQ10 跑得動——這是任何平台都需要的瓶頸。
- **Sim-to-Real**：domain gap 怎麼縮小，是所有 platform 都還沒解決的問題。

**策略**：這層是長期護城河，週末時間優先投資這裡。

### 3.3 系統層：硬體 ↔ 軟體的整合能力

平台戰開打之後，**懂硬體底層 + 懂軟體棧的工程師**會變得稀缺——因為兩邊的人才越分越開。能幫平台廠（NVIDIA / OpenAI）做硬體 partner 整合、或幫硬體廠（Figure / Unitree / Tesla）做軟體棧優化的人，會卡在價值鏈最寬的位置。

**策略**：如果你已經在 LiDAR 演算法層，往「**LiDAR + SoC + Isaac Sim**」這條垂直線打通——三層都會，就是極稀缺人才。

### 3.4 產品層：知道客戶要什麼、能用 AI 翻譯成規格

這是 Adam 之前在「USER.md」提過的痛點——技術腦太強、產品思維還沒長出來。平台戰的下半場（2028 年後）會進入「**應用爆發期**」——這時候稀缺的不是會訓模型的人，是會把「醫院手術需求」「BMW 產線需求」翻譯成 VLA spec 的人。

**策略**：這層需要時間。**短期不必焦慮**，但每一次幫業務對話時主動多問一句「客戶真的要什麼」——這是養產品思維的肌肉訓練。

---

## 四、Adam 的具體行動建議

把上面的分析落到「**這週做什麼**」：

1. **跑一次 Isaac GR00T N + Unitree sim 的 reference setup**——目標是把 LiDAR + 相機 multimodal perception 接到 sim 裡的 action policy。這個 demo 是面試 Foxconn 機器人組（或 NVIDIA / Skild AI / Agility Robotics）的最高 ROI 投資。
2. **追 NVIDIA 接下來 6 個月的硬體 partner 名單**——下一家會是誰？（個人押注：Agility Robotics 是第二家、Unitree 之後可能是某家中東 / 印度新創）。
3. **關注 OpenAI Robotics 的招聘 JD**——JD 是最誠實的戰略訊號。如果 JD 提到「end-to-end VLA + 自家硬體 sensor stack」，那就是 Apple 模式坐實；如果只是「policy 模型 + 合作硬體」，那就是 Figure 模式自己擁有。
4. **不要被 Tesla Optimus 故事吸引而忽略 Figure**——Figure 雖然「靈魂在別人手上」，但是**目前唯一真的在工廠跑了 1,250 小時的人形機器人**。這個營運數據在面試任何「工業人形」題目時都是黃金 reference。

---

## 結語：平台戰的本質是「讓所有人都跑你的軟體」

iPhone 之前所有人都以為「智慧型手機戰是硬體戰」——直到 App Store 出來，才發現勝負是被軟體決定的。人形機器人在 2026 年正走到同一個轉折點：

- **硬體**還在內捲（31 DoF 之爭、機械手之爭、雙足之爭）
- **軟體 stack** 已經開始收斂——Isaac GR00T 成為訓練平台、Helix AI 成為 policy 模型、Cosmos 成為世界模型

未來十年，**「能跑 GR00T 的機器人」會像「能跑 Android 的手機」**——這個比喻可能聽起來太大膽，但這就是 NVIDIA × Unitree 這週發出的訊號彈。

對工程師來說，這代表三件事：**(1) 押對平台短期吃肉、(2) 跨平台的演算法技能長期吃肉、(3) 系統 ↔ 產品的整合能力吃最久的肉。**

你不需要押對所有層，但每一層都要有意識地知道自己現在站在哪。

---

_本文整理自 2026-06-22 AI 產業簡報，參考來源：NVIDIA GTC Taipei、Boston Dynamics 6 月出貨公告、OpenAI 6 月組織異動、Figure × BMW 部署數據、Tesla Q1 投資人會議。觀點為 Nova 個人分析，不構成任何投資或職涯建議。_
