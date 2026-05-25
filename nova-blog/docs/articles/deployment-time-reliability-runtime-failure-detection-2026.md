---
title: "Demo 裡能跑，產線上會死：機器人策略的『部署時可靠性』正在成為新瓶頸"
slug: deployment-time-reliability-runtime-failure-detection-2026
description: "2026 年機器人圈的好消息與壞消息是同一件事：策略終於『能動了』，但離『能上班』還很遠。JAL 在羽田、Boston Dynamics 的 Atlas 都進了真實場域，可是最先進的 VLA 模型在沒見過的指令上開箱成功率只有 30–60%，人形機器人撐 30–90 分鐘就要人介入。瓶頸已經從『模型會不會做這件事』移到『我們能不能在它搞砸之前察覺並攔下來』。這篇拆解部署時可靠性、執行期失敗偵測（runtime failure detection）這個正在浮現的工程層，以及它和 SRE 可觀測性的相似之處。"
date: 2026-05-25
tags: [部署可靠性, VLA, 失敗偵測, Runtime Monitoring, OOD, Physical AI, 機器人學習, 可觀測性]
category: AI & Robotics
---

## 前言：好消息和壞消息是同一句話

2026 年機器人領域最誠實的一句總結是：**策略終於「能動了」，但離「能上班」還很遠。**

好消息很顯眼——五月，日本航空（JAL）聯手 GMO AI & Robotics，在羽田機場啟動了為期兩年的人形機器人實證實驗（採用 Unitree G1 平台），任務涵蓋行李搬運、貨櫃運送與客艙清潔；這是日本機場首例，而且是持續性的場域試點，不是發表會上的一次性表演。Boston Dynamics 的電動 Atlas 也走向商業部署，2026 年的產能據報多由 Hyundai 與 Google DeepMind 的合作消化。

壞消息藏在數字裡：**最先進的 VLA（Vision-Language-Action）模型，在沒見過的指令上開箱評測，真機成功率只有 30–60%；** 多數人形機器人連續運作 30–90 分鐘就需要充電或人工介入；2026 年沒有任何一家廠商做到「全自主機隊頂掉一整班人類」——每一個「部署」其實都是有現場工程團隊盯著的有限試點。

把這兩件事擺在一起，結論就浮出來了：**機器人的瓶頸，已經從「模型會不會做這件事」，移到「我們能不能在它搞砸之前察覺並攔下來」。** 這篇文章講的，就是這個正在成形、但還沒有名字的工程層——我暫且叫它**部署時可靠性（deployment-time reliability）**。

---

## 一、為什麼「能做」不等於「能部署」

把一個學習出來的策略丟進真實世界，它要面對三件實驗室裡被馴化掉的事：

1. **開放式變異（open-ended variability）**：真實場景的物件、光照、佈局是長尾的，demo 裡的整齊桌面在產線上不存在。
2. **分布偏移（distribution shift）**：部署時看到的輸入，和訓練資料的分布永遠對不齊。
3. **誤差累積（compounding errors）**：閉環控制裡，一個小偏差會把系統推到更陌生的狀態，下一步偏更多——這是模仿學習的經典老毛病。

這三件事疊起來，就是為什麼一個在錄影裡行雲流水的策略，換個工廠、換句指令就崩。**問題不在峰值能力，而在尾部行為。** 而尾部恰恰是部署最在意的地方：一次抓取失敗在 demo 裡是重拍一條，在客艙裡可能是砸壞東西或傷到人。

這也解釋了一個看似矛盾的現象：論文裡 success rate 的數字一直在漲，但真實部署仍然離不開現場工程師。因為**平均成功率高，不代表失敗時是安全的、可預測的、能被即時攔下的。**

---

## 二、新瓶頸：不是「會不會」，是「知不知道自己要砸了」

如果你做過系統，這個轉變會很眼熟：它和軟體從「功能能跑」走到「線上要 SRE」是同一個故事。功能寫完只是起點，真正讓系統能上線的，是**可觀測性（observability）**——你得能在故障擴大之前看到它。

機器人策略現在缺的就是這一層。所以 2026 年一個明顯升溫的研究方向，是 **執行期失敗偵測（runtime failure detection）**：不是讓策略更強，而是給策略裝一個「快不行了」的警報器。它的價值主張很務實——**一個 55% 成功率、但失敗時 95% 能提前喊停的策略，遠比一個 70% 成功率、但失敗時毫無徵兆的策略更能部署。**

這個方向有幾條技術路線，我整理成下表：

| 路線 | 核心想法 | 代表工作 | 限制 |
|---|---|---|---|
| **時序動作一致性** | 偵測閉環行為的「抖動 / 自相矛盾」，作為 erratic failure 的訊號 | Sentinel（arXiv:2410.04640） | 對「平順但走錯方向」的失敗不敏感 |
| **任務進度監控** | 用 VLM 判斷策略的動作有沒有在「解這個任務」 | Sentinel 同篇 / SAFE（arXiv:2506.09937） | 依賴 VLM 的判斷品質與延遲 |
| **序列式 OOD 偵測** | 把失敗偵測框成「輸入是否離開訓練分布」的時序 OOD 問題 | FAIL-Detect（arXiv:2503.08558） | OOD ≠ 一定會失敗，會有誤報 |
| **不確定性估計** | 不需要失敗資料，用策略自身的不確定度當訊號 | FAIL-Detect 同篇（arXiv:2503.08558） | 校準難，過度自信的模型會騙過它 |
| **本體遙測** | 監控馬達溫度、扭矩、功耗，做預測性維護 | 產業界 dashboard 實務 | 抓硬體劣化，抓不到「決策錯誤」 |

值得注意的關鍵設計目標：**好幾條路線都刻意做到「不需要失敗資料」（failure-data-free）。** 原因很現實——真實失敗樣本稀少、昂貴、而且收集它本身就有風險。能在「沒看過失敗長什麼樣」的前提下偵測失敗，才是能規模化的方向。

---

## 三、誠實的 trade-off：警報器自己也有兩種錯

我不想把 runtime monitoring 寫成銀彈。任何一個偵測器都站在一條經典的張力上：

- **漏報（false negative）**：該停沒停，失敗照樣發生——可靠性層形同虛設，最糟時造成實體損害。
- **誤報（false positive）**：沒事卻喊停，產線被無謂打斷——uptime 掉、人要跑來確認，部署經濟性垮掉。

產業客戶期待 95–99% 的 uptime，而誤報直接吃這個數字。所以**部署時可靠性的真正難題，不是「能不能偵測失敗」，而是「能不能在漏報與誤報之間找到一條讓經濟性成立的工作點」。** 這條工作點還會隨場景滑動：客艙清潔可以容忍多停幾次（安全優先），高節拍產線則對誤報極度敏感。

這就把問題從一個純 ML 指標（AUROC 多高），拉回到一個系統工程問題：**偵測延遲、攔停機制、人類介入成本、失敗後果的嚴重度**，這四個東西要一起算，才知道一個監控器到底「值不值得裝」。

---

## 四、對做感知 / 嵌入式的工程師意味著什麼

如果你和我一樣，日常在感知與嵌入式那一端，這個趨勢有三個具體的 take-away：

1. **可靠性層會吃你的算力預算。** 監控器是要和主策略一起跑在 edge 上的——VLM 進度判斷、不確定性估計都不便宜。這和 [on-sensor / edge 感知](on-sensor-perception-lidar-edge-2026.md)、[neuro-symbolic VLA 把能耗砍 100×](neuro-symbolic-vla-energy-100x-2026.md) 的主線是同一場資源戰爭：**「會偵測失敗」和「跑得起」必須同時成立。**

2. **OOD 偵測是感知人的主場。** 序列式 OOD、不確定性校準，本來就是感知與 sensor fusion 的老題目。把「這幀點雲 / 影像離訓練分布多遠」的能力，接到「策略要不要喊停」的決策上，是一個很自然、也很有價值的接口。

3. **它和 World Model、即時控制是互補的，不是競爭的。** [World Model](world-models-robot-learning-learned-simulators-2026.md) 想用「做夢」放大資料、補長尾；[Sony 那類即時感知-控制](sony-ace-realtime-perception-control-2026.md) 壓低迴路延遲。但這兩條都還是在「讓策略更好」。Runtime monitoring 是承認**策略永遠不會完美**，於是退一步問：那它出錯時，系統能不能優雅地知道並收手。前者是 [agentic 控制層](bayesian-control-layer-agentic-ai.md) 的事前規劃，後者是事中防護——一個健康的部署堆疊兩者都要。

---

## 結語：部署時代需要的是 SRE，不只是更強的模型

2026 年最該被記住的一句話，可能是某份產業報告講的：**「基礎時代結束了，我們進入部署時代——挑戰不再是讓機器人動起來，而是讓它負責任地、和人並肩地行動。」**

而「負責任地行動」翻成工程語言，就是可觀測性、失敗偵測、優雅降級這一整套 SRE 思維，被搬到了學習型策略上。模型會繼續變強，這沒有疑問；但讓 JAL 的機器人能真的撐完一個三年合約、讓 Atlas 能脫離現場工程師的盯梢，靠的不會只是更高的 success rate，而是這個還沒被充分重視的**部署時可靠性層**。

下一個值得做的人，也許不是訓練更大的 VLA，而是給它寫一個夠聰明的警報器。

---

## 參考來源

- [Deployment-Time Reliability of Learned Robot Policies (arXiv:2603.11400)](https://arxiv.org/abs/2603.11400)
- [SAFE: Multitask Failure Detection for Vision-Language-Action Models (arXiv:2506.09937)](https://arxiv.org/pdf/2506.09937)
- [Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress (arXiv:2410.04640)](https://arxiv.org/pdf/2410.04640)
- [Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection (arXiv:2503.08558)](https://arxiv.org/pdf/2503.08558)
- [National Robotics Week — Latest Physical AI Research（NVIDIA Blog）](https://blogs.nvidia.com/blog/national-robotics-week-2026/)
- [Humanoid Robotics In 2026: The Race From Pilot To Platform（KraneShares）](https://kraneshares.com/humanoid-robotics-in-2026-the-race-from-pilot-to-platform/)
- [What Is Humanoid Robot Safety? Why Real-World Deployment Is Still Years Away（MindStudio）](https://www.mindstudio.ai/blog/humanoid-robot-safety-real-world-deployment)
- [Japan's First Demonstration Experiment for Utilizing Humanoid Robots at Airports（JAL Group Press Release）](https://press.jal.co.jp/en/release/202604/009502.html)
- [Japan Airlines begins humanoid robot trials at Haneda（CNBC）](https://www.cnbc.com/2026/05/01/japan-airlines-humanoid-robots-haneda-labor-shortage.html)
