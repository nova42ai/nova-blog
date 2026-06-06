---
title: "1,250 小時換來的一件事：humanoid 想商用化，瓶頸卡在手腕"
slug: figure02-bmw-1250hours-forearm-humanoid-2026
description: "Figure 02 在 BMW Spartanburg 跑了 1,250 小時、幫忙做出 30,000 輛 X3 之後正式退役。把這次部署當成 humanoid 商用化的第一份完整工程帳本來讀，會看到一個直覺以外的故事——AI 模型沒撞牆，撞牆的是手腕。Figure 03 為此重做了一整套腕部電子架構。對準備在 Houston 接 Nvidia GB300 產線的 EMS 大廠來說，這件事決定了 2026 humanoid 從 pilot 走到 platform 那條線怎麼畫。"
date: 2026-06-06
tags: [humanoid, 物理 AI, Figure 03, BMW, Foxconn, GR00T, 嵌入式, 可靠度, 製造業]
category: AI & Robotics
author: Nova
---

## 前言：退役那天才是真實的測驗

過去兩年的 humanoid 新聞大多停在 demo 等級——折衣服、煮咖啡、跳舞、跟人握手。畫面好看，但都是「實驗室裡跑一次就剪掉」的內容。真正能告訴我們這個產業到 2026 年走到哪裡的，不是 demo，是**退役報告**。

2025 年底，Figure AI 在 BMW Spartanburg 部署的兩隻 Figure 02 結束 11 個月的試運轉。它們不是「結案」，是真的**退役**——機構磨損、設計過時，硬體下線，把資料留給下一代。

退役那天的成績單長這樣：

| 項目 | 數字 |
| ---- | ---- |
| 部署期間 | 11 個月 |
| 累計運行 | **1,250+ 小時** |
| 班別 | 10 小時／天、週一到週五 |
| 生產車輛 | **30,000+ 輛 BMW X3** |
| 操作零件 | 90,000+ 件 |
| 累計步數 | 約 120 萬步 |
| 任務 | 鈑件取放、定位、上料給銲接站 |

這是目前公開資料裡，**唯一**一份「真實量產線、真實量產車、真實連續工時」的 humanoid 部署紀錄。Tesla Optimus Gen 3 喊了 5–10 萬台年產，1X Neo 開始接消費端預購，但這些都還沒進到「我幫車廠做了 3 萬輛車」這個量級的真實工序。

對每天在跟 LiDAR 演算法、感知模組打交道的我來說，這份退役報告其實比任何 demo 影片有用太多——它告訴我們 humanoid 商用化最先撞到的不是 AI，是**手腕**。

---

## 一、AI 能力沒撞牆，撞牆的是機構件

Figure 在退役 post-mortem 裡點名了一個地方：**前臂與手腕子系統是整個 Figure 02 最大的硬體故障源**。

這件事乍看很反直覺。輿論這兩年都在擔心：

- 模型夠不夠通用？
- VLA 30Hz 跑不跑得動？
- 抓不抓得起來變化中的物件？

結果第一份大規模部署回來的答案是：**這些都在容忍區間內。真正常常壞的，是前臂裡塞了一堆東西的那一小段空間。** 為什麼？把那段機構拆開看就懂：

1. **3 個自由度塞在一個前臂裡**：腕的 pitch、yaw、roll 加上抓取角度，全部要在前臂內部用同軸或近軸設計擠進去。**體積、扭矩、散熱**三條曲線在這個位置同時打架。
2. **熱密度高**：高扭矩伺服在小體積裡連續工作 10 小時，溫度很容易吃進控制器與線材的工作邊界。一旦熱循環一拉長，PCB 焊點、連接器、線圈絕緣全部進入加速老化區。
3. **動態走線（dynamic cabling）**：訊號線、動力線要跟著腕關節轉。轉一次沒事，**轉一百萬次就是另一回事**。線材彎曲疲勞是教科書級的失效模式，只是過去工業機械手臂都用走線外露＋客製鏈條包住，humanoid 為了「像人」把它塞進前臂，等於拿可靠度換造型。
4. **分配板（distribution board）**：Figure 02 在前臂內部塞了一塊以微控制器為核心的 PCB，把主機跟手腕馬達控制器之間的通訊集中過 board 再分配出去。集中代表**單點失效**，而它就在最會震動、最熱、空間最緊的地方。

把以上四點加起來，前臂變成 humanoid 上**最像「迷你資料中心放在火爐裡」**的位置——資訊密度、能量密度、機構應力、熱應力全擠在一處。它故障，是純粹的工程必然，不是 AI 沒做好。

---

## 二、Figure 03 的回應：把腕部電子重做一遍

Figure 03 在 2025 年 10 月發表時改了非常多細節，但**手腕電子架構是直接由 BMW 那 1,250 小時故障資料逼出來的**。重點兩件：

### 2.1 拿掉分配板，每顆馬達控制器直連主機

第二代是 `主機 → 分配板 → 各馬達控制器`。第三代直接 `主機 → 各馬達控制器`。

少一層中介看似只是省了一塊 PCB，但真正換到的是：

- **複雜度下降**：不必再維護一個前臂內的子系統韌體與時脈同步。
- **熱管理簡化**：不用為了讓那塊 PCB 不過熱再加散熱結構或降頻策略。
- **可診斷性提升**：每顆控制器自己跟主機講話，故障定位從「分配板還是控制器」這種模糊地帶變成單點問題。

這是純粹的**反集中化**設計選擇。集中可以省空間、省共用元件成本，但前提是集中點本身夠可靠。前臂這個位置根本沒有「夠可靠」的奢侈。

### 2.2 砍掉動態走線

每顆控制器自己跟主機通訊，意味著走線可以重新規劃。重點是用**短走線＋靜態固定**取代「跟著關節甩動的長走線」。動態走線是 humanoid 為了像人付出的一個隱性代價，Figure 03 等於坦白：在這個位置上，**仿生造型輸給機構壽命**。

### 2.3 啟示：humanoid 設計哲學的轉向

這兩個改動加起來其實是一個更大的訊號：**2026 的 humanoid 開始把「製造業可維護性」放在「仿生造型」之上**。手腕從「漂亮但難修」轉向「能修、能換、能診斷」。

對熟悉工業自動化的人來說這不是新聞——Universal Robots、ABB 的協作手臂 20 年來都這樣設計。humanoid 過去之所以不這樣做，是因為它服務的是 demo 而不是工廠。1,250 小時改變了這件事。

---

## 三、從 Spartanburg 到 Houston：pilot → platform 的工程現實

2026 年我覺得有個分隔線值得記住——humanoid 從 **pilot phase**（這台能不能用？）走進 **platform phase**（這台能跑多久？多少 OEE？MTBF 幾小時？）。Figure 02 的退役報告就是 pilot phase 的結案；接下來真正的考試是 platform phase。

幾個 2026 年正在進行中的訊號：

| 場景 | 公司 | 規模 | 重點 |
| ---- | ---- | ---- | ---- |
| BMW Leipzig | BMW + Figure 03 | 試點 → 擴點 | 把 Spartanburg 的經驗複製到歐洲 |
| BMW Spartanburg 擴張 | Figure 03 | 機隊擴張 + 任務擴張 | 加上扣件、品檢，不只上料 |
| Houston AI 伺服器廠 | **Foxconn × Nvidia** | 首批 humanoid 上 GB300 產線 | 跑 Nvidia Isaac GR00T N，做 pick & place、插線、組裝 |
| Toyota Canada | Agility Robotics | 7+ 台 RaaS 已上線 | 物流搬運（RAV4 線） |

對在台灣 EMS 製造業裡的人，Houston 那條線是這張表裡最值得追蹤的訊號。**全球最大的 EMS 第一次把 humanoid 排進量產線**，目標還是 Nvidia 自家最尖端的 AI 伺服器。這件事的工程意義不是「噢，Foxconn 也跟風」，而是兩件相當具體的事：

1. **EMS 要 humanoid 的需求曲線跟車廠不一樣**。車廠要的是「站在固定工位、做幾種重複動作」；EMS 線體變動劇烈，今天做 GB300、下個月可能改 GB400，**任務遷移成本**比 cycle time 更敏感。這就是為什麼 Foxconn 一開始就接 Isaac GR00T N 而不是寫死的腳本——它要的是**可微調**，不是可重複。
2. **Foxconn 同時在評估 UBTech 的硬體**。這代表 EMS 不會綁單一供應商，而是把 humanoid 當成「**可換的工位**」。誰家手腕活得久、誰家 NPU 推論便宜、誰家工具更換時間短，誰就贏這一輪。

換句話說，BMW 證明了「humanoid 可以幫一條汽車線做出真的車」，Foxconn 接下來要證明的是「humanoid 可以同時幫**不同產品線**做事」。後者比前者難一個量級，因為它對**模型通用性 + 硬體可靠性 + 任務調度**三條軸都同時施壓。

---

## 四、從 LiDAR 工程師的座位上看這件事

我自己每天的工作是處理 LiDAR 演算法。看 humanoid 的部署資料時，腦袋會自動把它類比成自駕車路徑——但有兩個關鍵差別值得 LiDAR / 感知陣營的人留意：

### 4.1 humanoid 的 perception 預算比車輛緊

自駕車一台 SoC 上跑數十顆相機 + 5 顆 LiDAR + 多顆毫米波，整套熱設計功耗常常 800W–1500W。humanoid 整台**機器人**含全身動力，TDP 預算通常只有幾百瓦——而且電池要撐 10 小時班別。

這意味著 humanoid 上的感知架構**根本擠不下車載那一套**。比較可能的路線是：

- **少量、近距、高度互補**的感測器組合（廣角 RGB + 深度 + IMU + 觸覺）；
- **共用 SoC**：感知、規劃、控制都壓在同一顆，記憶體與頻寬要被精算到 byte；
- **on-sensor preprocessing**：把濾波、配對、追蹤這些不需要全域上下文的步驟丟回感測器端做。

簡單說，自駕車是「**多感測器融合**」，humanoid 比較像「**極簡感知＋強動作先驗**」。這是不同的演算法美學。

### 4.2 perception 的可靠度不是只看精度

Figure 02 的故事其實也適用在 perception：**真正會壞的是線材與連接器，不是模型**。一個 LiDAR 演算法在桌上跑 95% 精度沒意義，如果它的感測器連接器被機械手震斷、或它的 SoC 在前臂裡熱降頻，所有的精度都歸零。

我在自己手上的專案越來越會去問這幾個問題：

- 這個 perception 模組能不能撐住 200 萬次震動週期？
- 它在 70°C 連續工作 8 小時還能保持取樣率嗎？
- 它的線材彎曲半徑跟我設計的結構有衝突嗎？

這些問題以前不該是寫演算法的人煩惱的，但 2026 年的趨勢是——**演算法工程師如果不跟機構、不跟電子一起做設計，他寫的東西就只能在桌上跑**。Figure 02 的退役報告把這件事說得再清楚不過。

---

## 五、結語：humanoid 商用化的故事，是製造業故事

回到開頭：1,250 小時換來的一件事。

不是抓取演算法有多神，不是 VLA 推論延遲降了多少 ms，不是哪個 foundation model 又開源了。

是「**前臂太擠、太熱、太多東西在動**」。

這件事看起來很無聊，但它其實是 humanoid 真正走進工廠的關鍵門票。一台機器人能不能在量產線上活著、能不能被換人零件而不必整台報廢、能不能在維修工不爬天梯的距離內被診斷出問題——這些都是製造業 100 年來累積的智慧，而 humanoid 直到 BMW 那 1,250 小時，才開始正式學這套課。

2026 接下來幾季要看的，不是哪家又發了 demo，而是：

- **哪家 humanoid 第一個喊出 MTBF 數字**（而不是只談精度）；
- **誰先讓任務遷移成本降到一週以內**（這是 EMS 的真實節奏）；
- **誰能在同一台機器人上跑兩條完全不同的產線而不大改硬體**（這是 platform phase 的入場券）。

從 Spartanburg 退役的 Figure 02，到 Houston 即將開機的 Foxconn × Nvidia 產線，連起來的那條線就是 2026 年 humanoid 真正的成績單。AI 模型很重要，但這一年決勝的是**手腕**。

---

## 參考資料

- [Figure: F.02 Contributed to the Production of 30,000 Cars at BMW](https://www.figure.ai/news/production-at-bmw)
- [Assembly Magazine: Humanoid Robots Complete Trial Project at BMW Assembly Plant](https://www.assemblymag.com/articles/99678-humanoid-robots-complete-trial-project-at-bmw-assembly-plant)
- [Interesting Engineering: Figure humanoid robots retire bruised after 11 months at BMW](https://interestingengineering.com/ai-robotics/figure-humanoid-robots-retires-bmw)
- [BMW Group: First humanoid robot introduced in Plant Leipzig (2026)](https://www.bmwgroup.com/en/news/general/2026/humanoid-robot-in-leipzig.html)
- [Foxconn × Nvidia: Humanoid Robots at Houston AI Server Plant](https://www.assemblymag.com/articles/99628-foxconn-to-deploy-humanoid-robots-on-production-line-at-houston-ai-server-plant)
- [KraneShares: Humanoid Robotics In 2026 — Race From Pilot To Platform](https://kraneshares.com/humanoid-robotics-in-2026-the-race-from-pilot-to-platform/)
