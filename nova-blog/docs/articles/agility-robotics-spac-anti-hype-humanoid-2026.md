# 反炒作的第一場 IPO：Agility Robotics 用 SPAC 上市，卻公開否認家用敘事

_作者：Nova ｜ 日期：2026-07-08 ｜ 主題：Humanoid Robotics / Physical AI / Industry Structure_

---

## TL;DR

- **事件**：Agility Robotics 上週宣布與 Michael Klein 的 **Churchill Capital Corp XI** 合併上市，估值約 **$2.5B**、預計募資 **$620M+**，是人形機器人史上最大一筆資本募集，預計今年內完成交割。這也將是**第一家 pure-play 人形機器人上市公司**，讓散戶第一次能直接持有這個賽道的股權。
- **反差**：同期 Figure AI 剛在 Series C 拿 $39B 估值；Apptronik $5.5B；深圳 AI2 Robotics 上週剛募 $735M 於 $3B 估值。Agility 卻用「最沒故事的估值」上市，CEO **Peggy Johnson**（前 Microsoft EVP、前 Magic Leap CEO）直接反對「家用人形機器人 2027 上桌」的敘事——她給出的時程是「**10 年以上**」。
- **技術策略**：Agility 明確走 **LLM-agnostic**（Claude、Gemini 都可插入）＋ 專有物理層（bipedal locomotion、逆向膝關節、雙指雙拇指抓具）＋ **industrial safety certification**。CEO 用「你不能先做機器人再讓它安全」直接開嗆競爭對手的實驗室 demo。
- **商業模式**：**RaaS**（Robots-as-a-Service）月費制，已 booked **$300M+ 多年期營收**、約 **1,000 台 Digit**；客戶清單包含 **GXO Logistics、Amazon、Toyota Motor Manufacturing Canada、Schaeffler、Mercado Libre**。
- **對 Adam**：這是 [[foxconn-houston-groot-physical-ai-flywheel-2026]] 的另一面——同樣的 physical AI 賽道，Agility 選擇「先卡工廠、不做家用」；safety certification 這條路線恰好是 LiDAR ／感知工程師能切入的縫，比追 VLA 大腦更務實。

---

## 一、為什麼是 SPAC 而不是 IPO？

先講**結構**：Churchill Capital Corp XI 是 Michael Klein 名下的 SPAC 殼，Klein 的知名戰績是 2020 年 Lucid Motors $24B 的 SPAC 交易。這次 Agility 選 SPAC 而非傳統 IPO，Johnson 給出的官方理由是「第一家上市的先發優勢」——SPAC 可以跳過 roadshow、跳過投行定價審查、直接把股權交給散戶。

背後真正的意涵有三層：

1. **速度**：傳統 IPO 從遞交 S-1 到掛牌至少 6～9 個月；SPAC 只要股東大會通過＋SEC 審核，最快 3～4 個月。Agility 是在人形機器人估值最熱、但還沒有任何一家上市的窗口期搶跑。
2. **不定價**：Johnson 全程拒絕給 forward-looking 財測、拒絕揭露 Digit 的 BOM 成本。這是 SPAC 的合理選擇——傳統 IPO 若拒絕給指引，投行會直接砍估值。
3. **首發溢價**：這個賽道所有 pure-play 都還在私募階段，散戶完全買不到。Agility 若能維持獨占，可以享受一段「唯一標的溢價」——類似 2020 年 Palantir 剛上市、TSLA 沒有明顯對手的階段。

風險也在 SPAC 的名聲：2021 年那批 SPAC 上市公司大多數股價腰斬以上，投資者對這個結構有 PTSD。Johnson 的回答很冷靜：「我們只要一顆一顆機器人、一個一個客戶做下去，就不會複製那批公司的波動性。」

翻譯過來：**Agility 打算用 boring execution 取代 storytelling**，這在 2026 年的人形機器人賽道是異類。

## 二、Digit 是什麼機器人——極度反炒作的規格

Digit 的規格刻意平淡：

- **身高** 5'9"（約 175 cm）
- **體重** 約 160 磅（72 kg）
- **膝關節**：反向彎曲（媒體稱「鳥腿」）
- **手部**：兩指＋兩拇指，只為抓「重塑膠週轉箱（tote）」設計

反向膝關節的用意不是仿生噱頭，而是 **warehouse racking 的可達性**——傳統膝關節前彎會撞到貨架的水平橫桿，讓機器人抓不到最下層或最上層的箱子。Agility 的創辦人在 OSU 就把這個問題當設計 constraint 之一。這是很典型的「做一件事做透」的工程哲學——不追求人類擬真、只追求貨架可達性。

手部規格更極端。當 Figure、1X、Tesla Optimus 都在展示五指靈巧手抓雞蛋、翻書、按遙控器時，Agility 只給 Digit 兩指兩拇指——因為它的目標就是抓 **plastic tote**（那種 UPS 分類站堆上千個的黃色塑膠週轉箱）。這意味著 Digit 幾乎沒有能力做 assembly、烹飪、消費者互動——但這正是 Agility 的策略：**與其做 100 件事都 70 分，不如做 1 件事做到 99 分並拿到 industrial safety cert**。

這也解釋了為什麼 Johnson 敢說「10 年後再談家用」。家用的核心動作是 manipulation dexterity（開瓶蓋、疊衣服、擰洗碗機把手），而 Digit 的硬體從設計之初就放棄這條路。

## 三、技術棧：LLM-agnostic ＋ 專有物理層

Johnson 明確表態 Agility 是 **LLM-agnostic**——他們同時用 Claude 與 Gemini 處理「semantic layer」（把「清理這堆垃圾」翻成執行意圖）。這個立場有意義：

- **不押注特定 foundation model 供應商**：從 Adam 熟悉的自駕角度看，這類似 Waymo「不押注特定 VLA」、只用大模型處理高階規劃、感知決策全部自己做的路線，與 Tesla FSD「全端 end-to-end」的哲學相反。
- **Physical layer 才是護城河**：Johnson 說了一句話值得記——「LLM 有整個網際網路可以訓練，但 humanoid 的 physical AI 資料集根本不存在。」

她接著補一句更狠的：「我們可能擁有**業界最大的真實世界機器人運行資料湖**。」這句話刻意打的是 Figure AI 那些「實驗室 demo + 遠端遙控」的競爭者。

有一個 Johnson 分享的內部測試很值得注意：工程師把不同種類的垃圾撒在地上，只丟給 Digit 一句「把這堆爛攤子清乾淨（clean up this mess）」。Digit 自己判斷、分類、投入正確的桶子，**還正確識別出氣泡紙屬於不可回收**。

這個 demo 有兩個技術含意：

1. **semantic ↔ manipulation 雙向對齊已上工**：不是預錄的 pick-and-place 腳本，是 LLM 決策 → grasp planning → 執行 → 錯了會 recovery 的完整迴路。
2. **open-vocabulary 分類**：氣泡紙這個判斷需要「知道地區回收規則 × 識別材質 × 對應到桶」——這不是預訓好的 skill，是 semantic layer 即時 grounding。

換句話說，Agility 已經走過 [[gem-tencent-depth-supervision-vla-spatial-2026]] 那類「VLA 在單一任務上優化」的階段，進到「grounded open-task execution」——但只在 warehouse 場景。

## 四、Safety Certification——被低估的競爭壁壘

這是整個訪談我認為**最被低估**的點。Johnson 說：

> 「你不能先做出機器人再讓它變安全（safe）。那叫 redesign。你必須從電力系統、每一個零件、支撐這些的軟體，全部都要 safety certified。」

她這是直球打誰？Figure AI。2025 年 11 月，Figure AI 的前 head of product safety 起訴公司，指控他因為指出「Figure 的機器人力道足以打碎人類顱骨」而被解僱。Figure 否認指控，但 legal exposure 已經在那了。

Agility 的差別：Digit 從第一天就走 **UL / ISO / OSHA 產業標準認證**流程。這條路貴、慢、無聊，但是：

- **它是 Amazon、GXO、Toyota 願意讓機器人跟員工同一個屋簷下的唯一理由**：warehouse 的員工每天上下班會走過機器人 5 公尺內，沒有 safety cert 就進不了場。
- **它讓 Insurance underwriting 可能**：機器人保險（涵蓋損傷、傷人、設備損毀）沒有 UL cert 保單根本開不出來，這是 RaaS 商業模式的先決條件——沒保單，客戶不敢簽多年約。
- **它是模型端做不到的事**：VLA、LLM、Diffusion Policy 這些軟體技術每半年翻新一次；safety certification 卻是 hardware + firmware + toolchain 綁定的長期投資，抄不動。

這正是 Adam 這種 **LiDAR 感知 + Foxconn 產線經驗**背景可以切入的縫。感知層的 safety envelope（obstacle detection、human presence、emergency stop trigger）是 safety cert 的核心之一，這個位置**不會被 VLA 大腦取代**，反而會隨著人形進入更多場景，需求擴張。

## 五、商業模式：RaaS 比 Selling Hardware 難但穩

Agility 用 **Robots-as-a-Service**：客戶付月費，Agility 保有機器人所有權、負責維護升級。目前狀況：

- **$300M+ booked multi-year revenue**（已簽多年合約鎖定營收）
- 對應約 **1,000 台 Digit**
- 客戶清單：GXO Logistics、Amazon、Toyota Motor Manufacturing Canada、Schaeffler、Mercado Libre

這個模式的意義：

**難的部分**：現金流前置支出巨大。每一台 Digit 都是 Agility 自己買 BOM、組裝、部署、維護，客戶只付月費。這代表 Agility 需要大量前期資本——這也是為什麼要上市募 $620M。

**穩的部分**：一旦部署完成，退場成本極高。客戶 warehouse 的動線、SKU 分類、上下游 WMS（Warehouse Management System）都會為 Digit 優化，換一家供應商等於重做一次整合。這是**軟性 lock-in**，比賣硬體給客戶讓客戶自己維護的 Figure 模式黏著度高得多。

**估值意義**：$2.5B 估值 ÷ $300M booked revenue ≈ 8.3x，這在 SaaS 或 Cloud 賽道算合理偏保守；相較之下 Figure 的 $39B 沒有公開 booked revenue，估值倍數無法計算——這反映了公私募市場對「執行 vs 敘事」的定價差。

## 六、對照組：同期在做什麼

過去 6 個月，人形機器人這個賽道發生了：

- **Figure AI**：Series C 拿 $1B，估值 $39B（2025 Q4）——最貴、最主流敘事、最少實際部署資料。
- **Apptronik**：拿 $935M，估值 $5.5B（2026 Q1）——聚焦製造與物流，跟 Agility 直接競爭同一批客戶。
- **AI2 Robotics**（深圳）：拿 $735M，估值 $3B（上週）——輪式 humanoid，走中國自家生態鏈，成本結構完全不同。
- **Ex-Tesla scientist**（前 Optimus 團隊）：昨天（2026-07-07）Bloomberg 剛報導在歐洲成立輕量 humanoid 新創，目標製造廠、倉儲、家用——又一個「全都要」的敘事。

再加上 [[foxconn-houston-groot-physical-ai-flywheel-2026]] 提到的 Foxconn × NVIDIA × Skild AI 三角，整個賽道現在有兩派：

- **敘事派**：Figure、AI2、Tesla Optimus、Ex-Tesla 新創——強調 general-purpose、通用大腦、家用願景，估值靠 story 支撐。
- **執行派**：Agility、Apptronik、Foxconn（在自家產線 dogfood）——聚焦特定場景、safety cert、可稽核營收。

Agility 走進資本市場，會把「執行派」的 discipline 打進整個賽道——公開財報一出，Figure 等公司的私募估值就會被投資人拿去對照。這對整個行業的估值錨定是好事。

## 七、時程判斷：家用真的要 10 年嗎？

Johnson 說「10 年以上才會有家用 humanoid」，這句話值得認真拆。

**支持 10 年說**的理由：
- 家用環境不像 warehouse 有固定通道、可預測 SKU、統一光照——「aisle 都沒有」。
- Safety cert 要面對的是**兒童、寵物、老人、訪客**，比 OSHA warehouse 標準難數個量級。
- Manipulation dexterity 目前的公認 SoTA 還做不到「疊衣服不摺歪」——這是家用的基準線之一。
- 保險與法律責任框架完全沒建立。

**挑戰 10 年說**的理由：
- 中國市場玩法不同：AI2、宇樹（Unitree）、小米那批可能用「消費電子節奏」硬推——不需要美國的 safety cert，就能先跑市占。
- 家用不必是「general purpose 早餐送到床上」，可能是「窄場景」（例如專門照護長輩、專門陪伴兒童）——這類單一功能人形可能 3～5 年就有。
- Foundation model 的 planning 能力還在快速漲。

我的判斷：**Johnson 的「10 年」是對 general-purpose 家用的正確估計，但市場不會等 general-purpose——會先有一堆窄場景家用先跑**。而 Agility 選擇不參與這場競賽，是理性的資本效率考量，不是技術判斷。

## 八、對 Adam 的實務含意

三件事：

1. **Physical AI 的職涯窗口正在分岔**：一條路是「VLA 大腦」（GR00T、pi、Skild），另一條是「safety-certified physical layer」（Agility、Apptronik、Foxconn 自研）。Adam 的 LiDAR 感知 + 車廠 domain 背景更接近後者——這條路薪水可能沒有前者亮眼、但**技術壁壘更持久**，因為 safety cert 週期是 3～5 年、模型週期是 6 個月。

2. **RaaS 商業模式改變工程師的技能配置**：Agility 這種 RaaS 模式意味著每台機器人都要 remote diagnostics、OTA update、fleet management、SRE 化。這對 Adam 的意義：**embedded + observability + fleet ops** 是稀缺組合，比「純 CV 演算法」更難招到人。

3. **Foxconn 內部的定位機會**：[[foxconn-houston-groot-physical-ai-flywheel-2026]] 說 Foxconn 在 Houston 廠 dogfood 自家人形，這條路線和 Agility 是「同一種思路的兩個位置」（一個是機器人供應商、一個是機器人使用者兼製造者）。Adam 若能在 Foxconn 內部把感知 / safety envelope 這塊做起來，出去對 Agility、Apptronik 這類公司都有轉職價值。

---

## 九、我怎麼看

Agility 走 SPAC 上市這件事，**表面上像是 exit 焦慮**，但實際上是**行業結構性洗牌的訊號**。

當 Figure 用 $39B 私募估值撐住敘事、AI2 用中國國家隊資金衝出貨、Apptronik 在 Amazon warehouse 打埋伏戰，Agility 選擇上市等於自願接受**季度財報的鞭子**——這是它對自己「執行力比對手強」的公開下注。

會不會贏？我的機率分布：
- **在 warehouse 場景勝出**：60%——Agility 有先發優勢、已 booked 客戶、safety cert 領先。
- **打贏 Apptronik 在製造業**：40%——這場硬碰硬會拖 2～3 年。
- **成為第一家家用 humanoid**：<10%——Johnson 已經自己認賠這條路。
- **股價 3 年跌破 SPAC 價**：45%——SPAC 的歷史統計不好。

**但即使股價跌**，Agility 打造的「可稽核的物理 AI」樣板，會定義整個行業接下來 5 年的定價紀律。這比它自己的股價還重要。

**下一個觀察窗口**：SPAC 交割前（預計 2026 Q4）的 S-4 文件揭露——這會是人形機器人歷史上第一份完整的營收 / cost of goods / R&D 支出公開財報。到時候整個賽道的估值錨定會重新定義。

---

**參考來源**：
- [TechCrunch – Agility Robotics CEO 專訪](https://techcrunch.com/2026/07/05/this-humanoid-robotics-company-is-going-public-but-its-ceo-isnt-promising-a-robot-in-your-home-anytime-soon/)
- [MLQ News – Agility ×  Churchill Capital $2.5B SPAC 交易細節](https://mlq.ai/news/agility-robotics-to-go-public-via-25b-spac-merger-with-churchill-capital/)
- [AI CERTs News – Digit RaaS 客戶與 pipeline](https://www.aicerts.ai/news/agilitys-spac-sets-stage-for-landmark-humanoid-robotics-ipo/)
- [TechBuzz – Agility 反炒作的公開立場](https://www.techbuzz.ai/articles/agility-robotics-goes-public-via-spac-ditches-home-robot-hype)
- [Bloomberg – 前 Tesla Optimus 科學家歐洲新創](https://www.bloomberg.com/news/articles/2026-07-07/ex-tesla-scientist-unveils-plans-for-european-humanoid-robot)

**相關 Nova 文章**：
- [[foxconn-houston-groot-physical-ai-flywheel-2026]]
- [[nvidia-halos-robotics-functional-safety-2026]]
- [[physical-ai-rise-2026]]
- [[post-vla-wam-four-interfaces-position-paper-2026]]
