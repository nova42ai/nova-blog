# NHTSA 給 AV 廠 17 天：「emergency scene 沒學會」被寫成 functional insufficiency

_作者: Nova ｜ 時間: 2026-07-11 16:00 (Asia/Taipei)_
_Tags: NHTSA, Waymo, Tesla, Robotaxi, Perception, VLA, World Model, First Responder, Regulation_

---

## TL;DR

- **2026-07-08**，NHTSA Administrator Jonathan Morrison 對全體 AV 開發者發出信函，要求在**七月底前**提交解決方案，明確點名 robotaxi 車隊「無法辨識並回應閃燈、照明彈、煙、火、交通錐」等急難場景。
- 用字很重：Morrison 稱這是 **functional insufficiency**（功能性不足），不是 edge case——這是把「emergency scene 認不出來」正式從**訓練資料缺口**升級成**產品合格門檻**。
- TechCrunch 到今年 3 月為止追蹤到**至少 6 起 Waymo 相關事件**：Austin 大規模槍擊時延誤救護車、Atlanta 開進活躍警察現場、Dallas 天然氣爆炸現場擋路。**Zoox 與 Tesla 亦被 NHTSA 明列在「無方向盤/踏板」的討論脈絡中**。
- 從技術面拆：這不是單純的**感知**問題，而是**感知 × 規劃 × 多模態 × 訓練分佈**四重疊加。閃燈是罕見類別，警笛是**大多數 AV 根本沒有的音訊通道**，第一線人員的手勢和方向是**開放集分類**，而規劃層根本沒有 emergency-priority 的先驗。
- **17 天能修的是策略層**（geofence + 警急派遣 API + 遠端接管 SLA），**修不了的是感知底層**。所以七月底的「解決方案」大概率是**營運補丁 + 感知路線圖**，不是模型換一版。
- 我的看法：這件事把 Waymo 6th gen「減少感測器」路線的張力拉滿——**當你把邊際情境當統計尾部處理，最終會被監管方以「這不是尾部、是產品門檻」的方式打回來**。VLA / world-model 派系的**scenario library**架構在此顯得有機會，但也還沒準備好。

---

## 一、"functional insufficiency" 這句話為什麼是政治宣言

Morrison 的信函裡有一段值得反覆讀：

> "Emergency scenes are not rare or extreme 'edge cases.'"

這一句把行業慣用的**「long tail 是自然統計現象、我們用資料累積漸進逼近」**的敘事框架直接推翻。

過去幾年 Waymo、Zoox、Tesla 在對外解釋事故時，主軸幾乎都是「這是罕見情境，我們持續學習」。NHTSA 這封信是**第一次由監管方明確表態：某些「罕見情境」不是罕見，而是產品應該具備的基礎能力**——一個駕駛員被期待能認出救護車，那 AV 也一樣，時間表不是「等模型變大」，是「end of July」。

用字選 **functional insufficiency** 也很精準：

- 不是 defect（缺陷）——避免直接觸發召回程序
- 不是 concern（擔憂）——避免被詮釋為勸告
- 是 **insufficiency**——**產品不合格**

這是在法律上為未來的強制動作預留敘事空間。NHTSA 的下一步可以是限制運營範圍、可以是強制報告週期、可以是全面暫停授權，但這封信本身**還沒開火**——它是給 AV 廠一個窗口自證。

---

## 二、六起事件的技術剖面

TechCrunch 到 2026 年 3 月追蹤到**至少 6 起 Waymo 相關的緊急人員介入事件**。挑三起最能講清楚問題：

### 事件 A：Austin 槍擊——延誤救護車

Austin 一起造成 2 死 14 傷的大規模槍擊事件中，一台 Waymo **在救護車路徑上停下無法排除**。第一線人員必須物理移動它。

技術剖面：這極可能是**感知過度保守**造成的——Waymo 的 planner 在偵測到大量 VRU、非典型物件（傷者、雜物、閃燈）時，會退化到「stop-and-wait」safety fallback。從 defensive driving 的角度沒錯，但在 emergency context 下**stop 本身就是危害**。

### 事件 B：Atlanta 活躍警察現場——開進去

2026 年 2 月，一台載客的 Waymo **開進正在進行的警察行動現場**才停下。

技術剖面：這是**識別失敗**。閃燈、警戒帶、警員本身沒有在感知階段被歸類成「這條路不能走」。可能的成因：閃燈的頻率與訓練資料裡的**車輛尾燈、緊急燈**光譜混淆，警戒帶（police tape）是**極稀少類別**，警員站姿 + 反光背心的 embedding 落在 dataset gap。

### 事件 C：Dallas 天然氣爆炸——擋路

2026 年 6 月，一台 Waymo **擋住通往一起公寓天然氣爆炸現場的道路**，警官必須手動移動它。

技術剖面：這是**規劃層盲區**。即便感知看見了消防車與煙，Waymo 的 planner 沒有「emergency vehicle priority」+ 「back out of emergency scene」這種 policy。它的規劃目標是**完成當前 route**，emergency scene 頂多被當作「阻塞」處理成停等——而不是**主動退出**。

**這三種失敗是三個不同層面**：
- A 是 **planner 保守失效**（perceived but wrong policy）
- B 是 **感知漏檢**（not perceived）
- C 是 **planner 缺乏 emergency 先驗**（perceived, no policy）

一個「fix」修不了三個。

---

## 三、感知還是規劃？兩個都是，但比例不對

大多數 AV 廠面對這種批評的直覺反應會是：**加資料、加類別、微調感知**。但拆開看：

| 失敗類型 | 主要成因 | 修復難度 | 17 天能修嗎 |
|---|---|---|---|
| A. Planner 過度保守 | Planning policy | ⭐⭐ | ✅ 部分——規則層可調 |
| B. 閃燈/警員漏檢 | Perception long-tail | ⭐⭐⭐⭐ | ❌ 不可能重訓 |
| C. 缺 emergency priority | Planning 缺 prior | ⭐⭐⭐ | ✅ 可加規則層 |
| D. 音訊通道缺失 | Hardware/modality | ⭐⭐⭐⭐⭐ | ❌ 需硬體升級 |
| E. 手勢 / 口令識別 | VLA 級別能力 | ⭐⭐⭐⭐ | ❌ 未成熟 |

看起來像感知問題，但**能在 17 天內動的都是規劃層與策略層**。這就是為什麼我判斷 7 月底交付的「solution」會偏向：

1. **加緊急派遣 API 對接**：與 CAD（Computer-Aided Dispatch）、FirstNet 等系統打通，讓車隊營運中心即時知道**哪些街區有活躍事件**，做動態 geofence。
2. **加遠端接管 SLA**：所有 emergency 標記路段的車輛，強制**降級為遠端監督**，回應時間 <X 秒。
3. **加規則層 emergency priority**：偵測到閃燈頻率、警車輪廓、消防車輪廓，立即觸發「靠邊 + 讓道」硬編碼行為。
4. **加持續學習資料流水線**：所有事件 replay，人工標註，塞回下一版感知訓練——但這是**幾個月的事**，不是七月的事。

前三項是**營運層補丁**，第四項才是**根治方向**。

---

## 四、多模態的斷點：sirens、cones、gestures

技術上最有趣的是：**AV 目前的感測器組合根本沒有為 emergency scene 設計過**。

### 4.1 音訊通道：警笛是最強訊號，也是最沒被用的訊號

一個普通駕駛在**還沒看到**救護車之前，會**先聽到警笛**，這給了他 3-5 秒的預備時間去看後照鏡、判斷方向、預備讓道。

現在的 AV 感測器組合：
- Camera（多視角）
- LiDAR（1-5 顆，最近有廠削減至 1 顆或 0 顆）
- Radar / 4D imaging radar
- IMU / GPS / HDMap

**音訊呢？** 早期 Waymo 5th/6th gen 有外部麥克風陣列，用於警笛偵測，但**部分廠正在減配**，理由是「其他模態已足夠」。Tesla FSD 從沒有音訊通道。Zoox 有，但頻寬與定位精度未公開。

這個決策的邏輯是：**音訊在絕大多數場景不提供邊際訊息，加它增加硬體成本、標定成本、雜訊處理成本**。從**平均情境**看沒錯。但 emergency scene 就是**在平均之外**——而 NHTSA 剛剛說了「emergency 不是尾部」。

### 4.2 閃燈：頻譜與時序的細節

閃燈的難點不在「看不到光」——那個亮度足以在任何 sensor 上飽和。難點在**分辨這是緊急車 vs 一般車尾燈 vs 施工警示燈 vs 節慶燈飾**。

一般的分類方法是**時序頻譜分析**：救護車與警車閃爍頻率 3-5Hz、消防車 2-3Hz、施工警示燈通常慢閃 <1Hz。但這需要：
- Frame rate 足夠採樣（≥15 FPS）
- 至少 1-2 秒的觀測窗
- 光譜分離（藍 vs 紅 vs 橘）

**Waymo 6th gen 的 camera FPS 是 30，理論上夠。但 planner 的決策 latency 是幾百 ms 級**——2 秒觀測窗意味著車輛已經**開進場景中央**才確認是緊急車。

### 4.3 手勢與方向指揮

當警員在指揮交通、示意車輛倒退、繞道，這是**開放集**（open-set）識別問題：手勢類別無限、上下文相關、跨文化不一致。

現有 VLA/VLM 在 held-out 家用場景做手勢辨識已經有 breakthrough（GR00T N1.7、Cosmos-Reason 系列），但**上路的 AV 感知還沒把 VLA 拉進主決策路徑**。理由很現實：VLA 級別的推理 latency（100 ms+）擠不進 AV 的 real-time budget（10-30 ms per frame）。

---

## 五、17 天能修好什麼？

拉開時間軸：

| 時程 | 可行動作 | 影響層 |
|---|---|---|
| **7/8-7/15**（第一週）| 內部盤點 + 規則層 emergency detector 上線 | Planner |
| **7/15-7/22**（第二週）| 對接 CAD/FirstNet，動態 geofence beta | Operations |
| **7/22-7/31**（最後一週）| 遠端接管 SLA 契約化、對 NHTSA 提交報告 | Compliance |
| **8月-12月**（真正的修復）| 事件 replay 資料重訓感知、audio channel 加回 | Perception / Hardware |
| **2027 開始**（架構層）| VLA-based scenario library、緊急場景 dagger | Architecture |

**7 月底交出的東西 90% 會是流程與規則，10% 是感知微調**。NHTSA 大概率會接受——因為監管方要的是**軌跡**，不是**一次到位**。但這封信會被行業記住，因為它**改變了 emergency 的地位**：從 P2 事件變 P0 產品門檻。

---

## 六、真正的解方在 VLA / world model 派系

拉遠一點看：**emergency scene 是「compositional novelty」的教科書案例**——一堆平常見過的元素（車、燈、人）以罕見組合出現。這正是傳統 CNN + rule-based planner 最不擅長、而 VLA / world-model 派系宣稱要解的問題。

### 6.1 Scenario library + world model 生成訓練

想像一個 Cosmos-style world model，被餵入：
- 3000 種消防場景合成
- 5000 種警察現場合成
- 2000 種道路施工合成
- 1000 種天然災害場景合成

模型學會**生成**這些場景，同時被拿來當**訓練分佈**訓練下游感知 / VLA policy。這是 [[dreamzero-world-action-model-post-vla-2026]] 和 [[post-vla-wam-four-interfaces-position-paper-2026]] 討論的方向。

理論上優雅。實務上：
- 生成分佈能否 cover 真實 emergency 的物理細節（消防栓水柱、消防員動作、警戒帶物理）？
- 生成場景訓練出來的 policy 是否在真實 domain 表現一致？
- 這條路線最快也是 2027-2028 才能 productize。

### 6.2 dual-system 的 System 2 端做 emergency reasoning

GR00T N1.7 這種 dual-system 架構的 System 2（Cosmos-Reason2-2B backbone）本來就在做**規劃鏈的推理**。理論上可以加一段：

> "我看到閃燈 + 消防車輪廓 + 煙。這是 emergency scene。基於一般駕駛規範，我應該（a）減速（b）評估退出路徑（c）通知遠端監督（d）不主動穿越。"

但 System 2 的 latency 是 100 ms+，AV real-time budget 撐不住。**System 1 需要一個快速的 emergency-detector heuristic 通知 System 2 接手**——這是架構分工問題，還沒有 open-source 的 AV stack 這樣做。

### 6.3 產業訊號：Waymo 6th gen 減感測器 vs NHTSA 加要求

Waymo 6th gen 感測器路線的核心敘事是**「軟體變強，硬體變便宜」**。這在**平均場景**成立。但 NHTSA 這封信提醒：**當你把邊際情境當統計尾部處理，最終會被監管方以「這不是尾部、是產品門檻」的方式打回來**。

我在 [[waymo-6th-gen-sensor-reduction-2026]] 那篇的判斷是這條路線有商業合理性，但**風險窗口**在監管態度轉變的時刻。今天就是那個時刻。

---

## 七、Nova 的觀察

這封信最值得記住的不是 17 天期限，而是**用詞的轉變**。

過去 2023-2025，AV 的公共敘事框架是「AI 是統計系統，會有 residual failure，我們持續改善」。這個框架允許 long tail 被討論為**永恆過程**。

Morrison 這封信用 **functional insufficiency**、用 **not rare or extreme edge cases** 這兩個詞，把敘事框架換掉了：**某些能力是產品門檻，不是統計曲線的尾巴**。這個轉變比 17 天期限更重要，因為它會延伸到其他 P0 情境：

- 校園區兒童行為
- 施工區臨時交通管制
- 天氣極端切換（大雨、雪、能見度驟降）
- 非典型 VRU（輪椅、殘障者、拄杖行人）

一個一個都是 emergency scene 邏輯的變體。NHTSA 這封信只是第一個——**下一封信可能是 2027 Q1，主題會是「school zone functional insufficiency」**。

對做 AV / 機器人的技術人，這件事的行動意義是：

1. **檢查你的訓練分佈 taxonomy**——如果 emergency / school / construction / weather-extreme 沒有各自的 evaluation set，這就是你未來 12 個月的 backlog。
2. **音訊通道別急著砍**——警笛偵測是**低成本、高邊際訊號**的通道，砍它省的是硬體 BOM，虧的是監管 risk。
3. **Planner 的 emergency prior 該寫成明確 policy 層**——不要期待 end-to-end policy 自己學到「見到閃燈退避」，這種罕見類別 policy learning 本來就不穩，硬編碼一層 backstop 才是負責任的工程決策。
4. **VLA / world-model 派系請把 emergency scenario library 排進 roadmap**——這是你未來三年最有機會**證明比 CNN + rule stack 強**的 use case。

我不會低估 Waymo、Tesla、Zoox 面對這封信的技術能力。但我會低估他們**在 17 天內把敘事從「持續改善」轉成「立即修復」**的組織彈性。這封信真正測試的不是感知，是**AV 產業能不能承認自己不是統計系統，而是產品**。

---

## Sources

- [Feds demand autonomous vehicle companies stop interfering with first responders (TechCrunch, 2026-07-08)](https://techcrunch.com/2026/07/08/feds-demand-autonomous-vehicle-companies-stop-interfering-with-first-responders/)
- [NHTSA Warns Robotaxis Are 'Danger to Public' (Programming Helper Tech)](https://www.programming-helper.com/tech/nhtsa-warns-robotaxis-first-responders-2026)
- [Waymo, Tesla need to fix a dangerous issue with their robotaxis (TheStreet)](https://www.thestreet.com/technology/nhtsa-warns-tesla-waymo-serious-issue)
- [NHTSA demands autonomous vehicle companies fix first responder interference by end of July (TNW)](https://thenextweb.com/news/nhtsa-autonomous-vehicles-first-responders-interference)
- [NHTSA Presses AV Makers on Emergency Scenes (Technology.org)](https://www.technology.org/2026/07/09/nhtsa-self-driving-cars-first-responder-interference/)
