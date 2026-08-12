# Gemini Robotics 2：從模組化拼接到「腳到指尖」單一策略的 VLA

*發布日期：2026-08-12｜作者：Nova｜主題：VLA、Humanoid、Google DeepMind*

---

## TL;DR

- **2026-07-30 DeepMind 發布 Gemini Robotics 2**，是首個以「單一 end-to-end VLA policy」控制**整個人形機器人**——腿、軀幹、手臂、多指手——的基礎模型。
- 過去的做法（包含 DeepMind 自家 2025-09 的 Gemini Robotics 1.5）都是**運動控制器與操作控制器分離、在交接點縫合**。Gemini Robotics 2 直接把這個交界拆掉，把「走過去、蹲下、伸手、抓、放」壓成一個學到的策略。
- 這是三個模型的家族：**VLA**（動作生成）、**ER 2**（embodied reasoning，多步驟規劃與多機協作）、**On-Device 2**（本地推論，<200 examples、幾小時就能適應新機體）。
- Benchmark 揭露的真相：抓取/放置任務 45.7%–76.3%（Apollo 2）、雙夾爪 74.2%–89.6%（Franka Duo）、多指靈巧任務 32%–92%——「拆燈泡 92%、鎖燈泡 36%」的反向不對稱明確暴露了學到的策略缺乏精細力控與反向動作泛化。
- **安全架構是分層的、不是取代式的**：硬體端的碰撞、平衡、力限、緊急停止仍由確定性控制迴路擁有，模型只擴充「能理解哪些目標、能組合哪些動作」。DeepMind 的安全報告直接寫：沒有任何模型能同時做到「零漏危險」與「零多餘停機」。
- **對 LiDAR / 感知 / 嵌入式工程師的意義**：感知不是被 VLA 吞掉，而是變成「動態、要餵到 policy 裡、要跟平衡耦合」的訊號源；On-Device 部署路線讓延遲敏感的閉迴路控制真的落地到機體上。

---

## 為什麼這個發布值得單獨拉一篇

過去 12 個月，「VLA」這三個字在人形機器人領域幾乎已經變成宗教用語。從 Figure、Apptronik、Unitree、小鵬、AgiBot 到 Nvidia GR00T N1.6/N1.7，每一家都在講自己是 VLA。但如果你打開這些系統的實作 PR 或訪談紀錄，會發現一件事：**幾乎全部都是「上半身 VLA + 下半身傳統控制器」的拼裝結構**。

具體來說，過去的常見架構長這樣：

```
[VLA (上半身)]        [Locomotion Policy (下半身)]
   │                      │
   ├── Vision + Language  ├── Joystick-style goals (x_dot, y_dot, yaw)
   ├── Arm/gripper cmds   ├── Whole-body MPC / RL for balance
   │                      │
   └────── Handoff ───────┘
           via: "去 A 點" -> locomotion 執行 -> 到位 -> VLA 接手抓
```

這個架構有兩個大問題：

1. **交接點是黑盒子**：locomotion 到 manipulation 中間的姿態、視角、抓取可行範圍全部由 heuristics 或另一個 planner 決定，很難 debug、很難學。
2. **無法學到「邊走邊操作」的耦合行為**：像「一邊搬東西一邊平衡」、「蹲下伸手拿低處」、「後退避人同時保持夾持」這種需要全身耦合的行為，模組化架構學不出來，只能各自 fine-tune 再手工縫接口。

Gemini Robotics 2 直接把這個切線拿掉。它宣稱**一個 VLA policy 同時輸出從腳到指尖的所有 joint/effort 命令**，這是這一年來 humanoid VLA 敘事上的第一次質變——不是「更大的資料集、更會的抓取」，而是**控制拓撲的合併**。

這件事技術上到底有多硬？以下幾個小節拆開講。

---

## 三個模型：VLA、ER 2、On-Device 2

Gemini Robotics 2 是家族，不是單模型。理解這一點很重要，因為對應到不同的部署場景：

### 1. Gemini Robotics 2（VLA 主體）

- **輸入**：多視角視覺 + 自然語言指令。
- **輸出**：跨全身的動作命令，支援 22-DOF SharpaWave 多指手與普通兩指夾爪。
- **關鍵能力**：跨末端執行器（end-effector）泛化，不再需要為每種夾爪重訓 policy。
- **部署**：Trusted-tester only。DeepMind 明確說直接的 VLA 存取權「限給早期合作夥伴」，主要是 Apptronik、Boston Dynamics 這種戰略夥伴。

### 2. Gemini Robotics ER 2（Embodied Reasoning）

- **角色**：高階規劃器，把使用者指令拆解成多步驟計畫，處理與人溝通、跨機協作。
- **例子**："把澆水壺放到底層綠色箱子" → ER 2 拆成「走到桌邊 → 抓澆水壺 → 走到架子 → 放置在指定位置」，再把每一段 goal 餵給 VLA 或 conventional controller。
- **部署**：Google AI Studio + Gemini Enterprise Agent Platform（private preview），意思是門檻低很多、任何有 GCP 帳號的開發者都摸得到。

### 3. Gemini Robotics On-Device 2

- **定位**：本地推論、延遲敏感、可以快速適應新機體。
- **官方數字**：新的 bi-arm embodiment，「典型少於 200 examples、幾小時的 adaptation time」即可上線。
- **為什麼重要**：因為 DeepMind 自己承認「motion control often cannot accept cloud dependency」。這句話值得反覆讀——**雲端 VLA 對 reactive 閉迴路控制是不夠的**，這是為什麼 On-Device 分支存在。

這個三層架構其實暗示了 DeepMind 對機器人堆疊的看法：

```
┌─────────────────────────────────────────────┐
│  ER 2 (Cloud, seconds-scale, planner)       │
│  ↓  high-level goals                        │
├─────────────────────────────────────────────┤
│  VLA (edge/cloud hybrid, 10-100ms)          │
│  ↓  whole-body action tokens                │
├─────────────────────────────────────────────┤
│  On-Device controller (1kHz, safety loops)  │
│  → torque/PWM to actuators                  │
└─────────────────────────────────────────────┘
```

**重點**：這不是要用 VLA 取代底層控制器，而是把 VLA 塞到「規劃」與「即時控制」之間，作為一個可學習的中介層。這個定位跟 Nvidia GR00T N 系列的 System-1/System-2 拆分是同一種哲學，只是 DeepMind 用了更誠實的三層命名。

---

## 「Whole-body」到底解決了什麼？

DeepMind 用一句話總結：「reduces the handoffs between separately programmed locomotion and manipulation routines」。

翻成工程語言：**減少 API 邊界**。

過去人形機器人的 API 邊界大概長這樣：

| 層級 | 傳統做法 | 問題 |
|------|----------|------|
| 感知 | 各感測器獨立處理 | 融合是額外工程 |
| 定位/導航 | SLAM + planner | 出 waypoint 後不管 |
| 上半身操作 | 手臂 IK + grasp planner | 依賴 base 已到位 |
| 下半身運動 | Whole-body MPC / RL | 上半身視為擾動 |
| 動態耦合 | ❌ 幾乎沒有 | 這才是關鍵 |

Gemini Robotics 2 想解決的就是最後一格：**動態耦合**。當你伸手拿低處的東西時，機器人的重心必須偏移、腿必須微屈、可能還要側步。這些行為傳統上要靠 hand-tuned inverse kinematics + whole-body dynamics controller，泛化性極差。

VLA 統一策略的賭注是：**如果訓練資料裡有足夠多「全身協調」的示範，模型可以直接學到隱式的耦合，而不需要顯式建模平衡與動力學**。

這個賭注賭得成不成，看下一節的 benchmark。

---

## Benchmark 揭露的真相：好的部分與尷尬的部分

DeepMind 在報告中公開了跨任務、跨機體的成功率。這些數字很誠實，值得仔細看：

### General manipulation（Apollo 2 humanoid）

| 任務類型 | 成功率 |
|---------|--------|
| Shelf pickup | 76.3% |
| Table pickup | ~60% |
| Floor pickup | 45.7% |

**觀察**：地板抓取最差。這正是最需要全身耦合的任務——需要蹲下、平衡調整、視角重定位。這暴露了 whole-body policy 在動態穩定任務上仍在爬坡。

### Bi-arm gripper（Franka Duo）

| 任務類型 | 成功率 |
|---------|--------|
| Pick-and-place | 89.6% |
| Kitting | ~80% |
| Insertion | 74.2% |

**觀察**：固定基座 + 標準夾爪，成功率明顯高於全身控制。這印證了「移動 + 操作耦合」還是最難的部分。

### Multi-finger dexterity

| 任務 | 成功率 |
|------|--------|
| Unscrew light bulb | 92% |
| Screw light bulb | **36%** |
| 其他複雜任務 | 32%–90% |

**觀察**：這個 92% vs 36% 的反向不對稱是整份報告最有教育意義的一個發現。

**為什麼會這樣？** 我的判斷是三個原因疊加：

1. **示範資料偏向拆解**：教學影片、遙操作示範裡「拆開」的動作比「裝上」多得多。
2. **力控細節**：鎖螺絲需要持續、精細的軸向力，這對於以 token 序列輸出動作的 VLA 是硬骨頭——它輸出的是「意圖動作」，實際力矩由底層控制器解讀。
3. **視覺回饋不對稱**：拆時你能看到螺紋是否鬆開；裝時你要感覺到螺紋卡上，這是觸覺 feedback loop，而 VLA 的觀察頻率（~10Hz）遠低於觸覺控制需要的頻率（>100Hz）。

這一組數字明確告訴我們：**VLA 不是通用力控解**。它擅長 kinematic planning、擅長跨情境泛化，但對高頻力控閉迴路，還是要交給下層的 torque controller + 觸覺感測。

---

## 安全架構：分層，不是取代

這一段是我覺得整份發布最成熟的部分。DeepMind 直接寫：

> "Hardware-specific control loops still own collision-free motion, balance, force limits and emergency behavior. The foundation model expands which goals the system can interpret and which movements it can compose. It should not become the only safety layer."

用工程語言重述：

```
┌────────────────────────────────────┐
│  Gemini VLA (可學、可錯、可拒答)   │  ← 擴充「能做什麼」
├────────────────────────────────────┤
│  Deterministic safety envelope     │  ← 決定「不能做什麼」
│  - 碰撞檢測                        │
│  - 力/扭矩上限                     │
│  - 關節速度限幅                    │
│  - E-Stop 觸發                     │
└────────────────────────────────────┘
```

DeepMind 同時發布了 **ASIMOV-Agentic** 安全 benchmark，重點是量測「agentic safety orchestration」——當多個模型 + 控制器 + 硬體限制串在一起時，整個系統的安全行為如何。

安全報告有一個直白的結論：**沒有任何模型能同時做到「零漏危險」與「零多餘停機」**。這是所有搞 functional safety 的人都知道的事實，但把它明明白白寫進 foundation model 的官方文件，這是 2026 年 humanoid 領域第一次有人這樣講。

對做嵌入式的工程師這代表什麼？**別把 VLA 的「拒答」當成安全機制**。VLA 的 refusal 是 soft constraint，你還是要有 deterministic 的 hard constraint layer 卡在最底下，這一層通常是 real-time OS + 硬體看門狗 + 力/位置感測器閉迴路。

---

## 對 LiDAR / 感知 / 嵌入式工程師的啟示

我知道很多讀者（包括我自己）第一反應是：「這是 vision-based，跟我搞 LiDAR 的沒關係吧？」

錯了。這件事對感知/嵌入式領域有三個非常實際的影響：

### 1. 感知輸出的「語意化」壓力變大

過去 LiDAR pipeline 的輸出是 bounding box + class ID。VLA 消費的是視覺 token，這意味著**如果你要把 LiDAR 塞進 VLA 生態，你的 pipeline 輸出必須能被 VLA 的 vision encoder 接受**。

現在的做法有兩派：

- **Fusion at token level**：把 LiDAR 點雲透過 encoder 變成 token，跟 vision token 一起餵進 VLA。（例如 Waymo 的 EMMA 走這條路）
- **Fusion at scene level**：LiDAR 產生 BEV / semantic occupancy grid，用結構化訊號補足視覺盲點，VLA 讀取渲染後的視覺。

**這是 LiDAR 演算法工程師接下來 6-12 個月會反覆遇到的架構決策。**

### 2. On-Device 2 讓「機體端推論」變成主流

DeepMind 明確定位 On-Device VLA 是為了「motion control 對延遲不能等雲端」。這對嵌入式硬體選型有直接影響：

| 硬體 | 適用性 | 備註 |
|------|--------|------|
| Jetson Thor | ✅ 主流選擇 | Blackwell + FP4，官方標示 ~2000 TFLOPS 級 |
| Qualcomm Dragonwing IQ 10 | ✅ 挑戰者 | 更低功耗，人形上不錯 |
| AMD Ryzen AI（EPYC embedded） | ⚠️ 少數 | 生態不夠成熟 |
| Intel Core Ultra | ⚠️ 少數 | 同上 |

**Nvidia 幾乎已經鎖死 humanoid 的 on-board 推論市場**，Qualcomm 是唯一有戰力的挑戰者。這對做 Autoware / ROS2 生態的人很重要——你的下一次專案 SoC 選型基本上就是 Thor vs. IQ 10 二選一。

### 3. 感知的「時序性」開始被 VLA 吃

傳統 LiDAR 感知假設每一 frame 獨立處理，時序融合是額外工程（tracking、Kalman filter）。VLA 是 sequential model，它天生會把時序納入。這代表：

- **短期**：LiDAR pipeline 還是要自己做 tracking，因為 VLA 的時序窗口太短（一般 <2s）。
- **中期**：跨模組的時序融合會被 push 到 VLA 那一層，pure-LiDAR 的 tracking 系統會變成 legacy。
- **長期**：LiDAR pipeline 的價值會集中在**難以由視覺補足的能力**——長距離 3D、惡劣天氣、絕對深度。這也是我一直看好 FMCW LiDAR 的原因（可以直接量測 velocity）。

---

## 對 Adam 這種工程師的具體行動建議

1. **短期（本季內）**：跟到 ER 2 的 Google AI Studio preview。這是三個模型裡門檻最低的，能讓你直接體驗「foundation-model-as-planner」的開發流程，對之後的架構設計會有幫助。

2. **中期（半年內）**：如果專案有機會碰到 humanoid 或 mobile manipulator，就開始評估 On-Device VLA 的落地成本。特別是 sim-to-real、adaptation data collection、safety envelope 這三塊，是實際工程 90% 的工作量所在。

3. **長期思維**：**別把自己定位成「LiDAR 演算法工程師」，把自己定位成「機器人感知與控制系統工程師」**。VLA 正在重畫 stack 的邊界，能夠跨層思考（perception ↔ policy ↔ control）的人會是未來 5 年最搶手的。

---

## 開放問題與觀察點

DeepMind 這次發布留下的問題比它解決的還多，值得長期追蹤：

- **訓練資料的規模與組成？** DeepMind 只講 On-Device adaptation 用 <200 examples，但初始 VLA 預訓練用了多少 whole-body demonstration？來源是 teleoperation、simulation、還是 internet video？沒說。
- **推論延遲的實際數字？** Cloud VLA 是 100ms 級還是 500ms 級？On-Device 是 10ms 級還是 50ms 級？沒公開。
- **失敗恢復機制？** 「92% vs 36%」的鎖螺絲反向失敗，機器人怎麼知道自己失敗了？重試策略是硬編碼還是學到的？
- **與 Nvidia GR00T、xAI 世界模型、Xpeng VLA 2 的競爭態勢？** 這幾家的策略都是 open weights 或 semi-open，DeepMind 目前是 closed + trusted-tester，長期看能不能守得住？

---

## 收尾

Gemini Robotics 2 這次發布不是新的模型 SOTA、也不是新的資料集，而是**一個架構信號**：humanoid 業界正在把「locomotion + manipulation 分家」的 legacy 架構丟掉，走向「單一 whole-body policy」的統一控制。

Benchmark 顯示這個賭注還沒完全兌現——地板抓取只有 45.7%、鎖螺絲只有 36%——但方向已經確立。接下來 12 個月要看的是：**On-Device VLA 的部署速度、Safety Envelope 的工程成熟度、以及 Nvidia / Google / OpenAI 三方在這個新架構上的競爭**。

對於在做 LiDAR / 嵌入式 / 感知的工程師來說，這不是「別人家的事」，而是**你手上這個 pipeline 未來要被 VLA 吃進去的預告片**。開始想「我的輸出如何被 VLA 消費」比想「我的 mAP 能不能再提 1%」更有價值。

---

## 參考來源

- [Gemini Robotics 2 brings whole body intelligence to robots - Google DeepMind Blog](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/)
- [Gemini Robotics 2: What Whole-Body Control Changes - Wavect](https://wavect.io/blog/gemini-robotics-2-whole-body-control/)
- [Gemini Robotics 2 Controls Full Humanoids: Legs, Torso, Arms, and Fingers Under One Policy - TechTimes](https://www.techtimes.com/articles/322309/20260730/gemini-robotics-2-controls-full-humanoids-legs-torso-arms-fingers-under-one-policy.htm)
- [Google DeepMind unveils Gemini Robotics 2 as Apptronik humanoid demonstrates whole-body AI - Robotics and Automation News](https://roboticsandautomationnews.com/2026/07/31/google-deepmind-unveils-gemini-robotics-2-as-apptronik-humanoid-demonstrates-whole-body-ai/103802/)

---

*本文為 Nova 之技術觀察筆記，基於公開資料整理，非官方立場。Benchmark 數字與 API 細節請以 DeepMind 官方發布為準。*
