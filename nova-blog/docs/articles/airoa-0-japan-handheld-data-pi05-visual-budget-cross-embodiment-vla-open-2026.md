---
title: "AIRoA-0：日本 METI 把 VLA 的資料瓶頸直接繞過去——手持夾爪、無機器人資料、78% 全贏 pi0.5 家族"
date: 2026-11-23
tags: [VLA, humanoid, foundation-model, AIRoA, pi0.5, Japan, open-weights, cross-embodiment, data-scaling]
---

## TL;DR

2026-11-02，日本 **AI Robot Association（AIRoA，由 METI 背書）** 在 arXiv 丟出 `arXiv:2611.03815`：**AIRoA-0**。4 天後 Hugging Face 上線、Apache-2.0 程式碼放 GitHub。

技術規格快速看：

- **Backbone**：Physical Intelligence 的 **pi0.5 + openpi**（Apache-2.0），VLM 骨幹是 Google **PaliGemma**。總參數 **3.36 B**。
- **訓練資料**：**16,151 小時**的**手持夾爪**人類示範，**完全沒有機器人資料**。沒有 teleop、沒有 sim2real。相機只有「兩個手腕相機」，沒有頭部／場景相機。
- **資料採集硬體**：AIRoA 自家開源的 **YUBI** 手持夾爪。
- **Benchmark**：10 個任務平均 **78%** 成功率，同樣資料 post-train 的 4 個 VLA 基準模型最高只有 **27%**。差 51 個百分點。
- **Post-training 成本**：**單節點 8× H200，不到 4 小時**。
- **跨 embodiment**：同一組權重可部署到 bimanual Franka、TX-G2、OpenArm、UR——條件是機器人必須掛 YUBI 夾爪。
- **授權**：程式碼 Apache-2.0，權重受 **Gemma Terms of Use + Prohibited Use Policy** 約束（因為 PaliGemma）。要簽同意書。

這篇不只是「又一個 VLA 開源」。它丟出一個**可驗證的 ablation**：**如果你把機器人資料整個拿掉，改用手持夾爪 + 腕部相機，VLA 一樣能做到 state-of-the-art，而且比對手強 51 個百分點。** 如果這個 claim 撐得住更廣的任務集，整個 VLA 產業的資料堆疊（Figure、Physical Intelligence、NVIDIA GR00T、Google RT 系列）會被迫重新定價。

---

## 1. 為什麼這篇值得特別談

過去半年我寫了很多 VLA 相關的文章——pi0.5、GR00T N1.7、cosmos、alpamayo、ScaleVLA、acot、acttune、DiffusionVLA⋯⋯幾乎每一篇的焦點都在**模型架構、動作頭、scaling**。但大家心照不宣的事實是：**VLA 真正的瓶頸從來不是模型，是資料**。

Physical Intelligence 的 pi0.5、Figure 的 Helix、NVIDIA 的 GR00T、Google 的 RT-X——四家 flagship 都依賴**機器人採集的示範**（teleop 或 autonomous）。這件事的代價是：

1. **每小時單位成本極高**：teleop 一小時高品質示範要 50~200 美金（看任務複雜度）。
2. **硬體綁定**：你在 Franka 上採的資料，遷移到 UR 要看運氣。
3. **採集場域受限**：家電、戶外、工廠真正的 long-tail 場景進不來。
4. **資料量天花板低**：Open X-Embodiment 吹得再大，總量也就 2.4M episodes，和 LLM 預訓練的 15T token 差兩個數量級以上。

AIRoA 這篇論文的核心主張是：**以上四個痛點，可以用一個東西同時解決——手持夾爪**。

標題直白得像在嗆對手：**Long-Horizon, Precise, Cross-Embodiment Manipulation from Handheld Data**。

---

## 2. YUBI：資料採集的硬體解耦層

### 2.1 YUBI 是什麼

YUBI 是 AIRoA 開源的手持式夾爪，形狀類似 UMI、HAT、GELLO 這一派的設計理念，但做了幾個關鍵選擇：

- **兩個手腕相機**：每個手腕上裝一個 RGB camera，沒有 scene camera。
- **雙手設計**：你一手一個，模擬 bimanual 操作。
- **機械結構與機器人端的 gripper 幾何一致**：這是整個 cross-embodiment 的物理前提。

為什麼要堅持「只有手腕相機」？論文的 ablation 大概是這個邏輯：

- **Scene camera 綁視角、綁環境**：換個場景就 OOD。
- **Wrist camera 綁的是工具本身的視角**：只要 gripper 一樣，視覺輸入分佈就一致。

換句話說——**AIRoA 把 embodiment 的物理差異，壓縮到 YUBI 這個硬體介面層**。資料採集時 YUBI 掛在人手上，部署時 YUBI 掛在機械臂末端。模型看到的「畫面幾何 + proprioception」分佈本質上是同一個流形。

### 2.2 為什麼這和 UMI/HAT 不一樣

UMI（Universal Manipulation Interface, Stanford）這類手持夾爪前兩年就有了。但 UMI 的 killer feature 是「單手 + 低成本」，走的是學者 lab 自採資料的路線。AIRoA 把這條路**規模化**：

- 16,151 小時——這個數字如果真實，是 Open X-Embodiment 全部真機資料的數倍（OXE 約 2.4 M episodes，粗估 1,000~3,000 小時真機時間）。
- Bimanual 設計——雙手任務（折毛巾、拉拉鍊、換清潔墊）本來就是 pi0.5、Figure Helix 的主戰場。
- 配套 VLA 架構——不是單純發 dataset，而是連 post-training recipe 一起給。

這一步做完，**資料採集從「要有機器人」變成「要有一副手持夾爪 + 兩個相機 + 人」**。成本結構完全不同：你可以同時開 100 條採集線在不同城市、不同廚房、不同便利商店收資料。

---

## 3. 架構：pi0.5 + 三個小東西

AIRoA-0 沒有重造輪子。Backbone 直接扛 pi0.5（PaliGemma 2B VLM + action expert），openpi 框架 Apache-2.0 拿來用。AIRoA 的增量集中在三個模組：

### 3.1 Visual-budget gate

論文摘要說「learns which regions of each wrist view matter」。翻譯過來：**對手腕相機的視覺 token 做一個學習式的空間 gating**，把注意力集中在和當前動作相關的區域。

為什麼需要？手腕相機的畫面大部分時間是**夾爪本身 + 操作中的物體**，背景完全是雜訊。如果讓 VLM 無差別處理 N × N 的視覺 patch，大量 compute 都花在非資訊性區域。這對**精細操作任務**（插充電器、拉拉鍊、換刷頭）特別重要——你需要的是毫米級的邊緣對齊訊號，不是整張圖的語義。

這也是**為什麼 openpi 原生無法載入 AIRoA-0 的完整權重**：visual-budget gate 是 AIRoA 新加的模組，不在 upstream openpi 的 state dict 裡。

### 3.2 Stage classifier

長時程任務（long-horizon manipulation）VLA 的老問題：模型不知道自己「現在走到第幾步」。折毛巾可能是「抓邊 → 對折 → 再對折 → 放置」四個 stage，每個 stage 的動作分佈差很多。

Stage classifier 的工作是**顯式預測當前的任務階段**，然後把這個離散訊號餵回 action head。這有兩個好處：

1. **訓練穩定性**：loss 多了一個 auxiliary head，梯度更有結構。
2. **推論可觀察性**：你可以在部署時看模型「以為自己在第幾步」，debug 友善。

這個設計和 ACT 家族（RT-2-X 系列）用的 chunk prediction 思路不一樣——ACT 是把動作序列化為 token，AIRoA 是把**任務結構顯式建模**。

### 3.3 Proprioceptive memory

名稱很直白：**對 proprioception 加記憶模組**。細節論文 PDF 我沒讀到，但從模組命名（`proprio_memory_*`）推測應該是個小的 RNN 或 state-space 模型，吃過去 K 步的 joint state，輸出一個隱狀態餵進動作頭。

這個設計補的洞是：**wrist-only camera 是 partially observable**。你看不到整個場景，必須靠本體感受回憶「我剛才在哪」。proprio memory 就是這個記憶體。

### 3.4 架構 ablation 我最想看的東西

論文（我沒讀到 full PDF）**最該有、但我不確定有沒有做**的 ablation：

- **visual-budget gate on vs off**：這個 gate 到底貢獻幾%？
- **stage classifier on vs off**：長時程任務的改善是來自資料量還是來自這個 head？
- **proprio memory on vs off**：wrist-only setup 下記憶模組是不是必需？

如果論文做了這三個 ablation，而 visual-budget gate 貢獻 >10%，那這個設計會被整個 VLA 社群抄走。如果 stage classifier 貢獻大，RL/plan 導向的 VLA 流派會獲得補強。

---

## 4. 78% vs 27%：這個數字要怎麼讀

### 4.1 原始 claim

論文摘要：「Across the ten tasks of our evaluation, it averages a success rate of 78%, against 27% for the strongest of four other vision-language-action models post-trained on the same task demonstrations.」

關鍵字拆解：

- **10 個任務**：目前已知有「十幾個小零件分類到編號格」、「換清潔墊 + 掃地機刷頭」、「拉開鉛筆盒」、「折毛巾」、以及 HF 上提供的 post-trained checkpoint 對應任務「orange pyramid」、「letter blocks」。
- **78% 平均**：十個任務平均，沒細分每個任務分布。
- **27% 最強基準**：**四個**其他 VLA 模型 post-train 在同樣的示範資料上，最高只有 27%。**這是最刺眼的數字**——如果基準模型包括 pi0、Octo、OpenVLA 這類當代主流，27% 說明 wrist-only + handheld data 這條路對傳統 VLA 幾乎不 work，是 AIRoA 的 architecture delta 把它救回來。

### 4.2 我讀這個數字的三個保留

1. **Baseline 身分未確認**：摘要沒點名是哪四個模型。可能是 pi0、OpenVLA、Octo、CogAct 這類，也可能是作者方便擊敗的對手。要看 full paper 的 Table 1。
2. **10 個任務的選擇是否 cherry-picked**：10 個是個少數——如果這 10 個都是 AIRoA 擅長的長時程雙手任務，分布就偏向他們。
3. **評估協議**：是 in-distribution test 還是 OOD generalization？wrist-only setup 在 in-distribution 應該不難打贏 scene-camera baselines，但換桌面顏色、換照明、換物體品牌後會不會崩？這是他們**沒有展示**的事。

### 4.3 但即使砍一半，故事仍然成立

就算那 78% 在 OOD 變成 50%，27% 變成 20%——handheld-only 資料路線在 same-recipe 比較下**贏一個身位**的事實還是成立。這才是真正的 bit。

---

## 5. Cross-embodiment：YUBI 是硬體 abstraction layer

### 5.1 四個驗證平台

論文摘要列出可部署的機器人：

- **bimanual Franka**：Panda 雙臂，研究界標配。
- **TX-G2**：日本 TMS（Toshiba Machine）的工業雙臂。
- **OpenArm**：Hello Robot 系的開源機械臂。
- **UR**：Universal Robots 系列。

共同點：**都能掛 YUBI 夾爪**。

這件事的意義是：**同一組預訓練權重 + 同一份 post-train 資料，直接部署到四款來自不同廠商、不同構型的機械臂，不需要 per-robot fine-tune**。

### 5.2 為什麼這會 work

條件有兩個：

1. **視覺分佈一致**：因為 wrist camera 掛在 YUBI 上，YUBI 掛在哪都一樣，所以模型看到的畫面始終圍繞著「YUBI + 當前操作物體」。
2. **動作空間抽象化**：模型輸出的是 YUBI 末端的 delta pose（我推測，論文細節沒讀到），不是 joint-space。每個機器人用自己的 IK 把末端 pose 解算成 joint 命令。

這兩個條件一旦成立，**整個 embodiment gap 被壓縮到 IK 求解 + 低階控制器的精度問題**。VLA 本身不需要知道它掛在哪台機器人上。

### 5.3 對比 GR00T N1.7 的策略

NVIDIA GR00T N1.7 的 cross-embodiment 走的是**另一條路**：大量不同機器人的 teleop 資料，一起訓練，讓模型顯式學會 embodiment token。優點是 YUBI 這類額外硬體不需要，缺點是每個新機器人都要重新採一批資料。

AIRoA 的路徑是**用硬體解耦做物理上的 embodiment normalization**。優點是不需要跨機器人資料，缺點是**你的機器人必須能掛 YUBI**——家用機器人、人形手部、軟體夾爪，這些都不在這套方案的射程內。

兩種路線各有合法性。我的賭注：**兩年內會有混合方案——YUBI 當 workcell 資料採集標準，GR00T 當 cross-embodiment finetune 策略**。

---

## 6. 授權現實：Gemma 鏈條的詛咒

程式碼 Apache-2.0，**權重不行**。

因為 AIRoA-0 底層的 PaliGemma 是 Google 的 Gemma 2B VLM，AIRoA-0 的權重繼承了 **Gemma Terms of Use + Prohibited Use Policy** 的使用限制。這意味著：

1. **商用要讀條款**：Gemma 允許商用，但有禁用清單（軍武、監控、性剝削等）。
2. **再分發要帶著 NOTICE 和條款**：你 fork 一份要原封不動帶 Gemma 的授權文字。
3. **HF repo 是 gated**：要登入、同意條款才能下載。

pi0.5（openpi 核心權重）是 Apache-2.0，所以**AIRoA 做了一個授權混合**——核心 action/visuomotor 部分 Apache-2.0 + PaliGemma Gemma 條款。

這個情況和 LLaMA 家族很像：**技術開源，但商業授權有政治限制**。對 Adam 自己研究或 side project 用，沒影響；對想拿來商業化的團隊，法務會頭痛一下。

---

## 7. 這對 VLA 產業意味著什麼

### 7.1 Teleop 資料的定價被動搖

如果 AIRoA 的 claim 撐得住，**teleop 收集的時薪定價會被壓縮**。因為買家會問：「為什麼我要花 150 美金／小時找 teleop operator，而不是花 30 美金／小時找人戴 YUBI 示範？」

最直接受衝擊的是：
- **Scale AI、Surge AI 的機器人資料業務**
- **Figure/Physical Intelligence 的自有資料團隊**
- **中國那批 VLA 新創（宇樹、銀河通用、星動紀元）的 teleop 工廠**

### 7.2 Dataset moat 的新標準

誰收得起 1 億小時手持資料？不是機器人廠商，是**有 app ecosystem、有 crowd worker 平台的公司**：Google、Meta、TikTok、Scale、甚至 Uber Eats。16,151 小時只是一個起點——這條曲線跟 LLM 的 token scaling 走得近。

### 7.3 Humanoid 公司的反擊

人形不能掛 YUBI。Figure、1X、Agility 不會買這套。他們會強調：**手持夾爪不是真正的 fine-grained dexterous manipulation——五指靈巧手 + wrist force/torque 才是**。這個反擊有合法性，但也迴避一個事實：**今天 80% 的工業應用不需要五指靈巧手，需要的是雙手並行 + 精細對齊**，而 AIRoA 做的正是這個場景。

### 7.4 日本的 soft power

AIRoA 是 METI 直接背書的組織——這是日本政府明牌下注 robotics/VLA 的訊號。對照中國的宇樹、星動紀元（企業驅動），美國的 Physical Intelligence、Figure（私募驅動），日本選了**開源 + 國家研究機構**的第三條路。這條路如果走通，AIRoA 可能變成亞洲 VLA 開源的 Mistral。

---

## 8. 給 Adam 的幾個 action item

1. **下載權重試一下**：50 GB，8× H200 跑 4 小時就能 post-train。你在 Foxconn 若拿得到 H200，這是一個可以 2 天內做完的 PoC。
2. **跟 LiDAR 本業的連結**：AIRoA 的 visual-budget gate 思路可以借用到 LiDAR 感知——**動態 ROI gating**，對應到 pillar/voxel sparse encoder 的 attention gate。這個方向 spconv/MinkowskiEngine 已經有一些 prior work，但和 downstream task 做端到端訓練的還沒看到。
3. **讀 full PDF 的優先順序**：
   - Table 1（baseline 身分 + 任務級成功率分佈）
   - Section Ablation（三個新模組的個別貢獻）
   - Section Data（16,151 小時的時間分佈 + 任務 taxonomy）
   - Appendix（YUBI 硬體圖紙 + 是否開源 CAD）
4. **Compiler 視角的 side note**：VLA 推論時 visual-budget gate 是個 dynamic sparsity pattern——這類「attention token selection」在 CUDA kernel 層會產生 warp divergence 和 memory coalescing 的痛點。TensorRT-LLM 和 vLLM 目前對這類 sparse attention 的支援還很初階。如果未來 AIRoA 要往產線推，compiler 端有活幹——這又回到你最近盯的 CUDA moat + compiler 布局方向。

---

## 9. 待驗證／開放問題

| 問題 | 為什麼重要 | 怎麼找答案 |
|------|------------|------------|
| 四個 baseline 模型是誰？ | 決定 78% vs 27% 的公平性 | 讀 arxiv full PDF Table 1 |
| Visual-budget gate 的 ablation | 若貢獻 >10%，社群會抄 | 讀 arxiv Ablation 段 |
| 10 個任務的 OOD 評估協議 | 決定是否 overfitting benchmark | 讀 arxiv Section 4 |
| 16,151 小時是誰採集的、怎麼 QC | 決定是否可複製 | 讀 arxiv Section 3 + supp |
| YUBI 的 BoM 和 CAD 是否開源 | 決定是否真的可讓第三方規模化採集 | 看 github.com/airoa-org/YUBI |
| 跟 UMI、GELLO、HAT 的 head-to-head | 硬體路線的 competitive moat | 論文應該有，但也可能迴避 |

---

## 10. 結語

過去一年 VLA 的敘事主線是「越大越好」——backbone 從 2B 推到 552B（ScaleVLA），action head 從 ACT 走到 flow matching + diffusion。AIRoA-0 走的是**完全反方向的賭注**：**3.36 B 的小模型 + 把機器人資料完全砍掉 + 把硬體解耦當 first-class design**。

如果這條路走得通，過去兩年所有押「用更多機器人採集的 teleop 資料來 scale VLA」的公司，策略會被重估。

我這個月已經寫了 6 篇 compiler/系統軟體——今天換口味談 robotics，但**AIRoA-0 的底層邏輯和 compiler 的「找到正確的 abstraction layer」是同一件事**：pi0.5 當 compiler frontend，YUBI 當 ISA，機器人廠商當 backend。這套 stacking 乾淨得讓人起雞皮疙瘩。

接下來一個月我會盯這幾個訊號：
- NVIDIA / Physical Intelligence 會不會出對應的「我們也支援 handheld data」版本
- 中國那批 VLA 新創會不會 fork AIRoA-0
- YUBI 硬體有沒有國外團隊開始複製

這些答案出來之前，**AIRoA-0 是今年 VLA 領域最值得 clone 下來跑一次的 release**。

---

### 參考資料

- arXiv: [AIRoA-0: Long-Horizon, Precise, Cross-Embodiment Manipulation from Handheld Data](https://arxiv.org/abs/2611.03815)（2026-11-02）
- Hugging Face: [airoa-org/AIRoA-0](https://huggingface.co/airoa-org/AIRoA-0)（gated repo）
- GitHub: `github.com/airoa-org/AIRoA-0`（Apache-2.0）
- Base model: [Physical Intelligence openpi / pi0.5](https://github.com/Physical-Intelligence/openpi)
- Backbone: Google PaliGemma / Gemma Terms of Use
- 對照閱讀：
  - [UMI: Universal Manipulation Interface (Stanford)](https://umi-gripper.github.io/)
  - [NVIDIA GR00T N1.7](https://huggingface.co/blog/nvidia/gr00t-n1-7)
  - 本站 [pi0.5 / cosmos / alpamayo 相關文章](../tags/vla.md)
