---
title: "從 VLA 到 PLA：NEURA Neuron AI 加上觸覺、聲音、記憶之後，on-device 多模態融合的系統成本在哪裡？（2026-10）"
date: 2026-10-20
tags:
  - vla
  - pla
  - foundation-model
  - physical-ai
  - robotics
  - multimodal-fusion
  - on-device-inference
  - gr00t
  - pi-0
  - openvla
  - neura
  - humanoid
summary: >
  2026-10-14 NEURA Robotics 在 Metzingen 發布 Neuron AI，一個他們命名為
  Perception-Language-Action（PLA）的機器人基礎模型，刻意與 2024-2026
  年統治這塊領域的 Vision-Language-Action（VLA）典範拉開距離。對外的
  story 很乾淨：VLA 只看相機、PLA 加上觸覺／力矩、聲音、本體覺（proprioception）
  與記憶，讓機器人能在「看不到」的情況下完成動作。11 分鐘洗衣流水線的
  demo 預計在 2026-10-20 開幕的 NVIDIA GTC Berlin 現場展示。問題是：到
  2026-10-20 為止，NEURA 沒有公開任何 benchmark，沒有 parameter count，
  沒有 inference latency，沒有論文。只有一個架構圖與五隻 robot embodiment
  的宣稱。這篇不是公關稿改寫，是把 PLA vs VLA 的差距抓到「真實的系統
  成本」上討論：早期融合 5 個 modality 的 encoder 預算是多少？temporal
  memory 若真的要跨 10 秒 horizon 工作、要吃多少 KV cache？on-device
  推論跑在哪一塊 SoC、功耗預算是多少？對照 pi-0.5 的 3.3B flow matching
  action head、GR00T N1.7 的 2B diffusion policy、以及 MolmoAct2 的
  open-weight 公開狀態，Neuron AI 這篇發布到底是「新架構」還是「舊 VLA
  加上多幾個 modality encoder 加上更積極的 marketing」？本文從系統工程師
  的角度拆解。
---

# 從 VLA 到 PLA：NEURA Neuron AI 加上觸覺、聲音、記憶之後，on-device 多模態融合的系統成本在哪裡？（2026-10）

> **TL;DR**
>
> 1. **PLA 不是新典範，是 VLA 的補丁命名**：Neuron AI 把 Vision-Language-Action 中的 V 展開成「3D vision + touch + force-torque + audio + proprioception」，再加上跨時間的 memory，然後把這整包重新命名為 Perception-Language-Action。從模型架構的角度這是 encoder 端的擴充與一個 recurrent/attention-over-time memory，不是全新的 training paradigm。問題不在命名，而在**公開材料到 2026-10-20 都沒有揭露任何可以驗證的技術細節**。
> 2. **真正值錢的是 memory，不是多模態**：VLA 吃「當下這一幀」的設計，在洗衣、開鎖、取出看不見的物件這類任務上必然失敗 — object permanence 不是加上 5 個 modality 就會自動長出來。Neuron AI 若真的解決了「物件離開視野之後持續追蹤」，那是這次發布唯一有結構性意義的部分。但 implementation 的細節（state space model？memory bank？KV-cache 延伸？）完全沒說。
> 3. **五個 modality 的早期融合在 on-device 推論是極大的 bandwidth 成本**：3D vision 要吃點雲、tactile 要吃 force-torque stream（通常 kHz 等級更新）、audio 要吃連續音訊、proprioception 要吃 joint state、language 要吃 token。要把這些時間軸對齊、encode 到同一個 shared latent space，光是 modality-specific encoder 的計算就會把一顆 edge SoC（例如 Jetson Thor 等級）的 compute budget 吃掉一大半，留給 action head 的 budget 相對窄。NEURA 說「on-device, no cloud」— 具體跑在哪塊晶片、功耗多少、控制頻率多少 Hz，全部沒說。
> 4. **對照 pi-0.5 / GR00T N1.7 / MolmoAct2**：這三家在 2026 都公開了 parameter count、training mixture、open weights（至少部分）、評估 benchmark。Neuron AI 完全走閉源 + demo-only 路線。這在 2026 年中段 physical AI 競賽裡是一個清楚的策略選擇 — 不是技術領先才閉源，是**還沒準備好接受對比**才閉源。對研究社群的 signal 很明確。
> 5. **工程師怎麼看這則新聞**：如果你在做 robotics 相關研發，PLA 這個命名可以直接忽略，重點是 (a) 多模態早期融合 encoder 的運算配比怎麼切、(b) temporal memory 用什麼結構實作、(c) action parameterization（NEURA 沒說，pi-0 用 flow matching、GR00T 用 diffusion、OpenVLA 用 discretized tokens）。把這三點釘住，不管 marketing 怎麼寫，技術位置都看得清楚。

## 一、公開材料的事實清單（2026-10-14 ~ 2026-10-20）

這段只放可查證的事實，不加分析：

- **發布時間**：2026-10-14，NEURA Robotics 從德國 Metzingen 總部發布，官方 landing page [neura-robotics.com/neuron-ai](https://neura-robotics.com/neuron-ai/) 同步上線。
- **架構名稱**：Perception-Language-Action（PLA），刻意與 VLA 拉開命名距離。
- **輸入 modality**（五項）：
  1. 3D vision（深度／點雲）
  2. Tactile / force-torque（接觸力與力矩）
  3. Audio（接觸音與語音指令）
  4. Robot state / proprioception（關節位置、速度、末端姿態）
  5. Language（任務 prompt）
- **三層架構**：
  - Perception layer：五項 modality 的 encoder
  - Language model：將 encoded features 投影到 shared language space 做 reasoning
  - Action model：將 reasoning 結果轉成 motor command
- **memory 宣稱**：模型能「記住離開視野的物件」，支援 object permanence。
- **部署**：on-device，無雲端依賴。
- **embodiment coverage**：五隻 NEURA 機器人平台（4NE-1 Gen 3、4NE-1 Gen 4、wheeled 4NE-1、以及兩個 bimanual 平台）。
- **訓練資料**（三支柱）：NEURA Gyms 的真實機器人經驗、simulation 增強、人類穿戴感測服的 demonstration。
- **demo**：11 分鐘洗衣流水線任務，將於 2026-10-20 開幕的 NVIDIA GTC Berlin 現場展示。
- **沒有公開的**：parameter count、inference latency、控制頻率、硬體平台、training compute、benchmark 分數、paper 或 technical report、open weights。截至 2026-10-20，獨立評估完全沒有。

這一節其實已經把後面所有分析的邊界畫出來了：**公開可查的技術資訊幾乎為零，剩下都是從架構圖與行銷詞彙反推。** 這一點本身就是分析的重點。

## 二、VLA 典範在 2026 年中的地位，與它的真正弱點

要判斷 PLA 是不是「新東西」，得先把 VLA 這個典範的現狀講清楚。

### 2.1 VLA 典範的三個代表

到 2026-10，open-weight VLA 市場有三個清楚的參考點：

- **pi-0.5**（Physical Intelligence）：3.3B 參數的 flow matching action head + VLM backbone。從 pi-0 到 pi-0.5 的主要升級是在 continuous action space 的生成品質，以及更廣的 cross-embodiment training。2026-10-13 透過 Pi API beta 開放商用存取。
- **GR00T N1.7**（NVIDIA）：約 2B 參數的 diffusion policy，走 humanoid 專用路線，深度整合 Isaac Sim + Jetson Thor，是 NVIDIA 推 Jetson Thor 的 reference 模型。
- **MolmoAct2**（Allen Institute for AI）：open weights 完整釋出，走 permissive license 路線，是研究社群驗證各種 VLA 架構 claim 的實際對照組。

三家的共通點：都承認 VLA 的核心弱點是「只吃當下的視覺幀」，無法處理需要跨時間記憶或非視覺感知的任務。

### 2.2 VLA 的三個真實弱點（不是行銷話術）

這三個弱點是從 2024-2026 年各家 benchmark 報告反覆浮上來的：

1. **Current-frame bias**：policy head 的 condition 幾乎都壓在最近一兩幀的 vision token 上，object permanence 要靠外接 memory 系統或 heuristic workaround（例如把中間狀態寫到外部 scene graph）。
2. **純視覺的 modality 瓶頸**：抓握軟性物體（毛巾、衣物）、插拔連接器、確認鎖舌是否咬合 — 這些任務純視覺解析度不夠，需要 force-torque 或 contact audio 來補。多數 VLA 把這類資訊丟在 proprioception channel，當作 side input 而非一等公民。
3. **跨 embodiment transfer 的 action space 不一致**：pi-0 系列用 flow matching 處理連續 action、OpenVLA 用 discretized token、GR00T 用 diffusion。這三種 action parameterization 在不同 embodiment 之間 transfer 時各有包袱。

Neuron AI 的 pitch 顯然是在打 (1) 和 (2) — 加入 touch / audio modality 來補 modality 瓶頸、加入 memory 來解 current-frame bias。(3) 他們沒碰，實際 action parameterization 也沒公開。

## 三、把「PLA」拆回去看，剩下的新東西到底是什麼？

這是本文最想釐清的一節。

### 3.1 命名這件事

「Perception-Language-Action」比「Vision-Language-Action」多的只有一個抽象層級：把 vision 泛化成 perception。這在深度學習文獻裡不是新概念 —「multimodal perception」早在 2010 年代就是標準詞。NEURA 把 modality 列表擴充（+ touch, + audio, + proprioception）、把名稱改掉，就產生「與 VLA 不同的新典範」的 marketing framing。

技術上的誠實說法應該是：**Neuron AI 是一個 multimodal VLA，加上一個 explicit temporal memory 模組。** 把它叫 PLA 沒有錯，但把它寫成「與 VLA 不同的新架構」是誇大。

### 3.2 多模態 encoder 的擴充（第一個真實變化）

這是 Neuron AI 相對 OpenVLA / pi-0.5 / GR00T N1.7 的第一個實際技術差異。扣除 marketing 詞彙，具體是：

- **3D vision encoder**：若真的是 point cloud / depth-aware（不是只做深度圖 2D CNN），需要 PointNet++、Point Transformer、或 sparse 3D CNN 這類結構。這與多數 VLA 用 ViT / SigLIP 走 2D vision token 是結構性的不同。
- **tactile encoder**：force-torque sensor 通常以 kHz 等級更新，常見做法是用小型 1D CNN 或 TCN 做時間方向的降維，再 project 到 shared embedding。
- **audio encoder**：接觸音常用 mel-spectrogram + 小型 CNN，或直接用預訓練 audio foundation model（例如 CLAP-like）的 encoder。
- **proprioception encoder**：joint state 是低維 dense vector，通常 MLP 就夠，但要跟其他 modality 做 time alignment。

這五個 encoder 要**早期融合**（在送進 language reasoning 層之前就 fuse），不是晚期融合（各自算完 action 再 ensemble）。NEURA 的 landing page 明確說「one shared space over time」— 這幾個字對應的就是早期融合。

早期融合的技術成本很具體：**modality 數 × modality 更新頻率 × encoder FLOPs** 全部要擠在一個 forward pass 的頭幾個 layer。VLA 單一視覺 encoder 的 budget 原本就不便宜，PLA 要塞五個，若 inference 要維持 10-20 Hz 的 control rate，encoder 部分能分到的 compute 相當有限。

這也是為什麼我會懷疑 Neuron AI 的真實 modality 融合深度 — 很可能是「3D vision 吃主要 budget，其他 modality 做輕量 projection」的妥協設計，而不是真的對五個 modality 都做深層 encoding。沒有公開 architecture 或 ablation，這點無法驗證。

### 3.3 Temporal memory（第二個真實變化，也是最有價值的部分）

NEURA 說模型能「記住離開視野的物件」。這在 2026 中段的 VLA 文獻裡是個真實的空缺 — object permanence 幾乎所有家都走 workaround，沒有結構性解法。

可能的實作方案（NEURA 完全沒說，以下是逆推）：

- **方案 A：State space model（Mamba / RWKV 類）** — 用 recurrent-style 的 selective state space 做長 horizon memory。優點是推論複雜度 O(1) per step、無 KV cache 爆炸問題。缺點是訓練成本高、對 catastrophic forgetting 的控制還不穩。
- **方案 B：Episodic memory bank + retrieval** — 在推論過程中動態寫入 scene feature 到外部 memory，動作時用 attention retrieve 回來。優點是可解釋、可檢查、可以手動 edit。缺點是 memory 管理策略複雜，且 retrieval latency 不低。
- **方案 C：Sliding window KV cache + 強制 attention sinks** — 這是最簡單的做法，用大 window 的 transformer，配合 attention sink / landmark token 保持關鍵歷史。優點是工程成熟。缺點是 KV cache 隨 horizon 線性成長，on-device 推論在長任務（例如 11 分鐘洗衣）會爆記憶體。
- **方案 D：World model + model-based planning** — 在內部維護一個 learned world model，用它推演「我上次把這個東西放到哪了」。這條路最乾淨但訓練成本最高。

從「11 分鐘洗衣 demo」反推：11 分鐘以 20 Hz 控制頻率計算是 13200 個 timestep。方案 C 的 window transformer 要在這個 horizon 工作幾乎不可能（KV cache 會吃光所有 memory），所以 Neuron AI 的 memory 大概率是 **A 或 D 的某種變體**，或兩者混合。

這條線索其實是這則新聞唯一有工程含金量的部分。如果 NEURA 之後公開 technical report，第一個該翻的章節就是這塊。

### 3.4 Action parameterization（NEURA 沒說的那塊）

Action head 怎麼生成動作，NEURA 完全避開不提。這有兩個可能：

- **可能一**：用的是標準方案（flow matching 或 diffusion），沒什麼特別好說。
- **可能二**：用了特殊設計（例如 hierarchical action tokens、或某種 skill library + composition），但還沒成熟到可以公開討論。

從公開材料無法判斷。對一個 robotics 基礎模型來說，action head 的設計其實比 encoder 更關鍵（直接決定控制精度與穩定性），這塊的沉默是最刺眼的訊息。

## 四、On-device 推論：這五個 modality 的系統預算在哪裡？

這是本文最想拉回系統工程師視角的一節。

### 4.1 Edge inference 的真實限制

Humanoid robot 的 on-device compute 預算，到 2026 中段大致是這樣：

- **Jetson Thor（NVIDIA）**：約 2 PFLOPS FP4 / 1 PFLOPS FP8，TDP 大約 100W 等級。這是 NVIDIA 推給 humanoid 的 reference 平台，GR00T N1.7 的 reference deployment 走這條線。
- **自研 SoC**：各家做法不一，Figure、1X、Tesla Optimus、NEURA 自己可能都有部分自研 inference 加速。但 reference peak 通常不會超過 Jetson Thor 等級。
- **Apple M-series on humanoid**：少數玩家在試，但封裝與 I/O 限制讓它不是主流選擇。

假設 Neuron AI 跑在 Jetson Thor 等級的硬體，20 Hz 控制頻率下每個 step 有 50 ms 預算。這 50 ms 要分給：

1. 五個 modality 的 encoder（含 time alignment）
2. Language reasoning 層
3. Temporal memory 的 read / write
4. Action head 的生成
5. 感測器 I/O 與驅動器輸出

對照 pi-0.5 的 3.3B flow matching、GR00T N1.7 的 2B diffusion — 這些 VLA 單一模型的 on-device 推論在 Jetson Thor 等級硬體上已經是 20-40 ms 的 budget（視精度與 batching）。Neuron AI 加上三個額外 modality encoder 與一個 memory 模組，若還能維持相同 control rate，架構上必然做了某些壓縮或降 modality rate（例如 tactile 跟 audio 不是每個 timestep 都做深 encoding）。

這些工程取捨 NEURA 全部沒說。

### 4.2 多模態 bandwidth 的隱藏成本

另一個常被忽略的點：五個 modality 的 raw data bandwidth 不是 trivial。粗估：

- 3D vision：假設 depth camera 走 640×480 @ 30 Hz，約 100 MB/s raw depth stream。
- Tactile：假設 10 個 force-torque sensor @ 1 kHz × 6 自由度 × 4 bytes ≈ 240 KB/s。小。
- Audio：44.1 kHz × 16-bit stereo ≈ 170 KB/s。小。
- Proprioception：假設 30 joint × 1 kHz × 幾個 state ≈ 1 MB/s 等級。
- Language：事件驅動，不計入 steady-state。

總和 raw bandwidth 由 vision 主導，大約 100-150 MB/s 等級。這本身不是問題，但要把這五條流**時間對齊**到同一個 inference step、再送進早期融合 encoder，中間的 buffering、同步、timestamp alignment 本身就是一個 non-trivial 的 real-time system 問題。

這在 ROS2 / Isaac ROS 生態其實有部分成熟 primitive（例如 message filter 的 time synchronizer），但要讓一個 foundation model 在 20 Hz 控制率下穩定吃到五個不同頻率的 modality，實際工程 pipeline 複雜度遠超 VLA。

這也是為什麼 NEURA 強調「on-device」 — 不是雲端跑不動，是這套 pipeline 的 latency 根本不允許走雲端 round-trip。但「on-device」這個 claim 本身在系統上到底怎麼做到的，一個字都沒說。

## 五、對照競爭格局：為什麼 NEURA 走 demo-only 路線？

這一節講策略判讀，不是技術分析。

### 5.1 2026 年中 physical AI 的公開程度分布

- **完全 open**：MolmoAct2（AI2），weights + 訓練程式碼全公開。
- **部分 open**：pi-0 / pi-0.5（Physical Intelligence），有 technical report、API 存取，但 training mixture 細節有保留。
- **Reference open**：GR00T N1.7（NVIDIA），走 Jetson Thor + Isaac Sim 的 reference 路線，模型本身與 NVIDIA 硬體綁定。
- **Closed + demo**：Figure、1X 大致都在這個區間。
- **Closed + demo + 強行銷**：Neuron AI 現在的位置。

### 5.2 為什麼 NEURA 選擇不公開 benchmark

幾個合理解釋，由強到弱：

1. **還沒準備好接受對比**：如果 Neuron AI 在 LIBERO、SIMPLER、RoboCasa 這些標準 benchmark 上輸給 pi-0.5 或 MolmoAct2，公開數字只會傷到自己。這是最常見的解釋。
2. **demo 是為了融資 / 商業客戶，不是研究社群**：11 分鐘洗衣 demo 的目標受眾是會下採購單的工廠或家庭服務客戶，不是會翻 ablation table 的研究員。這個客群看 demo 比看 benchmark 有用。
3. **技術真的領先、但策略保留**：這是官方 narrative 希望大家相信的版本。有可能，但沒有獨立證據。
4. **架構本身就是漸進改進，沒有 benchmark 值得發**：如果 Neuron AI 的進步只是 modality 擴充與 memory 加法，相對 pi-0.5 / GR00T 可能並沒有顯著優勢，發 benchmark 只會凸顯這一點。

(1) + (4) 的組合概率最高。

### 5.3 對研究社群的 signal

這則發布對 academic / open research 社群的實際用途很有限：

- 不能 reproduce。
- 沒有論文可引用。
- 沒有可下載的 weights 做對比實驗。
- 唯一能參考的是「加上 touch + audio + memory 可能是正確方向」這個 directional claim — 但這個 claim 在 2024-2025 的多篇論文裡已經有討論。

這則新聞在 academic 傳播上大概會沉下去，在 industry / 商業圈會持續被引用，形成一個典型的「business press announcement vs research signal」落差。

## 六、工程師的 take：這則新聞該讀什麼、該忽略什麼

對正在做 robotics / 多模態 / foundation model 的工程師，這則新聞的實用訊號大概長這樣：

### 6.1 可以忽略

- **PLA vs VLA 的命名戰**：VLA 這個詞社群會繼續用，PLA 不會變成新 standard terminology。
- **「on-device, no cloud」的 marketing**：pi-0.5 / GR00T / Figure 都已經在做 on-device，這不是差異點。
- **11 分鐘洗衣 demo 的影片本身**：demo 可以精心編排，不是穩定性的證據。看待方式與 Figure 02 咖啡 demo、1X Neo 整理房間 demo 一樣，當 showcase 看、不當 benchmark 看。

### 6.2 值得追蹤

- **temporal memory 的實作細節**：若 NEURA 之後釋出 technical report，第一個該看的章節。這個領域的突破口比加 modality 更有結構性價值。
- **多模態早期融合的 encoder budget 配比**：這是真實的工程難題，不管 NEURA 怎麼做，實作時都會遇到。可以當作自己做多模態系統時的對照。
- **cross-embodiment 推論的 latency 數據**：若之後有獨立 third-party 測試（例如某個 benchmark organizer 拿到 API access），那是真實的技術訊號。
- **action parameterization 的揭露**：這塊的沉默代表這塊的重要性。

### 6.3 對 compiler / systems 工程師的額外觀察

這則新聞對 compiler path 的工程師其實也有訊號，雖然不直接：

- **多模態早期融合的 inference 是 compiler 的新戰場**：MLIR 生態目前對單一 vision transformer 的 fusion 已經相對成熟（iree、xla、torch inductor 都能處理），但對「五個不同 shape / 不同 dtype / 不同 update rate 的 stream 時間對齊」這類 heterogeneous pipeline，compiler 層還沒有乾淨的抽象。這塊是 2026-2028 compiler 社群會慢慢補的空白。
- **temporal memory 的 kernel 需求**：若 Neuron AI 用的是 state space model，Mamba selective scan kernel 在 Jetson Thor 等級硬體的優化到 2026-10 還不成熟（相較於 attention）。這是另一個會慢慢被 compiler 補起來的 kernel 類。
- **Edge inference 的 memory bandwidth**：五個 modality 的早期融合是明確的 memory-bound 問題，不是 compute-bound。對做 memory layout / HBM-aware optimization 的 compiler engineer，這類 workload 的 pattern 是未來幾年的研究熱區。

## 七、收束：這則新聞的真實重量

把前面六節壓縮成一句話：

> **Neuron AI 不是新的機器人架構典範，是 VLA 的 modality 擴充版 + 一個 explicit 的 temporal memory 模組，加上 NEURA 把這套包裝成 PLA 來與 pi-0.5 / GR00T / MolmoAct2 區隔。但因為沒有公開任何可驗證的技術細節或 benchmark，截至 2026-10-20，這則發布對研究社群的 technical signal 很弱、對商業市場的 marketing signal 很強。**

對 2026 的 physical AI 大戰來說，這則新聞主要是**競爭格局變化的信號**，不是**技術突破的信號**：

- NEURA 作為德國歐系 humanoid 玩家，第一次推出一個全棧自研的 foundation model，而不是在 NVIDIA GR00T 或 Hugging Face LeRobot 之上做應用。這在歐系 physical AI 自主性的 narrative 上是個 milestone。
- 但從模型本身的技術論述來看，突破點不足以單獨成立。真正的技術 story 必須等 technical report 或獨立 benchmark。

對 Adam 這類在 compiler / systems / robotics 交界工作的人，這則新聞的實用價值是：**memory 的實作選型** 與 **多模態早期融合的 edge inference pipeline 設計** 這兩個子問題，比命名戰或 demo 影片更值得花時間。這兩個問題不管 Neuron AI 做得好不好，未來三年你的工作裡都會遇到。

PLA 這個詞大概會跟 2024 年某些家喊過的「VLM-R」、「Robot-GPT」一樣，在 2027 中就被社群默默忘掉，但背後的工程問題不會。先把工程問題釘住，命名的事情留給 marketing 部門去打。

---

**參考材料**（截至 2026-10-20）：

- NEURA Robotics 官方 Neuron AI landing：[neura-robotics.com/neuron-ai](https://neura-robotics.com/neuron-ai/)
- Interesting Engineering 技術報導：[New robot brain lets humanoids tackle multi-step tasks autonomously](https://interestingengineering.com/ai-robotics/neuron-ai-combines-vision-touch-sound)
- Implicator.ai 分析：[NEURA Robotics Launches Neuron AI Robot Model With Touch](https://www.implicator.ai/neura-robotics-neuron-ai-touch-sound-memory/)
- Robotics and Automation News：[Neura Robotics launches multimodal AI model that gives robots perception and memory](https://roboticsandautomationnews.com/2026/10/14/neura-robotics-launches-multimodal-ai-model-that-gives-robots-perception-and-memory/105902/)
- Unite.ai 公告整理：[NEURA Robotics Unveils Neuron AI Foundation Model for Physical AI](https://www.unite.ai/neura-robotics-unveils-neuron-ai-foundation-model-for-physical-ai/)
