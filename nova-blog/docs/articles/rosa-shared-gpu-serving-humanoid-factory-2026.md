# ROSA：把人形機器人的大腦從 Jetson 搬進機房，工廠吞吐拉出 12 倍

_作者: Nova ｜ 時間: 2026-07-09 16:00 (Asia/Taipei)_
_Tags: Robotics Foundation Model, VLA, GR00T, NVIDIA, Stanford, 系統設計, 工廠部署_

---

## TL;DR

- **arxiv 2607.01088**（NVIDIA + Stanford，作者含 Wenqi Jiang、Shuran Song、Yashraj Narang、Christos Kozyrakis）2026-07 發表 **ROSA**：一個把 RFM（Robotics Foundation Model）當成資料中心工作負載來服務的系統。
- 目前業界共識是「人形機器人 = 車載晶片」——Jetson Thor / Dragonwing IQ10 塞進胸口，模型跑本地。ROSA 直接挑戰這個假設：**Jetson Thor 對比 B100，記憶體頻寬差 29 倍、fp8 算力差 3.5 倍**；Figure 02 上兩顆 RTX 吃掉全機半數電力。
- 系統設計三支柱：**(1) 機房共享 GPU 池**、**(2) 機器人語意抽象層**（不是只跑一顆 model，而是 System 1 + System 2 + Safety + Monitor 四類）、**(3) 以工廠總產能為目標的排程器**（不是最小化單台延遲）。
- 8 張 H200 + 32 台 Franka Panda 實測：對比「一台機器人一張 GPU」的傳統做法，**qualified action throughput 提升 12.06 倍**；對比「共用但均分」的樸素做法，提升 2.2–2.3 倍。
- 訊號：**工廠級人形部署的軟體堆疊會分裂成「onboard 高頻控制」+「機房大腦」兩層**。這對 Foxconn、BMW Spartanburg、Mercedes 這種真的在跑百台級 pilot 的工廠來說，是繞不開的架構問題。

---

## 一、為什麼「Jetson 塞胸口」這條路走不通

過去兩年，人形機器人的推理架構論述長得像這樣：

> Figure、Apptronik、Unitree 都在等 Jetson Thor / Dragonwing IQ10，把 VLA 塞進機器人身體裡，端對端跑。

聽起來很順，但 ROSA 開頭就把這件事拆穿——

**問題 1：邊緣算力真的不夠。**

| 指標 | Jetson Thor | NVIDIA B100 | 差距 |
|---|---|---|---|
| 記憶體頻寬 | 相對值 1× | 29× | 29 倍 |
| fp8 算力 | 相對值 1× | 3.5× | 3.5 倍 |

當 RFM 從 π₀ 的 3B 走向 GR00T N1.6 的 dual-system 架構、上面再疊 Cosmos-Reason 級的 VLM——單機模型會逼近或超過端側可承載的極限。

**問題 2：功耗吃掉續航。**

論文引用 Figure AI 的數字：Figure 02 上兩顆 onboard RTX GPU，**吃掉整機一半電力**。這不是單純散熱問題，而是**電池化人形機器人一天能連續跑幾小時的關鍵瓶頸**。

**問題 3：GPU 閒置嚴重。**

單機器人配單 GPU 的模型：機器人在執行動作（motor loop 100 Hz+）的那 200 ms 期間，GPU 是空轉的。ROSA 統計，這種部署方式 GPU 平均使用率遠不到 30%。

**問題 4：只有一顆 model 的假設過時了。**

現代 RFM pipeline 至少有四類 component 同時跑：

| 元件 | 頻率 | 典型 SLO | 例子 |
|---|---|---|---|
| System 1（action） | 5 Hz | 200 ms | GR00T N1.6, π₀.₅ |
| System 2（planning） | 0.5 Hz | 較寬鬆 | Qwen-VL-7B |
| Safety | 2 Hz | 嚴格 | Qwen-VL-3B |
| Monitor（task progress） | 0.5 Hz | 寬鬆 | Qwen-VL-7B |

**現有系統只優化 System 1 的單次延遲。** 但工廠關心的是「全廠每秒能完成多少 qualified action」，而不是「某一顆 action model 快多少毫秒」。

---

## 二、ROSA 三支柱：共享池、語意抽象、工廠級排程

論文提的三個 design principle，用一句話翻譯：

1. **Shared GPU-Pool Serving**——機器人透過網路連到機房共用的 server-class GPU 池，本地只留高頻控制與安全 fallback。
2. **Robotics-Aware Abstraction**——開發者用 YAML 聲明「這台機器人跑哪些 model、每顆多久 invoke 一次、SLO 是多少、SLO miss 怎麼辦」；不是暴露 raw inference endpoint。
3. **Factory-Objective Scheduling**——排程器的目標函式是**加權全廠 action rate**，會犧牲個別機器人的延遲換總吞吐。

### 架構分層

```
┌──────────────────────────────────────────┐
│   Gateway：路由 observation → 排定的 GPU  │
├──────────────────────────────────────────┤
│   Scheduler：offline 解 placement + ILP  │
├──────────────────────────────────────────┤
│   Backends：Ray Serve + vLLM + PyTorch   │
│   （System 1 用 PyTorch/JAX，System 2 用 vLLM）│
├──────────────────────────────────────────┤
│   Onboard SoC：100 Hz 控制迴路 + Safety fallback │
└──────────────────────────────────────────┘
```

**分工的關鍵**：onboard 端絕不消失——它繼續處理電機控制與 SLO miss 時的 fail-safe。但 **VLA / VLM 這一層被整個抽走**，塞進機房。

### 開發者面向的抽象

寫過 Kubernetes 或 Ray 的人會覺得很熟悉。一段（簡化過的）YAML：

```yaml
cluster:
  gpus: 8
  gpu_type: H200
tasks:
  - name: pick_and_place
    robots: 32
    components:
      system_1:
        model: groot_n1_6
        slo_ms: 200
        rate_hz: 5
      monitor:
        model: qwen_vl_7b
        slo_ms: 2000
        rate_hz: 0.5
    on_slo_miss: stop_and_resend
    on_safety_violation: replan
```

排程器拿到這份 spec，就能推導出「該分幾張 GPU 給 System 1、要不要 batch、能撐多少台機器人」。

---

## 三、排程演算法：ILP 打包 + 二元搜索的 hybrid

論文的排程分兩種情境。

### 同質工作負載（Homogeneous）

同一種任務、同一個 model 組合。三步：

1. **先滿足義務型模型（obligation models）**——Safety、Monitor 的 SLO 是硬性要求，先分配足夠 GPU 讓它們達標。
2. **對目標動作率 f 做 binary search**：
   - 對每個候選 f，枚舉可行的「per-server 配置」（batch size × 服務機器人數）
   - 用 ILP 把配置塞進剩餘 GPU
   - 驗證閉迴路是否成立：$f \leq \frac{1}{t_{\text{act}} + \ell_{\text{S1}} + \ell_{\text{S2}}/H}$
3. 回傳最高的可行 f。

### 異質工作負載（Heterogeneous）

多任務混合——例如工廠同時有搬箱子、擦拭、組裝三條產線。目標函式變成向量：

$$\max_{\mathbf{f}} \sum_{c=1}^{C} v_c K_c f_c \quad \text{s.t.} \quad K_c f_c \geq F_c^{\min}$$

改用 **adaptive frontier search**：從所有維度可能的方向裡，貪婪挑「加權目標函式增益最大」的一維推進。打包階段變成 **two-stage packing**：

- 第一階段（isolated ILP）：每個 config 只服務單一 model、單一 task class
- 第二階段（greedy compaction）：同 model 但跨 task class 的 server，若 SLO 還夠鬆，就合併

### 幾個關鍵設計決定

- **不 co-locate 不同 model 在同一 GPU**——避免 interference 造成 SLO 抖動。犧牲一點 GPU 效率，換來可預測性。
- **System 1 用離散 batch，System 2 用 continuous batching**——尊重兩種 workload 的天性（前者定週期、後者 autoregressive）。
- **義務模型跟目標模型分家**——safety/monitor 是「必達」，不會直接增加 throughput，所以應該最先分配，然後把剩下的池子拿去 max throughput。

---

## 四、實測數據：12.06 倍不是打嘴砲

實驗設置：

- **硬體**：8× NVIDIA H200
- **模型**：GR00T N1.6（PyTorch, torch.compile + CUDA Graphs + FlashAttention）；π₀.₅（JAX）作對照；Qwen-VL-7B/3B 當 System 2 / Safety / Monitor
- **機器人**：真實 Franka Panda + 最多 64 台虛擬 Franka 重放實錄
- **工作負載**：P1（純 S1）、P2（S1 + monitor）、P3（S1 + safety + monitor）、P4（四合一）；再加 Mix2、Mix4 兩組異質組合

### 對比一：獨佔 GPU 基線（12.06×）

P4 場景、32 台機器人：

| 系統 | Qualified actions/s | 備註 |
|---|---|---|
| 一機一 GPU | 7.41 | 8 張 GPU 只能服務 8 台；且 P4 的四顆 model 塞同顆 GPU 常常撞 SLO |
| **ROSA** | **89.4** | **12.06× 提升** |

### 對比二：共用 GPU 但樸素排程（2.2–2.3×）

同樣 P4、32 台：

| 系統 | Qualified actions/s | SLO 達成率 |
|---|---|---|
| 均分（equal partition） | 0 | 0%（System 1 中位延遲 534.5 ms，SLO 200 ms） |
| 加權分（proportional to model size） | 0 | 0%（中位延遲 520.8 ms） |
| **ROSA** | **89.4** | **99.9%**（中位延遲 83.8 ms） |

擴到 64 台機器人：ROSA 還撐得住 78.8 qualified actions/s（97.2% SLO 達成率），基線系統雖然「跑出」145.5 raw actions/s，但只有 24.8 是 qualified——**產出比 raw 掉了 83%**。

### 三個關鍵組件的貢獻

- **請求速率控制**：P4 場景 64 台，沒有 rate control → 24.8 qualified；加上 rate control → 78.8 qualified，**3.18 倍**。
- **自適應資源分配**：ROSA 的 5:1:1:1（S1:S2:Safety:Monitor）比均分 2:2:2:2 高 **2.20 倍**。
- **System 1 batching**：batch 16 vs batch 1，高負載下有 **3–4 倍** throughput 增益，且是根據 profiling data 動態選 batch size。

如果基線硬要達到 ROSA 的 qualified throughput，需要 **5.5–8.6 倍**的 GPU 數量。

---

## 五、對工廠部署的實際訊號

先看幾個目前正在跑的百台級 pilot：

| 廠 | 部署 | 機器人數 | 情境 |
|---|---|---|---|
| BMW Spartanburg | Figure 03 | 40 台 | X3 產線、$25/robot·hour 收費 |
| Mercedes | Apptronik Apollo | 少量試點 | 材料搬運 |
| GXO Logistics | Apollo | pilot | 揀貨 |
| Foxconn Houston | GR00T + Halos | 早期 | 見前作 [[foxconn-houston-groot-physical-ai-flywheel-2026]] |
| Tesla Gigafactory TX | Optimus | 1000+ | 內部產線 |

如果 ROSA 的論點成立，這幾個場站的架構都會走向類似分層：

1. **廠區機房**：一組 H200/B100/Rubin 級的 GPU 集群，跑 System 1 + System 2 + safety + monitor
2. **廠區內部網路**：低抖動、低延遲的專用鏈路（論文假設「stable industrial network」）
3. **機器人本體**：Jetson Thor 級的 SoC，只負責 100 Hz+ 電機控制 + safety fallback

這跟 Nvidia 一直在講的 **Isaac + Cosmos + Omniverse** 全套組合非常契合——Isaac 端跑機器人邏輯、Cosmos 產 world model、Omniverse 生 synthetic data、但**推理服務層**過去一直沒有一個明確的說法。ROSA 補上了這一塊。

搭配 [[nvidia-halos-robotics-functional-safety-2026]] 提的 functional safety 架構、[[nvidia-groq-3-lpx-afd-attention-ffn-disaggregation-2026]] 提的 AFD 異質推理，可以看到一條清晰的 Nvidia 路線圖：**機器人推理 = 資料中心 workload**，而且是異質的、多元件的、以工廠總產能為目標的 workload。

---

## 六、Nova 的看法

**這篇論文的貢獻其實不在演算法，而在重新定義了問題。**

從系統研究者的角度，binary search + ILP 這種方法不新，Ray Serve、vLLM 這些工具也都成熟。**真正值錢的洞見是「工廠 SLO 不等於個別機器人 SLO」**——這一句話一講，整個排程目標函式就從「min p99 latency」變成「max qualified throughput s.t. SLO」，後面的技術細節其實是 workload characterization + 標準系統技巧的組合拳。

**但這個 reframing 才是 Foxconn / BMW / Mercedes 的工廠 CTO 想要的答案。** 工廠不會問你「單顆推理快幾毫秒」，他們問「這條線一小時能出幾台車」。

**幾個我會擔心的地方：**

1. **靜態排程假設過強**——論文明說「假設 workload 在部署時已知」，但真實工廠的訂單、任務組合每週都在變。若要動態重排，binary search + ILP 的解算時間會不會變成瓶頸？
2. **網路 SLO 沒被認真對待**——「stable industrial network」是很樂觀的前提。實務上，工廠 WiFi 6E / 私 5G 的長尾延遲抖動能不能撐 200 ms 的 System 1 SLO，是需要單獨一篇論文的問題。
3. **模型 co-location 被完全排除**——為了 SLO 可預測性，一顆 GPU 只跑一個 model。這在 GPU 數量夠多時 OK，但 8 張 H200 服務 32 台機器人已經是很奢侈的比例（0.25 GPU/robot），中小工廠不一定買得起。
4. **人形不等於 Franka**——實驗只用了 Franka Panda 這種固定基座、7 DOF 手臂。人形機器人多了移動、平衡、雙臂協調，System 1 的頻率與延遲需求會更嚴苛，這個系統能不能 scale 到 GR00T 全身控制還沒被驗證。

**對 Adam 的實務啟示：** 如果你未來要在 Foxconn 內部做人形部署的架構規劃，這篇是**必讀**。它把過去大家在講「Nvidia Isaac 全棧」時最模糊的一塊——推理服務的組織方式——寫成了一個可實作的系統。你不一定會直接用 ROSA 的 code，但問題被拆解的方式（obligation vs goal-coupled、closed-loop feasibility、two-stage packing）已經是**未來這個問題的通用語言**。

至於論文沒回答的動態排程、網路 SLO、人形適配——這些其實都是可以出第二篇、第三篇 paper 的題目。**這是一個「開拓 sub-field」的工作，不是「刷 benchmark」的工作。**

---

## 參考資料

- **主論文**：Rosa: A Robotics Foundation Model Serving System for Robot Factories, arxiv 2607.01088 (2026-07). NVIDIA + Stanford.
- **相關系列文**：
  - [[foxconn-houston-groot-physical-ai-flywheel-2026]]
  - [[nvidia-halos-robotics-functional-safety-2026]]
  - [[nvidia-groq-3-lpx-afd-attention-ffn-disaggregation-2026]]
  - [[groot-n16-dual-system-cosmos-reason-2026]]
  - [[dragonwing-iq10-vs-jetson-thor-humanoid-soc-2026]]
- **背景數據**：Figure 03 BMW 部署（40 台、1250 hours、30k 車）；Tesla Optimus Gigafactory TX；Apptronik Apollo × Mercedes / GXO；Unitree G1 5,500+ 出貨。
