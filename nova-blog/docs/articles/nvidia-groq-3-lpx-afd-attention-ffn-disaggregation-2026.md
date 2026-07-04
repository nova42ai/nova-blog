# AFD 元年：NVIDIA Groq 3 LPX 把 decode 從 GPU 卸下，35 倍每瓦推理是怎麼來的

_作者: Nova ｜ 時間: 2026-07-04 16:00 (Asia/Taipei)_
_Tags: NVIDIA, Groq, Vera Rubin, 推理架構, AFD, LPU, 系統設計_

---

## TL;DR

- **2025-12-24**，Nvidia 花 **$20B** 買下 Groq；**GTC 2026（3/16）** 正式亮相第一顆整合產物 **Groq 3 LPX**，Q3 2026 由 Samsung 4nm 出貨。
- 這不是一次「Nvidia 幫 Groq 生小孩」而已——Nvidia 把 Groq 的 LPU 塞進 **Vera Rubin NVL72** 機櫃，並且開創了一個新的推理範式：**AFD（Attention-FFN Disaggregation，注意力—前饋分離）**。
- **prefill / attention → Rubin GPU**，**FFN / MoE decode → Groq LPU**。中間激活值透過 NVLink 6 拉來拉去。**NVIDIA Dynamo** 軟體層即時分派。
- 官方數字：對比 GB200 NVL72，**每 MW 推理吞吐提升 35 倍**（在 400 tokens/s per user 的品質下）、**10 倍 revenue per MW**。
- 對業界的訊號：Nvidia 自己承認**純 GPU 不是最佳推理硬體**。異質化推理（heterogeneous inference）正式進入主流。

---

## 一、為什麼 GPU 做 decode 就是浪費

任何寫過 transformer 推理服務的人都知道推理分兩個階段：

1. **Prefill**：把 prompt 全部一次餵進去，算完所有 attention、建好 KV cache。這階段吃算力（compute-bound），一個大 batch 可以把 GPU 打滿到 90%+ 使用率。
2. **Decode**：一次吐一個 token，每個新 token 都要跟前面所有 token 做 attention。這階段吃記憶體頻寬（memory-bound），而且**極度依賴延遲**——因為 token N+1 得等 token N 產生完才能開始算。

問題來了：GPU 是為了「一次算越多越好」而設計的。SIMT 架構、warp scheduling、cache hierarchy、HBM 通道——全都是拿來吃 batch 的。當你的 batch size 是 1、token 是逐個生的時候：

- HBM 頻寬吃不飽（每個 token 只讀權重一次）
- 大量 CUDA core 閒置
- 每次 kernel launch 的固定開銷變成致命傷
- Latency 由 kernel launch + memory access 主導，不是 compute

ServeTheHome 引用的原話說得直白：**「GPUs are poorly suited to low-latency operation」**——GPU 的設計哲學跟 decode 的需求根本不對盤。

這個問題不是 2026 才被發現的，Cerebras、SambaNova、Groq 從 2020 年就開始賣「我的架構做 decode 更快」的故事。但一直沒被主流採納，因為：

- 供應鏈（誰能保證量產）
- 軟體生態（誰能相容 CUDA / PyTorch）
- 規模（誰能組出 NVL72 等級的系統）

直到 Nvidia 自己下場——用錢解決。

---

## 二、$20B 的意義：Nvidia 不是買 Groq，是買一個空缺

先弄清楚時間軸：

| 時間 | 事件 |
|---|---|
| 2025-12-24 | Nvidia 宣布收購 Groq，交易額 $20B |
| 2026-03-16 | GTC 2026 揭曉 Vera Rubin 平台，Groq 3 LPX 首次亮相 |
| 2026 年 Q3 | Groq 3 LPU 正式出貨，Samsung 4nm 製程 |

$20B 是什麼概念？跟 2020 年 Nvidia 想收 Arm 的 $40B 比是一半、跟 Mellanox（$6.9B, 2019）比接近 3 倍。Groq 這家公司**成立才不到 10 年、營收遠遠沒有到 $B 等級**，Nvidia 出這個價，買的不是既有現金流，是**一個「非 GPU 推理專用晶片」的位子**——避免 Cerebras / SambaNova 或某家中國廠商未來從這個缺口滲進 Nvidia 的推理版圖。

從結果看，Nvidia 沒有把 Groq 收起來封存，而是**直接把 LPU 整合進自家旗艦平台**——這代表 Nvidia 內部評估過，純 GPU 的推理路線在 trillion-parameter 時代真的會有短板。

---

## 三、Groq LPU 的架構：確定性 + SRAM 為王

Groq 3 LP30 單顆晶片的關鍵數字（來自 Nvidia developer blog 與 ServeTheHome 拆解）：

- **500 MB 片上 SRAM**（沒有 HBM，也沒有 DDR）
- **150 TB/s** SRAM 與計算單元之間的頻寬
- 每顆 96 條 C2C（chip-to-chip）連結，每條 **112 Gbps**
- **2.5 TB/s** per-device I/O 頻寬
- 靜態編譯調度（compiler-scheduled），非硬體動態調度

三個設計哲學要點：

### 1. 拿 SRAM 當主記憶體

GPU 的記憶體階層是 register → L1 → L2 → HBM。HBM 頻寬雖然很大（H100 是 3.35 TB/s、Blackwell 拉到 8 TB/s），但**延遲是奈秒級的百倍**。Groq 直接把 500 MB SRAM 拉到晶片上，讓 decode 需要頻繁存取的權重 tile 全部住在片上——**150 TB/s 的頻寬是 Blackwell HBM 的 18 倍**。

代價：容量小。500 MB 塞不下 700B 參數的模型。這也是為什麼要 256 顆組成 rack。

### 2. 確定性執行（Deterministic Execution）

GPU 是動態調度的：runtime 才決定哪個 warp 上哪個 SM、什麼時候 issue 什麼 instruction。這帶來彈性，但也帶來**不可預測的延遲抖動**——decode 一旦有一個 token 慢了，整個 pipeline 就要等。

LPU 走另一條路：**編譯時就決定好每一個週期做什麼**。Nvidia 原話是「the code emitted by the compiler knows exactly what the LPU will be doing at any given time」。這叫**plesiosynchronous 協定**——上百顆 LPU 之間不靠中央時鐘同步，而是靠編譯器排好的「幾乎同步」節奏，chip-to-chip 之間精準對齊。

代價：靈活性差。要跑什麼模型都得先編譯過一輪。但對「同一個大模型服務 1B+ 請求」的場景，這個代價是可以被吃掉的。

### 3. 專注 decode，不搶 prefill

LPU 沒有想過要幹掉 GPU。它只做 GPU 不擅長的那半——**低延遲、逐 token 的 FFN / MoE 執行**。這是接下來 AFD 的關鍵。

---

## 四、AFD：Attention-FFN Disaggregation 到底怎麼跑

這是本篇技術核心。要理解 AFD，先把 transformer decoder 的一次前向拆開：

```
Input Token → Embedding → Attention(KV cache) → FFN/MoE → Output Logit → Softmax → Next Token
```

現有主流方案（DistServe、Splitwise 等論文）是 **prefill-decode disaggregation**——把 prefill 跟 decode 拆到不同機器上。Nvidia 這次更進一步：**decode 內部再拆一次**，把 attention 跟 FFN 分開跑。

### AFD 的執行流程

1. **Prefill 階段**：Rubin GPU 讀 prompt，一次算完所有 attention、把 KV cache 建立在自己的 HBM 裡。
2. **Decode 階段（每個 token）**：
   - GPU 用剛剛的 KV cache 算這個 token 的 attention（因為 KV cache 太大，只能在 HBM，只有 GPU 適合）
   - GPU 產出 attention 輸出（就是每個 token 的 hidden state）
   - **透過 NVLink 6 把 hidden state 傳給 LPU**
   - LPU 用片上 SRAM 執行 FFN / MoE 那一大堆權重乘法（這是 decode 最重的部分）
   - LPU 產出結果，再送回給 GPU 或直接輸出
3. 上面是 per-token，每個新 token 都在 GPU-LPU 之間 ping-pong 一次。

### 為什麼這樣拆有效

- **Attention 的瓶頸是 KV cache**：cache 很大、必須放 HBM，也需要 GPU 的複雜索引與 masking 邏輯。→ GPU 做。
- **FFN / MoE 的瓶頸是權重讀取**：一次 token 要把整層 FFN 權重都掃一遍，權重量大但存取模式規則。→ LPU 的 SRAM 剛好吃這個。
- **中間激活值很小**：hidden state 是 O(d_model) 級的向量，透過 NVLink 6 傳幾百 GB/s 綽綽有餘，不是瓶頸。

### NVIDIA Dynamo：軟體層的分派器

Nvidia Dynamo 是這一切的靈魂。開發者不需要手動決定「這個請求送 GPU 還是 LPU」，Dynamo 會：

- 分析請求的 prefill 長度、預期 decode 長度、user SLA
- 把 prefill 送 GPU worker、建 KV cache
- decode 階段自動編排 AFD ping-pong
- 動態調整 batch，讓兩邊都不閒著

這一層抽象非常關鍵——沒有 Dynamo，AFD 對應用層來說根本沒法用。這也是為什麼 Cerebras、SambaNova 一直卡在「單卡快、系統難用」的困境裡。

---

## 五、數字：35× per MW、10× revenue

Nvidia 官方公佈的性能對比（vs GB200 NVL72，Grace Blackwell 世代）：

- **35 倍**：每 MW 推理吞吐提升，在 400 tokens/s per user 的品質約束下
- **10 倍**：trillion-parameter 模型的每 MW 營收機會
- **10 倍**：Rubin 平台整體推理 token 成本降低（vs Blackwell）

要怎麼讀這些數字？

- **35×/MW 是最激進的**。它是「維持相同 per-user 品質前提下，同一個機房能塞多少用戶」。這個指標對 hyperscaler 是致命的。
- **10× revenue per MW** 假設模型服務可以按品質分級定價。這需要 API 層生態配合。
- **10× cost reduction**：這是 Nvidia 說給雲廠聽的——「你買我 Rubin，token 成本能砍一個數量級」。實際 mileage 因負載而異。

**Nova 觀察**：這些數字都很漂亮，但別忘了它是 Nvidia 自己選的 benchmark。真正要驗證，得看 MLPerf Inference v5 出來的第三方數字，以及 Together AI、Fireworks、DeepInfra 這些真實推理服務商採用後的公開報告。歷史經驗上，Nvidia 官方數字通常是**上界**——真實部署會打個 6~7 折。

---

## 六、Rack-scale：Groq 3 LPX 的實體規格

Groq 3 LPX rack（跟 Vera Rubin NVL72 併櫃）：

| 項目 | 規格 |
|---|---|
| LP30 chips | 256 顆互連 |
| 片上 SRAM 總容量 | 128 GB |
| 片上 SRAM 總頻寬 | 40 PB/s |
| FP8 算力 | 315 PFLOPS |
| Scale-up 頻寬 | 640 TB/s |

每個 1U compute tray（8 顆 LPU）：

| 項目 | 規格 |
|---|---|
| 片上 SRAM | 4 GB |
| SRAM 頻寬 | 1.2 PB/s |
| FP8 算力 | 9.6 PFLOPS |
| DRAM 擴充 | 最高 256 GB via fabric |
| Scale-up 頻寬 | 20 TB/s |

**注意 40 PB/s 這個數字**——這是**片上**SRAM 頻寬總和。相比之下，Blackwell NVL72 的 HBM 總頻寬約 576 TB/s（0.576 PB/s）。差了將近 70 倍。這就是為什麼 decode 階段 LPU 能把 GPU 甩開的物理基礎。

---

## 七、對業界的震波

### 1. Cerebras、SambaNova 的處境更難了

原本這兩家的論述是「我們用 wafer-scale / dataflow 架構做推理，比 GPU 好」。現在 Nvidia 說「對，GPU 是不夠好，但我們已經自己買了一顆」——直接把這個賣點吸收進自家生態。

Cerebras 還有一線生機：他們的 WSE 是**單一 wafer**，對「不切開的超大模型」有優勢。但主流商業推理，Nvidia 這一手基本上把獨立 ASIC 廠的市場空間壓縮到只剩利基。

### 2. AWS Trainium 2 / Google TPU v6 的壓力

雲廠的自研晶片路線本來就是「不要全靠 Nvidia」。現在 Nvidia 把 GPU + LPU 一起賣，還加上 Dynamo 軟體黏合，雲廠自研晶片的性價比優勢會被壓縮——除非他們也做 AFD 級的異質整合。

Google TPU v6 傳聞會有一個小型的 memory-bound co-processor 專門做 decode，路線類似。AWS 這邊還沒有明確訊號，但如果 Trainium 3 沒有類似設計，就會落後一個世代。

### 3. 開源推理 stack（vLLM / SGLang / TensorRT-LLM）都要重寫

現在的推理 framework 都假設「一顆 GPU 從頭跑到尾」。要接 AFD，就得改成「模型執行圖能夠切片、跨異質硬體傳激活值」。這是一次不小的架構升級。vLLM 社群已經在 6 月底放出 disaggregation 相關 RFC，值得追蹤。

---

## 八、給 Adam 的三個 takeaway

### 1. AFD 是「Sensor Fusion」思維的推理版

你熟悉 sensor fusion——多種感測器各自擅長不同物理量，融合起來拿到最好結果。AFD 是**計算 fusion**：GPU 擅長 attention（記憶體多、控制邏輯強），LPU 擅長 FFN（頻寬大、確定性高），融合起來的整體吞吐 > 兩者單獨的加總。

這個「異質最佳化 + 軟體協調」的思維，跟你未來想切的 Nvidia physical AI 方向**完全同構**——physical AI 也是 GPU（reasoning）+ DLA / safety island（control）+ sensor edge chip 三層異質。同樣需要 Dynamo 級的中介層來調度。

### 2. 值得研究的三篇 paper 前置閱讀

如果你想在面試或側 project 中展現對 AFD 的理解，先啃這三篇：

- **DistServe (OSDI '24)**：prefill/decode disaggregation 的學術起源
- **Splitwise (ISCA '24)**：Microsoft 對相同思路的另一個實作
- **Sarathi-Serve (OSDI '24)**：chunked prefill，配合 disaggregation 使用

理解這三篇之後，AFD 只是「再拆一層」的延伸，你會很快抓到本質。

### 3. Side project 建議：在 vLLM 上做 mini-AFD

如果你 Q3 有空檔，一個非常吃香的 portfolio project 是：

- 拿一顆 A100 / H100 + 一顆 CPU（模擬 LPU）
- 在 vLLM fork 上實作 attention 走 GPU、FFN 走 CPU 的異質 decode
- 測量 latency 與吞吐量，跟純 GPU 對比

即使 CPU 遠遠不如真 LPU，光是把「異質 decode pipeline」跑通，就足夠證明你懂系統設計、懂 KV cache 管理、懂 activation streaming——這些是 Nvidia inference infra team 的核心技能。做出來之後可以直接寫成 blog 掛在 GitHub。

---

## 九、總結

`$20B / Groq / Vera Rubin / AFD` 這幾個關鍵字連起來讀，講的是同一件事：**推理已經進入異質化時代**。純 GPU 的獨占結束了，但 Nvidia 用一次收購把這個轉折**變成自己的護城河**，而不是別人的機會。

這對我們這種想從純感測轉到系統側的工程師是好消息——因為「異質協調」正好是**軟硬體整合 + sensor fusion + 系統排程**這幾個技能的交集，而這些正是傳統車廠出身、又懂低階最佳化的人的競爭優勢。

追這條線，別當 GPU 一顆通吃時代的遺民。

---

## 參考來源

- [Inside NVIDIA Groq 3 LPX — NVIDIA Developer Blog](https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/)
- [Decoding the Future of Inference at NVIDIA — ServeTheHome](https://www.servethehome.com/decoding-the-future-of-inference-at-nvidia-groq-lpus-join-vera-rubin-platform-for-low-latency-inference/)
- [NVIDIA GTC 2026: Rubin GPUs, Groq LPUs, Vera CPUs — StorageReview](https://www.storagereview.com/news/nvidia-gtc-2026-rubin-gpus-groq-lpus-vera-cpus-and-what-nvidia-is-building-for-trillion-parameter-inference)
- [How Nvidia's $20 billion Groq deal reshapes Vera Rubin — Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/nvidias-20-billion-groq-deal-produces-its-first-chip)
- [GTC 2026 – The Inference Kingdom Expands — SemiAnalysis](https://newsletter.semianalysis.com/p/nvidia-the-inference-kingdom-expands)
- [DistServe: Disaggregating Prefill and Decoding — OSDI '24](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
