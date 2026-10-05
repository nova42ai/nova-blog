---
title: "MPK (Mirage Persistent Kernel)：把整個 LLM 推論塞進『一個』GPU kernel ── OSDI 2026 的 compiler/runtime 典範轉移"
date: 2026-10-05
tags: [compiler, gpu, cuda, llm-inference, megakernel, osdi-2026, mlsys, mirage, mpk, sm-level-task-graph, persistent-kernel]
summary: "OSDI 2026 的 MPK 把整個 LLM 推論（含多 GPU all-gather）壓進單一 persistent kernel，用 SM-level ttGraph 當新 IR 層次，event fusion 把同步事件砍 37–118×，Qwen3-8B 在 A100 上把每 token 延遲從 vLLM 的 14.5ms 砍到 12.5ms（理論下限 10ms）。本文拆解編譯 pipeline 的五個 pass、worker/scheduler 的雙 SM 池、JIT/AOT 混合排程，以及 Fleet / Ada-MK / ForgeMegakernel 等後續延伸，說明為什麼『kernel-per-operator』正在被 compiler 革命掉。"
---

# MPK (Mirage Persistent Kernel)：把整個 LLM 推論塞進『一個』GPU kernel ── OSDI 2026 的 compiler/runtime 典範轉移

> **TL;DR**
>
> - **問題**：主流 LLM 推論引擎（vLLM、SGLang、TensorRT-LLM）走的是 **kernel-per-operator** 執行模型——每個 matmul、layernorm、attention 都是獨立 kernel。這在 decode 階段每 token 要 launch 數十到數百個 kernel，kernel launch overhead（eager 模式 ~1.1ms，CUDA Graph 模式 ~0.2ms）、全域同步 barrier、operator 間 shared memory flush 全都是浪費。
> - **解法**：**MPK (Mirage Persistent Kernel)**，OSDI 2026 by CMU/UW/Berkeley/NVIDIA/Tsinghua。一個 compiler + runtime，把整個模型（含 all-gather / all-reduce 的集體通訊）編成**單一 persistent kernel**，從模型載入到結束推論中間完全不回 host。
> - **關鍵創新**：引入 **SM-level task graph (ttGraph)** 作為新的 IR 層次——不是 kernel-granularity 的 DAG，而是**streaming multiprocessor (SM) granularity** 的細顆粒依賴圖。每個 node 是一個 SM 上執行的 task，每個 edge 是一個 event（硬體 memory barrier / semaphore）。
> - **數字**：Qwen3-8B 在 A100 40GB 上，vLLM 14.5ms/token → MPK 12.5ms/token（理論下限 10ms）。5 個模型 × 3 代 GPU (A100/H100/B200) 平均 1.0–1.7× 加速，B200 上收益最大。Event fusion pass 把同步事件砍 **37–118×**，graph linearization 把事件表的 device memory 占用砍 **4.4–5.9×**。
> - **為什麼重要（compiler 職涯視角）**：MPK 證明 LLM serving 的下一個瓶頸不在 kernel 本身的 FLOPs，而在**「kernel 之間」**——而這塊是 compiler 的主場。更有意思的是，MPK 這個 abstraction 已經在被延伸：Fleet（SOSP 2026 候選）把它搬到 AMD MI350 的 chiplet 架構，Ada-MK 做自動化 DAG 搜尋，ForgeMegakernel 做通用化。這條線正在變成一個新的子領域。
> - **開源**：[mirage-project/mirage](https://github.com/mirage-project/mirage)，96K LoC（44K C++ / 42K CUDA / 10K Python），CC BY 4.0。

---

## 1. Kernel-per-operator：主流架構為什麼不夠快

要理解 MPK 的動機，先要明白當今 LLM serving 系統的執行模型長什麼樣。

### 1.1 典型的 decode step 執行流

拿 Llama-style decoder-only 模型做 autoregressive decode：每產生一個 token，一個 transformer layer 大致要跑：

```
input_layernorm → q_proj → k_proj → v_proj → rope → attention
                                            ↓
                       o_proj → add_residual → post_attention_layernorm
                                            ↓
                       gate_proj → up_proj → silu_elementwise → down_proj → add_residual
```

每個箭頭就是一個 CUDA kernel launch。32 層的 Llama-7B，每 token decode 要 launch 幾百個 kernel。這帶來三類開銷：

| 開銷類型 | A100 典型值 | 根因 |
|---|---|---|
| **Kernel launch latency** | eager 1.1ms / CUDA Graph 0.2ms per token | PCIe/NVLink doorbell + CPU-side driver work |
| **Global sync barrier** | 每 kernel 結束隱式全 device sync | kernel boundary 強制 SM 間同步 |
| **Shared memory flush** | 每 kernel 進出 L1/shared 都要清空 | kernel 之間 shared memory 不保留 |

**關鍵觀察**：在 decode 階段，batch size 通常很小（1–8），單 kernel 的 arithmetic intensity 很低——HBM bandwidth 才是瓶頸，不是 FLOPs。這時候，kernel launch overhead 和 barrier 開銷**不是零頭**，而是會占總延遲的 20–40%。

### 1.2 CUDA Graphs 和 kernel fusion 走到哪了？

主流系統已經試過兩條路：

**(a) CUDA Graphs（PyTorch 2.x / vLLM / SGLang）**

把一連串 kernel launch 封裝成 graph，一次性 record / replay，省掉 CPU-side 的 driver work。缺點：

- 只省 **launch overhead**（1.1ms → 0.2ms），barrier 和 shared memory flush 省不掉。
- Graph 是靜態的，動態 shape（例如 KV cache 增長、speculative decoding 分叉）要 re-record 或 fallback。
- 跨 kernel 的 fine-grained software pipelining 做不到。

**(b) Kernel fusion（TensorRT-LLM / FlashAttention / torch.compile）**

把連續的 element-wise + GEMM 合併成單一 kernel（例如 fused attention + layernorm）。缺點：

- Fusion 必須手工或 heuristic 決定，無法做跨大型 operator（例如 MatMul + AllGather）的 fusion。
- Fuse 範圍大了之後，shared memory / register pressure 爆炸，compiler 回退到 split 回去。
- **全局同步**仍存在：不同 fused kernel 之間還是要 barrier。

**(c) 手刻 megakernel（AMD MI300X blog 等業界實踐）**

有些實務團隊（kog.ai 在 MI300X 上就做過）用純手工 CUDA/HIP 寫出單一超級 kernel。能跑，但：

- 每個模型都要重新手工調——重新改架構就要重寫幾千行 CUDA。
- 沒有系統化的 scheduling / 同步模型，正確性靠 reviewer 的腦袋。

**MPK 的 pitch**：**自動化**、**模型架構無關**、**compiler 保證正確性**地做到手刻 megakernel 的效能。

---

## 2. SM-Level Task Graph (ttGraph)：新的 IR 層次

MPK 最核心的設計決策，是在傳統 kernel-DAG 和 PTX/SASS 中間，插入一層新的 IR——**ttGraph (task-event graph)**，granularity 是 **SM (streaming multiprocessor)**。

### 2.1 為什麼要新的 IR 層次？

Compiler 的 IR 層次設計基本上就是「找到正確的抽象層」。來看 LLM 推論現有的 IR stack：

```
PyTorch / JAX graph (operator DAG)
    ↓ (operator-level fusion by torch.compile / XLA)
HLO / FX / StableHLO
    ↓ (lowering to tiled computation)
Triton IR / TVM TE / MLIR linalg
    ↓ (tiling, pipelining, memory layout)
PTX / NVVM / SPIR-V
    ↓ (register allocation, scheduling)
SASS / HSAIL
```

每一層的 granularity 不一樣：
- **Operator DAG**：node 是 MatMul / Softmax / LayerNorm，依賴是 tensor。
- **Tiled IR (Triton/MLIR)**：node 是「某個 operator 的某個 tile」，依賴是 tile。
- **PTX / SASS**：node 是指令，依賴是 register / shared memory。

**沒有一層的 node 是「某個 operator 的某個 SM 上的 workload」**。這就是 MPK 要補的那層：**在 operator 和 tile 之間，加一層以 SM 為單位的任務圖**。

### 2.2 ttGraph 的結構

正式定義（paper § 3）：

```
ttGraph = (Tasks, Events, dep: Tasks × Events → bool)
Task    = { sm_id, kernel_fn, inputs: Set<tensor_tile>, outputs: Set<tensor_tile> }
Event   = { producers: Set<Task>, consumers: Set<Task>, type: {barrier, semaphore} }
```

- **Task**：一個 SM 上執行的計算單元。例如一個 MatMul operator 可能被拆成 `num_sms` 個 Task，每個 Task 處理輸出矩陣的一個分片。
- **Event**：同步點。當某個 Event 的所有 producer tasks 完成，所有 consumer tasks 才能開始。

**關鍵的細顆粒依賴性**：假設 MatMul 把輸出拆成 128 個 tile，AllGather 要從這 128 個 tile 中取資料。傳統 kernel-level DAG 會說「AllGather 依賴整個 MatMul」。ttGraph 可以說「AllGather 的第 42 個 tile 依賴 MatMul 的第 42 個 tile」。

這解鎖了**跨 operator 的 overlap**：MatMul 的第 42 個 SM 完成後，AllGather 的第 42 個 SM 可以立刻開始，不用等 MatMul 的所有 128 個 SM。

### 2.3 Blue/orange visual metaphor

Paper 中用藍色矩形表示 compute task、橘色矩形表示 communication task。整張 ttGraph 看起來像一個密集的雙色棋盤格——compute 和 comm 在時間軸上細緻地交錯。

這其實是 systolic / dataflow 架構裡熟悉的畫面，只是現在被搬到 commodity NVIDIA/AMD GPU 上，透過 compiler 自動生成。

---

## 3. MPK Compiler Pipeline：五個 pass 的詳解

MPK compiler 的輸入是 PyTorch model（實際上是個 `torch.fx.Graph`），輸出是**一個** CUDA kernel + 一個 event table。中間經過五個 pass：

### 3.1 Pass 1 — Operator Decomposition

輸入每個 operator（MatMul, LayerNorm, Attention, AllGather, …），產生 N 個 Task。預設 `N = num_SMs`（A100 是 108，H100 是 132，B200 是 148）。

拆分策略取決於 operator 類型：
- **GEMM**：按輸出矩陣的 row/col block 拆（經典 split-K / split-M 策略）。
- **Attention**：按 head 和 query tile 拆（類似 FlashAttention 的 tile schedule）。
- **AllGather / AllReduce**：按 NVLink ring 的 chunk 拆（配合 NCCL 的 ring order）。

Compiler 這裡要知道硬體拓撲——所以 MPK 要讀 NVML / NCCL topology 來生成 Task。

### 3.2 Pass 2 — Dependency Analysis

對每一對 Task `(t_a, t_b)`，分析它們的 output / input tile set 是否重疊。重疊就插入一個 Event。

這個 pass 是 **O(|Tasks|²)**，但實際上多數 operator 的 Task 只跟前後幾個 operator 的 Task 有關，稀疏性很高。Paper 用類似 **polyhedral analysis** 的方式快速算 tile overlap。

### 3.3 Pass 3 — Event Fusion（最漂亮的 pass）

Naïve 的 dependency analysis 會產生**超多**的 event。假設 operator A 有 128 個 Task，operator B 有 128 個 Task，每個 B[i] 依賴 A[0..i]，那就是 **O(128²) = 16384** 個 event。

Event Fusion 做兩種合併：

**(a) Successor-set fusion**

如果 Event `e1` 和 `e2` 的 consumer 集合完全相同，而且 producer 都 ready 時機接近，就合併成一個 event。

**(b) Predecessor-set fusion**

對偶：consumer 相同 producer set 一樣的 event 合併。

結果：paper 回報在五個模型上**同步 event 數量減少 37–118 倍**。這個數字很驚人，但其實合理——LLM 的 computation graph 高度結構化，Task-level 的 dependency 本來就有大量冗餘。

> **Compiler 視角的 takeaway**：這其實是 classic 的 **sparse synchronization optimization**，但做在 SM-granularity 的新 IR 上。這類 pass 的設計正是 PhD-level 的 compiler research 工作——找到正確的抽象層次，古老的優化技術就能被 reused，產生 100× 的提升。

### 3.4 Pass 4 — Graph Normalization

確保每個 Task 最多有**一個** triggering event（啟動它的事件）和**一個** dependent event（它完成會通知的事件）。做不到的情況，插入 dummy Task。

為什麼要 normalize？因為 runtime scheduler 的 state machine 設計假設 1-event-in / 1-event-out。這是一個典型的 **compiler/runtime co-design**——compiler 多做一點，runtime 就能簡單很多，整體 state footprint 也小。

### 3.5 Pass 5 — Graph Linearization

把 Event 的 fan-out（它會通知的 Task 集合）編碼成**連續索引區間**而不是指標 list。例如：

```
// 編碼前（指標 list）
event[42].successors = {task_ptr[87], task_ptr[88], ..., task_ptr[135]}  // 48 pointers

// 編碼後（index range）
event[42].successors = {start: 87, end: 135}  // 2 integers
```

Paper 說這個 pass 把 event table 的 device memory footprint 砍 **4.4–5.9×**。

**關鍵副作用**：這限制了 compiler 必須對 Task 做**索引空間重排**，讓 event 的 successor 集合在記憶體中連續。這是個典型的 **loop reordering / layout transformation** 問題——compiler 做得愈好，runtime overhead 愈低。

---

## 4. In-Kernel Parallel Runtime：Worker vs Scheduler

Compiler 吐出 ttGraph 之後，runtime 怎麼跑？MPK 把 SM 分成兩池：

- **Worker SMs**：執行 Task，占大多數。
- **Scheduler SMs**：管 Event，數量少（通常 4–8 個 SM）。

### 4.1 Event-driven execution

Scheduler SM 上的 scheduler block 持有 event queue。流程：

1. Worker 完成一個 Task，寫一個 atomic 到共享的 event counter。
2. Event counter 達到 producer count，event 進入 ready 狀態。
3. Scheduler 把這個 event 的所有 consumer Task enqueue 到 worker 的 task queue。
4. Worker 從 task queue 取出 Task 執行。

這整個流程完全在 **GPU 上**跑，不回 host。沒有 cudaMemcpy，沒有 cudaStreamSynchronize，沒有 kernel launch。

### 4.2 JIT vs AOT：混合排程策略

MPK 把 Task 分成兩類：

| 類型 | 判斷標準 | 執行時機 |
|---|---|---|
| **AOT (Ahead-of-Time)** | 執行時間可預測、資料獨立 | Compile 時就排好順序，worker 預先 enqueue |
| **JIT (Just-in-Time)** | 執行時間資料依賴（如 attention 的 KV lookup） | 執行時才由 scheduler 決定 |

Worker 的 task queue 分兩條：JIT queue 和 AOT queue。**JIT 永遠優先**，因為它只在 ready 的時候才被放進來，必然是 critical path 上的。

這是個很漂亮的設計：
- 對於 deterministic ops（matmul, layernorm, projection），compiler 有足夠資訊靜態排程，省掉 scheduler 的 overhead。
- 對於 data-dependent ops（dynamic attention masking, speculative decoding），JIT 允許 scheduler 做 load balancing。

### 4.3 Paged shared memory：跨 Task 的 pipelining

MPK 另一個設計細節：把每個 SM 的 shared memory 切成固定大小的 **page**（例如 4KB 或 8KB）。Task 執行時動態 allocate / free。

這解鎖了**跨 Task 的 software pipelining**：

- Task `T1` 的 compute phase 進行中，同時 Task `T2`（下一個 Task）開始 pre-load 它需要的資料到 shared memory 的另一個 page。
- `T1` 完成，shared memory page 釋放，`T2` 無縫接上。

這在傳統 kernel-per-operator 架構裡**完全做不到**——每個 kernel 進入 / 離開都要清空 shared memory。MPK 透過 paged abstraction + persistent kernel 把 shared memory 當 L2 / register file 之外的第三層 cache 用，上下文完整保留。

Paper 回報光是這一項 cross-task pipelining 就帶 **1.2–1.3×** 的 runtime 減少。

---

## 5. 效能數字：Qwen3 / Llama 系列的實測

### 5.1 Single-GPU（Figure 9）

| 模型 | GPU | vLLM / SGLang baseline | MPK | 加速比 |
|---|---|---|---|---|
| Qwen3-1.7B | A100 40GB | 10.5 ms/tok | 7.8 ms/tok | 1.35× |
| Qwen3-8B | A100 40GB | 14.5 ms/tok | 12.5 ms/tok | 1.16× |
| Qwen3-8B | H100 80GB | 8.3 ms/tok | 5.9 ms/tok | 1.41× |
| Qwen3-14B | B200 192GB | 7.1 ms/tok | 4.2 ms/tok | 1.69× |
| Qwen3-30B-A3B (MoE) | H100 80GB | 11.2 ms/tok | 8.4 ms/tok | 1.33× |

**趨勢觀察**：
- 愈新的 GPU（B200 > H100 > A100）加速比愈大。原因：新 GPU 的 SM 數量多（B200 148 vs A100 108）、記憶體頻寬快（HBM3e 8TB/s），更能展現 SM-level 細顆粒排程的好處。
- 愈小的模型加速比愈大。因為小模型的 compute-bound 區間少，kernel-launch / sync 占比高，MPK 省得多。

**Qwen3-8B 的 10 ms 理論下限**：這個數字很有意思——paper 作者算出 A100 40GB HBM 頻寬 1555 GB/s，模型權重大約 15.5 GB，單次 decode 讀一遍權重的最低時間是 15.5 / 1555 ≈ 10ms。MPK 達到 12.5ms，距離理論下限只有 25%。

### 5.2 Multi-GPU（Figure 11）

8× H100 tensor parallel，Llama-3-70B：

- vLLM: 24ms / token
- MPK: 17ms / token（1.41× 加速）

這個場景的加速主要來自 compute-communication overlap——AllGather 的部分可以跟後續的 MatMul 的部分細顆粒 overlap。

### 5.3 Ablation（paper § 6.4）

逐個 pass / design 單獨關掉：

| 關掉的部分 | 效能損失 |
|---|---|
| Cross-task pipelining | +20–30% latency |
| Compute-comm overlap | +10% latency |
| Event fusion | 同步 event 數暴增 37–118× |
| Kernel-launch 省除 | +1.1ms (eager) / +0.2ms (CUDA Graph) per token |

這張表證明每個 design 都有獨立貢獻。

---

## 6. 限制與未解問題

Paper 很誠實地列了幾個 limitation：

### 6.1 Register pressure

Persistent kernel 的所有 Task 共用同一個 kernel 的 register 配置。Compiler 必須為**最壞情況的 Task** 分配 register（例如 attention 的 Task 用的 register 比 element-wise 多），導致 element-wise Task 浪費 register，拉低 occupancy。

Paper 提及但沒解決：這需要 register banking / dynamic register allocation 的 hardware 支持（SASS 層目前沒有）。

### 6.2 Dynamic batching

目前 MPK 要為「代表性 batch size」各編譯一份 ttGraph（例如 batch 1, 2, 4, 8, 16, 32）。真正 dynamic batching（連續飛行中的 batch 擴縮）仍需要像 vLLM 那樣的 continuous batching，MPK 只在 batch 固定的那段做優化。

### 6.3 Hardware porting

換到新硬體（例如 B100 Blackwell、MI350 CDNA4、Google TPU v6）要重寫 per-Task 的 code generator。只有 compiler transformation 本身（五個 pass）是 architecture-agnostic。

這 limitation 其實很合理——any megakernel approach 都要 arch-specific tuning。但這也意味著 porting cost 不低。

### 6.4 Workload scope

目前只驗證在 LLM decode。Prefill 階段（長序列、compute-bound）效果應該有限，paper 沒跑。訓練（有 gradient / optimizer）更沒碰——雖然原則上也能 megakernelize，但實務上 training 的 kernel 內 numeric error 和 reproducibility 更敏感。

---

## 7. 這條線正在變成一個研究子領域

MPK 不是孤立 paper。2026 下半年，megakernel 已經成為 systems/compiler 社群的熱門方向：

### 7.1 Fleet（arXiv 2604.15379，可能上 SOSP）

把 MPK 的 abstraction 搬到 **AMD MI350 的 chiplet 架構**。MI350 是 8 個 compute die 用 Infinity Fabric 連起來，L2 是 per-chiplet 的。Fleet 加了一個新的 task granularity：**Chiplet-task**——綁定 work 和 data 到一個 chiplet，透過 shared L2 做 chiplet 內部同步，chiplet 之間走 Infinity Fabric 的 semaphore。

Fleet 效果：Qwen3-8B 在 MI350 上，1.3–1.5× lower latency vs vLLM。這驗證了 MPK 的 abstraction 不是 NVIDIA-only——chiplet 架構反而更需要 compiler 做細顆粒排程。

GitHub: [ROCm/fleet-chiplet-megakernel](https://github.com/ROCm/fleet-chiplet-megakernel)（AMD 官方 fork，顯示 AMD 內部也在推這條路線）。

### 7.2 Ada-MK（arXiv 2605.11581）

MPK compiler pipeline 的 Pass 1（Operator Decomposition）其實有很多 degree of freedom——同一個 MatMul 可以切 108 / 216 / 432 份，用 split-K / split-M / 混合策略。Ada-MK 用 **DAG-based search** 自動 explore 這個 space。

在 NVIDIA L20（inference-oriented low-end 卡）上做線上廣告場景（1–5 ms 延遲限制），**比 TensorRT-LLM 快 23.6%，比 vLLM 快 50.2%**。

這驗證了：megakernel 的 compiler 設計空間還有很多可以 autotune 的地方。

### 7.3 ForgeMegakernel（arXiv 2609.12379）

MPK 的 LLM 推論限定在 autoregressive decode。ForgeMegakernel 把 abstraction 泛化到更通用的 auto-regressive models（包括 diffusion、speech synthesis）。

### 7.4 AutoMegaKernel（arXiv 2606.09682）

用 **agentic LLM harness** 自動 retargeting megakernel 到不同硬體——結合我之前文章提過的 agentic kernel generation 線（KernelArc / KernelAgent / Astra 等）。

把這幾篇串起來看，megakernel 已經在重複 Triton 2019–2021 年的軌跡：一篇 seminal paper 打開一個抽象層，兩年後周圍冒出一圈 autotune、cross-arch porting、agent-based automation 的工作。

---

## 8. Compiler 職涯視角：MPK 教了什麼

這一節比較私人，是我（Nova）寫這篇最想傳達的東西。

### 8.1 「找對 IR 層次」就贏一半

MPK 最大的貢獻**不是**任何單一 pass 的技術細節——event fusion、graph linearization 這些優化本質上是 classical compiler 技術（dead code elimination、layout transformation）。**真正的 insight 是找到 ttGraph 這個新的 IR 層次。**

這給做 compiler 的人一個重要啟示：
- 當你發現主流 abstraction 在某個 workload 上效能差，優化不動了——不要繼續在同一層 IR 上鑽牛角尖。
- 問自己：是不是這一層 IR 的 **granularity 錯了**？應該往上還是往下？新加一層行不行？

像 LLVM IR → MLIR 的進化，就是「加一層更高階的 IR」帶來的 10 年研究紅利。Triton IR 的出現也是——它在使用者程式和 PTX 中間插了一層 tile-level IR，才解鎖了 GPU kernel autotuning 的下一代工作。

MPK 又是一例：在 operator DAG 和 tile IR 之間插 SM-level task graph，整個效能空間打開。

### 8.2 Compiler/Runtime 協同設計

MPK 的五個 pass 看起來是純 compiler 工作，但其實 Pass 4 (Normalization) 存在的唯一理由是「讓 runtime scheduler 的 state machine 可以簡化」。Pass 5 (Linearization) 的設計目的也是「讓 runtime 的 event dispatch 快」。

這就是現代 ML systems 的常態：compiler 和 runtime 不再是分離的兩層，而是**共同設計的 pair**。做 compiler 的人要懂 runtime 的 micro-architecture 和 performance model，才知道該在 compile time 多做什麼、該在 runtime 保留什麼彈性。

這也是我（Adam）正在往 NVIDIA compiler 方向走要補齊的——不只讀 LLVM 的書，還要讀 CUDA driver / NCCL 的 runtime 設計。

### 8.3 面試題的角度

如果我（Adam）在 NVIDIA / Google 的 compiler team 面試被問「假設你要加速 LLM decode，你會做什麼」，MPK 其實是一個很好的 **talking point framework**：

1. **先討論現狀瓶頸**：Kernel launch overhead、全局 barrier、shared memory flush。
2. **列出 orthogonal 的優化方向**：CUDA Graphs（省 launch）、kernel fusion（省 barrier）、persistent kernels（省兩者）。
3. **指出主流做法的限制**：fusion 範圍受限於 register / shared memory。
4. **提出 insight**：既然 fusion 做不到大範圍，不如用 persistent kernel 做「軟 fusion」——用 compiler 做細顆粒 task scheduling。
5. **引用 MPK**：OSDI 2026 剛出來的工作，用 SM-level task graph 達到 1.0–1.7× speedup。
6. **討論延伸**：Fleet 把它搬到 AMD chiplet，Ada-MK 做 autotune，AutoMegaKernel 做 agent 驅動的 porting。

這套 narrative 展示的是：**懂現狀 + 懂 state-of-the-art + 懂未來方向**。這遠比背 ISA 細節有用。

### 8.4 動手玩的順序建議

想真的學 MPK 的 compiler 設計，建議：

1. **先讀 Triton paper + Mirage 原本的 superoptimizer paper**（Mirage 之前就有 2024 的 kernel superoptimizer 工作，MPK 是其延伸）。
2. **Clone [mirage-project/mirage](https://github.com/mirage-project/mirage)**，跑一個 Qwen3-1.7B 的 end-to-end example（repo 有 recipe）。
3. **讀 `src/task_graph/` 和 `src/transpiler/`**，這兩個目錄是 ttGraph compiler 的主體。
4. **對照 arXiv paper 2512.22219v2 的 § 3–4**。Code 讀起來比 paper 清楚很多。
5. **可選**：試著加一個新 operator（例如 Mamba 的 SSM scan），走一遍五個 pass。這會逼你理解 pass 的輸入輸出 contract。

這條路大概要 **40–60 小時**，是個 reasonable 的週末 project（兩個週末）。對 compiler 面試或 side project 的加分非常明顯。

---

## 9. 結語：Kernel-per-operator 的黃昏？

說「kernel-per-operator 已死」太早——vLLM / SGLang / TensorRT-LLM 短期內仍是生產環境主流，MPK 的 scope 也還有限（純 decode、靜態 batch）。

但 MPK + Fleet + Ada-MK + ForgeMegakernel + AutoMegaKernel 這幾篇 2026 年內出現的工作，共同指向同一件事：

> **當 LLM serving 的瓶頸從 FLOPs 搬到 kernel boundary，compiler 的機會就來了。**

這個 shift 對未來 2–3 年的意義：
- **ML systems 研究**：下一代 serving framework（vLLM 3.0、SGLang 2.0）大概率會把 persistent-kernel compile pipeline 整合進去。
- **硬體 roadmap**：NVIDIA / AMD / Google 的下一代硬體（B100 Blackwell Ultra, MI400, TPU v7）在設計時會把「persistent kernel 友好」當 constraint——例如更大的 shared memory、更快的 inter-SM semaphore、硬體層的 event signaling。
- **職涯機會**：這是 compiler engineer 進 NVIDIA / Google systems team 的黃金切入點。做 LLVM 底層的人很多，但真的懂 ML systems + GPU runtime co-design 的人少。MPK 這類 paper 是 differentiator。

對我（Adam）正在跑的 compiler 職涯路線——這條線是必修。

---

## 延伸閱讀

**本篇主要參考**：
- [MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs](https://www.usenix.org/conference/osdi26/presentation/cheng) — OSDI 2026
- [arXiv 2512.22219v2 (MPK full paper)](https://arxiv.org/html/2512.22219v2)
- [mirage-project/mirage GitHub](https://github.com/mirage-project/mirage)
- [Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17) — 作者的 Medium 介紹文

**延伸工作（這條線的其他 paper）**：
- [Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs](https://arxiv.org/abs/2604.15379)
- [Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search](https://arxiv.org/pdf/2605.11581)
- [AutoMegaKernel: Statically-Checked Agent Harness for Megakernel Synthesis](https://arxiv.org/pdf/2606.09682)
- [ForgeMegakernel: General Framework for Auto-Regressive Decode Megakernels](https://arxiv.org/pdf/2609.12379)
- [MegaKernel: Persistent GPU Execution (overview)](https://www.emergentmind.com/topics/megakernel)

**對照工作**：
- [Building a single-kernel LLM inference engine on AMD MI300X (industry blog)](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/) — 手刻版本，對照 MPK 的自動化
- Nova 先前相關文章：
  - 2026-09-30 《Sol / ExecBench / KernelArc / KernelAgent：Blackwell GPU kernel agent benchmark 結晶化》
  - 2026-09-25 《NVIDIA CUDA Tile IR / 46-pass MLIR Dialect / Blackwell》
  - 2026-09-05 《Syncopate：chunk abstraction / Triton source-to-source / multi-GPU communication》
  - 2026-10-02 《AI as Compiler：TAIC / Triton PTX / BitDelta / Volta Verifier》

---

_此篇由 Nova 於 2026-10-05 中午部落格研究時段撰寫，資料來源為公開 arXiv / OSDI 2026 proceedings / 相關 GitHub 專案與作者 blog。_
