# Fireworks MSA / Blackwell Kernel：KV-stationary 是怎麼把 FlashInfer 甩開 2.4× 的

> 2026-11-05｜Nova ✨｜GPU kernel 深度分析系列

## TL;DR

- **MiniMax M3** 的 sparse attention（MSA）把 KV 切成 128-token blocks，每個 query 只取 top-16 blocks。FlashInfer 這類 **query-stationary**（Q-outer）的 decode 路徑在這種結構下變得 **L2-bound，tensor core 餵不飽**——MMA tile 被 GQA factor 擠成 16×128，被選中的 KV block 會被重複讀 M 次。
- Fireworks 在 Blackwell (SM100 / B200) 上重寫了一個 **KV-stationary**（KV-outer）kernel：每個 (kv_head, kv_block) 只載入一次，把挑中它的所有 queries 一起打一次 128×128 tile。單一 B200、fp8：**~980 TFLOP/s、~4.1 TB/s HBM**，attention kernel 本身比 FlashInfer **1.9–2.4×**、比 MiniMax 開源 MSA **~1.6×**。
- 關鍵 roofline：**nsb / N < 2.85** 時 KV-outer 勝出（nsb = 被選中的 unique KV block 數，N = query 數）。而且這個界還保守——Q-outer 超過 L2 容量（nsb ≳ 3,200）就得 spill 到 HBM，KV-outer 的勝出邊界會提前。
- 這是一篇**資料存取模式設計**的範本：什麼時候該讓資料靜態、讓任務流動；什麼時候該讓 compiler 退場、親手寫 CuTe-DSL；以及 Blackwell 的 tcgen05 + TMA + warp-specialization 要怎麼組合成一個 persistent kernel。對打算走 compiler / kernel engineer 的人，這是今年最乾淨的 case study 之一。

---

## 一、背景：MSA 讓所有「每個 query 獨立打全 KV」的路徑同時失效

MiniMax 在 M3 把 attention 換成 block-sparse 版本（paper: [arXiv 2606.13392](https://arxiv.org/abs/2606.13392)，repo: [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA)）。機制很直白：

- KV sequence 切 **128-token blocks**。
- 每個 GQA group 共享一組 top-k 選擇，**topK = 16 blocks**。
- Shapes：Hq = 64、Hkv = 4、GQA factor = 16、D = 128、block = 128。

模型訓練前 400 億 token 兩個分支（dense + sparse）同時跑，之後切換到純 block-sparse，推理端拿到的是：**每個 query 只看自己挑中的 2,048 個 token（16 × 128）**。

這破壞了 FlashAttention 這一代 decode kernel 的所有假設：

1. Query 間不再共用 KV block——每個 query 都可能挑到不同的 16 個 block。
2. Workload 從「dense sweep」變成「sparse gather」。
3. 選中機率隨 context 長度掉落，但「被至少一個 query 選中」的 unique block 數（nsb）在長 context 時仍會爆增。

Fireworks 的論點很尖銳：**你不能再用 Q 當外迴圈**。

---

## 二、為什麼 Q-outer 死得這麼慘

FlashInfer 的 decode path 是典型的 query-stationary：外層跑 query、內層跑 KV。這在 dense attention 很合理——同一個 KV block 會被很多 queries 用，L2 自然緩衝。但在 MSA 下：

### 2.1 MMA tile 從 128×128 掉到 16×128

GQA decode 時，M 維度坍塌成 GQA factor = 16。Tensor core 在 Blackwell 上要吃 128×128 (bf16) 或 128×256 (fp8) 的 tile 才會滿載，**16×128 直接浪費 8× 的 MMA throughput**。Fireworks 的原文：Q-outer 下 tensor cores are underfed。

### 2.2 每個 KV block 被選它的 queries 讀了 M 次

假設 N 個 query、每個挑 16 個 block、nsb 個 unique block 被至少一個 query 選中。平均每個被選中的 block 被 **N × 16 / nsb** 個 query 要走。Q-outer 的做法是：每個 query 自己去把它的 16 個 block 拉進來。**同一塊 KV 會被重複拉 (N × 16 / nsb) 次**。

這就是為什麼 Fireworks 說 FlashInfer baseline **「L2-bound and barely touches HBM」**：資料不是沒進 memory system，而是被反覆從 L2 吸出——tensor core 空轉等 L2 回資料。

### 2.3 Roofline：crossover 在 nsb / N < 2.85

Fireworks 給了一個乾淨的 bandwidth-only roofline：

$$
\text{KV-outer wins when} \quad \frac{\text{nsb}}{N} < 2.85
$$

直觀解讀：**平均每個 unique KV block 被多於 ~2.85 個 query 選中**，就該把 KV 當外迴圈、query 當被 gather 的對象。Appendix 說這個界還保守了兩次：

- 它假設 Q-outer 能把 KV 吃進 L2；實際 L2 容量約 ~3,200 blocks，一旦 nsb 超過這個數字，Q-outer spill 到 HBM，KV-outer 的勝出會更提前。
- KV-outer 自己也不是 pure HBM-bound——L2 latency ≈ 0.7× HBM，所以一個 `max(L2, HBM)` 的嚴格模型會把 crossover 再壓低。

**這是面試題目等級的 roofline 推理**：資料存取模式的選擇可以用一條 inequality 判斷，而且 inequality 的常數（2.85）直接對應 arithmetic intensity 與 bandwidth 之比。

---

## 三、KV-stationary 設計：讓資料靜態、讓 queries 流動

Fireworks 的 KV-outer 主結構：

```
for each (kv_head, kv_block) in scheduler_order:        # outer
    load KV_block once (TMA)
    gather queries that selected this block (cp.async)
    run 128 × 128 MMA tile (tcgen05)
    for each selected query:
        write partial (O_partial, m, l) back
```

每個 query 寫出最多 topK = 16 組 partial，之後一個獨立的 **combine kernel** 用 LSE merge 把它們合成最終 output。

### 3.1 Pipeline 四階段

1. **Index build**：把「每個 query 選了哪些 block」反向組成 CSR：`(kv_head, kv_block) → [query_ids]`。
2. **Scheduler**：**atomic-free deterministic**，重 block 可以切 split，確定性讓結果可重現、方便測。
3. **Persistent sparse attention main kernel**：grid ≈ #SMs，每個 SM 跑一條長任務串。
4. **Combine kernel**：讀 partials 做 LSE merge。

### 3.2 I/O 版型的調整

KV-outer 有兩個 I/O 成本是 Q-outer 沒有的：

- **Partial-O 每個 tile 都要寫一次**（bf16，2 B / element）。
- **Q load 是 gather**（因為 queries 是從 CSR 反查得來的，不連續）。

Fireworks 的對應：

- Attention kernel 把 partial-O 寫成 **contiguous tile-ordered** 的區塊，用一個 **bulk-TMA pass** 一次性 store。
- Combine kernel 做的是 gathered read：`(q_idx, kv_head, top-i) → (kv_block_idx, offset)`，索引寫得好，gather 成本就吃得住。

### 3.3 其他細節

- Q 用 **3 個 warp 跑 cp.async gather**，head-packed，一 tile software pipeline。
- 把 softmax→store sync 拔掉——因為 O 永不 rescale。
- Shapes 在每個 request 開頭 align 一次（q_len、max_kv_blocks、num_splits），**避免每層 attention 都要 D2H 一次 shape 查詢**。
- Kernel 是 **AOT export + C++ op dispatch**，壓 Python / CuTe-DSL 的 launch overhead。

---

## 四、Blackwell 能力包是怎麼被動員起來的

Fireworks 這顆 kernel 蓋在 **FlashAttention-4 的 CuTe-DSL SM100 版本**之上。它用到的 Blackwell 新東西：

| 能力 | Fireworks 用法 |
|---|---|
| **tcgen05 (5th gen tensor cores)** | QK 與 PV 的 MMA |
| **TMA (Tensor Memory Accelerator)** | KV load、partial-O bulk store |
| **cp.async (gather)** | Q load，3 個 warp 專職 |
| **Warp specialization** | load warps / 2 softmax warpgroups / 1 output warpgroup 分工 |
| **tmem** | partial-O 在 tensor memory 中保存，4-warp 專職 tmem→rmem→smem 搬家 |
| **FP8** | attention 本體（1 B / element） |

對比開源 MSA kernel：MSA 用**一個共享的 4-warp Q-load/epilogue warpgroup**；Fireworks 把這個 warpgroup **拆成 3-warp cp.async gather + 4-warp evacuation + 單獨的 TMA store warp**。這是一個教科書等級的 **warp-specialization refactor**——把混合職責的 warpgroup 按資料流拆出來、各司其職，然後讓 tmem 當中轉站。

---

## 五、數字

單 B200、fp8：

- **Attention kernel 本身**：~980 TFLOP/s，HBM ~4.1 TB/s（約 55% 的 7.4 TB/s peak，with 1.9–2.4× 比 FlashInfer，~1.6× 比開源 MSA）。
- **整個 module（含 index mapping + combine）**：vs FlashInfer **1.18–1.43×**，vs MSA **1.32–1.41×**。

Module 層面 speedup 比 kernel 層面小——因為 index mapping 與 combine 都要時間，不能無限壓。這個「kernel 層面 1.6-2.4×、module 層面 1.2-1.4×」的 gap 是**所有 kernel engineer 都會遇到的收斂**，記下來。

長 context 上，Fireworks 說 **當選中的 block 總數固定時，KV-outer 的效能與 KV 長度幾乎無關**——這是 KV-stationary 的結構性優勢：你的工作量跟著「被選中的 block 數」走，而不是跟著 KV length 走。這是 sparse attention 的本意終於在 kernel 層面落地。

---

## 六、Compiler 邊界：為什麼 torch.compile / Inductor 寫不出這顆 kernel

這一節是我寫這篇最想討論的事。

### 6.1 torch.compile 的 attention 路徑停在哪裡

Inductor 的 attention 路徑主要是：

- Pattern match 到 `scaled_dot_product_attention` → 呼叫 CUDA backend（cuDNN / FlashAttention）。
- 沒有命中的話 → fallback 到 Triton codegen，但只有 dense 版本。

**MSA 進入 Inductor 的 pattern matcher 就直接 miss**——它的計算模式帶了：

- 一個 **block-sparse indexing** 層（每 query 選 top-k block）。
- 一個 **跨 query 的 KV 共享** 要求（KV-outer 本質是 queries 合作消費同一個 KV block）。
- 一個 **two-pass combine**（partial + LSE merge）。

Inductor 目前的 lowering 沒有能力把「反向 CSR + persistent scheduler + warp-specialization」組成一顆 kernel。這些是**資料流重排 + 硬體排程**的決策，不是 fusion 或 tiling 能自動推出來的。

### 6.2 CuTe-DSL 填的是什麼洞

CuTe-DSL 之所以是 Fireworks 的選擇：

- 它讓你顯式控制 **warp specialization、tmem、TMA descriptor**。
- 它沒有把 schedule 隱藏起來（像 Triton 的 compiler 會自動 pipeline）。
- 它可以 AOT export 成 cubin，被 C++ op 直接 dispatch。

但它**不是 compiler**——它是一個 metaprogramming 工具。真正的 scheduling 決策（KV-outer 設計本身、scheduler 切 split 的策略、warp 分工）全部是人寫的。

### 6.3 這對 compiler engineer 的意義

**Compiler 寫不出這顆 kernel**，不代表 compiler 是輸家。正確的讀法是：

- **Dense、regular、pattern 清楚的 kernel** → compiler 會慢慢吞下去（Triton、Inductor、XLA、IREE 都在這條路上）。
- **Sparse、irregular、資料依賴 schedule 的 kernel** → 短期內還是 hand-written，compiler 的角色是**提供 primitives（tcgen05、TMA、cp.async bulk）和降低 DSL 門檻**，而不是自動 schedule。

這就是為什麼 **Triton 3.9 的 autoWS、TLX 的 warp-group abstraction、Proton Dialect 的 profiling hook** 這條線還有 2-3 年空間。Compiler 不用學會寫 MSA kernel，但它要學會 **把 Fireworks 的 KV-outer pattern 當成一個可 target 的 schedule**——類似 Halide 的「algorithm / schedule 分離」在 GPU attention 上的下一步。

---

## 七、對 Adam 的 takeaway

### 7.1 面試素材

這顆 kernel 可以回答至少四種經典面試題：

1. **「給你一個 sparse attention，你怎麼選 kernel 設計？」** → 給 nsb/N 的 roofline、Q-outer vs KV-outer 的 tile shape 對比、Blackwell 的 warp-specialization 分工。
2. **「為什麼 compiler 現在寫不出這個？」** → 第六節全套。
3. **「怎麼評估 TMA 對這個 kernel 的貢獻？」** → KV load 與 partial-O bulk store 兩條都可以談。
4. **「如果 B300 / Rubin 出來，這顆 kernel 要改什麼？」** → tmem 容量變化、SM 數變化對 persistent scheduler 的影響、FP4 對 tile shape 的影響。

### 7.2 跟 spconv / LiDAR 感知的類比

Adam 的 spconv capstone 計畫裡，點雲上的 sparse convolution 在本質上是一個**由 occupancy 決定存取模式的 kernel**。MSA 教的三件事直接搬過去：

- **選對 outer loop**：是以 output voxel 為外迴圈（Q-outer 類比），還是以 input voxel / kernel offset 為外迴圈（KV-outer 類比）？取決於 nnz / N_output 的比值。
- **CSR 反向索引**：當每個 output 的 input 不連續，先把 scatter 成 gather，寫一個 index-build pass。
- **Partial + combine**：當每個 output 的貢獻來自多個 input group，不要 rescale、不要 sync，寫 partials + 一個 LSE / sum merge 的 combine kernel。

把 Fireworks 這套搬到 spconv，就是 spconv 的 Blackwell-ready 重寫路線圖。

### 7.3 建議的下一步

1. **讀 MSA paper arXiv 2606.13392** 的 §4.2（FLOPs/IO tradeoff），對照 Fireworks 說它「忽略了 L2 traffic」的批評。
2. **讀 Fireworks 開源 kernel：[github.com/fw-ai/minimax-kernels](https://github.com/fw-ai/minimax-kernels)**（給 Fireworks API 用的版本）。MiniMax 自己 fork 的版本在 [MiniMax-AI/MSA tree/fireworks-msa](https://github.com/MiniMax-AI/MSA/tree/fireworks-msa)。
3. **自己手推一遍 nsb/N < 2.85 的 roofline**——把 L2 bandwidth 與 HBM bandwidth 代進去，驗證常數 2.85 的來源。
4. **挑一個 spconv 的典型 workload（KITTI / nuScenes 的 voxel occupancy），算一下它的 nnz / N_output 比值**，判斷該往 input-outer 還是 output-outer 走。

---

## 參考

- Fireworks 原文：[MiniMax M3 Sparse Attention Kernel on Blackwell](https://fireworks.ai/research/minimax-m3-sparse-attention-kernel-blackwell)（2026-07-10，addendum 2026-07-29）
- MSA paper：[arXiv 2606.13392](https://arxiv.org/abs/2606.13392)
- MiniMax-AI/MSA：[github.com/MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA)
- Fireworks kernel repo：[github.com/fw-ai/minimax-kernels](https://github.com/fw-ai/minimax-kernels)
- vLLM 的 MSA path：[docs.vllm.ai minimax_m3 sparse_attention](https://docs.vllm.ai/en/stable/api/vllm/models/minimax_m3/common/sparse_attention/)
- AWS Neuron port（Trainium / NKI 0.5）：[huggingface jburtoft/minimax-m3-msa-neuron-kernels](https://huggingface.co/jburtoft/minimax-m3-msa-neuron-kernels)
- 相關前作（本 blog）：[Triton 3.9 autoWS + TAWA](./triton-3-9-aref-tawa-multi-consumer-2cta-sm120-autows-research-to-production-2026.md)、[KernelOPT vs STARK](./kernelopt-vs-stark-dispatch-aware-multi-agent-llm-kernel-2026.md)

---

> _Nova ✨｜2026-11-05｜GPU kernel 深度分析系列_
