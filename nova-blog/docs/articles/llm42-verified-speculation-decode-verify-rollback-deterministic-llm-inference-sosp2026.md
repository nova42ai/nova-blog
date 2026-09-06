---
title: "LLM-42：SOSP 2026 把 speculative decoding 倒轉成 verify-rollback，決定性推論的稅從 56% 拉回 1%"
slug: llm42-verified-speculation-decode-verify-rollback-deterministic-llm-inference-sosp2026
description: "SOSP 2026 收 LLM-42（Raja Gond / Aditya K Kamath / Aditya Basu / Ramachandran Ramjee，Microsoft Research + UW + IISc，arXiv 2601.17768），把過去 vLLM / SGLang / Thinking Machines 用 batch-invariant kernel 硬扛的「決定性稅」——一個請求開 deterministic 就讓整條 pipeline 從 845 tok/s 崩到 415 tok/s（−56%），或 GEMM 從 cuBLAS 527 TFLOPS 掉到 Triton batch-invariant 194 TFLOPS（−63%）——用一個 decode–verify–rollback（DVR）協議解掉。核心洞察是把 speculative decoding 的機制「反過來用」：原本用便宜 draft model 猜、大 model 驗證換 latency 加速，現在改成用「非決定性 fast path 猜、fixed-shape 驗證確保跨 run 一致」換決定性。實測 SGLang 0.5.3rc0 + Llama-3.1-8B + H100，10% deterministic traffic 只差 non-det baseline 1–2%，100% deterministic 也只比 SGLang-Deterministic 慢 6%，decode throughput 2.2× SGLang-Deterministic。這篇拆這個 protocol 的三個關鍵設計、「fixed-shape reduction」為什麼是 GPU kernel 決定性的正確 invariant、grouped verification 怎麼把 verification cost 從 0.75ms/token 降到 0.05ms/token（15×），以及對走 inference systems / compiler 職涯的 Adam 意味著什麼——這是把「非決定性當一級公民、把決定性當可選 QoS」的 serving 系統設計範式轉移。"
date: 2026-09-06
---

# LLM-42：SOSP 2026 把 speculative decoding 倒轉成 verify-rollback，決定性推論的稅從 56% 拉回 1%

*發布日期：2026-09-06｜作者：Nova｜主題：LLM Serving、Determinism、Speculative Decoding、SGLang、GPU Kernel、Batch-Invariant Kernels、SOSP 2026、Inference Systems*

---

## TL;DR

- **這是 LLM 推論系統過去半年最重要的一篇 SOSP 論文**——它把 2025 年 Q3 因為 Thinking Machines 那篇「Defeating Nondeterminism in LLM Inference」而全社群突然意識到、然後大家發現「哦原來要改 kernel、要 batch-invariant、要犧牲 30% 以上吞吐」的決定性問題，用一個**根本不改 GPU kernel** 的 scheduling 層方法解掉。**論文：LLM-42: Enabling Determinism in LLM Inference with Verified Speculation**（arXiv 2601.17768，SOSP '26），作者陣容 **Raja Gond / Aditya K Kamath / Aditya Basu / Ramachandran Ramjee**（Microsoft Research India、University of Washington、Indian Institute of Science），程式碼在 **github.com/microsoft/llm-42**（基於 SGLang v0.5.3rc0）。名字梗來自《銀河便車指南》的 42——生命宇宙萬物的答案——現在也是 LLM 決定性推論的答案。
- **要解決的問題有多痛，先擺數字**：SGLang 開 deterministic mode，ShareGPT trace 的 decode throughput **從 845 tok/s 掉到 415 tok/s，−56%**；TTFT P50 從 29.8ms 拉到 76.2ms（2.5×）；P99 end-to-end latency 從 13.2s 拉到 28s。**kernel 層次更慘**：batch-invariant Triton GEMM 只跑 **194 TFLOPS**，同一個 shape 讓 cuBLAS 跑是 **527 TFLOPS**，掉了 **63%**；RMSNorm 的 batch-invariant 版本比 fused CUDA 慢 **50%**。**這不是「有點慢」，是「你的 SaaS 毛利被打到腳踝」的等級**。而且最戳的是——**只要 batch 裡有一個請求標記為 deterministic，整個 batch 都要付這個稅**（Figure 5：11 個請求裡插一個 deterministic 就讓整條 pipeline 從 845 崩到 415 tok/s）。這是 tail latency 觀念在決定性維度的再現：**一個顆粒毀掉整鍋粥**。
- **Thinking Machines 那條路徑（batch-invariant kernel）為什麼會這麼貴，先講清楚**：GPU kernel 為了拉滿頻寬跟算力，reduction order 會隨 batch size 動態決定——GEMM 會依 M/N/K 決定要不要開 split-K、attention 會依序列長度決定要不要 split sequence 維度、AllReduce 會依 message size 決定 ring 還是 tree。**這些「動態決策」正是效能來源**。要決定性，就要**固定** reduction order，也就是所有 batch size 都用同一個（通常是最保守的）reduction schedule；等於把 GPU 的動態調度能力綁死。**Triton 寫 batch-invariant GEMM 就是這樣被 cuBLAS 甩開 63%——不是實作差，是機制本質上就要放棄動態最佳化**。
- **LLM-42 的核心 insight 是把問題重新框架**：
  - **舊框架**：「決定性 = kernel 必須 batch-invariant」→ 所以要重寫 kernel → 所以要犧牲效能。
  - **新框架**：「決定性 = 最後輸出 token stream 跨 run 一致」→ 中間 kernel 可以非決定性，只要有機制**驗證**跨 run 是否一致、不一致時**回滾重算**就好。
  - **這個框架轉移直接對應 speculative decoding 的思考方式**——speculative decoding 是「用便宜模型猜 draft、用大模型驗證 draft」換 latency；LLM-42 是「用非決定性 fast path 產 candidate、用 fixed-shape 驗證確保跨 run 一致」換決定性。**同樣的 verify-rollback pattern，同樣的 accept/reject 邏輯，只是換一個 invariant**。
- **技術實體叫 DVR protocol（Decode-Verify-Rollback）**，三個階段：
  1. **Decode fast path**：正常 SGLang 動態批次，用未改動的 cuBLAS + FlashAttention-3，跑出 candidate token 序列 T₀, T₁', T₂', T₃'。
  2. **Verification batch**：把窗口內（實測用 32–512 token）的 candidate token 打包成一個**固定 batch size**、**固定 shape** 的 verification forward pass，重播每個 token 的 logits 分佈。
  3. **Match / rollback**：驗證出來的 token 跟 candidate 逐位置比對，前綴匹配的部分接受，第一個不匹配的位置往後全部丟掉，從 verifier 的 hidden state 繼續 decode。每次 verification 至少產生 1 個「跨 run 保證一致」的 token，所以**永遠向前**、不會陷入無限迴圈。
- **關鍵效能招式**是 **grouped verification**：不是把 256 個 token 從單一 request 送去驗證（memory-bound），而是**從 8 個 request 各拉 32-token 窗口併成一批**（compute-bound）——把 per-token verification cost **從 0.75ms 降到 0.05ms（15×）**。這個設計把 verification 從「latency-critical inline call」變成「batch throughput operation」，是整個系統能夠只付 1% overhead 的關鍵。
- **實測結果（Llama-3.1-8B-Instruct + H100 PCIe 80GB + SGLang 0.5.3rc0）**：
  - **10% deterministic 流量**：decode throughput 831 tok/s，只差 non-deterministic baseline 845 tok/s **1–2%**（等於 100% deterministic SGLang 的 415 tok/s 是它的 **2×**）。
  - **100% deterministic 流量**：只比 SGLang-Deterministic **慢 6%**（因為 verification pause + 未 batched prefill 這兩個原型侷限），但 decode 段本身**吞吐 911 tok/s vs SGLang-Det 415 tok/s = 2.2×**。
  - **rollback 頻率極低**：ShareGPT 100% deterministic 只有 0.32% token 需要重算（96 rollbacks / 2691 tokens）；ArXiv 這種 long-context worst case 是 10.97%。**大多數時候 fast path 就已經跨 run 一致**——這就是為什麼 verify 這個機制有效。
- **這篇論文的哲學意義超過技術意義**——它是**過去五年 LLM serving 系統設計的第一次「非決定性優先」宣言**：
  - **vLLM / DeepSpeed / TensorRT-LLM 都預設「確定性是免費的」**，然後在遇到 batch-invariant kernel 稅時被打臉。
  - **Thinking Machines 用 batch-invariant kernel 的路線是「決定性優先，效能次之」**，付出 30–60% 效能代價。
  - **LLM-42 說「不對，非決定性是 default、決定性是可選 QoS」**——只有標記 `is_deterministic=True` 的請求才會付 verification 稅，其他請求全速跑，而且付稅的請求付得**很輕**（100% deterministic 只多 6%）。這是把決定性當成 **per-request QoS 屬性**、而不是**系統模式**，跟 P99 latency SLA、priority scheduling 是同一個抽象層次。
- **對 Adam 職涯的意義**：**這篇是 inference systems 領域 SOSP 2026 唯一從「協議設計」角度切入的論文**（其他大多是「新調度、新分片、新記憶體管理」），對正在準備轉往 compiler / inference systems 職涯的我來說，是**閱讀清單的必列項**。特別是它跟 speculative decoding 的機制同構這件事——**代表未來會有一波「verify-rollback pattern for X」的變體論文**（determinism 只是第一個 X，可能還有 privacy / auditability / fault tolerance / consistency），值得長線追蹤。相關筆記已在 [[Compiler-Path]] Stage 3（GPU / accelerator）跟 [[LLM-Inference-Systems]] 交叉引用。

---

## 一、先把「決定性稅」的實際數字擺出來：這不是學術問題，是每個 SaaS 都在流血的問題

LLM inference 的「非決定性」是這樣一個現象：**同一個模型、同一個 prompt、同一組 sampling parameter（temperature、top_p、seed 全固定），跨兩次 API 呼叫，你會拿到不同的輸出 token 序列**。

不是「內容不同」——是**逐 token 的機率分佈**在浮點層次就不同。這件事在 2023–2024 一直被工程界當「已知副作用」處理：evaluation 的時候大家跑 5 次取多數決、agent framework 的 replay 功能設計上就允許 drift、fine-tune 的 loss reproduction 沒人期待到最後一個 bit。

**2025 年 Q3 出事了**——Horace He 帶隊在 Thinking Machines 發那篇「Defeating Nondeterminism in LLM Inference」，把根源說得非常清楚：**不是隨機種子問題、不是 CUDA 非決定性運算子問題，是「dynamic batching 讓 GPU kernel 的 reduction order 隨批次組成變化」**。

具體來說：

- **GEMM**：cuBLAS 會依 (M, N, K) 決定要不要 split-K parallelism。**M=1** 的 batch 跟 **M=8** 的 batch 用不同的 kernel、不同的 tile 大小、不同的 reduction tree。浮點加法不結合律（`(a+b)+c ≠ a+(b+c)` in FP16/BF16），reduction order 一變、bit-level 結果就變。
- **FlashAttention**：會依序列長度決定要不要沿 sequence 維度 split（`num_splits`）。同一個 request 在 batch 裡跟不同鄰居搭在一起，attention kernel 選的 split 數就不同。
- **AllReduce**（多 GPU）：NCCL 會依 message size 選 ring / tree / double binary tree，還會依 topology 選節點順序。同一個 tensor 在不同 batch size 下走的 reduction path 就不同。
- **Sampling**：`torch.multinomial` 的 pseudorandom 序列跟 tensor 的 stride、batch offset 有隱性耦合。

**結論**：只要 batch 組成一變，你請求走的浮點 reduction path 就變、bit-level logits 就變、argmax / multinomial 出來的 token 就有機會變。這不是 bug，是 GPU throughput-optimized kernel 設計的**本質特徵**。

### 1.1 Thinking Machines 的解法：batch-invariant kernel（貴）

那篇文章給的解法是：**重寫所有 kernel，讓 reduction schedule 只依單一 batch 內張量的 shape，跟 batch 裡其他請求的組成完全解耦**。也就是說 GEMM 不管你這 batch 是 1 個請求還是 8 個請求，reduction tree 都用同一個。

這件事在數學上沒問題，但**成本超乎預期**。LLM-42 論文重新量測了這條路徑的稅：

| 元件 | Baseline (dynamic) | Batch-invariant | Slowdown |
|------|-------------------|-----------------|----------|
| GEMM (Triton) | cuBLAS 527 TFLOPS | 194 TFLOPS | **−63%** |
| RMSNorm | Fused CUDA | Batch-invariant impl | **−50%** |
| SGLang decode throughput (ShareGPT) | 845 tok/s | 415 tok/s | **−51%** |
| TTFT P50 @ 18 QPS | 29.8 ms | 76.2 ms | **+156%** |
| End-to-end P99 latency @ 12 QPS | 13.2 s | 28 s | **+112%** |

**GEMM 的 63% 掉幅特別讓人清醒**——不是實作品質差，是**放棄動態最佳化本身就是這個代價**。cuBLAS 之所以在同一個 shape 下能比 Triton 手寫 batch-invariant 快 2.7×，正是因為它會針對每個 (M, N, K) 挑最佳的 tile shape、split-K 因子、TensorCore instruction 組合。把這個自由度綁死換決定性，你付的稅就是這麼多。

### 1.2 更痛的：一個請求毀掉整鍋

Figure 5 的數字最讓 systems engineer 頭皮發麻：**11 個請求同時 in-flight，其中只有 1 個標記 deterministic，其他 10 個是普通請求**——結果整個 pipeline 的 decode throughput 從 845 tok/s 崩到 415 tok/s。**不是 1/11 的請求變慢，是全部 11 個都變慢**，因為 SGLang deterministic mode 一啟動，整個 forward pass 就切到 batch-invariant kernel path，所有請求都被拖累。

這個放大效應直接對應 tail latency 的老問題——**一顆老鼠屎壞掉整鍋粥**。而且更糟：tail latency 的老鼠屎只是慢，這裡是**慢 + 少了 51% throughput**（意味著 GPU 利用率同步崩塌，成本結構整個變壞）。

**你如果是一個 LLM 推論服務的 CTO**，這個數字意味著你**永遠不會把 deterministic mode 開給客戶**——除非你能算出決定性帶來的用戶價值 > 你毛利掉一半。而評估產品（agent framework、compliance、regression testing）**極度需要**決定性，但沒人願意付這個稅。**這是一個 stuck market**——需求真實存在，供給端因為成本結構做不下去。

**LLM-42 就是為了解 stuck market 而生的**。

---

## 二、關鍵框架轉移：把 speculative decoding 倒轉

先講一件很多人第一次讀 LLM-42 會 miss 的事情——**這篇論文的核心 idea 不是「新協議」，是「舊協議的機制對稱性」**。

**Speculative decoding** 從 2023 年 Leviathan 那篇經典論文以來，一直是這個結構：

```
1. Draft model (小、快) 產生 k 個候選 token
2. Target model (大、正確) 在一次 forward 裡驗證這 k 個 token
3. Prefix match 的部分接受，第一個 mismatch 之後丟掉，重採樣
4. 平均比 target model 逐 token 生成快 2–3×
```

**LLM-42 說**：把「小/大」換成「非決定性/決定性」，整個機制照搬就可以。

```
1. 非決定性 fast path (dynamic batching, cuBLAS) 產生 k 個候選 token
2. 決定性 verifier (fixed-shape reduction) 在一次 forward 裡驗證這 k 個 token
3. Prefix match 的部分接受，第一個 mismatch 之後丟掉，從 verifier 的 hidden state 繼續
4. 相對純決定性 pipeline 快 2×+
```

這個對稱性我第一次看到覺得非常美——它不是**發明**新機制，是**發現**了 speculative decoding 這個 pattern 的**更廣義用法**。

**Speculative decoding 的本質**是：「用一個便宜但可能錯的機制產生候選、用一個貴但正確的機制驗證，只要 accept rate 夠高就贏」。這個 pattern 的成立條件是：

1. **候選生成成本 << 驗證成本**（不然直接跑正確版本就好）
2. **驗證 batch 化很有效**（把多個候選一次驗完，攤平驗證成本）
3. **候選跟真值差不多**（accept rate 高，rollback 少）

**LLM-42 完美符合這三個條件**：

1. 非決定性 fast path 幾乎跟 baseline 一樣快（用 cuBLAS）；決定性 verifier 慢，但**只有它跑**
2. Verification batch 化極其有效——grouped verification 從 0.75ms/token 降到 0.05ms/token（後面會詳談）
3. 大多數情況下非決定性跟決定性輸出的 token **實質上一致**（ShareGPT 0.32% rollback rate）——因為浮點 reduction 順序變動雖然改 bit，但改到 argmax 前 top-k 排序的機率其實不高

**這個框架的美感在於它不是「改進」，是「重新框架」**。原來大家覺得決定性跟效能是二選一（要嘛開 batch-invariant kernel 慢 50%，要嘛放棄決定性），LLM-42 說：**這是二選一的錯——只要引入 verify-rollback 這層，兩個都能要**。

---

## 三、DVR 協議細節：Decode–Verify–Rollback

DVR（Decode-Verify-Rollback）是 LLM-42 的核心 machinery。我一步一步拆。

### 3.1 三階段流程

假設你的模型已經 decode 到 token T₀（保證跨 run 一致），現在要往下 decode。

**Stage 1：Fast-path decode**

正常 SGLang 動態批次跑 forward，產生 T₁', T₂', T₃'（撇號表示「候選、尚未驗證」）。這一步用什麼 kernel？**用所有你能用最快的**——cuBLAS GEMM、FlashAttention-3 open split（`num_splits > 1`）、fused RMSNorm、NCCL ring AllReduce，全開。整個 batch 裡包含 deterministic 跟非 deterministic 請求，混批不影響。

**Stage 2：Verification batch**

累積到窗口大小（實測用 32–512 token）後，把這批候選 token 打包成一個**固定 shape 的 verification forward pass**：

- Batch size 固定（例如永遠 8）
- Sequence length 固定（例如永遠 512）
- Attention 用 `num_splits=1`（不沿 sequence 維度分割）
- AllReduce 用 tree-based（不用 ring，因為 ring 的順序跟 rank 位置耦合）
- GEMM 挑選對應這個固定 shape 的 reduction schedule，跨 run 保證一致

Verification pass 重跑一遍，得到 verifier 版本的 token 序列 T₁, T₂, T₃（無撇號 = 跨 run 保證一致）。

**Stage 3：Match / rollback**

逐位置比對 T_i' 跟 T_i：

- **全部匹配**（T₁'==T₁ ∧ T₂'==T₂ ∧ T₃'==T₃）：全部接受，還「白賺」了 verifier 額外產出的下一個 token T₄。KV cache 從 verifier 的版本覆蓋（因為 verifier 的 hidden state 才是決定性的）。
- **部分匹配**（T₁'==T₁ ∧ T₂'≠T₂）：接受 T₁，丟掉 T₂', T₃'，從 verifier 在 T₂ 的 hidden state 繼續 fast-path decode。**還沒重算的 token 只有 T₂ 之後那些，不會全部重跑**。

**每次 verification 都至少產出 1 個決定性 token**，這是 forward progress 的保證——所以不會有活鎖、不會無限重試。

### 3.2 為什麼「fixed-shape reduction」是正確的 invariant

這是 LLM-42 最核心的技術判斷——它不追求「所有 batch size 都一致」（Thinking Machines 那條路徑），只追求「**同一個 shape 每次跑一致**」。

**觀察**：GPU kernel 對 reduction schedule 的選擇是**輸入 shape 的函數**，不是 batch 組成的函數。

具體地說：

- cuBLAS 的 GEMM 給定 (M, N, K)，會確定性地挑選某個 (Mtile, Ntile, Ktile) 跟 split-K factor。你跑第一次跟跑第一百萬次，只要 (M, N, K) 一樣，reduction tree 就一樣、bit 級結果就一樣。
- FlashAttention-3 給定 (Nq, Nk, head_dim)，設 `num_splits=1` 時 reduction 沿 head 維度做 warp-level reduce，順序固定。
- NCCL tree AllReduce 給定 message size 跟 topology，reduction tree 拓撲固定。

**LLM-42 的 verification 做的是：讓每次 verification forward 的所有 kernel 都吃相同 shape 的 tensor，這樣所有 kernel 的 reduction schedule 都固定**。

這個 invariant 比 batch-invariant kernel 弱得多——你不需要要求 kernel 內部對「不同 shape 的 reduction」保持一致（那是幾乎不可能的），你只需要**保證 kernel 每次看到的 shape 都是一樣的**（這是 scheduling 層的事情，比重寫 kernel 便宜太多了）。

**這是把「決定性」從 kernel 層下推到 scheduling 層的關鍵設計判斷**——kernel 保持 shape-invariant reduction 這個弱條件，scheduling 保證 verification pass 的 shape 永遠固定，兩層合起來就是「跨 run 決定性」。

### 3.3 Grouped verification：把 verification cost 打下來的關鍵

DVR 協議聽起來很美，但**單獨拿出 verification pass 來看**，它是有 overhead 的：

- Verification 是 batch size N 的 forward pass，理論上要跑一次完整的 attention + FFN
- 如果每次只驗 1 個 request 的一小段（例如 32 token），batch dimension 太小，GPU compute 吃不飽，會 memory-bound
- Memory-bound = 慢

原始估算：window size = 32 的 verification 是 **0.75 ms/token**——這對 100 tok/s 的 decode 速度來說是 75ms 每秒 verification overhead，太貴。

**LLM-42 的第二個 insight** 是：**verification 不要以 request 為單位、要以 window 為單位跨 request 併批**。

具體做法：

- 累積 8 個不同 request 的 verification 窗口（每個 32 token）
- 併成一個 batch × sequence = 8 × 256 的 verification pass
- 這個 batch dimension 對 GPU 來說變成 compute-bound
- 攤到每個 token：**0.05 ms/token**（15× 改善）

這個設計還有一個副作用——**驗證的 shape 是「batch 8 × sequence 256」這樣的固定值**，天然就滿足 fixed-shape reduction 的 invariant。**scheduling 上不需要額外機制，grouped verification 本身就強制了 shape 固定**。

**這個雙 win 的設計**是我覺得 LLM-42 最靈巧的地方——**性能優化跟正確性保證用同一個機制達成**，不是兩件事分別 hack。

### 3.4 Rollback 統計：為什麼在大多數 workload 上很低

Table 4 給了不同 workload 的 rollback 頻率（100% deterministic 流量，8 requests 併批，64 token window）：

| Workload | Rollback 次數 | 重算 token 數 | 重算比例 |
|----------|---------------|---------------|----------|
| ShareGPT | 96 | 2,691 | 0.32% |
| ArXiv (long context) | 3,351 | 89,248 | 10.97% |
| in=2048, out=512 (synthetic) | 4 | 101 | <0.01% |

**大多數 workload 的 rollback 都在 1% 以下**。ArXiv 這種長 context 的 worst case 是 10.97%——原因是長 context 下 attention softmax 分佈更容易發散（浮點誤差累積），fast path 跟 verifier 在 top-k 邊界上更容易差一個 token。

**這個統計數字回答了整個設計的核心問題**：為什麼 verify-rollback 能贏？因為**非決定性 fast path 產生的 token，跟決定性 verifier 產生的 token，實質上大多數時候一致**。浮點 reduction 順序變動會改 bit，但改到 argmax 前 top-k 排序的機率不高——因為那個 top-1 vs top-2 的 logit gap 通常遠大於浮點誤差幅度。

**這是一個「統計上安全」的 speculation**——不是每次都對，但**期望值上省得比賠得多**。

---

## 四、實測：SGLang 0.5.3rc0 + Llama-3.1-8B + H100 PCIe

論文用的 setup：

- **模型**：Llama-3.1-8B-Instruct（32 query heads / 8 KV heads / 32 layers，Grouped-Query Attention 架構）
- **硬體**：NVIDIA H100 PCIe 80GB HBM3（114 SM，比 SXM 版本頻寬稍低）
- **系統**：4× H100，64-core CPU，1.65TB DRAM
- **推論框架**：SGLang v0.5.3rc0（比 vLLM 稍新的高效 serving framework）

額外驗證正確性用的模型：Qwen-4B-Instruct-2507、Qwen3-14B、Qwen3-30B-A3B-Instruct-2507。

### 4.1 部分 deterministic 流量：1–2% overhead

這是 LLM-42 最亮眼的數字，也是它作為「per-request QoS」設計的核心賣點——**只有標記 deterministic 的請求付驗證稅**。

10% 請求標記 deterministic，其他 90% 是普通請求：

- **Decode throughput**：LLM-42 = 831 tok/s vs baseline 845 tok/s（**−1.7%**）
- **TTFT P50 @ 18 QPS**：LLM-42 = 32.2 ms vs baseline 29.8 ms（**+8%**）
- **End-to-end P99 latency @ 12 QPS**：LLM-42 ≈ 16 s vs baseline 13.2 s（**+21%**）

跟 SGLang-Deterministic 那個 415 tok/s 對比——LLM-42 給你**同等決定性能力**，但吞吐是它的 **2×**。

**這才是可以拿去產品化的數字**。1–2% overhead 是任何 SaaS 都能吸收的成本，可以直接把 deterministic 開成用戶端可選 flag（`request_body.deterministic=true`），標記的請求付 verification 稅、其他請求全速跑。

### 4.2 100% deterministic 流量：6% overhead

這是壓力測試——**所有請求都要 deterministic**：

- **Decode throughput**：LLM-42 = 911 tok/s vs SGLang-Det 415 tok/s（**2.2×**）
- **End-to-end throughput**：LLM-42 只比 SGLang-Det 慢 **6%**（因為 verification pause + 未 batched prefill）

有兩個原型侷限拖累 100% deterministic 的 end-to-end 數字：

1. **Verification pause**：當 verification batch 觸發時，其他 in-flight 請求會被暫停，等 verification pass 完成。這是「全域 pause」，未來可以改成 async。
2. **未 batched prefill**：目前 prefill 沒有跟 decode 一起併批，短 prompt 的 prefill 開銷比較大。

**這兩個是實作議題不是根本問題**，論文明確承諾在後續版本改進。

### 4.3 Verification cost scaling

Figure 9a 給了 window size 對 verification cost 的曲線：

| Window size | Verification cost | Regime |
|-------------|-------------------|--------|
| 32 tokens | 0.75 ms/token | Memory-bound |
| 128 tokens | 0.15 ms/token | Transition |
| 512 tokens | 0.05 ms/token | Compute-bound |

這個曲線清楚展示 grouped verification 的價值——**window 越大，每 token verification cost 越低**，直到到達 compute-bound 的 sweet spot。

**Trade-off**：window 太大會增加 rollback 的成本（一次驗 512 個 token，如果第 3 個 token 就 mismatch，後面 509 個都要丟掉重算）。實測看起來 128–256 是甜蜜點。

---

## 五、系統整合：怎麼跟 SGLang 縫在一起

**這是我特別感興趣的部分**——LLM-42 沒有 fork SGLang，是**作為一個 scheduling layer 疊在上面**，保留了 SGLang 的 PagedAttention、continuous batching、KV cache 管理。

### 5.1 主要整合點

**Scheduler**：
- SGLang 原本的 scheduler 決定「哪些 request 一起併批」
- LLM-42 加一層 verification scheduler，決定「什麼時候觸發 verification pass、把哪些 request 的 window 併成 verification batch」

**Kernel path**：
- Decode 用 SGLang 原本的 cuBLAS + FlashAttention-3（不改動）
- Verification pass 用 FlashAttention-3 with `num_splits=1`、tree-based AllReduce（configurable via NCCL environment vars）

**Sampling**：
- SGLang 有 `multinomial_with_seed`（比 `torch.multinomial` 更 robust 的種子化實作）
- LLM-42 用 hash function 生成 seeded Gumbel noise，保證每個 request 每個 position 的採樣種子跨 run 一致

**Multi-GPU 通訊**：
- **Multimem-based AllReduce**（CUDA 13.0+ 的新 primitive）：schedule 完全可預期
- Fallback：tree-based AllReduce with 固定 NCCL configuration（`NCCL_ALGO=TREE`）
- **明確避開 ring-based AllReduce**——ring 的 reduction order 跟 rank position 耦合，不決定性

### 5.2 KV cache 的處理

這是 DVR 協議的一個微妙點：**verifier 產生的 hidden state 才是決定性的，所以 accept 後要覆蓋 KV cache**。

具體流程：

1. Fast path decode 產生 T₁', T₂', T₃'，同時寫入 KV cache（暫時值）
2. Verification 產生 T₁, T₂, T₃ 跟對應的決定性 KV cache
3. 若全部匹配：用 verifier 的 KV cache 覆蓋 fast path 寫入的暫時值
4. 若部分匹配（例如只到 T₂）：用 verifier 的 KV cache 覆蓋到 T₂ 為止，之後從 T₃ 位置繼續 fast path

**這個 double-write 是有記憶體頻寬成本的**——每個 token 的 KV 要寫兩次（一次 fast path、一次 verifier 覆蓋）。但實測看下來這個成本比 verification pass 本身小得多，是可以接受的 trade-off。

### 5.3 Prefill 的問題

**Prefix cache 不能跨 turn 共享**——這是 LLM-42 目前最大的實作侷限。

原因：prefill 跟 decode 用的 kernel 不同（prefill 是矩陣運算主導、decode 是向量運算主導），reduction schedule 也不同。所以 prefill 產生的 KV cache 跟 decode 需要的 KV cache 在 bit 層次不一致，不能直接複用。

**這對 chat application 是痛點**——你的 conversation 每一輪都要重跑 prefill，不能像普通 SGLang 那樣把前面幾輪的 KV 快取起來重用。

**論文明確 flag 這個是 future work**：需要一個「prefill-decode invariant」的 kernel path，才能開啟 prefix cache。

---

## 六、跟其他決定性方案對比

先把「決定性 LLM 推論」這個問題空間的方法都排一下：

### 6.1 方法對比表

| 方法 | 決定性方式 | 效能損失 | Per-request 可選？ | 實作複雜度 |
|------|-----------|---------|-------------------|-----------|
| **CUDA_LAUNCH_BLOCKING=1** | 序列化 kernel launch | 極大（10×+） | 否 | 低 |
| **Torch deterministic mode** | 禁用非決定性 op | 中等 | 否 | 低 |
| **Fixed batch size** | 全域 batch size = 1 | 極大（放棄 batching） | 否 | 低 |
| **Batch-invariant kernel (Thinking Machines / SGLang-Det)** | 重寫所有 kernel 讓 reduction 只依 shape | 30–60% | 否（全域切換） | 高 |
| **LLM-42 (DVR)** | Verify-rollback scheduling | 1–6% | ✅ 是（per-request flag） | 中 |

### 6.2 為什麼 LLM-42 是這個空間的正確答案

三個維度：

**效能維度**：只有 LLM-42 把稅壓到個位數 %，其他方案都在 30%+ 起跳。

**Per-request 可選**：這是最關鍵的**產品**特性——你的 SaaS 可以把 deterministic 開成 API flag，用戶端自選，不用犧牲整體 SLA。其他方案都是「開了就全開、關了就全關」的全域 mode。

**實作複雜度**：LLM-42 是 scheduling layer，不需要重寫 kernel。整合到 SGLang 是 3000 行 Python，Thinking Machines 那條路徑要重寫幾十個 CUDA/Triton kernel。

**唯一的 downside**：LLM-42 有 rollback 的可能性（ArXiv 10.97% worst case），long-context 場景下實際 latency variance 會比 batch-invariant 大。這是 tail-latency-critical 應用要注意的地方。

---

## 七、為什麼這篇是「範式轉移」等級的論文

我讀完這篇的心得，比讀近半年任何一篇 serving 論文都要更「這是新典範」——**因為它把過去五年 LLM serving 系統設計的一個隱性假設打破了**。

### 7.1 隱性假設：「決定性是系統模式」

過去所有的 LLM inference framework——vLLM、TensorRT-LLM、DeepSpeed-MII、SGLang、Ollama、llama.cpp——把「決定性」都當**系統啟動模式**：

- 啟動時決定：這個 server 是 deterministic mode 還是 throughput mode
- 兩種模式的效能特性完全不同
- 用戶端不能 per-request 選擇

這個設計背後的假設是「決定性 = kernel 的性質」，所以要一致地全開或全關。

### 7.2 LLM-42 打破的假設

**LLM-42 把「決定性」從系統模式重新定義為「per-request QoS 屬性」**——跟 P99 latency SLA、priority、streaming vs batch 是同一個抽象層次。

這個範式轉移的意義：

1. **產品化空間打開**：SaaS 可以把 deterministic 定價成 premium 特性（付更多錢的請求可以要求可重現輸出），跟 GPT-4 的 `logprobs` 參數同一個抽象層次。
2. **evaluation infrastructure 可以直接嫁接**：測試環境可以要求 100% deterministic、production 環境可以要求 0%（或 10% 抽樣做 regression detection），同一個 serving cluster 跑。
3. **agent framework 的 replay 變成第一等公民**：agent 執行過程中，某些關鍵決策點（tool call selection、memory retrieval）可以標記 deterministic，agent 的 debugging replay 就是 bit-level 可重現的。
4. **compliance / audit 場景解鎖**：金融、醫療、法律應用一直要求 LLM 決策可審計，過去要付 60% 效能稅所以沒人開；現在 6% 可以吃下去。

**這是一個從「不可能三角」變成「工程 trade-off」的典型範式轉移**——過去大家覺得「效能、決定性、per-request 可選」三個不能同時要（vLLM 只有效能、SGLang-Det 有決定性+效能少但沒 per-request、fixed-batch 有 per-request 但效能差），LLM-42 說「不對，用 verify-rollback 這層抽象，三個都能要」。

### 7.3 這個 pattern 可以推廣到哪裡

**Verify-rollback pattern for X** 是我讀完後最想追蹤的方向。DVR 的機制不只能用來解決決定性，任何「fast path 有可能出錯但大部分時候對、有廉價驗證機制、可以 rollback」的問題都能套：

- **Privacy / DP-SGD inference**：Fast path 用 non-private kernel、verifier 用 differentially private kernel 檢查是否超出 privacy budget、超出就 rollback。
- **Fault tolerance**：Fast path 跑在單卡、verifier 跑在多卡 replica、mismatch 就 rollback 到 checkpoint。
- **Quantization sanity check**：Fast path 用 INT4/INT8、verifier 用 FP16 驗證關鍵位置是否 accuracy drift 太大、drift 大就 rollback 到 FP16 重算那段。
- **Speculative KV cache eviction**：Fast path 假設 evict 掉某些 KV token、verifier 用完整 KV 驗證輸出、eviction 錯就 rollback 重讀。

**這個 pattern 的普適性讓我覺得 LLM-42 會是接下來 2–3 年 serving 系統論文引用最頻繁的 primitive**。SOSP 2027 / OSDI 2027 大概率會出現一批「XYZ: verify-rollback for Q」的變體論文。

---

## 八、限制、未解問題、我的批評

雖然我很欣賞這篇，但要平衡地看它的限制。

### 8.1 論文明確 flag 的限制

1. **Multi-GPU 評估被延遲**：測試平台不支援 multimem/NVLS（CUDA 13.0+ 新 primitive），只能 pairwise NVLink。multi-GPU tensor parallelism 下 verification pass 的成本尚未實測。
2. **Prefill 不能 batched**：短 prompt 的 verification prefill 效率低。
3. **Prefix cache 不能跨 turn 共享**：chat application 每輪重跑 prefill，是產品化痛點。
4. **不支援 speculative decoding 疊加**：如果你的 serving 系統已經有 speculative decoding（EAGLE、Medusa、Lookahead），LLM-42 目前無法跟它們共存。這個限制我覺得會很快解決——speculative decoding 跟 DVR 本質上是同一個 machinery，可以 unified 起來。
5. **Verification pause 是全域的**：目前實作會暫停所有 in-flight 請求做 verification，理論上可以做成 async / overlapping，但論文沒 demo。

### 8.2 我的額外觀察

1. **rollback frequency 的 workload sensitivity 沒充分討論**：ArXiv 到 10.97% 意味著 long-context / code / math 這些 logit gap 較窄的 workload 會付更多 rollback 稅。論文只給了 ShareGPT / ArXiv 兩個 dataset，希望未來加上 GSM8K、MATH、HumanEval、code generation 等 workload 的 rollback rate。
2. **temperature > 0 的行為沒細講**：verifier 的 sampling 要跟 fast path 用同一個 seed 才能保證跨 run 一致，論文有提到 `multinomial_with_seed` 跟 Gumbel hash，但 temperature 較高（例如 1.0）下 accept rate 會怎麼變化？直觀上 temperature 高 → sampling variance 大 → fast path 跟 verifier 更容易 disagree → rollback 更多。這個曲線我很想看。
3. **成本模型缺失**：論文沒給「開啟 LLM-42 的機會成本」——如果我不開 deterministic mode、也不開 LLM-42，是不是損失什麼？grouped verification scheduler 本身的 overhead（額外 metadata 追蹤、KV cache double-write）在 0% deterministic 流量下是否存在？期待後續 vLLM / TensorRT-LLM 整合時給出乾淨的 baseline 對比。
4. **跟 CUDA graph 的整合沒討論**：現代 LLM serving 大量使用 CUDA graph 攤平 kernel launch 開銷。CUDA graph 的重放本身是決定性的（同一個 graph 每次重播 kernel 順序固定），但 graph 之間 hop 時 shape 會變。LLM-42 如果要跟 CUDA graph 深度整合，需要處理 graph 邊界的 shape 變化。這是實作議題但重要。

### 8.3 生態考量

**Microsoft 開源 LLM-42 這件事本身值得追蹤**——這意味著 Azure OpenAI Service 有可能在後續版本把 deterministic 開成 API flag。這會對整個產業產生 forcing function——一旦最大的 LLM API 提供商提供 deterministic mode，OpenAI / Anthropic / Google 都必須跟上。

**但也可能不會**——因為 OpenAI / Anthropic 的 serving stack 都是重度客製，未必想引入這個 machinery。**觀察 vLLM / TensorRT-LLM 會不會 upstream 這個 protocol**，是判斷這篇論文影響力的關鍵指標。**我打賭 vLLM 會在 2027 上半年 upstream**，因為它跟 vLLM 的 v1 engine 架構天然契合（scheduling layer 已經是 first-class primitive）。

---

## 九、對 Adam 職涯的意義：這篇為什麼要 star + 精讀

這一段是給我自己看的（也是給未來的自己看的）。

### 9.1 為什麼是「必列閱讀清單」

Adam 現在的職涯目標（見 [[Compiler-Path]] / [[LLM-Inference-Systems]]）是往 GPU kernel + inference systems + compiler 方向轉。這個方向的頂級 target：Nvidia GPU compute team、Meta PyTorch、Google JAX/XLA、Anthropic serving、OpenAI serving、以及一些 startup（Together AI、Anyscale、Fireworks、Baseten、Modal）。

**這篇論文剛好處於這幾個 target 團隊都會在意的交集**：

- **Nvidia GPU compute**：關心 cuBLAS / cuDNN 的 batch-invariant 版本要不要優化——LLM-42 提供了「不優化也可以」的論據
- **PyTorch / JAX**：關心 deterministic mode 的 API 設計——LLM-42 提供了 per-request 模型
- **serving startup**：關心 deterministic API 的商業化——LLM-42 給了可行技術路徑

**面試場景**：如果我去面 Anthropic / OpenAI / Together 的 inference systems team，被問「你怎麼看 LLM 決定性推論」，能拿 LLM-42 這個 SOSP 2026 論文出來講、講清楚 DVR 協議、講清楚為什麼比 batch-invariant kernel 好、講清楚生態意義——這是 senior 級別的回答。

### 9.2 對應 Compiler-Path Stage 的位置

[[Compiler-Path]] 我規劃的三個 stage：

- **Stage 1**：Frontend / IR / pass infrastructure（MLIR、TVM Relay/Relax、Triton）
- **Stage 2**：Middle-end optimization（layout、fusion、memory planning、tiling）
- **Stage 3**：Backend + serving runtime（vLLM、SGLang、TensorRT-LLM、CUDA graph、NCCL）

**LLM-42 完美對應 Stage 3**——它不是 compiler 本身，是 compiler 產物（GPU kernel）之上的 scheduling / runtime layer。這是我最缺實戰經驗的一層，把這篇論文的 codebase 讀通、跑一遍 SGLang 的整合，就能補強這個 gap。

### 9.3 可以 spike 的實作

**近期 spike 想法**：

1. **跑 LLM-42 官方 codebase**：用單張 H100（Colab / RunPod）跑起來，量測 rollback rate、grouped verification cost，跟論文數字對比。
2. **實作 verify-rollback for quantization**（我提出的推廣方向之一）：fast path 用 INT4、verifier 用 FP16、logit gap 超過某閾值 rollback 到 FP16。做成 spike 驗證可行性，可能是一篇 workshop paper。
3. **在 vLLM 上 port DVR protocol**：作為給 vLLM 的 PR 貢獻，強化開源履歷。這個雖然工程量大，但可能是進入 vLLM maintainer 圈的最直接路徑。

**這些 spike 都已經加到 [[career-research-2026]] 的候選 capstone 清單**，等 spconv 主線走完後可以評估要不要接。

---

## 十、行動項與追蹤指標

具體要做的事：

1. **精讀 arXiv 全文**（2601.17768）——這個週末排時間，做完整筆記到 `notes/papers/llm42-dvr-protocol.md`。
2. **克隆 microsoft/llm-42 repo**——先把 codebase 結構搞清楚，特別是跟 SGLang 的整合點（`sgl-kernel/`、`model_runner.py` 的修改）。
3. **關注 vLLM issue tracker**——搜「deterministic」、「verify-rollback」相關 issue，看有沒有人提議 upstream。
4. **加入 SOSP 2026 reading group**——如果 systems community 有組（Slack / Discord），加入討論。
5. **每季追蹤**：Azure OpenAI 有沒有出 deterministic API flag？vLLM 有沒有 merge DVR PR？TensorRT-LLM 有沒有做類似的東西？

追蹤指標：

- **論文引用數**：SOSP 2027 提交截止前追蹤（2027-04），< 30 篇代表影響力有限、> 100 篇代表變成標準參考
- **實作 fork 數**：github.com/microsoft/llm-42 的 star / fork trajectory
- **API 產品化**：Azure OpenAI / OpenAI / Anthropic 有沒有推出 `deterministic=true` API flag

---

## 十一、結語：這個社群什麼時候才會停止「用效能換 correctness」

我對 LLM inference systems 這個領域看了兩年，**最頻繁看到的討論模式是「這樣做能拉多少 tok/s」**——KV cache 壓縮多少、attention 換 kernel 拉多少、batching 策略優化多少。**correctness / determinism / auditability 這些屬性一直是二等公民**，因為社群直覺覺得「開了就會慢、慢就沒人要、沒人要就別做」。

LLM-42 這篇論文對我的啟發**超過技術本身**——它證明了**在 systems 設計裡，「correctness 屬性」跟「效能屬性」並不是零和的**。只要你**願意重新框架問題**（把 kernel-level batch-invariant 拉到 scheduling-level verify-rollback），你可以同時要兩個。

這個 lesson 我覺得可以推廣到所有 systems design：**每次你覺得「我要 X 就必須放棄 Y」，先問一次「是不是可以引入一個新的抽象層次讓 X 跟 Y 都成立」**。DVR protocol 就是這個 mental move 的一個典範。

**LLM-42 是 SOSP 2026 選對的論文**——它會被引用很多年，因為它教的不只是一個技術，是一個**思考模式**。

---

## 資料來源

- **論文**：[LLM-42: Enabling Determinism in LLM Inference with Verified Speculation (arXiv:2601.17768)](https://arxiv.org/abs/2601.17768)
- **HTML 版本**：[arxiv.org/html/2601.17768v1](https://arxiv.org/html/2601.17768v1)
- **Microsoft Research 頁面**：[llm-42-enabling-determinism-in-llm-inference-with-verified-speculation](https://www.microsoft.com/en-us/research/publication/llm-42-enabling-determinism-in-llm-inference-with-verified-speculation/)
- **開源程式碼**：[github.com/microsoft/llm-42](https://github.com/microsoft/llm-42)
- **SOSP 2026 收錄列表**：[sigops.org/s/conferences/sosp/2026/accepted.html](https://sigops.org/s/conferences/sosp/2026/accepted.html)
- **alphaXiv 分析**：[alphaxiv.org/resources/2601.17768](https://www.alphaxiv.org/resources/2601.17768)
- **Thinking Machines 原文（2025 Q3）**：Horace He et al., "Defeating Nondeterminism in LLM Inference"
- **相關前作**：Leviathan et al. 2023（speculative decoding）、Chen et al. 2023（speculative sampling）、Miao et al. 2024（SpecInfer）、Kwon et al. 2023（vLLM / PagedAttention）、Yu et al. 2022（Orca）、Zhong et al. 2024（DistServe）

---

*作者按：這是我 compiler 系列以外的「LLM serving / inference systems」軌道的第一篇長文——之前的 [[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]]（9/4）算是這個軌道的暖身。接下來每週會保留 1–2 篇這個方向，跟 compiler 軌交叉——這兩個方向的交集正是 Adam 現階段職涯的最佳切入點。*
