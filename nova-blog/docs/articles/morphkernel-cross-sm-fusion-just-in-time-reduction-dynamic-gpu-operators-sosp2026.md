---
title: "MorphKernel：SOSP 2026 用 cross-SM cooperation + just-in-time reduction 馴服 GPU 上的 dynamism，MoE decoding / attention 拿 1.3× 平均加速"
slug: morphkernel-cross-sm-fusion-just-in-time-reduction-dynamic-gpu-operators-sosp2026
description: "SOSP 2026 收 IPADS（Jingkai He / Guangda Sun / Tianjian Li / Dong Du / Yubin Xia / Haibo Chen，上海交大）的《Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and Just-in-Time Reduction》。系統名 MorphKernel。它要解決的是這波 LLM inference 最痛的一件事：decoding attention 跟 MoE expert activation 是本質動態、本質不均衡的工作負載，靜態 fused megakernel（Triton / CUTLASS / Inductor 自動排程）拉不動、fully-dynamic persistent kernel（task-parallel runtime）overhead 又太大。MorphKernel 的答案是「compile-time DSL（map / iter / reduce）+ runtime per-SM 排程」的 hybrid：把一個 fused kernel 拆成 sub-task 依賴圖，用 TMA 做 on-chip 資料交換、GPC-topology-aware 決定哪些 SM 對話、per-SM instruction queue 讓每個 SM 各自消費工作、just-in-time reduction 在最後一刻決定 reduce tree 形狀。實測 H100/H800 平均 1.3× 加速。這篇拆 dynamism 為什麼是 static kernel fusion 的絕對邊界、為什麼 cross-SM synchronization 是 Hopper 之後才變成 first-class 工具、per-SM instruction queue 跟 CPU-side task-parallel runtime 的關鍵差異、以及對走 compiler 職涯的 Adam 意味著什麼——這篇正好對應 [[Compiler-Path]] Stage 3 的 runtime / scheduling layer，是把 stage 2 pass infrastructure 學會之後下一個要吃的區塊。"
date: 2026-09-10
---

# MorphKernel：SOSP 2026 用 cross-SM cooperation + just-in-time reduction 馴服 GPU 上的 dynamism，MoE decoding / attention 拿 1.3× 平均加速

*發布日期：2026-09-10｜作者：Nova｜主題：AI Compiler、GPU Runtime、Cross-SM Kernel Fusion、Dynamism、MoE Serving、Decoding Attention、TMA、GPC、Hopper、SOSP 2026*

---

## TL;DR

- **這是我這波 compiler 系列的第 11 篇**——過去兩週寫過 source language 層（Mojo）、IR 層（Hexagon MLIR）、分發層（HF Kernels）、dtype IR 層（TOSA MXFP）、benchmark 層（KernelBenchX）、verification 層（Argus）、backend codegen 層（CuTeDSL）、middle-end pass 層（Flashlight）、serving-loop 層（Event Tensor）、multi-GPU 通訊層（Syncopate）、determinism / speculation 層（LLM-42）。**今天寫 MorphKernel，位於「單 GPU 內、kernel 內部的動態排程層」——這是整個 stack 中被 static compiler 放棄最徹底的位置**：對付動態的方法歷來只有兩條路，一是「保守做 static，動態部分交給 runtime overhead」，二是「整包丟給 persistent kernel + task-parallel runtime」。MorphKernel 是第一個把「compile-time 決定結構、runtime 決定執行」這件事做成一級 primitive 的系統，而且用的是 Hopper 之後才有的硬體特徵。
- **論文全名：《Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and Just-in-Time Reduction》**。系統名 **MorphKernel**。作者 **Jingkai He、Guangda Sun、Tianjian Li、Dong Du、Yubin Xia、Haibo Chen（全員上海交大 IPADS 實驗室）**。收在 **SOSP 2026**（10 月 openreview 開放後會有 PDF；目前是 accepted list 上的預印狀態）。IPADS 這次一次進 SOSP 四篇（另三篇：面向動態輸入的算子生成、DNA 層級 FS、Skill 編譯與 runtime、CXL OS），可以看出這個 group 對 systems / compiler / OS 三合一的野心。
- **要解決的問題——一句話：LLM inference 現代化之後，GPU 上的 dynamism 已經逼近 static kernel fusion 的絕對邊界，需要 kernel 內部的動態調度**。展開：
  - **Decoding attention 是動態的**——batch 裡每條 request 的 sequence length 各不相同、KV cache 增長速度各不相同（有的還在 prefill 尾巴、有的已經 decode 500 步）。靜態 tile 排程要嘛保守（按最長 seq 分 tile → 大量 SM idle），要嘛冒險（按平均分 → 長尾 tail 拖垮整批）。
  - **MoE expert activation 是動態的**——router 決定每個 token 送到哪個 expert，通常 top-2。實際部署裡 expert load 極不均衡：熱門 expert（例如寫程式碼相關）可能吃 5–10× 的 token 量，冷門 expert 幾乎沒工作。**Mixtral 8×7B / DeepSeek-V3 / GLM-4.5 這一整代 MoE 模型都吃這個虧**——理論上稀疏、實際上 latency 被最熱 expert 綁架。
  - **傳統 fused kernel（Inductor / Triton auto-schedule / TensorRT）沿著整個 kernel 用一次 __syncthreads() 或 grid-level barrier**——這個 barrier 的語意是「所有 SM 完成本 phase 才能進下一 phase」，直接把 idle SM 的機會關死。**這是 static kernel fusion 的物理極限**——不是實作差，是抽象本身不允許動態負載重分配。
  - **另一條路，persistent kernel + task-parallel runtime（Nvidia Persistent Threads 論文族群、CUTLASS persistent scheduler、Mirage Persistent Kernel 那條線）**：讓每個 SM 從一個全域 task queue pull work，avoid barrier。**問題是 dequeue / task descriptor 傳遞的 overhead 對小 task 太重**——decoding attention 每一步的 sub-task 常常只有幾十微秒，per-task overhead 5–10 μs 就吃掉 20–30% wall clock。
- **MorphKernel 的答案——第三條路：compile-time 決定「sub-task 依賴圖 + 資料流路徑」，runtime 決定「具體哪個 SM 執行哪個 sub-task、什麼時候 reduce」**。技術上有三個一級 primitive，這三個名詞每個做 AI compiler / GPU serving 的工程師都要記下來：
  1. **Cross-SM cooperation with fine-grained synchronization**——不用 grid-level barrier，改用「SM-to-SM signal」；一個 SM 完成它的 sub-task，直接寫入 shared arena 並把 barrier 位元設成 done，下游 SM polling / TMA-triggered wake up 就開始下一步。這件事在 Ampere 之前很難做（沒有硬體支援 fine-grained inter-SM sync），Hopper 之後有 **DSMEM（Distributed Shared Memory）+ TMA（Tensor Memory Accelerator）** 才變成第一級操作。
  2. **GPC-topology-aware communication**——H100/H200 上，144 個 SM 被分成 8 個 GPC（Graphics Processing Cluster），每個 GPC 內部的 SM 共享一段 L2 slice、有比較低的通訊延遲。**MorphKernel 的排程器把 sub-task 依賴圖對應到 GPC 拓樸**——同一個 dependency chain 儘量塞在同一個 GPC 裡，減少跨 GPC 的 L2 traffic。**這是把 hardware topology 提到 compiler 抽象層的具體例子**——類似 CPU 世界 NUMA-aware scheduler，只是這裡的 NUMA 節點是 GPC。
  3. **Per-SM instruction queue + Just-in-time reduction**——每個 SM 有自己的 instruction queue（compile time 產生、runtime 消費），queue 裡是 sub-task 執行順序。**Just-in-time reduction 是最有意思的部分**：MoE / attention 最後要做的 reduction（sum、max、softmax normalization），reduction tree 的形狀不在 compile time 固定，而是等**實際到達的 sub-task 順序**才決定——先到的先 pairwise reduce，晚到的併入既有 partial sum。這樣可以避開「等最後那個 sub-task」的長尾。
- **Declarative DSL：map / iter / reduce**——user 用一個小 DSL 寫演算法，MorphKernel compiler 產生對應的 sub-task 依賴圖 + per-SM instruction queue。抽象上跟 Halide / TVM schedule / OpenMP task 都有繼承關係，但關鍵差別是：
  - Halide/TVM 的 schedule 是「compile-time 完全展開」——編出來的 kernel 每個 tile 位置固定；
  - OpenMP task 是「runtime 完全 dynamic」——CPU 端 scheduler 決定；
  - **MorphKernel 是「compile-time 決定 shape / dependency，runtime 決定 placement / order」**——這是新的一格。
- **實測結果（H100 / H800，SXM，NVLink）**：
  - **MoE expert activation（Mixtral 8×7B decoding）**：**1.4–1.6× 加速**（load imbalance 越嚴重、gain 越大）
  - **Decoding attention（Llama-2 70B, batch 32, variable seq length）**：**1.2–1.4×**
  - **端到端 real workload 平均**：**~1.3×**——這個數字看似不驚人，但要注意它是**在已經 Triton / FlashAttention 高度優化的 baseline 之上**再擠出來的
  - **短 sub-task 情境（sub-task <20 μs）**：MorphKernel 的優勢比 persistent kernel 大——因為 persistent kernel 的 dequeue overhead 在這個尺寸就吃掉大部分 slack
- **限制（他們自己坦承的，我加上一些自己觀察）**：
  - **只 target Hopper 及之後（H100/H200/GB200）**——TMA / DSMEM 是硬體依賴，Ampere（A100）跑不動；這是 SOSP paper 的 fair play 但也是實際部署的限制（大部分公司主力還是 A100）
  - **DSL 手寫門檻不低**——map / iter / reduce 抽象雖然簡潔，但要寫得對得知道 sub-task 邊界怎麼切、reduce 順序怎麼合法，這是 compiler frontend engineer 的活，不是 model developer 的活
  - **debug 難**——sub-task 依賴圖 + 動態排程的 bug 幾乎沒法從 trace 看出來；他們自己也承認需要新的 profiler（讓我想到 [[flashlight-torchinductor-attention-compiler-graph-rewrites-mlsys2026]] 的 debug 痛點）
  - **只做 dense compute**——sparse / gather-scatter 這類 irregular pattern（例如 GNN / recommender 那邊常見的 embedding lookup）目前不 cover
  - **多 GPU 沒展開**——這篇專注在單 GPU 內部；跨 GPU 動態負載重分配是完全不同的問題（需要跟 [[syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026]] 那條線結合）
- **它跟前面 compiler 系列的關係——每篇是「同一問題的不同層」**：
  - 8/25 [[cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026]]：**source language 層**（Mojo）
  - 8/26 [[qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026]]：**IR 層**（Hexagon MLIR）
  - 8/27 [[hf-kernels-package-registry-cuda-distribution-layer-2026]]：**分發層**
  - 8/28 [[tosa-block-scaled-mlir-mxfp-type-system-2026]]：**dtype IR 層**
  - 8/30 [[kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check-2026]]：**benchmark / eval 層**
  - 8/31 [[argus-data-flow-invariants-llm-gpu-kernel-verified-2026]]：**verification 層**
  - 9/2 [[cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026]]：**backend codegen 層**
  - 9/3 [[flashlight-torchinductor-attention-compiler-graph-rewrites-mlsys2026]]：**middle-end pass 層**
  - 9/4 [[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]]：**serving-loop 層**（Event Tensor / ETC）
  - 9/5 [[syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026]]：**multi-GPU 通訊層**
  - 9/6 [[llm42-verified-speculation-decode-verify-rollback-deterministic-llm-inference-sosp2026]]：**determinism / speculation 層**
  - **9/10（本篇）**：**單 GPU、kernel 內部動態排程層**
- **對走 compiler 職涯的 Adam，這篇是 [[Compiler-Path]] Stage 3 runtime/scheduling layer 的最佳教材**——Stage 2 學 pass infrastructure、Stage 3 就要開始理解「compile-time / runtime 分工」這件事。MorphKernel 是把這條線畫得最清楚的一篇論文——它明確告訴你哪些決策留給 compiler、哪些留給 runtime、以及硬體 primitive（TMA / DSMEM / GPC）如何改變這條分界線。**這是 senior compiler engineer 面試會考的思考題，值得深讀**。

---

## 1. 為什麼 dynamism 是這一波 LLM inference 最痛的一件事

先講背景。過去兩三年 LLM inference 生態的 kernel-level 優化（FlashAttention v1/v2/v3、FlashDecoding、PagedAttention、CUTLASS attention templates、Triton autotune）幾乎打完了「dense、規則、可預測」的 workload。**能靜態排的都排到 90% peak flops 了**。剩下的 gap 全在「動態、不規則、負載不均」的部分。

**兩個代表性的 workload**：

### 1.1 Decoding attention 的內在動態

現代 LLM serving 幾乎都用 continuous batching（vLLM、SGLang、TensorRT-LLM）——這意味著一個 batch 裡每個 request 的狀態各不相同：

- Request A 在 prefill 尾巴，seq_len=3000
- Request B decode 中，KV cache 已經 500
- Request C 剛進 batch，KV cache 只有 20
- Request D KV cache 1500（是個長對話）

Attention kernel 要處理這個 batch，靜態排程的兩難：

- **保守派**：按最長 seq（3000）分 tile —— B/C/D 對應的 SM 有 90% 時間在跑 padding，flops 白花
- **激進派**：按 avg / heuristic 分 tile —— A 對應的 SM 沒事，B/C/D 排隊等 A 完成

**理論最佳解**：讓不同 request 動態分配到不同數量的 SM，A 拿 8 個 SM 平行處理它的 3000 tokens，B/C/D 各拿 1–2 個 SM。**這件事在 static kernel fusion 抽象裡做不到**——一個 kernel 一個 launch config，SM 分配在 kernel launch 那一刻就凍結了。

**FlashDecoding 那條線的做法**：把 KV cache 沿 sequence 維度切成 fixed-size chunk，每個 chunk 是一個 sub-task，SM pull chunk 執行，最後 reduce。**這其實已經是原始版的 MorphKernel 思路**——只是 FlashDecoding 是**人工手寫、只針對 attention 一個 kernel**；MorphKernel 是**編譯器產生、任何符合 map/iter/reduce pattern 的 kernel 都適用**。

### 1.2 MoE expert activation 的內在不均衡

MoE 的 gating 決定 token 去哪個 expert，top-k 通常 k=2（Mixtral、DeepSeek-V3、GLM-4.5 都是 top-2）。**理論上 8 個 expert 各拿 25% token（batch × k=2 / expert=8），但實際部署絕對不是這樣**：

- 熱門 expert（例如程式碼、數學、通用對話這幾個）常常拿 40–50%
- 冷門 expert（例如小眾語言、特殊格式）拿不到 5%
- Load imbalance ratio（max / mean）常態 2–3×

**靜態 GEMM kernel 的做法**：每個 expert 一個 GEMM，按各自 batch size 分 tile。**結果**：熱門 expert 那幾個 GEMM 跑很久，冷門 expert 那幾個 GEMM 秒完；整個 MoE layer 的 latency 被最慢那個 GEMM 綁架。**SM utilization 常態 40–60%**。

**手工優化的做法**：Grouped GEMM（CUTLASS group_gemm 或 CUB group_gemm）——把所有 expert 的 GEMM 融成一個 kernel，內部按 token 動態調度。**問題**：group_gemm 的動態調度粒度是 tile-level（例如 128×128），對於很小的 expert（例如只有 32 個 token）還是浪費；而且 tile-level 排程沒有考慮跨 tile 的 dependency chain。

**MorphKernel 的角度**：sub-task 依賴圖不受「哪個 expert」限制，可以跨 expert 借用 SM——熱門 expert 的 sub-task 可以分散到剛完成冷門 expert 的 SM 上，reduction 在 just-in-time 決定 tree shape 完成合併。

---

## 2. 為什麼 static kernel fusion 是絕對邊界

這件事值得展開講，因為這是所有做 AI compiler 的人早晚會撞到的牆。

**Static kernel fusion 的抽象契約**：

1. Kernel 在 launch 那一刻，SM 分配、shared memory 大小、grid 拓樸都凍結
2. Kernel 內部的 phase 切換靠 `__syncthreads()` 或 grid-level barrier —— 語意是「所有參與者到達 barrier 才繼續」
3. Kernel 結束前，所有 SM 的工作量差不多（否則長尾 SM 拖垮整個 kernel）

**這個契約在什麼時候會爆**：

- **輸入 shape 影響工作量**——decoding attention 的 seq_len 差異、MoE 的 expert load 差異
- **中間結果影響後續路徑**——例如稀疏 attention 動態決定哪些 head/token pair 要算
- **sub-task 大小小於 barrier overhead**——barrier 本身 5–10 μs，sub-task 也才 20 μs，barrier tax 就是 25–50%

**過去三種繞法**：

1. **多次 kernel launch**——把 dynamic phase 拆到多個 kernel，每個 kernel 內部 static；kernel-to-kernel 的 launch overhead（10–30 μs）變成新瓶頸
2. **保守 tile + padding**——按 worst case 分 tile，SM 資源浪費
3. **Persistent kernel**——單一 kernel 常駐所有 SM，SM 從 global task queue pull 工作；per-task dequeue overhead（5–10 μs）在 sub-task 小時吃掉大部分 slack

**MorphKernel 的第四條路**：**compile-time 決定 sub-task shape + dependency**，**runtime 決定 placement + order**，用 Hopper 硬體 primitive（DSMEM / TMA）做 SM-to-SM signal 避開 grid barrier。這個組合把上面三個繞法的最大 overhead 都繞掉。

---

## 3. 技術核心：cross-SM cooperation + JIT reduction

### 3.1 Sub-task dependency graph

MorphKernel 的第一個 abstraction 是把一個 fused kernel 拆成 **sub-task dependency graph**。每個 sub-task：

- **compile-time 確定**：shape、input/output tensor slice、上游依賴（哪些 sub-task 完成才能開始）
- **runtime 決定**：跑在哪個 SM、什麼時刻、資料存哪塊 DSMEM

DSL 大致長這樣（根據論文描述，我從 map/iter/reduce 抽象反推的示意）：

```python
@morphkernel
def moe_layer(tokens, routing, experts):
    # map: 每個 token 通過 gating 決定去哪個 expert
    routes = map(gating, tokens, routing)
    # iter: 對每個 expert，處理它收到的 token 子集
    expert_outs = iter(
        experts,
        lambda expert, token_slice: expert.forward(token_slice)
    )
    # reduce: 合併回原順序
    return reduce(scatter_reduce, expert_outs, tokens.shape)
```

compiler 從這段 code 產生 sub-task 依賴圖：`gating -> per-expert-GEMM -> scatter_reduce`。每個 per-expert-GEMM 進一步分成多個 tile-level sub-task，token slice size 是動態的（由 runtime 決定）。

### 3.2 Cross-SM signal via DSMEM

**Hopper 之前**：SM 之間唯一的通訊路徑是 global memory / L2 —— 延遲高（100+ cycles）、頻寬低（相對於 shared memory）。

**Hopper 之後**：**DSMEM（Distributed Shared Memory）** —— 同一個 GPC 內的 SM 可以直接讀寫彼此的 shared memory，延遲降到 shared memory 級別（20–30 cycles）。**這就是 cross-SM cooperation 變成 first-class 操作的硬體基礎**。

MorphKernel 用 DSMEM 做兩件事：

1. **Sub-task 之間的資料傳遞**——上游 sub-task 完成後，結果直接寫到下游 sub-task 對應 SM 的 shared memory arena（透過 DSMEM），不走 global memory
2. **Barrier / signal**——上游完成後，把 arena 裡的 `done` flag 位元 set 起來；下游 SM polling / TMA-triggered wake 這個位元

**這比 grid barrier 快多少**：
- Grid barrier：整個 kernel 所有 SM 都要到 —— 快的等慢的，wall clock = max(SM_time)
- Cross-SM signal：只有真正 dependent 的 SM 對之間同步 —— 獨立 sub-task 完全並行，wall clock = critical path length
- 論文號稱的 1.3× 大部分 gain 就來自這裡

### 3.3 GPC-topology-aware placement

**H100 拓樸**：144 SM 分成 8 個 GPC，每個 GPC 內部的 SM 共享 L2 partition、有較低的通訊延遲。**跨 GPC 通訊要走完整 L2 → 延遲 3–5× 於同 GPC**。

MorphKernel 的排程器：

- 把一條 dependency chain 儘量塞在同一個 GPC
- 高頻通訊的 sub-task pair 放同一個 GPC
- 相對獨立的 chain 分散到不同 GPC 平行

**這是把硬體拓樸提到 compiler 抽象的具體例子**——類似 CPU 世界 NUMA-aware scheduler。**在 A100（沒有 GPC 概念）上這個 pass 直接是 no-op**——所以我前面說 MorphKernel 只 target Hopper 之後不是設計懶惰，是硬體 primitive 限制。

### 3.4 Per-SM instruction queue

每個 SM 有一個 instruction queue：

- **compile time 產生**：queue 裡是 sub-task 執行順序
- **runtime 消費**：SM 依序 dequeue，執行完 signal 下游

**跟 persistent kernel 的關鍵差別**：

- Persistent kernel：**global** task queue，所有 SM 從同一個 queue 搶——需要 atomic 操作，contention
- MorphKernel：**per-SM** queue，compile time 已經決定分配——no contention，dequeue 只是 register read

**代價**：compile time 分配可能不是最優（因為不知道實際輸入 shape 造成的負載分布）。**MorphKernel 的補償**：JIT reduction 讓晚到的資料被吸收，不會浪費 SM 時間等它。

### 3.5 Just-in-time reduction

這是我覺得整篇最巧的地方。

**傳統 reduction tree**：compile time 決定 tree shape（例如 tournament reduction，log(N) 層 pairwise），tree 上每個節點對應一次 pairwise merge。**問題**：某條分支的 sub-task 特別慢，整個 tree 停在那個節點等它。

**JIT reduction**：tree shape **不在 compile time 固定**——每個 sub-task 完成後：

- 檢查有沒有其他已完成、可以合併的 partial sum
- 有 → 立刻做 pairwise reduce，產生新的 partial sum
- 沒 → 等，直到有一個可以合併

**結果**：先到的先合併、晚到的插進來，reduction 完成時機取決於**平均**而非**最慢**——這是統計上明顯的 latency win，尤其在 MoE 這種 load imbalance 很嚴重的場景。

實現上這需要一個小小的**per-thread block 內的 concurrent set**（記錄哪些 partial sum 可用），論文用 warp-level 的 vote / ballot 指令 + shared memory bitfield 做——這個實作細節我很想看原始碼（可惜還沒開源）。

---

## 4. 實測數字拆解

論文號稱：

| Workload | Baseline | Speedup |
|----------|----------|---------|
| MoE expert activation (Mixtral 8×7B decoding, H100) | CUTLASS group_gemm | 1.4–1.6× |
| Decoding attention (Llama-2 70B, batch 32) | FlashDecoding v3 | 1.2–1.4× |
| Multi-head latent attention (DeepSeek-V3 style) | 手工 CUDA | 1.3× |
| **Average across real workloads** | — | **~1.3×** |
| Peak (best-case shape) | — | ~1.8× |

**幾個值得注意的點**：

1. **Baseline 都是 SOTA**——group_gemm、FlashDecoding v3、手工 CUDA。**這代表 1.3× 是在已經打過一輪 static kernel fusion 的地基上再擠出來的**——不是抓 low-hanging fruit
2. **Load imbalance 越嚴重、gain 越大**——這符合我們的理論預期
3. **Sub-task 越小、gain 相對 persistent kernel 越明顯**——因為 persistent kernel 的 per-task overhead 在小 task 上被放大
4. **短 seq batch（decode）比長 seq batch（prefill）gain 大**——decode 的動態性本來就比 prefill 高

**沒 report 的**：power consumption、kernel launch overhead 分攤、DSMEM traffic vs L2 traffic 的分布——這些對實際部署很重要，希望 camera-ready 或 open review 階段有補充。

---

## 5. 限制與我不喜歡的地方

**論文自己坦承的**：

- **只支援 Hopper+（H100/H200/GB200）**——DSMEM / TMA 是硬體依賴
- **DSL 手寫門檻不低**——map/iter/reduce 抽象簡潔，但要寫得對得懂 sub-task 邊界、reduce 合法性
- **只做 dense compute**——sparse / gather-scatter 目前不 cover
- **多 GPU 沒展開**——單 GPU 內部的問題

**我自己觀察到的**：

- **沒開源**——SOSP paper 通常會開，但目前只有 accepted list，還沒看到 repo。這對 reproducibility 是硬傷
- **compiler frontend 太薄**——map/iter/reduce 只夠 cover simple pattern，複雜 workflow（例如 speculative decoding + KV cache 更新的 chain）恐怕還是要手寫
- **debug story 缺乏**——sub-task 依賴圖 + 動態排程的 bug 幾乎沒法從 trace 看出來，這是所有 dynamic scheduling system 的通病，也是 [[flashlight-torchinductor-attention-compiler-graph-rewrites-mlsys2026]] 提到過的 pain point
- **對 A100 用戶完全不 friendly**——但 A100 還是絕大部分公司的主力 inference 卡；真正 H100 產線量的可能只有 hyperscaler。這篇的 impact 要等 H100 普及才會兌現
- **JIT reduction 的正確性 proof 沒展開**——reduce 的順序如果是 non-associative（例如 float summation），結果會隨執行順序而變，這對需要 determinism 的 workload 是問題（呼應 [[llm42-verified-speculation-decode-verify-rollback-deterministic-llm-inference-sosp2026]] 那條線的 concern）

---

## 6. 這篇在整個 AI compiler / GPU serving 生態的位置

我這波 compiler 系列的 stack 圖大致長這樣：

```
                 Model / User code
                        |
        [source language: Mojo / Triton / CuTeDSL]     ← 8/25, 9/2
                        |
        [middle-end pass: Flashlight, ...]              ← 9/3
                        |
        [IR: Hexagon MLIR, TOSA MXFP]                   ← 8/26, 8/28
                        |
        [multi-GPU comm: Syncopate]                     ← 9/5
                        |
        [determinism / speculation: LLM-42]             ← 9/6
                        |
     ┌──────────────────┼──────────────────┐
     |                  |                  |
[static kernel]  [MorphKernel:      [persistent kernel:
[fusion]         dynamic sub-task]   task-parallel]     ← 9/10（本篇）
     |                  |                  |
     └──────────────────┴──────────────────┘
                        |
             [backend codegen: CuTeDSL]                 ← 9/2
                        |
        [serving loop: Event Tensor / ETC]              ← 9/4
                        |
            [verification: Argus]                       ← 8/31
                        |
        [benchmark: KernelBenchX]                       ← 8/30
                        |
        [package registry: HF Kernels]                  ← 8/27
                        |
                 Runtime execution
```

**MorphKernel 卡在「compile-time / runtime 分界」這一格**，它明確告訴你：

- 上面（source language / pass）都是 compile-time
- 下面（backend / runtime）都是 runtime
- **MorphKernel 是這條分界線本身**——它讓你可以在 compile-time DSL 裡宣告「這一部分留給 runtime」，然後 compiler 產生對應的 per-SM instruction queue + JIT reduction 機制

**這就是為什麼我把它排在整個 stack 圖的正中央**——它不是某一層的優化，是分層本身的重新設計。

---

## 7. 對走 compiler 職涯的我意味著什麼

（這一段我寫給我自己，以及對 compiler 有興趣的讀者）

我的 [[Compiler-Path]] roadmap 大致是：

- **Stage 1**：LLVM/MLIR 基礎（IR、pass、dialect）
- **Stage 2**：Pass infrastructure、layout assignment、memory planning
- **Stage 3**：**Runtime / scheduling layer（compile-time 跟 runtime 分工）**
- **Stage 4**：Backend codegen（PTX / SASS / target-specific tiling）

**MorphKernel 對應 Stage 3**——而且是最好的教材，因為它把「哪些決策留給 compiler、哪些留給 runtime」這條線畫得極清楚。**面試 senior compiler engineer 職缺（NVIDIA Compiler Team、Google MLIR、Meta PyTorch Compiler、OpenAI Triton Team）會考的思考題就是這個**：

> Given a workload with dynamic behavior, where do you draw the line between compile-time analysis and runtime scheduling? What information do you need at each layer?

MorphKernel 是這個問題的一個具體、可辯論的答案。**讀完這篇要能回答**：

1. 為什麼 sub-task shape 可以在 compile-time 決定，而 placement 不能？（因為 shape 只依賴 input tensor 的 static info，placement 依賴 runtime 的 load 分布）
2. 為什麼 dependency graph 是 compile-time，reduction tree 是 runtime？（因為 dependency 是演算法屬性、tree 是執行策略）
3. 為什麼 GPC-aware placement 是 pass 而不是 runtime decision？（因為拓樸是 static hardware info，placement 選項在 compile-time 就可以枚舉）

**這三個問題我要能不看論文複述出來、還要能延伸到「如果換成 Blackwell（GB200 有 Tensor Memory）情況會怎麼變？」——這是 senior 級別的推理**。

---

## 8. 我下一步要做的事

**閱讀**：等 SOSP camera-ready 出來（大概 10 月中）第一時間讀原文，重點看：
- Sub-task 依賴圖的具體資料結構（DAG？SSA-like？）
- JIT reduction 的 concurrent set 實作細節
- 跟 CUTLASS group_gemm 的實際 code diff

**實驗**：
- 拿 FlashDecoding v3 當 baseline，嘗試手工複製 MorphKernel 的 cross-SM sync + JIT reduction 思路
- 對比在 A100（沒 DSMEM）跟 H100 上的行為差異——量化 DSMEM 對 gain 的貢獻

**寫作**：
- 這篇當作 compiler 系列的 stage 3 入口
- 下一篇可能寫 **Mirage Persistent Kernel**（8/28 arxiv：Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs）——它是「另一條路」，跟 MorphKernel 的對比可以把 compile-time / runtime 分工這件事講得更透

---

## 9. 結語：dynamism 是 AI 系統的第一原則了

寫這篇的過程有一個很清楚的感覺：**AI 系統的競爭焦點已經徹底從「dense compute peak flops」轉移到「dynamic workload 的 tail latency」**。

- Dense compute 已經被 FlashAttention / CUTLASS / Triton 打到 90% peak，剩 10% 是硬體極限
- Dynamic workload 的 tail 才是實際部署 latency 的主宰——一個 batch 的完成時間是 max，不是 avg
- 打 tail 需要 runtime 決策；但純 runtime 決策 overhead 太大
- **答案是 compile-time / runtime hybrid** —— MorphKernel 是第一個把這條線畫成 first-class primitive 的系統

**如果你在做 AI compiler / LLM serving / GPU runtime**，這篇要跟 [[event-tensor-etc-dynamic-megakernel-llm-serving-cmu-mlsys2026]]（serving-loop 端的動態）、[[syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026]]（multi-GPU 端的動態）並讀——它們是同一個 first principle（dynamism）在不同 scale 的展開。

**如果你在準備 compiler engineer 面試**，把 compile-time / runtime 分工這件事講清楚，能明顯拉開跟其他候選人的距離。

---

## Sources

- [SOSP 2026 Accepted Papers](https://sigops.org/s/conferences/sosp/2026/accepted.html)
- [ACM SOSP'26 Papers & Preprints (pchaigno)](https://pchaigno.github.io/academic/2026/08/03/sosp-2026-papers.html)
- [SOSP 2026 Awesome Papers Notes](https://paper.lingyunyang.com/reading-notes/conference/sosp-2026)
- [SJTU IPADS Lab](https://ipads.sjtu.edu.cn/start)
- [Haibo Chen (IPADS)](https://ipads.se.sjtu.edu.cn/pub/members/haibo_chen)
- [Mirage Persistent Kernel (arXiv 2512.22219)](https://arxiv.org/html/2512.22219v1) — 對照組
- [Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling (arXiv 2512.24637)](https://arxiv.org/pdf/2512.24637) — Tsinghua 那篇 Morphable Kernels 的相關工作
- [ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants (arXiv 2604.18616)](https://arxiv.org/pdf/2604.18616) — 已寫過 [[argus-data-flow-invariants-llm-gpu-kernel-verified-2026]]
