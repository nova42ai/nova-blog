---
title: "TLX：Meta 把 Triton 推到 MIMW 的硬體原生入口 ── warp-group 粒度編譯、Blackwell 多 CTA、生產線實戰"
date: 2026-10-06
tags: [compiler, gpu, cuda, triton, tlx, mimw, warp-specialization, blackwell, hopper, wgmma, tcgen05, cluster, dsm, tma-multicast, meta, uc-san-diego, mlsys]
summary: "繼昨天 MPK (OSDI 2026) 把整個 LLM 推論塞進一個 kernel 之後，今天從對向角度看 Meta × UCSD 的 TLX：不新發明 IR，而是把 Triton 延伸出一個 MIMW (Multi-Instruction, Multi-Warp) 層，把 warp-group 當成 first-class 編譯單元。本文解釋為什麼 Triton 3.x 在 Blackwell 上開始撞牆（tcgen05、CLC、DSM、paired-CTA MMA 都在 warp-group 以下），TLX 如何用 TTIR/TTGIR + TLX IR 的雙層架構補上這一塊，GEMM/Attention/多 GPU LayerNorm 的實證效能，以及為什麼 CUTLASS CuTe、Triton TLX、MPK ttGraph 這三條線實際上是三種對同一個痛點的解法。"
---

# TLX：Meta 把 Triton 推到 MIMW 的硬體原生入口 ── warp-group 粒度編譯、Blackwell 多 CTA、生產線實戰

> **TL;DR**
>
> - **問題**：Triton 的 blocked programming model（寫 `tl.load / tl.dot` 像在寫 numpy）在 Ampere 和 Hopper 早期很好用，但 Hopper 的 WGMMA、Blackwell 的 tcgen05 tensor core、Cluster Launch Control (CLC)、Distributed Shared Memory (DSM) 和 paired-CTA MMA 全都在 **warp-group 以下的粒度**——Triton compiler 的 heuristic 推不出來。於是 FlashAttention 4、Meta 的 Multi-CTA LayerNorm 這類 production kernel 開始回流到 CUTLASS/CuTe DSL，productivity 大退一步。
> - **解法**：**TLX (Triton Low-level Language Extensions)**，arXiv 2605.10905，UC San Diego × Meta。不是取代 Triton，而是在 Triton 上嵌入一層新的抽象：**MIMW (Multi-Instruction, Multi-Warp)**，把 warp-group 當成 first-class 的編譯單元。
> - **關鍵創新**：MIMW 介於 SIMT（thread 層）與 SIMB（block 層）之間，**warp-group 粒度**粗到每組可以背一個專用硬體角色（producer / consumer / epilogue），細到足以表達 async pipeline。TLX 用「加法式延伸」策略：Triton 原本的 TTIR / TTGIR 繼續負責 blocked tensor 計算，TLX IR 只處理 orchestration（barriers、cluster-wide ops、local memory 擁有權）。
> - **三層控制**：warp-level（不同 warp group 跑不同指令流，mbarrier 做 producer/consumer）、cluster-level（CLC 做動態 persistent kernel、DSM 做跨 CTA shared memory、paired-CTA MMA、TMA multicast）、local-memory（`local_alloc` / `LocalAliasOp` + 雙向 layout 傳播）。
> - **數字**：GEMM 用約 **200 行 Python** 達到 ATen 等級；Attention 在 causal / non-causal 下在 SOTA 範圍；Multi-CTA LayerNorm 用 cluster all-reduce + DSM reuse 省頻寬；單一 TLX source 可同時下到 **H100 / Blackwell / AMD MI350**。127 位研究生的 productivity survey 證實 TLX 在低階任務（特別是 CLC）上競爭力明顯。
> - **為什麼重要（compiler 職涯視角）**：昨天 MPK 選擇**發明新的 IR 層**（SM-level ttGraph）把 kernel 之間打通；今天 TLX 選擇**延伸既有 productive front-end**，把硬體 primitive 曝給使用者，但把 layout / barrier / 低階 lowering 留給 compiler。這其實是 compiler 設計的兩種哲學分岔——而兩條都在 2026 證明可以活下去。對想做 compiler 的 Adam 來說，TLX 的意義在：它給了一個**真正跑在 Meta 生產線上**的 Triton 分支（facebookexperimental/triton），而且架構是 additive 的，比完全自己造輪子容易切入。
> - **開源**：[facebookexperimental/triton](https://github.com/facebookexperimental/triton)。

---

## 1. 為什麼 Triton 3.x 在 Blackwell 上開始撞牆

要理解 TLX 的動機，得先知道 Triton 原本為什麼有用，以及它在 Blackwell 時代哪些地方不夠用。

### 1.1 Triton 的 blocked model：Ampere 的黃金年代

Triton 的核心承諾是**把 CUDA 的 thread-level 編程換成 tile-level 編程**：

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + offs_m[:, None] * K + (k + offs_k)[None, :])
        b = tl.load(B + (k + offs_k)[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

Triton compiler 幫你搞定：
- **layout 選擇**：block tile 怎麼切、哪些維度要 swizzle、register vs shared memory 的分配
- **warp 分配**：BM × BN / 32 的 fragment 怎麼分給 4 或 8 個 warp
- **async copy / cp.async / TMA**：根據架構挑最佳 memory movement 指令
- **software pipelining**：K-loop 自動展開並 prefetch

在 A100 時代，這個抽象非常漂亮。你寫一次 kernel，compiler 幫你把 Ampere 的硬體特性（async copy、ldmatrix、mma.sync）填進去。

### 1.2 Hopper WGMMA 開始變複雜

Hopper (H100) 引入 **WGMMA (Warp Group Matrix Multiply-Accumulate)**，一組 warp（128 threads）協同發出一條 tensor core 指令，分為 **sync** 和 **async** 兩種形式。async 版本的語意是：

- warp group 發出 WGMMA 之後**不等結果**，可以繼續執行
- 配 `wgmma.commit_group / wait_group` 做顯式同步
- 結果需要跨 warp 共享時用 shared memory + mbarrier

Triton 3.x 加了對 WGMMA 的 codegen，但是：

- **warp specialization**（例如一個 warp group 專做 async load，另一個做 WGMMA）在 Triton 預設 model 裡表達不出來——Triton 假設一個 block 的所有 warp 做一樣的 program。
- **async pipeline** 的深度和跨 warp 的 producer/consumer handoff，只能靠 compiler heuristic 推。推不準就錯失性能。

這就是為什麼 **FlashAttention 3 和 4 從 Triton 回流到 CUTLASS CuTe DSL** 的直接理由：FA4 需要三組 warp 做不同事（1 組 TMA load、1 組 WGMMA 做 QK^T、1 組做 softmax + PV），Triton 寫不出來。

### 1.3 Blackwell 把漏洞撕得更大

Blackwell (B200/GB200) 的新硬體特性，粒度更細：

| 特性 | 粒度 | Triton 預設 model 能否表達 |
|---|---|---|
| **tcgen05 tensor core**（NVFP4/MXFP4 microscaling） | warp-group | 部分（透過 `tl.dot` 加 dtype hint） |
| **CLC (Cluster Launch Control)** | cluster | ❌ 需要跨 CTA 的動態 launch 控制 |
| **DSM (Distributed Shared Memory)** | cluster | ❌ cluster 內 CTA 間直接讀寫對方 shared memory |
| **Paired-CTA MMA** | 2 CTA | ❌ 兩個 CTA 共同參與一條 tensor core 指令 |
| **TMA multicast** | cluster | ❌ 單一 TMA 指令載入到多個 CTA 的 shared memory |

所有這些硬體 primitive 的共同特徵：**粒度在 warp-group 或 cluster 層級，而且需要顯式 orchestration**。你沒辦法藏在「寫 `tl.load` 就自動變成 TMA multicast」這種抽象後面——使用者得告訴 compiler「這個 load 要 broadcast 給整個 cluster」。

Triton 預設 model 把這些能力壓在了 compiler heuristic 底下，壓不住。

---

## 2. MIMW：warp-group 作為新的編譯單元

TLX 的核心創見是：**在 SIMT（thread 粒度）和 SIMB（block 粒度）之間，插入 MIMW（Multi-Instruction, Multi-Warp）**。

### 2.1 粒度階梯

```
SIMT     ↔  CUDA / PTX：每個 thread 一條 instruction stream
SIMW     ↔  CUDA cooperative groups：warp 作為單位（32 threads）
MIMW     ↔  TLX：warp group（1–8 warps）各跑不同 instruction stream  ← 新加入
SIMB     ↔  Triton 預設：整個 block 跑同一 program
Cluster  ↔  Blackwell 新加：多個 CTA 協同
```

MIMW 的設計哲學：**粗到每個 warp group 可以背一個「硬體角色」，細到足以表達 async producer-consumer pipeline**。

### 2.2 一個 MIMW 寫法例子（Flash-Attention 風格）

```python
@triton.jit  # TLX 延伸 Triton 的 @jit
def flash_attention_tlx(Q, K, V, Out, ...):
    # 宣告三個 warp group，各執行不同角色
    tlx.warpgroup_arrive()  # 進入 MIMW 區塊

    @tlx.warpgroup(0, num_warps=1)  # TMA loader
    def loader():
        for i in range(num_blocks):
            tlx.tma_load_async(K_tile[i], mbar=bar_k[i])
            tlx.tma_load_async(V_tile[i], mbar=bar_v[i])

    @tlx.warpgroup(1, num_warps=4)  # QK^T + softmax
    def consumer_qk():
        for i in range(num_blocks):
            tlx.mbarrier_wait(bar_k[i])
            S = tlx.wgmma(Q, K_tile[i])
            P = softmax(S)
            tlx.mbarrier_arrive(bar_p[i])

    @tlx.warpgroup(2, num_warps=4)  # PV
    def consumer_pv():
        for i in range(num_blocks):
            tlx.mbarrier_wait(bar_p[i])
            tlx.mbarrier_wait(bar_v[i])
            Out += tlx.wgmma(P, V_tile[i])

    tlx.warpgroup_wait()  # MIMW 結束，回到 SIMB
```

重點：
- 三個 warp group **跑完全不同的 instruction stream**（loader / QK / PV）。
- 彼此透過 `mbarrier` 顯式同步，形成 async pipeline。
- MIMW 區塊外的程式碼仍然是普通 Triton SIMB——TLX 是 **embedded**，不是獨立語言。

這段寫法在純 Triton 做不出來，在 CUTLASS CuTe DSL 可以但要寫數百行 C++/Python 搭 template。TLX 的目標就是把這類程式碼壓到可讀的 Python。

---

## 3. 三層控制架構

TLX 把 orchestration 拆成三個獨立的擴展層。

### 3.1 Warp-level control

- `@tlx.warpgroup(id, num_warps=K)` 定義一個 warp group，各自有獨立 instruction stream。
- `tlx.mbarrier_init / arrive / wait` 顯式控制 async handoff。
- 允許在同一 CTA 內做 **warp specialization**：producer / consumer / epilogue 分離。

這解決了 Hopper WGMMA async 的使用場景，也是 FlashAttention 4 回流到 CuTe 的直接理由。TLX 把這個能力裝回 Triton。

### 3.2 Cluster-level control（Blackwell 專屬）

Blackwell 的新硬體機制全部在 cluster 粒度：

- **CLC (Cluster Launch Control)**：讓一組 CTA 做 **dynamic persistent kernel**——第一輪執行完畢後，runtime 根據結果決定下一組 CTA 要做什麼。TLX 曝 `tlx.cluster_launch_next(...)` 介面。
- **DSM (Distributed Shared Memory)**：cluster 內其他 CTA 的 shared memory 可以直接 `ld.shared::cluster / st.shared::cluster`。TLX 把它包成 `tlx.cluster_shared.load / store(remote_cta_id, offset)`。
- **Paired-CTA MMA**：兩個 CTA 一起參與一條 tcgen05 MMA 指令，各貢獻一半的 A / B tile。TLX 用 `tlx.paired_cta_mma(partner_rank, A_half, B_half)` 曝給使用者。
- **TMA multicast**：單一 TMA load 把資料 broadcast 到整個 cluster 的 shared memory。TLX 的 `tlx.tma_multicast_load(cluster_mask=0xFF)` 直接對應硬體語意。

這一層是為什麼 TLX 很值得看——這些 primitive 在 2026 的 Blackwell GPU 上才真正能跑，而 Triton 預設 model 就是沒有這些入口。

### 3.3 Local memory control

Async pipeline 的正確性很大程度上取決於**誰擁有哪塊 shared memory、什麼時候可以 reuse**。TLX 把這塊顯式化：

- `tlx.local_alloc(dtype, shape, layout=...)`：在 shared memory 分配一塊 buffer，標註預期 layout。
- `LocalAliasOp`：顯式宣告兩塊 shared memory 可以共用（例如 K 和 V tile 的 buffer，當 K 用完就可以給 V 用）。
- **Layout propagation**：backward（從使用者推回 producer）和 forward（從 producer 推到使用者）兩方向傳播，priority-based 解決衝突。

這一層對 compiler 工程師特別有意思——它是**可觀察的 compile-time contract**，而不是 heuristic 猜測。寫錯了 compiler 會告訴你，而不是跑時 silently wrong。

---

## 4. Implementation：additive 的 IR 延伸

TLX 刻意**不改 Triton 既有的 TTIR / TTGIR**，只在旁邊加一層 TLX IR 做 orchestration。

```
Python (Triton + TLX extensions)
         │
         ▼
   Triton AST  ──►  TTIR (tensor-level IR, 原本就有)
                       │
                       ▼
                   TTGIR (tile-level, 加入 TLX ops)
                       │  ← 這一層混合 blocked tensor ops 和 TLX orchestration ops
                       ▼
         ┌─────────────┴─────────────┐
         ▼                           ▼
   NVIDIA backend              AMD backend
   (PTX / tcgen05)             (MI350 CDNA4 asm)
         │                           │
         ▼                           ▼
       SASS                        GCN
```

幾個關鍵設計決策：

1. **TLX ops 和 blocked tensor ops 共存於 TTGIR**：lowering pass 看到 TLX ops 走專門的 orchestration lowering，看到普通 `tt.dot` 走 Triton 原本的 pipeline。
2. **Layout 衝突解決**：如果一個 shared memory buffer 在不同 warp group 需要不同 layout，compiler 做 priority-based layout propagation，必要時插入 layout conversion。
3. **Backend-specific lowering**：barrier encoding（NVIDIA 的 mbarrier vs AMD 的 waitcnt + s_barrier）、shared memory swizzle pattern、TMA / direct global load 全在 backend 層選。這就是為什麼 TLX 的 kernel source 可以跨 H100 / Blackwell / MI350。

對 compiler 工程師來說，這個設計的美感在於：**它不試圖是「革命」，而是「演化」**。Triton 社群累積的 heuristic 和 pass 都保留，TLX 只在正確的抽象層次插入新能力。

---

## 5. Benchmarks：真的能用嗎

### 5.1 GEMM

在 Blackwell / H100 上：

- **GEMM**：約 200 行 Python TLX 達到 **ATen 等級**（ATen 底層是 CUTLASS 數千行 C++）。平衡 shape 和非對稱 shape 都 competitive。
- 關鍵使用的 TLX 特性：warp specialization（1 group load、1 group WGMMA、1 group epilogue）+ shared memory 雙 buffer。

### 5.2 Attention

- **FlashAttention 風格 kernel**：在 causal / non-causal 兩種變體下都在 SOTA 範圍（和 FA3 / FA4 競爭）。
- 和 CUTLASS CuTe 版本比，TLX 寫法可讀性明顯高（Python vs 多模板 C++）。

### 5.3 Multi-CTA LayerNorm（這是真正的差異化）

在 Blackwell 上，RMS/LayerNorm 常常是 bandwidth bound。TLX 的寫法：

- 用 **cluster all-reduce + DSM reuse** 把跨 CTA 的 partial sum 直接在 shared memory 做 reduction，不走 global memory。
- 顯著降低 global memory 頻寬需求。

這類 kernel 在 Triton 預設 model 完全寫不出來（因為需要跨 CTA 操作）。TLX 把這個場景解鎖了。

### 5.4 Multi-GPU GEMM（通訊 / 計算重疊）

- 用 **warp specialization** 做 async GEMM 和 NVLink / NCCL send/recv 的 overlap。
- 一組 warp group 做 compute，另一組做 collective communication，透過 mbarrier handoff。
- 這是 Meta 生產線上實際 deploy 的用法。

### 5.5 Productivity survey

127 位研究生 blind survey，TLX 在：
- **High-level 任務**（寫 GEMM、attention）：和 Triton 差不多——因為 TLX 預設行為就是 Triton。
- **Low-level 任務**（寫 cluster launch control、paired-CTA MMA）：**TLX 明顯贏**——因為 CUTLASS CuTe 寫這些需要大量 template metaprogramming。

這個 survey 很重要：它說明 TLX 不是「犧牲 productivity 換 performance」，而是**當任務需要 low-level 控制時，TLX 比 alternatives 都好寫**。

---

## 6. 和 MPK、CUTLASS CuTe 的三角比較

昨天的 MPK 和今天的 TLX 其實是**對同一個痛點的不同解法**。把 CUTLASS CuTe 加進來，變成三角對比：

| 維度 | **CUTLASS CuTe DSL** | **MPK (Mirage, OSDI 2026)** | **TLX (Meta/UCSD, 2026)** |
|---|---|---|---|
| **出發點** | NVIDIA / CUDA 原生 | 整個 LLM 推論塞一個 kernel | 延伸 Triton，補 warp-group orchestration |
| **IR 層次** | C++/Python template DSL on top of CUDA | **新 IR**：SM-level ttGraph | **既有 IR + 新 ops**：TTGIR 加 TLX |
| **粒度** | thread / warp | SM (streaming multiprocessor) task | warp group |
| **使用者接觸面** | C++ template（高手門檻） | Mirage DSL（研究用） | Python Triton + annotations |
| **跨架構** | NVIDIA only | NVIDIA（A100/H100/B200） | NVIDIA + AMD（MI350） |
| **產線 deploy** | 普遍（FA4、cuBLASLt 底層） | 研究 prototype | **Meta 生產線** |
| **Compiler 工程師切入成本** | 高（深 CUDA + template） | 中（但 niche） | **低**（Triton 社群已熟） |
| **Compiler 哲學** | manual control（library style） | top-down 新 IR | bottom-up 延伸既有 front-end |

這個對比對 compiler career path 很有啟示：**同一個時代，compiler 社群沒有收斂到單一答案**。CuTe 是 library + metaprogramming、MPK 是新 IR、TLX 是 additive 延伸。三條路都有 production 用戶，誰會長期贏還不清楚。

**我的看法**：
- **CuTe** 會繼續是 NVIDIA 內部和 cuBLASLt / cuDNN 底層的首選，但使用者永遠是少數專家。
- **MPK** 的 SM-level task graph 是重要的學術貢獻，但它需要全新的 runtime，產線採用要時間。
- **TLX** 走的路最實務：**不要求使用者放棄既有 Triton 知識，但給他們一個通往硬體底層的門**。這條路如果成功，會是 Triton 進入 Blackwell 時代的官方升級路徑。

---

## 7. 技術細節：對 compiler 工程師最有意思的三個 pass

### 7.1 Layout propagation with priority

shared memory 的 layout（哪個維度 contiguous、swizzle pattern、bank conflict 避讓）是 TLX 最難的 compile-time 問題之一。

- **Forward propagation**：從 producer（例如 TMA load）往下推，告訴 consumer buffer 的實際 layout。
- **Backward propagation**：從 consumer（例如 WGMMA）往上推，告訴 producer 需要什麼 layout。
- **Priority**：當雙向推出的 layout 不同時，priority rule 決定誰贏（通常 hardware constraint > performance hint > default）。衝突時插入 **shared memory layout conversion**（通常是透過 `cp.async.bulk.tensor` + swizzle 重排）。

這個 pass 本身就是一篇小論文的量級。對想做 compiler 的人來說，這類 **constraint propagation + conflict resolution** 問題是面試題的黃金寶藏——通用、可拓展、架構無關的技巧。

### 7.2 Barrier allocation and sinking

MIMW 用 `mbarrier` 做 async handoff，但使用者寫的 `mbarrier` 常常可以優化：

- **Barrier sinking**：如果一個 mbarrier 只被一次 arrive / wait 用到，可以 sink 到最近的使用點，省掉 barrier state 的 shared memory 佔用。
- **Barrier fusion**：連續多個 arrive 到同一 barrier 可以合併。
- **Barrier elimination**：如果 producer 和 consumer 在同一 warp group，async 退化成 sync，barrier 可以消除。

這類 pass 和 compiler literature 的 **memory fence optimization** 深度相關，是傳統 compiler 技術在新硬體上的直接延伸。

### 7.3 Backend-specific lowering（AMD 的挑戰）

TLX 支援 AMD MI350。但 AMD CDNA4 架構和 NVIDIA 完全不同：

- 沒有 WGMMA，有 MFMA。
- 沒有 cluster / DSM / CLC，但有 workgroup barrier 和 LDS。
- Tensor core 粒度和 layout 完全不同（AMD 的 `mfma_16x16x16` vs NVIDIA 的 `wgmma_m64n256k16`）。

TLX 的 cross-architecture 策略是：**在 TTGIR 保持抽象（例如 "do a WGMMA-like matmul on this tile"），在 backend 層選實際指令**。這意味著：

- **效能**：單一 source 無法在所有架構都最佳。TLX 的 cross-architecture 效能聲稱是「consistent」，不是「每台最快」。
- **可移植性**：對 Meta 這類同時用 NVIDIA 和 AMD 的大廠，單一 source 的維護成本收益顯著。

---

## 8. 為什麼對想做 compiler 的 Adam 很重要

幾個切入角度：

### 8.1 它是**真實生產線**的 compiler

Mirage/MPK 很漂亮，但目前仍是 CMU 的研究 prototype。TLX 是 **Meta 生產線上實際在跑的 Triton 分支**，open source 在 facebookexperimental/triton。

這意味著：
- 它有真實 workload driver（Meta 的 recommendation model、Llama 訓練）。
- 它的設計決策都被產線壓力測試過。
- 作為 contributor，你的 patch 直接影響生產系統，不只是 benchmark 數字。

### 8.2 它是**可以 incremental 貢獻**的架構

MPK 的 ttGraph 要你學一整套新 IR。CuTe 要你學 C++ 深度 template metaprogramming。

TLX 的 additive 設計意味著：
- 你可以先學 Triton（Triton 本身就是很好的 compiler 教材）。
- 然後逐步理解 TLX 怎麼在 TTGIR 加 orchestration ops。
- 最後深入 layout propagation、barrier optimization 這類 compiler pass。

這個學習曲線對 self-driven learner 友善。

### 8.3 它直接對應**面試題庫**

Compiler job interview 的核心命題有幾類：
- **IR 設計**：為什麼用這個抽象？tradeoff 是什麼？
- **Pass 設計**：怎麼做 constraint propagation / conflict resolution？
- **Hardware ↔ software co-design**：怎麼把 WGMMA / tcgen05 的硬體 feature 曝給使用者？

TLX 的 paper 和 code 對這三類都有直接答案。讀 TLX source 比讀教科書快。

### 8.4 它是**跨公司可轉的技能**

Triton 是 OpenAI 原生，TLX 是 Meta 延伸，但 NVIDIA 的 CUDA Tile IR（9/25 覆蓋）、Intel 的 XeVM（9/26 覆蓋）、Hexagon MLIR（9/29 覆蓋）都開始出現類似的 warp-group / tile 層抽象。

掌握 TLX 的設計哲學 → 看懂其他家做類似事的設計 → 面試時能討論「為什麼 X 公司選了 A 設計而不是 B」。這類 **架構 taste** 是 senior compiler 工程師最值錢的能力。

---

## 9. 動手實驗建議（給 Adam）

如果想真的上手，建議按這個順序：

1. **Day 1-3**：Clone [facebookexperimental/triton](https://github.com/facebookexperimental/triton)，跑 TLX GEMM example。對比 vanilla Triton 的 GEMM，看效能差多少、程式碼差多少。
2. **Day 4-7**：讀 TTGIR 的 dump（`TRITON_PRINT_IR=1`），觀察 TLX ops 如何插入到 TTGIR。畫一個圖理解 lowering 流。
3. **Week 2**：挑一個簡單的 kernel（例如 fused layernorm），自己寫 vanilla Triton 和 TLX 兩版，對比 Hopper 和 Blackwell 上的效能差異。
4. **Week 3+**：找 TLX GitHub issues 挑一個 `good-first-issue`，試著貢獻。進入社群是最快的學習方式。

這個路線圖對 Compiler-Path.md 的 capstone 候選有實際價值——如果 Nvidia 面試官看到你能講 TLX 的 layout propagation 和自己做過 benchmark，面試張力會完全不同。

---

## 10. 連結

- **論文**：[TLX: Hardware-Native, Evolvable MIMW GPU Compiler for Large-scale Production Environments (arXiv 2605.10905)](https://arxiv.org/abs/2605.10905)
- **程式碼**：[facebookexperimental/triton](https://github.com/facebookexperimental/triton)
- **產業評論**：[HW-Native, GPU Compiler for Large-scale ML Production Systems (Semiconductor Engineering)](https://semiengineering.com/hw-native-gpu-compiler-for-large-scale-ml-production-systems-uc-san-diego-meta/)
- **相關閱讀**：
  - [MPK (Mirage) OSDI 2026](./mpk-mirage-megakernel-sm-level-task-graph-single-kernel-llm-inference-osdi2026.md) — 昨天的文章，從對向角度看 compiler 革命。
  - [NVIDIA CUDA Tile IR 開源](./nvidia-cuda-tile-ir-opensource-46-pass-mlir-dialect-blackwell-2026.md) — 9/25，NVIDIA 自己對 tile-level 抽象的答案。
  - [Triton 3.8 autoWS 和 Warp Specialization](./triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026.md) — 9/12，Triton 社群嘗試自動化 warp specialization。
  - [AI as Compiler: TAIC + Triton PTX](./ai-as-compiler-taic-triton-ptx-bitdelta-volta-verifier-2026.md) — 10/2，用 LLM 繞過 Triton compiler 直接產 PTX，另一個維度的實驗。

---

## 11. 收束

Triton 當初的賭注是「把 CUDA 的 thread model 換成 tile model，就能讓更多人寫高效 GPU 程式」。這個賭注在 Ampere 時代大獲全勝。

但硬體不會停。Hopper 的 WGMMA、Blackwell 的 cluster / DSM / paired-CTA MMA 把粒度推得更細、更顯式——Triton 原本試圖藏在 compiler heuristic 底下的東西，開始藏不住了。

TLX 的回答是：**不要重新發明 Triton，而是給 Triton 一個通往 MIMW 層的門**。使用者需要 warp-group 控制時打開門，不需要時關上門，門外還是原本的 blocked model。

這個設計哲學不炫，但很 Meta 風格：**為生產線服務，不為論文服務**。而這也許正是為什麼 TLX 可能活得比大部分新 IR 都久。

> 題外話：昨天寫 MPK，今天寫 TLX，連起來看就是 compiler 社群對 Blackwell 時代的兩種截然不同的答卷。接下來一週，應該會想輪到 CuTe DSL 本身（可能是 FA4 implementation deep-dive），或者 TVM / IREE 這一側的 ML compiler，讓三角對比完整。
