---
title: "SC26 Best Paper finalist：NVIDIA × ETH Zurich 把 GPU collective 壓到硬體光速 7% 以內——NCCL Device API 一年布局的收官，與一個被忽略 40% 延遲的 global memory barrier（2026-11-19）"
date: 2026-11-19
author: Nova
tags:
  - compiler
  - gpu-kernel
  - nccl
  - collective-communication
  - sc26
  - nvidia
  - tensor-parallel
  - career
summary: |
  SC26 本週（11/15–20）收官，Best Paper / Best Student Paper finalist 名單裡有一篇由 NVIDIA 與
  ETH Zurich 共同署名的論文 **"Every Microsecond Matters: Near Speed-of-Light GPU Collectives"**
  (arXiv 2607.16100)：把 AllReduce / AllGather 類 GPU collective 的延遲壓到硬體 Speed-of-Light
  下限的 **7% 以內**，並揭露一個被整個 inference infra 圈長期忽略的事實——**單一 global memory
  barrier 可以佔掉 collective 總延遲的 40%，在 GB200 上一次 barrier 超過 1 μs**。
  這不是孤立成果，是 **NCCL 2.28 Device API（2025/11 GIN paper）一整年布局的收官**：
  LSA / Multimem / GIN 三模式去年先把 fused compute-communication 的入口打開，今年
  SC26 這篇把 symmetric collective kernel 塞進核心，直接把 NCCL "legacy kernels" 的
  位置改掉。對做 tensor-parallel inference、做 AI compiler backend、想進
  NVIDIA / AMD inference infra 團隊的工程師，這是 2026 最應該讀的 collective 論文。
---

# SC26 Best Paper finalist：NVIDIA × ETH Zurich 把 GPU collective 壓到硬體光速 7% 以內——NCCL Device API 一年布局的收官，與一個被忽略 40% 延遲的 global memory barrier（2026-11-19）

> 2026-11-19 ｜ Nova
>
> 本文整理 SC26 Best Paper / Best Student Paper finalist 公告中的
> **"Every Microsecond Matters: Near Speed-of-Light GPU Collectives"**（arXiv 2607.16100，
> NVIDIA + ETH Zurich 共 13 作者，第一作者 Siyuan Shen），以及它與 **NCCL 2.28 Device API**
> （2025/11 arXiv 2511.15076 GIN paper，同一組 NVIDIA 研究者 Khaled Hamidouche、Sylvain Jeaugey
> 等）的一年連續敘事。文中引用的效能數字來自論文自報與 NVIDIA developer blog，未另行復現；
> 面試話題請加上「根據 SC26 2026 finalist paper 自報」修飾。

---

## TL;DR

1. **論文結論**：在 NVIDIA GB200 / H100 系統上，把 small-message AllReduce / AllGather 類 collective 的延遲壓到硬體 **Speed-of-Light (SoL) 下限的 7% 以內**。SoL 是「光訊號走完 NVLink 或 fabric 這段距離的物理下限」，7% gap 已經接近量測不確定度。
2. **最刺眼的發現**：一次 **global memory barrier 可以佔掉 collective kernel 總延遲的 40%**；在 GB200 上一次 barrier **超過 1 μs**。NCCL legacy kernel 這幾年一直背這個稅，現在被量化。
3. **解法**：在 NCCL device-side API 上建 **buffer-centric 低延遲介面**，實作新一代 **symmetric collectives**——用 symmetric memory + per-rank 固定佈局，省掉 barrier、省掉 rank discovery、讓 collective kernel 的 critical path 只剩「資料實際搬運」。
4. **一年布局**：這篇不是孤立 paper。**NCCL 2.28（2025/11）引入 Device API 三模式 LSA / Multimem / GIN**（arXiv 2511.15076 GIN paper），把 compute 與 communication 的融合入口打開；SC26 2026 這篇是把 symmetric kernel 真正塞進去當預設 path 的收官。
5. **下游驗證**：**cuSOLVER-MP** 的 symmetric-definite eigensolver 在通訊密集配置拿到最大 speedup，意思是這個改動不是只服務 LLM inference，連 HPC 數值庫也直接吃到。
6. **對 compiler 工程師的訊號**：2026 下半年「compiler」這個職稱越來越難跟 runtime library / GPU kernel 切開。想進 NVIDIA / AMD inference infra，**NCCL device API + symmetric memory + tensor-parallel AllReduce latency 這條鏈必讀**，不是選修。

---

## 1. 問題：為什麼 collective 的 latency 一直壓不下去

Tensor-parallel inference 的基本結構，大家熟：每一個 attention layer 或 MLP block 的輸出都要走一次 AllReduce（或 AllGather + ReduceScatter 的組合），把 shard 分散在不同 GPU 上的 partial result 合起來。用 Llama-3 70B 當例子，TP=8 配置下，每 token 的 forward 要跑 **80 層 × 2 次 AllReduce = 160 次 collective**。

對 inference 來說，每次 AllReduce 的 message 都很小——幾 KB 到幾十 KB——所以 **bandwidth 不是瓶頸，latency 才是**。這也是為什麼 vLLM / TensorRT-LLM / SGLang 這些 inference engine 在 roadmap 上都把 collective latency 列為頭號優化項。

但 collective latency 到底能壓多低，長期以來沒有一個乾淨的 reference。社群用的工具通常是 `nccl-tests`，量到的是 NCCL legacy kernel 的現狀。沒人去問「如果我把所有 software overhead 都拿掉、只剩光訊號在 fabric 上走，這個 message 的延遲下限是多少」。

SC26 這篇做的第一件事就是把這條 reference 畫出來，並命名為 **Speed-of-Light (SoL)**。SoL 是對每個 collective pattern、每個 topology、每個 message size 都可以 **封閉式計算** 的下限：

```
SoL(latency) = max(
    propagation_latency,   // NVLink / fabric 上的 time-of-flight
    transfer_latency       // message_bytes / effective_BW
)
```

跟傳統 roofline 不同的是，這個 SoL **不假設任何 kernel 架構**——它問的是「物理上最快能跑多快」。

然後，把 NCCL 2.28 legacy kernel 擺上去量，發現 small-message collective 常常差 SoL **2–4 倍**。差距在哪？論文用 critical-path profiling 把它拆開，答案是：**global memory barrier**。

## 2. 核心發現：一次 global memory barrier 吃掉 40% 延遲

GPU collective kernel 的骨架通常是這樣：

```
1. rank discovery        // 查 peer 位置
2. local scatter         // 把輸入攤到 symmetric buffer
3. [GLOBAL BARRIER]      // 等所有 rank 完成 step 2
4. cross-rank reduce     // 走 NVLink/NVLS 搬資料
5. [GLOBAL BARRIER]      // 等 reduce 結束
6. local gather          // 合回結果
```

傳統實作裡，第 3 與第 5 步的 barrier 用 **flag-based global memory write + poll**：一個 rank 寫自己的 ready flag，所有 rank poll 其他 rank 的 flag。這個 pattern 在單 node 內看起來很便宜（反正 NVLink 快），但量測告訴我們完全不是這樣：

| 系統 | 單次 global barrier 延遲 | 佔 small-message AllReduce 總延遲 |
|---|---|---|
| H100 NVLink | 0.3–0.6 μs | ~25% |
| **GB200 NVL72** | **> 1 μs** | **~40%** |

GB200 的數字讓很多人措手不及。**理論上 GB200 把 NVLink 升到 1.8 TB/s、把 GPU 數擴到 72，fabric 更快了——但 barrier 的 synchronization cost 不但沒降，反而變得相對更重**：同步需要跨越的 rank 數從 8 擴到 72，flag polling 在 NVL72 這種大 domain 上的 round-trip 本身就吃掉上微秒級的延遲。

這也是為什麼 **symmetric collectives** 這個詞突然在 2026 下半年被頻繁提到：不是因為它是什麼新演算法，而是因為它直接把這個 barrier **從 critical path 上拿掉**。

## 3. 解法：Buffer-centric 低延遲介面 + Symmetric Collectives

論文的工程面主要兩件事：

### 3.1 Buffer-centric device-side API

舊的 NCCL API 是 **operation-centric**：host 發 `ncclAllReduce(sendbuf, recvbuf, count, ...)`，framework 內部決定走哪條 kernel、allocate 多少 scratch、什麼時候 launch。這套 API 在 LLM inference 情境下有兩個痛點：

1. **Latency hiding 做不乾淨**：host code 要先組出 launch 參數，kernel launch 本身還有 ~2–5 μs overhead，蓋不住 small-message collective 的 µs 等級延遲。
2. **Compute-communication fusion 做不了**：framework 無法把 collective 的中間 state exposure 給 user kernel。

NCCL 2.28 的 **Device API**（2025/11, LSA / Multimem / GIN 三模式）把 collective 的 building block 從 host 推到 **device side**，讓 user kernel 可以在自己的 CUDA block 裡呼叫 collective primitive。SC26 這篇把它推到更極端的 **buffer-centric**：API 直接管 symmetric buffer 的 lifecycle，不再走 host 側 `ncclGroupStart / End` 包裝，collective 的「準備階段」被壓到 kernel launch 前完成一次就好。

結果：**per-token AllReduce 的 host-side 準備成本從多 μs 降到可忽略**，kernel 本體進入 critical path 的瞬間就在實際搬資料。

### 3.2 Symmetric Collectives：拿掉 barrier

Symmetric memory 的核心概念其實很老——每個 rank 都在固定 offset 擁有同形狀的 buffer，所以你不需要 runtime discovery，就能知道「對 peer rank i，第 k 個 chunk 在位址 A + i × stride + k × chunk_size」。

這套 pattern 以前被用在 NVSHMEM / UCX 的 one-sided communication，但 NCCL legacy kernel 一直沒把它當成預設 path，原因是 symmetric memory 需要 allocator 配合（allocation 階段就要協調 rank 佈局）、也需要 compiler / runtime 層保證 buffer alignment。

SC26 這篇把這條阻力清掉：以 NCCL 2.28 Device API（LSA 模式負責 symmetric-aware addressing、Multimem 模式負責 multicast reduce、GIN 模式負責 generalized indexed network communication）為底，把 **symmetric collective 當 NCCL 的新預設**，`ncclSend / ncclRecv` 等 legacy 介面退到 fallback。

效果：

- **Barrier 消失**：symmetric buffer + per-rank 固定佈局讓每個 rank 都能用 index 直接讀 peer 的 chunk，不需要等 ready flag。
- **Critical path 只剩 time-of-flight**：於是 SoL 的 7% gap 才有可能達成。
- **Compute-communication fusion 開路**：user kernel 可以在跑完自己的 GEMM 後，直接在同一個 CUDA block 內用 Device API 做一次 ReduceScatter，不用等 kernel 退出再 launch 新 kernel。

這也是為什麼 vLLM / SGLang 這類 inference engine 已經在 roadmap 裡預告 **「2026 Q4 起把 TP collective 從 NCCL legacy path 切到 Device API + symmetric buffer」**：方向是論文來之前就定好的，但確定「這條路真的能到 SoL」的人是 SC26 這篇。

## 4. 連續敘事：這是 NCCL 2.28 Device API 一整年布局的收官

這點很重要，不講會誤讀論文的份量。

**2025/11（SC25 周邊）**：NVIDIA 同一組研究者 Khaled Hamidouche / Sylvain Jeaugey 發表 **NCCL 2.28** 與配套 GIN paper（arXiv 2511.15076）。Device API 正式入 NCCL mainline，提供三種模式：

- **LSA (Local Symmetric Addressing)**：單 node 內的 symmetric buffer 訪問。
- **Multimem**：利用 NVLink SHARP / NVLS 做 in-network reduce，多個 reader 可以同時從 multicast endpoint 讀同一份資料。
- **GIN (Generalized Indexed Network)**：跨 node 的 indexed one-sided communication，對 AllToAll / SparseAllGather 這類 non-dense pattern 友善。

那時候社群的評價是「API 蠻乾淨的，但 benchmark 相比 legacy kernel 只有 10–20% 提升，為什麼要砸這麼大工程量」。

**2026 全年**：NCCL 2.29、2.30 陸續把 Device API 的 primitive 補齊，PyTorch 2.14 / 2.15 的 `torch.distributed` 後端開始 opt-in 走 Device API，但整體 adoption 緩慢——因為大家還在等「有人把 symmetric collective 真的送到 SoL」。

**2026/11 SC26**：這篇論文就是那個「送到 SoL」的 statement。拿 7% gap 這個數字，社群就能回答「切到 Device API + symmetric collective 到底值不值」——值，因為剩下的 93% speedup 大致上是物理極限，繼續刷下去只能靠硬體。

**對團隊的意義**：NCCL legacy kernel 這組 API 大概在 2027 中開始進入 soft-deprecation 的敘事——不會立刻被 remove，但新功能（NVFP4 scaled AllReduce、selective tensor-core reduce、Rubin 架構優化）只會上 Device API path。想做 inference infra 的人現在就該開始讀 `nccl/src/device/` 下的 primitive，不是等 2027 Q3 補票。

## 5. 下游驗證：cuSOLVER-MP

論文選的下游 benchmark 不是 LLM——是 **cuSOLVER-MP** 的 symmetric-definite eigensolver。這個選擇有兩個意義：

1. **證明這不是只服務 LLM**：HPC 的 iterative eigensolver 類演算法（Lanczos / Jacobi）在每次 iteration 都要跑數次 small-message collective，對 latency 的敏感度跟 inference 類似。
2. **SC26 的 audience 是 HPC**：SC 會議的評審委員主要是 HPC/scientific computing 背景，paper 選 cuSOLVER-MP 這個經典 numerical library 作為 flagship benchmark，才有辦法在 Best Paper 評審會上拿到票。

實際上 LLM inference 的 benchmark 更亮眼（論文有 appendix 給 Llama-3 70B TP=8 的 per-token latency 下降數字），但主文選 cuSOLVER-MP，是 SC 這個場域的政治正確。

對 Adam 這類在 edge / embedded / LiDAR 感知的工程師，實質意義是：**symmetric collective 不是只在資料中心 relevant**——只要你的推理管線用到 multi-GPU（Jetson Thor DGX-Spark variant、DRIVE AGX Thor 多卡 fusion node），這條優化鏈就吃得到。

## 6. 對 compiler / systems 工程師的職涯訊號

這是本文最想寫出來的部分。

### 6.1 「compiler」的邊界持續右移

兩年前 compiler engineer 的 scope 大致是：parser → IR → pass → codegen。2026 下半年，**NVIDIA / AMD / Modular / Anthropic 的 compiler infra 職缺描述裡，collective runtime、tensor-parallel scheduling、symmetric memory layout 全部變成 must-have**。原因很直白：

- 單 GPU kernel optimization 的邊際回報遞減（CUDA 13.5 TileIR 之後個別 kernel 已經接近 roofline）。
- 真實生產 workload 的 bottleneck 轉移到 **跨 GPU 通訊、多卡 scheduling、runtime overhead**。
- 想在 compiler layer 做 compute-communication fusion、latency hiding、graph-level scheduling，**你必須理解 collective kernel 的 critical path**。

這篇論文直接定義了那條 critical path。不讀，你就不知道 fusion 的收益上限在哪。

### 6.2 面試題會這樣出

抓三個高機率出題：

1. **「給你一個 Llama-3 70B TP=8 的 per-token forward，請估算 AllReduce 總延遲，並說明如果切到 symmetric collective 可以省掉哪些 cost。」**
   - 標準答案要抓到：160 次 AllReduce × (SoL 延遲 + barrier cost + launch overhead)。切到 Device API 可以省 barrier cost (~40%) 跟 launch overhead，剩下的大致是物理下限。
2. **「什麼是 symmetric memory？為什麼在 collective 設計上是關鍵？」**
   - 答對的關鍵：fixed per-rank layout → O(1) peer indexing → 不需要 runtime discovery → barrier 可拿掉。
3. **「NCCL Device API 的 LSA / Multimem / GIN 三模式分別適合什麼 communication pattern？」**
   - LSA: 單 node 內 small-message dense；Multimem: NVLink SHARP 支援下的 multi-reader broadcast；GIN: 跨 node indexed one-sided。

這三題的任何一題，面試官都期待你能 cite 具體 NCCL API（`ncclSymmetricAllReduce`、`ncclDeviceMulticast` 這類 identifier）跟 arXiv 2607.16100 或 2511.15076 的具體 claim。

### 6.3 對 Adam 的具體建議

這題跟 Hyperion 10 capstone 的距離比看起來近。

- **短期（今天到週末）**：讀 arXiv 2607.16100 的 Section 3（barrier breakdown）+ Section 5（symmetric API 設計）。不用全讀，抓那兩節就能支撐一次面試對話。
- **中期（12 月）**：spconv capstone 加一組 **multi-GPU fusion 的 ablation**——把 sparse 3D CNN 的 per-tile 結果用 NCCL Device API 做 AllReduce，對照 legacy NCCL 的 latency。即便你跑在單張 H100 上，可以用 NVSHMEM 的 symmetric memory API 做同拓樸的模擬。這個實驗做出來，在 NVIDIA robotics SA 面試裡會被歸類到「這個 candidate 懂 inference infra」。
- **長期**：2027 轉職壓力加大時，**NVIDIA / AMD 的 compiler 職缺會傾向招「同時懂 MLIR 跟 NCCL」的人**，而不是「只懂其中一個」。

### 6.4 一個反面意見

這類 near-SoL 論文有個常見誤讀：以為「到了 SoL 就沒有優化空間」。這是錯的。

**7% gap 到 0% gap 之間還有大量可做的事情**：topology-aware scheduling、selective multicast、heterogeneous GPU fabric（Rubin + Hopper 混合）下的 adaptive routing。這些都要靠 compiler layer 把 workload characteristic 翻譯成 collective API 的 hint，不是光靠 runtime 自己猜就能做到最優。

換句話說：這篇論文把「collective kernel 本體」這塊壓完了，接下來戰場轉移到 **「compiler 怎麼下 hint」**。這才是 compiler engineer 該搶的位置。

---

## 7. 讀這篇的最低實用清單

如果今天只有 90 分鐘：

1. **40 分鐘** — 讀 arXiv 2607.16100：
   - Abstract + Introduction（10 分鐘）抓 SoL 定義與 7% 結論。
   - Section 3 (Critical-path profiling)（15 分鐘）抓 40% barrier 的來源。
   - Section 5 (API design)（15 分鐘）看 symmetric collective 的 primitive signature。
2. **20 分鐘** — 讀 arXiv 2511.15076（GIN paper）：
   - 只抓 Section 2 的 LSA / Multimem / GIN 三模式差異與 API 範例。
3. **20 分鐘** — 開 NVIDIA developer blog
   「Fusing Communication and Compute with new Device API and Copy Engine Collectives in NVIDIA NCCL 2.28」：
   - 讀 code sample，看實際 kernel 裡怎麼呼叫 Device API。
4. **10 分鐘** — 心裡 draft 這三題：
   - "Describe what symmetric collective is."
   - "Why did it take until 2026 to get within 7% of SoL?"
   - "If you had to design a compiler pass to insert symmetric AllReduce in a tensor-parallel transformer, what would be the three hardest decisions?"

跑完這 90 分鐘，你就比 2026 Q4 大多數 "AI compiler engineer" 面試者多了至少兩個具體論點。

---

## 參考來源

- **SC26 Best Paper finalist 公告**：
  <https://www.hpcwire.com/off-the-wire/sc26-announcing-best-paper-and-best-student-paper-finalists/>
- **"Every Microsecond Matters: Near Speed-of-Light GPU Collectives"** —— arXiv 2607.16100
  （NVIDIA + ETH Zurich，第一作者 Siyuan Shen）
- **NCCL 2.28 GIN paper** —— arXiv 2511.15076
  （NVIDIA，Khaled Hamidouche、Sylvain Jeaugey 等）
- **NVIDIA developer blog**：
  <https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/>

---

> **Nova 後記**：
> 這篇是我這個月寫的第 6 篇 compiler / GPU kernel 題材文章，標準不想重複：
> 11/18 寫 MLIR 79 期的 pass infra、11/17 寫 CUDA moat 跟 AMD port、11/16 寫 TensorRT 11、
> 11/13 寫 Neptune FlashAttention reduction fusion、11/12 寫 HIERA kernel agent。
> 看出一條線沒？2026 下半年，compiler 的戰場已經從「寫 pass」橫向展開到
> **「pass × kernel × runtime × collective」四個維度一起**。單點技術強不夠，要能把這四個
> 維度在一題面試裡串起來。這篇論文剛好提供一個把四個維度全踩一遍的話題——SC26 的
> Best Paper finalist 身份加持下，下個月的 NeurIPS 2026、CGO 2026 submission deadline
> 都會引用它。不讀的話，compiler 社群這 3 個月在聊什麼你會跟不上。
