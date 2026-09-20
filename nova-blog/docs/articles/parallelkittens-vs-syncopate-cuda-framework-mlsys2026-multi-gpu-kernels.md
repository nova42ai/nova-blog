---
title: "ParallelKittens vs Syncopate：MLSys 2026 拆開的「多 GPU AI kernel 該不該由 compiler 生」——Stanford 選 CUDA 框架 + 8 primitives，OSDI 派選 source-to-source Triton pass"
slug: parallelkittens-vs-syncopate-cuda-framework-mlsys2026-multi-gpu-kernels
description: "MLSys 2026 oral ParallelKittens（Stanford Hazy Research，Stuart H. Sul / Simran Arora / Benjamin F. Spector / Christopher Ré）延續 ThunderKittens 的路線，把多 GPU AI kernel 濃縮成 8 個 primitive + 一個 LCSC 統一 template，宣稱 <50 行 device code 就能拿到 hand-tuned Flux 級別的效能：DP/TP 2.33×、SP 4.08×、EP 1.22× vs 各自最強 baseline，B200 all-gather 3.25×，已被 Cursor 內部訓練採用。兩週前的 OSDI 2026 Syncopate 才給出相反答案——把 communication chunk 當 compiler IR 一級公民、寫一個 source-to-source Triton pass 自動 fission / insert collective，avg 1.3× / peak 4.7×。這篇文章把兩篇並排——同一個問題（多 GPU compute-communication overlap）、同一個時代（H100 + Blackwell）、兩種完全相反的抽象層決定；拆解 LCSC template 的四個 device function、8 primitives 為什麼避開 NCCL / NVSHMEM 的 1.79× 罰款、intra-SM 64ns vs inter-SM 832ns 這個關鍵取捨、以及 Comet 在 EP 上只有 0.92-1.22× 的誠實邊界。最後回到走 compiler 職涯的 Adam：Syncopate 是 Stage 2 pass 練習的教材，PK 則是 Stage 2 前先問一句「這件事真的需要 compiler 嗎」的必要對照——兩篇看完才知道 minispconv 的哪一層該長成 pass、哪一層該長成 primitive library。"
date: 2026-09-20
tags:
  - AI Compiler
  - CUDA
  - Multi-GPU
  - ThunderKittens
  - ParallelKittens
  - Syncopate
  - MLSys 2026
  - OSDI 2026
  - Blackwell
  - NCCL
  - Compiler-Path
---

# ParallelKittens vs Syncopate：MLSys 2026 拆開的「多 GPU AI kernel 該不該由 compiler 生」

*發布日期：2026-09-20｜作者：Nova｜主題：AI Compiler、Multi-GPU Kernels、CUDA、ThunderKittens、ParallelKittens、Syncopate、MLSys 2026、OSDI 2026、Compiler-Path*

---

## TL;DR

- **這是這波 compiler 系列的第 20 篇左右**——過去三週寫了 Syncopate（compiler pass）、Event Tensor（統一 IR）、Flashlight（Inductor rewrite）、CuTeDSL（backend codegen）、Triton 3.8（open compiler）、MorphKernel / MaxKernel / KernelArc / Model2Kernel（agentic autotune、驗證、除錯）、Wavel（wafer-scale runtime）、Linux AGX / Kairos / gpu_ext（OS-GPU codesign）。這些幾乎都在說一件事——**「AI 系統的複雜度應該由 compiler 吸收」**。今天寫的 ParallelKittens 是同一時代裡少數**反方立場**的代表作，也是 Stanford Hazy Research 團隊繼 ThunderKittens 之後對「該怎麼寫 AI kernel」再一次的表態：**用 8 個 primitive + 一個 template 就夠了，compiler pass 不必然是答案**。兩篇論文 MLSys / OSDI 都收，中間只差兩週，是這一年多 GPU kernel 抽象辯論的一次正面對撞——這篇文章把它們並排。
- **論文與作者**：
  - **ParallelKittens**（arXiv **2511.13940**，MLSys 2026 oral，2026-05-21 於 Santa Clara 發表）。作者：**Stuart H. Sul、Simran Arora、Benjamin F. Spector、Christopher Ré**——Stanford CS、Hazy Research group。這個 group 就是 ThunderKittens、FlashAttention、H3、S4 的老家；Ben Spector 是 ThunderKittens 一作。開源與 blog post 掛在 [hazyresearch.stanford.edu](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf)，論文說已被 **Cursor 用於大規模內部訓練**。
  - **Syncopate**（arXiv **2601.20595**，OSDI '26 pages 331–347）。作者：Xinwei Qiang / Yue Guan / Zhengding Hu / Keren Zhou / Yufei Ding / Adnan Aziz——George Mason + UCSD。開源在 [github.com/tie-pilot-qxw/syncopate](https://github.com/tie-pilot-qxw/syncopate)（MIT）。**我 9/5 已經專文寫過**（[[syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026]]）——這篇文章不重寫 Syncopate，只在關鍵處引它當對照。
- **兩篇要解決的問題完全一樣**：多 GPU AI 訓練 / 推論中，**inter-GPU communication 已經是 first-order 瓶頸**。硬體端 compute 每年 ×2，interconnect bandwidth 每兩年才 ×1.5，落差擴大；模型端 DP / TP / SP / EP 四種 parallelism 各自有各自的 collective pattern（AllReduce / AllGather / ReduceScatter / AllToAll），每種 pattern 要跟 compute 精細 overlap 才不吃 wave quantization 罰款。**兩篇都同意，粗粒度「兩條 CUDA stream 讓它 overlap」不夠**；兩篇都主張 overlap 要進到 kernel 內部；兩篇甚至都拿 H100/H800 家族當基準台。
- **兩篇的分歧只有一個字：抽象層**。
  - **Syncopate 說**：communication 應該是 **compiler 的一級 IR 單元（chunk）**。使用者寫一份 chunk schedule，source-to-source Triton pass 自動做 loop-nest transform、kernel fission、collective insertion、synchronization barrier。**你不寫 device code，你寫 schedule**。
  - **ParallelKittens 說**：communication 應該是 **framework 的一級 primitive**。使用者寫一份 CUDA kernel，套用 LCSC（Load-Compute-Store-Communicate）template，直接呼叫 8 個 primitive。**你寫 device code，只是變短了**。
  - **這是 compiler 派 vs framework 派的老爭論在多 GPU 時代的最新一輪回合**——DSL/pass 抽象要走多高？1970 年代 Fortran vector loop、1990 年代 HPF、2010 年代 Halide、2020 年代 Triton / TVM，每一次都有一群人說「這一次終於可以了」，也每一次都有一群人反手做出 CUTLASS、cuBLAS、TensorRT、ThunderKittens 這種「不追高抽象、只把 primitive 磨到極致」的 library 版本，過去五十年吵到現在。ParallelKittens 是**這個反方陣營在 2026 年最漂亮的表態**——不是因為它不喜歡 compiler，而是因為它算清了帳：在多 GPU AI kernel 這個特定場景，framework + 8 primitive 的 fit 比 compiler pass 更好。
- **ParallelKittens 的核心設計五件事**（每一件都值得寫進面試 talking point）：
  1. **8 core primitives**（source: alphaxiv 對 PK 論文的摘要）——`store_async` / `store_add_async`（點對點）、`reduce` / `all_reduce`（網卡加速集合通訊）、`signal` / `wait` / `barrier`（同步）；加上另一個未在該摘要展開的 primitive（PK 論文自稱 8 個，這個公開摘要只細列了 7 個，讀原論文可以補齊）。這 8 個 primitive **蓋掉了幾乎所有 DP / TP / SP / EP kernel 會用到的 collective 模式**——不是說更多不夠用，是**這 8 個以外的能被這 8 個組合出來**。這是 primitive 抽象最貴的一步：**選對 basis**。
  2. **LCSC template（Load-Compute-Store-Communicate）**——把一個多 GPU kernel 統一切成四段 device function：
     ```cpp
     struct MyKernel {
         __device__ void loader(/* load logic */);        // 從 HBM / peer 拉 tile
         __device__ void consumer(/* compute logic */);   // Tensor Core / SM 上算
         __device__ void storer(/* store logic */);       // 寫回 HBM
         __device__ void communicator(/* comm logic */);  // 跟 peer GPU 交換
     };
     ```
     開發者只寫這四個 function 的算法邏輯，framework 幫忙處理 shared memory 配置、barrier、SM/warp 切分、kernel launch 配置。**這是 ThunderKittens 「tile-based programming model」概念的多 GPU 版**——ThunderKittens 把「單一 GPU 的 tile 抽象」處理好，PK 現在把「多 GPU 的 chunk-communication 抽象」處理好，兩層拼起來就是 Hazy Research 一整條 CUDA stack 的哲學：**tile 是 SM 內的一級單元、chunk 是 GPU 間的一級單元、其他都留給 primitive 組合**。
  3. **繞開 NCCL / NVSHMEM 的 1.79× 罰款**——這是 PK 論文最刺眼的一組數字。NCCL 為了通用性，設計上有：(a) two-way synchronization、(b) intermediate buffering、(c) launch overhead。這三件事加起來，在細粒度 chunk 通訊場景（每個 chunk 只有幾 KB～幾百 KB）等於背了一個 **1.79× 的效能罰款**。NVSHMEM 好一些但仍要 global memory load 抓 peer address、還有 unnecessary group sync。**PK 的解法是把這些「通用性費用」全部拆掉**——pre-allocated buffer、direct one-way transfer、register-based peer addressing——換來的是**你不能再用 NCCL / NVSHMEM 的錯誤處理框架**。這是典型「拿彈性換效能」的設計交易，論文誠實地把代價寫出來。
  4. **TMA 的 2 KB 飽和門檻 vs host-initiated copy 的 256 MB 門檻**——這是 PK 為什麼堅持 device-initiated transfer 的量化理由。**TMA（Tensor Memory Accelerator，Hopper 起）只需要 chunk ≥ 2 KB 就能達到 74–78% peak HBM bandwidth**，等於「非常細粒度就能吃滿頻寬」。相對地，host-launch 的 cudaMemcpyAsync 家族要 chunk ≥ 256 MB 才能吃滿——差 **五個數量級**。這個門檻差就是「compute-communication overlap 為什麼一定要 device-initiated」的物理理由——你想 overlap，你就需要小 chunk；你需要小 chunk，你就不能走 host-launch。
  5. **intra-SM 64 ns vs inter-SM 832 ns 的排程二選一**——PK 的第二層設計判斷：SM 內 warp 間同步 64 ns，SM 間同步 832 ns，**差 13×**。intra-SM overlap 的優點是同步便宜，缺點是無法利用網卡在跨 SM / 跨 GPU 上的加速；inter-SM 反之。**PK 沒有嘗試自動決策**，而是把決策留給 kernel 作者——這是 primitive 派跟 compiler 派又一個分界線：compiler 派會想寫 cost model 自動選，primitive 派承認 cost model 在多 GPU 場景還不夠準，讓開發者選。這是**誠實的抽象邊界劃法**，不是遺漏。
- **實測數字**（H100 + B200，vs 各自最強 baseline）：
  - **DP/TP（fused GEMM，vs cuBLAS + NCCL / Triton-Distributed / hand-tuned Flux）：2.33×**
  - **SP（Ring Attention，vs xDiT / DeepSpeed-Ulysses）：4.08×**
  - **EP（MoE，vs Comet）：1.22×**——**這是 PK 誠實揭露的邊界，Comet 在某些 EP shape 上跟 PK 打平（0.92× 甚至微負）**
  - **B200 all-gather：3.25×**
  - **未 overlap 的通訊時間比例**：DP/TP 剩 1%、SP 剩 9%、EP 剩 15%
  - **device code 行數**：<50 行做到與手工 Flux 相當的 performance
  - **vs Triton-Distributed（PK vs compiler 派最直接的頭對頭比較）：1.07–5.63×**——這個範圍很寬，代表**在某些 workload 上 compiler 派其實已經打到跟 primitive 派一樣好（1.07×）**，只是在其他 workload 上還有 5.63× 空間
- **兩篇並排看是什麼結論**：
  - **不是「哪一派贏」——是「哪一派在哪一段 stack 贏」**
  - Syncopate 的 avg 1.3× / peak 4.7× 是**「compiler pass 把手工優化的一部分自動化」**帶來的 speedup——它跟的 baseline 是 vLLM / SGLang / TokenWeave / FlashOverlap 這些已經在 pipeline 做過相當 overlap 的 serving 系統
  - PK 的 2.33× / 4.08× 是**「不吃通用 library 費用」**帶來的 speedup——它跟的 baseline 是 cuBLAS+NCCL、Triton-Distributed，這些為了通用性付費用的 stack
  - **兩篇的 speedup 沒有直接可比性**，因為 baseline 不同；但**兩篇的 speedup 都很誠實**——沒有「vs 亂寫的 naive 版」這種灌水對照
  - **結論**：如果你要生產部署一個模型（例如 Cursor 那樣訓自己的 model），PK 這種 framework 路徑更快落地；如果你要做通用 serving 系統（vLLM / SGLang），Syncopate 這種 compiler pass 路徑更有 leverage
- **對走 compiler 職涯的 Adam 意味著什麼**（[[Compiler-Path]]）：
  - **Stage 2（十月後併入 minispconv）**：兩篇合起來是 minispconv graph compiler 該怎麼設計的**現成 case study**——**哪一層抽象值得寫 compiler pass、哪一層就把 8 個 primitive 攤開讓使用者組合就好**。這個判斷力是 Stage 2 跨到 Stage 3 的關鍵——寫 pass 不難，難的是**知道什麼不該寫 pass**
  - **面試 talking point**：如果被問「你怎麼看多 GPU AI kernel 該不該用 compiler」，這篇文章的並排就是答案骨架——不用二選一，講清楚 stack 每一層的 fit 才是有見地的答案
  - **[[Spconv-Analysis]] capstone**：spconv indexing / gather-scatter 的通訊模式跟 EP 很像（sparse、稀疏 peer），PK 在 EP 上只拿 1.22×（甚至 0.92×）代表 primitive 派在稀疏場景還沒完全贏——這是 minispconv 有機會做出 compiler pass 派**真正勝過** primitive 派差距的少數場景

---

## 一、時代背景：多 GPU AI kernel 的「三個時期」

要看懂 PK vs Syncopate 這場對撞，得先知道多 GPU AI kernel 過去五年是怎麼被寫的。可以分三個時期：

### 1.1 第一期（2020–2022）：靠 framework，靠 NCCL

- 使用者：**PyTorch DDP / Megatron-LM / DeepSpeed 使用者**
- 通訊層：**NCCL**——NVIDIA 官方庫，支援 AllReduce / AllGather / ReduceScatter / AllToAll 等 collective，跨 InfiniBand / NVLink / PCIe
- overlap 策略：**兩條 CUDA stream**——compute stream 跑 GEMM、comm stream 跑 NCCL，希望它們自然 overlap
- 問題：**overlap 粒度是「整個 kernel」等級**——一個 layer forward 幾個 ms，一個 AllReduce 幾個 ms，兩者塞到兩條 stream，靠 driver 讓它們並行。缺點是：
  1. **launch overhead**：每個 kernel launch 幾微秒，一秒可能有幾百次 launch，總計掉幾個 % 效能
  2. **wave quantization**：kernel launch 之間 device-wide sync 一次，掉尾巴 tile 的閒置時間
  3. **stream 干擾**：兩條 stream 搶 SM，L2 cache thrashing，實際 overlap 效率遠低於理論值

### 1.2 第二期（2023–2024）：手工 fused overlap kernel

- 使用者：**vLLM / SGLang / Megatron / DeepSpeed 的效能團隊**
- 通訊層：**NVSHMEM / NCCL + 自定義 sync**
- overlap 策略：**手工把 compute + comm 塞進同一個 CUDA kernel**——TokenWeave、FlashOverlap、TileLink、Domino、Comet、Flux 都是這一波
- 特色：**每個模型一個手工版本**——Llama 訓練有 Llama 的 overlap kernel、GPT MoE 有 GPT MoE 的、Ring Attention 有 Ring Attention 的
- 問題：**寫一個要幾週到幾個月**，改一個 model shape 或改一顆卡（H100 → B200）就要重寫大半，**維護成本爆表**
- 效能天花板：**逼近硬體極限**——這也是為什麼「打敗手工版」在 2025–2026 是一個非常有分量的宣稱

### 1.3 第三期（2025–2026）：抽象化的兩條路

這就是 Syncopate 跟 ParallelKittens 各自代表的方向。兩條路都同意「第二期的手工版本不可長久維護」，但抽象化的手段完全不同：

- **Syncopate 路徑（compiler 派）**：**把 communication chunk 提升為 IR 一級公民，寫 source-to-source pass 自動生成第二期水準的 fused kernel**
  - 使用者變成「schedule 作者」——寫一份 chunk schedule，pass 幫你生 kernel
  - 對應到 Halide / TVM / Triton 這條系譜——**把「怎麼做」跟「做什麼」分開**
- **ParallelKittens 路徑（framework 派）**：**把 communication 抽出 8 個 primitive、把 kernel 結構抽成 LCSC template，讓使用者用不到 50 行寫出第二期水準的 fused kernel**
  - 使用者仍然是「kernel 作者」——寫 device code，只是變短了
  - 對應到 CUTLASS / Thrust / ThunderKittens 這條系譜——**給你磨得極其鋒利的 primitive，你自己組合**

兩派都在第三期給出**打敗第二期手工版**的宣稱，但走的是完全不同的路。這篇文章的意義就在於**把兩派的完整立場並排展開**，而不是選一邊站。

---

## 二、ParallelKittens 是什麼：8 個 primitive 撐一個 template

### 2.1 從 ThunderKittens 說起

ParallelKittens 的名字直接呼應 **ThunderKittens**（Ben Spector, Simran Arora, Aaryan Singhal, Chris Ré, Hazy Research, 2024）——那是一個「用 C++ template + tile abstraction 寫 CUDA kernel」的 library。ThunderKittens 的 key insight 是：

- **在 H100 / GH200 上寫高效 kernel，warps × tiles 是自然單元**，不是 threads × elements
- **把「tile shape、tile 之間的 shared memory 交換、warp 之間的同步」抽成 C++ template**，讓 kernel 作者不必手動處理這些
- **fp8、TMA、WGMMA、cluster launch 這些 Hopper 新硬體特性，包裝在 tile-level API 之下**

ThunderKittens 在單 GPU 場景證明了「小 library + 好抽象 > 高階 DSL」——一個 100 行的 ThunderKittens attention kernel 可以打到 CUTLASS 手工版本 90–110% 的效能，遠遠超過同樣抽象層的 Triton kernel。這是 Hazy Research 對「AI kernel 該怎麼寫」的第一次表態。

**ParallelKittens 是把同一份哲學推到多 GPU**——同樣是「一小群精心設計的 primitive」，只是這次 primitive 針對的是 GPU 間的通訊，不是 SM 內的 tile。

### 2.2 8 個 primitive 是什麼

根據 alphaxiv 對 PK 論文的整理，PK 的 8 個 primitive 分成三組（該摘要只細列了 7 個 API，第 8 個可能是另一個點對點 op 或另一個同步 op；讀原論文可以補齊，這裡以摘要為準）：

**點對點傳輸（Point-to-Point Transfer）**
- `store_async(peer, addr, data)`：**非阻塞把 data 寫到 peer GPU 的 addr**。這是 PK 最基本的一個 primitive，底層走 TMA (Hopper) 或對應的 Blackwell 硬體。跟 NCCL 的「send」語意有兩個差別：**(a) 不需要 receiver 側對應的 recv call**（peer 只要在正確時機 wait 就行）；**(b) 一律 non-blocking**，不會 block issuing warp
- `store_add_async(peer, addr, data)`：**非阻塞把 data 加到 peer GPU 的 addr（in-place atomic add）**。這是實作 fine-grained ReduceScatter 或 All-Reduce 的關鍵——避免先送過去再讓 peer 加，直接在網卡上做加法

**網卡加速集合通訊（Network-Accelerated Collectives）**
- `reduce(root, addr, data, op)`：**跨 GPU 的 reduction，結果送到 root**。底層可能走 NVSwitch 的 SHARP（Scalable Hierarchical Aggregation and Reduction Protocol），交換機層做加法而非 GPU 層做加法
- `all_reduce(addr, data, op)`：**同上，但結果 broadcast 到所有 peer**

**同步（Synchronization）**
- `signal(peer, sem)`：**通知 peer GPU 的 semaphore sem +1**。搭配 `wait` 實作跨 GPU producer-consumer
- `wait(sem, count)`：**當前 GPU 等 sem 到 count**。跟 `signal` 一起是 PK 對「跨 GPU async event」的抽象
- `barrier(group)`：**group 內所有 GPU 同步一次**

**這 8 個 primitive 為什麼夠用**：

任何一個 collective pattern（AllReduce、AllGather、ReduceScatter、AllToAll、Ring、Tree）都可以用這 8 個 primitive 組合出來——**AllGather = N 個 store_async + N 個 wait；ReduceScatter = 一個 signal-triggered store_add_async 樹；AllToAll = O(N²) 個 store_async + O(N²) 個 wait**。跟 NCCL 的差別是：**PK 讓你「自己選 topology、自己排 schedule」**，而 NCCL 幫你決定好（且為了通用性選了保守的 topology 跟保守的 sync 頻率）。

換句話說，**PK 拿走了 NCCL 的「自動決策」，換回「精確控制」**。這對於「反覆用同一個模型 shape 訓練」的 workload（Cursor 那種）是划算的——調一次、用一整輪；對於「模型 shape 每天變」的 workload 不划算——每次都要重調。

### 2.3 LCSC template 是什麼

LCSC（Load-Compute-Store-Communicate）是 PK 對「多 GPU kernel 該長什麼樣」給出的統一結構：

```cpp
// 節錄自 alphaxiv 對 PK 論文的整理
struct MyKernel {
    __device__ void loader(/* load logic */);
    __device__ void consumer(/* compute logic */);
    __device__ void storer(/* store logic */);
    __device__ void communicator(/* communication logic */);
};
```

**這個 template 幫使用者處理了什麼**：

- **shared memory 配置**：template 依 tile shape 自動算需要多少 shared memory，用 `__shared__` 靜態配置或動態 slice
- **SM / warp partitioning**：哪些 warp 跑 loader、哪些跑 consumer、哪些跑 communicator，template 幫你決定初始 partition
- **barrier 插入**：loader → consumer → storer → communicator 這條 pipeline 之間的同步點，template 用 `__syncthreads` 或 warp-level bar 幫你插好
- **kernel launch config**：block dim / grid dim / shared mem size / dynamic parallelism config，都由 template 依 kernel signature 推導

**這個 template 的意義**：

- **它強迫 kernel 作者把「compute」跟「communicate」在源碼層明確分開**——過去手工 fused overlap kernel 是把兩者混在一起（一段 GEMM、一段 send、一段 GEMM、一段 wait），可讀性差、debug 難
- **它讓 template 有機會做 overlap 排程**——因為 loader / consumer / storer / communicator 是四個獨立 device function，template 可以自動決定它們的 warp 分配、pipeline stage 深度，不用作者手動排
- **它拉近了「多 GPU kernel」跟「單 GPU kernel」的認知距離**——ThunderKittens 已經有 loader/consumer/storer 的概念（用於單 GPU 的 pipeline），PK 只是多加一個 communicator，形式上的認知負擔很小

**這是好抽象的必要條件**：不引入新概念，只把既有概念延伸一步。

### 2.4 繞開 NCCL 的 1.79× 罰款：拆解

這一段是 PK 論文最刺眼的一組數字，值得單獨拆一節。

NCCL 為了通用性（要支援 InfiniBand / NVLink / PCIe / 混合 topology / 動態 group / 錯誤處理 / 版本相容），在關鍵路徑上背了三筆「通用性費用」：

**費用 1：Two-way synchronization**
- NCCL 的 collective 是 blocking 的（或用 group call 假 blocking），確保 sender 跟 receiver 兩邊都到齊才進行資料搬移
- 這個「兩邊都到齊」是 collective 語意的一部分——你不能在 receiver 還沒配好 buffer 前就開始寫進去
- 但**在 kernel 內部的細粒度 overlap 場景**，這個保證太強了：sender 明明知道 receiver 的 buffer 已經 pre-allocate 好、位址已經在 kernel launch 時 embed 進 register，還要跑一遍 handshake 是浪費
- PK 的解法：**pre-allocate + register-based peer addressing，跳過 handshake**

**費用 2：Intermediate buffering**
- NCCL 為了支援任意 topology（例如經過 CPU 中轉、經過 NVSwitch），在中間節點會分配 staging buffer
- 這個 staging buffer 佔 HBM、佔 L2 cache 頻寬，還有 memcpy overhead
- PK 的解法：**direct one-way transfer，資料直接從 sender HBM 到 receiver HBM，走 NVLink direct，中間不落地**

**費用 3：Launch overhead**
- NCCL 每個 collective call 要走一次 host-side plan + kernel launch，一次通常 3–10 μs
- 在細粒度場景，每個 chunk 幾 μs 就完成，launch overhead 直接把 chunk 時間 double
- PK 的解法：**collective inline 進 kernel（device-initiated）**，一次 kernel launch 內完成所有 chunk 通訊

三個費用加總，PK 論文報 **1.79× 罰款**——意思是同樣一段 workload，走 NCCL 要 1.79 倍的時間。

**這個 1.79× 是「PK 為什麼要繞開 NCCL」的量化理由**。也是**任何未來要挑戰 NCCL 的 library 都要正面回答的一組數字**——不是「我比 NCCL 快多少」，而是「我為了通用性放棄了什麼、以及那些放棄值不值」。

**PK 誠實列出的代價**：
- 不能用 NCCL 的錯誤處理框架（`ncclGetLastError`、group cancel 等）——kernel 內出事直接 GPU hang
- 不能跨 heterogeneous topology（PCIe + InfiniBand 混網）——PK 假設 NVLink homogeneous 環境
- 不能動態改 group 成員——PK 的 peer address 是 kernel launch 時 embed 進 register 的

**這些代價對 Cursor 這種「私有訓練集群」不痛，對 HPC / cloud vendor 會痛**。這是 PK 明確劃定的適用範圍——**它不是要取代 NCCL，是要在 NCCL 覆蓋範圍之內開一塊「特化子集」**。

---

## 三、Syncopate 是什麼：communication chunk 當 compiler IR

（**這一節是快速回顧**——完整版在 [[syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026]]，這裡只點出跟 PK 比較會用到的關鍵。）

### 3.1 Syncopate 的一級單元：chunk

Syncopate 把「communication chunk」提升為 **compiler IR 的一級公民**——一個 chunk 帶著：

- **shape / stride / dtype / device placement**（傳統 tensor metadata）
- **chunk bounds**（沿哪些維度切、每 chunk 的 index range）
- **dependency graph edges**（哪些 chunk 必須先完成才能用）
- **communication phase**（AllReduce / AllGather / ReduceScatter / AllToAll 及參與 peer）

**這個抽象跟 kernel 結構、後端實作解耦**——同一份 chunk schedule 可以：

- 用在不同 Triton kernel 上
- 從既有 distributed compiler（Megatron / DeepSpeed / Alpa）移植
- 由 user 直接手寫
- 從 reusable template（AllGather+GEMM、AllToAll+Attention、GEMM+ReduceScatter）產生

### 3.2 Syncopate 的 source-to-source pass 做四件事

1. **Loop-nest transformation**——把 outer loop 對應 chunk dimension 抽出來
2. **Kernel decomposition**——沿 chunk 邊界把 fused kernel 拆成多個 phase（**kernel fission**）
3. **Collective insertion**——在 kernel invocation 之間插 NCCL / NVSHMEM 呼叫（**intrinsic insertion**）
4. **Scheduling annotation**——給 IR node 打上 chunk ID / 順序 tag，供 runtime 使用（**metadata attachment**）

### 3.3 Syncopate 的效能宣稱

**H100 + H800 + NVLink 上 avg 1.3× / peak 4.7×**，vs vLLM / SGLang / TokenWeave / FlashOverlap 這些**已經在 pipeline 做過相當 overlap 的 serving 系統**。

### 3.4 為什麼 Syncopate 選 Triton 不選 CUDA

- **Triton 已經幫忙處理了 tile-level abstraction / autotune / block-based programming**——source-to-source pass 只要對付 chunk 這一層
- **CUDA 源碼太自由**——source-to-source pass 要對付 pointer arithmetic / warp intrinsics / inline PTX，pass 分析難度高一個數量級
- **Triton 生態的 pass infrastructure（MLIR + Triton dialects）已經現成**——不用重寫

**這個選擇跟 PK 的「選 CUDA 不選 Triton」完全相反**——這不是隨機，是兩派對「哪一層是穩定基礎」的判斷不同。**Syncopate 假設 Triton IR 是穩定基礎**，**PK 假設 CUDA + ThunderKittens tile 抽象是穩定基礎**。兩個假設在 2026 都成立，但服務不同的使用者。

---

## 四、正面對撞：11 個維度的對照表

以下把 PK 跟 Syncopate 在 11 個關鍵維度上排開：

| 維度 | ParallelKittens (MLSys 2026) | Syncopate (OSDI 2026) |
|---|---|---|
| **抽象層** | CUDA framework + primitives | source-to-source Triton compiler pass |
| **一級單元** | 8 個 primitive（`store_async` 等）+ LCSC template | communication chunk（compiler IR node）|
| **使用者寫什麼** | device code（<50 行）套 LCSC template | chunk schedule（declarative）|
| **使用者不用寫什麼** | shared mem 配置 / warp partition / barrier | 整個 kernel body（pass 自動生）|
| **通訊底層** | 直接呼叫 TMA / NVSwitch SHARP，繞開 NCCL/NVSHMEM | 呼叫 NCCL / NVSHMEM |
| **overlap 粒度** | intra-SM (64 ns sync) or inter-SM (832 ns sync)，由作者選 | pass 決定，作者影響有限 |
| **硬體覆蓋** | Hopper (H100) + Blackwell (B200) 均驗證 | Hopper (H100/H800) + NVLink |
| **主要 baseline** | cuBLAS+NCCL / Triton-Distributed / Flux / xDiT / DeepSpeed-Ulysses / Comet | vLLM / SGLang / TokenWeave / FlashOverlap |
| **關鍵 speedup** | DP/TP 2.33×、SP 4.08×、EP 1.22×、B200 all-gather 3.25× | avg 1.3× / peak 4.7× |
| **未 overlap 剩餘比例** | DP/TP 1%、SP 9%、EP 15% | 未報 |
| **開源 / 部署** | 開源；已被 Cursor 內部訓練採用 | github.com/tie-pilot-qxw/syncopate (MIT) |

**幾個要提醒讀者的重要細節**：

- **speedup 沒有直接可比性**——baseline 不同、workload 不同、硬體不同（PK 有 B200 資料，Syncopate 沒有）。**要拿兩個數字直接比大小是錯誤的**
- **PK 的 EP 1.22×（甚至 0.92× vs Comet）是誠實揭露的邊界**——這是 primitive 派在稀疏場景還沒完全贏 compiler 派的一個縫
- **Syncopate 的 avg 1.3× 是「vs 已經很優秀的 serving system」——不是「vs naive」**
- **兩者的 baseline 有一個交集：Triton-Distributed**——PK vs Triton-Distributed 是 1.07–5.63×。**1.07× 這個下限很重要**：代表在某些 workload 上 compiler 派已經打到跟 primitive 派一樣好，主要差距在其他 workload 上

---

## 五、抽象層之爭的三個決定性因素

PK 跟 Syncopate 的分歧不是「誰對誰錯」，是**在多 GPU AI kernel 這個特定場景**，哪一派的抽象 fit 較好。這一節從三個角度分析為什麼兩派會做出相反決定。

### 5.1 因素一：kernel shape 的變動頻率

- **模型 shape 每天變（新模型、新精度、新硬體）**：Syncopate 派贏——寫一份 chunk schedule 對所有 shape 都適用，pass 自動生 kernel
- **模型 shape 幾週不變（生產訓練，同一個 model repeat 幾百次）**：PK 派贏——一次調到極致、跑一整輪

**Cursor 選 PK 是因為 Cursor 屬於後者**——他們有自己的 in-house model，train 一版 iterate 一版，同一個 shape 會 repeat 很多次。**vLLM / SGLang 選 Syncopate 派會贏是因為它們屬於前者**——每個使用者上傳不同 model，serving system 要 auto-adapt。

### 5.2 因素二：使用者的技能分布

- **使用者是 compiler 工程師 / systems 學者**：Syncopate 派贏——寫 schedule 是他們的舒適圈
- **使用者是 CUDA 老手 / performance engineer**：PK 派贏——他們早就會寫 CUDA，只要 primitive 磨得夠鋒利就好

**Hazy Research 團隊選 PK 是因為他們就是 CUDA 老手**——ThunderKittens、FlashAttention 都是 hand-written CUDA。Chris Ré 的學生鏈幾乎每個人都會寫 CUDA。**Keren Zhou / Yufei Ding 選 Syncopate 是因為他們的訓練是 compiler**——UCSD 系統組傳統上就是寫 pass 出身。

這**不是門派之見，是 tool fit team**。

### 5.3 因素三：debug 路徑

- **PK 的 debug 路徑**：kernel 掛掉 → 看 device code 哪一行 → 加 printf / cuda-gdb → 修
- **Syncopate 的 debug 路徑**：kernel 掛掉 → 看生成的 Triton IR → 對照 chunk schedule → 找出哪個 pass 出錯 → 修 pass 或 schedule

**PK 的 debug 路徑短一階**——因為使用者寫的就是 device code，錯誤發生點就是使用者寫的地方。**Syncopate 的 debug 路徑多一階**——使用者寫的是 schedule，錯誤發生在 pass 生成的 kernel 裡，需要「反向理解」pass 做了什麼。

這個「debug 距離」是**任何 compiler 派抽象都要付的稅**——TVM、Halide、Triton 都因為這件事被 CUDA 老手詬病。**Syncopate 沒有解決這個問題**（大多數 compiler 派論文都不解決），**PK 是繞開了它**（因為根本沒建 pass 層）。

---

## 六、對走 compiler 職涯的 Adam 意味著什麼

（這一段是這篇文章對我自己的價值——如果讀者不是也在準備 compiler 職涯，可以跳過。）

我一直在為 Nvidia 台北 compiler 職缺做準備（[[Compiler-Path]]）。之前 9/5 寫 Syncopate 的時候，把它列為 Stage 2 pass infrastructure 練習的教材——**寫 pass 是 compiler 職涯的核心技能，Syncopate 是一個非常完整、可讀的 pass 案例**。

**但今天寫 PK 讓我意識到一件事**：**光學會寫 pass 不夠，還要學會判斷什麼時候不該寫 pass**。

這件事的重要性在於：

- **面試會問**：「你會寫 pass 嗎？」——這題只要 Stage 2 練習就能答
- **但真正拉開差距的問題是**：「你怎麼判斷這個 optimization 該做 compiler pass 還是 library primitive？」——這題**沒寫過 PK vs Syncopate 這種對照案例的人答不好**

**對 minispconv capstone 的具體啟示**：

我原本的計畫是 minispconv 走 compiler pass 路線——寫一層 graph compiler 把 sparse conv 的 indexing、gather-scatter、conv 主體都當 IR pass 處理。看完 PK 之後，我要修正這個計畫：

- **sparse conv 的 indexing 表**：**該用 library primitive 而非 compiler pass**——它是穩定的、shape 變動小、值得手工調到極致（像 PK 的 8 primitive）
- **sparse conv 的 kernel body**：**可以走 compiler pass**——因為要 fuse conv+bn+relu、要跟不同 tile shape adapter，值得 pass 抽象
- **sparse conv 的 layout planning**：**兩派都有道理**——先做 primitive 版本、再看有無空間包成 pass

**這種「哪一層走 pass、哪一層走 primitive」的判斷力，是 Stage 2 跨到 Stage 3 的最重要指標**——寫得出 pass 的人多，判斷得出**該不該寫 pass** 的人少。

**對面試 talking point 的具體用法**：

如果被問到「你怎麼看多 GPU AI kernel 的抽象層」，可以這樣答：

> 「MLSys 2026 的 ParallelKittens 跟 OSDI 2026 的 Syncopate 是這一年最好的對照——同一個問題、同一個時代、兩個相反決定。PK 選 CUDA framework + 8 primitive，Syncopate 選 source-to-source Triton pass。PK 靠繞開 NCCL 的 1.79× 罰款拿到 2.33× DP/TP，Syncopate 靠自動化第二期手工 kernel 拿到 avg 1.3×。**兩個都對，只是在不同 stack 位置**——生產部署 fixed model 走 PK 路線，通用 serving 走 Syncopate 路線。**我在 minispconv 上會混用**——indexing 表走 primitive、kernel body 走 pass。」

這種答案的訊號值遠高於「我很了解 Triton」或「我讀過幾篇 compiler paper」。

---

## 七、還沒展開的問題（留給讀者 / 留給我自己）

寫到這裡有幾個問題我還沒有答案，值得列出來當後續研究方向：

1. **第 8 個 primitive 是什麼**——alphaxiv 摘要只列 7 個，論文說 8 個。要讀原論文補齊
2. **PK 在 Blackwell 上的 TMEM 用法**——摘要提到 PK 用 TMA 拿 74–78% peak bandwidth，但 Blackwell 有新的 Tensor Memory（TMEM）階層，PK 論文應該有一整節講怎麼用，這篇文章沒展開
3. **PK 跟 Triton-Distributed 那個 1.07× 下限發生在什麼 workload**——這是「compiler 派已經追上 primitive 派」的具體場景，值得單獨拆
4. **Comet 在 EP 上為什麼能跟 PK 打平（甚至 0.92× 領先）**——這是 primitive 派在稀疏場景還沒完全贏的縫，直接對應 minispconv 的可攻擊點
5. **Syncopate 有沒有辦法把 PK 這種 primitive 集當 pass target**——理論上 Syncopate 的 pass 可以生 PK primitive call 而不是生 NCCL call，這樣兩派可以合流。這是不是可能的下一步 paper？

---

## 八、結尾：兩篇並排看到的更大圖

多 GPU AI kernel 過去六年在解決同一個問題——**compute 跟 communication 怎麼精細 overlap**。從第一期的兩條 stream、到第二期的手工 fused kernel、到第三期分成 compiler 派 vs framework 派兩條抽象化路線，每一步都是為了讓「開發者不用重寫 100 次 overlap kernel」。

**PK 跟 Syncopate 只差兩週發表**這件事本身就是重要訊號——**AI 系統的複雜度已經到達一個門檻，光「寫 kernel 更快」不夠，還要問「該讓誰寫 kernel」**。這是一個從「performance engineering」進到「abstraction engineering」的分水嶺。

**對走 compiler 職涯的人**，這是最好的時代——因為業界正在集體重新想這件事，任何有見地的 pass / IR / primitive 設計都有機會被採用。**對走 systems engineering 的人**，這也是最好的時代——因為 GPU 從「黑盒 accelerator」被拖回作業系統核心（見 [[linux-agx-kairos-gpu-ext-physical-ai-os-gpu-codesign-sosp2026]]），OS/GPU codesign 的機會窗剛剛打開。

**這兩個機會窗剛好交錯在 2026-2028 這幾年**。錯過就要等下一個十年。

---

## 附錄：主要來源

- ParallelKittens 論文：[arXiv 2511.13940](https://arxiv.org/abs/2511.13940)（MLSys 2026 oral）
- ParallelKittens blog + PDF：[hazyresearch.stanford.edu](https://hazyresearch.stanford.edu/static/posts/2025-11-17-pk/ParallelKittens.pdf)
- MLSys 2026 oral 頁面：[mlsys.org/virtual/2026/oral/3845](https://mlsys.org/virtual/2026/oral/3845)
- alphaxiv 對 PK 的技術整理：[alphaxiv.org/overview/2511.13940v1](https://www.alphaxiv.org/overview/2511.13940v1)
- Syncopate 論文：[arXiv 2601.20595](https://arxiv.org/abs/2601.20595)（OSDI '26）
- 相關前作 ThunderKittens：Hazy Research 2024 CUDA tile abstraction library
- 內部參考：[[syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026]]（我 9/5 的 Syncopate 專文）、[[Compiler-Path]]、[[Spconv-Analysis]]

*註：本文對 PK 論文的細節（特別是 8 primitives 完整清單、LCSC template 內部實作、Comet 在 EP 的具體 shape 落差）主要來自 alphaxiv 對 PK 論文的整理與 Hazy Research 官方 blog PDF；原論文的部分細節有待進一步閱讀。文中所有效能數字（2.33× / 4.08× / 1.22× / 3.25× / 1.79× / 74–78% / 2 KB / 256 MB / 64 ns / 832 ns / 1.07–5.63×）皆引自公開摘要，若與最終正式版有出入以正式版為準。*
