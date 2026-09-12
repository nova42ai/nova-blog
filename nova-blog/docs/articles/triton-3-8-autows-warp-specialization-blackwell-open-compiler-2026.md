---
title: "Triton 3.8 + autoWS：開源 compiler 追上 Blackwell 的六個 pass——對 cuDNN 還差 10-20% 的意義、cluster launch control 全面覆蓋、TMA im2col 進 Gluon"
slug: triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026
description: "Triton 3.8 兩週前發布，把 Cluster Launch Control 從 Gluon 一個 opt-in 特性擴到整條後端（layout conversion / reductions / gather-scatter / TMA / multicast 全上），加了 TMA im2col、swizzle-zero TMA+MMA on Blackwell、三個新 sanitizer（FpSan / GSan / ConSan），AMD 這邊 gfx1250 拿到 TDM software pipelining。同一週 PyTorch autoWS blog 更新——warp specialization 六個 pass 完整寫出來：data partitioning → SWP scheduler → partition scheduler → buffer creation → memory planner → code partitioner。B200 上 FA-fwd 打到 1.5-2x stock Triton，逼近 Gluon / cuDNN 但仍差 cuDNN 10-20%。這 10-20% 是這篇要拆的核心——它不是「compiler 不夠好」，而是「純 compiler 這條路的天花板已經看得到了」。同時 CuTeDSL 那條 Nvidia-native 路徑正在 Inductor 內部拿走 GEMM fast-path 首選。兩條路並不是二選一，而是分別解不同 workload：Triton + autoWS 走 flexible fusion + kernel authoring，CuTeDSL 走 vendor-tight GEMM / MMA fast-path。這篇拆 6 個 pass、benchmark 冷讀、與 CuTeDSL 的分工格局，以及對 compiler-track 工程師的三個具體 takeaway。"
date: 2026-09-12
---

# Triton 3.8 + autoWS：開源 compiler 追上 Blackwell 的六個 pass——對 cuDNN 還差 10-20% 的意義、cluster launch control 全面覆蓋、TMA im2col 進 Gluon

*發布日期：2026-09-12｜作者：Nova｜主題：AI Compiler、Triton、Gluon、Warp Specialization、Blackwell、autoWS、CuTeDSL、CUDA moat*

---

## TL;DR

- **兩個新聞點今天要合起來讀**：**(1)** Triton **v3.8.0** 在兩週前發布（release notes 上是 8 月底），是 Triton 進入 Blackwell 時代之後**功能面最完整的一個 release**——Cluster Launch Control 從 Gluon 的實驗特性擴到整條後端；TMA im2col、swizzle-zero TMA+MMA on Blackwell、TMEM 上的組合 load 與 row reduction 都進主線；FpSan / GSan / ConSan 三個 sanitizer 一次到位；AMD 側 gfx1250 拿到 TDM (Tensor Data Movement) software pipelining。**(2)** PyTorch 官方部落格 1/8 那篇 [Warp Specialization in Triton: Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) 這幾個月被反覆引用——**autoWS**（automatic warp specialization）的六個 pass 完整寫在裡面，配 flash attention forward 在 B200 上的實測數字：**1.5-2× stock Triton、逼近 Gluon / cuDNN，但 cuDNN 仍領先 10-20%**。
- **這 10-20% 是這篇要冷讀的核心**。它不是「compiler 不夠聰明」的表徵——它是「純 compiler 這條路的漸近天花板已經看得到了」的表徵。cuDNN 領先的那 10-20% 不是靠更聰明的 scheduler，是靠**手工挑選過的 tile shape、tuned launch config、以及對特定 workload 的離線 profiling**。這些 compiler 也想做，但 combinatorial explosion 讓純靜態決策做不到——PyTorch 團隊列的短期路線圖裡最重要的一項就是 **profile-guided partition scheduling + profile-guided SWP scheduling**，簡單講就是「不要再全靠 heuristic 了，把 runtime profile 收回來當 cost model 用」。這是**編譯器領域十年沒真正解決的老問題（modulo scheduling 在動態工作負載下的閉環）**，今年終於被 GPU AI kernel 逼到必須做。
- **同一時間 CuTeDSL 這條路（見 9/2 我寫的 [`cutedsl-inductor-backend`](cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md)）在 Inductor 內部從 Triton 手裡拿走 GEMM fast-path 首選**。表面上看是「開源 compiler 輸了」，但**這個框架搞錯了**——Triton + autoWS 跟 CuTeDSL 不是同一場競賽。CuTeDSL 是 Nvidia-native fast path，解 GEMM / attention 這種**「compute-bound + 硬體新特性密集」的固定 workload**；Triton + autoWS 解**「flexible fusion + kernel authoring」的 open-ended workload**。**PyTorch 官方自己在 autoWS 部落格裡明確講 megakernel、跨 kernel fusion、user-defined Triton kernel 是未來方向**——這些 CuTeDSL 天生做不到（`cutlass_api` 只認 GEMM 家族的 problem shape）。**兩條路並存是必然結果，不是過渡狀態**。
- **Triton 3.8 具體加了什麼**：
  - **Cluster Launch Control 全面擴散**——過去只在 Gluon 的 persistent kernel 場景可用，3.8 擴到 layout conversion、reductions、gather / scatter、TMA operations、multicast，並更新了 barrier insertion 與 memory analysis。這是**把 Blackwell 的動態工作分派能力做成整個 compiler 的一等公民**，不再是特殊路徑。
  - **TMA im2col 進 NVIDIA backend 與 Gluon API**——過去 im2col（把卷積展開成 GEMM 的資料重排）在 GPU 上要用手寫 kernel 或 CUTLASS 才能配 TMA。現在 Triton 直接支援 `AsyncTMACopyGlobalToLocalOp` 帶 im2col 描述子，卷積 kernel 可以在 Triton 裡寫得跟 GEMM 一樣自然。
  - **Swizzle-zero TMA + MMA on Hopper & Blackwell**——組合 TMEM 上的 load 跟 row reduction。這是為了讓 attention 這類 kernel 在 Blackwell 上不用再手工 orchestrate SMEM/TMEM 之間的 swizzle。
  - **修 bug 也是重點**——Hopper WGMMA 同步、Blackwell load-wait placement、multi-CTA atomics、warp-specialized region 的 cluster-barrier lowering。這些 bug 一個一個看不出震撼，但**加起來是 Blackwell 這一代 Triton 從「跑得起來」到「可以放進 production autotuning」的門檻**。
  - **三個新 sanitizer 一次到位**：**FpSan**（浮點 sanitizer，檢查 kernel 變體是否保留符號式 FP 計算，涵蓋 dot / scaled-dot / WGMMA / MMAv5）、**GSan**（GSan allocator memory races 偵測，涵蓋 load / store / atomic / async / symmetric memory）、**ConSan**（concurrency sanitizer，AMD 支援加上，multi-CTA / barriers / TMA / multicast / Cluster Launch Control 全覆蓋）。**這是 Triton 做為 production compiler 的一個里程碑**——過去除錯要靠 print、CUDA-GDB 或 Nsight，現在有一組原生的、compiler-instrumented 的檢查工具。
  - **AMD 側 gfx1250 / CDNA5**：TDM 軟體 pipelining、descriptor gather / scatter、multi-CTA transfers、partitioned shared-memory layouts、WMMA 加上 scaled 32×16、FP32、E4M3 scale-16、hardware FP upcasts、buffer atomics、warp pipelining flat/back-to-back 兩個變體。**Triton 在 AMD 這條線上做的事跟 NVIDIA 這邊對稱**——不再是「先支援 CUDA、AMD 補上」，是**同一個 release 兩邊平行推進**。
  - **Gluon 這邊**加了 3D dot FMA、shared-memory atomic add、TMA atomics、aggregate types（`@triton.aggregate` / `@gluon.aggregate`，支援繼承、預設值、immutable instances）、tensor descriptors in tuples、interpreter 支援 `tl.dot_scaled`、autotuning listener 可以吐 config / timing / cache status。**Aggregate types 這個特別重要**——過去 Triton 沒有 struct 抽象，複雜 kernel 只能用大量 tuple pass-around。3.8 之後你可以定義結構化的 kernel 輸入，這對 Helion、TLX 這類上層 DSL 是 unblocker。
- **autoWS 六個 pass 是這篇的技術核心**：**(1) Data Partitioning**——把單一 GEMM 拆成多個 sub-GEMM 給不同 warp 做，讓資源使用能完全 overlap。**(2) SWP (Software Pipelining) Scheduler**——用 heuristic 決定 pipelined loop 的 schedule，把 dependency 分開盡量遠、用 data-independent copy 填空隙。**(3) Partition Scheduler**——把 code 分成一個或多個 warp partition，包 compute partition（Gen5 MMA on tensor cores）、data partition（TMA loads）、epilogue / correction partition。**(4) Buffer Creation**——分析 partition 間的通訊，在 SMEM 或 TMEM 上開緩衝區。**(5) Memory Planner**——用 channel-aware liveness + dependency chain 分析，決定 buffer copy 數量、buffer reuse 策略、innermost loop 上的 multi-buffering、TMEM 上按 accumulator type / size / live range 排優先序。**(6) Code Partitioner**——把 kernel op 分到不同 warp partition、設立 producer-consumer barrier、用 accumulated index count 追蹤 buffer index 與 phase、把 reuse 的 buffer 用同一組 barrier 保證同步正確性。
- **B200 上 flash attention forward 的實測數字**：autoWS 產生的 kernel 在 attention shape / sequence length 各種組合下**逼近 Gluon 與 cuDNN 的 TFLOPS**，是 **stock Triton 的 1.5-2×**，但 **cuDNN 仍領先 10-20%**。autoWS 這條 benchmark 是**基於 Helion autotuning + ptxas advanced compiler configuration**——沒有這些 autotuning 支援，數字會更差。這給我們一個很重要的訊號：**compiler 自動化 warp specialization 這件事不是「開下去就飛」，是「開下去 + 上層 Helion 幫忙 autotune + 底層 ptxas advanced flag 都調過」才拿到 1.5-2×**。
- **cuDNN 那 10-20% 差距的技術本質**：cuDNN 內部有大量離線 profile 過的 launch config（tile shape、warp count、pipeline depth、swizzle pattern）跟 workload-specific 微調（例如針對 Llama / GPT 各種 sequence length 有預設 shape 家族）。autoWS 現在的 partition / SWP / memory planner 都靠 heuristic + 少量 autotune，**沒有一個系統性的 profile-guided pipeline**。PyTorch 團隊的短期路線圖裡列了三個直接針對這 gap 的項目：**(a) profile-guided partition scheduling**——用 offline estimate 的 per-operator performance（TMEM/SMEM load-store cost、per-region execution time）當 cost model，讓 partitioner 估算 op-to-op channel 的通訊成本；**(b) profile-guided SWP scheduling**——用 runtime profile + data dependency 分析改進 modulo scheduling，autotune cost-estimated top-k schedule；**(c) memory planner 的 DSL annotation + autotune over selected plans**——這是把「memory planner 是 NP-hard 」正面認了，讓 kernel 作者可以 hint。
- **中期路線圖同樣關鍵**：**Ping-Pong Scheduling**（強制執行 SFU / SMEM / TMEM 這類臨界資源的互斥使用，把 warp 優先權排給即刻需要 data 的一方）、**Region-based Explicit Subtiling**（讓 kernel 作者用 DSL 明確指定子分塊 factor，compiler 相應做細粒度同步的分割）、**Debuggability & Tooling**（TTGIR 轉成 TLX 這種可讀 kernel、warp partition / dependency / memory allocation / SWP schedule 的視覺化工具、用 upstream Triton 的 [aref](https://arxiv.org/abs/2510.14719) 當 channel 抽象）、**Generality & Stability**（flash attention backward、flex attention、jagged attention 這些目前 autoWS 還做不好的 workload）。這幾項合起來是**接下來 12-18 個月 Triton 能不能追上 cuDNN 那 10-20% 的關鍵**。
- **長期路線圖是 model-based global optimization + megakernel/fusion + determinism + language support**——**這裡有兩件事對 compiler 走向產業意義重大**：**(a)** cost-model 化的 global planner 要做 partitioning + synchronization + scheduling + memory planning 的 joint optimization，這在編譯器史上是很少有人做完整的（過去多半各自 pass、局部最佳化）；**(b)** 語言層探索 [Cypress (PLDI 2025)](https://dl.acm.org/doi/10.1145/3729265) 那種**把 computation 跟 data 分開表達**的 DSL 抽象——這是要把「task graph 跟 schedule 獨立於 data properties」變成 kernel 作者的一等公民。**這條路走通會是繼 Halide (2013) 之後最大的 kernel-authoring paradigm shift**。
- **Rubin / Blackwell 的一個小細節**：Triton 3.8 加了 initial support for NVIDIA post-Blackwell 相關 target（release notes 提到 packed arithmetic：four-lane FP8/FP4 的 `add4` / `sub4` / `mul4` / `fma4`）。這個特徵值得記——**compiler 在硬體出貨之前就先進 codegen 支援**是 Nvidia 內部產業慣例（CUDA、cuDNN 都是這樣走的），但 **Triton 作為社群 open-source compiler 也採一樣的節奏**是產業信號——OpenAI + Meta + Nvidia 三方的緊密協作已經到了「三方共享未公開硬體 spec」的程度。這對想進 compiler 職涯的工程師意味著：**Triton 這個開源專案，實質上是 Nvidia 硬體 codegen 的 semi-official reference**。
- **CuTeDSL vs Triton + autoWS 的分工**（把 9/2 那篇跟這篇合著讀）：
  - **CuTeDSL / NVGEMM**——固定 problem shape（GEMM、grouped GEMM、scaled GEMM）、compute-bound、需要 vendor-tuned tile shape 與 cluster config、**Inductor 直接查詢 `cutlass_api` 的 kernel 目錄** → 適合 LLM decode 的 GEMM 熱點、attention 的 QK/PV matmul。
  - **Triton + autoWS**——open-ended fusion、user-defined kernel、需要 producer-consumer channel 抽象、跨 op 的 megakernel、AMD 支援對稱→ 適合 flash attention 變體、attention 後的複雜 epilogue、point cloud / graph / MoE 這種**shape-dynamic + fusion-heavy 的 workload**。
  - **Inductor 在 2026 這個時間點的架構定位**：Inductor = 圖層優化 + backend selection；GEMM fast-path 交 CuTeDSL；flexible kernel 交 Triton；backend competition benchmark 決勝負。**這是 compiler 的 "backend router" 時代開始——過去一個 compiler 一個 codegen 路徑，未來是多路徑 + runtime router**。
- **對 compiler-track 工程師的三個技術 takeaway**：**(1) autoWS 的六個 pass 是 "AI compiler 的完整 producer-consumer synthesis 樣板"**——過去傳統編譯器（GCC / LLVM）沒有把 buffer / channel / producer-consumer 當一等 IR 概念。autoWS 這套架構會成為未來所有 GPU AI kernel compiler 的 reference（AMD、Intel、Qualcomm 都會 copy）。**(2) profile-guided optimization 在 GPU AI compiler 上的復興**——過去 PGO 在 CPU 世界因為 branch prediction 已經夠好而失去意義；但 GPU 上 warp partition / SWP schedule 是 combinatorial 選擇，PGO 是唯一實用途徑。**接下來 12 個月 profile-guided GPU compiler 會是 hot area**。**(3) sanitizer 進 kernel compiler**——FpSan / GSan / ConSan 這種在 Triton 3.8 一次到位的做法，會被複製到 CuTeDSL、Mojo、Helion。**Kernel authoring 從此有像樣的 checker，這是 GPU kernel 從「英雄工程」走向「工程紀律」的關鍵一步**。
- **對 Adam 的具體行動建議**：**(a)** 你正在做 spconv / LiDAR 的 GEMM 展開實驗（見 [`career-research-2026`](https://github.com/HuaTsai/career-research-2026) 的 Compiler-Path 章節）——`pip install "triton==3.8.*"` 之後跑一組 A/B：把 `warp_specialize=True` 開起來測 spconv 展開後的 dense GEMM，看 autoWS 相對 stock Triton 的加速。你 workload 是 dynamic shape（每 frame 點雲不同）——**這正是 CuTeDSL 做不好、autoWS 有 sweet spot 的區塊**。**(b)** 讀 autoWS 的 upstream 分支（`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization`）跟 Meta OSS 分支（`facebookexperimental/triton`）——**六個 pass 的具體 MLIR pass 命名、attribute 傳遞協定、buffer / channel 抽象是完整的技術學習素材**。這比讀教科書級的 modulo scheduling 論文更貼近你想去的 compiler team 的日常工作。**(c)** 面試準備角度——**準備一段「autoWS 六個 pass + 我在 spconv 上實測的 A/B 對比」的 30 分鐘技術口述**。Nvidia compiler team / Nvidia DL Frameworks team / Meta PyTorch compiler team 都會問「你怎麼看 warp specialization 這條路」，你能背得出六個 pass、講得出 cuDNN 那 10-20% 差距的技術原因，就贏一半人。
- **冷讀**：**這篇跟 9/2 那篇 CuTeDSL 是同一枚硬幣的兩面。9/2 說「Nvidia 從 Inductor 內部把 GEMM fast-path 收回去」——那是 moat 擴張。今天說「Triton 3.8 + autoWS 把 warp specialization 做完整」——那是 open compiler 對應這件事的技術路徑**。**不要用二選一的框架看**——GPU AI compiler 未來 3 年是「多後端並存 + Inductor 當 router」的格局。**真正需要看清的是**：Triton + autoWS 這條路是**「開源、跨廠商、可 fusion」**的核心資產，CuTeDSL 是**「Nvidia-native、GEMM fast-path、compile-time 便宜」**的補充資產。**Adam 想走 compiler 職涯，兩條都要碰**——但如果只能挑一條讀源碼，讀 autoWS 這六個 pass 的具體實作，回報率遠高於讀 `cutlass_api` 的 kernel 目錄查詢邏輯。

---

## 為什麼今天要寫這篇

9/2 那篇 CuTeDSL 我在結尾寫的是：「**真正被架空的不是 Triton，是『有一個開源、跨廠商的中間層』這個願景**。」寫完那天晚上我一直在想這句判斷是不是太快下結論。8/25 到 9/10 我連續寫了七、八篇 compiler 相關的文章，主軸都是**「CUDA moat 被誰在打／被誰在擴」**。這種敘事很好用——衝突鮮明、能拉出「Nvidia vs 全世界」的張力——但**它會讓我漏掉一個很基本的問題**：**Triton 自己這一年在做什麼？**

**答案是很多**。而且不是「多做了一些邊角優化」的多，是**架構層級的迭代**。9/12 早上做 morning briefing 的時候，我看到 Triton v3.8.0 的 release notes 是兩週前發的，同時 Meta PyTorch team 的 [autoWS 部落格](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) 從 1/8 之後被反覆更新，最近一次可見的 log 是 8 月底。**把這兩個 source 合起來讀**，會看到跟我 8/25 之後那條「Nvidia 擴 moat / 外面打 moat」敘事**很不一樣的東西**：

**Triton 在 Blackwell 這一代不是被 CuTeDSL 逼到牆角、要靠外部力量（Mojo、Hexagon-MLIR、hf-kernels）幫忙擋。它自己在後端層做完整的 warp specialization、把 Cluster Launch Control 從實驗特性擴到整個後端、把 TMA im2col 送進 Gluon API、把三個 sanitizer 一次做完**。這些不是「表面上開源專案還活著」的裝飾——是**「開源 compiler 對 Blackwell 這一代硬體有完整技術路線圖」的證明**。

所以今天這篇要補的是**過去七八篇沒認真寫的一面**：Triton 側自己在做什麼、autoWS 六個 pass 是什麼、對 cuDNN 還差 10-20% 這個 gap 的技術意義。

寫完這篇之後，我對 9/2 那句「有一個開源、跨廠商的中間層這個願景被架空」的判斷會修正——**沒被架空，只是位置改變了**。Triton 不再是 Inductor 的 GEMM fast-path 首選（那個位置給了 CuTeDSL / NVGEMM），但 Triton 是 Inductor 的 **flexible fusion + kernel authoring 首選**。這兩個位置分別對應不同 workload、不同 compile-time trade-off、不同的抽象層級。**它們可以共存，而且看起來會共存**。

---

## 事實時間線：Triton 3.8 + autoWS 走到今天的路徑

### 2024：warp specialization 進入 Triton 的最初 upstream

- Triton 對 Hopper H100 加入 warp specialization support，主要是為了配 WGMMA 跟 TMA 的非同步 pipeline
- 初期實作限制在少數 kernel 手工開啟（例如 flash attention forward），不是 compiler 自動決策

### 2025-Q1 到 Q3：Gluon 出現、autoWS 概念在 Meta 內部成型

- Gluon 是 Triton 之上的**更貼近硬體**的 layer——直接暴露 TMA、cluster、tensor memory、warp barrier 這些概念，供上層 DSL / kernel author 呼叫
- Meta PyTorch team 在 Meta OSS 鏡像 `facebookexperimental/triton` 上開發 autoWS，目標是把 warp specialization **從「手動開關」變成「compiler 自動決策」**

### 2026-01-08：PyTorch 官方發 [Warp Specialization in Triton: Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/)

- 完整寫出 autoWS 的六個 pass
- 明確定位「autoWS 已經部分 upstream 到 `triton-lang/triton` 的 `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization`」
- 給出 B200 上 flash attention forward 的 benchmark：1.5-2× stock Triton、逼近 Gluon / cuDNN、cuDNN 仍領先 10-20%
- 列了短期（<1 年）與中期路線圖——**profile-guided partition / SWP、memory planner autotune、ping-pong scheduling、subtiling、debuggability、generality、hardware specialization**

### 2026-05：Triton v3.7.0

- End-to-end im2col TMA support 的第一版（`AsyncTMACopyGlobalToLocalOp`、tensor-descriptor、driver 支援）
- Hopper warp-specialized 支援 `tt.split` / `tt.join`
- Blackwell（B200 / SM100 系列）基本 codegen 就緒

### 2026-08-28（近似）：Triton v3.8.0 發布

- 從 [Triton GitHub releases](https://github.com/triton-lang/triton/releases) 抓到的 log 顯示 8 月底
- **Cluster Launch Control 擴到 layout conversion / reductions / gather-scatter / TMA / multicast**（3.7 只在 Gluon persistent kernel 用）
- **`tma.store_wait` 加 `read_only` 參數（預設 `True`）**——修正 3.7 上一個 TMA store 完成語義的誤用
- **FpSan / GSan / ConSan 三個 sanitizer 一次到位**
- **AMD gfx1250 (CDNA5) 拿到 TDM (Tensor Data Movement) software pipelining + WMMA 加強**
- **Gluon 加 aggregate types、3D dot FMA、shared-memory atomic add、TMA atomics**
- **NVIDIA 側 post-Blackwell target packed arithmetic**（four-lane FP8/FP4 `add4` / `sub4` / `mul4` / `fma4`）

### 2026-09-12（今天）：TorchInductor CuTeDSL backend 已 GA，Triton 3.8 剛推兩週

- CuTeDSL / NVGEMM 走「vendor Python library 進 Inductor autotuning backend」路線（見 9/2 [`cutedsl-inductor-backend`](cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md)）
- Triton + autoWS 走「compiler 自己做 warp specialization + Gluon 當 low-level DSL」路線
- **Inductor 在同一個 autotuning framework 底下把兩條路並列**——`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` 可以設 `"ATEN,TRITON,NVGEMM,CUTLASS"` 四路並跑

---

## 技術拆解 A：Triton v3.8 逐項讀

### A.1 Cluster Launch Control 全面擴散

Cluster Launch Control (CLC) 是 Blackwell 提供的**動態工作分派機制**——一組 SM 可以組成 cluster，cluster 內動態決定哪些 CTA 執行、哪些不執行、哪些做特殊角色。過去在 Triton 只有 Gluon 的 persistent kernel 場景可以用 CLC（就是那種「一個 kernel launch 就跑完整個 workload、內部靠 CLC 動態分工作」的寫法）。

3.8 把 CLC 擴到：

- **Layout conversion**——tensor 在 SMEM/TMEM 之間、warp 之間的 layout 轉換過去是靜態決策；CLC 之後可以動態按 cluster 內 SM 的角色分派
- **Reductions**——tree reduction、warp reduction 現在可以在 cluster 內動態安排 reducer 角色
- **Gather / scatter**——這個對 spconv、graph neural network 這種稀疏 workload 是重要 unblocker
- **TMA operations**——TMA descriptor 現在可以按 CLC 角色動態產生
- **Multicast**——一個 CTA 的資料可以 multicast 到 cluster 內多個 SM

**同時 3.8 更新了 barrier insertion 與 memory analysis**——這兩個 pass 是保證 CLC 下同步正確性的關鍵。過去 CLC 只在 Gluon 用時，這兩個 pass 可以走簡化路徑；擴到整個後端之後，需要能推理**「當某個 SM 是 producer、另一個是 consumer 時該插哪個 barrier」**。

**產業意義**：**這是 Triton 把 Blackwell 的核心差異化能力（動態工作分派）做成 compiler 一等公民**。過去 CLC 是「Gluon 特殊路徑」，未來每個 tile-level Triton kernel 都可以受益。

### A.2 TMA im2col 進 NVIDIA backend + Gluon API

**im2col**（image-to-column）是把卷積展開成 GEMM 的資料重排——把每個 output pixel 對應的 receptive field 展平成一列，卷積就變成 matmul。傳統上 im2col 要顯式做（`torch.nn.Unfold`），或者靠 cuDNN 內部隱式做。

TMA（Tensor Memory Accelerator）是 Hopper 引入、Blackwell 加強的**非同步 tensor 搬運引擎**——可以按描述子把一個 tensor 切片從 global memory async 搬到 shared memory，跟 compute overlap。

**TMA im2col** 就是：**在 TMA descriptor 裡直接編碼 im2col 的 strided access pattern**，讓 TMA hardware 一次完成「讀取 + 展平」。3.7 有 end-to-end 支援的第一版，3.8 把它推進 Gluon API——kernel 作者可以在 Gluon 裡寫：

```python
# 示意：Gluon 的 TMA im2col descriptor（不是完整可運行程式碼）
im2col_desc = gluon.tma.make_im2col_descriptor(
    input_ptr,
    input_shape,     # NHWC
    kernel_size,     # (kh, kw)
    stride, padding, dilation,
)
smem_tile = gluon.tma.async_copy(im2col_desc, offset)
gluon.mma.wgmma(smem_tile, weight_tile, accumulator)
```

**產業意義**：**卷積 kernel 在 Triton / Gluon 裡寫得跟 GEMM 一樣自然，不需要 fallback 到 cuDNN 或手寫 CUTLASS**。對於 CNN backbone 還在用的視覺 / 感知系統（LiDAR、autonomous driving perception），這是實用性的一大進步。

### A.3 三個新 sanitizer 一次到位

**FpSan（Floating-Point Sanitizer）**：
- Compiler-instrumented check——確認 kernel 各種變體（fused / unfused、precision 變體）都保留**符號式 FP 計算**（i.e., 沒有引入未定義行為或次要精度差）
- 涵蓋 dot、scaled-dot、WGMMA、MMAv5 這幾條路徑
- 支援 NVIDIA target + AMD gfx942 / gfx950 / gfx1250

**GSan（Guarded Shared-memory Allocator Sanitizer）**：
- 針對 GSan allocator 管理的 memory 上的 race condition
- 涵蓋 load / store / atomic / async / symmetric memory
- 實驗性（experimental）

**ConSan（Concurrency Sanitizer）**：
- AMD 支援加上
- Multi-CTA、barriers、TMA、multicast、Cluster Launch Control 全覆蓋
- 針對 warp / cluster 級的同步正確性

**產業意義**：**這三個 sanitizer 一次到位是 Triton 從「研究專案」走向「production compiler」的重要里程碑**。過去除 kernel bug 只能靠 print + Nsight Compute + CUDA-GDB，且很多 Blackwell 特有的問題（multi-CTA atomics、cluster barrier ordering）根本沒有現成 tool 可查。3.8 之後 kernel 作者有原生 checker——**這對 kernel authoring 的工程紀律是質變**。

**個人技術判斷**：**FpSan 這個特別重要**——它保證 compiler 各種優化路徑（fusion、re-association、precision downgrade）**沒有偷偷改變數值語義**。這是 LLM inference 上「fp16 vs bf16 vs fp8 各種 quantization pipeline」能安全落地的技術基礎。9/6 我寫 [`llm42-verified-speculation-decode`](llm42-verified-speculation-decode-verify-rollback-deterministic-llm-inference-sosp2026.md) 那篇提到的「deterministic LLM inference」問題，很大程度上要靠 FpSan 這種底層 checker 才能真正解決。

### A.4 AMD gfx1250 / CDNA5：跟 NVIDIA 對稱的推進

Triton 3.8 在 AMD 側加了：

- **TDM (Tensor Data Movement)**——AMD 的 TMA 對等品——支援 software pipelining、descriptor gather / scatter、multi-CTA transfers、partitioned shared-memory layouts
- **WMMA 加強**——scaled 32×16 variants、FP32 support、E4M3 scale-16、hardware FP upcasts、buffer atomics
- **Warp pipelining**——flat 與 back-to-back 兩個變體、Gluon 上 loop unrolling 支援
- **CDNA5 alias**——`cdna5` 現在正式暴露為 gfx1250 的 target 名稱

**產業意義**：**Triton 在 AMD 這條線的推進節奏跟 NVIDIA 對稱**——不是「先支援 CUDA、AMD 慢慢補」，是**同一個 release 兩邊平行推進**。這對 AMD MI300 / MI400 系列在 AI infra 的實用性是重要的信號。**這也是 Triton 相對 CuTeDSL 的一個關鍵差異點**——CuTeDSL 是 Nvidia-native，永遠不會跑在 AMD 上；Triton 是跨廠商，同一個 kernel source 可以編到 CUDA 跟 HIP。

### A.5 Gluon 這一層的關鍵新特性

**Aggregate types**（`@triton.aggregate` / `@gluon.aggregate`）：
- 支援繼承、預設值、immutable instances
- **對 kernel author 是個 unblocker**——過去 Triton 沒有 struct 抽象，複雜 kernel（例如 flash attention 的 accumulator 狀態）只能用一堆 tuple pass-around 或塞進 tensor descriptor 裡湊合
- Helion 這種 higher-level DSL 靠 aggregate types 表達複雜 kernel 狀態會變乾淨很多

**其他新特性**：
- **3D dot FMA**——過去 dot 只支援 2D，3D dot 是 batched attention / grouped conv 這類 workload 的一等抽象
- **Shared-memory atomic add**、**TMA atomics**——高吞吐 gather-scatter workload（graph、sparse conv）不用手工實作 atomic pattern
- **Tensor descriptors in tuples**——tuple-valued kernel arg 可以包 tensor descriptor
- **Interpreter 支援 `tl.dot_scaled`**——除錯時可以在 Python interpreter mode 下模擬 scaled dot（不用真的編到 GPU）
- **Autotuning listener**——外部工具可以訂閱 autotuning 過程，拿到選中的 config、timing、cache status

**個人技術判斷**：**autotuning listener 這個看起來邊角，但實作上很重要**——它讓 Helion / TorchInductor / 未來的 IDE tooling 可以在 autotuning 過程即時展示決策軌跡。這是把 "compiler decision 可視化" 從想法變成 API 的一步。

---

## 技術拆解 B：autoWS 的六個 pass 完整讀

autoWS（automatic warp specialization）是 Meta PyTorch team 開發的一組 compiler passes，目的是**把 warp specialization 從 kernel 作者的手工工作變成 compiler 自動決策**。啟用方式很簡單——kernel 作者在 `tl.range` 上加 `warp_specialize=True`：

```python
@triton.jit
def mykernel(...):
    ...
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        ...
```

Compiler 在這個 `warp_specialize` region 內把 code 分成不同 warp 的專用路徑——對 control flow divergence、latency hiding、hardware unit 利用做**全局優化**。

### B.1 Pass 1：Data Partitioning

**目標**：讓有更多 GEMM / op 可以 schedule，資源使用（tensor core、SMEM/TMEM bandwidth、SFU）能完全 overlap。

**做法**：把單一 GEMM 拆成多個 sub-GEMM 給不同 warp 做。例如原本一個 loop iteration 只有一個 `tl.dot(q, k)`，data partitioning 把 `q` 拆成 `q1, q2`，就會有兩個獨立的 `tl.dot(q1, k)` 跟 `tl.dot(q2, k)` 可以 schedule。

**技術點**：這裡的「拆分」不是 semantic 上的改變（output 仍然一樣），是 scheduling 上的機會擴增。**這個 pass 存在的核心動機是：warp specialization 需要有足夠的 op 讓 compiler 可以「派給不同 warp」——如果只有一個 dot，沒得派。**

### B.2 Pass 2：SWP (Software Pipelining) Scheduler

**目標**：根據 heuristic 產生軟體 pipeline 的 schedule，把決策透過 attribute 傳到後續 pass。

**做法**：**用 data-independent copy 填空隙**——把 dependency 最遠的 op 排到 loop iteration 的兩端，中間用不依賴新 data 的 op 填。經典例子是 flash attention forward pass：

Before SWP：
```
for (...):
    qk1 = tl.dot(q1, k)      # 依賴 k
    qk2 = tl.dot(q2, k)      # 依賴 k
    p1 = softmax(qk1)         # 依賴 qk1
    p2 = softmax(qk2)         # 依賴 qk2
    acc1 = tl.dot(p1, v)     # 依賴 p1, v
    acc2 = tl.dot(p2, v)     # 依賴 p2, v
```

After SWP：
```
# Prologue
qk1 = tl.dot(q1, k0)
qk2_prev = tl.dot(q2, k0)
p1 = softmax(qk0)
acc1 = tl.dot(p1, v0)

for (...):
    qk1 = tl.dot(q1, k)
    p2 = softmax(qk2_prev)     # 用上個 iter 的 qk2
    acc2 = tl.dot(p2, v_prev)  # 用上個 iter 的 v
    qk2_prev = tl.dot(q2, k)
    p1 = softmax(qk0)
    acc1 = tl.dot(p1, v)

# Epilogue
p2 = softmax(qk2_prev)
acc2 = tl.dot(p2, v_prev)
```

連續 iteration 的操作被 pipeline，兩個 dot + 一個 softmax 從不同 iteration 疊在一起執行。

**技術限制**：**當前實作主要針對 flash attention forward**——因為它主要基於獨立的 `tl.dot` 鏈。**廣義化到 flash attention backward、flex attention、jagged attention 是短期路線圖的一部分**。

### B.3 Pass 3：Partition Scheduler

**目標**：把 code 分成一個或多個 warp partition，決策透過 op attribute 傳遞。

**當前支援的 partition 類型**：
- **Compute partition**——例如 Gen5 MMA on tensor cores 專用的 warp
- **Data partition**——TMA loads 專用的 warp
- **Epilogue partition**——後處理（例如 attention 的 output projection）
- **Correction partition**——例如 flash attention 裡的 max / sum 追蹤與 rescale

**路線圖上的未來 partition 類型**：
- **Mixed partitions**——把 CUDA op 放進 data partition
- **TMA loads + SFU + atomic add 混合的 partition**——讓 memory op 跟 special function unit op 共用 warp

**技術點**：**目前 partition scheduling 用簡單 heuristic**（based on NVIDIA warp specialization in Triton 的原型實作）。**PyTorch team 明確講這是 profile-guided optimization 的首要目標**——未來會用 offline estimate 的 per-operator cost 當 signal。

### B.4 Pass 4：Buffer Creation

**目標**：分析 partition 間的通訊，決定要開多少 buffer、開在 SMEM 還是 TMEM。

**做法**：**Channel 抽象**——source op 跟 destination op 之間定義一個 channel，channel 的 physical realization 是 SMEM 或 TMEM 裡的一組 buffer。**Buffer 的所在（SMEM vs TMEM）是重要決策**——TMEM 對 MMA accumulator 是最佳位置，SMEM 對 TMA 讀入的 tensor 是最佳位置。

**技術點**：這裡的 heuristic 涉及**「什麼樣的 op 產生什麼樣的 tensor 應該進哪個 memory」**——例如 MMAv5 的 accumulator D 應該進 TMEM，TMA 讀入的 tensor 應該進 SMEM。**Blackwell 上多了 distributed shared memory（cluster 內 SM 之間共享的 SMEM 區塊），buffer creation 需要判斷是否把 buffer 放在 distributed SMEM**——這是 3.8 CLC 全面擴散帶來的新決策空間。

### B.5 Pass 5：Memory Planner

**目標**：決定每個 channel 用多少 buffer copies、channel 之間怎麼 reuse buffer。

**技術點**：
- **Channel-aware liveness analysis**——追蹤每個 channel 的 buffer 從何時 live 到何時 dead
- **Dependency chain analysis**——沿著同 partition 內 loop 之間的依賴鏈 reuse buffer
- **Innermost loop 上多 buffer 分配**——multi-buffering 讓 pipeline depth 夠深
- **TMEM 上按 accumulator type / size / live range 排優先序**——TMEM 空間比 SMEM 小得多，需要精細管理
- **Aggressive reuse**——若 live range 不重疊、優先 reuse 現有 buffer；若必須新配、才 allocate

**Combinatorial explosion 問題**：**這是 autoWS 目前最痛的一個 pass**。Memory planning 本質是 NP-hard——channel × buffer × copies × reuse decision 的 joint space 隨 kernel 複雜度 exponential 成長。當前用 heuristic，路線圖上要靠**DSL annotation + autotune over selected plans** 才能收斂。

### B.6 Pass 6：Code Partitioner

**目標**：把 kernel op 真正分到不同 warp partition、設立 producer-consumer 同步機制。

**做法**：
- **Small-scope instruction ordering**——重排同一 warp partition 內來自不同 data partition 的 op，降低 live range 與 register pressure
- **Barrier setup**——每個 channel 用兩個 barrier 做 producer-consumer 同步
- **Accumulated index count**——追蹤 channel 執行次數，據此算 buffer index 與 phase
- **Barrier 復用**——如果 source op 或 destination op 來自 TMA / gen5 GEMM，可以 reuse 現有的 TMA / gen5 barrier
- **Reused buffer 的同步保證**——如果 channel A 與 B reuse 同一塊 memory，用同一組 barrier + 單一 accumulated index count 保證 channel A 與 B 的執行順序正確

**個人技術判斷**：**「Reused buffer 的同步保證」這個是 autoWS 最工程化的一段**——它讓 memory planner 可以放心 reuse 而不擔心產生 race condition。這是**「NP-hard 問題用一個工程 hack 化簡成 tractable 問題」的經典例子**，值得記進技術武器庫。

---

## Benchmark 冷讀：cuDNN 那 10-20% 差距的技術意義

PyTorch 官方部落格給出的 B200 上 FA-forward benchmark：

- **stock Triton**：baseline
- **autoWS**：1.5-2× stock Triton
- **Gluon**：跟 autoWS 相近（略高一點）
- **cuDNN**：領先 autoWS / Gluon **10-20%**

autoWS benchmark 是**基於 Helion autotuning + ptxas advanced compiler configuration**——這個前提要記住：**不是「開下去就飛」，是「Helion 幫忙 autotune + ptxas 進階 flag 調過 + autoWS 內建 heuristic」三個組合起來**才有 1.5-2×。

### 為什麼 cuDNN 還快 10-20%？

**技術本質**：cuDNN 內部有大量**離線 profile 過的 launch config**——tile shape、warp count、pipeline depth、swizzle pattern——這些對特定 (M, N, K, dtype, sequence length) 組合是**手工挑選 + 大規模自動 profile** 過的。cuDNN 也有 workload-specific 微調——例如針對 Llama、GPT 各種常見 sequence length 有預設 shape 家族。

autoWS 現在的 partition / SWP / memory planner **都靠 heuristic + 少量 autotune**，**沒有一個系統性的 profile-guided pipeline**。這就是那 10-20% 的技術根源。

### PyTorch team 的短期路線圖（直接針對這 gap）

**(1) Profile-guided partition scheduling**：
- 用 offline estimate 的 per-operator performance 當 cost model
- 涵蓋 TMEM/SMEM load-store cost、per-region execution time、e2e execution time
- 讓 partitioner 估算 op-to-op channel 的通訊成本
- **要用 [Triton-MPP (Multi Pass Profiler)](https://github.com/pytorch-labs/triton-mpp) 或類似工具收集 profile**

**(2) Profile-guided SWP scheduling**：
- 用 runtime profile + data dependency 分析改進 modulo scheduling
- 支援 outer loop（loop flattening）
- Autotune cost-estimated top-k schedule
- **這是傳統編譯器裡 modulo scheduling 十幾年沒真正解決的閉環問題**——用 profile 當 cost model 幫忙剪枝跟排序

**(3) Memory planner 的 DSL annotation + autotune over selected plans**：
- 讓 kernel 作者用 DSL 標記關鍵 channel 的優先權
- Compiler 估算每個候選 memory plan 的 cost，autotune top-k
- **正面認了 memory planning 是 NP-hard、必須 human-in-the-loop 或 profile-in-the-loop**

### 中期路線圖（12-18 個月，把差距壓到 < 5% 的關鍵）

**Ping-Pong Scheduling**：
- 強制執行 SFU / SMEM / TMEM 這類臨界資源的互斥使用
- 當這些資源被競爭時、優先讓「生產下一步需要 data 的 warp」執行
- 目前用 pattern matching 識別 critical section，路線圖是**做成一個系統性 pass、上下文感知地排 barrier**

**Region-based Explicit Subtiling**：
- 讓 kernel 作者用 DSL 明確指定子分塊 factor（例如 `subtile=4`）
- Compiler 據此把 channel 分割成更細粒度的多個 channel
- 讓 producer 只完成一部分時、consumer 就能開始
- **這是把「fine-grained producer-consumer overlap」變成一等語言特性**——比等 compiler 自己推理出來實用

**Debuggability & Tooling**：
- TTGIR 轉 TLX（Meta 內部的更可讀 kernel format）
- Warp partition / dependency graph / memory allocation / SWP schedule 的視覺化
- 用 upstream Triton 的 [aref](https://arxiv.org/abs/2510.14719) 當 channel 抽象
- **Interactive tooling**——讓作者做 compile-time 決策後繼續 pass pipeline

**Generality & Stability**：
- 支援 flash attention backward、flex attention、jagged attention
- 這些目前 autoWS 還做不好，因為它們不是「純獨立 tl.dot 鏈」

**個人判斷**：**這四項合起來是決定 Triton 能不能追上 cuDNN 那 10-20% 的關鍵**。ping-pong 跟 subtiling 是「新的優化維度」，debug tooling 跟 generality 是「基礎工程紀律」。**前兩項壓性能，後兩項擴適用範圍**——兩者都做完才能真的和 cuDNN 平起平坐。

---

## 技術拆解 C：Triton + autoWS vs CuTeDSL，兩條路的分工

**這是這篇的核心產業判斷**——不能用「二選一」框架看這兩條路。

### CuTeDSL / NVGEMM 的定位

- **問題形狀**：固定 problem shape（GEMM、grouped GEMM、scaled GEMM、attention 的 QK/PV matmul）
- **性能特徵**：compute-bound，需要 vendor-tuned tile shape 與 cluster config
- **compile 成本**：Python→MLIR JIT，秒級
- **可 fusion 範圍**：epilogue fusion（bias、activation），但**跨 op 的 flexible fusion 做不到**——`cutlass_api` 只認 GEMM 家族的 problem shape
- **廠商鎖定**：Nvidia-only，永遠不會跑在 AMD 上
- **Inductor 路徑**：`cutlass_api.query_kernels(problem_shape) → List[KernelSpec]` → `nvMatmulHeuristics.rank(candidates)` → top-5 benchmark → cache winner

**適合的 workload**：
- LLM decode 的 GEMM 熱點（M ≤ 256 的 decode-regime）
- Attention 的 QK matmul、PV matmul
- MXFP8 / NVFP4 這類 quantized GEMM 的固定 shape
- 任何**「pattern 固定、shape 已知、fusion 侷限於 epilogue」**的 workload

### Triton + autoWS / Gluon 的定位

- **問題形狀**：open-ended fusion、user-defined kernel、跨 op 的 megakernel
- **性能特徵**：memory-bound 到 compute-bound 都要支援，flexible tile shape
- **compile 成本**：JIT，秒到分鐘級（複雜 fusion 較久）
- **可 fusion 範圍**：**任意 op 的 IR-level fusion**——這是路線圖的長期目標（見「Kernel/Operator Fusion and Megakernels」）
- **廠商鎖定**：跨 NVIDIA / AMD（gfx1250 / CDNA5 對稱推進）
- **Inductor 路徑**：TorchInductor 生成 Triton IR → 走 autoWS pipeline → 產出 warp-specialized kernel

**適合的 workload**：
- Flash attention 變體（forward、backward、flex、jagged）
- Attention 後複雜 epilogue（RMSNorm + rotary + KV cache write 等 fused）
- Point cloud / graph / MoE 這種 **shape-dynamic + fusion-heavy** workload
- 任何**「kernel author 需要 flexible 表達」**的 workload

### Inductor 是 backend router

在 2026 這個時間點：

- Inductor = 圖層優化 + backend selection
- GEMM fast-path 交 CuTeDSL / NVGEMM
- Flexible kernel 交 Triton + autoWS
- Backend competition benchmark 決勝負：`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="ATEN,TRITON,NVGEMM,CUTLASS"` 讓四路並跑、跑最快的贏
- 這是 compiler 的 **"backend router" 時代**開始——過去一個 compiler 一個 codegen 路徑，未來是**多路徑 + runtime router**

### 為什麼兩條路必然共存

**技術原因**：**沒有一條路徑能覆蓋所有 workload**。
- CuTeDSL 的 Python→MLIR JIT 快、但**表達能力限於 GEMM 家族**
- Triton 表達能力廣、但**GEMM fast-path 的 vendor-tuned 性能追不上 cuDNN**

**生態原因**：**Nvidia 想擴 moat，PyTorch 想跨廠商**。
- Nvidia 推 CuTeDSL / NVGEMM 進 Inductor，把 GEMM fast-path 綁在 Nvidia hardware
- Meta 推 Triton + autoWS，保留「同一 kernel source 跨 NVIDIA / AMD」的選項

**產業原因**：**Inductor 作為 PyTorch 官方 compiler，必須讓兩條路並存**——不能只綁 Nvidia（否則 AMD / Intel 側 PyTorch 使用者會流失），也不能只綁 Triton（否則失去 Nvidia 新硬體的 fast-path）。**backend router 是唯一政治上可行的架構**。

---

## 對 compiler-track 工程師的三個技術 takeaway

### Takeaway 1：autoWS 六個 pass 是「AI compiler 的完整 producer-consumer synthesis 樣板」

傳統編譯器（GCC、LLVM）沒有把 buffer、channel、producer-consumer 當一等 IR 概念——這些是 runtime library 的職責（例如 pthread、MPI）。**autoWS 這套架構把 producer-consumer 抽象搬進 compiler IR**，六個 pass 分別做：

1. 產生足夠的 op 給 producer / consumer 分工（Data Partitioning）
2. Schedule 好 producer / consumer 的執行順序（SWP）
3. 決定誰是 producer、誰是 consumer（Partition Scheduler）
4. 決定 buffer 開在哪裡（Buffer Creation）
5. 決定 buffer 用多少、怎麼 reuse（Memory Planner）
6. 落地 barrier 與 index tracking（Code Partitioner）

**這六個 pass 會成為未來所有 GPU AI kernel compiler 的 reference**——AMD ROCm compiler、Intel Xe compiler、Qualcomm Hexagon-MLIR、華為 CANN、Ascend、寒武紀 MLU compiler 都會 copy 這個結構。**看懂這六個 pass，是理解「後 CUDA 時代 GPU AI compiler 長什麼樣」的最短路徑**。

### Takeaway 2：profile-guided optimization 在 GPU AI compiler 上的復興

傳統 PGO（Profile-Guided Optimization）在 CPU 上因為 branch predictor 已經夠強、加上主流 workload 是 latency-sensitive，實用性有限。**GCC / LLVM 的 PGO 過去十年沒有大的架構突破**——因為需求端不夠強。

**GPU AI compiler 把 PGO 的需求重新點亮**：
- Warp partition 是 combinatorial 選擇——heuristic 只能到某個位置
- SWP schedule 對 GPU 而言決定 pipeline depth 與 register pressure——static 分析看不出來
- Memory plan 對 TMEM 這種小空間的 packing decision——需要 runtime signal

**PyTorch team 短期路線圖上三個項目**（profile-guided partition、profile-guided SWP、memory planner autotune）**全都是 PGO 在 GPU AI compiler 上的具體形式**。**接下來 12 個月 profile-guided GPU compiler 會是 hot area**——不僅 Triton，CuTeDSL / Mojo / IREE 都會加類似機制。

**對 compiler engineer 意義**：**如果你能同時懂 PGO 傳統理論 + GPU AI kernel 的 profile 收集實務 + cost model 建構，你在 compiler team 面試會非常搶手**。這是一個「傳統 compiler 知識 + 新 domain expertise」交叉的甜蜜點。

### Takeaway 3：sanitizer 進 kernel compiler

FpSan / GSan / ConSan 這種在 Triton 3.8 一次到位的做法，**會被複製到 CuTeDSL、Mojo、Helion**。

**技術判斷**：Kernel authoring 從 2020 前後的「英雄工程」（每個 kernel author 手工除 bug、每個 bug 靠 print + Nsight）走向 2026 之後的「工程紀律」（compiler-instrumented checker 保證正確性）。**這對 GPU AI infra 的可維護性是質變**——不再是「你要有一個超神的 kernel author，否則你的 fast path 隨時炸」。

**對面試 / 職涯意義**：**能夠架構 kernel sanitizer 的工程師是稀缺資源**。傳統 compiler 領域有 UBSan、TSan、MSan 這一系列 sanitizer 的專家；GPU AI kernel 這一側才剛開始。**這個交叉領域 2026-2028 會缺人**。

---

## 對 Adam 的具體行動建議

### (a) spconv workload 上實測 autoWS 加速

你 [`career-research-2026`](https://github.com/HuaTsai/career-research-2026) repo 的 Compiler-Path 章節裡 Option C 是走 spconv capstone——把 LiDAR point cloud 的 sparse convolution 展開成 dense GEMM 之後，用 Triton 寫 kernel、跟 spconv 原始實作比。

**Triton 3.8 給了你一個新的實驗維度**：`warp_specialize=True`。你可以：

```bash
pip install "triton==3.8.*"
```

然後在你的 Triton spconv kernel 主 loop 上開 `warp_specialize=True`，跑 A/B：
- Baseline：stock Triton（`warp_specialize=False`）
- autoWS：`warp_specialize=True`（如果需要，先開 `TRITON_ENABLE_LLVM_OPT_LEVEL=3` + Helion autotune）

**為什麼這對你的職涯是履歷級別的實測**：

1. **你的 workload 是 dynamic shape**（每個 frame 點雲密度不同、非零位置不同）——這正是 CuTeDSL 做不好、autoWS 有 sweet spot 的區塊
2. **中文技術社群目前幾乎沒人跑過 autoWS 在 LiDAR / point cloud 上的實測**——你做出一組乾淨的 A/B 對比，可以寫成一篇 short paper 或 blog，直接放進面試作品集
3. **Blackwell 上 Cluster Launch Control 對 gather / scatter 的全面支援**（3.8 剛加）——這對 spconv 這種稀疏 workload 是理論上 huge unlock，但實務上還沒人系統 benchmark 過

**具體 timeline 建議**：
- 週末（9/12-13）先 install Triton 3.8，跑 hello-world autoWS example 確認環境
- 下週（9/14-20）把你手上的 spconv Triton kernel 加 `warp_specialize=True`、跑 A/B、記錄 nsys profile
- 週末（9/20-21）寫一頁技術報告——這個內容夠當 12/15 之後 Nvidia / Meta / AMD compiler team 面試的技術口述素材

### (b) 讀 autoWS 的上游源碼

**上游分支**（相對輕）：
- `github.com/triton-lang/triton/tree/main/third_party/nvidia/hopper/lib/Transforms/WarpSpecialization`
- 這裡有部分 upstream 的 autoWS pass 實作，是六個 pass 中最先進主線的幾個
- 讀順序建議：Partition Scheduler → Buffer Creation → Code Partitioner（先讀「決策」，再讀「落地」）

**Meta OSS 鏡像**（相對完整）：
- `github.com/facebookexperimental/triton`
- 這是 autoWS 主開發分支，六個 pass 都在
- 讀順序建議：Data Partitioning → SWP Scheduler → Memory Planner（先讀「機會擴增」，再讀「時間 / 空間規劃」）

**閱讀策略**：
- **不要一開始就試圖懂所有細節**——先讀每個 pass 的 top-level entry function，理解 pass 之間的 attribute 傳遞協定
- **重點看 buffer / channel 抽象在 IR 上長什麼樣**——這是 autoWS 相對傳統 compiler 最大的創新
- **對照 PyTorch autoWS 部落格讀**——部落格是 high-level 敘述，源碼是具體實作，兩邊對照最快理解

**為什麼這比讀 modulo scheduling 教科書實用**：Nvidia compiler team、Nvidia DL Frameworks team、Meta PyTorch compiler team 的日常工作**就是**在改這種 pass。你能在面試裡指著某個 pass 說「這裡的 heuristic 我認為可以改成 profile-guided」，你就贏了。

### (c) 準備一段「autoWS + spconv 實測」的 30 分鐘技術口述

面試準備角度——**專門為 compiler-track 面試準備一段技術口述**，結構：

1. **背景**（3 分鐘）——warp specialization 是什麼、為什麼 Blackwell 這代需要 compiler 自動化
2. **autoWS 六個 pass**（10 分鐘）——按順序講、每個 pass 一句話說做什麼、講清楚 buffer / channel 抽象怎麼串起來
3. **cuDNN 那 10-20% gap 的技術本質**（5 分鐘）——為什麼靠 profile-guided 是路線圖首要項目
4. **你的實測**（10 分鐘）——spconv workload 上 autoWS vs stock Triton 的 A/B、你觀察到的 3-5 個 insight（例如「Cluster Launch Control 的 gather/scatter 對稀疏 workload 幫助最大」）
5. **開放討論**（2 分鐘）——你想在這個團隊做什麼

**Nvidia compiler team / Nvidia DL Frameworks team / Meta PyTorch compiler team 都會問「你怎麼看 warp specialization 這條路」**——你能背得出六個 pass、講得出 cuDNN 那 10-20% 差距的技術原因、還有一組自己的實測數字，你已經贏過 80% 的候選人。

---

## 冷讀結論

**這篇跟 9/2 那篇 CuTeDSL 是同一枚硬幣的兩面**。9/2 說「Nvidia 從 Inductor 內部把 GEMM fast-path 收回去」——那是 moat 擴張。今天說「Triton 3.8 + autoWS 把 warp specialization 做完整、Cluster Launch Control 全面擴散、三個 sanitizer 一次到位」——那是 open compiler 對應這件事的技術路徑。

**不要用二選一的框架看**——GPU AI compiler 未來 3 年是「多後端並存 + Inductor 當 router」的格局。

**真正需要看清的是**：

- **Triton + autoWS** 這條路是**「開源、跨廠商、可 fusion」**的核心資產——它服務 flexible kernel authoring、跨 op megakernel、shape-dynamic workload、AMD 對稱推進這些**開放性需求**。
- **CuTeDSL** 是**「Nvidia-native、GEMM fast-path、compile-time 便宜」**的補充資產——它服務 LLM decode 的 GEMM 熱點、attention 的 QK/PV matmul、vendor-tuned tile shape 這些**確定性需求**。

**Adam 想走 compiler 職涯，兩條都要碰**——但如果只能挑一條讀源碼，讀 autoWS 這六個 pass 的具體實作，回報率遠高於讀 `cutlass_api` 的 kernel 目錄查詢邏輯。因為 autoWS 的六個 pass 是**「AI compiler 的完整 producer-consumer synthesis 樣板」**——這個抽象會被 AMD、Intel、Qualcomm、華為、寒武紀複製。**看懂這一套，等於看懂了未來 5 年 GPU AI compiler 的骨架**。

CuTeDSL 是 Nvidia 的商業產品，看懂它學到的是「一個廠商的 vendor library plugin shape」。autoWS 是社群 open-source compiler 的完整技術路徑，看懂它學到的是「AI kernel compiler 這門學問的方法論」。**兩者的技術密度差一個數量級**。

**最後一個判斷**：**cuDNN 那 10-20% 領先在 12-18 個月內會被壓到 < 5%**。理由是：短期路線圖上的 profile-guided partition / SWP / memory planner autotune 三個項目，一旦落地，就會把「cuDNN 靠離線 profile 領先」的技術基礎搬進 open compiler。剩下的 < 5% 是 vendor 對「特定 shape 家族」的手工微調——這個 gap 存在也可以接受，因為 Inductor 是 router、它會選最快的那條路。**Triton 的目標從來不是打敗 cuDNN，是把「非 cuDNN 覆蓋的 workload」做到 near-optimal**。這是一個更務實、也更可行的定位。

---

## 附錄：技術參考

### Triton v3.8 相關

- [Triton GitHub Releases](https://github.com/triton-lang/triton/releases) — 完整 release notes
- [Warp Specialization in Triton: Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) — 2026-01-08 PyTorch 官方部落格
- [Meta OSS Triton mirror](https://github.com/facebookexperimental/triton) — autoWS 主開發分支
- Triton upstream autoWS pass: `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization`

### 相關前作

- 2026-09-02 [CuTeDSL 進 Inductor：PyTorch 把 GEMM fast-path 外包給 Nvidia](cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md) — 這篇的鏡像動作
- 2026-09-05 [Syncopate：OSDI 2026 把 communication chunk 提升成 compiler 一級抽象](syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026.md) — Triton source-to-source compiler 另一路徑
- 2026-08-31 [ARGUS：用資料流不變量把 LLM 寫的 GPU kernel 拉到 99-104% 手工 assembly](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md) — kernel 正確性驗證

### 相關論文與資源

- Cypress (PLDI 2025) — 分離 computation 與 data 的 DSL 抽象
- Halide (PLDI 2013) — kernel-authoring paradigm 的前輩
- [aref (arXiv 2510.14719)](https://arxiv.org/abs/2510.14719) — Triton 未來 channel 抽象
- Triton-MPP (Multi Pass Profiler) — profile-guided GPU compiler 的資料收集工具

---

*Nova 記：這篇補齊了 9/2 那篇 CuTeDSL 的另一面。之前七、八篇 compiler 系列的主軸太集中在「CUDA moat 被擴 / 被打」，容易讓人以為 Triton 這條開源路徑已經被邊緣化。今天把 Triton 3.8 + autoWS 的技術路徑寫完整之後，敘事回到平衡狀態——**兩條路必然共存、Inductor 當 router、Adam 應該兩條都懂但先讀 autoWS 六個 pass**。Adam 加油——你選這條 compiler 職涯的路是對的，2026 這個時間點 open compiler 內部技術密度大、產業需求急、缺人。*
