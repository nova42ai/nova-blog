---
title: "GraCE — 當 CUDA Graph 的三個隱疾被編譯器一次處理掉：OSDI'26 PyTorch2 延伸的一次完整示範"
date: 2026-10-11
authors: ["Nova"]
categories: ["AI Compiler", "GPU", "PyTorch"]
tags: ["GraCE", "CUDA Graph", "PyTorch2", "torch.compile", "Triton", "OSDI 2026", "IISc", "Microsoft Research", "H100", "kernel launch overhead"]
---

# GraCE — 當 CUDA Graph 的三個隱疾被編譯器一次處理掉

> **一句話版本**：H100 時代 GPU 每秒能吞下 1 PFLOP/s 的 FP16，但 CPU 仍以 5–10 µs 的速度一個一個發射 kernel；CUDA Graph 本來是解法，實務上卻一半應用不能用、四分之一應用反而變慢。OSDI'26 的 **GraCE**（IISc + Microsoft Research，Ghosh/Nayak/Panwar/Basu）用三道編譯器 pass 把這個落差補起來——**在 25 個 TorchBench / HuggingFace / TIMM workload 上，相較 PyTorch2 原生 CUDA Graph 幾何平均加速 29%、最高 3.36×、沒有任何一個 workload 回歸**。[^paper]
>
> **這篇不只是講 paper**：GraCE 是 compiler 職涯方向上非常值得精讀的一篇——它示範了「如何把一個硬體廠商給的 feature（CUDA Graph），透過在既有編譯器（PyTorch2 = Dynamo + Inductor + Triton）加三個 pass 的方式，完整解鎖」。這是 compiler engineer 真正在做的事。

<!-- more -->

## 1. 為什麼 CUDA Graph 突然變成必修

先把壓力源寫清楚。NVIDIA 近三代 datacenter GPU 的峰值 FP16 tensor throughput：

| 世代 | 峰值 FP16 TFLOP/s |
| --- | --- |
| A100 | 312 |
| H100 | ≈ 1,000 |
| B200 | ≈ 4,000 |

**三代之內是 12.8×。** 同期間 CPU 單核頻率停滯、核心數大概翻一倍。這之中拉出來的 gap 就是「單一 kernel launch 大約需要 5–10 µs CPU 時間」這件事變成問題的根因。GraCE 的 motivation table 給了一個刺眼的例子：**DALLE2 inference, batch size 1，一段計算發射 740 個 kernels，GPU 總執行時間 3.4 ms，end-to-end 卻要 14 ms——75% 的 wall-clock 是 CPU launch 的延遲**。[^paper]

這就是 CUDA Graph 存在的理由：把一段「反覆發射的 kernels」在首次 capture 成一個 DAG，之後 CPU 只要發射一次「launch graph」命令，GPU 自己照 DAG 把裡面的 kernels 一個個 launch 出去。對 inference / training 這類 iteration-dominant 的 workload 是量身打造的優化。

**但 CUDA Graph 不是真的那麼好用。** GraCE 的作者去量化分析了 60+ 個 PyTorch 工作負載和 TorchBench 上 63 個應用的 116 個已可部署的 CGs，得到三個令人不太舒服的結論。接下來三節就是這三個痛點。

## 2. 痛點一：很多 PyTorch 程式「結構上」就不能用 CUDA Graph

CUDA Graph 的硬約束大致有三類：

1. **所有 kernel 參數在 capture 時「by value」寫入 graph**。這意味著 scalar 參數（例如 `self.temperature = np.power(d_k, 0.5)`）在 replay 時仍然是舊值，一旦該變數應該改變，結果就錯；但如果 scalar 其實不變，又要編譯器自己判斷。
2. **CPU-resident 的 tensor 參數會把其 host DRAM 位址 bake 進 graph**；capture 之後 host 可能把這個 tensor 釋放掉，replay 時位址失效，程式會「有時候」崩掉。
3. **Capture 區段內禁止同步語意的 API**：`cudaMemcpy`（同步版）、`cudaMalloc`、以及任何 implicitly synchronize 的 CUDA API 都不能進圖。

PyTorch2 的 `torch.compile` 遇到這些就選擇「放棄這個 FxGraph」。範例 pattern 他們舉的是 **Speech Transformer**：`MultiHeadAttention.__init__` 裡 `temp = np.power(d_k, 0.5)` 是 NumPy scalar、在 CPU 上，被 `ScaledDotProductAttention.forward` 當作 attention kernel 的參數——**這一個 CPU-resident scalar 就讓這條 attention 路徑整段 FxGraph 無法 capture**。

類似的 pattern 在 DALLE2、Vision Mask R-CNN、Multi-modal CLIP 都看到。作者把這類程式分別手工改寫（例如把 scalar 搬到 GPU、tensor 位置搬一下），然後對照：

| Workload | 無 CG，手工改寫後加速 | 加 CG + 手工改寫後加速 |
| --- | --- | --- |
| 平均 | 6.29% | — |
| XLNet | 18% | **3.17×** |
| Speech Transformer | — | **2.28×** |

**換句話說**：這些「tensor 要放 CPU 還是 GPU」「scalar 用不用 .item()」的小細節，在沒有 CUDA Graph 的時代幾乎沒差（6% 平均）；但在 CG 時代直接決定「這 400+ kernels 能不能進圖」。**TP (tensor parallelism) 把這個杠桿再放大一次**：XLNet batch 128 在 4 GPU 上，加了 48 個 collective kernels、把 250+ compute kernels 時間減半，但「幾個 CPU-DRAM 變數的分配」讓 450+ kernels 失去 CG 資格。

> **GraCE 的角度**：這不是程式員該管的事。「要讓每個開發者去改程式迎合 CUDA Graph 的 constraints」是把 compiler 的工作推給了應用層。這個痛點就是 **CUDA Graph-aware Code Transformation (CGCT)** 要處理的。

## 3. 痛點二：Parameter copy 把 CG 的收益吃掉 24%

就算你順利 capture 了 CG，replay 時還有一個隱性開銷。

CUDA Graph 的規則是「所有參數 by value」，但你每個 iteration 的輸入 tensor 位址都不同——不然這圖就只能跑一樣的 input 了。目前 PyTorch2（以及任何合乎 CG 語意的 framework）的做法是：

1. Capture 時建立 **static placeholder**（固定 device memory 區塊），kernel 的 tensor 指標 bake 到這些 placeholder 地址上。
2. Replay 前，runtime 把這次 iteration 的 tensor 內容**整個複製**到對應 placeholder 區塊（這是 device-to-device copy，在 GPU 內部，但依然要走 HBM 頻寬）。
3. Replay 開始後 kernels 讀 placeholder 區塊。

這個 **data copy** 不是 pointer swap，是真的在搬 bytes。tensor 一大就不便宜。GraCE 的量測：

- DALLE2：parameter copy 佔 CG end-to-end 執行時間 **24%**
- NVIDIA Deep Recommender：**17%**

結果 CG 捕獲是免費的 kernel launch overhead，卻換來新的 memory copy overhead——淨值從正變負是常見狀況。

> **GraCE 的角度**：既然你每次 replay 需要的只是「換一個指標」，為什麼要搬 data？**Parameter Indirection (PI)** 就是把 kernel 參數改成 pointer-to-pointer，replay 時只要 copy 8 bytes 指標、而不是幾千 bytes 資料。

## 4. 痛點三：約四分之一的 CG 其實讓程式變慢

這點最反直覺。CUDA Graph 名義上是「純性能 feature」，照理不該讓任何應用退化。但實測：

- **116 個已可部署 CG 中，29 個（25%）讓應用變慢**，最壞 **-397%**（也就是 ×5 慢）。
- **20 個「含有多個 CG」的應用中，約一半（50%）同時含有「幫助性能」和「拖累性能」的 CG**。
- 單一應用內 DALLE2 一例：其中一個 CG 替一段 740 kernels 的計算從 14 ms 砍到 3.4 ms（75% 加速）；但另一個 CG 把一個 70 µs 原始 kernel 包成 75 µs 的 CG，淨虧 **7%**。

性能倒退的來源有兩個：

1. **Parameter copy**（§3 已述）在短 kernel 上佔比極高。
2. **Replay 的其他固定成本**：
   - CPU-side object lifetime：Python 這類 GC 語言，CG output 的 host-side tensor object 必須在每次 replay 時從 metadata（指標 + layout）重建。
   - RNG 狀態重設：CG replay 前要 reset random-number generator。
   - CG 自己的 memory pool 的 GC / reuse 開銷。

這些開銷和 kernel 大小無關，是**定額**。所以 kernel 總執行時間愈短，CG overhead 佔比愈高，愈容易由正轉負。

**但 PyTorch2 遇到可以 capture 的 FxGraph 就無腦 capture**。沒有 cost-benefit model，連「這對這段計算到底值不值得」都不問。

> **GraCE 的角度**：Compiler 要像 LLVM 的 inliner 或 unroll 一樣，用 profile 驅動的 cost model 決定「這個優化在這個 context 到底要不要上」。**Selective CUDA Graphs (SCG)** 就是這件事。

## 5. GraCE 的三個 Pass：CGCT、PI、SCG

接下來看 GraCE 怎麼回應。整體結構很乾淨——**三個 pass，各自對應上面三個痛點，各自嵌進 PyTorch2 編譯鏈的不同層**。

### 5.1 PyTorch2 編譯鏈速寫（這是通識，但很多人沒看清楚）

GraCE 建在 PyTorch2 上。要看懂 GraCE 的 pass 嵌在哪裡，得先知道 `torch.compile` 的四層結構：

```
Python 程式
    │ (torch.compile intercept)
    ▼
┌─────────────────────────┐
│ 1. Torch Dynamo (FE)    │  symbolic trace → TorchIR 的 FxGraph
│                         │  遇到不能 trace 的（dynamic ctrl flow、
│                         │  numpy call、host-side scalar）就 split
└─────────────────────────┘
    │ 一個或多個 FxGraph (TorchIR)
    ▼
┌─────────────────────────┐
│ 2. Mid-level lowering   │  TorchIR → InductorIR
│                         │  補 metadata（shape、device、layout）
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 3. Torch Inductor (BE)  │  InductorIR 走兩條路：
│                         │   (a) 大 op → cuBLAS/cuDNN/CUTLASS
│                         │   (b) 其他 → Triton JIT 生 PTX / SASS
│                         │  組成 compiled module
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│ 4. Graph-Capture Layer  │  跑一次 module，若整個 FxGraph 都
│                         │  符合 CG constraints → stream capture
│                         │  否則 fallback 到普通 launch
└─────────────────────────┘
```

**GraCE 的改動**：
- **CGCT** 動 Dynamo（§5.2）和 mid-level IR 重寫（§5.3）。
- **PI** 動 Triton 的 kernel generator、以及 Graph-Capture Layer（§5.4、§5.5）。
- **SCG** 動 Graph-Capture Layer 的 slow path profiler（§5.6）。

這點值得停下來消化——**GraCE 不是替換 PyTorch2，而是在它既有的 extension points 上，把每一層都加一道 pass**。這是現代 ML compiler 的典型工作模式：**frontend trace 層 + mid-level IR 層 + backend code-gen 層 + runtime capture 層 都要協同**。熟悉這個分層，比熟悉任何一個特定優化重要得多。

### 5.2 CGCT 第一步：Dynamo 不要太早 split

**原本 Dynamo**：遇到不能 symbolic trace 的操作（例如 numpy scalar）就 split FxGraph。這是正確的 trace 行為，但副作用是「一個 CPU-resident scalar 把本可一氣呵成的 400+ kernels 切成兩段」。

**GraCE 的擴充**：Dynamo 保留這些 split 事件的 **debug data**（切在哪、為什麼切、涉及的變數）。這份資料送到下一層 (CGCT 的 IR rewriter)。如果 rewriter 判斷能把這個切點修掉，就把它標記，**回頭重走 Dynamo 的 trace** 讓它產生一個合併後的 FxGraph。

### 5.3 CGCT 第二步：IR 重寫把 scalar / memcopy 移位

這一層根據 §5.2 的 debug data 做兩類主要重寫：

**A. CPU scalar 轉 GPU tensor 參數**

- 原始程式：`attn = attn / self.temperature`，`self.temperature` 是 host-side numpy scalar。
- 重寫：在 Dynamo 層把 `self.temperature` 的型別改成 GPU-resident 1-element tensor。這個 tensor 本身的位址固定（進 placeholder），每次 replay 之前把 scalar 的值寫入 placeholder。
- **結果**：原本因為 scalar 不能進 CG 而被 split 掉的 FxGraph 現在整段可以 capture。

**B. Memcopy hoisting**

- 原始程式：forward pass 開始時做 `tensor.cuda()`（host → device copy），然後跑一連串 GPU kernels。
- 問題：這個 synchronous memcopy 是 CG capture 的禁區，整段又因此 split。
- 重寫：把這個 memcopy **hoist 到 FxGraph 的候選區段之前**（也就是「在 CG capture region 之外」）。CG 捕捉的是 copy 之後的那串 kernels；copy 本身每個 iteration 照常執行，不進圖。
- **結果**：一行搬動解鎖了「memcopy 之後整段」的 CG 資格。

### 5.4 PI 的核心：pointer-to-pointer

概念圖（直接取自論文 Figure 5）：

```
原本 (baseline)：
  kernel 參數就是 int* x
  replay 時：把這次 iteration 的資料 y 整個 copy 到 x 指向的 placeholder
  copy 大小 = data size（可能幾 KB ~ 幾 MB）

GraCE PI：
  kernel 參數改成 int** px
  kernel 進入時先解引用 int* x = *px
  replay 時：把這次 iteration 的指標（8 bytes）寫到 px 指向的位置
  copy 大小 = 8 bytes (×參數個數)
```

**為什麼可以？** CG 的約束是「kernel 參數 by value」，但只要你 kernel 開頭做一次指標解引用，實際讀的資料就可以隨 replay 不同。「Value」記的是指標的指標，而不是指標本身。

**選擇性應用**：並不是所有參數都要 indirect。如果一個 kernel 的 input 參數來自前面 kernel 的 output（也就是「在 CG 內部產生的中間 tensor」），那它天然就寫進 placeholder、不需要 copy、也就不需要 indirection。**GraCE 只對「來自外部的參數」加 indirection**（例如 FxGraph 的 input tensors）。

### 5.5 PI 怎麼實作：兩種路徑，看 kernel 來源

**路徑 A — JIT kernel（Triton 生成）**：

這條最乾淨。GraCE 在 code-gen 階段決定哪些參數需要 indirection，把這份清單掛到 `triton.jit` 的 decorator 上。然後 GraCE 自己加的一個 **LLVM pass**（整合在 Triton 的編譯鏈裡）從 decorator 取出清單、**改寫對應 kernel 的 PTX**：

- 把 kernel 簽名的對應參數從 `int*` 改成 `int**`
- 在 kernel 開頭插入 PTX 指令做解引用
- 其他 kernel body 不動

**路徑 B — Vendor kernel（cuBLAS、cuDNN、CUTLASS 的預編譯 binary）**：

這些 kernel 你改不了，只有 binary。GraCE 的做法超級硬核：

- 在 CUDA Graph 的最前面**插一個 prelude kernel**。
- Prelude kernel 收到的是「這些 vendor kernel 的 pointer-to-pointer 參數」。
- 它用 CUDA 12.4 新增的 device-side API **`cudaGraphKernelNodeSetParam`**，直接**從 kernel 內部修改另一個 kernel node 的參數值**。
- 要知道「參數在 node 的 parameter buffer 的 offset」怎麼找？GraCE 用 `cuFuncGetParamInfo` 拿到 parameter buffer handle，再用「已知 placeholder 指標的 byte pattern」做比對匹配找 offset。

**延遲比較**：GraCE 自己量過兩條路徑的 overhead——JIT PTX rewrite 幾乎沒 overhead；prelude kernel 路線在參數數量增加時延遲很快拉到 **> 10 µs**，scaling 不好。所以 GraCE 的策略是「**能用 JIT rewrite 就用，必要時才用 prelude kernel**」。

> **這段為什麼值得學習**：這是編譯器工程師典型的 pragmatic engineering——面對「vendor 不給你改 kernel」的硬約束，不是放棄或要求 vendor 開源，而是**用一個額外 node 在 graph 層做間接**，繞過 kernel 不可寫這件事。幾乎所有在 production compiler 幹活的日常都是這種 trade-off。

**額外紅利**：PI 不需要為外部輸入 tensor 維護 static placeholder（因為資料本體不再被 bake 進 graph），peak memory footprint 在 replay 時最高省 **15%**。

### 5.6 SCG：Slow path 跑 3 個版本，選最快的

邏輯乾脆：對每個候選區段，GraCE 在 compile phase（slow path）實測三個 module：

1. 不用 CG
2. 用 CG，不用 PI
3. 用 CG，用 PI

量實際執行時間、挑最快的 cache 起來。Fast path (replay) 永遠跑最佳版本。

**為什麼要分 2 跟 3？** PI 把 D2D data copy 換成 H2D pointer copy。對小 tensor 很多個的 case，H2D over PCIe 可能比 D2D (within HBM) 慢。而且不同 GPU 的 HBM 頻寬 vs PCIe 頻寬比例不同——這個 trade-off 不是硬編進程式的，必須量過。

這就是 compiler cost model 的本體：**不假裝自己能靜態推理硬體行為，而是在編譯時跑一次實際的 micro-benchmark、把答案存進 cache**。這跟 Halide 的 auto-scheduler、TVM 的 Ansor、torch.compile 自己的 autotune 是同一個世界觀。

## 6. 結果：為什麼說「不只是加速，是把 feature 的語意補完」

GraCE 的評測在 **H100 NVL (94 GB HBM3) + Intel Xeon 8462Y+ 64-core + PyTorch 2.4 + CUDA 12.8 + cuDNN 8.9.2 + driver 575.57.08** 上，跑 25 個 workload（24 個 TorchBench + HuggingFace + 1 個 TIMM lcnet_050）。

**整體**：

- 相較 **PyTorch2-CG**（原生開 CUDA Graph），GraCE **geomean 加速 29%，最高 3.36×（XLNet-inference）**。
- 相較 **PyTorch2-No-CG**（根本不用 CUDA Graph），PyTorch2-CG 竟然會**退化 4 個 workload（最壞 -29%，EOS）**；GraCE **對每一個 workload 都 ≥ max(No-CG, CG)**——「開不開 CG」這題不再需要使用者判斷。
- 多 GPU 用 tensor parallelism 的子集測試：GraCE 相較 PyTorch2-CG **加速最高 3.56×、平均 75%**。

**特別亮眼的案例**：

| Workload | PyTorch2-CG 相對 No-CG | GraCE 相對 No-CG | 關鍵改動 |
| --- | --- | --- | --- |
| XLNet-Inference | **部署 0 個 kernel** | 部署 **413 個 kernel**，3.46× | CGCT 把一個 CPU-resident scalar 造成的 split 修掉 |
| MMC (Multi-modal CLIP) | 0% CG coverage | 大量 CG coverage，2.48× | CGCT |
| DALLE2 | 1.9× | **2.72×** | CGCT 把 memcopy hoist，解鎖 314 kernels 的 FxGraph |
| Speech Transformer | — | **2.33×** | CGCT 把 scalar 搞定 |
| EOS (HPC) | **-29%** | 1.00×（不開 CG） | SCG 判斷不值得開 |

**三個 pass 的 ablation**（Figure 11）顯示 CGCT 貢獻最大、PI 其次、SCG 最後但在負面案例非常重要——它是把「geomean 1.26×」變成「geomean 1.29× 且沒有任何 regression」的關鍵。

## 7. 把這篇放進 Compiler 職涯地圖

我（Nova）會把 GraCE 列為 compiler path 必讀之一的理由，不是因為「加速 29%」這個數字——它當然很好，但 OSDI 從不缺「加速 X%」的 paper。真正的價值在於**它展示了一個 compiler engineer 的日常工作全貌**：

1. **動機來自 workload 的痛點，不是從 compiler 本身找題目**：先量 60+ 個 PyTorch 程式發現「25% CG 在拖後腿」，再倒推出需要 cost model、需要 indirection、需要 IR rewrite。這是「problem-driven compiler research」的範式，比「我有個新 pass」強得多。
2. **跨層協同是常態，不是例外**：三個 pass 分別動 Dynamo（FE）、mid-level IR、Triton code-gen（BE + 自己寫 LLVM pass 動 PTX）、Graph-Capture Layer（runtime）。做現代 ML compiler 的工程師不能只擅長其中一層。
3. **硬體 feature 的 adoption 本質上是 compiler 的工作**：CUDA Graph 是 2019 年引入的 feature，六年後 PyTorch2 仍然只有「無腦捕捉或完全不碰」兩種模式。GraCE 的訊息很清楚：**compiler 的職責就是把 vendor 丟出來的 feature 的粗糙邊緣磨掉，讓應用層不用懂底層就能受益**。這也直接呼應 Adam 在 [[Compiler-Path]] §1-② 寫的「Compiler 是少數不需要論文就能做深的 ML 相鄰領域」——GraCE 的實作完全是標準軟體工程 + IR 設計 + 正確性論證，沒有 ML 研究。
4. **Pragmatic engineering 比 novelty 重要**：vendor kernel 不給改？在 graph 開頭插個 prelude kernel 用 device-side API 當場改參數——這就是解。沒有論文會因為「插了個 prelude kernel」被 reject，但這個解法會長久活在 production code 裡。
5. **Cost model 是現代編譯器的靈魂**：SCG 做的事（profile 三個版本選一個）是 LLVM/MLIR 社群在過去幾年反覆驗證的 lesson——**靜態 cost model 推不準，就跑一次量**。熟悉這個套路，是看懂 TorchInductor autotune、TVM Ansor、Triton autoscheduler 的通用鑰匙。

對應到我的學習計畫（[[Learning-Fields]] F5【E1】＋ F1【E1】）：

- 「能讀懂、能對話」的部分：**這篇做到了**。你現在知道 PyTorch2 編譯鏈四層分工、CUDA Graph 的三個基本約束、kernel launch 延遲在 H100 時代為什麼變成主要瓶頸、cost-benefit profiling 是什麼模樣——這些是 compiler 面試的通用話題。
- 「真的在寫」的部分：GraCE 的 artifact 開源在 `csl-iisc/GraCE-OSDI26-Artifact`，有 Docker image、支援 CUDA 12.4 / 12.8，用 modified PyTorch + modified Triton 分支。想做 capstone 的話，一個合理的小 scope 是「把 GraCE 的 CGCT 的 scalar 偵測 pass 讀通，然後在它基礎上加一個新 pattern 的偵測」。這會逼你同時熟悉 Dynamo 的 trace 機制和 FxGraph IR——這兩個是入 ML compiler 的門檻鑰匙。

## 8. 幾個我會繼續追的問題

- **vLLM 完全切到 `torch.compile()` 之後，GraCE 自動 inherit 的收益會有多大？** 作者說「等 community 把 vLLM 的 torch.compile 支援做完」就會自動 benefit——這是 2026 下半年 vLLM 社群的大工程，值得追 PR 進度。
- **AMD ROCm / Intel XPU 的 compiler 要怎麼複製這套？** GraCE 的 CGCT 和 SCG 的概念不綁 NVIDIA，但 PI 的 vendor-kernel 路徑用的是 `cudaGraphKernelNodeSetParam`。ROCm 的 HIP Graph 有對等 API 嗎？這會決定 GraCE idea 能否平移。
- **跟 Blackwell 時代新 feature 的互動**：B200 的 FP4、新的 scoped sync、CUDA 13 的更新型 CG API 都會重新洗牌 cost model。SCG 的 profile-based 設計理論上是硬體無關的，但 PI 的具體增益會在 B200 上變多還是變少？看 GraCE 下一代 paper 或 artifact 更新。

---

**Bottom line**：GraCE 回答的核心問題是「如何讓一個 hardware feature 真正發揮它應有的價值」。答案是「加三個 compiler pass」。這個答案的優雅之處在於——它每一個 pass 都在既有編譯器的既有 extension point 上，不另造新 framework、不要求 vendor 配合、不要求應用改程式。這是我會持續 benchmark 自己的工作品質的標準。

[^paper]: Ghosh et al., *GraCE: Unlocking CUDA Graphs with Compiler Support for ML Workloads*, Proceedings of the 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI'26), July 13–15, 2026, Seattle, WA, USA. [USENIX page](https://www.usenix.org/conference/osdi26/presentation/ghosh)，[PDF](https://www.usenix.org/system/files/osdi26-ghosh.pdf)，[Artifact](https://github.com/csl-iisc/GraCE-OSDI26-Artifact)，[YouTube](https://www.youtube.com/watch?v=p4Zq535V1j8)。
