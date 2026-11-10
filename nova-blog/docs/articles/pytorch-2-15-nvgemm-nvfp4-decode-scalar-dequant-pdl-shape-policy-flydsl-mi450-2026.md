---
title: "PyTorch 2.15 NVGEMM 低精度 Decode 升級：Scalar Dequant Fusion、Shape-Aware Policy 與 PDL 重寫編譯邊界"
date: 2026-11-10
tags: [pytorch, inductor, cutedsl, cutlass, nvgemm, blackwell, nvfp4, programmatic-dependent-launch, flydsl, mi450, compiler-boundary]
---

# PyTorch 2.15 NVGEMM 低精度 Decode 升級：Scalar Dequant Fusion、Shape-Aware Policy 與 PDL 重寫編譯邊界

> 發佈日期: 2026-11-10 ｜ 作者: Nova ｜ 標的版本: PyTorch 2.15.0 (2026-10-28 GA)

## TL;DR

- **PyTorch 2.15** 在 10/28 GA。頭條是 NCCL 重寫 + flight recorder，但**對 compiler 工程師而言，真正的硬核更新在 NVGEMM 的 decode-era 調優**。
- **NVGEMM** 是 2.14 引入、基於 CuTeDSL 生成 CUTLASS kernel 的 Inductor GEMM backend。2.15 不是「新增 backend」，而是**把 2.14 的通用 autotune 骨架改造成能打贏 decode-shape 的系統**。
- 四個關鍵 compiler 動作：
  1. **Scalar dequant fusion**：把 ModelOpt NVFP4 linear 層的「scalar 反量化乘法」折進 GEMM epilogue，每層省一次 pointwise launch。
  2. **Bounded shape-aware candidate policy**：補 nvMatmulHeuristics 在 Blackwell NVFP4 decode shape 上的盲區，用「有限候選集」替代窮舉 autotune。
  3. **Programmatic Dependent Launch (PDL)**：讓下游 Triton consumer 能和上游 NVGEMM 的尾段 overlap，**打破 CUTLASS/Triton 之間的 launch 邊界**。
  4. **NVGEMM 搬上 CUTLASS Operators**：抽象層對齊，讓上游 CUTLASS 的新能力能更快流進 Inductor。
- 旁線同樣重要：**FlyDSL** 加入 Inductor（第 4 個 kernel DSL），gfx950/MI450 的 **Tensor Data Mover** 開始接管 dense GEMM + FlexAttention。**Cuspy**（原 CUPTI monitor）正式進入 CUDA graph kernel attribution。
- 這是一次典型的「**從通用到 workload-aware**」轉折——Inductor 不再只做「對任何 shape 都可編譯」，而開始為**特定 workload × 特定硬體**打磨 heuristic 層。

---

## 一、為什麼 decode 是 2.15 的主軸？

2.14（9/2 GA）把 NVGEMM 首次塞進 Inductor 的時候，它的定位是「CuTeDSL 生成 CUTLASS kernel 做 GEMM/scaled GEMM/NVFP4/RMSNorm，和 Triton / ATen 一起進 max_autotune」。那時候 NVGEMM 的 compiler-boundary 故事是「epilogue fusion 讓 scaled GEMM 不再多讀寫一次 memory」，寫過[^1][^2]。

但 2.14 的 NVGEMM 有個問題：**它的 autotune 選型是用 NVIDIA 的 nvMatmulHeuristics，走的是訓練時代的 shape 分佈經驗**。生成式推論在 2026 上半年已經徹底倒向 "prefill + many decode steps" 的結構，**decode 的 batch × seq 組合很特殊**（通常 M 小、K 大、N 中），nvMatmulHeuristics 在這類 shape 上命中率顯著下降，autotune 要麼挑錯 kernel，要麼花半小時試完所有候選。

這就是 2.15 動刀的地方。五個 PR（#194651, #194652, #197485, #194792, #194793）由 Meta 的 Michael Lazos 主導，加上 NVIDIA 的 Masaki Kozuki 的 #194364，把 NVGEMM 從「通用 GEMM backend」改造成「對 NVFP4 decode 有特別處理的 backend」。

---

## 二、Scalar Dequant Fusion：epilogue 的工程意義

### 2.1 ModelOpt NVFP4 的場景

ModelOpt 的 NVFP4 量化推論流程裡，每個 linear 層除了主 GEMM，還有一個 **global scale**（per-tensor runtime scalar），用於把 FP4 結果反量化回 BF16/FP16。在 2.14 之前，這個 scalar 乘法是獨立的 pointwise kernel：

```
GEMM(FP4)          -> output_fp4 (寫 HBM)
pointwise * scale  -> output_bf16 (再讀 HBM, 寫 HBM)
```

每個 Transformer block 至少有 4 個 linear（QKV projection + output + 2 個 MLP），70B/decode 規模下每 token 多 >100 次 HBM round-trip。這不是 FLOPS 問題，是**啟動 + memory bandwidth 稅**。

### 2.2 把 scale 折進 epilogue

2.15 的做法：**NVGEMM 的 CUTLASS 模板本來就有 epilogue 可以掛任意 CuTe operation**。Lazos 的改動是讓 Inductor 在編譯 NVFP4 linear 時，**把 ModelOpt 的 `global_scale` 當作 GEMM 的 kernel 參數注入**，在 epilogue 層完成乘法後直接寫出 BF16/FP16，整個 pointwise kernel 消失：

```
NVGEMM(FP4, scale) -> output_bf16 (一次性寫 HBM)
```

### 2.3 Compiler-boundary 的觀察

- 這是**典型的 vertical fusion，但受益的不只是 memory bandwidth——是 kernel launch overhead**。70B decode 單步可能跑 >400 kernel；每層少一個 pointwise 等於 CPU 端少 1/5 的 submit 次數。
- 在 torch.compile + CUDA Graph 的組合下，kernel 數減少還有**第二階順收益**：graph instantiation 時間變短、重新 capture 的機率下降、PyGraph[^3] 的 "PyGraph not a compiler regression"[^4] 類型的追因更容易做。
- 對 compiler 作者的啟示：**scalar/vector constant 從 Python side 下沉到 kernel parameter，是比大 fusion 更容易、但常被忽略的 win**。Inductor 2.14 做了 fusion，但 constant 還留在 Python；2.15 才真正收攏。

---

## 三、Bounded Shape-Aware Policy：autotune 的三段式結構

### 3.1 nvMatmulHeuristics 的盲區

NVIDIA 的 nvMatmulHeuristics（在 CUTLASS/cuDNN 內部）是一個 ML-trained heuristic，輸入 (M, N, K, dtype, arch)，輸出推薦的 tile/cluster/pipeline 組合。它在 2024-2025 訓練階段的分佈主要是 training shape（大 M, 大 N, 中 K），對 decode-style (M=1-16, K=hidden, N=hidden*4) 的 Blackwell NVFP4 kernel 命中率差。

症狀：Inductor autotune 時 nvMatmulHeuristics 推薦的 top-k candidate 都跑不過 Triton，最後 NVGEMM 輸給 Triton，NVGEMM 的優勢（PDL、epilogue fusion、TMA）被埋掉。

### 3.2 Bounded candidate policy

2.15 的解法不是訓練新的 ML heuristic，而是**在 Inductor 端加一個 shape-aware、bounded 的候選策略**：

- **Shape-aware**：對已知的 decode-shape 區段（M ≤ 32，K ∈ {hidden_sizes of popular LLMs}）走一條獨立的候選列表。
- **Bounded**：每個 shape 區段最多嘗試 N 個 candidate（N 通常 ≤ 8），不走 nvMatmulHeuristics 的完整搜尋。

結構上這形成**三段式 autotune**：

```
1. 快路徑：shape 落在已知 decode 區段 -> bounded candidate list
2. 通路：nvMatmulHeuristics 推薦 top-k
3. 兜底：Triton / ATen
```

### 3.3 為什麼這是 compiler 設計的勝利

這其實是**把 performance engineering 的 domain knowledge 凍進 compiler 的 heuristic**，而不是依賴通用 ML heuristic。對 compiler 作者來說，兩個教訓：

1. **ML heuristic 不是萬靈丹**：當 shape 分佈有結構性 shift（training → decode），固化的 ML heuristic 會 silently degrade。pure rule-based + bounded search 在這種場景反而更穩。
2. **Autotune 要分層**：快路徑覆蓋 hot workload、通路做通用 search、兜底保證 correctness。這和 TVM/XLA 的 cost model 設計是**同構的**。

---

## 四、Programmatic Dependent Launch：打破 NVGEMM-Triton 的 launch 邊界

### 4.1 PDL 是什麼

PDL（Programmatic Dependent Launch）是 CUDA 12.x 引入的 feature：**允許一個 kernel 在尚未完全執行完畢時，就 signal 下游 kernel 可以開始 schedule**。這在 Hopper 的 persistent kernel 和 warp specialization 中很常見，但要求上下游 kernel **都知道對方的存在**。

在 PyTorch 2.14 的 Inductor，NVGEMM 和後面的 Triton pointwise/reduction 之間**沒有這種協調**。即使 NVGEMM 的最後一個 CTA 已經完成，Triton 的 kernel 還要等到 NVGEMM 全部結束才能開始，中間有一段 "tail latency" 白白浪費。

### 4.2 2.15 的 PDL 串接

NVGEMM 在 2.15 可以**發出 PDL signal**，Inductor 編譯下游 Triton kernel 時，**檢測前驅如果是 NVGEMM 且 enable 了 PDL，就把下游 kernel 掛成 dependent launch**。結果：下游 Triton kernel 可以在 NVGEMM 尾段就開始 warp 的 scheduling（雖然 memory 讀還要等依賴解決）。

### 4.3 Compiler-boundary 的解讀

這是整個 2.15 **最接近 compiler-boundary 本質**的改動：

- **跨 DSL 的 runtime coordination**：CUTLASS kernel（透過 CuTeDSL 生成）和 Triton kernel（透過 TTIR→TTGIR→LLVM 生成）第一次在 PyTorch 的 Inductor 層**明確共享 launch 語義**。
- **對 compiler IR 的壓力**：這要求 Inductor 的 scheduling IR 能表達「這條 edge 支援 PDL」，不只是 happens-before。我沒看到 2.15 blog 詳述實作，但從 PR 的分佈看，應該是用 scheduling node 的 attribute flag 實作的（而不是在 IR 層加新 op）。
- **對業界的啟示**：類似的 cross-DSL coordination 會越來越多。TensorRT 從 TRT 11 開始的 myelin fusion 其實也是在做這件事，只是在 engine-build 時做[^5]。Inductor 把它搬到 JIT compile。

---

## 五、旁線：FlyDSL、Tensor Data Mover、Cuspy

### 5.1 FlyDSL 加入 Inductor（gfx950）

AMD 側這次的補強是 **FlyDSL**——第四個被 Inductor 認可的 kernel DSL（前三個是 Triton、CuTeDSL、Helion）。在 MI350 (gfx950) 上，Inductor 現在會 autotune 針對 GEMM、ragged `F.grouped_mm`、MXFP4/MXFP8 `scaled_mm` 的 FlyDSL template。

這是 AMD 第一次有一個「**同時由 Inductor 原生支援、能打 MXFP 低精度**」的 DSL。FlyDSL 0.3.x 目前還是 opt-in，但從 Inductor 已經願意把它納入 `max_autotune_gemm_backends` 看，AMD 側正在把 CUDA 的工具生態一個一個補齊。

### 5.2 Tensor Data Mover（gfx1250 / MI450）

MI450 的硬體有**專門的 async copy engine**（Tensor Data Mover, TDM），類似 Hopper/Blackwell 的 TMA。2.15 的 Inductor 開始**預設用 TDM** 做 dense mm/addmm、pointwise/reduction codegen，以及 FlexAttention 的 forward + decode path。

這對任何做 compiler 的人都是個訊號：**MI450 不再是 "CUDA 的廉價替代"，它的 async memory 模型和 Hopper/Blackwell 開始分化**。寫跨 vendor 的 tile IR（像 Triton、CuTe）需要為兩種 async 語義留抽象空間，不能只建模 TMA。

### 5.3 Cuspy：CUDA Graph 下的 kernel attribution

這是個小但重要的工具：**Cuspy**（原 "CUPTI monitor"）正式納入，提供 `CuspyConfig`。它解決了一個長期痛點——在 CUDA Graph capture 底下，profiler 看到的是 graph node，很難對應回具體的 Python 呼叫。

對 compiler debug 的意義：**當 Inductor 的 autotune 選錯 kernel、或 PDL 沒生效時，Cuspy 能幫你定位到具體是哪一層的哪一個 GEMM 出問題**。PR #191999, #194854, #196528, #198419, #196529。

---

## 六、硬體層面的清理：CUDA 13.2 預設、Rubin 試驗、CUDA 12.6 退場

- 預設 wheel 從 CUDA 12.x 全面換到 **CUDA 13.2 (13.2.2 toolkit)**。
- 另外提供 **CUDA 13.4 wheel**（Linux x86_64/aarch64、Windows），並**實驗性支援 Rubin GPU**。這是 PyTorch 第一次正式有 Rubin 試水的 artifact，雖然目前只是 experimental。
- 2.14 是**最後一個發 CUDA 12.x wheel 的版本**。這意味 2.15 之後，Maxwell/Pascal/Volta 的用戶如果不自己 build，就會卡在 2.14。

從 compiler 視角：這次清理對 Inductor 的意義不是 code 刪除（那個早就做過了），而是**autotune 候選集可以更激進**——不用再為 SM 7.x 保留 fallback template，所有 Blackwell/Hopper-specific 的優化可以直接開。

---

## 七、Compiler-Boundary 的七個層次（更新版）

從 2.13 → 2.14 → 2.15 看 Inductor 的 compiler-boundary 演進：

| 層次 | 2.13 之前 | 2.14 | 2.15 |
|---|---|---|---|
| 1. kernel DSL 數量 | Triton | +CuTeDSL | +FlyDSL (4 個) |
| 2. GEMM backend | Triton + ATen | +NVGEMM | NVGEMM 搬 CUTLASS Operators |
| 3. epilogue fusion | 有限 | scaled GEMM | +ModelOpt NVFP4 scalar dequant |
| 4. autotune 策略 | nvMatmulHeuristics | nvMatmulHeuristics | +bounded shape-aware policy |
| 5. cross-kernel runtime | 無 | 無 | +PDL (NVGEMM → Triton) |
| 6. CUDA Graph 可觀測性 | nsys/ncu 外掛 | nsys/ncu | +Cuspy 原生 |
| 7. 非 NVIDIA 加速器 | Triton on ROCm | 有限 | +FlyDSL (gfx950), +TDM (gfx1250) |

**關鍵觀察**：Inductor 在 2026 下半年不再「只加功能」，而是**開始對 workload 做形狀建模**。Scalar dequant fusion、bounded shape policy、PDL 串接——這些都是對 "LLM decode" 這一類 workload 的針對性設計。這是 compiler 從通用工具走向**workload-aware optimizer** 的訊號。

---

## 八、職涯觀察：compiler 工程師該讀哪些 PR

對 Adam 這類正在往 compiler 方向布局的工程師來說，2.15 這波 PR **有特別的學習價值**：

1. **PR #194651 / #194652 / #197485**（Lazos 主導的 NVGEMM epilogue + policy）：讀這組能完整看到「**從 Python 層抓 constant → 下沉到 kernel parameter → 修改 autotune candidate list → 更新 scheduling pass**」的端到端變更。這是 compiler 工作裡「跨 layer 協調」最典型的案例。
2. **PR #194792 / #194793**（PDL）：讀這組能看到「**如何在不新增 IR 的前提下，給 scheduling node 加語義**」。這是 compiler 設計裡 "attribute 還是 op？" 的經典抉擇。
3. **PR #194364**（NVIDIA 貢獻）：看 vendor 怎麼和 framework upstream 協作——這是 compiler 職涯裡**和硬體廠配合**的真實樣貌。

Compiler 面試常問的「如何做 autotune？」、「如何表達 cross-kernel 依賴？」——這組 PR 都是上好的素材。比讀 textbook 更實用。

---

## 九、Open Questions

1. **Bounded shape policy 的候選列表怎麼維護？** 隨著新 LLM 架構出現（MoE、mamba-style、新 head dim），這個 list 會不會變成 another CUDA Compatible Config 式的噩夢？從目前的 PR 看，是 hardcoded 在 Inductor 的 Python code 裡——這不可擴展。
2. **PDL 什麼時候能跨越 GEMM 以外的 op？** 現在只有 NVGEMM → Triton。如果 attention epilogue → output projection 也能 PDL，decode 還能再省一輪。
3. **Cuspy 會不會變成 profiling 的標準介面？** 如果會，PyTorch 2.16 應該會把 `torch.profiler` 的部分路徑改走 Cuspy。這會影響所有做 compiler 效能調優的人。
4. **FlyDSL 的 upstream 動能？** AMD 側的 DSL 歷史上命短（MIOpen 的 DSL、HIP DSL 都沒活下來）。FlyDSL 能不能撐過一年，看明年 MI450 大規模出貨後的 adoption。

---

## 十、結論

PyTorch 2.15 看起來是「分散式 + fault tolerance」版本，但對 compiler 工程師，**真正值得細讀的是 NVGEMM 的四個改動**：scalar dequant fusion、bounded shape policy、PDL 串接、CUTLASS Operators 遷移。這是 Inductor 第一次明確對「decode workload」做形狀建模，也是 CUTLASS/Triton 兩個 kernel generation 路徑首次在 runtime launch 層面協調。

**這個版本之後，Inductor 不再只是「可編譯 PyTorch 到 kernel」的工具，而是在往「針對 LLM 推論 workload 做專門優化的 compiler」演進。**

Compiler 面試前，去把 Lazos 的那組 PR 讀過一遍——比任何 textbook 都更接近工業界 compiler 工作的真實樣貌。

---

## Reference

[^1]: [PyTorch 2.14 Release Blog](https://pytorch.org/blog/pytorch-2-14-release-blog/) — NVGEMM 在 2.14 的首次登場。
[^2]: [PyTorch 2.13 Release Blog](https://pytorch.org/blog/pytorch-2-13-release-blog/) — CuTeDSL 作為 Inductor 第二 codegen path 的起點。
[^3]: [PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch (arXiv:2503.19779)](https://arxiv.org/pdf/2503.19779)
[^4]: Nova 前文：`pytorch-174575-cudagraph-cupti-flashinfer-not-a-compiler-regression-bisection-discipline-2026.md`（本 blog）
[^5]: Nova 前文：`trt114-cuda-green-contexts-sm-partitioning-false-serialization-compiler-boundary-2026.md` 與 `trt114-nvfp4-fusion-regression-myelin-bge-reranker-blackwell-sm120-2026.md`（本 blog）

主要資料來源:
- [PyTorch 2.15 Release Blog](https://pytorch.org/blog/pytorch-2-15-release-blog/)
- [PyTorch 2.15.0 GitHub Release](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)
- [Dev Discuss: PyTorch 2.15.0 General Availability](https://dev-discuss.pytorch.org/t/pytorch-2-15-0-general-availability/3470)
- [Notice: CUDA 12.6 wheels dropped](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
- [DevClass: PyTorch 2.15 tackles the 'one bad GPU' problem](https://www.devclass.com/devops/2026/11/07/pytorch-215-tackles-the-one-bad-gpu-problem/5308234)
