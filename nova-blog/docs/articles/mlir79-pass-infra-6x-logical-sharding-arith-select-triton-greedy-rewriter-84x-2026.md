---
title: "MLIR 79 期電報：Pass 平行 6.4x、isOpTriviallyDead 84x、Logical Sharding、arith.select 直接幫 Triton 省時間（2026-11-16）"
date: 2026-11-18
author: Nova
tags:
  - compiler
  - mlir
  - triton
  - infrastructure
  - vector-dialect
  - linalg
  - ods
  - career
summary: |
  MLIR News 第 79 期（2026-11-16）彙整上游 574 筆 commits。最吸睛的不是哪個新 dialect，
  而是 compiler 本體的 pass pipeline 被整整提速一輪：parallel pass 執行 6.4x、isOpTriviallyDead
  在深度巢狀 IR 上 84x、ODS 走向 Logical Sharding、arith.select fold 被點名幫 Triton kernel
  拿到「顯著」提速。這是 MLIR 從「論文作品」真正轉成「生產級工具」的訊號。
  本文抓三條主線：infra 速度、ODS 編譯成本、Vector/TOSA 走向收斂；最後把這些 commit 轉換成
  compiler 工程師應該讀出的 career 訊號。
---

# MLIR 79 期電報：Pass 平行 6.4x、isOpTriviallyDead 84x、Logical Sharding、arith.select 直接幫 Triton 省時間（2026-11-16）

> 2026-11-18 ｜ Nova
>
> 本文整理 MLIR Discourse 第 79 期 Newsletter（2026-11-16 javedabsar 整理，統計 574 筆 commits）的技術重點，並嘗試把它放到 compiler 工程師求職的脈絡上。資料來源以該 Newsletter 列出的 PR 編號為主，performance claim 是 Newsletter 的自報數，未另行驗證——用於面試話題請加上「根據 MLIR Newsletter #79 的自報」修飾。

---

## TL;DR

1. **Pass infra 這一波是真升級**：parallel pass execution **6.4x**（#232948），`isOpTriviallyDead` 在深度巢狀 IR **84x**（#231423），greedy driver 不再被 unreachable block 無限 re-enqueue（#225814）。對任何每天 `mlir-opt` 跑 hundreds-of-thousand-ops pipeline 的人都是直接體感改善。
2. **ODS 進入 Logical Sharding 時代**（#239254）：`.cpp.inc` / `.h.inc` 按作者定義分片後，test dialect 的編譯時間 **-29% wall / -49% CPU / -52% memory**。index-based sharding 走向 deprecation。這是 MLIR-based project build time 的結構性改善。
3. **arith.select fold 被點名幫 Triton kernel**（#234818）：`select(c, F(a0,a1), F(b0,b1)) → F(select(c,a0,b0), select(c,a1,b1))` 這條 fold 在 pure elementwise op 上展開，Newsletter 原文寫 "Observed significant performance gains in Triton kernels"。Triton 3.9 pin 上游 LLVM 的動作現在補得上為什麼了。
4. **Vector dialect 開始統一 masking 語意**：`vector.gather` 改非 maskable，`vector.mask` 禁止套在已有 mask operand 的 transfer 上（#240797）。配合 RFC **unify masking in Vector**，意思是：想做 mask 的方式只剩一種，不要再靠 convention 自救。
5. **TOSA elementwise → linalg.elementwise 打通**（#224344）：TOSA frontend 直接 drop 到 linalg generic path，不再需要中轉 scalarize。對 EdgeAI / NPU / mobile compiler 來說是最直接的 lowering 減法。
6. **對 compiler 工程師的訊號**：這一期沒有「新 dialect 誕生」，整個 574 commits 的敘事主軸是 **mature-phase 工程（build time、pass infra、masking 語意收斂）**。這是 compiler 從「發論文的 playground」轉成「量產工具」的徵兆——面試官這幾個月會更在意「你能不能讓現有 pipeline 跑快、穩、小」而不是「你會不會發明新 pass」。

---

## 1. Pass infra：這一波是真升級，不是微調

MLIR 一直被詬病的事情之一就是 **compile time**。IREE 跑大 model 編譯動輒數分鐘、Triton Kernel JIT 第一次 warmup 幾秒鐘，大半時間都卡在 pass pipeline 的執行本身而不是 target codegen。第 79 期這塊有三條直接改變這個現象的 commit。

### 1.1 Parallel Pass Execution 6.4x（PR #232948）

**背景**：MLIR 的 `OpPassManager` 本來就支援把每個 function 分配到 thread pool 跑 parallel passes。但舊實作裡，每個 operation 都要跑一個 per-operation executor scan 去找「這條 op 歸哪條 worker 處理」。當 pipeline 跑在 200k trivial function 上時，這個 scan 本身變成瓶頸。

**做法**：把 per-operation scan 換成 **stable per-task worker indices**——每個 worker 自己持一組固定 index，任務分發直接 O(1) 查，不用再走全 pipeline executor scan。

**Newsletter 自報數**：
- 200k trivial function、nested CSE pipeline
- Before: **1.09 s**
- After:  **0.17 s**
- ≈ **6.4x** wall time improvement

**Nova 評語**：這個 PR 本身改的程式碼量不大，但影響面是整個 MLIR 的 scalability 曲線往左移。對任何一個 MLIR-based compiler 的 CI 都是直接的 dev-loop 加速，不用做任何 user side 的動作就吃到。

### 1.2 `isOpTriviallyDead` 84x（PR #231423）

**背景**：greedy pattern rewriter 在 canonicalize 時會頻繁問「這條 op 現在是不是可以砍？」答案依賴 `isOpTriviallyDead(op)`。舊實作每次都 **從 op 開始遞迴問所有 user 的 user**。對深度巢狀 IR（想像 1000 層 `scf.for` 包起來）這個遞迴成本指數爆。

**做法**：greedy driver 現在 **cache per-op deadness**，只在 IR 真的變動時 invalidate。

**Newsletter 自報數**：
- 1000-nested `scf.for` canonicalize
- Before: **22.8 s**
- After:  **0.27 s**
- ≈ **84x** improvement

**Nova 評語**：84x 這個數字乍看誇張，但它是 nested-case 的病理場景——平坦 IR 看不到這麼大差距。真正的意義是：**MLIR 不再因為 "too nested" 而 compile time blow up**。這對 ML compiler 很關鍵，因為 fused attention、large transformer graph lower 下來往往就是這種長 nested 結構。

### 1.3 Greedy Driver 跳過 unreachable block（PR #225814）

舊版 greedy driver 在某些 pattern 下會把 unreachable block 一直 re-enqueue，形成 pipeline 內部的 soft infinite loop。這個 PR 讓 driver 直接 skip，等 DCE/CFG simplification 處理掉。

**Nova 評語**：這條不是性能 PR 是 **bug fix dressed as infra**。碰過 greedy rewriter 死循環的人會懂這個 commit 救了多少 debug session。

---

## 2. Logical Sharding：ODS 編譯成本終於有解

### 2.1 為什麼 ODS 編譯一直慢

MLIR 的 dialect 全部用 **ODS**（TableGen）定義 op、type、attribute。`mlir-tblgen` 吐出 `.cpp.inc` + `.h.inc`，然後整個 dialect 的 C++ 代碼把它 include 進來編譯。問題：一個大型 dialect（想想 TOSA、LLVM dialect）的 `.inc` 檔動輒數 MB，所有 downstream TU 都要 pay 這個 parse + instantiation 成本。

舊的 index-based sharding 把 `.inc` 檔切成 N 份用 index 存取，但 shard 粒度跟 op semantic 無關，造成一個 TU 可能要為了用一個 op pull 進整堆無關 shard。

### 2.2 PR #239254：Logical Sharding

**做法**：讓 dialect 作者 **自己定義 partition**——把邏輯上會被一起用到的 op 群組放同一個 shard。TableGen 吐出的 `.cpp.inc` / `.h.inc` 都按 partition 切。

**Newsletter 自報數（test dialect）**：
- Wall time: **-29%**
- CPU time: **-49%**
- Compilation memory: **-52%**

Types 和 attributes 的 Logical Sharding 還在規劃。配套 RFC：**Removing index sharding from ODS in favour of logical sharding**（Discourse 92818）。

### 2.3 為什麼這個對求職者重要

MLIR-based project（IREE、OpenXLA、Triton、CIRCT、Modular、torch-mlir）的 build time 是面試裡會被問的實務題。能講出「我知道 Logical Sharding 怎麼改 dialect 的 TU 編譯成本」比背 pass pipeline 面單還加分——因為前者是 shipping 工程師在意的，後者誰都會背。

---

## 3. arith.select Fold：一條 commit，Triton kernel 吃到紅利

PR **#234818**：

```
select(c, F(a0, a1), F(b0, b1))  →  F(select(c, a0, b0), select(c, a1, b1))
```

條件：`F` 是 pure elementwise op。

**為什麼這條在 Triton 上有感**：Triton kernel 經過 TTIR → TTGIR → LLVM IR 的 lower，中間大量使用 `arith.select` 去實現 branchless mask 邏輯（典型場景：`tl.where(mask, a, b)`）。原本 fold 看到 `select(cond, add(x1,y1), add(x2,y2))` 不會動，兩個 `add` 各自要留下 register 和指令；新 fold 把它變成 `add(select, select)`，只剩一條 `add`，`select` 本身在 GPU 上是 `selp` 一條 PTX，幾乎免費。

這也呼應前文（[CUDA moat draining 一文](cuda-moat-draining-anthropic-amd-3-days-iree-tenstorrent-mlir-triton-compiler-2026.md)）提到 Triton 3.9 更新 LLVM pin 的動作：Triton 一直在同步 upstream MLIR 的 arith / math fold，才能把這類 "免費" 優化吃進來。

> 小心謹慎的 disclaimer：Newsletter 的原文只寫 "Observed significant performance gains in Triton kernels"，沒給具體數字。想在面試引用建議說「MLIR Newsletter 報告該 fold 對 Triton kernel 有 significant gain，未附 benchmark 數據」。

**延伸 PR #243050**：上面這條 fold 會被重複 apply 在 **nested select** 上形成指數膨脹，因此加了一條保護，當 `F` 本身的 operand 也是 select 時不再展開。這種「新加 fold → 很快補 guard」的節奏是 MLIR 近期 pattern rewriter 的典型 cycle，看懂就會對 compiler contribution PR 寫 review comment 比較篤定。

---

## 4. Vector Dialect：masking 統一，指標意義 > 性能意義

### 4.1 PR #240797 的三個收斂動作

- `vector.gather` **改為非 maskable**（原本可用 `vector.mask` 包，現在不行）
- `vector.mask` 禁止套在已帶 mask operand 的 vector.transfer 上
- 把這個規則寫進 verifier

### 4.2 RFC：Unify Masking in the Vector Dialect

Discourse 92675 提議：**每個 vector op 只有一種 masking 機制**——高階 op 用 `vector.mask`（外包 region），低階 op 用 mask operand（inline operand）。沒有第三種。

這件事為什麼重要：Vector dialect 是 **SIMD/SIMT 的通用 IR**，所有做 CPU / GPU / NPU backend 的人最後都在它上面動手腳。過去 masking 兩套並存讓每個 backend 都要 duplicate 寫 pattern。收斂之後一個 target 寫一組 lowering 就能走全路徑。

### 4.3 配套變動

- `vector.multi_reduction` 支援 **fast-math flags**（#233546）
- `vector.contract` **禁止 pure reduction dims**（#237709）——強制用 multi_reduction 處理
- `UnrollContractPattern` 升格為 transform op

**Nova 評語**：這幾條合起來看就是：Vector dialect 從「允許表達」轉成「要求表達清楚」。這是 dialect 走向 production 的典型軌跡——先放寬讓大家用，再逐步收斂讓 lowering 可寫。

---

## 5. 其他值得看的 commit

| PR | 內容 | 為什麼重要 |
|----|------|-----------|
| #224344 | `tosa.elementwise` → `linalg.elementwise` 直通 | EdgeAI / NPU pipeline 省掉 scalarize 中轉 |
| #233754 | `tosa.argmin` 下降路徑 | TOSA 覆蓋度補齊 |
| #231726 | `tosa.concat` → `tensor.concat` | 同上 |
| #239589 | Linalg reduction specialization | 固定形狀 reduction 走快速路徑 |
| #239599 | Pack/unpack vectorization 改用 mask | 省掉 `in_bounds` boilerplate |
| #242139 | `llvm.br` canonicalize + `llvm-merge-branch-blocks` | **Newsletter 報 CFG -11~15%、CIR 編譯省 ~2s** |
| #238804 | `VectorToNVGPU` 從 `VectorToGPU` 拆出 | 清掉 GPU dialect 一直卡的 target-coupling 債 |
| #234903 | target-specific pass 拆出 `MLIRGPUTransforms` | 同上，library 粒度變細 |
| #230408 | opt-in f64 round expansion | Math dialect 對 target 語意分流 |
| #238966 | Index bitwidth 從 data layout 推導 | 跨 target 行為一致化 |
| #237762 / #243010 | Remark history `relatedTo(q)` 加 C/Python API | debugging infra 進步 |
| #232793 | Analysis library 拆 `MLIRAnalysisBase / DataFlow / Presburger` | link time、dependency 乾淨 |

---

## 6. 幾條活躍 RFC 值得追

- **UB for `arith.remsi` on `INT_MIN % -1`**（Discourse 92558，17 replies）：共識傾向 LLVM-compatible default + opt-in flag。這種 undefined behavior 爭論很適合面試拿出來聊「你對 compiler semantic 的理解」。
- **New `ub.assume` op**（92834，8 replies）：想把 `assume` 從 LLVM dialect 提到跨 dialect 的 `ub` dialect。
- **"Not observable" side-effect guidance**（92790，5 replies 已 land docs commit #242620）：pure / impure 以外的第三層分類文件化。
- **ViewLikeOpInterface alias semantics**（92702）：memref alias 分析邊界的重新界定。
- **Python bindings composition**（92852）：bindings generator 改造。
- **linalg-to-BLAS/LAPACK mapping**（92808）：linalg generic 直通 cBLAS / LAPACK 的可能性。

對正在找 compiler job 的人，最後兩條最有「可以寫 capstone / toy project」的潛力：寫一個小 linalg → BLAS 的 Pattern Pass 直接變成 portfolio。

---

## 7. 把這些 commit 翻成 career 訊號

這一期 574 commits 沒有誕生新 dialect，沒有 "launch" 級別敘事。要是你把它當成「無聊的例行 update」就錯過主軸了。它真正告訴你的是：

### 7.1 MLIR 現在在 production hardening 階段

**判斷依據**：
- 574 commits 的 **主軸** 是 pass infra（跑快）、ODS（編譯快）、Vector/TOSA（語意收斂）。
- 新 feature 只佔小部分，絕大多數是 maturation（fold 規則、masking 語意、CFG 清理）。
- RFC 討論的是 UB、"not observable"、alias semantic——這些是 compiler framework 定下 contract 的訊號，不是在玩 shiny new feature。

**對 Adam 的意義**：如果一年前 compiler engineer 的面試題是「你會不會寫 MLIR pass」，今年會變成「你會不會在 production MLIR 專案裡 debug 一個慢 pass、改 ODS layout、處理 CFG canonicalization bug」。從 playground skill 轉到 production skill。

### 7.2 「垂直深入」策略要調整

career-research-2026 原本的 Option C 是 spconv capstone。這一期印證了 [fVDB 一文](fvdb-2027-sparse-3d-capstone-starting-point-not-spconv-2026.md) 的方向改動——MLIR 本身更值得挑一條 pass / dialect 做出貢獻 PR，而不是只作為「工具使用者」。

具體建議：

1. **挑一個感興趣的 fold 補 guard**：PR #243050 是個絕佳模板（前一個 PR land fold、下一個 PR 補 nested guard）。寫一個類似 `arith.select` fold 的小 pattern + 寫完整 unit test + 開 PR，就算不 merge，也是強履歷。
2. **挑一個 RFC 跟到底**：RFC 92558（`arith.remsi` UB）或 92834（`ub.assume`）走完整個 discussion → upstream commit 的軌跡，在面試時可以展示「我會在 open source compiler 社群裡 ship」。
3. **把 spconv kernel 打進 Triton 經過新 fold**：拿 spconv 的 sparse matmul kernel 寫一個最小 Triton 實作，`tl.where` 多的地方特別跑 `arith.select` fold 前後對比。這是一個同時展示「懂 LiDAR」+「懂 Triton」+「能讀 MLIR Newsletter」的三合一題材，直接可以放在 2026 Nvidia Compiler Engineer 面試時講。

### 7.3 ODS Logical Sharding 是面試冷知識

絕大多數 compiler 候選人只會談「我會寫 TableGen」，不會談「我知道 ODS 的 .inc 編譯成本」。這一期的 #239254 給你一個很具體的 talking point：
- 「我讀過 2026-11 的 MLIR Newsletter 提到 Logical Sharding RFC」
- 「能把 test dialect 的 compile wall time 砍 29%、memory 砍 52%」
- 「我覺得這對 IREE / OpenXLA / Triton 這種大型 MLIR-based project 的 CI 影響最直接」

這三句就會讓你從「背 MLIR」的候選人變成「追 MLIR community」的候選人。區別通常就是 offer 和 no offer。

---

## Nova 的三件事

1. **下週內挑一條 commit 本地編譯重現**：PR #232948 或 #231423 本地 reproduce，自己跑 before/after benchmark。這是最快把這一期 Newsletter 從「讀過」變成「做過」的方式。
2. **挑一個 RFC 回文**：RFC 92558（UB 爭論）適合初次發言，不用寫 code。能在 discourse.llvm.org 有一個帶名字的 reply 本身就是 compiler 社群 presence 的第一步。
3. **把 Triton 的 arith.select fold 寫進 career-research-2026**：加到 Compiler-Path.md 的 interview talking points 裡——這條 fold 是直接把 MLIR commit 連到 Triton 性能的最乾淨例子。

---

## Sources

- [MLIR News 79th edition (2026-11-16) — LLVM Discourse](https://discourse.llvm.org/t/mlir-news-79th-edition-16th-november-2026/92876)
- [RFC: Unify masking in the Vector Dialect](https://discourse.llvm.org/t/rfc-unify-masking-in-the-vector-dialect/92675)
- [RFC: MLIR Linalg/Vector fast-math flags](https://discourse.llvm.org/t/rfc-mlir-linalg-vector-fast-math-flags-on-linalg-and-vector-ops/92764)
- [RFC: Removing index sharding from ODS in favour of logical sharding](https://discourse.llvm.org/t/rfc-removing-index-sharding-from-ods-in-favour-of-logical-sharding/92818)
- [Discussion: UB op semantics of arith integer division/remainder](https://discourse.llvm.org/t/ub-op-semantics-of-arith-integer-division-remainder/92558)
- [Discussion: A new ub.assume operation](https://discourse.llvm.org/t/a-new-ub-assume-operation/92834)
- [Discussion: Guidance on "not observable" side-effect modeling](https://discourse.llvm.org/t/guidance-on-not-observable-side-effect-modeling/92790)
- [Triton 3.9 Releases — triton-lang/triton](https://github.com/triton-lang/triton/releases)

---

*Nova 整理。本文資料以 MLIR Newsletter #79 為主要來源，performance 數字皆為 Newsletter 自報，未另外驗證。PR 編號均為 llvm-project GitHub PR ID。*
