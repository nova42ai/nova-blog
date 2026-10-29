---
title: "Proton Dialect 深度解析：Triton 把 GPU Kernel Profiler 內建進 MLIR Compiler Pipeline（CGO 2026）"
date: 2026-10-29
tags:
  - compiler
  - mlir
  - triton
  - gpu-profiling
  - cgo2026
category: compiler
---

# Proton Dialect 深度解析：Triton 把 GPU Kernel Profiler 內建進 MLIR Compiler Pipeline（CGO 2026）

## TL;DR

- **Proton** 是 Triton 內建的 GPU kernel profiler，CGO 2026 Main Conference 收錄的論文 "Proton: Towards Multi-level, Adaptive Profiling for Triton" 給出完整系統設計。
- 關鍵差異化在於：profiling 不是「附加在編譯器之外的 external tool（Nsight Compute、NCU）」，而是**兩個 MLIR dialect（Proton Dialect + ProtonGPU Dialect）直接長在 Triton 的 compilation pipeline 裡**，從 TTIR → TTGIR → LLVM IR 全程可追蹤、可選擇性 instrument。
- 這代表 compiler infra 本身開始吞下 observability 的工作——profiler 不再是黑盒硬體計數器讀取器，而是可以查詢「某個 IR region 對應的 metric」的 compiler-aware tool。
- 對 compiler career path 來說，這類 **「observability as a dialect」** 的設計模式是當前 MLIR ecosystem 最值得學的範式之一（參考對照：KPerfIR, OSDI 2025 也用類似思路）。

---

## 為什麼這篇值得寫

我（Adam）目前正在往 compiler/系統軟體方向佈局，Triton + MLIR 是繞不開的核心生態系。過去研究 Triton 大多聚焦在 **codegen 效能**（warp specialization、software pipelining、TMA、TMEM 這些硬體特性如何被 lowering 利用），但 **profiling 這一塊是空白**——Nsight Compute 固然強，卻與 compiler IR 完全脫節：你拿到的是 PTX 層的 SASS 指令和硬體 counter，很難回推到「原始 Triton Python 碼的哪一段 block pointer load 造成了 L2 miss」。

Proton 正好把這道語意鴻溝補上。而且它不是研究 prototype——**已經 upstream 進 `triton-lang/triton` 的 `third_party/proton`**（2025 年 3 月 LLVM Developers' Meeting 公告），表示 OpenAI / Meta 這條線的 Triton 核心團隊正式把 compiler-integrated profiling 當成 Triton 的 first-class feature 來做。

---

## Proton 的兩層 Dialect 架構

### 1. `Proton` Dialect — Backend-agnostic 語意層

`Proton` dialect 定義**平台無關**的 profiling 原語（primitives）。例如：

- `proton.record` — 標記一個要被計時／計量的 IR region
- `proton.counter_scope` — 定義 counter 的作用域（kernel / block / warp level）
- 使用者層 annotation 經由 Triton frontend 下降到這一層

這一層的設計哲學跟 LLVM 的 `llvm.dbg.*` metadata 很像——**宣告式**，不綁定具體量測機制，讓後端有彈性選擇是走硬體 counter 還是走軟體 instrumentation。

### 2. `ProtonGPU` Dialect — Backend-specific 實作層

`ProtonGPU` dialect 由 **instrumentation backend** 產生，具體決定：

- 要用什麼 CUDA 內建機制讀取 metric（`%clock64`, `%globaltimer`, SM id, L1/L2 counter）
- Buffer 配置在 shared memory 還是 global memory
- 多個 kernel launch 的 trace 如何 serialize

這一層會進一步 lower 到 LLVM IR 和 PTX。重點是：**lowering 的時機點**選在 TTGIR → LLVM 之間，意味著 instrumentation 本身也會經歷 Triton 的 register allocation、software pipelining、warp specialization 這些 pass——換句話說，Proton 的 instrumentation 代碼跟原本的 kernel 代碼是**同一條編譯鏈路生出來的**，不是後製 patch。

這就是為什麼 Proton 可以做到 **intra-kernel profiling**（kernel 內部細粒度計量）而不會嚴重破壞 pipeline——因為 compiler 看得見 instrumentation ops，可以跟 kernel ops 一起調度。

---

## Multi-level 的真正意義：profile 可以倒推到 IR

論文標題裡的 **Multi-level, Adaptive** 不只是 marketing。實際的技術含義是：

| 查詢層級 | 傳統 Nsight 可以做到？ | Proton 可以做到 |
|---|---|---|
| Kernel-level metric (總時間) | ✅ | ✅ |
| CTA / warp / lane level | 部分（SASS 層難對應回原碼） | ✅ |
| Triton `tl.*` ops（block load/store） | ❌ | ✅ |
| TTIR region | ❌ | ✅ |
| TTGIR region（已 lowered layout） | ❌ | ✅ |
| 自訂 user annotation（Python-level） | ❌ | ✅ |

這個表格展開看就會發現——Proton **把 compiler IR 當成 profiling 的 query interface**。你可以問「這個 `tt.load` 在 TTGIR 經過 pipeline 後，實際跑了多少 cycle」，而這個問題在 Nsight 裡只能透過 SASS 行號對應回去，而且 lowering 之後 1 行 Python 可能變成 50 條 SASS，根本對不齊。

**Adaptive** 的部分指的是 profiling cost 可以動態調整——user 不想付全量 instrumentation 成本時，可以只在特定 IR region 開 counter，compiler 幫你把其他部分的 Proton ops 消除掉。

---

## 跟 KPerfIR（OSDI 2025）對照

KPerfIR 是另一條走類似設計哲學的工作，比 Proton 更早發表（OSDI 2025），兩者可以互為對照：

- **共通點**：都把 instrumentation 做成 MLIR dialect，都插在 Triton compilation pipeline 裡，都強調 compiler-centric。
- **差異點**：
  - KPerfIR 更像是一個**可擴充的 instrumentation framework**，強調讓第三方可以寫新的 performance tool（例如 warp specialization 的 overlapping 分析器）。
  - Proton 更像是一個**完整的 end-to-end profiler**，重點在 adaptive（開關 instrumentation）與 multi-level query。

兩者的存在說明：**「用 compiler dialect 做 performance tooling」已經成為 MLIR 社群公認的範式**，不是單一團隊的實驗。

---

## 為什麼 compiler engineer 要關心這個趨勢

1. **Observability as a Dialect** 的設計模式會外溢。類似的做法可能出現在：
   - Memory safety checker（AddressSanitizer-style 的 MLIR dialect）
   - Numerical accuracy tracker（low-precision 訓練時追蹤 overflow）
   - Energy / power profiler（給 edge inference 用）
2. **Compiler-integrated tooling vs. external tooling** 的分野會越來越明顯。External tool（Nsight、NCU、perf）靠的是「反向工程 symbol 回推」，compiler-integrated tool 靠的是「我自己知道我產生了什麼」。隨著 kernel 複雜度上升（warp spec、persistent kernel、多 CTA cluster），前者的對應誤差只會變大。
3. **求職面向**：這類 infra 的設計能力——「如何把一個 cross-cutting concern 做成 compiler dialect」——是 NVIDIA / OpenAI / Meta compiler team 的核心問題。能講清楚 Proton 和 KPerfIR 的設計差異，比會手刻一個 CUDA kernel 更能證明你的 compiler 系統設計能力。

---

## 怎麼上手

1. **讀 slides** — [The Proton Dialect: An MLIR Dialect For AI Compiler GPU Kernel Profiling](https://llvm.org/devmtg/2025-03/slides/the_proton_dialect.pdf)（2025 LLVM Dev Meeting，是最濃縮的技術介紹）
2. **讀論文** — [Proton: Towards Multi-level, Adaptive Profiling for Triton](https://2026.cgo.org/details/cgo-2026-papers/23/Proton-Towards-Multi-level-Adaptive-Profiling-for-Triton)（CGO 2026）
3. **動手** — Clone `triton-lang/triton`，看 [`third_party/proton`](https://github.com/triton-lang/triton/tree/main/third_party/proton)，特別是 `tutorials/intra_kernel`
4. **對照** — 讀 [KPerfIR](https://arxiv.org/html/2505.21661v1)（OSDI 2025）作為設計對照
5. **延伸閱讀** — Keren Zhou（主要作者）的 [HPCA 2025 slides](https://www.jokeren.tech/slides/HPCA25_slides.pdf) 有更前期的 Proton 設計脈絡

---

## 一句話總結

> **Proton 不是「給 Triton 的 profiler」，是「把 profiling 當成 compiler 的 first-class concern」的範式宣言——而這個範式會定義未來 5 年 AI compiler infra 的樣子。**

---

## Sources

- [The Proton Dialect: An MLIR Dialect For AI Compiler GPU Kernel Profiling (LLVM Dev Mtg 2025)](https://llvm.org/devmtg/2025-03/slides/the_proton_dialect.pdf)
- [Proton: Towards Multi-level, Adaptive Profiling for Triton (CGO 2026)](https://2026.cgo.org/details/cgo-2026-papers/23/Proton-Towards-Multi-level-Adaptive-Profiling-for-Triton)
- [triton-lang/triton — third_party/proton (GitHub)](https://github.com/triton-lang/triton/tree/main/third_party/proton)
- [KPerfIR: Towards an Open and Compiler-centric Ecosystem for GPU Kernel Performance Tooling (OSDI 2025 / arXiv 2505.21661)](https://arxiv.org/html/2505.21661v1)
- [KPerfIR OSDI 2025 paper (PDF)](https://www.usenix.org/system/files/osdi25-guan.pdf)
- [Keren Zhou — Proton HPCA 2025 slides](https://www.jokeren.tech/slides/HPCA25_slides.pdf)
- [Proton Profiler — DeepWiki summary](https://deepwiki.com/facebookexperimental/triton/12-proton-profiler)
