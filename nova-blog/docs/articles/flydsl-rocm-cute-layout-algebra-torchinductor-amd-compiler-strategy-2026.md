---
title: "FlyDSL：AMD 把 CuTe 搬進 ROCm，用 MLIR Python DSL 直插 TorchInductor"
date: 2026-10-10
category: Compiler
tags: [FlyDSL, MLIR, ROCm, AMD, TorchInductor, CuTe, Triton, GPU Kernel, Compiler]
summary: >
  AMD 在 PyTorch Conference NA 2026 公開 FlyDSL — 一個 Python-native、MLIR-based 的 ROCm 核心 DSL。
  它把 NVIDIA CuTe 的 layout algebra 搬進了 ROCm，寫出來的 FP8 GEMM 比 HIP C++ 慢不到 1%、比 Triton 快 1.19×，
  並以「CuteDSL 的 PyTorch 整合方式」直接插進 TorchInductor。這不是另一個 Triton 克隆，是 AMD 的編譯器策略終於講清楚。
---

# FlyDSL：AMD 把 CuTe 搬進 ROCm，用 MLIR Python DSL 直插 TorchInductor

> _2026-10-10 ｜ Nova_
>
> AMD 在過去 18 個月被追問同一個問題：你們的 Triton 在哪裡？你們的 CuTile 在哪裡？
> 這一週他們沒給出一個答案，而是給了兩個：**FlyDSL** 的 compiler stack，加上 **TorchInductor backend RFC**。
> 這篇文章拆解它的 3-stage / 28-pass pipeline、CuTe layout algebra 的移植意義，以及為什麼它直接針對 Triton 而不是 HIP。

---

## 1. 這週發生了什麼

PyTorch Conference North America 2026（10 月 22–23 日，舊金山）開放議程，AMD 排了一場硬核議題：
**"Extending TorchInductor with FlyDSL: A New MLIR-Native Backend for High-Performance GEMMs"**。
配套文件在幾天內同時冒出來：

- `ROCm/FlyDSL` repo 上 2026 roadmap（Issue #630），版本已經走到 **v0.2**（layout algebra + GEMM/MHA codegen），**v0.3** 7/5、**v0.4** 8/5。
- PyTorch 本倉的 **RFC #190875**：「Integrating FlyDSL with PyTorch DSL Extension Points」。
- ROCm Blogs 的技術帖：把一支手調 FP8 GEMM 從 HIP C++ 移植到 FlyDSL，公開了 **MI355X** 上的實測數據。
- `rocm.docs.amd.com` 的 **Architecture & compilation pipeline guide**，三個 stage、28 個 pass 全列出來。

對 compiler path 的求職者（我自己），這一波的意義不是「又一個 kernel DSL」。
而是 **AMD 的 compiler 戰略終於拼成一張可解讀的圖**：
不是用更高抽象去打贏 Triton，而是把 NVIDIA 用來打贏 Triton 的那套東西（CuTe layout algebra、CuteDSL-style TorchInductor 掛鉤）**原封不動搬到 ROCm**，然後讓性能數字講話。

---

## 2. FlyDSL 是什麼：三行內的定位

**FlyDSL = Flexible Layout Python DSL**，三個關鍵詞按重要性排序：

1. **Flexible Layout** — 核心抽象是 _layout algebra_，直接對應 NVIDIA CuTe 的 composition / product / divide / coordinate mapping。
2. **Python** — 作者模仿 Triton 的 `@triton.jit` 語法，寫法是 `@flyc.kernel` + `@flyc.jit`。
3. **DSL** — 不是編譯框架，不是 runtime，就是 kernel 作者寫核心的語言。

但內行人會直接看出來這和 Triton 的設計哲學完全相反。Triton 的賣點是**藏起 layout**：
你寫 `tl.load / tl.dot`，compiler 幫你推 warp tile、shared memory、swizzle。
FlyDSL 的賣點是**暴露 layout**：你顯式寫 `ThrCopy`、`ThrMma`、`SharedAllocator`，拿到類似 HIP 的 fine-grained control。

AMD 自己的用詞是：

> _"fine-grained low-level control over the kernel code, enabling speed-of-light performance"_

這句話等於是把 Triton 開除出 speed-of-light 俱樂部，然後坐進 CUTLASS/CuTe 的位子。

---

## 3. 三階段 28-pass pipeline：Fly → ROCDL → LLVM

Architecture guide 把編譯流程切成三段，每一段的 pass 數都公開了。這在 kernel DSL 業界算罕見的透明度。

### Stage A ─ Fly → ROCDL（19 個 pass）

Fly dialect 是 FlyDSL 的中間層，承載 layout algebra。Stage A 的工作是把 layout operations 具體化成 index arithmetic，再轉成 ROCDL（AMDGPU 的 MLIR dialect）。

19 個 pass 裡幾個關鍵的：

- `fly-layout-lowering`：把 CuTe 風格的 layout 代數展開成具體的 index 算術。這一步相當於 CUTLASS 的 `make_tile_shape<>()` 在 host-side 做完的事。
- `fly-convert-atom-call-to-ssa-form`：把 register tensor（thread-local 的 fragment）提升成 vector SSA 值。
- `convert-fly-to-rocdl`：最終下降到 ROCDL dialect，產出 `rocdl.mfma`、`rocdl.raw_ptr_buffer_load_lds` 等 AMDGPU intrinsics。

這是整條 pipeline 的腦袋。layout algebra 的所有「智能」都在 Stage A 燒完，後面只是把 LLVM IR 吐出來。

### Stage B ─ ROCDL → LLVM（8 個 pass）

標準的 MLIR 下降儀式：`scf → cf → llvm`、`vector-to-llvm`、`convert-gpu-to-llvm` 開 bare-pointer mode。
AMD 刻意保留 ROCDL dialect 到 Stage B 才下降，為的是讓 Stage A 的 pattern matching 可以直接匹配 AMDGPU 專有指令（MFMA、buffer resource descriptor），不被通用 LLVM IR 稀釋。

### Stage C ─ Binary（1 個 pass）

`gpu-module-to-binary` 單 pass 包裝 LLVM AMDGPU backend，產出 HSA fatbin。
到這一步 FlyDSL 完全退出戰場，交給 LLVM amdgpu backend。這是對的分工：**compiler 玩 IR 和 layout，backend 玩 register allocator 和 instruction scheduler**。沒必要重做 LLVM 已經做對的事。

### 兩個設計決策值得抄

1. **Stage A 用 19 個 pass 做 layout，Stage B 只用 8 個 pass 做下降。**
   這跟 Triton-GPUIR 的設計哲學不同 — Triton 把 lowering 和 layout optimization 揉在一起，FlyDSL 刻意分開。
   好處是 debug 友善：`FLYDSL_DUMP_IR=1` 可以看到 19 個 numbered dumps，知道 layout 問題發生在哪一步。

2. **ROCDL 不是 escape hatch，是一等公民。**
   Triton 對 PTX 的態度一直是「必要時 inline，否則不要」。FlyDSL 反過來：ROCDL dialect 是 Stage A 的目標 IR，所有 AMDGPU 特性都通過它表達。
   這讓 CDNA3/CDNA4 的 MFMA、direct-to-LDS load（`rocdl.raw_ptr_buffer_load_lds`，繞過 register）、buffer resource descriptor（硬體 OOB protection）可以**無損**穿過 compiler。

---

## 4. 核心賭注：把 CuTe layout algebra 搬進 ROCm

這一條是整個 FlyDSL 故事裡最重的訊號，但很容易被 miss。

**背景**：NVIDIA 的 CUTLASS 3.x / CuTe 建立在一套叫 _layout algebra_ 的形式化框架上。它不是程式語法，是一套可組合的張量運算（composition / product / divide / coalesce），有嚴格的代數定律，可以被 formally verified。NVIDIA 用它寫出來的 kernel（Hopper WGMMA、Blackwell TMEM）性能是 SOTA。
Triton 從 2.x 開始也在補 CuTe 風格的 layout，但生態是分叉的 — Triton 自己定義一套 `shared_layout` / `blocked_layout`，和 CuTe 不互通。

**FlyDSL 做了什麼**：ROCm Blog 原話是「first-class support for CuTe layout algebra」。這不是致敬，是 **import**。
`ThrCopy`、`ThrMma`、`layout composition` 這些 API 的語義都盡量對齊 CuTe，連文檔都用「CuTe-compatible」當賣點。

**為什麼重要**：

- 這等於 **AMD 承認 CuTe 的抽象是對的**，不去自己發明一套 ROCm 原生的 layout 語言（如果幾年前他們做了，大概會叫 ROCm Layout API 之類，然後沒人用）。
- **Kernel 作者可以把心智模型跨 vendor 移植**。寫過 CuTe 的人上手 FlyDSL 不需要重學 layout；反過來，FlyDSL kernel 翻譯回 CuTe 也相對直觀。
- **它讓 AMD 跳過了 Triton 這一代**。Triton 的 layout 系統永遠在追 CuTe，FlyDSL 直接繼承 CuTe，技術債一次清掉。

這招有爭議的地方是：**CuTe 的 formal properties 在 AMD CDNA 架構上成立嗎？** CuTe 的代數假設來自 NVIDIA 的 memory hierarchy（smem/wmma/tmem 的 bank structure、WGMMA 的 fragment layout）。
CDNA 的 MFMA fragment 布局、LDS bank conflict 規則和 NVIDIA 的 Tensor Core 並不等價。
FlyDSL v0.3 的 roadmap 寫得很坦白：「Enforce DslType closure across expression primitives」、「Define coherent IR location policies for traced primitives」— 就是在補 layout algebra 在 ROCm 上的語義一致性。這是正在進行的實作，不是已完成的證明。

---

## 5. 怎麼接到 PyTorch：抄 CuteDSL 的作業

RFC #190875 把整合策略寫得非常誠實：

> _"explicitly models itself after CuteDSL's integration pattern for NVIDIA, treating FlyDSL as an elective backend that wins through performance evidence rather than policy."_

兩個 hook 點：

### 5.1 Eager path：dispatcher override

```python
# 粗略示意；實際命名以 RFC 為準
torch._native.register_override("rmsnorm", flydsl_rmsnorm, cond=is_gfx950)
torch.backends.python_native.enable("flydsl")   # user-controlled
```

Runtime 檢測架構（gfx950 / gfx942 / gfx1250），不支援就 fallback eager。
RFC 給的首個 showcase 是 **RMSNorm**：MI355X 上 1.50× geometric-mean 全幅加速。

### 5.2 Compile path：TorchInductor template heuristics

```python
# torch/_inductor/template_heuristics/flydsl.py
# torch/_inductor/async_compile.py  → flydsl 分支
```

TorchInductor 的 autotune 把 FlyDSL candidate 加進 choices 列表，和 Triton / Composable Kernel / rocBLAS 一起 benchmark，誰快選誰。
首個 operator 是 **GEMM（FP16/BF16 static shape, `aten.mm(A, B.T)`）**，MI355X 上比 Triton 快 **1.19× geomean**。

### 5.3 Packaging：外掛，不綁定

PyTorch 不 vendor FlyDSL compiler/runtime，用 CuteDSL 的套路：
使用者或 CI 自己 `pip install flydsl`。PyTorch 這邊 own 的只是「reviewed kernel snapshots + wrappers」。
這是個聰明的架構決策 — AMD 的 kernel DSL 版本演進可以不被 PyTorch 的 release cycle 綁死。

### 5.4 設計原則三條

RFC 自己列的：

1. **Additive**：是候選項，不是替換項。
2. **Architecture-gated**：按 GPU 代次 opt-in（初期只有 gfx950）。
3. **Performance-driven**：只在 benchmark 證明有收益的 operator 上啟用。

這三條聽起來很保守，但對 upstream review 是必要的 — Triton 當年進 PyTorch 的時候也是這種姿勢。你不能一上來就說「我是你的新 default」。

---

## 6. 性能數字：1% from HIP、1.19× over Triton

ROCm Blog 把一支手調 **FP8 GEMM** 從 HIP C++ 完整翻譯到 FlyDSL，MI355X / ROCm 7.2.2 / FlyDSL 0.2.0。

| Matrix Size   | HIP C++ (TFLOPS) | FlyDSL (TFLOPS) | Delta  |
|---------------|------------------|-----------------|--------|
| 4096×4096     | 3283             | 3261            | −0.7%  |
| 8192×8192     | 3292             | 3312            | +0.6%  |
| 12288×12288   | 3362             | 3444            | **+2.4%** |
| 16384×16384   | 3472             | 3411            | −1.8%  |

**解讀**：

- 平均在 ±1% 內波動，大矩陣（12k）FlyDSL 甚至反超 — 這一條是最重要的訊號。
  HIP C++ 的那支 kernel 是手調的，用了 inline asm 控制 `s_waitcnt`、手寫 LDS allocation。
  FlyDSL 把這些手調步驟換成 compiler 自動插入（`s_waitcnt`）、`fx.SharedAllocator()` 抽象。
  能在高抽象下維持 speed-of-light，意味著 compiler 做對了底層工作。
- 12k 矩陣反超的 2.4% 不是魔法。可能原因是 FlyDSL 的 **direct-to-LDS load path**（`rocdl.raw_ptr_buffer_load_lds`）比手寫 HIP 的 register-pumping 路徑少一個暫存器中轉。這是**抽象讓性能更好**而不是更差的例子 — compiler 看得見全局，程式員看不見。
- 配上 TorchInductor 的 **1.19× over Triton**（GEMM FP16/BF16），這套 pipeline 的性能故事就講得通了：**不是打 HIP，是打 Triton**。

這也解釋了為什麼 FlyDSL 選擇暴露 layout 而不是藏起來。Triton 在 AMD GPU 上的性能一直被詬病（和 NVIDIA GPU 上的體質不同），根本原因不是 Triton 編譯器不夠聰明，是**藏起 layout 的抽象在 CDNA 的 memory hierarchy 下損失太多**。FlyDSL 讓 kernel 作者直接表達 layout，是對這個問題的誠實回應。

---

## 7. 戰場地圖：Triton / CuTile / TLX / Mojo / FlyDSL

2026 這一年，GPU kernel DSL 的戰場已經從「Triton vs CUDA C++」變成多邊戰爭。我把目前的位置整理成一張表：

| DSL          | Vendor         | 抽象高度     | Layout 策略        | PyTorch 整合           | 主打場域             |
|--------------|----------------|--------------|--------------------|------------------------|----------------------|
| Triton       | OpenAI/community | 高（藏 layout） | 自家 blocked_layout | TorchInductor default  | NVIDIA 為主、AMD 補強 |
| CuteDSL      | NVIDIA         | 低（曝 layout） | CuTe algebra       | 外掛 backend           | Blackwell SOTA kernel |
| CuTile       | NVIDIA         | 中            | tile-based         | 內建 (CUDA 13.1+)     | Blackwell portability |
| TLX          | Meta           | 高            | Triton extension   | TorchInductor 分支     | Meta production      |
| **FlyDSL**   | **AMD**        | **低**        | **CuTe-compatible** | **TorchInductor 外掛** | **CDNA3/4/MI355X**   |
| Mojo kernels | Modular        | 低—中         | KGEN               | 外掛                   | 跨 vendor            |

**座標讀法**：

- **抽象高度低 + layout 顯式** = CuteDSL、FlyDSL、Mojo。這一派承認：**要榨乾新一代 GPU（Blackwell、CDNA4）的性能，必須給 kernel 作者 layout 控制權**。
- **抽象高度高 + layout 隱藏** = Triton、TLX。這一派賭的是：**compiler 可以學會隱式最佳化**。
- CuTile 是 NVIDIA 自己承認 Triton 的方向還是對的，所以做了個 tile-based 的 portable 層給 CUDA 使用者。

FlyDSL 選的位子是「**AMD 的 CuteDSL**」— 和 Triton 共存，但在高性能 operator 上壓制 Triton。
這是個合理的戰略選擇，因為 AMD 的客戶（主要是 hyperscaler）本來就不介意為 MI355X 這種高階卡寫 CuTe-level 的 kernel。

---

## 8. 風險、未解決的問題、該觀察的事

為了這篇文章的誠實度，我要列出一些 FlyDSL 還沒證明的東西：

### 8.1 CuTe algebra 在 CDNA 上的語義一致性

Section 4 已經說過了。CuTe 的 formal properties 來自 NVIDIA 的 memory model，CDNA 的 LDS bank、MFMA fragment 排列不同。
v0.3 roadmap 寫「enforce DslType closure」，就是在補這個洞。要觀察的是：**v0.3 release 時會不會公布 layout algebra 在 CDNA 上的 formal properties**？如果不公布，這個「CuTe-compatible」就只是語法相容，不是語義相容。

### 8.2 Autotuning 還沒 land

Roadmap 寫 v0.4 才有「Automatic kernel tuning support」。目前的性能數字都是**手寫 kernel + static shape**。
TorchInductor 的 autotune 需要 FlyDSL 給出 fast benchmark loop + cache 機制，這一塊 v0.2 還不完整。
**1.19× over Triton 這個數字在 dynamic shape 下會不會縮水，是個未知數**。

### 8.3 架構支援碎片化

v0.2 只保證 **gfx950 (MI355X / CDNA4)** 和 **gfx942 (MI300X / CDNA3)**。gfx1250（下一代、320KB LDS、Wave32、TDM）是 v0.3 目標。gfx1201（RDNA）、gfx90a（MI250X）支援不完整。
對 AMD 的客戶來說這是現實 — MI250X 是上一代，寫新 DSL 不划算。
但對 ROCm 的整體敘事（「我們在所有 AMD GPU 上都能跑」），FlyDSL 是個反例。

### 8.4 Mega MoE 這個賭注

Roadmap 把 **Mega MoE v1**（dispatch + grouped GEMM + combine）列為 v0.3/v0.4 的核心目標。
這對應的是 DeepSeek / Mixtral 這類 MoE 推理場景 — 當前 Triton 的 grouped GEMM 性能在 AMD GPU 上非常弱。
如果 FlyDSL 能在 MoE 上做出 Triton 做不到的東西，這個 DSL 的採用率會立刻上一個數量級。
反過來，如果 Mega MoE 延期到 v0.5，FlyDSL 的 narrative 就只剩 GEMM 這一個 showcase。

### 8.5 ROCm 的「總是晚一步」詛咒

歷史上 ROCm 的 DSL / compiler stack 多次做到接近 CUDA 的水準，但差 6–12 個月。這段時間裡 NVIDIA 又推進了下一代（Blackwell → Rubin → …）。
FlyDSL 的 roadmap 排 v0.3（7/5）、v0.4（8/5）、PyTorch Conference 10/22，節奏不算慢；但 10 月初 v0.2 仍是 current release，兩個版本都未如期落地——這是個該盯住的 slippage 訊號。
**另一條要觀察的是：MI400（CDNA5）出來之後，FlyDSL 的 layout algebra 需要重寫多少？** 如果 CuTe-compatible 的抽象讓 CDNA4→CDNA5 的升級路徑變短，FlyDSL 這一筆投資就贏了。

---

## 9. 對 compiler 職涯求職者（我自己）的意義

我把這篇文章放進 _Compiler-Path.md_ 的學習清單裡，有幾個明確的 take-aways：

**9.1 CuTe layout algebra 是必修**
它不再是 NVIDIA-only 的 CUTLASS 技術 — AMD 已經確認這套抽象可以跨 vendor。對 compiler engineer 來說，**理解 layout composition / divide / coalesce 的代數定律，比理解任何單一 DSL 的語法更保值**。

**9.2 MLIR dialect 設計 = 戰略決策**
FlyDSL 用「Fly dialect（layout algebra）→ ROCDL（hw-specific）→ LLVM（generic）」的分層，把 _語義層_ 和 _下降層_ 明確分開。
這和 Triton 的 dialect 混合不同、和 ClangIR 的設計哲學也不同。
我該能講出這些 dialect 選擇的 trade-off — 面試裡被問「如果讓你設計一個 ROCm 的 kernel compiler，你會把 layout optimization 放在哪一個 dialect」是個高概率問題。

**9.3 TorchInductor 的 extension points**
RFC #190875 把 TorchInductor 的 backend 擴充點公開了：`template_heuristics`、`async_compile`、autotune choice injection。
這幾個 API 之後任何 vendor（Intel XeVM、Hexagon、Qualcomm）加入都會沿用同一套。**讀懂這個 RFC = 讀懂了 TorchInductor 的 backend 架構**。

**9.4 Capstone 候選**
spconv capstone 之外，如果想做個小而強的 compiler side-project，**「把一個 ROCm HIP kernel 翻譯成 FlyDSL，寫一篇 benchmark report」是個 2–3 週可以完成、簡歷上立刻可讀的題目**。
FlyDSL 已開源，MI300X / MI355X 可以在 LambdaLabs 或 Hyperbolic 租到，成本比 Blackwell 低。

---

## 10. 收尾：這是 AMD 的 MLIR 覺醒時刻

過去兩年，AMD 的 compiler 戰略總被人形容成「追著 NVIDIA 跑」。
FlyDSL 讓這個敘事第一次站不住腳 — 不是因為 AMD 做了些 NVIDIA 沒做的事，是因為 **AMD 承認了 NVIDIA 做對的事（CuTe、CuteDSL 的 PyTorch 整合模式），然後在 ROCm 上重新實作，用 MLIR 當基礎設施**。

這很像 2019 年 Google 把 Jax 做出來 — 不是發明新東西，是把 PyTorch 的 autograd + numpy 的 API + XLA 的 compiler 組裝到一起，然後讓性能數字講話。

**FlyDSL 到了 1.0 的時候，衡量它成功的標準不是「超越 Triton」而是「讓 TorchInductor 在 AMD GPU 上和 NVIDIA GPU 上一樣好用」**。這條底線遠比技術炫技重要。

我會持續追蹤 v0.3、v0.4 兩個 release 的真實落地時間，以及 PyTorch Conference（10/22）後 RFC 的 upstream 進度。
Compiler path 上這類 vendor-level 編譯器策略的轉折，比單篇 paper 更值得當作學習座標。

---

## 延伸閱讀

- RFC: [Integrating FlyDSL with PyTorch DSL Extension Points (pytorch/pytorch#190875)](https://github.com/pytorch/pytorch/issues/190875)
- Roadmap: [FlyDSL Development Roadmap 2026 (ROCm/FlyDSL#630)](https://github.com/ROCm/FlyDSL/issues/630)
- Architecture: [FlyDSL Architecture & Compilation Pipeline Guide](https://rocm.docs.amd.com/projects/FlyDSL/en/latest/architecture_guide.html)
- Blog: [Porting High-Performance HIP Kernels to FlyDSL (ROCm Blogs)](https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html)
- Talk: [Extending TorchInductor with FlyDSL (PyTorch Conference NA 2026)](https://www.youtube.com/watch?v=eItm4ncwkOc)
- 相關站內：`cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md`、`clangir-maturity-rfc-polybench-gpu-cuda-hip-mlir-takeover-2026.md`、`cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026.md`

---

_— Nova_
