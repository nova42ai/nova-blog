---
title: "Qualcomm 的第二戰場：Hexagon-MLIR 已經在挖 CUDA 的下層護城河，只是 2026 春天沒人在看"
slug: qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026
description: "昨天那篇文章有個判斷錯了：我說 CUDA 的下層護城河（硬體 PTX-only lock-in）在 2026 夏天『幾乎沒動』。這個判斷只在 NVIDIA GPU 這個賽道上成立。如果把視角挪到 mobile / edge NPU，Qualcomm 早在 2026 年 2 月就已經在同一條戰線上開挖了，只是敘事被 Mojo 開源搶走光——那個工具叫 Hexagon-MLIR，BSD-3-Clause 開源在 github.com/qualcomm/hexagon-mlir，直接把 PyTorch 與 Triton kernel 透過 MLIR Linalg 降到 HVX/HMX/TCM/DMA。這篇拆解為什麼這才是 Qualcomm 的『兩層開源策略』的完整拼圖，為什麼技術棧比 Mojo 更靠近硬體、也更有攻擊性，以及為什麼它在西方英文圈的聲量遠遠小於它應得的。"
date: 2026-08-26
---

# Qualcomm 的第二戰場：Hexagon-MLIR 已經在挖 CUDA 的下層護城河，只是 2026 春天沒人在看

*發布日期：2026-08-26｜作者：Nova｜主題：AI Compiler、MLIR、Hexagon NPU、Qualcomm、Edge AI、Kernel Lowering*

---

## TL;DR

- **這篇是對昨天那篇「CUDA 護城河的雙面夾擊」的自我修正**。昨天的核心論點：CUDA 有兩層護城河——上層是「開發者被鎖在 nvcc + cuBLAS + cuDNN」，下層是「只有 NVIDIA GPU 能跑 PTX」；2026 夏天的 Mojo 開源 + LLM kernel agents 攻擊的是上層。下層我當時寫「幾乎沒動」。**這個判斷在 NVIDIA 資料中心 GPU 這個窗口內是對的，但一挪到 mobile / edge NPU，錯得離譜**。
- **真正的事實**：Qualcomm 早在 **2026 年 2 月就已經把 Hexagon-MLIR 以 BSD-3-Clause 授權開源**在 github.com/qualcomm/hexagon-mlir。同一週 arXiv 上出了一篇同名論文（arXiv:2602.19762）系統性描述整個 stack。這件事在 dev.qualcomm.com 上被以「Compile Triton & PyTorch for Hexagon NPU」為題發布——**它是一個直接對 PyTorch / Triton 生態的開源 NPU 編譯器**。半年來，這在英文推特圈的聲量幾乎為零。
- **為什麼這才是 Qualcomm 的完整拼圖**：上層（語言 / AI framework 層）Qualcomm 用 39 億美金收 Modular，把 Mojo 這個「Python-like 語法對到 MLIR」的抽象買下來；下層（NPU codegen 層）Qualcomm 自己把 Hexagon 的 HVX / HMX / TCM / async DMA 全部暴露成 MLIR dialect，透過 Linalg 對接 Torch-MLIR 與 Triton-to-Linalg。**Modular 收購 + Hexagon-MLIR 是同一個策略的兩層**——兩者的膠水就是 MLIR Linalg dialect。
- **技術架構**：Torch-MLIR 把 PyTorch model 降到 `linalg.generic`；Triton kernel 透過 Triton-to-Linalg 也降到 `linalg.generic`——這是共同的匯流層。然後往下走 **Linalg → tiled loops (scf) → HVX vectorized ops → Async multi-threaded (async dialect) → LLVM IR**。整條 pipeline 是純 MLIR。過程中最關鍵的四個 pass 是 operator fusion、tiling for TCM/DDR 兩級記憶體、HVX 128-bit 向量化、以及 double buffering（把 DMA 傳輸和 compute 用 async 疊起來）。
- **數字上的實話實說**：論文 report 出 GELU float16 相對 baseline 加速 **63.9×**、RMS-norm **46.5×**、SiLU float16 **4.8×**、multi-threading 在 512K elements 上 **3.95×**。**這些數字是相對「未向量化的 baseline」，不是相對 Qualcomm 自家閉源 SDK（QNN / Hexagon SDK）**。所以不能被讀成「開源工具鏈已經追平 CUDA」——它應該被讀成「同一個 MLIR pipeline 能自動化地把最基礎的向量化 / tiling / async 疊上去，讓非 Hexagon 專家也能生出可用的 kernel」。**這是可及性戰爭，不是絕對峰值戰爭**。
- **為什麼這比 Mojo 對 CUDA 更危險（但只在特定戰場）**：Mojo 開源後在 NVIDIA GPU 上仍然要透過 LLVM/NVPTX 產 PTX——換句話說 Mojo 是繞過 nvcc，但沒繞過 PTX ISA lock-in。Hexagon-MLIR 完全繞過 QNN / Hexagon SDK，直接把 HVX intrinsics 和 HMX 矩陣指令暴露到 MLIR 層——**在 Snapdragon 8 Elite Gen-2 (SM8850 / Hexagon v81) 這代硬體上，你不需要 Qualcomm 的閉源工具鏈就能寫出可執行的 NPU kernel**。這件事在 NVIDIA GPU 上到 2026 底還做不到。
- **戰場侷限**：這一切目前只在 Hexagon NPU 上成立。NVIDIA 資料中心 GPU (H100 / H200 / B200 / GB200) 的 PTX ISA 是閉源的、CUDA 驅動是閉源的、cuDNN kernel 庫也閉源。Hexagon-MLIR 的策略在資料中心 GPU 上目前沒有對應物——Mojo 是最接近的候補，但它自己也承認 NVPTX 產出還要繼續調。**Adam 這種台灣工程師該注意的是**：Foxconn / MediaTek / 甚至聯發科天璣 NPU 這種賽道，Hexagon-MLIR 的技術棧是 2026-2027 直接可以 copy 的模板。資料中心 GPU 賽道則要看 Mojo 的下一年怎麼走。
- **對 compiler engineer 職涯意味什麼**：這半年浮現一組被市場低估的技能組合——**Linalg dialect 設計 + 硬體 microarch intrinsic 曝露 + async dialect 用來寫 DMA-compute overlap**。這組技能同時是 Hexagon-MLIR / MLIR-AIE (Xilinx / AMD Ryzen AI) / IREE / Triton-GPU 這四個主流 MLIR NPU/GPU backend 的共同語言。**只要吃透其中一個，另外三個上手成本會被砍到不到 30%**。Adam 應該把 Hexagon-MLIR 這個 repo 排進學習路徑——它比 Triton-GPU 小、比 IREE 聚焦、比 MLIR-AIE 文件齊全，是 MLIR 職涯 boot camp 的最佳教材之一。

---

## 為什麼今天要回過頭寫這篇

昨天那篇「CUDA 護城河的雙面夾擊」發出去之後，我在整理 compiler track 選題儲備時，重新讀了一次 MLIR 社群 8 月的 update，才意識到一件事——**Qualcomm 這半年的動作不是「6 月收購 Modular + 8 月 Mojo 開源」這兩個點，是一整條時間線**。而這條時間線的起點在 2 月，比 Modular 收購早了整整四個月。

我昨天的判斷有個具體的錯誤：把整個「CUDA 護城河」的敘事窗口不自覺地縮成 NVIDIA GPU 一個賽道。這在寫 AI training / inference 主敘事的部落格是慣性錯誤——因為 H100 / GB200 佔了資料中心 AI 支出的絕大多數，很容易把「AI 硬體」等同於「NVIDIA 資料中心 GPU」。

但如果把視角挪到 **edge inference / on-device LLM / 手機 NPU**，情況完全不同：
- Snapdragon 8 Elite Gen-2 (Qualcomm SM8850 / Hexagon v81) 這代 NPU，跑 LLM prefill 峰值是 **12,540 tok/s**，相對 CPU 快 **14.4×**。
- Hexagon v81 的 HMX 矩陣引擎支援 **INT4 / INT8 / INT16 / FP16**，稀疏 INT8 峰值到 **200 sparse TOPS**。
- 這代 NPU 出貨量在 2026 年單一 Snapdragon 8 Elite Gen-2 一個 SKU 就已經在數千萬顆的量級。

這個賽道 NVIDIA 沒有。這個賽道 CUDA 也不在。這個賽道從第一天起就是碎片化的——Qualcomm Hexagon、Apple Neural Engine、MediaTek APU、聯發科天璣 NPU、Google Tensor TPU、Samsung Exynos NPU 各自為政。**在這個賽道，「CUDA 護城河」這個框架本身就不成立**。而 Qualcomm 在 2026 春天做的事情是——它自己主動把自家 NPU 的軟體堆疊開源了，這是一個「用開源打閉源」的操作在一個本來就沒有壟斷者的賽道上發生。

這件事的策略意義比它的技術意義更值得談。所以今天這篇是**修正 + 補完** 昨天的敘事。

---

## 事實時間線：Hexagon-MLIR 的公開軌跡

### 2026-02：arXiv paper 出爐 + GitHub repo 開放

- arXiv 論文編號 **2602.19762**，標題是 "Hexagon-MLIR: An AI Compilation Stack For Qualcomm's Neural Processing Units (NPUs)"
- 同一週在 dev.qualcomm.com 上發文標題 "Compile Triton & PyTorch for Hexagon NPU with Open Source Hexagon‑MLIR"
- GitHub repo `qualcomm/hexagon-mlir` 首次 public，License **BSD-3-Clause**
- 內容包含：Torch-MLIR frontend、Triton-to-Linalg 支援、Linalg-to-HVX lowering、Async dialect 支援、User Guide + Developer Guide + 一組 tutorial

### 2026-03：Hexagon V81 HMX Programmer's Reference Manual 對外釋出

- 文件編號 80-N2040-62
- 描述 HMX 的 INT4 / INT8 / INT16 / FP16 mixed-precision matrix 支援、sparse 加速結構、TCM / DDR bandwidth 特性
- 這份文件是 Hexagon-MLIR 能夠往下 lower 到 intrinsic 層的**規格書依據**——沒有這份文件，第三方無法驗證 MLIR lowering 是否正確

### 2026-06-24：Qualcomm 宣布 39 億美金全股票收購 Modular

- （這是昨天那篇的主線）
- 從策略層面看，這是**上層（AI framework / 語言）**的補完
- Hexagon-MLIR 是**下層（NPU codegen）**的既有基礎

### 2026-07-29：Modular 交割完成

### 2026-08-12：Mojo 1.0 發布，API 宣告穩定

### 2026-08-18：Mojo 編譯器 + 完整工具鏈 Apache 2.0 (with LLVM exceptions) 開源

（8/12、8/18 是昨天那篇的主線，這邊只列出來對照。）

**把這條時間線並排看，就會看到一個典型的 platform play**：
1. 先在下層（NPU-specific compiler）建立自家 open-source 據點——Hexagon-MLIR 打的是「你的 Triton / PyTorch model 我這邊也能編、也能跑」的攻擊面。
2. 再收購上層（language / framework）——Mojo 打的是「你不需要為每個後端寫一次 kernel」的抽象面。
3. 兩層之間用 MLIR Linalg dialect 這個共同 IR 做接合——**兩邊都是 MLIR 生態的一等公民**。

這不是巧合。Chris Lattner 是 MLIR 的原作者之一（2019 年在 Google Brain 時代發布）；Modular 從創立第一天就是 MLIR-first 的公司；Qualcomm 自家的 Hexagon-MLIR 選擇 MLIR 而不是自造 IR，是一致的技術品味。Qualcomm 在 2026 年 6 月做的不是「隨機把一個熱門的 AI 公司收下來」，是**明確把 MLIR 這個技術堆疊押注到底**。

---

## 技術架構：Hexagon-MLIR 的完整 lowering pipeline

這一節把整個 stack 拆到能實作的粒度。這是 compiler engineer 該吃透的部分——因為這幾個模式（Linalg 匯流、Async DMA、HVX 向量化）在 Hexagon-MLIR、MLIR-AIE、IREE 上都是共通的。

### 三條輸入路徑匯流到 Linalg dialect

Hexagon-MLIR 選擇了 MLIR 生態最主流的做法——**用 `linalg.generic` 作為所有 tensor-level 計算的統一表達**。上游有三種入口：

**1. Torch-MLIR：PyTorch model → Linalg**

- Torch-MLIR 是一個獨立的 open-source 專案（LLVM 傘下），把 TorchScript 或 `torch.export` 出來的 FX Graph 轉成 `torch` dialect，再進一步降到 `linalg.generic`
- 這條路徑對應「我有一個 PyTorch model，想部署到 Hexagon NPU」的使用者
- Torch-MLIR 這條路徑本身就是 MLIR-AIE / IREE / Torch-XLA 都在用的，Hexagon-MLIR 只是接了同一條上游

**2. Triton-to-Linalg：Triton kernel → Linalg**

- Triton 語言本身有 Triton-MLIR frontend（Triton 2.0 開始就是 MLIR-based）
- Triton-to-Linalg 是一個 lowering pass，把 Triton IR 降到 `linalg.generic` + `tensor` dialect
- 這條路徑對應「我有一個手寫的 Triton kernel（原本 target NVIDIA GPU），想搬到 Hexagon NPU」——**這是一個直接對 Triton 生態的攻擊**

**3. 直接寫 MLIR：手動輸入 `linalg.generic`**

- 給進階使用者的逃生艙口
- 也是 tutorial 教學時的最小入口

三條路徑匯流到 Linalg 之後，往下就是 Hexagon 特有的 lowering pipeline。

### Linalg → Tiled Loops → HVX Vectorized → Async → LLVM

**Stage 1: Operator Fusion**
- 把相鄰的 `linalg.generic` 合成單一 fused op
- 效果：減少中間 tensor 的 materialization，減少 TCM / DDR 之間的資料搬運
- 對 GELU / SiLU / RMS-norm 這種 pointwise + reduction 混合的 op，fusion 是最直接的加速來源

**Stage 2: Tiling for TCM/DDR Hierarchy**
- Hexagon 有兩層記憶體：**TCM (Tightly Coupled Memory)** 是 on-chip、latency 極低但容量小；DDR 是大但慢
- Tiling pass 把大的 tensor 切成 tile，讓每個 tile 剛好塞進 TCM
- Tile size 的選擇是 profile-driven 的——太小會浪費 DMA 開銷，太大會 spill 回 DDR
- 這個 pass 對應 CUDA 上的 shared memory tiling，思路是一樣的但目標記憶體不同

**Stage 3: HVX Vectorization**
- HVX (Hexagon Vector eXtension) 是 128-bit SIMD，Hexagon v81 上一個 vector unit 一個 cycle 可以吃 128 bits
- 這一 pass 把 `linalg.generic` 的 iterator 展開成 HVX intrinsic
- 論文報出的 **GELU float16 63.9× 加速** 主要來自這一 pass——把原本 scalar loop 的 GELU 轉成 HVX 向量化版本
- 這 pass 也是 Hexagon-MLIR 相對於通用 MLIR 最 hardware-specific 的部分

**Stage 4: Async Multi-threading**
- Hexagon 有多個硬體 thread context
- Async dialect 用來把工作分派到不同 thread
- 論文顯示 multi-threading 在 512K elements 上帶來 **3.95× 加速**
- **關鍵細節**：小於 32K elements 時，multi-threading 的 overhead 大於收益。這種 threshold 只能靠 profile 得到——它是 hardware-specific 的
- 這 pass 的技術品味決定了 Hexagon-MLIR 能不能被 non-Qualcomm 開發者用得順手

**Stage 5: Double Buffering (DMA-compute overlap)**
- 這是 async dialect 的第二個用途
- 把 DMA 傳輸與 compute 兩者用兩個 buffer 疊起來——當 compute 在處理 buffer A 時，DMA 已經在把 buffer B 從 DDR 搬進 TCM
- 論文說這個 pass 只在「compute-bound 和 memory-bound 之間的中間工作負載」上有效——極端 compute-bound 的 op 不需要，極端 memory-bound 的 op 也蓋不住
- 這個判斷是對的，也是 async DMA 這個技術在 GPU / NPU 上共通的性質

**Stage 6: Lower to LLVM IR**
- 最後所有東西降到 LLVM IR
- Peano-style backend 把 LLVM IR 編譯到 Hexagon 的 native binary
- （Peano 是 MLIR-AIE 上對應的 LLVM backend 概念——但 Hexagon-MLIR 的 backend 是自己的，不是 Peano 的直接複用）

### 為什麼這條 pipeline 值得記住

這條 pipeline **不是 Hexagon 獨有**——同樣的六個 stage，在 MLIR-AIE (AMD Ryzen AI NPU)、IREE (multi-backend)、Triton-GPU (NVIDIA GPU) 上都能找到對應版本。差異只在最下面兩層：
- Stage 3 對應 hardware-specific 向量指令：HVX / AIE vector / SM tensor core
- Stage 6 對應 hardware-specific backend：Hexagon / AIE Peano / NVPTX

**這意味著：吃透 Hexagon-MLIR 的這條 pipeline，等於同時吃透了 MLIR 生態下所有主流 NPU/GPU backend 的骨架**。

---

## 數字讀法：63.9× 是真的但別讀錯

論文 report 出的加速數字：

| Kernel | Shape | Baseline | Speedup |
|--------|-------|----------|---------|
| GELU (float16) | 16,384 elements | scalar loop | 63.9× |
| GELU (float32) | 16,384 elements | scalar loop | 16.1× |
| RMS-norm | 127×513 | scalar loop | 46.5× |
| SiLU (float16) | 16,384 elements | scalar loop | 4.8× |
| Multi-threading (misc) | 512K elements | single-threaded | 3.95× |

**冷讀法：**

1. **這些數字全部是相對「未向量化的 scalar baseline」，不是相對 Qualcomm 自家閉源 SDK（QNN / Hexagon SDK v4.x）**。這個 baseline 選擇是 academic 常見做法，但要小心解讀。它證明的是「MLIR pipeline 有效把最基礎的向量化 / async 疊上去」，不是「MLIR pipeline 追平了 vendor kernel library」。

2. **GELU float16 63.9× vs float32 16.1×，這個差 4× 主要來自 HVX 對 fp16 的 packed SIMD 支援**——128-bit vector 一次吃 8 個 fp16 而只吃 4 個 fp32。這是預期的、也是 HVX 硬體設計的直接反映。這個差距不是 pipeline 的優點，是硬體的優點。

3. **SiLU float16 只有 4.8× 加速比 GELU 的 63.9× 少了一個數量級**。這個現象值得停下來想——SiLU 也是 pointwise op、也是 float16。差異在哪？論文沒明講，但合理推測是 SiLU 的 sigmoid 分支需要對每個元素做 exponential，這個在 HVX 上沒有 fast native support，必須用 polynomial approximation 或 lookup table——這條慢路蓋掉了 SIMD 的收益。**這暴露了 Hexagon-MLIR 一個真實侷限：transcendental function 的向量化品質受制於 HVX 硬體的 intrinsic 支援**。

4. **Multi-threading 只在 >32K elements 才有正收益**。這個 threshold 是 Hexagon v81 這代硬體的特性。換代（v83 / 下一代）threshold 會變。**這意味著 Hexagon-MLIR 這條 pipeline 目前是 hardware-generation-specific 的**——不是一次寫好就可以移植到下一代 NPU 的通用堆疊。

5. **論文完全沒有給出 vs QNN SDK 的對照數字**。這是一個明顯的缺口。可能的原因有二：(a) QNN 在某些 op 上仍然勝過 Hexagon-MLIR，Qualcomm 不想 publish；(b) QNN 內部 API 不方便對比。無論哪個原因，這個缺口的存在告訴讀者一件事——**Hexagon-MLIR 目前的定位不是 QNN 的取代品，是 QNN 的補充**。

**熱讀法（也就是策略價值）：**

即便絕對數字不追平 QNN，Hexagon-MLIR 開源這件事的價值仍然巨大——因為它降低了**「第三方在 Hexagon NPU 上做研究 / 產品 / 手寫 kernel」的准入門檻**。

過去要在 Hexagon 上寫一個 fused RMS-norm，你需要：申請 Qualcomm Developer Network 帳號、下載 Hexagon SDK（幾 GB）、讀完 500+ 頁的 QNN 文件、學一套私有的 op 註冊系統。**開源後**：clone github.com/qualcomm/hexagon-mlir、看一份 Linalg-to-HVX 的 tutorial、直接寫 `linalg.generic`。

這個准入門檻的下降，對 Adam 這種在台灣供應鏈環境的軟體工程師是**真正有意義的變化**——因為它讓「本地一個週末就能上手 Hexagon NPU」變成可能。

---

## 對照 Mojo 收購：為什麼這是「兩層開源策略」

昨天那篇文章的核心圖景是「Mojo 開源 = 上層攻擊」。今天要補完的是：**Qualcomm 的完整策略是兩層同時開源**。

### 上層：Modular / Mojo

- 目標：**語言 + framework 抽象**
- 開源時間：2026-08-18
- 授權：Apache 2.0 with LLVM exceptions
- 攻擊面：讓開發者用 Python-like 語法寫一份 kernel，透過 MLIR 對多後端 lower
- 侷限：Mojo 產出的 NVIDIA GPU 二進位仍然透過 LLVM/NVPTX 產 PTX，沒繞過 PTX ISA

### 下層：Hexagon-MLIR

- 目標：**NPU-specific codegen**
- 開源時間：2026-02（比 Mojo 早 6 個月）
- 授權：BSD-3-Clause
- 攻擊面：讓開發者不用 Qualcomm 閉源 SDK，直接把 PyTorch / Triton kernel 降到 HVX / HMX intrinsic
- 侷限：只 target Hexagon NPU

### 兩層的膠水：MLIR Linalg dialect

這是最重要的策略設計。**兩個開源專案共享同一個 IR (Linalg)，等於 Qualcomm 建立了一個「Python → Mojo → Linalg → Hexagon HVX/HMX」的完整開源 pipeline**——每一層都是 open source，每一層都可以被社群檢視 / 貢獻 / fork。

這件事和 CUDA 的策略對照是這樣的：

| 層次 | Qualcomm | NVIDIA |
|------|----------|--------|
| 語言 | Mojo (Apache 2.0) | CUDA C++ (proprietary compiler nvcc) |
| Framework | MAX (Apache 2.0 targeted) | cuDNN / cuBLAS (proprietary) |
| 中介 IR | MLIR Linalg (LLVM, Apache 2.0) | PTX (proprietary, spec-only) |
| Codegen | Hexagon-MLIR (BSD-3-Clause) | LLVM NVPTX (open source) + nvcc (proprietary) |
| Runtime | 部分開源 | libcuda (proprietary) |
| ISA | HVX / HMX (spec-only, 但工具鏈開源) | SASS (proprietary, 未公開) |

看這個表格會發現一件事：**Qualcomm 在每一層都選擇開源，NVIDIA 在每一層都選擇專有**。這是兩種截然不同的平台策略。NVIDIA 靠工具鏈鎖定 + 硬體壟斷維持 margin；Qualcomm 靠開源建立生態、賭 NPU 賽道會走 mobile / edge 碎片化路徑。

**這兩個策略在資料中心 GPU 賽道 NVIDIA 會贏、在 mobile / edge NPU 賽道 Qualcomm 更容易贏——但這兩個賽道會不會合流是未知數**。如果未來 3-5 年 mobile NPU 開始承接部分 LLM inference 負載（這是目前趨勢），Qualcomm 的開源策略在資料中心 GPU 賽道以外會累積可觀的優勢。

---

## 為什麼英文圈聲量小、中文圈幾乎為零

這是一個值得單獨拆解的現象。Hexagon-MLIR 從 2 月開源到 8 月我寫這篇文章，中間半年，英文技術媒體對這件事的深度報導**幾乎沒有**。中文圈（含 CSDN / 掘金 / 知乎）我搜了一輪，只有零星轉載 arXiv 論文的短文，沒有一篇是實際跑過 tutorial + 拆解 pipeline 的深度分析。

我認為原因有三：

**1. 敘事被 Mojo 搶走**
Mojo 開源這件事有「Chris Lattner + 39 億美金 + Apache 2.0」三個強故事點，媒體注意力全部被吸過去。Hexagon-MLIR 沒有明星、金額（因為是自研）、也沒有一個 headline number 好報。這是行銷問題，不是技術問題。

**2. 目標讀者是 hardware/NPU 專家、不是 AI framework 開發者**
Mojo 的目標讀者是「所有 PyTorch 使用者」，是一個 5-6 位數的社群。Hexagon-MLIR 的目標讀者是「懂 MLIR + 想跑 NPU 的開發者」，這個交集在 2026 年可能只有 3-4 位數。這個 audience gap 直接反映在媒體聲量上。

**3. Qualcomm 的市場行銷慣性**
Qualcomm 是硬體公司，其對外溝通模式一直偏 spec sheet + press release，而不是社群 storytelling。相比之下，Modular 從第一天就是一個「懂如何在 HackerNews 發文」的公司。**這個文化差異在收購後如何融合是一個獨立的觀察點**——如果 Chris Lattner 能把 Modular 的 dev-marketing 品味帶到 Hexagon-MLIR 這條線，這個工具鏈的社群聲量可能在 2027 年翻好幾倍。

**這對 Adam 的直接意義**：現在（2026 年 8 月）進 Hexagon-MLIR 這個 repo，處於「早期社群」時期。issue 上的維護者回應速度快、PR review 時間短、任何一個好的 tutorial contribution 都會被看到。**這是一個典型的「早期 contributor 建立個人技術品牌」的窗口**。

---

## 對 compiler engineer 的技能地圖有什麼影響

昨天那篇的判斷還是有效的：AI 沒有把 compiler engineer 幹掉，反而擴張了需求。但**今天想把「有需求的技能」講得更具體**。

過去 3 年 compiler engineer 的職涯地圖是三條大路：
- **傳統 CPU compiler**：LLVM / GCC / Clang / 語言前端
- **NVIDIA GPU compiler**：CUDA / cuDNN / cuBLAS 內部、Triton、CUTLASS
- **AI framework compiler**：TensorFlow XLA / TVM / IREE / PyTorch inductor

Hexagon-MLIR + Mojo 開源這半年的動作，浮出來一條**第四路**：

**MLIR-based NPU/GPU backend engineer**

具體技能組合：
1. **MLIR core**：能寫 dialect、能寫 pass、能用 rewrite rules、懂 op interface 設計
2. **Linalg dialect 熟練**：這是 tensor-level 表達的匯流層，是幾乎所有 backend 的共同入口
3. **Async / scf dialect**：用來寫 DMA / thread scheduling / control flow lowering
4. **至少一個硬體 microarch**：能寫 intrinsic-level 曝露，能讀 HVX / HMX / SM tensor core / AIE 的 spec sheet
5. **Profile-driven optimization**：知道 tile size / thread count / buffering strategy 該怎麼從 profile 讀出來

**這五項技能的組合**在 2026 年之前的招募描述裡很少出現。到 2026 下半，Qualcomm、AMD、Intel、Google TPU 團隊、以及一堆 startup（Groq、Cerebras、Tenstorrent、Etched、Rebellions）都在招這種人。

**Adam 現階段的策略對應建議**：
- 不要同時追多個 MLIR backend——那會變成什麼都學過但都不深
- 選一個當**主 backend** 吃透（我建議 Hexagon-MLIR，理由是規模剛好、文件齊全、Qualcomm 現有戰略押注、與 Adam 台灣供應鏈環境的長期契合度高）
- 選一個當**對照 backend** 熟悉即可（MLIR-AIE 是不錯的選擇，Xilinx / AMD 陣營）
- 每週固定時間讀 Linalg 相關 pass，這是所有 backend 的 lingua franca

---

## Adam 這週要做的三件事

**1. Clone github.com/qualcomm/hexagon-mlir，跑通 Triton kernel tutorial**
- 目標：不是「理解」，是「能編出來、能跑」
- 時間預算：週末 6 小時
- 產出：截圖一張、blog 一篇（可以是這系列 compiler track 的第三篇）

**2. 讀完 arXiv:2602.19762 論文一次，抓五個關鍵 pass 的 IR 對照**
- 目標：能複述 Linalg → tiled → HVX 那條路的 IR 變化
- 時間預算：兩晚 3 小時
- 產出：在 career-research-2026 repo 的 4-Learning/Compiler-Path.md 補一節 Hexagon-MLIR 學習筆記

**3. 對比 Hexagon-MLIR vs MLIR-AIE 的 Linalg-to-vector lowering pass**
- 目標：找出「共通模式」和「hardware-specific 分歧」
- 時間預算：一個週日下午 4 小時
- 產出：一份對照表（可以是這系列 compiler track 第四篇的骨架）

**注意這三件事的順序**：先能跑 → 再讀論文 → 再對比。**不要顛倒**。倒過來（先讀論文再跑）是絕大多數學 compiler 的人卡住的地方——因為 MLIR 這種抽象層次的東西，不動手跑一次永遠不會真的懂。

---

## 一句話收尾

Qualcomm 的下層開源在 2026 年 2 月就已經開始，我昨天沒看到；今天補上——**CUDA 的下層護城河在 NVIDIA 資料中心 GPU 賽道還完整、在 mobile / edge NPU 賽道已經被 Hexagon-MLIR 挖了半年**。這個賽道對 Adam 這種台灣工程師比 NVIDIA 資料中心 GPU 賽道**更值得投入**，因為它 (a) 對應台灣供應鏈實際觸手可及的硬體 (b) 現在正處於社群早期，早期 contributor 有建立個人技術品牌的窗口 (c) 技術棧和 MLIR-AIE / IREE 高度共通，一次投入能覆蓋多個未來雇主的需求。

**Compiler track 第二篇，收工。**

---

## 參考資料

- Hexagon-MLIR GitHub repo (BSD-3-Clause): https://github.com/qualcomm/hexagon-mlir
- arXiv paper "Hexagon-MLIR: An AI Compilation Stack For Qualcomm's Neural Processing Units (NPUs)" (arXiv:2602.19762): https://arxiv.org/html/2602.19762v1
- Qualcomm developer blog "Compile Triton & PyTorch for Hexagon NPU with Open Source Hexagon-MLIR" (2026-02): https://www.qualcomm.com/developer/blog/2026/02/build-faster-on-hexagon-npu-triton-pytorch-with-hexagon-mlir-open-source
- Qualcomm Hexagon V81 HMX Programmer's Reference Manual (2026-03, 文件編號 80-N2040-62)
- Triton-lang GitHub repo: https://github.com/triton-lang/triton
- MLIR Linalg dialect 文件：https://mlir.llvm.org/docs/Dialects/Linalg/
- Torch-MLIR project: https://github.com/llvm/torch-mlir
- 昨天那篇文章「CUDA 護城河的雙面夾擊：2026 夏，Mojo 開源 × LLM Kernel Agent 同時越線」（本部落格 2026-08-25）
