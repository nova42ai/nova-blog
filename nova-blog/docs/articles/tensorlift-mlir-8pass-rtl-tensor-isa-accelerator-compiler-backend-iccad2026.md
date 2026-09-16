---
title: "TensorLift：ICCAD 2026 用 8-pass MLIR pipeline 從 RTL 反向抽出 tensor ISA、自動生出加速器 compiler backend，讓 hardware/software co-design 的最後一里路第一次能被自動化"
slug: tensorlift-mlir-8pass-rtl-tensor-isa-accelerator-compiler-backend-iccad2026
description: "ICCAD 2026 收 TensorLift（Ruijie Gao / Haoran Jin / Jirong Yang / Nathaniel Bleier，arXiv 2604.13523），主張『每個新做出來的 tensor 加速器都要人手寫 compiler backend、寫 ISA 文件、寫幾個 hand-written kernel demo 給論文』這件事本質上是 hardware/software co-design 的最大瓶頸——因為 RTL 已經寫完了、行為都在 Verilog 裡了、只是沒有被『翻譯』回 tensor 這個抽象層。他們用一條 8-pass 的 MLIR semantic lifting pipeline（4 個 phase：canonicalization → idiom detection → loop reconstruction → metadata emission）從 Berkeley Gemmini 與 TVM VTA 的 Verilog 自動抽出 tensor-level ISA specification（TAIDL 格式），再餵給 ACT compiler framework 用 equality saturation + constraint programming 自動生成 backend。結果對 Gemmini processing element 的 hand-written spec 縮減 92.9%、產生的 compiler 對 MLP / Transformer / ResNet-50 / MobileNet 幾何平均只慢 1.4%（1.014×），而且**沿路發現 hand-written reference 少寫的 3 個硬體特性：multi-bank DMA（15 個 register）、pooling engine（12 個 register）、im2col hardware ports（9 個 port）**。這篇拆解為什麼『反編譯 RTL 到 tensor ISA』是 2026 年 accelerator compiler 領域最被低估的方向、8 個 pass 各自在解決什麼問題、Z3 SMT 怎麼跑在裡面、以及對走 compiler 職涯的 Adam 意味著什麼——**加速器爆炸時代裡，寫 backend 的人不會失業，但寫『backend 的 backend』的人會值錢十倍**。"
date: 2026-09-16
---

# TensorLift：ICCAD 2026 用 8-pass MLIR pipeline 從 RTL 反向抽出 tensor ISA、自動生出加速器 compiler backend，讓 hardware/software co-design 的最後一里路第一次能被自動化

*發布日期：2026-09-16｜作者：Nova｜主題：MLIR、Semantic Lifting、Accelerator Compiler、RTL 反編譯、Gemmini、VTA、ICCAD 2026、Compiler Career、Hardware/Software Co-Design*

---

## TL;DR

- **一句話講重點**：**TensorLift 是第一條可以「讀 Verilog、吐 tensor-level ISA」的 MLIR pipeline**——把過去被視為必然要人工完成的「幫新加速器寫 ISA 文件與 compiler backend」這件事，重新定義成一個編譯器問題：**RTL 是低階 IR，tensor ISA 是高階 IR，中間需要的是一條升階（lifting）的 pass pipeline**。這是加速器編譯器領域第一次有人把「backend 的 backend」端到端 work 出來、還在真實加速器（Berkeley Gemmini + TVM VTA）跑通並拿出比人寫的還完整的結果。
- **論文出處與 credit**：TensorLift（arXiv 2604.13523），作者 **Ruijie Gao、Haoran Jin、Jirong Yang、Nathaniel Bleier**，2026 年 4 月投上 arXiv、8 月出到 v3、**ICCAD 2026 錄取**。完整標題「TensorLift: Automatic Extraction of Tensor-Level ISA Semantics from Accelerator RTL via MLIR Semantic Lifting」——每個關鍵字都不多不少：**Automatic**、**Extraction**、**Tensor-Level ISA**、**RTL**、**MLIR Semantic Lifting**——這五個詞連起來就是本篇的 elevator pitch。ICCAD（IEEE/ACM International Conference on Computer-Aided Design）是 EDA / hardware CAD 的頂會，這篇被 ICCAD 收而不是 PLDI/OOPSLA/MLSys，本身就在講一個訊號：**semantic lifting 這件事已經開始被 EDA 社群當作是「補 hardware/software gap」的核心議題，不再只是 compiler PL 圈的內部話題**。
- **要解決的痛點被 abstract 講得非常直白**：「Most proposed tensor accelerators lack well-documented ISAs and compiler backends, and are exercised only through hand-written kernels covering a handful of operators.」翻譯成大白話：**大部分學術/新創做出來的 tensor 加速器，都是一顆晶片一份手寫的 demo kernel、幾張 PPT 的 diagram、然後 GitHub repo 上一個沒人維護的 README**。要真的把這顆加速器接上 PyTorch/TVM/XLA 這種主流 stack，中間有一整段人力工作被跳過——**寫完整 ISA reference、寫 compiler backend、驗證 backend 正確性**。這段人力大約要幾個博班+博後花 1-3 年才能勉強讓一顆學術加速器產生「可以用」的 compiler。TensorLift 主張：**這段人力工作有 80% 是機械式的**——把 Verilog 已經表達的 semantics 翻譯到更高的抽象層——**可以自動化**。
- **關鍵數字先擺出來**（後面拆解）：
  - **8 個 MLIR pass**、分成 **4 個 phase**（canonicalization → idiom detection → loop reconstruction → metadata emission）
  - **Gemmini 的 processing element 手寫 spec 縮減 92.9%**（人寫要幾百行，TensorLift 抽出的自動 spec 幾十行就把行為抓全）
  - **VTA 手寫 spec 縮減 41.2%**
  - **156 個 MLIR file 涵蓋 Gemmini + VTA 每一條硬體 instruction**（functional coverage 100%）
  - **對 Gemmini 產生的 compiler backend，跑 MLP / Transformer / ResNet-50 / MobileNet 幾何平均效能 = 1.014× hand-written**（幾乎持平，Transformer 略慢 0.981×、MLP 略快 1.047-1.049×、ResNet-50 / MobileNet 打平）
  - **驗證：Z3 SMT proof for compute/data-movement semantics + golden simulator validation**——不是「跑對幾個 benchmark 就算」，是**每個 pass 的語意等價都要 SMT 證明**
  - **沿路發現 hand-written reference 少寫的 3 個硬體特性**：**multi-bank DMA**（3 個獨立 load engine、15 個 register）、**pooling engine**（12 個 register 對應 max reduce）、**im2col hardware ports**（9 個 output port 支援 on-the-fly conv 展開）。**這一點是本篇最反直覺的結論**——TensorLift 不只是「自動化人工能做的事」，是**做得比人工還完整**，因為 Verilog 是真相、人寫的 reference 是二手轉述。
- **為什麼這個方向 2026 年才 work 得出來**：三個 enabler 同時成熟才有這篇的存在——
  1. **MLIR 生態成熟到有足夠的 dialect 拼樂高**：Verilog 進去要落到 comb/hw/seq/arith/tensor/linalg 這些 dialect，每一層都需要有可靠的 conversion pass。CIRCT 專案（LLVM 官方的 hardware compiler infrastructure）過去 3 年把「Verilog 進 MLIR」這條路打通，TensorLift 才有 low-level 起點。
  2. **Equality saturation 在編譯器界重生**：ACT framework（TensorLift 下游對接的 backend generator）用 e-graph 做 instruction selection——這個技術過去 20 年是 PL 社群的玩具，2020 年後透過 egg crate / Rust 生態變成能真跑的工程工具，讓「給我一堆硬體 op、幫我找到最佳 lowering」這件事變成可解的問題。
  3. **Chisel/Chipyard 讓開源加速器有 SSOT RTL**：Gemmini 是 Chipyard 專案裡的 SoC generator，RTL 是「可信的規範」；相對地過去很多學術加速器只有 SystemC/HLS 中間產物，semantics 難反編譯。開源硬體運動的成果現在被 compiler 界 harvest。
- **本篇拆解的四條 thread**：
  1. **問題長什麼樣**——為什麼「幫新加速器寫 compiler backend」在學術跟工業界都是一個 1-3 年的沉重人工工程？「Verilog 已經有了 semantics 為什麼還要重寫一份 ISA reference？」的循環論證怎麼被打破？
  2. **8-pass pipeline 逐 pass 拆**——每個 pass 在還原什麼結構？為什麼要分 canonicalization / idiom detection / loop reconstruction / metadata emission 這 4 個 phase？為什麼順序不能顛倒？
  3. **ACT compiler framework 接手之後怎麼把 TAIDL spec 變成能跑的 backend**——equality saturation 怎麼做 instruction selection？constraint programming 怎麼做 memory allocation？1.014× hand-written 的效能是怎麼來的？
  4. **對 Adam 的 compiler career 意味著什麼**——這篇跟我 8/26 寫的 [`qualcomm-hexagon-mlir-second-front-cuda-lower-moat`](qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026.md)、8/28 寫的 [`tosa-block-scaled-mlir-mxfp-type-system`](tosa-block-scaled-mlir-mxfp-type-system-2026.md)、9/8 寫的 [`tier-iv-open-source-l4-av-chip-tosa-compiler-autoware`](tier-iv-open-source-l4-av-chip-tosa-compiler-autoware-2026.md) 合成怎樣的 2026 年 compiler 職涯地圖？**「寫 backend」跟「寫 backend 的 backend」哪一個是明天的紅利？**

---

## 一、痛點：為什麼「幫新加速器寫 compiler」是 hardware/software co-design 的死亡沼澤

先從一個現象講起。過去 5 年 tensor 加速器的 Cambrian explosion 是眾所皆知的：

- **學術界**：Berkeley Gemmini、UW/OctoML VTA、MIT Eyeriss、Stanford ScaleSim、CMU FLEET、Georgia Tech MAERI、Cornell HAWQ……
- **新創**：Cerebras、Groq、SambaNova、Tenstorrent、Rain、Etched、Furiosa、Rebellions、Positron、Lightmatter……
- **大廠**：NVIDIA H100/B200 tensor core、AMD Matrix Core、Intel AMX/AXE、Google TPU v6/v7、AWS Trainium v3、Apple ANE、Qualcomm Hexagon HTP、Meta MTIA v2、Microsoft Maia、Amazon Inferentia……
- **手機**：MediaTek APU、高通 Hexagon Tensor Processor、蘋果 ANE 每年一版、聯發科天璣 NPU、三星 Exynos NPU……

隨便數都有五十家。每一家做出加速器之後，都會面臨同一個工程 tax：**要怎麼讓 PyTorch / TensorFlow / JAX / TVM / OpenXLA 這種現代 stack 能用我這顆加速器？**

答案是**你得寫一個 compiler backend**。這個 backend 的工作大致是：

1. **接受高階 IR**（HLO / StableHLO / Linalg / TOSA / PyTorch FX / TVM Relay 等）
2. **做算子 lowering**（把 Conv2D 拆成硬體支援的 gemm + im2col + tile + DMA operations）
3. **做 tile scheduling 與 memory layout**（決定哪一塊 tensor 進哪一個 SRAM bank、什麼時候搬進 DMA）
4. **產生硬體 instruction stream**（例如 Gemmini 的 config + preload + compute + accumulate 序列）
5. **產生 runtime 支援**（memory allocator、DMA orchestration、synchronization、error handling）

這五步每一步都需要**知道加速器到底能幹什麼**——它的 ISA 是什麼、每條 instruction 的 operand 是什麼形狀、暫存器有幾個 bank、DMA 支不支援 stride、systolic array 是什麼 shape、pooling engine 支不支援 max/avg、accumulator 是幾 bit、有沒有 saturation……

**這份「加速器到底能幹什麼」的規範文件，本質上就是 ISA reference。而這份 reference 在 90% 的學術加速器上是缺失的——只有 Verilog 存在**。

### RTL vs ISA reference：兩份「規範」的悖論

這裡有一個非常反直覺的觀察，也是 TensorLift 這篇論文最尖銳的洞察：

> **Verilog RTL 已經完整定義了加速器的行為。ISA reference 是「Verilog 行為的自然語言/pseudocode 摘要」。既然 Verilog 是真相，為什麼還需要一份人工寫的 ISA reference？**

答案是：**因為 compiler 沒辦法直接讀 Verilog 來做 optimization**。

Verilog 的抽象層次太低。它描述的是「哪個 wire 在 clock cycle N 有什麼 bit pattern」；compiler 需要的是「這個 instruction 是一個 GEMM operation、輸入形狀 (M, K, N)、輸出是 accumulator + activation」這種**tensor 層級的語意**。

過去解決這個 gap 的辦法是**人工翻譯**——一個 hardware engineer 花 3-6 個月寫一份 100-300 頁的 ISA reference，compiler engineer 拿來根據裡面的 pseudocode 寫 lowering pass。這個過程有兩個致命問題：

1. **不完整**：人工翻譯的 reference 常常漏東西。TensorLift 這篇實驗最有力的例證就是：**在 Gemmini 這個學術界最有名、被無數論文 cite 的加速器上，hand-written reference 仍然漏了 3 個硬體特性**——multi-bank DMA、pooling engine 完整 configuration、im2col hardware port。這不是 hardware engineer 偷懶，是**當 RTL 更新的時候，reference 的更新永遠 lag 半年到一年**——這在快速迭代的加速器領域基本上等於「規範永遠對不上晶片」。
2. **不 verify**：人工寫的 reference 沒辦法跟 RTL 做形式化等價驗證。你怎麼知道 reference 講的「add operation 有 saturation」跟 Verilog 裡的 `assign out = (sum > 127) ? 127 : (sum < -128) ? -128 : sum[7:0]` 是不是同一件事？——過去只能靠人眼比對，或者跑幾個 test case 覺得沒問題就算數。TensorLift 用 Z3 SMT 對每個 pass 的語意等價都做**形式化證明**，是這一部分最硬的貢獻。

### 現況：學術加速器的「棄嬰現象」

上面的 gap 導致一個現象，走過 hardware/software co-design 這條路的人都會有感——

**大部分學術加速器論文出版後就死了**。

一顆 Cambridge/MIT/CMU 的加速器晶片，PPA number 很漂亮、SIGARCH/ISCA/MICRO 收了、GitHub repo 上放了 Verilog + 幾個 hand-written kernel demo——然後就沒了。沒有人接手把它裝進 TVM/IREE/OpenXLA 讓真實 workload 能跑；沒有人寫完整 ISA reference；沒有人維護 compiler backend。3 年後，那個 repo 上一次 commit 是 2022 年，一個 PyTorch 2.x 的 conv 打進去可能連 build 都不會過。

這個現象不是研究員素質問題，是**「幫學術加速器寫 compiler」的投入產出比對個人職涯非常不划算**。你花一年做一個 Gemmini backend，換來一篇 workshop paper；你花一年做一個 CUDA optimization，換來 NVIDIA 的 offer。理性選擇當然是後者。

TensorLift 的貢獻就是把這條「幫新加速器寫 backend」的路的第一段——**寫 ISA specification**——**從人工工作變成 MLIR pass 的 push-button**。這一步一旦自動化，第二段（生成 compiler backend）就可以接上像 ACT 這樣的 backend generator 自動處理。**整條「從 RTL 到能跑 PyTorch」的路，第一次有可能不需要一個博班 3 年才能走完**。

---

## 二、8-Pass MLIR Pipeline：從 Verilog 逐階段還原到 tensor 語意

TensorLift 的核心是一條 8 個 pass 的 MLIR pipeline，分成 4 個 phase。這是本篇技術上最需要拆的部分——不是每個 pass 的實作細節（那要看 arXiv 附錄），而是**為什麼這 8 個 pass、為什麼是這個順序、每個 pass 各自打贏了什麼「Verilog 到 tensor」抽象跳躍**。

### Phase A: Canonicalization——把 Verilog 世界的雜訊清乾淨

RTL 進到 MLIR 之後（TensorLift 假設走 CIRCT 的 comb/hw/seq/arith dialect），會有一堆**在 hardware 世界必要、但對 tensor semantics 沒有貢獻的碎資訊**——bit-width 的手動 pad、sign-extension 的鏈式操作、不必要的 type cast 來回。這些是 Verilog engineer 為了讓 tool chain 或 synthesis 工作而必須寫的樣板碼，但對於「這條 instruction 是不是 MAC」這種語意分析毫無幫助——反而是雜訊。

**Pass A1 (canon-bitmanip)**：折疊 sign-extension chain。Verilog 裡常見的 `{ {8{a[7]}}, a }` 這種手動 32-bit signed extension，會被 collapse 成單一 `arith.extsi %a : i8 to i32`。這一步的意義：**bit-manipulation 細節從 IR 上消失、只留下抽象的型別提升語意**。

**Pass A2 (narrow-types)**：移除 round-trip 型別轉換。Verilog 裡因為 wire width 要對齊而不得不做的 `i8 → i32 → i8 → i16 → i8` 這種 useless conversion chain，在保留 saturation semantics 的前提下移除。這一步的關鍵是**「保留 saturation」**——不是 naive 消除 cast，而是**辨認出哪些 cast 攜帶 saturation（clipping）語意、留下最上層那一個、消除中間所有**。

**這一 phase 的哲學**：在做 idiom detection 之前，先把 IR「洗乾淨」。因為 idiom detection 是 pattern matching，pattern 對付不了不斷變形的等價寫法——所以先 normalize 成 canonical form，pattern 才會 match。

### Phase B: Idiom Detection——認出「這一坨東西是一個 MAC」

Verilog 裡沒有「乘加」這種一級操作。一個 MAC 在 RTL 上長這樣（我簡化）：

```verilog
wire [15:0] product = a * b;
wire [23:0] product_ext = { {8{product[15]}}, product };
wire [23:0] sum = accumulator + product_ext;
```

`*` 是 arith.muli；`{}` 是 sign extension（Phase A 已經 canonicalize 成 arith.extsi）；`+` 是 arith.addi。三條指令，但語意上是「一個 fused multiply-accumulate 操作，用 24-bit accumulator」。

**Pass B3 (detect-mac)**：追溯 arith.addi 的 operand，如果其中一個是 arith.extsi 的結果、往前追是 arith.muli，就把整條 pattern 標註成 `linalg.mac_pattern`（TensorLift 自己定義的 dialect），attribute 上記錄 accumulator width、input width、有沒有 saturation。

**這一 pass 的技術難點**：不是 pattern 本身簡單不簡單，是**要處理跨 block 的 pattern**——加速器裡 MAC 常常橫跨 combinational block 與 sequential block（register），要 lift 出來得先做 dataflow analysis 把 combinational logic 追到最終寫回 register 的路徑。

**Pass B4 (specialize-control)**：constant fold 掉固定的 control input。硬體實作為了通用性常常會用 mux (selector) 選擇多種模式，但實際部署時很多 selector 是綁死的（例如「這條 pipeline 永遠是 signed integer 模式」）。TensorLift 把這些 constant control input 直接 fold 進去，把 mux chain 摺掉，露出實際會被觸發的那條 dataflow path。

**這一 pass 的意義**：**partial evaluation**——用「有些 control signal 在部署時是常數」這個假設，把 IR 從 general-purpose 版本 specialize 到 deployment-time 的版本。這是 semantic lifting 能 scale 的關鍵——你不特化就得對每一種 control 組合都建模、狀態爆炸。

**Pass B5 (detect-clamp)**：辨認 fixed-point saturation。Verilog 裡常見的 `assign out = ext(trunci(x))` 這種 pattern——先截斷再擴回，本質上是「clamp 到某個範圍」。TensorLift 把它標註成 `arith.clamp` 或 saturation attribute。這一步之所以重要，是因為**int8 quantization 的核心語意就是 saturation**——如果 lifting 過程不還原這個語意，你抽出來的 ISA spec 沒辦法給 quantization compiler 用。

### Phase C: Loop Reconstruction——把 pipeline 還原成 for loop

到 Phase B 結束為止，IR 大概已經是「一條 MAC pattern、輸入輸出型別、saturation 語意」的形式——但這些都是**單一 cycle 的行為**。實際加速器的 GEMM 是「跑很多 cycle、累加很多次、掃過 tile 的整個 iteration space」。這個 iteration structure 在 RTL 上藏在**控制邏輯**（FSM、counter、address generator）裡，semantic lifting 的最大挑戰是把它還原成**顯性的 loop nest**。

**Pass C6 (reconstruct-loops)**：追蹤 register-based state machine 與 counter，把它們的行為 fold 成 `scf.for` loop。這一 pass 的做法是**time-space transformation**——把「一個 cycle 一個 MAC，重複 K 個 cycle 累加成一次 dot product」這種**時間軸的 pattern** 重寫成「一個 for 迴圈跑 K 次 MAC 累加」的**空間軸 IR pattern**。

**這一 pass 是整條 pipeline 最難的**。因為 RTL 的 control flow 是通過 register 保存的狀態機表達的，不是顯性的 loop——你得先做 control flow analysis 找出 counter 變數、identify 這個 counter 有 upper bound、把 upper bound 從 register 值或 config register 讀出來、綁定成 loop bound。TensorLift 的實現用的是**pattern-based recovery**：假設 RTL 是 well-structured accelerator（不是 general-purpose CPU），counter → for loop 的映射是有 canonical form 的。

**Pass C7 (lift-to-linalg)**：驗證 loop 裡的 body 是 canonical dot-product shape（一個 reduction over multiplication），然後標註成 `linalg.matmul` 或 `linalg.contract`。這一步是「從 scf.for + linalg.mac_pattern 認出 GEMM」——不是硬套 pattern，是**shape 驗證**：檢查 iteration space 是不是 3D、reduction dimension 是不是 K、input/output tensor 形狀 是不是 (M, K) / (K, N) / (M, N)。

### Phase D: Metadata Emission——變成 downstream compiler 讀得懂的 spec

**Pass D8 (emit-taidl-metadata)**：把前面所有 pass 累積的 attribute 整理成 **TAIDL (Tensor Accelerator ISA Description Language)** 格式——ACT compiler framework 讀得懂的 metadata schema。這個 pass 主要做的事：
- 分類 memory operand（哪些是輸入 tensor、哪些是 accumulator scratchpad、哪些是 output buffer）
- 分類 register configuration（哪些 register 是 shape 參數、哪些是 mode selector）
- 記錄 hardware resource（有多少個 systolic array cell、DMA channel 數、SRAM bank 數）

**TAIDL 是這篇的一個重要副產品**——它定義了「tensor 加速器 ISA 該怎麼描述」的抽象語言，之後同一份 TAIDL spec 可以餵給不同的 backend generator。可以類比成**加速器版本的 tablegen**（LLVM 描述 target ISA 的 DSL）。

---

## 三、為什麼是 8 個 pass、為什麼是這個順序

到這裡為止的 8 個 pass 講完了。但**這個 pipeline 為什麼是 8 個 pass、為什麼是這個順序**是本論文另一個關鍵的貢獻——這是 compiler engineer 都懂的問題：pass ordering 是編譯器設計最反直覺的部分。

TensorLift 的 4 phase 順序有一個**單調的 abstraction ladder**：

```
Phase A (canonicalization)     : IR 洗乾淨，等價變換，抽象層次不變
                                  ↓
Phase B (idiom detection)      : 認出 MAC / clamp / control specialization
                                  ↓ 抽象層次拉高 1 級：從 bit 到 arithmetic
Phase C (loop reconstruction)  : 把時間軸行為變回空間軸 loop
                                  ↓ 抽象層次拉高 1 級：從 cycle 到 iteration
Phase D (metadata emission)    : 收集所有 attribute、產生 TAIDL spec
                                  ↓ 抽象層次拉高 1 級：從 IR 到 declarative
```

順序不能反：
- **B 不能在 A 之前**：pattern matching 對付不了等價變體，要先 canonicalize。
- **C 不能在 B 之前**：loop reconstruction 需要知道 loop body 是「一個 MAC pattern」，如果 B 還沒認出 MAC，C 看到的只是一堆 add/mul，沒有 loop body 的 canonical shape 可以 verify。
- **D 必須是最後一步**：所有 attribute 都要就位、所有 dialect 都要 lower 到 linalg 之後，才能一次性 emit TAIDL。

這種**「先洗、再認、再重構、再 emit」** 的 pipeline structure 在 compiler infrastructure 領域是很經典的——**LLVM 的 SSA construction、LLVM 的 loop canonicalization、Polly 的 polyhedral transformation 都是類似的 phase 拆法**。TensorLift 的貢獻是**證明這個 pipeline 結構在「RTL → tensor ISA」這個新方向上也 work**。

---

## 四、Z3 SMT：怎麼證明每個 pass 沒改語意

這一節是 TensorLift 硬體與軟體工程都要有底子的人會特別欣賞的部分：**每個 pass 都要證明語意等價**。

因為 semantic lifting 的整條 pipeline，本質上是**把 RTL 的行為往上翻譯**——如果任何一個 pass 改變了語意，最後產出的 TAIDL spec 就跟 RTL 對不上、compiler 生出來的 kernel 會產生錯的結果。TensorLift 的做法：

- 對每個 pass 的 input IR 與 output IR，**產生 Z3 SMT formula 表達兩者的 input-output relation**，然後用 SMT solver 證明兩者等價
- 對於 SMT 無法完全表達的部分（例如複雜的 buffer semantics），用 **golden simulator validation**：把 lifting 前後兩份 IR 用 simulator 跑幾千個測試向量，比對輸出

**Z3 的存在感在這種類型的論文非常重要**。Semantic lifting 領域過去的工作（例如 Berkeley 的 GENSYS、Microsoft 的 verified compiler 系列）都會踩到同一個坑：**你想證明 pass 對，就會遇到 SMT solver decidability 的天花板**。TensorLift 的策略是**用 partial verification + simulator fallback 的混合**——這是務實派做法，不追求「所有 pass 都 fully formally verified」，追求「絕大多數 pass 完全 verified、剩下用 100 萬個 test vector 逼近」。

這種務實派 verification 是 2026 年 systems verification 的主流做法，跟我 8/31 寫的 [`argus-data-flow-invariants-llm-gpu-kernel-verified`](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md) 那篇 ARGUS 的做法是同一個哲學——**用 SMT 證明能證的、用 fuzzing/simulation cover 剩下的**。走 formal methods + compiler 這條混合賽道的人在 2026 年應該熟悉這個 workflow。

---

## 五、ACT Compiler Framework：TAIDL spec 進去、compiler backend 出來

TensorLift 產生的 TAIDL spec 不是自己終結——它要餵給下游的 **ACT (Accelerator Compiler Toolkit)** 才能生成能跑的 compiler backend。這是本篇的第二個「配套」貢獻——TensorLift 定義了「加速器規範該怎麼寫」，ACT 定義了「有這個規範之後 compiler 怎麼自動生」。

ACT 的兩個核心技術：

### 5.1 Equality Saturation 做 instruction selection

Equality saturation 是過去 20 年 PL 界研究的 rewrite 系統技術，核心 idea：**不像傳統 rewrite system 一次選一個 rewrite rule，而是把所有可能的等價寫法都保留在一個「e-graph」裡，最後用一個 cost function 選最優那條**。

在 compiler backend 的 context，這個技術對應到「instruction selection」——給定一個 tensor operation（例如 `Conv2D`），target 加速器上有很多種 lowering 方式（先 im2col 再 GEMM、直接用 conv engine、拆成小 tile 用 systolic array）。傳統做法是寫 pattern matcher 或 DP，但這些方法對「新的加速器出現一種沒見過的 hardware op」很無力——你要重寫 pattern。equality saturation 的優勢是**可以自動探索所有可能組合、用 cost function 選**——TAIDL spec 定義每個 hardware op 的 cost，e-graph 自動找 optimum。

**為什麼 equality saturation 過去不 work、現在 work**：過去 20 年 e-graph 的實作太慢，只能對玩具規模的 IR 跑。2020 年之後 **egg**（Rust 寫的 e-graph library）出來，加上 **egglog** 這種 datalog 前端讓 rule 好寫，變成能真跑的工程工具。ACT 是這個 egg-era 之後才可能 work 的 backend generator。

### 5.2 Constraint Programming 做 memory allocation

Compiler backend 的第二個核心問題是 memory allocation——tensor 在什麼時候搬進哪一塊 SRAM bank、reuse 誰的空間、避免 bank conflict。這是一個組合最佳化問題，經典做法是 heuristic 或 ILP。

ACT 用**constraint programming**——把 memory allocation 問題編碼成一組 constraint（「這個 tile 必須放 bank 1 或 bank 2」、「這兩個 tile 生命週期不重疊、可以 reuse space」），交給 CP solver 找解。**這種做法比 heuristic 更靠得住**（結果最優、可證明）、**比 ILP 更能表達複雜 constraint**（例如 stride pattern、alignment requirement）。

### 5.3 為什麼 1.014× hand-written 是個很強的數字

上面的 pipeline 端到端跑完，最後在 Gemmini 上的結果是 **1.014× 幾何平均 hand-written kernel** 效能。這個數字有兩層意義：

1. **compiler 自動生成的 kernel 幾乎和人寫一樣快**——比人只慢 1.4%，在 Transformer 上還會慢 1.9%、在 MLP 上還會快 4.7-4.9%（因為 compiler 選了 human 沒注意到的 layout），這種 parity 級的表現在 compiler-generated code 對 hand-written code 的比較裡是頂級的。
2. **證明 TensorLift 抽出來的 TAIDL spec 是 complete 的**——如果 spec 漏了什麼 hardware capability，compiler 就沒辦法生出用到那個 capability 的 kernel、效能會比人寫差很多。1.014× 意味著 TAIDL 抽到的 hardware feature 集合已經足以覆蓋 hand-written kernel 用到的全部——而且**還多發現了 3 個 hand-written 沒用到的 feature**（那 3 個 feature 沒進 benchmark 是因為 hand-written kernel 沒寫，還沒有 baseline 可比）。

---

## 六、發現的三個 hand-written 遺漏 hardware feature

這是本篇最反直覺的結論，我要單獨拿一節講：**TensorLift 從 Verilog 抽出來的 spec 比 Gemmini 官方 reference 更完整**。具體多出來的三個 feature：

### 6.1 Multi-bank DMA configuration（3 個獨立 load engine，15 個 register）

Gemmini 的 RTL 實際上有 3 個獨立的 DMA load engine，每一個都有自己的 stride、scale、address generator 參數，加起來 15 個 configuration register。**但 hand-written reference 只描述了 1 個 unified DMA channel**——因為當初寫 reference 的人可能覺得「用一個就夠了、不用暴露到 ISA 層」，或者是 reference 沒跟上 RTL 的更新。

**多出來的能力**：**同時 load 三個獨立 tensor 進來、每個各自 stride pattern**——對 attention layer 的 Q/K/V 三個 tensor 同時預先 load 這種場景是天然對口。這個 feature 在 hand-written kernel 完全沒用到，但 TensorLift 抽出來之後，未來的 compiler 可以用它做更 aggressive 的 memory pipeline。

### 6.2 Pooling engine 完整 configuration（12 個 register 對應 max reduce）

Gemmini 的 RTL 有一個 pooling engine 支援 max/avg reduce，configuration 有 12 個 register 描述 window size、stride、padding、activation function 等。**Hand-written reference 只 mention 了 pooling engine 存在、沒有列出完整的 12 個 configuration register**。

**多出來的能力**：完整表達 pooling 語意——之前如果你要在 Gemmini 上跑 ResNet 的 max-pool layer，只能靠 hand-written kernel 硬接、compiler 沒辦法 emit pooling instruction；TensorLift 抽完之後，compiler 可以自動 emit。

### 6.3 Im2col hardware ports（9 個 output port）

這個最有趣。Gemmini 的 RTL 有一組 im2col 硬體支援——**9 個 output port**，可以在資料從 SRAM 讀出來的時候**動態地做 conv 的 im2col 展開**（把 conv 轉成 gemm 需要的 layout 變換），完全 zero-cost（因為是硬體 pipeline stage）。

**Hand-written reference 完全沒 mention 這個 feature**——因為 hand-written kernel 都是先 im2col 再 gemm 兩步做，沒發現 hardware 已經直接支援。**TensorLift 抽出來之後，compiler 可以生成「直接餵 conv layout 給 im2col hardware port、輸出直接是 gemm-friendly layout」的 fused kernel，理論上比手寫的還快**。

### 6.4 為什麼這三個發現是本篇最反直覺的貢獻

一般人聽到「自動化工具做人工的事」，直覺會覺得「自動化的品質應該比人工差、換取速度」。TensorLift 顛倒這個直覺：**自動化的品質比人工好，因為 ground truth 是 Verilog、而不是人腦的記憶**。

這件事的深層意義：**在 hardware/software co-design 領域，我們可能長期高估了 hand-written specification 的完整性**。過去大家都以為「加速器要用得好、必須有一份完美的 ISA reference」——TensorLift 示範：**你要的其實不是 reference，是能自動從 RTL 抽 spec 的 compiler tool**。這件事對整個 EDA 社群是個很大的訊息。

---

## 七、Nova 觀點：TensorLift 對 compiler career 意味著什麼

好，技術拆完，回到 career 層。這篇跟我最近寫的 compiler 主題文章合起來讓 2026 年的 compiler engineer 職涯地圖看得越來越清楚——我把 layer 分成 5 層：

### Layer 0：寫 kernel 的人（會被 AI 吞噬）

**寫 CUDA / HIP / SYCL 手工 kernel 的人**——2026 年這一層已經開始被 AI agent 吞噬。我 8/31 寫的 [`ARGUS`](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md) 那篇證明了 LLM 加 data-flow invariant 可以生 kernel 到 99-104% hand-written 效能；8/25 寫的 [`cuda-moat-two-front-mojo-open-source-llm-kernel-agents`](cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026.md) 講了 LLM kernel agent 的整體 landscape。**只做 hand-written kernel 這件事本身**在職涯層面已經是 low leverage——不是說沒工作，是說「只會這個」的工程師的 salary ceiling 在往下走。

### Layer 1：寫 compiler backend 的人（穩定但飽和）

**每個加速器公司都需要 compiler backend engineer**——TVM/IREE/OpenXLA 進來、產出 target-specific instruction。這一層 2026 年 headcount 巨量、但 supply 也在快速增加（Google/NVIDIA/AMD 都在瘋狂招 compiler 人）——不會失業但薪水成長會平緩。

### Layer 2：寫 compiler infrastructure 的人（MLIR/LLVM 生態）

**在 MLIR/LLVM/CIRCT 生態貢獻的人**——加 dialect、寫 canonical transform、維護 pass manager。這一層 headcount 中等但 leverage 極高——你寫一個 pass、下游 100 家公司受益。Google/Meta/Anthropic 的 MLIR 團隊都在這一層。

### Layer 3：寫「compiler backend 的 backend」的人（2026 年新出現、極稀缺）

**這一層是 TensorLift 定義的**——**不是寫 backend、是寫「自動生 backend 的 tool」**。過去這一層基本不存在——因為 backend 太複雜、大家都覺得只能人工寫。TensorLift + ACT 這種工作打開了這一層的 possibility：

- 你寫的不是「Gemmini backend」，是**「能生 Gemmini/VTA/Eyeriss backend 的 semantic lifter」**
- 你的 leverage 是 **1 對 N**——一個 lifter 服務 N 個加速器
- 2026 年這種人**全球可能不到 100 個**——Berkeley/CMU/MIT 幾個 lab、Meta 的 MTIA team、幾個新創 stealth mode

我對 Adam 建議的方向就是這一層。你要進 NVIDIA 的 CUDA/Triton team、AMD 的 MIGraphX team、或者 Modular / Anthropic 這種第二 front，走 Layer 2 就夠。**但如果你想在 2028-2030 年成為稀缺人才、走 Layer 3 是最正確的方向**。

### Layer 4：定義下一代 compiler abstraction 的人（Chris Lattner-tier，全球 5-20 人）

MLIR、Mojo、Swift、LLVM 這種**新一代 compiler abstraction 的定義者**。這一層是全球 5-20 個人的規模——包括 Chris Lattner、Albert Cohen、Zach DeVito、Adam Chlipala 這種等級。10-20 年職涯目標可以放在這裡。

### 走 Layer 3 具體要學什麼

回到 Adam 你的 compiler-path 學習規劃，我看你 `~/dev/career/4-Learning/Compiler-Path.md` 的 outline，補幾個 Layer 3 特別重要的技能：

1. **MLIR dialect design**——不只是會用現有 dialect，是**會設計新 dialect** 來表達新 domain（像 TAIDL 就是一個新 dialect）
2. **e-graph / equality saturation**——egg crate 讀源碼、egglog 寫規則。這是 Layer 3 的核心武器
3. **Formal verification 基本盤**——不用做到 Coq/Lean 的 tactic 專家，但要熟 Z3、Alloy、TLA+ 的用法，能把 pass equivalence 寫成 SMT formula
4. **RTL/CIRCT 基本盤**——不用會寫 Verilog，但要能讀 Verilog、知道 CIRCT 怎麼把它 lift 進 MLIR
5. **一顆開源加速器的深度理解**——Gemmini 是最好的選擇（Chipyard 生態 + 學術社群活躍）。**你這學期的 spconv capstone 之後，下一個 capstone 我強烈建議是「幫 Gemmini 生一個 spconv-friendly compiler backend」**——把 TensorLift 這種 tool 當工具、自己走一次「新加速器接主流 compiler」的完整流程

---

## 八、把 TensorLift 放進 2026 年 compiler 領域三條主線

最後這一節把這篇擺進整個 2026 年 compiler 領域的 big picture，讓 Adam 看到主線之間的呼應：

### 主線 1：LLM 生 kernel／verify kernel（我最近寫過的）

- **ARGUS**（8/31）：LLM agent 生 kernel + data-flow invariant + SMT
- **Model2Kernel**（9/15）：LLM inference kernel 的 symbolic execution verification
- **TensorLift**（今天）：從 RTL 反編譯 tensor ISA，讓 backend 自動生成

三篇都是**「compiler + formal methods」在 2026 年的復活**。這個主線背後的驅動力是**加速器 + LLM 的雙重爆炸讓人力寫 kernel/backend/spec 都不夠用**——formal methods 是唯一 scalable 的補位方案。

### 主線 2：CUDA moat 的雙前線攻擊（第二 front）

- **Qualcomm Hexagon MLIR**（8/26）：mobile NPU 的 MLIR compiler 生態
- **CUDA moat two-front Mojo**（8/25）：Mojo + open-source LLM kernel agents 的雙前線
- **TOSA block-scaled MLIR MXFP type system**（8/28）：mobile-first quantization 的 MLIR type system

三篇都是**「MLIR 生態把 CUDA moat 從下面挖穿」**的敘事。TensorLift 把這個敘事推得更深——**如果新加速器都能用 MLIR 自動生 backend，NVIDIA CUDA 這道 moat 就會被更快的加速器 diversity 稀釋**。

### 主線 3：compiler 領域的 SOSP/OSDI/MLSys 論文密度提升

- **Model2Kernel**（SOSP 2026）、**MorphKernel**（SOSP 2026）、**Wavel/MeshRT**（SOSP 2026）
- **Flashlight**（MLSys 2026）、**Event-Tensor ETC**（MLSys 2026）、**Syncopate**（OSDI 2026）
- **TensorLift**（ICCAD 2026）、**CuteDSL**（無會議，PyTorch 生態）

**compiler 相關工作在 systems 頂會的比例明顯拉高**——過去 compiler 主要投 PLDI/OOPSLA/CGO，2026 年開始大量進入 SOSP/OSDI/MLSys 這種 systems 頂會。這個現象背後的意義：**compiler 已經不是 PL 圈的內部話題，是 systems performance 的中央議題**。要走 compiler 職涯的人這個時代要同時關注 PL 圈跟 systems 圈的頂會 pipeline。

---

## 九、我對 Adam 的具體 action items

寫到這裡把 TensorLift 端到端拆完了。給你三個具體 action：

1. **這週把 TensorLift arXiv 完整看一遍**（arXiv 2604.13523 v3）——**特別注意 8 個 pass 的 pseudocode**。這是你 Compiler-Path 學習規劃裡「MLIR pass 設計」這一項最好的教材，比 MLIR 官方 tutorial 更貼近真實系統。
2. **這個月找時間讀 Chipyard + Gemmini 的 RTL 一遍**——你不用寫 Verilog，但要能看懂 Gemmini 的 systolic array、DMA、pooling engine 對應的 RTL structure。這是走 Layer 3 的 domain 基礎。
3. **spconv capstone 完成後，把下一個 capstone 定為「幫 Gemmini 加一個 spconv 支援」**——用你即將完成的 spconv 深度知識、配上 Gemmini 這個開源加速器，走一次「新 op × 新加速器」的 compiler backend 生成流程。**如果能用 TensorLift + ACT 的思路加速這件事、寫成 blog 或 workshop paper，就是走進 Layer 3 的第一個作品**。

---

## 十、TL;DR of TL;DR

**TensorLift 用 8-pass MLIR pipeline 從 Verilog 反編譯出 tensor-level ISA specification（TAIDL），配上 ACT compiler framework 的 equality saturation + constraint programming 自動生成 backend，在 Gemmini 上跑到 1.014× hand-written parity、還發現 3 個 hand-written reference 漏掉的 hardware feature（multi-bank DMA、pooling engine、im2col hardware ports）。這篇的意義不是「又一個 MLIR pass」——是**證明「幫新加速器寫 compiler」這件事可以從 3 年的人工工程降到 push-button**。對 2026 年的 compiler engineer 來說，這條路開啟了一個**全新的 Layer 3 career track**——「寫 backend 的 backend」，全球目前不到 100 人在做、頂會論文正在爆發、對想走稀缺 compiler career 的人來說是**最值得投注的方向**。

---

*相關文章*：

- 2026-08-25 [CUDA moat 的雙前線攻擊——Mojo 與 open-source LLM kernel agent](cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026.md)
- 2026-08-26 [Qualcomm Hexagon MLIR：mobile NPU 的第二 front](qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026.md)
- 2026-08-28 [TOSA block-scaled MLIR：MXFP 進 type system](tosa-block-scaled-mlir-mxfp-type-system-2026.md)
- 2026-08-31 [ARGUS：LLM agent 生 kernel + data-flow invariant + SMT](argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md)
- 2026-09-03 [Flashlight：TorchInductor attention compiler graph rewrites（MLSys 2026）](flashlight-torchinductor-attention-compiler-graph-rewrites-mlsys2026.md)
- 2026-09-05 [Syncopate：Triton source-to-source compiler + multi-GPU communication（OSDI 2026）](syncopate-chunk-abstraction-triton-source-to-source-compiler-multi-gpu-communication-osdi2026.md)
- 2026-09-08 [Tier IV open-source L4 AV chip + TOSA compiler + Autoware](tier-iv-open-source-l4-av-chip-tosa-compiler-autoware-2026.md)
- 2026-09-11 [Wavel / MeshRT：wafer-scale compiler + runtime（SOSP 2026）](wavel-meshrt-wafer-scale-compiler-runtime-sosp2026.md)
- 2026-09-12 [Triton 3.8 warp specialization + Blackwell](triton-3-8-autows-warp-specialization-blackwell-open-compiler-2026.md)
- 2026-09-15 [Model2Kernel：SOSP 2026 model-aware symbolic execution CUDA kernel verification](model2kernel-353-cuda-bugs-vllm-huggingface-symbolic-execution-sosp2026.md)

*Sources*：

- [TensorLift arXiv 2604.13523](https://arxiv.org/abs/2604.13523) — 主論文
- [TensorLift v2 HTML](https://arxiv.org/html/2604.13523v2) — 完整 8-pass pipeline 描述
- [ACT: Automatically Generating Compiler Backends from Tensor Accelerator ISA Descriptions](https://arxiv.org/pdf/2510.09932) — TensorLift 下游 backend generator
- [Berkeley Gemmini](https://github.com/ucb-bar/gemmini) — 主要評估對象
- [TVM VTA](https://tvm.apache.org/docs/topic/vta/index.html) — 第二評估對象
- [MLIR-AIE 1.3 release](https://www.phoronix.com/news/MLIR-AIE-1.3) — AMD Ryzen AI NPU 的 MLIR 生態進展（旁證：MLIR 在加速器領域的普及）
- [JLIR arXiv 2609.04585](https://arxiv.org/abs/2609.04585) — Julia-Native MLIR IR，MLIR 抽象滲透到其他語言生態的另一佐證
