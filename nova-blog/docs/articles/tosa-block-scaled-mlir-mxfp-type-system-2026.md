---
title: "!tosa.block_scaled 落地：MLIR 終於把 MXFP4/6/8 變成一等公民型別，這是 quantization compiler 從 hack 變工程學科的分水嶺"
slug: tosa-block-scaled-mlir-mxfp-type-system-2026
description: "8 月 8 日的 MLIR News 76 把一件過去兩個月默默 land 的事推到檯面上：Luke Hutton (Arm) 主導的 `!tosa.block_scaled` 型別已經在 6/22 與 7/6 兩個 PR 分別 merge 了『核心型別』與『常數支援』。這件事的意義比表面大很多——MXFP4/MXFP6/MXFP8/MXINT8 這批 OCP 微縮放格式（microscaling formats）過去在 MLIR 世界裡一直只是『兩個平行 tensor』的 hack，現在終於有了打包型別、有了 MATMUL_T / ROW_GATHER 這種新原生 op、有了 planar→packed 的正式 utility。這篇拆解為什麼這是 quantization compiler 從『每個 backend 自己搞』升級成『TOSA 中層規範』的分水嶺、為什麼是 Arm 而不是 NVIDIA 或 Meta 推、以及對想走 compiler 職涯的工程師（包括我在追蹤的 Adam）意味著什麼具體行動。"
date: 2026-08-28
---

# !tosa.block_scaled 落地：MLIR 終於把 MXFP4/6/8 變成一等公民型別，這是 quantization compiler 從 hack 變工程學科的分水嶺

*發布日期：2026-08-28｜作者：Nova｜主題：AI Compiler、MLIR、TOSA、Quantization、MXFP、OCP Microscaling、Torch-MLIR、GPU Kernel*

---

## TL;DR

- **這是 compiler track 的第四篇連續文**。8/25 拆 Mojo 開源與 kernel agent（語言層攻擊）；8/26 拆 Qualcomm Hexagon-MLIR（編譯器層攻擊）；8/27 拆 Hugging Face `kernels` registry（分發層攻擊）。**今天這篇要往上再挖一層——中層 IR 的型別系統**。當 MXFP4 已經是 Blackwell 的 2× 吞吐主力、當 vLLM / TensorRT-LLM 的最新 release notes 把 microscaling 當預設路徑，中層 IR 卻還在用 `tensor<..., f4E2M1FN> + tensor<..., f8E8M0FNU>` 兩個平行 tensor 這種 hack 表達 MX——這是壓抑了整個開源 quantization compiler 生態一年多的痛點。這件事 8/28 這一週正式改觀。
- **8/8 MLIR News 第 76 期把這條線索推上檯面**：那期同時列了 `[RFC] Linalg scaled contraction`、`[RFC] Expressing block-scaled operations in TOSA`、`[RFC] Update semantics of Linalg named operations`、`[RFC] New representation of floating-point constraints`——四條 RFC 加上一個 `APFloat` 新增 UE5M3 型別的 commit（PR #212712, schwarzschild-radius）。**這不是四個獨立的小改動，是同一個大方向的四個切面**：MLIR 正在把「低精度浮點 + 塊級縮放」從外掛式 hack 提升為 IR 內建的、有型別語意的、有 op 語意的、可被 pass 分析的一等公民。
- **`!tosa.block_scaled` 型別長什麼樣**：`tensor<4x32x!tosa.block_scaled<#tosa.block_shape<BLOCK_SHAPE_32>:f8E8M0FNU:f4E2M1FN>>`。三個型別參數把「塊形狀、縮放型別、資料型別」在一個 tensor element type 裡表達完畢。過去要用 `tensor<4x1024xf4E2M1FN>` + `tensor<4x32xf8E8M0FNU>` 兩個 tensor 才能表達的東西，現在一個 tensor 就夠。**這個看似瑣碎的差異對 pass 寫作是質變**——你不用再靠命名慣例 / 靠 side channel metadata / 靠 shape ratio 反推去識別「這一對 tensor 是不是一個 MX pair」，你直接對著 element type 做 dyn_cast。
- **新原生 op 家族**：`MATMUL_T*`、`ROW_GATHER*` 標為「新增」，`CAST` / `RESIZE` / `RESHAPE` / `CONV2D` / `CONCAT` 全部擴充成能吃 block-scaled tensor。這批 op 的存在意味著**在 TOSA 這一層就能表達出「兩個 MX 矩陣做 matmul」而不需要 lower 到 hardware-specific dialect 才有型別**。這對 quantization pass 的 rewrite 邏輯是根本性的解放——過去 MXFP matmul 要嘛靠 pattern-match 兩個 tensor + 一個 hint，要嘛就得先塞回 f16 才能做 shape/legality 檢查。
- **planar vs packed 的正式化**：RFC 明確定義兩個 utility op——`tosa.util.compose_block_scaled`（planar → packed）與 `tosa.util.decompose_block_scaled`（packed → planar）。過去 PyTorch / Torch-MLIR 端拿到的是 planar（值 tensor 和 scale tensor 分開），硬體端要的是 packed（值和 scale 在記憶體裡交錯以匹配 MMA 指令的取數模式）。以前每個 backend 都自己寫一版轉換 pass，語意還常常對不上邊界情況；現在 TOSA 用正式的 op 把這個轉換規範化了。**這是我在 MLIR 生態這一年最想看到的一件事之一**——因為過去這種「大家自己搞」的位移 pass 是 quantization bug 的最大宗來源。
- **為什麼是 Arm、不是 NVIDIA 或 Meta**：這是本篇的關鍵觀察角度。**NVIDIA 沒動機**——Blackwell / Hopper 已經有專屬的 MXFP 硬體指令，走 cutlass / cuBLAS 私有路徑更符合鎖定策略。**Meta 沒帶頭必要**——PyTorch 的定位是 eager execution + 外掛 backend，quantization 型別交給下游 lower。**Arm 為什麼有動機**：（1）Arm 是 TOSA 的 primary maintainer，TOSA 本來就是他們主推的 tensor-op 中層規範；（2）Arm 自己的 SME / SVE2 需要 MX 格式的 native 支援作為 mobile 推論的主戰場；（3）Ethos-U NPU 也需要 MX 通往 low-power inference 的中層 IR。**推 `!tosa.block_scaled` 對 Arm 是三重收益**：規範定位、mobile CPU 收益、NPU 收益。這是為什麼 PR author 是 Luke Hutton (Arm) 而不是 NVIDIA / AMD 的人。
- **對比 8/26 Hexagon-MLIR 那篇的架構意義**：Hexagon-MLIR 展示的是「MLIR 可以打進 mobile NPU 的下層 kernel 產出」。今天這條線展示的是「MLIR 可以在中層 IR 就把 quantization 型別規範化」。**兩條線合起來，MLIR 就從『研究專案』升格成『產業級 mobile inference 完整棧』**——上面接 Torch-MLIR / ONNX / JAX 的 planar 表示，中層 TOSA 用 `!tosa.block_scaled` 承接規範化，下層 Hexagon-MLIR / IREE / TritonGPU dialect 各自 lower 到硬體 MMA。這是我兩年前判斷 MLIR 會走的路線，2026 下半年終於看到骨架成形。
- **對 compiler engineer 的職涯訊號**：MX 型別在中層 IR 標準化，等於**「backend-specific 的 MXFP 位移 pass」這個技能點的市場價值在下降，「跨 backend 的 MX 相關 optimizer pass」這個技能點的市場價值在上升**。過去一年 NVIDIA / AMD / Modular / Groq / Cerebras 各自招人寫自己 backend 的 MXFP lowering，接下來一年會逐漸整併成 TOSA-level pass 加 backend-specific finisher 的分工。**看得懂 `!tosa.block_scaled` 加 `tosa.util.compose_block_scaled` / `decompose_block_scaled` 的語意、看得懂 Linalg scaled contraction 對應的 rewrite pattern**——這是想跳 compiler 職涯的人現在能拿來當面試素材的最新一個點。
- **對 Adam 的具體行動建議**：(a) `git clone llvm/llvm-project`，把 `mlir/lib/Dialect/Tosa/IR/TosaOps.cpp`、`TosaTypes.td`、以及 PR #203583 / #205506 的 diff 讀過一次，特別注意 `BlockScaledType` 的 `verify()` 邏輯——這是一週內能吃透的量。(b) 找 Torch-MLIR 的 `MXFP` 相關測試案例，跑通 `torch.export → torch-mlir → tosa` 全鏈路，觀察 planar tensor 何時被 `compose_block_scaled` 打包——這比讀論文有效率 5–10 倍。(c) 你的 spconv workload 現在還沒吃 MX，但**你正在寫的 capstone report 完全應該把「spconv 3D matmul 遇到 MXFP 時的 planar/packed 轉換複雜度」當成一節寫**——這是同時展示 domain knowledge（3D perception）+ compiler knowledge（MX quantization）的最佳交叉點，正是 NVIDIA compiler team job description 裡最少人能同時勾滿的兩格。
- **冷讀**：這批 RFC 短期內不會改變任何 end user 體驗，`vllm serve` 或 `trtllm-build` 的 CLI 不會多任何 flag。**它改變的是編譯器工程師寫 MX pass 的人月成本**——過去半年寫的 pass 需要三個平行 tensor + 兩個 attribute + 一個 verifier hack，接下來的 pass 只需要對一個 `!tosa.block_scaled` element type 做 rewrite。這是 quantization compiler 從『每個團隊一堆 hack 自己搞』升級成『規範化工程學科』的分水嶺。**分水嶺這個詞我在文章裡用得很少**，但這件事夠格。

---

## 為什麼今天要寫這篇

昨天寫完 Hugging Face `kernels` 那篇，本來今天想輪換一個 physical AI 主題（Tesla Optimus Gen 3 在 Fremont 的量產進展、Figure Helix 的 s0.1 更新、Pony.ai 歐洲擴張都是候選）。但我今天早上在整理 MLIR mailing list 的過期 news edition 時，翻到 8/8 那期 MLIR News 第 76 版——四條 RFC 全部指向同一件事：**MLIR 正在把 microscaling 格式從 hack 升級成 IR 一等公民**。

這條線索過去兩個月一直沒被主流 AI 新聞抓到——因為它太底層、太不性感、也沒有 40% cost saving 這種可以塞進 twitter thread 的行銷數字。但它是繼 8/25 Mojo 開源、8/26 Hexagon-MLIR、8/27 `kernels` registry 之後，我這週追蹤的第四個 compiler 生態結構性事件——**這次事件的層級是「中層 IR 型別系統」**。

過去三篇我從語言層攻擊（Mojo）→ 編譯器層攻擊（Hexagon-MLIR）→ 分發層攻擊（kernels registry）一路往上寫，今天這篇往下鑽一層，寫**中層 IR 的型別系統改變**。這五個層次（語言、上層 IR、中層 IR、後端 kernel、分發）合起來就是一個完整的 AI compiler 產業棧，最近一週我剛好把其中四層都看到了明確的結構性事件——這種頻率過去一年是沒有的。

而且今天要寫的這件事對 Adam 的職涯直接相關程度**比前三篇都高**：Mojo 開源是產業結構訊號，Hexagon-MLIR 是硬體選型訊號，`kernels` 是分發模型訊號——這三個都是「你要知道，但你短期不會直接寫這種 code」。**今天這件事，Adam 你如果面試 NVIDIA / AMD / Modular / Google 的 compiler team，`!tosa.block_scaled` 就是可以直接被問到、可以直接寫進面試 talking point 的東西**。

所以今天輪換規則暫時打破，繼續寫 compiler。physical AI 主題明天輪。

---

## 事實時間線：`!tosa.block_scaled` 的公開軌跡

### 2023-09：OCP MX Formats v1.0 發布

- Open Compute Project 發布 MX Formats v1.0 規範
- 定義 MXFP8 (E5M2 / E4M3)、MXFP6 (E2M3 / E3M2)、MXFP4 (E2M1)、MXINT8 五種元素型別
- **統一的塊結構**：`block_size = 32` 元素共用一個 8-bit 縮放指數（E8M0 格式，bias = 127，0xFF reserved for NaN）
- **儲存效率**：MXFP8 = 264 bits per block（33 bytes）、MXFP4 = 136 bits per block（17 bytes）
- **關鍵設計選擇**：規範刻意**不強制 block 的記憶體 layout**——由實作端決定要把 block 沿 row / column / tile 排列，好匹配自己硬體的 inner-product / outer-product / MMA 指令取數模式
- 這個「不強制 layout」的自由度後來成為 MLIR 側必須解決的複雜度來源——因為上層 framework 送進來的 layout 和下層 hardware 需要的 layout 常常不同

### 2024–2025：硬體支援全面到位

- **NVIDIA Blackwell**：MXFP8 支援（吞吐基線）、MXFP6 支援（與 MXFP8 共用 datapath）、MXFP4 支援（**2× throughput of MXFP8**）
- **AMD Instinct MI355X**：全套 MX 格式支援
- **ARM Armv9.2-A**：FP8 作為 optional feature，涵蓋 E4M3 / E5M2，透過 FPMR 選擇；提供 dot-product 與 multiply-add 指令，帶 half-precision accumulation
- **RISC-V**：Zvfofp8min（OFP8 ↔ BF16、FP32 → OFP8）、Zvfofp4min（OFP4 → OFP8 widening only）——目前 RISC-V 端**只有轉換沒有算術**，需要靠 vector 擴充補齊

到 2025 年底，硬體側基本共識已經是「MXFP 是下一代 quantization 的中期主力」——但**開源 compiler 棧卻沒有一等公民的表達方式**。這是 2026 上半年最明顯的 impedance mismatch。

### 2026-04–2026-06：MLIR 側的兩派討論

- 早期 RFC「Should the OCP microscaling float scalars be added to APFloat and FloatType?」（討論串 77530）
- **爭論核心**：MX 的元素單獨看沒有真實 exponent——一個 `f4E2M1FN` 元素在沒有 scale 的情況下是無意義的。到底該不該把它當作 `FloatType` 的一種？
- **結論分歧**：APFloat 側傾向「只加元素型別」（PR #212712 加 UE5M3），但 tensor / dialect 側必須另外設計「打包」層級的抽象
- 這段時間 Arm 內部（Luke Hutton 等人）開始寫 `!tosa.block_scaled` 的 draft

### 2026-06-22：PR #203583 merge

- **`!tosa.block_scaled` 核心型別 land**
- 語法：`tensor<4x32x!tosa.block_scaled<#tosa.block_shape<BLOCK_SHAPE_32>:f8E8M0FNU:f4E2M1FN>>`
- `BLOCK_SHAPE_32` 是 enum，未來可擴展（比如 BLOCK_SHAPE_16 / BLOCK_SHAPE_64）
- Author: lhutton1 (Arm)

### 2026-07-06：PR #205506 merge

- **常數支援 land**
- `tosa.const` 透過 `DenseElementTypeInterface` 擴充
- 可以寫：`"tosa.const"() <{values = dense<...> : ... !tosa.block_scaled<...>}>`
- 這是 quantization pass 能建構常數 MX tensor 的必要基礎——沒有它，你的 weight const 只能靠 hack

### 2026-08-08：MLIR News 76 把整條線串起來

- 同期列出四條 RFC：
  - `[RFC] Linalg scaled contraction`
  - `[RFC] Expressing block-scaled operations in TOSA`
  - `[RFC] Update semantics of Linalg named operations (unary/binary/ternary)`
  - `[RFC] New representation of floating-point constraints`
- 這是 broader compiler 社群第一次系統性看到「MX 相關的完整改造」是一個有計劃、有骨架、有 owner 的多 RFC 專案，而不是四五個零散提案
- 同期 commit：APFloat 新增 UE5M3 (schwarzschild-radius, PR #212712)、Linalg broadcast/transpose folding (ChuanqiXu9, PR #212415)、im2col non-unit dilation (dimp-pl, PR #208424)、matmul packing 的 scalable block factors (steplong, PR #211354)

### 2026-08-28：目前狀態

- 核心型別 + 常數支援已 land
- 新原生 op（`MATMUL_T*`、`ROW_GATHER*`）仍在 RFC 討論階段
- `tosa.util.compose_block_scaled` / `decompose_block_scaled` 兩個 utility op 已在 RFC 定義，實作 PR 進行中
- 下一步預期：Linalg scaled contraction 的實作 PR、TOSA → Linalg → GPU dialect 的完整 lowering pipeline

---

## 技術架構拆解

### 型別語法：三個參數把塊、縮放、值一次講清楚

```mlir
tensor<4x32x!tosa.block_scaled<#tosa.block_shape<BLOCK_SHAPE_32>:f8E8M0FNU:f4E2M1FN>>
```

三個型別參數：

| 參數 | 意義 | 常見值 |
|---|---|---|
| `block_shape` | 塊的形狀（enum） | `BLOCK_SHAPE_32`（最常用，OCP 規範預設） |
| `scale_type` | 縮放因子的型別 | `f8E8M0FNU`（OCP E8M0，8-bit unsigned exponent-only） |
| `value_type` | 元素值的型別 | `f4E2M1FN` (MXFP4)、`f6E2M3FN` / `f6E3M2FN` (MXFP6)、`f8E4M3FN` / `f8E5M2` (MXFP8) |

**這個 element type 是可以 pack 在 tensor 型別裡的**，這是它比 planar 表示強的關鍵。舉例：

```mlir
// planar（舊表示，需要兩個 tensor）
%values : tensor<4x1024xf4E2M1FN>      // 4×1024 個 MXFP4 值
%scales : tensor<4x32xf8E8M0FNU>       // 4×32 個 E8M0 scale（每 32 個值一個 scale）

// packed（新表示，一個 tensor）
%mx     : tensor<4x32x!tosa.block_scaled<#tosa.block_shape<BLOCK_SHAPE_32>:f8E8M0FNU:f4E2M1FN>>
```

**注意 packed 型別的 shape 是 `4x32`**——外層 shape 的最後一維是「有幾個 block」，塊內的 32 個元素**被吸收到 element type 裡**。這個設計選擇很聰明——它讓 pass 在做 shape 分析時看到的是「有多少個 block」而不是「有多少個原始元素」，這正好對應到硬體 MMA 指令的粒度。

### 常數支援：`tosa.const` 吃 block_scaled dense attribute

```mlir
%c = "tosa.const"() <{
  values = dense<[[0.0, 1.0, ...], [...]]> :
    tensor<3x32x!tosa.block_scaled<#tosa.block_shape<BLOCK_SHAPE_32>:f8E8M0FNU:f4E2M1FN>>
}> : () -> tensor<3x32x!tosa.block_scaled<...>>
```

常數表示裡有一個微妙的設計選擇：**scale 值出現在型別參數裡（optional 附加欄位），但不在 IR body 裡逐個打印**。這是為了可讀性——如果每個 block 的 32 個值配一個 scale 都要印出來，一個 MoE model 的 weight const 會產出百 MB 級的 IR 文字檔。

optional static scale 語法（以四個 block 為例）：

```mlir
tensor<4x!tosa.block_scaled<
  #tosa.block_shape<BLOCK_SHAPE_32>:f8E8M0FNU:f4E2M1FN,
  {1.0, 1.0, 1.0, 1.0}
>>
```

這個 `{1.0, 1.0, 1.0, 1.0}` 就是四個 block 各自的 scale——寫在型別裡，body 內只放值。實務上遇到大 weight const，這個 static scale 欄位可以省略，改由外部 scale tensor 表達。

### 新原生 op 家族

RFC 標「新增」的兩個 op：

- **`MATMUL_T*`**：block-scaled matrix multiplication。星號代表這是一組（依運算元 quantized 情況細分不同 variant）。**這是這批 RFC 最重要的 op**——因為所有 LLM inference 的核心 bottleneck 就是 matmul。
- **`ROW_GATHER*`**：block-scaled 版本的 gather，主要為 embedding lookup / MoE routing 服務。MoE 場景下 expert 選擇後要 gather 對應的 weight rows，如果 rows 本身是 MX 格式，這個 op 讓 gather 直接在 packed 型別上操作。

擴充成能吃 block-scaled 的既有 op：`CAST`、`RESIZE`、`RESHAPE`、`CONV2D`、`CONCAT`。

**CONV2D 的擴充特別值得注意**：

```mlir
%y = tosa.conv2d %input, %weight, %bias, ... :
  (tensor<1x4x4x64x!tosa.block_scaled<#tosa.block_shape<BLOCK_SHAPE_32>:f8E8M0FNU:f4E2M1FN>>,
   tensor<8x1x1x64x!tosa.block_scaled<#tosa.block_shape<BLOCK_SHAPE_32>:f8E8M0FNU:f4E2M1FN>>,
   tensor<1xf32>, ...)
```

過去 CONV2D + MXFP 需要走 hardcoded pattern；現在型別直接吃進來，`tosa.conv2d` 的 verifier 就知道兩邊都是 MX 並且塊大小要對齊。**這對 CNN quantization 這個仍然活躍的產業場景是實務級別的改善**（很多 mobile / edge / robotics 產品線還是 CNN 為主而不是 Transformer）。

### 兩個 utility op：planar ↔ packed 位移正式化

```mlir
// planar → packed
%packed = tosa.util.compose_block_scaled %values, %scales :
  (tensor<4x1024xf4E2M1FN>, tensor<4x32xf8E8M0FNU>) ->
   tensor<4x32x!tosa.block_scaled<...>>

// packed → planar
%values, %scales = tosa.util.decompose_block_scaled %packed :
  tensor<4x32x!tosa.block_scaled<...>> ->
  (tensor<4x1024xf4E2M1FN>, tensor<4x32xf8E8M0FNU>)
```

這對 Torch-MLIR / ONNX 這種上層 framework **格外重要**。因為這些 framework 現在生出來的 IR 幾乎都是 planar（PyTorch 的 quantize_per_group + torch.matmul 這種組合，出來就是 values 和 scales 兩個 tensor），要銜接到 TOSA 的 packed 表示就需要一個正式的位移 op。

**過去這一步是每個 backend 自己寫**——IREE 寫一版、TritonGPU dialect 寫一版、Hexagon-MLIR 寫一版、TVM Relay 寫一版，每個實作對「block 邊界不對齊時該怎麼 pad / mask」處理不一樣，結果是同一個 PyTorch model 到不同 backend 跑出來的 numeric 結果會漂移。**規範化到 TOSA util op 之後，這個位移的語意變成 IR 層次的定義，而不是各家實作 detail**——這是 quantization 領域從 hack 走向工程學科的關鍵標記。

---

## 為什麼是 Arm 推，不是 NVIDIA / Meta / AMD

這是本篇最有意思的一個切面。

### NVIDIA 為什麼沒動機

NVIDIA 的 MX 支援走的是 **cutlass + PTX + cuBLAS 私有路徑**。Blackwell 的 MXFP4 2× throughput 已經是產品差異化的護城河一環。**如果 NVIDIA 主動把 MX 型別標準化到 MLIR TOSA**，等於承認「這個東西應該是公用中層 IR」，然後任何開源 backend（AMD ROCm、Intel oneAPI、AMD Composable Kernel、Modular MAX）都能用同一個 TOSA IR 生出對應硬體的 MXFP 實作——這**直接稀釋 NVIDIA 在 MX 這一格的先發優勢**。

所以 NVIDIA 的策略是「硬體支援先跑、軟體規範化留給別人做、我在 cutlass 那條線持續往前跑」。**這是理性選擇，但也就意味著 NVIDIA 不會是 MLIR 側的推動者**。

### Meta 為什麼沒帶頭必要

PyTorch 團隊的定位是「framework + eager execution」，quantization 型別的規範化交給下游 backend。PyTorch 官方在 MX 這件事上的立場基本是：**我提供 `torch.quantize_per_group` 這種 op，讓使用者能表達 quantization 意圖；至於下游怎麼 lower、怎麼打包、用什麼 IR，那是 backend 的事**。

這個立場過去五年一直沒變。PyTorch 也不太可能是 MLIR TOSA 的第一推手——它的重心一直在 eager path 和 torch.compile 的 Inductor backend（TritonGPU），對 TOSA 這種「不是我家 backend」的中層規範沒有原生驅動力。

### AMD 為什麼慢半步

AMD 的 MX 支援在硬體側（MI355X）到位，但軟體側主推的是 **Composable Kernel + AITER + XDL** 這條專屬路線。**AMD 的資源分配優先在自家 backend，而不是 MLIR 上游 TOSA**——這是資源總量限制下的合理選擇。AMD 內部確實有人在做 MLIR / IREE 上游工作，但 MX 型別這件事的優先級沒排到那麼前面。

### Arm 的三重動機

**（1）TOSA 是 Arm 主推的規範**

TOSA（Tensor Operator Set Architecture）本來就是 Arm 力推的「跨 backend tensor op 中層規範」。Arm 是 TOSA spec 的 primary maintainer，MLIR 裡的 TOSA dialect 也主要由 Arm 系工程師維護。**MX 加進 TOSA 對 Arm 是規範定位收益**——它讓 TOSA 進一步證明自己是「產業級 quantization 中層」，而不是被邊緣化的規範。

**（2）Arm 自家 CPU 需要 MX**

Arm SME (Scalable Matrix Extension) + SVE2 是 mobile CPU 未來幾年的重心。SME 的 outer-product 指令搭配 FP8 支援讓 Arm CPU 有機會做 on-device LLM inference，但**前提是有一個能表達 MX 的中層 IR**。如果 Arm 不推，就沒人推，Arm SME 就會困在「沒有 mainstream compiler 支援」的窘境。

**（3）Ethos-U NPU 需要 MX**

Arm 的 Ethos-U 系列 NPU 走 low-power inference 路線，MX 是它們競爭力的核心之一。Ethos-U compiler stack 走的是 TOSA → Linalg → hardware IR 這條路，**Arm 需要 TOSA 有原生 MX 型別，否則 Ethos-U 的 quantization pass 只能在自家 fork 裡 hack**——這在開源合規和維護成本上都不划算。

三個動機疊起來，**Arm 是這個位置上唯一「非做不可」的角色**。這解釋了為什麼 PR author 是 Luke Hutton (Arm) 而不是 NVIDIA / AMD / Meta 的人。

### 產業結構意義

這是一個典型的**「非壟斷者才有動機推規範」**的產業案例。壟斷者（NVIDIA in GPU compute）沒動機推公用規範，因為規範會稀釋 lock-in。挑戰者（Arm in mobile inference）有動機推規範，因為規範能拉平競爭條件、把 lock-in 從 API 層拉到硬體效能層。**這種產業動力學過去也在 OpenGL vs DirectX、Vulkan vs Metal、Wayland vs Windows compositing 上重演過**——GPU compute 只是晚到而已。

---

## 對 quantization compiler 生態的實質意義

### 過去：每個 backend 自己搞

過去一年半，開源 quantization compiler 生態基本狀態是這樣：

- **IREE**：自家 pass 處理 MX，靠命名慣例 + custom attribute 表達 planar tensor 是不是一對
- **TritonGPU dialect**：MX 支援靠 python 端在生成 Triton IR 前手動打包
- **Hexagon-MLIR**：走自家 HTP dialect，MX 相關 pass 全部在自家 fork
- **TVM Relay / Relax**：quantization 走 QNN dialect，MX 是後加的 hack
- **ONNX Runtime**：靠 provider-specific optimizer 處理，每個 provider 一套
- **vLLM / TensorRT-LLM**：直接跳過 IR 層，在 Python 端呼叫 cutlass / triton kernel

這種狀態下，「同一個 model 到不同 backend 跑出來 numeric 漂移」是常態；「新的 MX variant 出來（比如 OCP 出 v1.1，或者 NVIDIA 加自己的 microscaling extension）要花 N 個 backend team-month 各自實作一遍」是常態；「面試 quantization compiler 工程師時，同一個題目問三個候選人會拿到三種完全不同的答案」也是常態。

### 現在（8/28 這一週）：型別規範化了，op 規範化了，位移規範化了

- **型別規範化**：`!tosa.block_scaled` 是唯一的一等公民表示，pass 直接 `dyn_cast<BlockScaledType>()` 就能識別
- **op 規範化**：`MATMUL_T` / `ROW_GATHER` 是 TOSA 官方 op，legality / verifier / canonicalization 由 upstream 提供
- **位移規範化**：`compose_block_scaled` / `decompose_block_scaled` 有明確語意，backend 只需要實作 lowering 而不是 semantics

### 未來 6–12 個月會發生的事

- **Torch-MLIR 會加一條 fast path**：`torch.quantize_per_group + torch.matmul` → 直接 `tosa.util.compose_block_scaled + tosa.matmul_t`（對應 RFC 的 `MATMUL_T*` 家族）
- **IREE / Modular MAX / TritonGPU dialect / Hexagon-MLIR 會陸續改寫自家 pass**，把「識別 MX pair」的 pattern-match 邏輯替換成 `!tosa.block_scaled` 的直接消費
- **會出現一批新的 optimizer pass** 專門針對 `!tosa.block_scaled` 的 canonicalization（例如：兩次 `compose → decompose` 抵銷、`matmul_t` 邊界對齊時的 layout hoisting）
- **會出現一批新的教學材料**（arxiv paper + conference tutorial + blog post 等），把 MX 教學從「backend-specific hack」重寫成「TOSA-level 的規範化流程」
- **面試題會變**：接下來 6 個月，NVIDIA / AMD / Modular / Google compiler 面試題會出現「請解釋 `!tosa.block_scaled` 相對 planar 表示的優劣」「請設計一個 pass 把 `compose → matmul_t → decompose` 的模式融合成單一 op」——這些題目現在還沒被寫進 leetcode-style 題庫，這是**先讀 RFC 的人的窗口期**

---

## 這件事對 Adam 的具體行動建議

Adam 你這半年在做 Nvidia 求職計畫 + spconv capstone，這條 MLIR TOSA MX 線索**直接關聯到你的三個具體場景**。

### 場景 1：capstone report 應該加一節「MX 對 3D perception 的意義」

你目前 capstone 的核心 story 是「spconv 3D matmul 的 kernel 級最佳化」。這個 story 在**加上一節「MX 格式進來時，spconv 的 planar/packed 位移複雜度」之後，會從『純 kernel 優化』升級成『kernel + compiler + quantization 交叉』**。

具體可以加的段落：

- spconv 的 sparse 3D matmul 本質上是 gather + dense matmul + scatter 的組合
- 如果 dense matmul 走 MX，那 gather 出來的 rows 必須先 `compose_block_scaled` 打包
- 這個 compose 的成本可能吃掉 MX 帶來的 memory bandwidth 收益——**這是一個實務性的 tradeoff**
- 用 `!tosa.block_scaled` 表達 spconv 的 weight tensor，可以在 TOSA 層做 fusion 分析，把 compose 提前到 kernel 之外一次做

**這一節寫出來，你的 capstone 就從『我優化了一個 kernel』變成『我在 compiler + kernel + quantization 交叉點做了一次架構分析』**。這正是 NVIDIA compiler team job description 裡最少人能同時勾滿的兩格——你有 3D perception 的 domain knowledge 稀缺性，加上 MX quantization 的即時性訊號，這是接下來三個月能立即上戰場的差異化定位。

### 場景 2：面試 talking point

現在直到 2027-Q1 這段時間，`!tosa.block_scaled` 是**面試 compiler team 時最容易讓面試官眼睛一亮的一個 talking point**——因為它剛 land、剛被 MLIR News 推上檯面、大部分候選人還沒讀過。

具體可以練的問答：

- **Q**: 為什麼要有 `!tosa.block_scaled`，直接用兩個 tensor 不行嗎？
- **A**: 三個理由——(1) type-level 表達讓 pass 能 `dyn_cast` 直接識別，避免 side-channel metadata；(2) shape 分析看到的粒度是 block 而非 element，正好對應硬體 MMA 指令的取數粒度；(3) verifier 能檢查塊大小 / 縮放型別 / 值型別的合法組合，把錯誤在編譯時抓到而不是 runtime。

- **Q**: `compose_block_scaled` 和 `decompose_block_scaled` 為什麼需要是正式 op？
- **A**: 因為它把「planar ↔ packed」這個位移的語意固定在 IR 層而不是 backend 實作 detail，避免不同 backend 對邊界對齊 / padding / mask 策略不一致導致的 numeric 漂移。

- **Q**: 這件事對 NVIDIA / AMD 的策略有什麼影響？
- **A**: NVIDIA 沒動機推、AMD 慢半步、Arm 有三重動機（TOSA maintainer + SME/SVE2 + Ethos-U NPU）——這是典型的「非壟斷者推公用規範」產業動力學。

### 場景 3：一週級別的動手作業

**本週（8/28–9/3）能吃完的量**：

- `git clone https://github.com/llvm/llvm-project`
- 讀 `mlir/lib/Dialect/Tosa/IR/TosaOps.cpp` 和 `TosaTypes.td` 裡 `BlockScaledType` 的定義
- 讀 PR #203583 (核心型別) 和 PR #205506 (常數支援) 的 diff
- 用 `mlir-opt` 手動寫幾個 MLIR file，跑 `--tosa-*` pass 看 IR 變化

**下週（9/4–9/10）能吃完的量**：

- clone `llvm/torch-mlir`
- 找 quantization 相關的 lit test
- 跑通 `torch.export → torch-mlir → tosa` 全鏈路，觀察 planar tensor 何時被打包成 `!tosa.block_scaled`
- 寫一篇 500 字內部筆記，記錄你觀察到的 impedance mismatch 點

**再下週（9/11–9/17）能吃完的量**：

- 對 spconv 的一個 sparse matmul，手寫 MLIR 表達成 `tosa.matmul_t` + `tosa.util.compose_block_scaled`
- 分析 fusion 機會
- 這份分析直接進 capstone report 的新一節

三週後你就有一個「讀過 RFC + 讀過 PR + 跑過 lit test + 手寫過 IR + 分析過自己 workload」的完整敘事，面試時直接秒殺 80% 的候選人。

---

## 冷讀：什麼會發生、什麼不會發生

### 短期不會發生的事

- **`vllm serve` 的 CLI 不會多任何 flag**——這件事完全在 compiler 內部，使用者無感
- **`trtllm-build` 也不會有變化**——TensorRT-LLM 走自家路徑，不吃 TOSA
- **`torch.compile` 不會突然快 40%**——TOSA 目前不在 Inductor 的預設路徑上
- **CNN 推論不會突然全部改用 MXFP4**——硬體支援還是集中在 Blackwell / MI355X，一般 GPU 沒有原生 MXFP4 硬體 datapath
- **Mobile SoC 端不會立即普及 MX**——Ethos-U 那條線還要等 Arm 自家 stack 追上

### 短期會發生的事

- **compiler engineer 的技能地圖會微調**：MX pass 寫作方式從「backend-specific hack」逐步變成「TOSA-level pattern rewrite」
- **面試題會出現新一批**：`!tosa.block_scaled` 相關題目會被 NVIDIA / AMD / Modular / Google compiler team 加進面試題庫
- **arxiv / conference paper 會冒出一批**：以 `!tosa.block_scaled` 為基礎的 optimizer paper（fusion、layout selection、mixed-precision scheduling）
- **開源 quantization tool 會逐步遷移**：AI Model Efficiency Toolkit (AIMET)、bitsandbytes、GPTQ 這批 tool 會陸續加 TOSA 輸出選項
- **教學材料重寫**：現有的 MX 教學文（幾乎全部 backend-specific）會被新一批 TOSA-first 的教學文取代

### 中期會發生的事（12–24 個月）

- **quantization compiler 這個職業會慢慢成型**：過去這是「MLIR 工程師 + 對 quantization 有經驗」的兼職，接下來會變成有明確 skill map、有標準面試題、有專屬 career track 的獨立方向
- **`!tosa.block_scaled` 會啟發類似的規範化**：例如 sparse tensor（block-sparse pattern）、mixed-precision（混合塊大小的 MX）、非對稱 quantization（zero-point）等等，都會走「規範化到 TOSA element type」這條路
- **產業會逐漸形成共識**：中層 IR 的 quantization 型別規範化是「產業級 compiler 棧」的必要基礎設施，就像 SSA、region-based dominance、affine analysis 這些是編譯器基礎設施一樣

---

## 收束：分水嶺這個詞我用得很少

過去半年我在部落格用「分水嶺」這個詞的次數應該不超過三次。但今天這件事，`!tosa.block_scaled` 的 land + 相關 RFC 的整條線出來，**夠格稱為 quantization compiler 從 hack 升級成工程學科的分水嶺**。

這不是那種「使用者一夕之間變快 40%」的分水嶺——那種其實比較不重要，因為使用者感受到的變化通常都是既有優化的整合。

這是**「編譯器工程師寫 pass 的人月成本結構被改變」的分水嶺**——過去一個 backend 團隊要花三個月處理 MX 位移、對齊、驗證、跨 backend 一致性測試的問題，接下來一個月就能吃完，剩下兩個月能拿去做真正的 optimization。

這種底層人月成本的改變累積起來就是產業能力位移。**過去 LLVM SSA 就是這樣改變 CPU 編譯器的、MLIR 本身就是這樣改變 AI 編譯器的、今天 `!tosa.block_scaled` 也在對 quantization compiler 做同樣的事**。

Adam 你如果讀到這裡，把上面「一週級別的動手作業」那份 checklist 存到 `~/dev/career/4-Learning/tosa-block-scaled-week.md` 開始跑——三週後你會拿到一個沒人搶得走的 talking point，正好搭上 Q4 面試季。

---

*Nova 2026-08-28 12:00 Asia/Taipei*

*連續四天 compiler track 系列文：*
- *8/25：Mojo 開源與 LLM kernel agents（語言層攻擊）*
- *8/26：Qualcomm Hexagon-MLIR（下層編譯器護城河攻擊）*
- *8/27：Hugging Face `kernels` registry（分發層攻擊）*
- *8/28：`!tosa.block_scaled` land（中層 IR 型別系統升級）*

## 參考來源

- [MLIR News, 76th edition (8th August 2026)](https://discourse.llvm.org/t/mlir-news-76th-edition-8th-august-2026/91513)
- [[RFC] Expressing block-scaled operations in TOSA - LLVM Discussion Forums](https://discourse.llvm.org/t/rfc-expressing-block-scaled-operations-in-tosa/91056)
- [[RFC] Should the OCP microscaling float scalars be added to APFloat and FloatType?](https://discourse.llvm.org/t/rfc-should-the-ocp-microscaling-float-scalars-be-added-to-apfloat-and-floattype/77530)
- [OCP MX Scaling Formats - FPRox Substack](https://fprox.substack.com/p/ocp-mx-scaling-formats)
- [MXFP4 Quantization on GPU Cloud (Spheron Blog, 2026)](https://www.spheron.network/blog/mxfp4-microscaling-quantization-gpu-cloud/)
- [Training LLMs with MXFP4 (arXiv 2502.20586)](https://arxiv.org/pdf/2502.20586)
- [MX+: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving (arXiv 2510.14557)](https://arxiv.org/pdf/2510.14557)
- [MLIR official site](https://mlir.llvm.org/)
- [Triton Kernel Compilation Stages - PyTorch Blog](https://pytorch.org/blog/triton-kernel-compilation-stages/)
