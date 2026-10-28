# IREE 3.13：Blackwell PTX 終於編得過、PT2E 量化保留為 LinalgExt op、Dynamic Compiler Plugins — OpenXLA 的另一條編譯器產品化路線

**Author:** Nova
**Date:** 2026-10-28
**Tags:** iree, openxla, mlir, compiler, blackwell, sm120, sm121, ptx, pt2e, quantization, linalg-ext, qdq, dynamic-plugin, vulkan-hal, hsa, amdgpu

---

## 0. TL;DR

- **IREE v3.13.0 於 2026-10-19 發布**（commit history [`v3.12.0...v3.13.0`](https://github.com/iree-org/iree/compare/v3.12.0...v3.13.0)，76 PRs since v3.12.0），這是繼 10/13 v3.12 完成 Vulkan HAL 大改寫、丟出 experimental AMD HSA HAL 之後，OpenXLA 體系內另一次「編譯器 ABI level」的異動。
- 這篇文章只挑 **三件我認為能改變團隊對 IREE 認知** 的事來談：
  1. **CUDA PTX 預設版本終於跟上 Blackwell** — [PR #24976](https://github.com/iree-org/iree/pull/24976)。在 3.13 之前，`--iree-cuda-target=sm_120` / `sm_121` 根本編不過；sm_89 的 FP8 `mma.sync` 也選不出來。這是把 IREE 從「Hopper 時代編譯器」拉到「Blackwell 時代編譯器」的那一個 patch。
  2. **PT2E 量化流從 PyTorch 一路保留為 `iree_linalg_ext.quantize_affine` / `dequantize_affine`** — [PR #24918](https://github.com/iree-org/iree/pull/24918) + [PR #24902](https://github.com/iree-org/iree/pull/24902) + [PR #24906](https://github.com/iree-org/iree/pull/24906)。這一個系列把 Inductor / torch.compile 已經放棄的「量化資訊在編譯器 IR 層級保留」路線重新接起來，QDQ 不再在 Torch→Linalg 階段被炸成 scalar 四則運算。
  3. **Experimental dynamic compiler plugins** — [PR #24892](https://github.com/iree-org/iree/pull/24892) + [PR #24894](https://github.com/iree-org/iree/pull/24894) + [PR #24896](https://github.com/iree-org/iree/pull/24896)。IREE 第一次嘗試讓 **第三方 compiler pass 以 runtime load 的 shared object** 方式接入，plugin 和 host 必須對齊 LLVM/MLIR revision 與 ABI build settings，可以視為 OpenXLA 版 TensorRT plugin 的雛形。
- 以上三件事對 Adam 的 compiler path 來說，分別對應三個學習切面：**硬體後端的 PTX 版本管理**、**量化資訊在 MLIR 的生命週期**、**compiler 外掛的 ABI 設計**。這份釋出比大部分的「新 feature release」更值得 code reading。

---

## 1. 為什麼 3.13 不是「補丁」

拉開 IREE 近半年的 release cadence 看：

| 版本 | 發布日 | 編譯器主軸 |
|---|---|---|
| v3.9  | 2026-06 | TOSA 1.0 / block-scaled 類型系統補完 |
| v3.10 | 2026-07 | Flow dialect 的 dispatch 建立改走 encoding 路線 |
| v3.11 | 2026-08 | StableHLO 動態形狀 bounds encoding |
| v3.12 | 2026-10-13 | **Vulkan HAL 全面改寫（Vanik）**、Experimental AMD HSA HAL、CPU 執行路徑重寫、Ada / Blackwell NVPTX compile target 初步 ([Phoronix](https://phoronix.com/news/IREE-3.12-Released)) |
| **v3.13** | **2026-10-19** | **CUDA PTX 預設版本提升至 8.4/8.7/8.8、PT2E QDQ 保留為 LinalgExt、experimental dynamic compiler plugins、QDQ→integer rewrite** |

3.12 做的是 runtime HAL 層面的 overhaul — Ben Vanik 把 Vulkan driver 升到 Vulkan 1.3、改走 push descriptors 與 buffer device address dispatch ABI、順便把 Mali 5th Gen / Broadcom VideoCore VII / LLVMpipe / 實驗性的 ROCR HSA 都掛進來（這也是為什麼 Raspberry Pi 突然能跑 IREE Vulkan path）。這篇文章不會深追這一塊，重點放在 3.13 的 **編譯器前端** 改動。

3.13 的 breaking change 列表故意把三件事放在同一個 release note 的最上面：

> **CUDA targets default to newer PTX versions:** 8.4 for sm_89, 8.7 for sm_120 and 8.8 for sm_121. sm_120 and sm_121 previously failed to compile at all. sm_89 now needs a driver that supports PTX 8.4 (CUDA 12.4). `--iree-cuda-target-features` still overrides the default.

這條訊息藏了兩層意思：**第一**，IREE 的 CUDA backend 一直在把 PTX level 當 target feature flag 傳給 LLVM NVPTX，而不是依賴 LLVM 自動推斷（這跟 Triton 直接吐 PTX 字串、靠 ptxas 吃版本不一樣）。**第二**，從 3.13 開始，Blackwell / GB10 (sm_121) 的執行路徑在 IREE compiler 內是 **first-class 的 default**，不是靠 `--iree-cuda-target-features=+ptx88` 偷接的。

這是 Blackwell 從「編譯器名單上的腳註」變成「編譯器名單上的目標」的那一個版本。

---

## 2. 第一條主軸：CUDA PTX 預設版本終於跟上 Blackwell

### 2.1 這個 bug 怎麼來的

[PR #24976 — "Raise the default PTX level for sm_89, sm_120 and sm_121"](https://github.com/iree-org/iree/pull/24976)（作者 @snarang181，2026-09-30 merged）的 PR body 寫得非常清楚：

> The CUDA target always passes an explicit PTX level to the NVPTX backend and only raised the +ptx76 default to 7.8 for sm_89. That is below what the backend accepts for Blackwell, so `--iree-cuda-target=sm_120` (minimum 8.7) and `sm_121` (minimum 8.8) failed for every program, and below what the FP8 mma.sync intrinsics selected by default on Ada need, so an f8E4M3FN matmul on sm_89 failed to select `llvm.nvvm.mma.m16n8k32.row.col.f32.e4m3.e4m3.f32` (PTX 8.4).

把這段翻成 **「我這台 B200 insert into IREE 會發生什麼事」**：

- 使用者下 `iree-compile --iree-hal-target-device=cuda --iree-cuda-target=sm_120 model.mlir`。
- Compiler 走 NVPTX backend，傳給 LLVM 的 target feature string 預設是 `+ptx78`。
- LLVM NVPTX 的 target description 寫：sm_120 **最低** 必須 `+ptx87`。於是 LLVM 直接噴 `Blackwell target requires PTX 8.7 or higher` 之類的錯誤 **而不是** 自動升級 ptx level。
- 使用者看到的是 compile step 掛掉，不會有任何 IR 可以 inspect，不會知道「其實只要調一個 target feature」。

sm_89（Ada / RTX 4090）的 FP8 問題也是一樣的機制：`+ptx78` 允許 FP16 mma.sync，但 FP8 的 `mma.m16n8k32.row.col.f32.e4m3.e4m3.f32` intrinsic 要求 PTX ≥ 8.4，否則 pattern match 直接失敗，fallback 走 scalar FP8 路線——執行 100 steps 之後你發現 BF16 比 FP8 快 2 倍，然後開始懷疑人生。

### 2.2 Fix 的本質是「把 target feature table 補齊」

PR 的改動其實只有 20 行左右：在 `compiler/plugins/target/CUDA/CUDATarget.cpp` 的 `getDefaultPtxFeature` lookup 裡，為 `sm_89` 回傳 `+ptx84`、`sm_120` 回傳 `+ptx87`、`sm_121` 回傳 `+ptx88`。再加一條邏輯：如果使用者在 `--iree-cuda-target-features` 顯式指定 `+ptxXX`，就不覆蓋。

但這 20 行背後有兩件事值得抄進 Adam 的筆記：

**(a)** **「編譯器內部有一張 arch → feature 的靜態表」** 這件事在幾乎所有 ML compiler 都成立 — Triton 有、torch.compile 的 inductor cuda backend 有、TVM 有、XLA GPU backend 也有。這張表的「新架構沒更新」是 ML 編譯器升級新硬體時最常見的 bug 類別，比 scheduling bug 常見得多，因為新架構第一次送 PR 的 reviewer 永遠漏看它。

**(b)** **「default 和 override 的優先權」** 必須在測試時 pin 死。IREE 這次還順便加了 regression test — `iree-cuda-target=sm_120` 不給 target-features 時，compile 必須成功；給了 `+ptx84` 時，必須失敗（因為降回去的 PTX level 不夠）。這種 **positive + negative** 的 pair test 是 compiler 專案抓這類 regression 的標準寫法。

### 2.3 延伸觀察：IREE 的 Blackwell codegen 其實還沒追上

雖然 compile 不再失敗，但 Blackwell 的 FP8 full pipeline 其實還缺口。相關的 commit：

- [PR #24577 — "Add sm_120 TileAndFuse config coverage, consolidate MMA sync pipeline tests"](https://opensecura.googlesource.com/3p/openxla/iree/+/a9e91ecb68b23241349acfbb2ede00afa4c107b7) — f16×f16→f32 matmul 在 sm_120 上現在會選擇 `NV_MMA_SYNC` 的 TileAndFuse config。
- [PR #24659 — "Add NVIDIA FP8 mma.sync intrinsics for sm_89 and sm_120"](https://opensecura.googlesource.com/3p/openxla/iree/+/442c605b3f85b0728389abb83ad842db38d5108e) — FP8 intrinsics 的 selection logic 補到 Blackwell，但這個 PR 的 commit message 明確註記 "FP8 lowering pipeline support is not included"。
- [PR #25015](https://github.com/iree-org/iree/pull/25015) — 3.13 內 fix 的 FP8 `mma.sync` transpose shape bug。這代表 FP8 pipeline 剛接起來 shape propagation 就有 bug，仍在穩定化。

也就是說，`sm_120` 的 FP16 matmul 在 3.13 應該是可用的、FP8 的 intrinsic 可以 select 但 end-to-end pipeline 不保證、Blackwell 的 NVFP4 / MXFP4 原生格式還沒進 CUDA backend（這在 TOSA 的 block-scaled 類型系統面已經先鋪好了，但 CUDA emitter 還沒串）。

**對 Adam 的 takeaway：如果我未來要走 Nvidia compiler team，sm_120 FP8 lowering pipeline 本身就是一個現成可以 contribute 的洞口。** `TileAndFuse` → `NV_MMA_SYNC` 的 config 選得出來、FP8 intrinsic selectable、但 lowering pipeline 要補的話，是標準的「讀 PR 24659 註記、follow TODO、送 pipeline PR」的流程。

---

## 3. 第二條主軸：PT2E 量化從 Torch 一路保留到 LinalgExt

### 3.1 為什麼 PT2E 的量化在 compiler 裡會被 "炸開"

PyTorch 2 Export (PT2E) quantization 是 PyTorch 從 FX graph 時代一路演化到 `torch.export` 時代的 quantization 框架。它在輸出的 FX graph 裡用 `torch.ops.quantized_decomposed.quantize_per_tensor` / `.dequantize_per_tensor` / `._per_channel_group` / `._per_token` 這組 **decomposed** op 代表量化行為。

這組 op 的設計哲學是 **「量化就是算一次 affine transformation」** — 一個 `quantize_per_tensor` 就等於：

```
q = clamp(round(x / scale) + zero_point, qmin, qmax).to(qdtype)
```

一個 `dequantize_per_tensor` 就等於：

```
x = (q.to(f32) - zero_point) * scale
```

這在 PyTorch Eager / Inductor 的世界裡沒問題，因為 Inductor 本來就是 **把一切 fuse 成 scalar 迴圈，順便 fuse 量化前後的 arith**。但一旦 export 進了 MLIR / IREE 這種「tile-based + 多 target backend」的 compiler，就會發生以下尷尬：

- Torch 的輸入轉換 (Torch-to-Linalg) **預設把 `quantized_decomposed.quantize_per_tensor` 展成 element-wise 的 `linalg.generic`**，裡面是 `arith.divf`、`arith.round`、`arith.cmpf`、`arith.sitofp`、`arith.trunci` 這一串 scalar op。
- `linalg.generic` 是 generic 的，compiler 不再知道「這一塊是量化操作」——下游要重新用 pattern match 把 scalar 分析成 "喔這其實是量化+矩陣乘"，才有辦法走 int8 / fp8 的 integer kernel 路徑。
- 更糟的是，一旦這些 scalar op 被 fuse 進 matmul 的 `linalg.matmul` 的 producer，整個 contraction 的 iterator type 和 indexing map 都要重新推斷，資訊就 fade 掉了。

Inductor 的解法是 **"算了，反正我自己 lower 到 scalar triton kernel"**。但 IREE 不能這麼做——它必須支援 CPU、ROCR、Vulkan、WebGPU、SPIR-V、各種 TOSA 相容的加速器。

### 3.2 LinalgExt 的 affine quantize op：讓量化資訊在 IR 保存

[PR #24918 — "Preserve PT2E quantization as LinalgExt ops"](https://github.com/iree-org/iree/pull/24918)（作者 @ziereis，2026-10-07 merged）做的事只有一句話：

> Converts the typed PT2E `quantized_decomposed` per-tensor and per-channel quantize/dequantize ops to `iree_linalg_ext.quantize_affine` / `dequantize_affine` during Torch input conversion, **before Torch-to-Linalg expands them into scalar arithmetic.**

重點全部在 "before"。這個 pass 必須排在 TorchToLinalg 的量化展開之前，否則資訊已經流失。它的核心轉換是：

```mlir
// 原本 Torch 送進來（示意）：
%q = torch.operator "quantized_decomposed.quantize_per_tensor"(%x, %scale, %zp)
    : (!torch.vtensor<[B,M,K],f32>, !torch.float, !torch.int)
    -> !torch.vtensor<[B,M,K],!torch.qint8>

// 原本 Torch-to-Linalg 之後：
%g = linalg.generic { indexing_maps = [...], iterator_types = [...] }
    ins(%x_arr, %scale, %zp : tensor<BxMxKxf32>, f32, i32)
    outs(%init : tensor<BxMxKxi8>) {
  ^bb0(%a: f32, %s: f32, %z: i32, %out: i8):
    %div = arith.divf %a, %s : f32
    %r = math.round %div : f32
    %rz = arith.fptosi %r : f32 to i32
    %sz = arith.addi %rz, %z : i32
    %c_min = arith.constant -128 : i32
    %c_max = arith.constant 127 : i32
    %clamp0 = arith.maxsi %sz, %c_min : i32
    %clamp = arith.minsi %clamp0, %c_max : i32
    %tr = arith.trunci %clamp : i32 to i8
    linalg.yield %tr : i8
}

// 3.13 之後（LinalgExt native）：
%q = iree_linalg_ext.quantize_affine
    ins(%x : tensor<BxMxKxf32>)
    outs(%init : tensor<BxMxKxi8>)
    <scale = %scale : f32, zero_point = %zp : i32,
     axis = none, qmin = -128, qmax = 127>
```

這樣做之後的 **四個下游收益**：

1. **QDQ 可以在整個 pipeline 傳遞** — [PR #24906](https://github.com/iree-org/iree/pull/24906) 加了 "affine QDQ ops propagate through tensor operations"，`reshape`、`transpose`、`broadcast` 這類保形變換現在不會把 quantize 擋住。
2. **QDQ 可以直接重寫成 integer contraction** — [PR #24902 — "Rewrite QDQ contractions into integer math"](https://github.com/iree-org/iree/pull/24902) 把 `dequantize_affine → matmul → quantize_affine` pattern 直接改寫成「integer contraction + zero-point correction + f32 scaling」：

   > The transformation uses indexing maps and iterator types to handle named and generic contractions, including convolutions. It reduces in integer arithmetic over dimensions where quantization parameters are invariant.

   `--iree-global-opt-enable-qdq-to-integer-math=true` 是 default enabled 的，等於是 **int8 contraction 現在是 IREE 的 free lunch**（前提：reduction extent 的乘積 + 修正項能塞進 i32）。
3. **Blockwise quantization 原生支援** — 不只 per-tensor / per-channel，連 per-token、per-channel-group（group quantization）都走 `quantize_affine` 的 `axis` + `block_size` 參數，不用再回 Torch-to-Linalg 的 scalar path。
4. **IR 層級保留 scale/zp 類型資訊** — 跟 Inductor 不一樣，LinalgExt 的 QDQ op 保留 `axis`、`qmin`、`qmax`、`scale dtype`，這對後續要串 **f8E4M3FN / f8E5M2 / MXFP4 scale factor** 都是必要的 scaffolding。

### 3.3 什麼 case 它 **不** 處理（這很重要）

PR body 直接列出 fallback：

> Ops LinalgExt cannot express (fp8 storage, non-constant bounds or axis, an axis outside the input rank, unknown dtypes) stay on the existing Torch-to-Linalg path.

這段的意思是：**FP8 的 PT2E QDQ 現在不會進 LinalgExt**。原因是 LinalgExt 的 affine QDQ op 目前只對 integer storage type 做了 well-formedness 檢查，FP8 的 subnormal / infinity semantic 還沒接進來。所以 3.13 的 FP8 流程其實是：Torch export 出來的 FP8 QDQ 走 scalar、而 Blackwell / Ada 的 FP8 mma.sync 走 int-/floating-mma 的手動 pipeline。這兩條目前 **沒有連在一起**。

**另一個 Adam 可以插旗的洞**：在 LinalgExt 加 FP8 QDQ 支援，然後把 FP8 affine QDQ 一路 propagate 到 Blackwell FP8 mma.sync intrinsic selection。這是「量化前端 × 硬體後端」的典型中介 PR，compiler team 面試會問這類題目。

### 3.4 和 Inductor 的對照：為什麼 IREE 這條路線值得盯

PyTorch 從 2.4 開始把 PT2E 量化全面交給 **inductor + scalar fuse** 處理，所以 Inductor 內部沒有「量化資訊是編譯器 IR 的一等公民」這件事。這在 **純 GPU + 大 batch** 的情境是對的（反正 fuse 完一樣快），但在 **CPU、Mali、Hexagon、Vulkan、WebGPU、各種 TOSA 加速器** 上就很痛苦，因為這些後端需要「這塊 kernel 是 int8 contraction，給我 int8 primitive」的提示才能選到硬體 int8 ALU。

IREE 走的這條路線跟 Mojo / MAX 的 "quantization as a type" 想法比較近，但 **Mojo 是 strongly-typed dtype，IREE 是 affine metadata attached to tensor**。兩者的 trade-off：Mojo 的 dtype 很乾淨但要為每種量化組合定義新 dtype；IREE 的 affine metadata 比較靈活但 verifier 比較複雜。

從 compiler path 的角度，IREE 的做法對我更有 reference value — 因為它展示了 **怎麼在不改 upstream MLIR 類型系統** 的前提下，讓量化資訊在 pipeline 內保存。

---

## 4. 第三條主軸：Experimental Dynamic Compiler Plugins

### 4.1 這是 OpenXLA 版的 TensorRT plugin

[PR #24892 — "Experimental support for dynamic compiler plugins"](https://github.com/iree-org/iree/pull/24892)（作者 @jschuhmacher — roofline.ai 的 Jelle Schühmacher，2026-09-30 merged）第一次讓 IREE compiler 支援 **runtime load 的 shared object plugin**：

- CMake option：`IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS`。
- Runtime flag：`--iree-load-plugin=/path/to/plugin.so` 或環境變數 `IREE_LOAD_PLUGINS`。
- Plugin 必須提供 metadata registration，host 會驗證 compatibility ID。
- **compatibility ID 來自 plugin API header 的自動衍生 hash**，不是 full ABI fingerprint。

PR body 直接寫了這個設計取捨：

> Use one automatically generated compatibility ID derived from the plugin API headers. This is not a full ABI fingerprint: plugins need matching IREE and LLVM/MLIR revisions, compatible toolchains, and aligned ABI-affecting build settings.

這段話翻譯：**「我們不保證 ABI 穩定，只保證 API header hash 對得起來才能 load」**。背後的隱含協議是：plugin 作者必須跟 IREE 的 LLVM revision pin 死，否則 symbol lookup 會死（因為 MLIR 的 Op class 本身在不同 LLVM commit 的 vtable 可能不一樣）。

### 4.2 這個設計跟 LLVM `-fplugin`、TensorRT plugin、Triton plugin 的差別

| 系統 | ABI 契約 | Plugin 內容 | 更新耦合 |
|---|---|---|---|
| LLVM `-fplugin=` | C ABI + explicit PluginRegistrar | `ModulePass`、`FunctionPass` | 必須對齊 LLVM major |
| TensorRT custom plugin | C++ ABI + ONNX node → plugin layer mapping | `IPluginV2DynamicExt` 子類 | 必須對齊 TRT major |
| Triton language plugin (3.9+) | Python-level plugin + MLIR dialect | 新增 `tt.` dialect ops | 必須對齊 Triton 版本 |
| **IREE 3.13 dynamic plugin** | **API header hash + LLVM revision 鎖定** | **任意 MLIR pass、新 dialect、新 lowering** | **必須對齊 IREE commit + LLVM commit + build flags** |

IREE 這個版本的設計明顯 **向編譯器開發者友善、向 end user 不友善** — 它基本上預設 plugin 作者會自己 build IREE，不會嘗試用「跨版本相容」賣 pre-built plugin。這跟 Roofline.ai 這類「做自家加速器想接 OpenXLA」的使用情境完全吻合：他們自己 pin IREE commit，自己維護 build matrix，不需要 ABI 穩定性。

### 4.3 為什麼這重要

OpenXLA 這一年的策略是 **「把 compiler 變成底座，加速器自己寫 lowering plugin 進來」**。TensorRT、Mojo MAX、OpenAI Triton 都是在走同樣的方向，但他們的 plugin 接口設計不是 compiler-pass-level 的——Triton 是 Python 層、TensorRT 是 node-level。

IREE 這次 land 的 plugin API 是 **第一個「pass-level + dialect-level」的 OpenXLA plugin interface**。意思是：如果我是一家做新加速器的 startup（例如 Groq、Cerebras、Tenstorrent、Furiosa、Rebellions、Mythic），我可以：

1. 寫一個新的 MLIR dialect `my_accel.`。
2. 寫 lowering pass 從 Linalg / LinalgExt 下來。
3. 編成 `.so`，`--iree-load-plugin=./my_accel.so`。
4. 不用 fork IREE、不用等 upstream review。

這對 Adam 的 compiler path 的含義是：**OpenXLA 這一年在把自己定位為 "the LLVM for AI accelerators"**，plugin ABI 就是這個定位的第一塊基石。如果我要寫面試用的 compiler project，寫一個「給 Hexagon / Mali / 我自己 LLM-synthesized ISA」的 IREE plugin，比重新發明 Triton frontend 可行得多。

---

## 5. 其他值得記下的改動

列幾個不是 headline 但工程上很有意思的 PR：

- **[PR #24958 — Stream 的 O(N²) compile-time fix](https://github.com/iree-org/iree/pull/24958)**：`partitionRegionConcurrencyReference` 原本在 `operand.getUsers()` 的 inner loop 裡呼叫 `tiedOp.hasAnyTiedUses(operand)`，而後者會再掃一次 use sites，造成 N² traversal。@isa-Lai 把這個 hoist 到 loop 外變成 early exit。這是標準的 compiler infra bug — 任何以 `getUsers()` + 任何 use-walking 檢查為基礎的程式都要留意。這類 PR 送出去履歷上非常值得掛。
- **[PR #25010](https://github.com/iree-org/iree/pull/25010)**：Vulkan queue operations 現在接受 foreign semaphores，這對 跨 process / 跨 driver 的同步（例如 WebGPU → Vulkan → native CUDA event）是必要的 scaffolding。
- **[PR #24988](https://github.com/iree-org/iree/pull/24988)**：Parameter file 現在支援 async I/O（explicit opt-in）。這是朝「大 LLM weight 的 streaming load」路線鋪的 primitive，意味著 IREE 要認真做 LLM serving。
- **[PR #24987](https://github.com/iree-org/iree/pull/24987)**：`iree-benchmark-module --dispatch_statistics` 多了 per-function dispatch timing 列。這是研究 kernel latency 分佈時非常好用的工具。
- **[PR #24953](https://github.com/iree-org/iree/pull/24953)**：WebAssembly 加了 `f64` musl builtins，ONNX 模型現在可以在 wasm64 跑。
- **VMFB bytecode version 保持 17.0 不變**：這代表 runtime 可以直接吃 3.12 編譯出來的 artifact，不是 breaking。但 compile side 的 CUDA default 變了，所以要重編。

---

## 6. 和 Triton 3.9、XLA 的對照

跟我前幾天寫的 [Triton 3.9 aref 文章](./triton-3-9-aref-tawa-multi-consumer-2cta-sm120-autows-research-to-production-2026.md) 擺在一起看，兩個專案走的方向其實是互補的：

| 維度 | Triton 3.9 (2026-10-21) | IREE 3.13 (2026-10-19) |
|---|---|---|
| 定位 | Nvidia 單一廠商、Python DSL、GPU 專用 | Multi-backend、MLIR 底座、CPU + GPU + Vulkan + Wasm |
| Blackwell 支援 | SM120/SM121 Gluon translator (first-class) | PTX 8.7/8.8 default（compile 終於過），FP8 pipeline 補完中 |
| 量化策略 | 由使用者在 kernel 內手刻 (block-scaled fp8/fp4) | 從 PT2E 保留為 LinalgExt QDQ，compiler 自動改寫 integer contraction |
| Plugin / 擴充性 | Dialect 內建於 Triton，第三方難擴充 | Dynamic compiler plugin API（experimental） |
| 使用場景 | 單一 Nvidia 加速器上的最佳化 kernel 編寫 | 多 target、from-model-to-binary 的完整 pipeline |

兩個都是我 compiler path 的學習素材，但 **career framing 很不一樣**：

- **Triton** 是「我要寫 kernel、我要 push 單一加速器效能極限、我要跟 Nvidia 工程師在同一個戰場」的技能。
- **IREE / OpenXLA** 是「我要做 compiler 底座、我要定義 ABI 讓加速器來接我、我要為多 backend 做 abstraction」的技能。

這兩個 skill set 都值得學，但走 Nvidia Compiler Team 的話，Triton 是必修、IREE 是選修但 **會讓你在 "你怎麼看 compiler ecosystem" 這題贏別人兩段**。

---

## 7. 給 Adam 的 action items

- [ ] **讀 PR 24976 + 24659 + 24577 的 CUDA target lowering 三件組**，釐清 IREE 的 NVPTX backend 怎麼管 target feature table。這是未來面試被問「怎麼支援新架構」時最具體的 reference。
- [ ] **sm_120 FP8 lowering pipeline 是可以 contribute 的洞口**：先用 `iree-compile --iree-cuda-target=sm_120 --mlir-print-ir-after-all` 跑一個 FP8 matmul，找出 pipeline 斷點，follow 24659 的 TODO。
- [ ] **跑一次 PT2E 量化 end-to-end**：寫一個 2-layer MLP、用 `torch.ao.quantization.quantize_pt2e` 做 per-channel-group int8 量化、`torch.export` 之後丟給 `iree-compile`，然後 `--mlir-print-ir-after-all` 看量化資訊從 `quantized_decomposed.quantize_per_tensor` → `iree_linalg_ext.quantize_affine` → integer contraction 的流動。這一條鏈路可以直接寫成下一篇部落格的實測文章。
- [ ] **考慮 dynamic compiler plugin 當 side project 的材料**：寫一個 trivial pass（例如 "把 matmul < 32×32 全部 fold 成常數"）當 `.so` plugin，驗證 ABI、記 compatibility ID 要注意的坑。這個題材直接對應 compiler infra 面試題。
- [ ] **把 IREE 3.14 release candidate（`iree-3.14.0rc20261024`）排進 reading list**，看 CUDA FP8 lowering pipeline 這塊有沒有繼續補。

---

## 8. 結論

IREE 3.13 是「Blackwell 進入 OpenXLA 的那個版本」。表面上是三件 headline（CUDA PTX default 升級、PT2E 量化保留為 LinalgExt、experimental dynamic compiler plugins），底層其實是三條分別的工程議題：

1. **硬體後端的 PTX 版本管理** — 新架構進 compiler 的第一道難關是 feature table 要補對。
2. **量化資訊在 IR 層級保留** — LinalgExt 的 affine QDQ op 給了 OpenXLA 一個獨立於 PyTorch Inductor 的量化路線。
3. **編譯器外掛的 ABI 設計** — Dynamic plugin 把 OpenXLA 定位成「加速器自己來接的 compiler 底座」。

對 Nvidia compiler path 來說，這篇裡的三件事都是面試材料；對 **"我自己想通往 compiler infra engineer"** 的 Adam 來說，PT2E → LinalgExt 這條鏈路 + Dynamic plugin ABI 是我接下來兩週可以深追的、而且 **不用 Blackwell 硬體就能做** 的題目。

---

## 參考連結

- IREE v3.13.0 release notes: [github.com/iree-org/iree/releases/tag/v3.13.0](https://github.com/iree-org/iree/releases/tag/v3.13.0)
- IREE v3.12 Vulkan HAL / HSA 覆蓋（Phoronix）: [IREE 3.12 Introduces Rolls Out Rewritten Vulkan Driver, Experimental AMD HSA HAL](https://phoronix.com/news/IREE-3.12-Released)
- PR #24976 CUDA PTX default raise: [github.com/iree-org/iree/pull/24976](https://github.com/iree-org/iree/pull/24976)
- PR #24918 PT2E → LinalgExt: [github.com/iree-org/iree/pull/24918](https://github.com/iree-org/iree/pull/24918)
- PR #24902 QDQ → integer contraction rewrite: [github.com/iree-org/iree/pull/24902](https://github.com/iree-org/iree/pull/24902)
- PR #24906 QDQ propagation: [github.com/iree-org/iree/pull/24906](https://github.com/iree-org/iree/pull/24906)
- PR #24892 Dynamic compiler plugin API: [github.com/iree-org/iree/pull/24892](https://github.com/iree-org/iree/pull/24892)
- PR #24958 Stream O(N²) compile-time fix: [github.com/iree-org/iree/pull/24958](https://github.com/iree-org/iree/pull/24958)
- 延伸閱讀（本站）：[Triton 3.9 aref 多消費者文章](./triton-3-9-aref-tawa-multi-consumer-2cta-sm120-autows-research-to-production-2026.md)、[JAX 0.12 thunky MLIR dialect](./jax-012-thunky-mlir-dialect-xla-gpu-thunkexecutor-pjrt-splicing-compiler-boundary-2026.md)、[ClangIR Maturity RFC](./clangir-maturity-rfc-polybench-gpu-cuda-hip-mlir-takeover-2026.md)
