---
title: "`--fp16` 被砍：TensorRT 11 把「精度決策」從 backend compiler 推回到前端 ONNX graph —— ModelOpt AutoCast 是新的 frontend（2026-11-16）"
date: 2026-11-16
tags:
  - compiler
  - tensorrt
  - modelopt
  - autocast
  - mixed-precision
  - strong-typing
  - quantization
  - onnx
  - compiler-boundary
  - compiler-career
  - gpu-kernel
summary: >
  TensorRT 11.x 已經把從 TRT 1 時代就陪伴所有部署工程師的 `--fp16`、`--int8`、
  `--bf16`、`--fp8`、`--int4`、`--fp4`、`--best` 這一整組 trtexec precision
  flag 全部移除，IInt8Calibrator 全家族 class 也在 DriveOS build 被整欄刪
  掉。所有 network 強制改為 strongly typed，weak typing 作為一個 TensorRT
  feature 在 11.0 正式宣告死亡。表面上這是一個 API deprecation，但往下看
  三層才是真正的故事：precision policy 這個「選精度」的 compiler 階段，被
  NVIDIA 整段從 TensorRT backend builder 搬到 ONNX graph，上搬了一層 ——
  由 ModelOpt AutoCast 這個 Python-side graph rewriter 以明寫 Cast node 的
  方式負責，後端只剩「照單全收」的 strong-typing dispatcher。這是一個
  compiler boundary shift：「compiler 決定」的時代結束了，「graph annotate」
  的時代開始了。這跟 LLVM 2010 年代把 auto-vectorization 從 middle-end
  heuristic 搬去 explicit intrinsics / OpenMP SIMD pragma 的動作同構 ——
  都是把政策從 compiler 推給使用者 / 前端工具。對 LiDAR 感知部署來說，
  過去 `trtexec --fp16 model.onnx` 的 one-liner 已經不再成立；標準部署
  流程多了一個強制前置步驟。對正在往 compiler path 布局的工程師來說，
  這是一份活教材 ── 什麼決策屬於 frontend、什麼屬於 backend、兩層怎麼
  用 type system 當 contract 綁定、以及 Torch-TensorRT 用一個軟體 autocast
  pass 蓋在 strong typing 之上「補回舊 UX」的技術決策 ── 全部是當前
  Nvidia / Modular / OctoAI 面試桌上會出現的題目。
---

# `--fp16` 被砍：TensorRT 11 把「精度決策」從 backend compiler 推回到前端 ONNX graph —— ModelOpt AutoCast 是新的 frontend（2026-11-16）

> **TL;DR**
>
> 1. **TRT 11 移除 7 個 precision flag**：`--fp16`、`--int8`、`--bf16`、`--fp8`、`--int4`、`--fp4`、`--best` 從 trtexec 全部拿掉，`BuilderFlag.FP16/INT8/BF16/FP8/INT4/FP4` 這幾個 Python API flag 也被刪。所有 engine 強制 strongly typed。
> 2. **IInt8Calibrator 全家族消失**：TRT 10 時代還留著給 legacy workflow 用的 Int8 calibrator interface 在 11.x DriveOS build 完全移除。Implicit quantization 這個 feature 整段被砍。
> 3. **替代方案是 ModelOpt**：FP16/BF16 mixed precision → `python -m modelopt.onnx.autocast`；INT8/FP8/INT4/FP4 quantization → `python -m modelopt.onnx.quantization`（需要 calibration data）。全部在**build engine 之前**做 ONNX graph rewrite，把 Cast node 和 Q/DQ node 明寫到 graph 裡。
> 4. **這是 compiler boundary 上移**：舊世界 `trtexec --fp16 model.onnx` 的意思是「graph 是 FP32，builder 你自己決定哪些 op 該降到 FP16」—— 這是一個 **backend heuristic**。新世界 graph 是 strongly typed 的、每個 Cast 都明寫，builder 只負責 **selection**（挑 FP16 kernel）、不負責 **policy**（哪些 op 要 FP16）。policy 搬到 ModelOpt 這個前端工具去。
> 5. **這跟 LLVM 的歷史同構**：十多年前 LLVM middle-end 的 auto-vectorizer 用 heuristic 決定哪些 loop 該 vectorize，後來業界慢慢轉向 `#pragma omp simd`、ISPC、Highway 這種 **explicit intrinsics / frontend annotation** 的路線，原因一樣 —— compiler 的 heuristic 在新架構面前總是慢一拍，不如把決策權移交給比較熟 model 的前端工具。TensorRT 11 在做同一個 trade-off。
> 6. **Torch-TensorRT 的對策是「蓋一層軟 autocast」**：strong typing 的部署 UX 很差，Torch-TensorRT 2.12 直接在 dynamo 前端做了一個 rule-based autocast pass，`enable_autocast=True` 會**自動**把 FP32 graph 轉成混合精度（並且 silently 把 `use_explicit_typing` 也打開）—— 這是一個典型 **frontend 蓋在 strict backend 上補 UX** 的 compiler pattern，很值得單獨學。
> 7. **對 LiDAR 感知部署的直接影響**：PointPillars / CenterPoint / BEVFusion 這類模型如果還用 `trtexec --fp16 model.onnx` 的 10.x 流程，11.x 升級後**直接編譯失敗**。要麼先過 ModelOpt AutoCast、要麼用 Torch-TensorRT 的 dynamo 流程。這是 2026 Q4 必修的部署動作。
> 8. **compiler path 面試素材**：這篇列出的四個設計決策 —— boundary 怎麼劃、type system 怎麼當 contract、policy vs mechanism 怎麼分、frontend 怎麼補 backend 的 UX ── 全部是當前 compiler 職缺的必考題。拿這個 case 做 talking point，面 Nvidia TensorRT team、Modular、OctoAI、PyTorch Compile team 都通用。

---

## 1. API 層的變動：七個 flag 一次刪，兩個 class 家族整欄砍

先看表面動作。TRT 11.x 的 migration guide 明寫了這幾件事。

### 1.1 trtexec：所有 precision flag 全部移除

```bash
# TensorRT 10.x — 2024/2025 的標準部署 one-liner
trtexec --onnx=model.onnx \
        --fp16 \
        --saveEngine=model.plan

# TensorRT 11.x — 上面的指令直接編譯失敗，unknown flag
trtexec --onnx=model_autocast.onnx \   # ← graph 要先過 AutoCast
        --saveEngine=model.plan
```

被移除的 flag 清單：

- `--fp16`
- `--int8`
- `--bf16`
- `--fp8`
- `--int4`
- `--fp4`
- `--best`（這個是「打開所有 precision，讓 builder 自己挑」的 Swiss-army flag）
- `--restricted`（相關的 strict-type flag）

### 1.2 Python builder API：對應的 BuilderFlag 也被刪

```python
# TensorRT 10.x
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.INT8)
# → builder 內部跑 precision propagation + 選 kernel

# TensorRT 11.x
# BuilderFlag.FP16, INT8, BF16, FP8, INT4, FP4 全部不存在
# config.set_flag(trt.BuilderFlag.FP16)  # AttributeError
# 唯一的 precision 意圖只能透過 graph 本身的 dtype 和 Q/DQ node 表達
```

### 1.3 IInt8Calibrator 全家族：class 整欄移除

DriveOS build 的 migration guide 寫得最直白：`IInt8Calibrator`、`IInt8EntropyCalibrator`、`IInt8EntropyCalibrator2`、`IInt8MinMaxCalibrator`、`IInt8LegacyCalibrator` 這一整組 TRT 從 v5 時代就有的 calibration interface，**class 本身**在 11.x 被刪除。這代表 implicit quantization 這整段 feature —— 「給我校正資料集，builder 內部跑 KL divergence 或 min-max，自己決定 per-tensor scale」—— 作為一個 TensorRT 功能徹底消失。

### 1.4 用什麼替代

NVIDIA 給的官方替代流程：

| 舊 API / Flag | TRT 11 替代 | 怎麼用 |
|---|---|---|
| `--fp16` / `BuilderFlag.FP16` | **ModelOpt AutoCast (ONNX)** | `python -m modelopt.onnx.autocast --onnx_path=m.onnx --low_precision_type=fp16 --output_path=m_fp16.onnx` |
| `--bf16` / `BuilderFlag.BF16` | ModelOpt AutoCast with BF16 | 同上，`--low_precision_type=bf16`（需要 opset ≥ 22） |
| `--int8` + IInt8Calibrator | **ModelOpt quantization** | `python -m modelopt.onnx.quantization --onnx_path=m.onnx --quantize_mode=int8 --calibration_data=calib.npz` |
| `--fp8` / `BuilderFlag.FP8` | ModelOpt FP8 quantize | `--quantize_mode=fp8` |
| `--int4` / `--fp4` | ModelOpt INT4 / FP4 quantize | `--quantize_mode=int4` / `fp4` |
| `--best` | **沒有直接替代** | 使用者自己決定要 AutoCast 到什麼、要不要再加 Q/DQ |

共通的動作模式：**在 build engine 之前**先跑 ModelOpt 對 ONNX graph 做改寫，把 Cast node 和 Q/DQ node 明寫到 graph 裡，然後 trtexec 看到的是一張**已經強 typed 的 graph**，不需要做任何 precision 決策。

## 2. 真正的設計意圖：把 policy 從 backend 搬去 frontend

表面看這是一次 API spring cleaning。但稍微拉開看就會發現這是一個**compiler boundary 上移**的動作 —— 把「precision policy」這個階段從 TensorRT builder 整段搬到 ONNX graph，上搬了一層。

### 2.1 舊世界：policy 藏在 backend

TRT 10.x 時代，`trtexec --fp16 model.onnx` 的語意是這樣的：

1. **Input**：一張 FP32 ONNX graph（Cast node 很少）。
2. **Builder 要做的事**：
   - **Policy**：在這 500 個 op 裡面，哪些**應該**降到 FP16？哪些會掉精度、不能降？這是一個決策問題。
   - **Mechanism**：對選中的 op，發射對應的 FP16 CUTLASS / cuDNN kernel、處理 layout、處理 scale。這是一個實作問題。
3. **Builder 內部的 precision propagation**：給一組 heuristic（這個 op 是 reduction 不能降、這個 op 在 attention 的 softmax 不能降、這個 op 的 dynamic range 太大不能降），跑一趟圖遍歷，把每個 op 的實際 precision 標好。
4. **Output**：一個 engine，裡面 kernel 用混合精度跑。

**關鍵觀察**：步驟 2 的 policy 是 backend compiler 做的決定。使用者只給了一個 flag（「我願意用 FP16」），具體**哪些 op 真的用 FP16** 是 builder 內部的 heuristic 說了算。

### 2.2 新世界：policy 搬到 frontend，backend 只做 mechanism

TRT 11.x 的流程被重切成兩段：

1. **ModelOpt AutoCast（frontend）**：
   - 吃 FP32 ONNX graph。
   - 跑 **classification rule**：根據每個 node 的 I/O magnitude（default 門檻 512）、initializer 絕對值（default 門檻 65504）、特定 op 類型（LayerNorm reduction、softmax、exp、log 等）、是否需要 opset ≥ 22（BF16）等等規則，決定哪些 node 保留 FP32、哪些可以降到低精度。
   - **輸出**：一張**已經明寫 Cast node 的 ONNX graph**，每條 edge 的 dtype 是確定的。
2. **TensorRT builder（backend）**：
   - 吃上面那張 strongly typed graph。
   - 不做任何 precision 決策。
   - 只做 kernel selection：這個 FP16 GEMM 選哪個 CUTLASS template、這個 layout 怎麼配、fusion 怎麼排。
   - 輸出 engine。

這個切法的核心是 **policy 和 mechanism 分家**。

### 2.3 為什麼是「上移」而不是「下移」或「橫移」

這裡要特別標記一下：**policy 的位置變了** —— 從 backend compiler 的 heuristic，變成 frontend graph 上的 explicit annotation。

- 上移到 frontend：ModelOpt AutoCast 看到的是 **ONNX graph + 使用者意圖**（例如 `--low_precision_type=fp16`、`--calibration_data=...`），它有更多**資訊**可以做決策 —— 它可以跑一次 calibration、可以看統計分布、可以讓使用者排除特定 node。
- 留在 backend 的：純 mechanism —— kernel 選擇、fusion、memory layout。這些需要**硬體資訊**（SM 版本、SMEM 容量、Tensor Core 能力），而且這一層 NVIDIA 本身最熟。

這就是典型的 **"make the easy things in the frontend, the hardware things in the backend"** compiler philosophy。TRT 10 之前 builder 同時做兩件事，是歷史遺產；11 把這個遺產清掉。

## 3. 這跟 LLVM 的歷史同構：auto-vectorizer 向 explicit SIMD intrinsics 讓位

寫到這裡熟悉 compiler 歷史的人應該會想起一件事。這個「policy 從 backend heuristic 搬去 frontend annotation」的動作，在 LLVM 歷史上發生過一次，就是 **auto-vectorization**。

### 3.1 LLVM 的故事（2010-2020）

早期 LLVM middle-end（`-O3`）有一個 loop vectorizer，看到可向量化的 loop 自動發射 SSE/AVX 指令。但實務上出現幾個問題：

- **Heuristic 永遠慢一拍**：AVX-512 出來那一年，middle-end vectorizer 的 cost model 還沒理解 mask register。AVX2 跟 AVX-512 的 vectorization policy 不一樣，middle-end 要兩套 cost model。
- **使用者比 compiler 更懂**：寫數值 library 的工程師知道這個 loop 一定可以 vectorize，但 compiler heuristic 保守不敢碰。
- **Debugging 很慘**：vectorizer 做了什麼、為什麼沒做，使用者看不到。

業界的回應是 **把 policy 推去 frontend / annotation**：

- `#pragma omp simd`：使用者在 source level 明寫「這個 loop 我要 vectorize」。
- `__restrict__`、`alignas`：使用者明寫 alias 和對齊資訊。
- ISPC、Highway、std::simd：使用者直接寫 SIMD intrinsics，compiler 只做 scheduling。

LLVM middle-end 的 auto-vectorizer 沒死，但在性能敏感的部署路徑裡，大家越來越少依賴它。Policy 搬去 frontend 了。

### 3.2 對應關係表

| LLVM 故事 | TensorRT 故事 |
|---|---|
| middle-end auto-vectorizer 用 cost model heuristic 選 loop | TRT 10 builder 用 precision propagation heuristic 選 op |
| AVX-512 出來 heuristic 跟不上 | NVFP4 / FP8 / FP4 出來 heuristic 跟不上 |
| 使用者比 compiler 更懂這個 loop | ModelOpt 比 TRT 更熟這個 ONNX graph |
| debugging 很慘 | TRT 內部 precision 決策不透明，engine inspector 要翻 layer info |
| 業界轉向 `#pragma omp simd` / intrinsics | 業界轉向 ModelOpt AutoCast / 明寫 Cast node |

這個同構不是偶然。兩者都是 **compiler 面對新硬體 feature 的 adoption lag 問題**，兩者都用 **把 policy 推去 frontend** 解。如果你要面 compiler 職缺，這個類比在白板上畫出來就是半個 offer。

## 4. ModelOpt AutoCast 內部：rule-based 的 graph rewriter

AutoCast 的實作不是「跑一個 ML model 來決定 precision」那種花俏做法，而是一個 **rule-based graph rewriter**。NVIDIA 的 Model-Optimizer 文件描述得很直白。

### 4.1 核心分類規則

AutoCast 遍歷 ONNX graph，對每個 node 跑一組決策：

1. **Op kind blocklist**：預設把 `LayerNormalization`、`RMSNormalization`、`Softmax`、`Exp`、`Log`、`Reciprocal` 等**數值敏感 op** 保留 FP32。
2. **Dynamic range check**：對 node 的輸入/輸出做 shape inference + 統計分析，如果 I/O magnitude 預估超過 `data_max`（default 512），保留 FP32。
3. **Initializer magnitude check**：如果 node 的 constant initializer 絕對值大於 `init_max`（FP16 max = 65504），保留 FP32。
4. **Calibration data path（可選）**：如果使用者給了 `--calibration_data`，實際跑一輪 shape propagation + tensor stats，比上面的保守估計更準。
5. **Opset gating**：BF16 需要 opset ≥ 22，FP16 需要 opset ≥ 13，不符合直接升 opset 或 fallback。

### 4.2 輸出：一張 Cast node 明寫的 graph

```
# 舊 FP32 graph（node 省略成節點名）
Conv → BN → ReLU → Conv → LayerNorm → Softmax → MatMul

# AutoCast 改寫後
Conv_fp32 → Cast(fp16) → BN_fp16 → ReLU_fp16 → Conv_fp16 → Cast(fp32) → LayerNorm_fp32 → Softmax_fp32 → Cast(fp16) → MatMul_fp16
```

每一個 Cast 都是**明寫**的 ONNX node，TRT builder 看到這張圖只會想：「FP16 Conv → 我有 FP16 CUTLASS kernel，選 kernel 完成。」builder 不會再思考「要不要降 LayerNorm」—— 因為 graph 已經寫得死死的。

### 4.3 使用者 override：node-level exclusion

這個設計最重要的 affordance 是：**使用者可以 override**。

```bash
python -m modelopt.onnx.autocast \
    --onnx_path=pointpillars.onnx \
    --low_precision_type=fp16 \
    --excluded_node_names=/head/cls_conv,/head/box_conv \
    --calibration_data=kitti_calib.npz \
    --output_path=pointpillars_fp16.onnx
```

LiDAR 感知的 class head / box regression head 有時候對 FP16 很敏感（small object 的 classification score 容易飽和），這時候使用者可以直接**在 ONNX node name 層級明寫排除**。這是舊 TRT `--fp16` 做不到的 granularity —— 舊 flag 是 all-or-nothing，要不全開要不全不開。

## 5. Torch-TensorRT 2.12 的對策：蓋一個軟體 autocast 在 strong typing 之上

接下來是這個故事真正 compiler 味最濃的一段。TRT 11 的 strong typing 雖然在**系統設計上是對的**，但**對 PyTorch 使用者的 UX 是災難** —— 誰想在 `torch.compile(model, backend="tensorrt")` 之前還要手動跑 ONNX 導出 + ModelOpt AutoCast？Torch-TensorRT 團隊的回應很 compiler：

### 5.1 設計：frontend 自己做 autocast pass

Torch-TensorRT 2.12.0 的 `enable_autocast=True` 做了這件事：

1. 使用者在 `torch.compile` 時設定 `options={"enable_autocast": True, "low_precision_type": torch.float16}`。
2. Torch-TensorRT 在 **dynamo trace 之後**、**build TRT engine 之前**，跑一個 **rule-based autocast pass**（功能等價於 ModelOpt AutoCast，但實作在 PyTorch FX graph 層級）。
3. 這個 pass 自動插 `aten._to_copy` node（PyTorch 的 Cast），讓 graph 變成 strongly typed。
4. **自動**設定 `use_explicit_typing=True`（因為 autocast 的輸出就是 strongly typed，必然要搭配 strong typing）。
5. 把這張 graph 送進 TRT builder，builder 看到 strongly typed graph，不做 precision 決策，直接 build。

### 5.2 這是什麼 compiler pattern

這是經典的 **"shim a frontend policy pass on top of a strict backend"** pattern。幾個這個 pattern 的範例：

- **Clang 用 `__attribute__((target("avx2")))` + `__builtin_expect`** 等 frontend annotation，把 policy 推給 source level，backend 只做 mechanism。
- **Rust 的 `unsafe` block**：policy（我負責保證這段是對的）在 frontend，backend 不做 safety 檢查。
- **LLVM `#pragma clang loop vectorize(enable)`**：frontend 強制 policy，backend 不再 argue。
- **Torch-TensorRT autocast**：frontend 做 precision policy pass，backend（TRT 11 builder）只做 kernel selection。

每一個都是 **"backend 把政策權力交出，frontend 用 annotation / pass 填補"** 的同一個設計模式。TRT 11 + Torch-TensorRT autocast 是這個模式最新、最鮮的教案。

### 5.3 代價：policy 從一個地方變成多個地方

這個設計有一個不漂亮的副作用：policy 從**一個地方**（TRT builder）變成**多個地方**（ModelOpt AutoCast、Torch-TensorRT dynamo autocast、使用者手寫的 Cast node、未來可能的 TF-TensorRT autocast）。兩個 frontend 跑同一個 model，很可能給出**不同的**精度決策。

這是把 policy 外移的**必然代價**。優點是每個 frontend 可以根據自己的使用者群客制 heuristic；缺點是 reproducibility 變差了 —— 「為什麼我 PyTorch flow 和 ONNX flow 的 FP16 engine 性能不一樣」會變成一個比以前更常見的 bug report。NVIDIA 要怎麼處理這個 fragmentation，是 2027 要看的戲。

## 6. 對 LiDAR 感知部署的直接影響

拉回 Adam 的職涯軸。這一波 API 變動對實際在部署 LiDAR 模型的工程師，會有幾個立即的動作項目。

### 6.1 升級 TRT 11 時，10.x 的 one-liner 直接失效

如果你目前在用這個：

```bash
trtexec --onnx=pointpillars.onnx --fp16 --saveEngine=pp.plan
```

TRT 11 升級後**直接編譯失敗**（unknown flag）。最小改動是：

```bash
# Step 1: AutoCast
python -m modelopt.onnx.autocast \
    --onnx_path=pointpillars.onnx \
    --low_precision_type=fp16 \
    --output_path=pointpillars_fp16.onnx

# Step 2: Build engine from strongly typed graph
trtexec --onnx=pointpillars_fp16.onnx --saveEngine=pp.plan
```

### 6.2 感知模型常見的 FP16 敏感位置

LiDAR 感知在 FP16 下常見的數值敏感位置（AutoCast 預設會保護，但你應該知道為什麼）：

1. **Sparse conv 的 hash table index 計算**：這個是純整數邏輯，不會降。
2. **PointPillars 的 voxelization scatter**：index + max_pool，前面是 index 不降，後面是 reduction 看 magnitude。
3. **CenterPoint 的 center regression head**：box size / heading 的 regression target 磁性大，建議 exclude。
4. **BEVFusion 的 cross-attention softmax**：預設會保護。
5. **NMS 的 score 比較**：純比較，不敏感。

這些位置現在可以用 `--excluded_node_names` 明寫排除，這是 TRT 10 給不了的工程 affordance。

### 6.3 INT8 workflow 的徹底改變

TRT 10 時代 INT8 部署的核心是 `IInt8EntropyCalibrator2`：寫一個 subclass，餵校正資料，builder 內部跑 KL divergence。這整組 API **在 11.x 被刪除**。

替代做法：

```bash
python -m modelopt.onnx.quantization \
    --onnx_path=pointpillars.onnx \
    --quantize_mode=int8 \
    --calibration_data=kitti_calib.npz \
    --output_path=pointpillars_int8.onnx

trtexec --onnx=pointpillars_int8.onnx --saveEngine=pp_int8.plan
```

ModelOpt quantization 內部一樣跑 calibration + Q/DQ node 插入，但**結果是明寫在 ONNX graph 的 Q/DQ node**，不是 TRT 內部的 per-tensor scale table。這個差異的工程意義是：**量化結果可以 diff、可以 version control、可以在 PyTorch 和 TRT 之間轉換**。舊的 IInt8Calibrator 結果只存在 TRT engine binary 裡，可攜性極差。

### 6.4 一個不方便的現實：legacy model 的轉換成本

如果你的團隊有一堆 TRT 10 時代建好的 engine binary，**升級 TRT 11 後這些 engine 都要 rebuild**（engine binary 本來就不跨 major version）。rebuild 的時候要補 AutoCast / quantization 流程。如果原本的校正資料不在了（只剩 engine），這是一個**不可逆**的技術債 —— 建議趁還有校正資料的時候就把 ONNX + Q/DQ 流程建起來，不要依賴 engine binary。

## 7. 對 compiler 職涯的意義：為什麼這是一份面試素材

Adam 正在往 compiler path 布局（Compiler-Path.md），這篇文章列出的四個設計決策，是當前 compiler 職缺必考的題目類型。整理一下：

### 7.1 「Boundary 怎麼劃」

**題目**：你的 compiler 吃一個中間表示，有一個 optimization 需要知道 user intent。你會把這個 optimization 放在 frontend 還是 backend？

**答題框架**：

- 這個 optimization 需要什麼**資訊**？source-level 的、ir-level 的、hardware-level 的？
- 這個 optimization 跟 **hardware** 綁得多緊？緊 → backend；鬆 → frontend。
- **使用者**比 compiler 更懂這個 optimization 嗎？懂 → frontend annotation；不懂 → backend heuristic。

**用 TRT 11 當 case**：precision policy 需要知道 model 的數值分布（frontend 可以 calibrate）、不太需要知道具體 SM 版本（backend 晚一點算）、使用者對 model 熟 → 放 frontend 對。放對了。

### 7.2 「Type system 怎麼當 contract」

**題目**：兩個 compiler stage 中間，怎麼確保前一段的輸出能被後一段正確消費？

**答題框架**：

- **用 type system 做 interface**：輸出型別明寫、輸入型別明寫。後端只處理 well-typed input，不處理「可能需要補 cast」的 input。
- **用 pass manager 保證不變量**：每個 pass 聲明 pre-condition / post-condition。
- **用 verifier**：中間 IR 有一個 verifier，型別不對直接 abort。

**用 TRT 11 當 case**：strong typing 就是 **「backend 只接受 well-typed ONNX graph」** 這個 contract 的體現。AutoCast 的輸出 = TRT builder 的輸入，兩邊用 ONNX graph 的 dtype annotation 當 interface。

### 7.3 「Policy vs Mechanism 怎麼分」

**題目**：一個 compiler pass 同時做 policy（決策）和 mechanism（執行），怎麼拆？

**答題框架**：

- **拆成兩個 pass**：一個 pass 做決策、產生標註；另一個 pass 看標註執行。
- **決策 pass 可以 pluggable**：不同 target / 不同 user 用不同決策 pass。
- **執行 pass 固定**：對同一個標註，執行永遠一樣。

**用 TRT 11 當 case**：AutoCast 是 **policy pass**（決策哪個 node 降精度），TRT builder 是 **mechanism pass**（照標註挑 kernel）。拆開後 AutoCast 可以被 Torch-TensorRT 的 dynamo autocast 替換、可以被使用者的手寫 Cast 替換、可以被未來的 LLM-based policy agent 替換 —— policy 變 pluggable 了。

### 7.4 「Frontend 怎麼補 Backend 的 UX」

**題目**：你的 backend 很嚴格（strong typing / 不接受 ambiguous input），但使用者抱怨 UX 差。怎麼辦？

**答題框架**：

- **在 frontend 加一個 convenience pass**：吃鬆散的 input、產生嚴格的 output。
- **讓 convenience pass 可選**：進階使用者可以 bypass。
- **convenience pass 要透明**：使用者可以看、可以 override、可以 debug。

**用 TRT 11 當 case**：Torch-TensorRT 的 `enable_autocast=True` 就是這個 pattern 的教科書實作。使用者可以不用（手寫 Cast）、可以 override（`excluded_node_names`）、可以 debug（看 AutoCast 輸出的 graph）。

### 7.5 一個可以帶去面試的 talking point

這段可以直接背下來當 interview talking point：

> 「TensorRT 11 最近做了一個我覺得很乾淨的 compiler boundary 動作 —— 把 precision policy 這整段從 backend builder 搬到 ONNX graph 和 ModelOpt AutoCast，backend 只剩 kernel selection。這在設計上同構於 LLVM 當年把 auto-vectorization 從 middle-end heuristic 轉去 `#pragma omp simd` 的動作。兩者都是把 policy 外移，因為 (1) heuristic 在新硬體（AVX-512 / NVFP4）面前總是 adoption lag，(2) 使用者 / 前端工具比 compiler 更熟 workload。有趣的是 Torch-TensorRT 2.12 在這個 strong-typed backend 上又蓋了一個軟 autocast pass，補回舊 UX ── 這是 frontend 蓋 backend 的經典 pattern，但也引入了 policy 多處實作的 fragmentation，我覺得 NVIDIA 2027 會要處理這個。」

這段話在 compiler 職缺的 interview 裡可以撐 10 分鐘 Q&A，涵蓋 boundary、type system、policy/mechanism、fragmentation 四個面向。

## 8. 延伸閱讀與行動項目

### 給 Adam 的直接行動

1. **這週**：翻一遍自己手上的 LiDAR 部署腳本，把所有 `--fp16` / `--int8` 標出來。如果計畫升 TRT 11，列一份 ModelOpt AutoCast / quantization 的遷移 checklist。
2. **這個月**：找一個小的感知模型，走一次 ModelOpt AutoCast flow，觀察輸出 ONNX graph 的 Cast node 分布，看 AutoCast 保護了哪些 node。
3. **放進履歷 / 面試素材**：上面 §7.5 那段 talking point，背下來。下一次跟 compiler 職缺的工程師聊天時直接用。

### 要追的下一步

- **ModelOpt 的 ONNX quantization LLM 擴展**：FP4/INT4 LLM quantization（AWQ、GPTQ）的 ONNX workflow 2026 Q4 正在成形，值得盯著 GitHub 看。
- **Torch-TensorRT 的 autocast heuristic**：這個軟 autocast 的 rule 2.12 還很保守，後面會不會變複雜（甚至放 LLM 進去決策）是看點。
- **Myelin fusion 對 strong-typed graph 的新 pattern**：舊 weak-typed graph 的 fusion pattern DB 要重寫，這直接影響 FP4 / FP8 的端到端性能（參考前一篇 TRT 11.4 FP4 fusion 回歸的文章）。

---

_來源：NVIDIA TensorRT 11.x Migration Guide（docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/），NVIDIA Model-Optimizer AutoCast docs，Torch-TensorRT v2.12.0 Autocast docs 和 PR #4995（pytorch/TensorRT），LLVM 23.1 release notes 作為 compiler policy 遷移類比的歷史參照。本文對 LLVM 類比屬技術評論 / 類比，不是 LLVM 官方 position。_
