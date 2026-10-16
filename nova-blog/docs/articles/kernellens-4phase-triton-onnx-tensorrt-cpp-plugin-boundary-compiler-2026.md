---
title: "KernelLens 4 階段編譯器：Triton @triton.jit 到 ONNX Runtime / TensorRT C++ plugin 的最後一哩路（2026-10-16）"
date: 2026-10-16
tags:
  - Compiler
  - Triton
  - TensorRT
  - ONNX-Runtime
  - PyTorch
  - GPU-Kernel
  - AI-Compiler
  - BoundaryTranslator
  - CareerPath
authors:
  - nova
---

# KernelLens 4 階段編譯器：Triton @triton.jit 到 ONNX Runtime / TensorRT C++ plugin 的最後一哩路

> **TL;DR**
> 2026-10-06 公開在 Zenodo 的 KernelLens（v1.1.9）是一個把 PyTorch 模型裡的 `@triton.jit` kernel 自動編譯成 ONNX Runtime `OrtCustomOp` 與 TensorRT `nvinfer1::IPluginV2DynamicExt` C++ plugin 的框架，官方 benchmark 在 LLaMA 3 RMSNorm、SwiGLU、Qwen 2.5 RoPE 上達到 **3.4x – 7.0x 推論加速**、**最大浮點差 0.0**。它值得單獨寫一篇，不是因為「又一個 autogen tool」——而是因為它第一次把「Python-first kernel 開發文化」與「C++-first production inference stack」中間那條，過去只能用「手工 500 行 boilerplate + PTX 內嵌 + grid lambda 翻譯」才能跨越的斷層，收斂成一個 **4 階段 compiler**。更重要的，是這個 compiler 揭示了 2026 年 AI compiler 職場上最被低估的一條專業路徑：**boundary translator**——不是發明新 IR，而是在兩個已存在、彼此不相容的 IR 之間建立形式化的橋。本篇把 KernelLens 的 4 個 phase 拆開看，點出 16-byte alignment guard 與 PTX ISA clamping 這兩個不起眼卻關鍵的 engineering decision，並討論為什麼這類「翻譯器職位」正在 Nvidia / Modular / OpenAI compiler team 內部快速增加。

<!-- more -->

## 一、Triton kernel 進入 production 時的斷層

### 為什麼 `torch.compile` 不夠

Triton 從 2021 年開始推、2024 年進入 PyTorch Inductor 作為第一等後端，到 2026 年已經是 Python 側寫 kernel 的事實標準。寫一支 RMSNorm 或 flash attention variant，用 Triton 可能 60 行 Python 搞定；用 CUDA C++ 動輒 400 行、還要處理 WMMA / cp.async / shared memory bank conflict。

但是——當模型從「研究 notebook」要搬進「C++ production inference engine」時，整個 story 斷掉。原因很具體：

- **`torch.compile` 面向的是 PyTorch runtime**：它在 PyTorch process 裡 JIT，依靠 Dynamo 抓住 Python frame。production inference 的 C++ engine（ONNX Runtime、TensorRT、TVM runtime、自家 serving stack）不會啟動 PyTorch runtime。
- **`torch.export` 走的是 ATen op set**：它把模型凍結成 ONNX 或 ExportedProgram，但 `@triton.jit` kernel 不在 ATen op set 裡，exporter 看到 Triton launch 就直接拒絕或 trace 成不透明 fallback。
- **ONNX Runtime 與 TensorRT 只認 standard ONNX op**：你的 custom kernel 要變成 `OrtCustomOp`（ONNX Runtime 的 C++ 註冊介面）或 `nvinfer1::IPluginV2DynamicExt`（TensorRT 的 plugin 介面），每一個 custom op 都要手寫 shape inference、serialize、plugin factory、CUDA kernel launch glue。

### 手工橋接長什麼樣

一個生產級的 Triton → TensorRT plugin，手工寫 typically 包含：

1. **Python 端**：用 `triton.compile(...)` 得到 PTX，`cuModuleLoadData` 載入，`cuLaunchKernel` 直接 call。這一段還好。
2. **C++ 端 header**：繼承 `nvinfer1::IPluginV2DynamicExt`，宣告 `getOutputDimensions`、`getWorkspaceSize`、`configurePlugin`、`enqueue`、`serialize`、`clone`、`getNbOutputs`、`getPluginType`、`getPluginVersion`。十幾個 virtual function。
3. **C++ 端 enqueue**：embed PTX string（通常是幾 KB 的 base64），`cuModuleLoadDataEx` + `cuModuleGetFunction`，查詢 `nvinfer1::Dims` 推 grid，`cuLaunchKernel`。
4. **Grid 計算**：Triton 的 `grid=(lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),))` 是 Python lambda，要翻成 C++。手工翻幾支還行，一個模型裡 20 支 fused kernel 的話，就是機械化的苦工。
5. **Plugin factory 與 version registry**：讓 ONNX graph 看到自定 op name 時能實例化對應 plugin。
6. **Shape inference**：TensorRT 需要在 engine build 階段知道 output shape；要在 C++ 裡重現 Triton kernel 的 shape 語意。

對一個典型的 7B LLM，若包含 fused RMSNorm、SwiGLU、RoPE、attention variants 等 4–6 支 custom kernel，手工 C++ plugin 的 boilerplate 行數大約 1500–2500 行、驗證週期 2–4 週。這就是 Python kernel 到 C++ serving 之間，**實際上存在、但很少人公開討論的工程稅**。

KernelLens 要做的事就是把這個稅砍到——官方 claim——**1 行 Python**：

```python
from kernel_lens import compile
compile(my_model, example_inputs, backend="tensorrt")
```

這類 claim 的可信度，取決於它背後的 compiler 架構是不是真的收斂了這個問題。我們進入 4 個 phase 看。

---

## 二、4 階段架構拆解

### Phase 1：Analysis & Manifest Assembly

這個階段處理一個哲學問題：**Triton kernel 的「真實語意」在 Python runtime 執行期才存在**。你看一支 Triton kernel 的 source：

```python
@triton.jit
def rmsnorm_fwd(X, Y, W, N, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    ...
```

你**看不到**的資訊，但 compiler 必須知道的：

- `BLOCK` 的實際值（constexpr，launch 時決定）；
- grid tuple 的解析（通常是 lambda，依賴 `meta['BLOCK']`）；
- 哪些 arg 是 tensor pointer、哪些是 scalar、哪些是 constexpr；
- `num_warps`、`num_stages` 等 autotune 配置；
- 每個 argument 的 dtype 與記憶體對齊假設。

KernelLens 的 Phase 1 用兩個手段同時抓這些資訊：

1. **Dual-Pass Tracing**：第一次 pass 用一個 FX Proxy 攔截 `triton.*.launch` call，記錄「這個 kernel 被哪個 PyTorch Module 以什麼 argument 呼叫」。第二次 pass 走 symbolic shape propagation，推出 output shape。
2. **AST Inspection**：對 kernel source 做 Python AST 走訪，抓出 `tl.constexpr`、`num_warps`、grid lambda 的具體 Python 表達式（還沒 evaluate）。

兩路結果合成一個 **manifest**：一份結構化描述，記下每支 kernel 的 (name, PTX 來源, grid formula, constexpr table, launch config, input tensor spec, output tensor spec)。

這個設計看似簡單，但它藏著一個聰明的切割：**所有依賴 Python runtime 的分析都壓縮在這個 phase 完成**。Phase 2–4 不再需要碰 Python；它們只讀 manifest。這等於把 compiler 的 Python frontend 與 C++ backend 之間建立了一個靜態邊界。

### Phase 2：ONNX Graph Transformation

這一階段的任務是**改寫 ONNX graph，讓它用 custom op node 指向我們將要生成的 C++ plugin**。

原本 `torch.export → ONNX` 遇到 Triton 時會：

- 拒絕 export（如果 Triton kernel 不在 fallback 清單內）；
- 或者 fallback 成 ATen aten.rms_norm + 其他 op 的組合（但這會失去 Triton 原本的 fusion 收益）。

KernelLens 的做法是：

1. 先 export 整個 model 的 ONNX，碰到 Triton launch 時插入一個 **placeholder node**（用 ONNX custom domain，例如 `com.kernellens`）；
2. placeholder node 的 op type 名稱對應 Phase 1 manifest 裡的 kernel name；
3. node 的 attribute 塞入該 kernel 的 constexpr 值、PTX blob 的 content hash、serialize 需要的 metadata。

執行時，ONNX Runtime 看到這個 custom op，透過 plugin registry 找到對應的 `OrtCustomOp` 實作；TensorRT 看到 ONNX 匯入的 custom op，透過 plugin factory 找到對應的 `IPluginV2DynamicExt`。

這個設計避開了「要修 PyTorch ONNX exporter」這條政治難路——它把 Triton 當作「黑盒子但有合約」處理。合約就是 manifest。

### Phase 3：C++/CUDA Code Synthesis & Hardening

這個 phase 是 compiler 工程最密集的地方。目標：根據 manifest，自動生成兩種 target 的 C++ source（+ 嵌入的 PTX blob）：

- `OrtCustomOp` 實作（ONNX Runtime target）；
- `IPluginV2DynamicExt` 實作（TensorRT target）。

幾個關鍵技術決策：

**SymPy-based grid transpilation**：Triton 的 grid 通常是 Python lambda，像 `lambda meta: (triton.cdiv(N, meta['BLOCK']),)`。KernelLens 用 SymPy 把這個 lambda 當作符號表達式解析，再 lower 成 C++ AST——`(N + BLOCK - 1) / BLOCK` 之類的 integer arithmetic。不是文字替換，而是真的走 symbolic simplification，所以 `triton.cdiv(N, 128)` 跟 `triton.cdiv(N, 1 << 7)` 會收斂成相同的 C++ code。這個設計避開了「手工翻 lambda 時容易漏整數溢位」的 bug。

**16-byte memory alignment guards**：這個細節非常值得注意。PTX 的 `ld.global.v4.f32` / `st.global.v4.f32`（128-bit coalesced load/store）要求位址 16-byte aligned，否則觸發 Misaligned Address exception、或悄悄 degrade 成 scalar access。Python 端 `torch.empty_like()` 分配的 tensor 通常是 256-byte aligned（CUDA caching allocator 的語意），但從 ONNX Runtime 或 TensorRT 的 workspace 分配的 tensor **不保證**同樣對齊。KernelLens 在生成的 plugin `enqueue` 裡注入 runtime 檢查，若發現 pointer 不對齊，走一條 scalar fallback kernel。這個細節不出現在論文的「主要貢獻」清單裡，但它是「論文 demo 能跑」與「真實 production 不會踩炸」的差距。

**PTX ISA version clamping**：Triton compile 時會根據當前 GPU 的 compute capability 選最新的 PTX ISA 版本（例如 Hopper 用 PTX 8.3+、Blackwell 用 PTX 8.5+）。但生成的 plugin 若要 forward-compat（舊 CUDA driver 載入新 PTX 會失敗），需要把 PTX 版本 clamp 到一個已知可用的下限。KernelLens 做法是：編譯時掃 PTX 文字，把 `.version` directive 改寫成 target-specific 的最低可行版本，同時 strip 掉那個版本不支援的 instruction（若有的話，整支 kernel 標記為「compute cap X 以下不可用」）。

**Zero-Copy Runtime Binding**（這部分跨 Phase 3/4）：plugin 不自己管 tensor 記憶體——它接收 ONNX Runtime / TensorRT 傳進來的 `OrtValue*` / `void*`，直接把 `.data_ptr()` 傳給 `cuLaunchKernel`。沒有中間的 `cudaMemcpyDeviceToDevice`。這點聽起來理所當然，但手工 plugin 的新手版本經常為了「安全」做一次 copy，把 1 GB 的 KV cache 無意義地搬一次，直接吃掉所有 fusion 收益。

### Phase 4：Zero-Copy Runtime Binding

Phase 4 把生成的 C++ source 編譯成 `.so`/`.dll`，產出一個 **shippable artifact bundle**：

- 編譯好的 shared library（含嵌入的 PTX blob）；
- 修改過的 ONNX graph；
- 一個 loader script（註冊 custom op 到 ONNX Runtime、或把 plugin library 餵給 TensorRT builder）。

Runtime 時：

1. ONNX Runtime / TensorRT 看到 custom op，查 plugin registry；
2. 找到 KernelLens 生成的 plugin 實作；
3. `enqueue` 階段：直接拿 input tensor 的 device pointer，`cuLaunchKernel`，寫入 output tensor pointer；
4. 沒有 Python、沒有 PyTorch runtime、沒有 memcpy。

這就是「zero-copy」的具體語意：**從 model-level graph 的角度看，custom kernel 跟 built-in op 走同一條記憶體管線**。

---

## 三、Benchmark 與要保守看的地方

### 官方數據

- LLaMA 3 RMSNorm：3.4x+ 加速；
- Qwen 2.5 RoPE：接近 7.0x 加速；
- SwiGLU：中段區間；
- 最大浮點誤差：**0.0**（exact bitwise parity）。

浮點誤差 0 是個強聲明。它暗示：生成的 plugin 執行的 PTX 跟 PyTorch 裡 Triton 執行的 PTX 是**同一份 bitcode**。這點很合理——Phase 1 本來就是從 Triton 的 compile cache 抓 PTX，所以不是「翻譯 + 重新編譯」，而是「同一支 PTX，換個 host-side launch 機制」。這也意味著性能提升來自：（a）不走 Python interpreter；（b）不走 PyTorch dispatcher；（c）不經過 Inductor 的 fallback 機制。對小 kernel（RMSNorm、RoPE 這類 memory-bound 的 element-wise），Python overhead 本來就是主要瓶頸，所以 3–7x 符合預期量級。

### 要保守的地方

- **Benchmark 選的全是 element-wise / reduction 類的 small kernel**。這類 kernel 的 Python overhead 占比最大，才會有 7x。若把 flash attention 之類的 compute-bound kernel 搬進來，加速比會壓到 1.1x 以下——因為瓶頸在 GPU 本身，不在 host dispatch。這不是 KernelLens 的缺點，是它適用範圍的真實邊界。
- **Dynamic shape 支援目前不明**。TensorRT 的 `IPluginV2DynamicExt` 介面雖然允許動態 shape，但要寫對 `getOutputDimensions`（給 shape expression，不是 concrete number）。KernelLens 的 Phase 1 若靠 example input trace，只能抓到那個 shape 的 constexpr——實際 production 的 LLM 要處理變長 sequence，這塊會卡 shape inference 的實作深度。文件沒明說怎麼處理，我持懷疑態度。
- **autotune 的處理**。Triton `@triton.autotune` 會根據 input shape 動態選 config。生成的 C++ plugin 需要在 runtime 做同樣的 config lookup，或者 freeze 到一個 config。freeze 的話性能可能不理想，runtime autotune 的話生成程式碼複雜度會上升。這在 v1.1.9 的描述裡沒細談。
- **Hopper / Blackwell 的 TMA、WGMMA**。Triton 3.x 對這些指令有編譯支援，但 KernelLens 的 PTX ISA clamping 若把版本壓低，可能把這些指令整個 strip 掉。對 Blackwell 使用者等於失去主要加速源。

這些保守評論不是在抨擊 KernelLens——是提醒自己：**v1.1.9 是一個「打通典型路徑」的里程碑，不是「涵蓋所有 production corner case」的終點**。而這也正是這類 boundary translator 專案進入下一階段最有工作可做的地方。

---

## 四、Compiler 職涯視角：boundary translator 是機會

### 一個被低估的職位類別

2026 年如果你去 Nvidia / Modular / Meta PyTorch / OpenAI compiler team 翻 JD，會發現職位分成三類：

1. **Core IR 設計**：MLIR dialect、Mojo 的 KGEN、Triton 的 TTIR / TTGIR——人少、資深、天花板高、進入門檻陡；
2. **Backend code generation**：PTX / SASS、ROCm GCN、AMX，碰硬體文件、碰 ISA——需要電子/計算機架構底子；
3. **Boundary translator**：把 A 系統的語意翻成 B 系統——Triton ↔ TensorRT、PyTorch ↔ ONNX、CUDA ↔ ROCm（HIPIFY）、CUDA ↔ Metal（Modular 做的 kernel portability）、JAX ↔ XLA ↔ IREE。

第三類過去被視為「低地位的 glue code」，但 2026 看來不是——**它是硬體與生態系爆炸的直接受益者**。GPU 從 Nvidia 一家獨大變成 AMD MI300X、Intel Battlemage、Apple M4 Pro、華為昇騰、各家 NPU 都想接 AI 工作負載；framework 從 PyTorch 一家變成 PyTorch / JAX / Mojo / TensorFlow 殘黨 + 一堆 inference engine。笛卡兒積 N × M 的翻譯對組合在爆炸性成長，每一對都需要形式化的 compiler 來處理。

### 這類職位的技能剖面

從 KernelLens 這個例子回推，一個合格的 boundary translator 需要：

- **兩邊的語意要熟到可以區分 "same" 與 "equivalent"**（PTX bitwise 相同 vs 數值相同是兩回事）；
- **AST / IR 操作要日常**（Phase 1 做 Python AST 走訪、Phase 3 用 SymPy 符號計算）；
- **對目標系統的 C API / plugin interface 要能徒手寫**（`OrtCustomOp` / `IPluginV2DynamicExt` 不是看文件 20 分鐘能摸熟的，要寫過幾支才會穩）；
- **硬體 corner case 要知道**（16-byte alignment、PTX ISA 相容性、CUDA driver forward compat）；
- **不迷戀「發明新 IR」**——學會接受既有 IR 的歷史包袱，在上面做翻譯層。

這個技能剖面的門檻比 Core IR 低、工作量比 Core IR 大、Nvidia 等公司急缺的程度卻跟 Core IR 一樣高。對從電機工程（如我自己）跨進 compiler 的人來說，這是最務實的切入點之一。

### 自我盤點清單

如果你讀到這裡，想評估自己距離「能做這類工作」還差多少，建議動手試這幾件事：

1. **手寫一個 `IPluginV2DynamicExt`**：挑一個很簡單的 element-wise op（例如 scaled dot），從 .cu 寫到 ONNX 匯入到 TensorRT engine build 成功。目的不是性能，是把這條 C++ 流程走通。
2. **用 Triton 寫同一個 kernel，compile 出 PTX，手動 embed 到剛才的 plugin 裡取代 CUDA 版本**。這個練習會逼你搞清楚 PTX 的 `.version`、`.target`、ABI 細節。
3. **讀 KernelLens 的 PyPI 包 (`pip install kernel-lens`)**：看它生成的 C++ source 長什麼樣。這是「compiler output 的 reverse engineering」最便宜的一次。
4. **挑一個開源 LLM（Qwen 2.5 7B）試著跑一次 `kernellens.compile`**：看它在你的卡上失敗在哪裡。失敗訊息會告訴你 v1.1.9 的真實邊界。

---

## 五、更大的脈絡：Python / C++ 邊界正在被形式化

把 KernelLens 放進 2026 這一年的時間軸：

- **ClangIR** 把 Clang AST → LLVM IR 之間插入了 MLIR-native 中介層；
- **Mojo 1.1 開源 KGEN** 讓 Python-looking syntax 直達 MLIR；
- **CUDA Tile IR** 讓 Triton 的 TTGIR 多了一條可以直接下到 NVVM 的新路；
- **KernelLens** 讓 Triton 的 PTX output 可以直達 ONNX Runtime / TensorRT 的 C++ plugin interface。

這四件事看似分散，但方向一致：**過去「Python 寫、C++ 跑」的 impedance mismatch 正在被 compiler 層逐個收斂**。每一個收斂，都對應一個新的 IR 介面或 code synthesis layer。這些層的作者，正是未來三年 AI compiler 職場的核心人力。

想走 compiler 職涯的人，不必從「我要發明一個新的 IR」開始。從「我要把 A IR 翻譯到 B IR，且數值 bitwise 相同、性能不劣化」開始，就是一個完整的、有商業價值的、面試可以講出來的專案。KernelLens 是個 v1.1.9，不是終點——而 production AI stack 裡的「斷層」還有很多條。

挑一條，把它變成下一個 KernelLens。

---

## 參考資料

- [KernelLens: Deploying Triton Kernels to C++ Inference Engines（Zenodo v1.1.9，2026-10-06）](https://zenodo.org/records/23194523)
- [KernelLens — torch.compile 討論區貼文](https://discuss.pytorch.org/t/kernellens-deploying-triton-kernels-to-c-inference-engines-tensorrt-onnx/225514)
- [Triton 官方 repo（triton-lang/triton）](https://github.com/triton-lang/triton)
- [NVIDIA Blackwell Compatibility Guide 13.4（2026-10-12）](https://docs.nvidia.com/cuda/pdf/Blackwell_Compatibility_Guide.pdf)
- [NVIDIA Developer Blog：CUDA Tile IR Backend for OpenAI Triton](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton/)
- [PyTorch 2.9 Release Notes](https://pytorch.org/blog/pytorch-2-9/)
- [TensorRT Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
