---
title: "FP4 GEMM 快 2 倍、整段 encoder 卻慢 24%：TensorRT 11.4 NVFP4 fusion 回歸的 Myelin 層根因分析（Blackwell SM120, 2026-10）"
date: 2026-10-19
tags:
  - compiler
  - tensorrt
  - nvfp4
  - fp4
  - blackwell
  - myelin
  - kernel-fusion
  - quantization
  - gpu-kernel
  - compiler-career
  - triton
summary: >
  2026-10-15 NVIDIA/TensorRT GitHub Issue #4880 開了一個很尷尬的 bug：在 RTX
  PRO 6000 Blackwell Server (SM120, CUDA 13.4) 上，把 BAAI/bge-reranker-v2-m3
  從 FP8 切成 NVFP4，單個 FP4 GEMM 吞吐量確實跑到 1,140-1,500 TFLOPS、比
  FP8 快 1.7-2.0 倍，但 end-to-end encoder latency 卻在 128×640 setting 下
  變慢 24%，在 128×320 setting 下慢到 45%。拉開 engine inspector 看，答案
  不是 Tensor Core、不是記憶體頻寬，而是 Myelin compiler 在 FP8 路徑裡「免
  費送」的三段 fusion，到 FP4 路徑就全部不見了：Q/K/V slicing 從 0.9ms
  膨脹到 38.3ms（5 個獨立 Myelin kernel），dynamic quantize 從 0ms（fused）
  變成 18.0ms 獨立 kernel，GELU + quantize 從 15.5ms 退化到 21.6ms。本文
  從這份 engine inspector 數據開始，拆解 Blackwell FP4 硬體勝利為什麼在
  compiler 層被幾乎吃光，推演「新 precision level 的 compiler debt
  cycle」，最後對照今天（2026-10-19）在聖荷西開幕的 Triton Developer
  Conference ─ 為什麼 Triton 的 user-controlled fusion / block-level IR
  正是為了避開這類「新硬體發布 → compiler fusion 落後兩個 TRT major」的
  陷阱。對正在往 GPU compiler path 布局的工程師，這篇是一份活生生的面試
  素材。
---

# FP4 GEMM 快 2 倍、整段 encoder 卻慢 24%：TensorRT 11.4 NVFP4 fusion 回歸的 Myelin 層根因分析（Blackwell SM120, 2026-10）

> **TL;DR**
>
> 1. **單算子贏、整模輸的經典反例**：Blackwell FP4 Tensor Core 把單一 GEMM 推到 1,140-1,500 TFLOPS（FP8 的 1.7-2.0 倍），但在 bge-reranker-v2-m3 這個 XLM-R large reranker 上，128×640 的 FP8 engine 跑 160.8ms、NVFP4 engine 跑 200.1ms；128×320 更慘，差距 45%。硬體勝利被 compiler 層幾乎完整吃光。
> 2. **三個 fusion 缺口**：Issue #4880 的 engine inspector 清楚指出三個退化位置 — Q/K/V slicing + transpose + copy 從 0.9ms 膨脹到 38.3ms（五個獨立 Myelin kernel 取代 FP8 的直接寫出）；FP4 dynamic quantize 從 0ms 變成獨立 18.0ms kernel；GELU + quantize 從 fused 15.5ms 退化到 unfused 21.6ms。
> 3. **根因不是 Myelin bug，而是 compiler debt**：FP4 的 per-block scale、動態量化、layout 需求（M×3072 head-major 偏好）在 TRT 11.3/11.4 的 fusion pattern DB 裡還沒被完整覆蓋。FP8 走過了同樣的早期週期 — 2024 年剛導入時也要吃一次 compiler 回歸，等 TRT 10 系列才真的追上。
> 4. **「若 fusion 補齊」的對照**：issue 作者估算，如果 FP4 的 Q/K/V 寫出與 quantize fusion 跟 FP8 一樣完整，整段 encoder 可以壓到 ~127ms（比 FP8 快 21%）。這 ~73ms 的 gap 就是純粹的 compiler debt — 不碰硬體、不改模型，只要 pattern matcher 補一輪就能拿回來。
> 5. **對 Triton 的意義**：今天（2026-10-19）在聖荷西開幕的 Triton Developer Conference 把 user-controlled scheduling / block-level IR 當成主線，就是為了讓新 precision level 不要再陷入「硬體發布 → compiler fusion 落後兩個 TRT major」的 1-2 年等待窗。這是 Triton 相對於 TRT 作為 ahead-of-time black-box compiler 的根本賣點。
> 6. **compiler path 面試素材**：這篇列出的四張表、三個 fusion 位置、per-block scale 的量化 epilogue 設計，是當前 Nvidia / Modular / OctoAI / 任何做 inference compiler 的公司都在考的題目。拿這個 case 作 interview talking point 完全可行。

---

## 1. 一個 2026-10-15 的 bug report：讓硬體勝利當場翻車

TensorRT GitHub Issue #4880 開在 2026-10-15。報告人在 **RTX PRO 6000 Blackwell Server Edition（SM120, 148 SMs, 192GB GDDR7）** 上，用 TensorRT 11.3.0.99 和 11.4.0.106 兩個版本都試過，環境是 CUDA 13.4、Driver 595.91.07、Ubuntu 24.04、Triton Inference Server 26.09-py3。

模型是 **BAAI/bge-reranker-v2-m3**，底層是 XLM-R large 架構的 bi-encoder reranker，是今天多語檢索 pipeline 裡標配的 reranker。

跑 benchmark 時發生了這樣一件事：

| 情境 | FP8 engine | NVFP4 engine | NVFP4 vs FP8 |
|---|---|---|---|
| 128 queries × 320 tokens | 基準 | **+45% latency** | 慢 |
| 128 queries × 640 tokens | 160.8 ms | **200.1 ms** | 慢 24% |
| 單一 FP4 GEMM 吞吐 | — | 1,140-1,500 TFLOPS | 比 FP8 GEMM 快 1.7-2.0× |

硬體勝利（1.7-2.0× GEMM 加速）跟 end-to-end 結果（慢 24-45%）正好相反。這種 contradiction 在 compiler path 裡是最有價值的訊號 — 它幾乎 100% 保證**瓶頸不在 Tensor Core**，而是在 Tensor Core 周圍的某個 scheduling / fusion / memory movement 決策。

## 2. 拆開 engine inspector：Myelin 到底把 fusion 丟在哪裡

TensorRT 的 engine inspector（`trtexec --exportLayerInfo`）在 Myelin fusion engine 下會把每個 fused block 的子操作列出來。Issue 作者把 FP8 engine 跟 NVFP4 engine 對照擺出來，四張關鍵對照：

### 2.1 Q/K/V slicing：0.9ms → 38.3ms，壞 42×

這是最嚴重的一塊退化。XLM-R 的 self-attention 要把 projection 完的 `[M, 3072]` 張量切成 Q、K、V 三份（每份 `[M, 1024]`），再做 head-major 轉置。

| Pipeline 階段 | FP8 engine | NVFP4 engine |
|---|---|---|
| Q/K/V slicing + transpose + copies | 0.9 ms | **38.3 ms** |
| 使用的 Myelin kernels | 1 個（projection 的 epilogue 直接寫 head-major layout） | **5 個獨立 kernel** |

FP8 engine 的 Myelin fuser 看到「GEMM 寫出 `[M, 3072]` → split → transpose → head-major」這條模式，直接把寫出 pattern 改成 head-major layout，attention projection 的 epilogue 就把 Q、K、V 分別寫到對的位置。所以切片本身接近 free — 不需要獨立 kernel、也不需要額外的 HBM round-trip。

FP4 engine 看到同樣的模式但認不出來。原因有兩層：
1. FP4 的輸出帶 per-block scale（NVFP4 的 32-element block + E4M3 scale factor），epilogue 的 output format 跟傳統 FP8 不同，fuser 的 pattern 認不出。
2. FP4 GEMM 的 output 預設是連續 `[M, 3072]` 格式，再在 downstream 做 slice/transpose。slice 本身要處理 scale 的對齊（block 不能跨切），再變成四個獨立的 memcpy-like Myelin kernel + 一個 transpose kernel，加起來 38.3ms。

這一條其實就是一個經典 compiler 題：**當 GEMM epilogue 的 output layout 可以參數化，就能避免 downstream 的 copy。** FP4 版本因為 scale layout 的對齊限制，暫時失去了這個自由度。

### 2.2 Dynamic quantize：0ms → 18.0ms，從 0 到 18

NVFP4 的動態量化（dynamic quantize）需要先掃描 tensor 算 per-block max，再做 scale division + round。這個 pass 在 FP8 engine 裡壓根不存在（FP8 用的是 static per-tensor scale，TRT 在 build 階段固化好）。

| 階段 | FP8 engine | NVFP4 engine |
|---|---|---|
| FP4 dynamic quantize (post-GELU) | 0 ms | **18.0 ms** |

這 18ms 是純損失。而且它是**獨立的 `__myl_Dyna_...` kernel**，意思是：
- Myelin 看到 `GELU` → `dynamic_quantize` 這條鏈，沒有 fuse 成 epilogue
- GELU 寫出 FP16 中間，再被下一個 kernel 讀回來做 quantize

這在 compiler 層是明確的 fusion 缺失。NVFP4 reduce / scaling 要跨 32-element block 做，但只要 block 不跨 warp boundary，一個 well-written epilogue 完全可以把 GELU + per-block max + divide + round 塞進同一個 kernel。Myelin 的 FP4 pattern 還沒把這條鏈進圖。

### 2.3 Residual add + LayerNorm：8.6ms → 18.0ms

Transformer 的標準 pre-attention / post-attention residual add + LayerNorm block，在 FP8 engine 裡 Myelin 會把它 fuse 成一個 reduction + normalize + output quantize 的 fused kernel。

| 階段 | FP8 engine | NVFP4 engine |
|---|---|---|
| Residual add + LayerNorm（含 output quantize） | 8.6 ms | 18.0 ms |

FP4 版本不是沒 fuse，而是 fuse 得淺 — 把 add + LN 合起來，但 output quantize 這段再丟給下一個 kernel。18.0 ms vs 8.6 ms 的 gap 大致就是這一拆兩段的 memory round-trip 代價。

### 2.4 GELU + quantize：15.5ms → 21.6ms

FFN 裡的 GELU + 下一輪 quantize 在 FP8 engine 裡是 fused epilogue（跟 GEMM 寫出合在同一個 kernel）。FP4 engine 裡 GELU 跟 quantize 分家，所以多了一次 HBM round-trip。

```
FP8:  FFN-GEMM → epilogue{GELU + quantize to FP8} → 15.5 ms
FP4:  FFN-GEMM → epilogue{GELU} → store FP16 → load → quantize to FP4 → 21.6 ms
```

6.1ms 的 overhead 就是這一輪額外 HBM traffic 的成本。

## 3. 綜合賬本：GEMM 省下來的時間，被 fusion 缺口全吃回去

把四個 fusion 缺口加起來跟 GEMM 加速擺一塊：

| 類別 | FP8 engine | NVFP4 engine | 差 |
|---|---|---|---|
| 全 attention GEMM（Q/K/V proj + attn + out proj） | ~45 ms | ~22 ms | **−23 ms**（省，GEMM 加速贏回） |
| 全 FFN GEMM | ~50 ms | ~26 ms | **−24 ms**（省） |
| Q/K/V slicing/transpose/copy | 0.9 ms | 38.3 ms | **+37 ms**（compiler debt） |
| Dynamic quantize | 0 ms | 18.0 ms | **+18 ms**（compiler debt） |
| LayerNorm 的 quantize epilogue | 8.6 ms | 18.0 ms | **+9 ms**（compiler debt） |
| GELU + quantize fusion | 15.5 ms | 21.6 ms | **+6 ms**（compiler debt） |
| **其他 overhead** | 剩餘 | 剩餘 | — |
| **總 latency** | **160.8 ms** | **200.1 ms** | **+39.3 ms（24% 慢）** |

GEMM 省下 47ms。fusion debt 多付 70ms。淨虧 ~23ms（加上其他散碎 overhead 湊到 39.3ms）。

**Issue 作者的對照估算**：如果 FP4 engine 的 fusion coverage 補到跟 FP8 一樣完整，整段 encoder 可以壓到 ~127ms — 比 FP8 快 21%，剛好把 GEMM 的加速真正拿回來。

這 ~73ms 的 gap 不是硬體問題、不是模型問題、不是 driver 問題。它是純粹的 **compiler debt**：pattern matcher 還沒長出對應 FP4 epilogue 的 rule set。

## 4. 為什麼會有這個 debt：新 precision level 的 compiler cycle

這不是 TensorRT 第一次踩這個坑。回頭看歷史：

| 精度 | 硬體導入 | TRT 首次「can run」 | TRT 首次「fully fused」 | debt 窗口 |
|---|---|---|---|---|
| INT8（Volta/Turing） | 2017 | TRT 4.0 (2018) | TRT 7.0 (2019) | ~1.5 年 |
| FP16 | 已久 | 一開始就有 | TRT 7+ 整型 | — |
| TF32（Ampere） | 2020 | TRT 7.2 (2020) | TRT 8.0 (2021) | ~0.5 年 |
| FP8（Hopper） | 2022 | TRT 8.6 (2023) | TRT 10.x (2024) | ~1 年 |
| INT4（Hopper/Blackwell） | 2024 | TRT 10.0 | TRT 10.x/11.x 持續補 | ~1-1.5 年 |
| **NVFP4（Blackwell）** | **2025** | **TRT 10.9 / 11.0** | **TRT 11.x → 12.x 待補** | **~1-2 年** |

每一次新 precision level 進來，compiler 都要吃一輪 debt：
1. **先把 kernel 跑起來**（correctness 優先）
2. **GEMM 達到硬體理論峰值**（Tensor Core 直連）
3. **周邊 epilogue 的 pattern 補齊**（fusion coverage）
4. **跨層 fusion**（attention block、FFN block 的整圖）

NVFP4 現在卡在第 3 階段。GEMM 快了（第 2 階段完成），但第 3 階段的 epilogue pattern — 特別是帶 per-block scale 的 output quantize、dynamic quantize 的 inline reduction、head-major slicing 的 output layout — 還沒被 TRT 11.4 的 Myelin 完整覆蓋。

**預測**：
- TRT 11.5 / 11.6 大機率會把 dynamic quantize fusion 補上（這條最 low-hanging，Pattern 很清楚）
- TRT 12.x 才會把 Q/K/V projection 的 head-major epilogue + scale layout 這條大活幹完
- 期間 ModelOpt 可能會先用 PTQ 的 static quantize 幫使用者繞過 dynamic quantize 這條最痛的（但 reranker 這類 encoder 對靜態 PTQ 很敏感）

這整個 cycle 就是為什麼大型 inference 公司（Perplexity / Groq / Together）都在內部維持自己的 patch set — compiler debt 是 TRT 這類 AOT closed-source compiler 的結構性成本。

## 5. 這就是 Triton 這個對手的切入點（今天剛好開 DevCon）

今天（2026-10-19），**Triton Developer Conference** 在聖荷西會議中心開幕，就在 PTC 的前一天同場地。免費、含早餐午餐 happy hour、NVIDIA 的工程師同時參加。這個時間點很有意思 — 剛好是 TRT 11.4 的 NVFP4 fusion 問題被公開點名、而 Triton 把自己定位成「給你 user-controlled fusion」的 compiler 時。

Triton 的設計哲學跟 TRT 這條 AOT black-box compiler 剛好對立：

| 面向 | TensorRT | Triton |
|---|---|---|
| Fusion 由誰決定 | compiler 的 pattern matcher | 使用者的 block-level IR |
| 新 precision level 支援 | 跟著 release note 等 | 使用者可以自己寫 scale layout |
| Epilogue 控制 | 間接（看 fusion rule） | 直接（寫在 kernel body） |
| 整圖優化 | 強 | 弱（kernel-level） |
| 新硬體 enablement | 高 debt | 低 debt（但要重寫 kernel） |

對 bge-reranker-v2-m3 這個 case 來說，**Triton 使用者可以直接寫一個 attention block 的 kernel，把 Q/K/V split / transpose / FP4 quantize 全塞進 projection 的 epilogue**，繞過 Myelin 現在這個 fusion 缺口。代價是你得自己寫（而且是 Blackwell SM120 的 ldmatrix / wgmma / tcgen05 這些新指令）。

這其實解釋了為什麼 2025-2026 這一年 Meta / OpenAI / Together / Modal 等大戶的 production stack 愈來愈倒向 Triton + torch.compile + Inductor 這條路：**對他們來說，手寫 kernel 的工程成本已經小於等 TRT 補完 compiler debt 的機會成本。**

對正在往 compiler 布局的工程師，這條分叉決定了未來一到兩年的題庫方向。TRT 這條路考的是 pattern matcher、整圖 schedule、Myelin fusion rule。Triton 這條路考的是 block-level IR、Tensor Core 的 WGMMA / TCGEN05 直接呼叫、per-warp scheduling。兩條都是 compiler 工作，但技術棧截然不同。

## 6. 給面試用的三個 talking points

把上面這些壓成三個 interview-ready 的 talking points：

### Talking point 1：為什麼 single-op benchmark 不可信

> 「2026-10 有一個 Issue #4880 很好的例子 — bge-reranker-v2-m3 在 Blackwell 上，FP4 GEMM 比 FP8 快 1.7-2.0 倍，但 end-to-end 慢 24%。瓶頸是 Myelin compiler 在 FP4 路徑的三個 fusion 缺口：Q/K/V slicing epilogue、dynamic quantize fusion、LayerNorm output quantize。這告訴我們 Tensor Core 的理論 TFLOPS 跟實際 throughput 之間最大的差距，往往不在 Tensor Core 本身，而在它周圍的 epilogue 和 memory layout。」

這一段展示：
- 讀 release notes 跟 issue tracker 的習慣
- engine inspector / nsys 這類 profiling 工具的熟悉度
- 「硬體勝利能不能被 compiler 兌現」這個系統視角

### Talking point 2：compiler debt 的生命週期

> 「每個新 precision level 進入 TRT 都會有一個 1-2 年的 compiler debt 窗口 — INT8 從 2017 的硬體到 2019 才完整 fuse，FP8 從 2022 的 Hopper 到 2024 的 TRT 10；NVFP4 從 2025 的 Blackwell 到現在的 TRT 11.4 還在補。這個 cycle 的瓶頸永遠是 epilogue fusion 的 pattern coverage — 新 precision 的 output layout、per-block scale、動態量化 reduction 這些 pattern 都要重新寫進 fuser。」

這一段展示：
- 歷史趨勢的掌握
- 對 compiler internals 的理解不是「學過一次」而是「追過幾個版本」
- 預測性（下一個 TRT 版本會補哪一塊）

### Talking point 3：AOT closed-source 跟 user-controlled JIT 的 trade-off

> 「這就是為什麼大型 production stack 愈來愈倒向 Triton + Inductor 這條路 — 手寫 kernel 的成本正在小於等 TRT 補 compiler debt 的機會成本。TRT 的優勢是整圖優化跟 build-time 常數折疊，Triton 的優勢是遇到新 precision / 新硬體時可以當場把 epilogue 寫對。對 inference 公司的 compiler 工程師來說，這兩條路的技術棧很不一樣 — 一邊是 pattern matcher 和整圖 schedule，一邊是 block-level IR 和 warp-level scheduling。」

這一段展示：
- 工具選型的策略思考
- 不是「哪個比較好」而是「什麼情況下該選哪個」
- compiler 工程師不只是寫 pass，還要理解自己這個 compiler 在生態系的位置

## 7. 這對 Nova 讀者的實用面

如果你正在：

- **跑 TRT 做 production inference**：先別急著把所有模型切 NVFP4。encoder-only 的 reranker 和 embedding 模型目前 fusion debt 最嚴重；decoder-only LLM 因為 attention + FFN 的結構單純、Myelin fusion 覆蓋更完整，NVFP4 的 end-to-end 收益反而比較穩。等 TRT 11.5-11.6 再回頭評估 encoder 類。
- **寫 Triton kernel**：Blackwell SM120 的 FP4 epilogue 現在是最可以展現價值的位置 — TRT 補不過來的地方，就是你能幹的地方。specifically 看一下 `tl.dot` 的 FP4 output + per-block scale 怎麼跟後面的 GELU/LN/quantize fuse 到同一個 kernel。
- **做 LiDAR / 感知 pipeline（Adam 的 case）**：這個 case 其實跟 encoder-only reranker 高度相似 — PointNet / spconv backbone + Transformer-based detection head 的結構，fusion debt 的模式會非常像。做 benchmark 時務必拿 engine inspector 看清楚每個 kernel，不要只看 GEMM 的 TFLOPS。
- **準備 compiler 相關面試**：這一篇的四張表、三個 fusion 缺口、compiler debt cycle 是面試 Nvidia / Modular / OctoAI / xAI compiler team 的現成素材。issue 連結就是你 due diligence 的證據。

## 8. 一句話收尾

Blackwell 的 FP4 Tensor Core 把算術往前推了 2 倍，但 compiler 層的 fusion rule set 還沒跟上。這 ~73ms 的 gap 不是硬體問題，是 **compiler debt** — 一個每次新 precision level 進來都要重走一遍的 1-2 年週期。對 compiler path 的工程師來說，這個 gap 就是這個領域為什麼有工作可做。

今天去 Triton Developer Conference 現場的人，或多或少都在賭這個 debt 要花多久才會被 TRT 補上，以及這段時間裡 user-controlled 的 Triton 能不能把這些機會握住。這個賭盤不會太快收。

---

## 來源

- [TensorRT Issue #4880 — NVFP4 encoder slower than FP8 with TensorRT 11.4 on RTX PRO 6000 Blackwell (2026-10-15)](https://github.com/NVIDIA/TensorRT/issues/4880)
- [TensorRT 11.2.1 Release Notes — NVFP4 diffusion transformer NaN fix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.2.1.html)
- [TensorRT 11.3.0 Release Notes — FP4 double quantization refit](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.3.0.html)
- [NVfp4 Explained: Blackwell 4-Bit Precision 2026 — packet.ai](https://packet.ai/blog/nvfp4)
- [NVIDIA TensorRT Model Optimizer (ModelOpt): FP8, INT4, and FP4 Quantization Guide (2026) — spheron.network](https://www.spheron.network/blog/tensorrt-model-optimizer-modelopt-quantization-guide/)
- [FP4 Quantization on Blackwell GPUs: Throughput, Cost — spheron.network](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/)
- [Quantization (INT8 / FP8 / FP4) — Torch-TensorRT](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html)
- [Triton Developer Conference 2026 — San Jose Convention Center, 2026-10-19](https://tritonconference2026.splashthat.com/)
- [triton-lang/triton on GitHub](https://github.com/triton-lang/triton)
- [LLVM 23.1.3 Release — 2026-10-06](https://en.wikipedia.org/wiki/LLVM)
