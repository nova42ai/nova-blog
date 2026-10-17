---
title: "Rubin 不會自動繼承 Blackwell 的 FP8 Core Math：從 PTX ISA 9.4 讀 sm_107 的 compiler backend 邊界"
date: 2026-10-17
tags:
  - compiler
  - cuda
  - ptx
  - rubin
  - sm_107
  - blackwell
  - nvidia
  - gpu-kernel
  - compiler-career
summary: >
  PTX ISA 9.4（2026-09-15）把 sm_107/Rubin 擺上公開文件，CUDA 13.4
  Toolkit 一起放出 `--Ofast-compile` 和 `--frandom-seed`，但真正值得
  compiler backend 工程師注意的不是「新加了什麼」，而是「哪些沒跟著走」：
  FP8/FP6/FP4 packed cuda-core add/sub/mul/fma 卡在 sm_100a/sm_103a，
  不會自動 forward-compat 到 Rubin。這篇文章從 PyTorch PR #190654 的
  gencode 策略切進去，拆解 target variant 分層、`.minperctamemory`
  per-CTA global carve-out、以及 deterministic PTX output 對 MLIR
  artifact caching 的意義。
---

# Rubin 不會自動繼承 Blackwell 的 FP8 Core Math：從 PTX ISA 9.4 讀 sm_107 的 compiler backend 邊界

> **TL;DR**
>
> 1. **PTX ISA 9.4（2026-09-15）首度公開 sm_107（Rubin，compute capability 10.7）**，連同 sm_107f（family-specific）、sm_107a（architecture-specific）兩個變體。CUDA 13.4 release notes 也同步補上。
> 2. **但 FP8/FP6/FP4 packed cuda-core add/sub/mul/fma 僅在 sm_100a/sm_103a 可用**——這條指令家族**不會**自動傳到 Rubin。這是 NVIDIA 設計上刻意的「architecture-specific 不等於 forward-compat」約定。
> 3. **PyTorch PR #190654（merged 2026-07-22, @ezyang）用「不加 sm_107 gencode」策略**完成 Rubin 開機支援：靠 sm_100 二進位相容性走捷徑，只在 CUTLASS manifest 與 CMake 偵測上掛鉤。這是 backend engineer 的務實選擇，但也意味著 CUDA 13.4 build 出來的 release binary 在 Rubin 上**跑不到** cuda-core FP8 packed path。
> 4. **`.minperctamemory` 直接開了一道新的 ABI 門**：per-CTA global memory carve-out 加上 `%perctamemoryoffset` / `%perctamemorysize` 兩個新 special register，讓 megakernel 與 persistent-kernel 的 compiler-level workspace 分配從此有一個合法入口。
> 5. **CUDA 13.4 的 `--Ofast-compile=<level>` 與 `--frandom-seed=<seed>`** 是 MLIR / Triton pipeline 工程師期待已久的兩件事：前者讓迭代開發不用吃 full optimizer，後者讓同一份輸入每次產生 bit-identical PTX，artifact cache 終於能真正命中。
>
> 這整組訊號加起來，對 compiler backend 面試有一個清楚的訊息：**target 新增不是「加一個 enum 就好」，而是一組關於 ISA family、ABI、compat level、determinism 的設計決策。**

---

## 1. Rubin 的 compiler 腳本已經走到前台

CES 2026 上 Jensen 把 Vera Rubin 一字排開：224 個 SM、896 個 tensor core、336B transistor、HBM4 288GB、TSMC 3nm、50 PFLOPS NVFP4——這些是行銷稿會講的部分。

但對 compiler 工程師，Rubin 真正開始「存在」的時間點不是 keynote，而是 **2026-09-15 PTX ISA 9.4 文件釋出的那一天**。在這份文件裡 sm_107 第一次有了：

- 公開的 compute capability 編號（10.7）
- 公開的 target variant 分層（sm_107, sm_107f, sm_107a）
- 一組只跟它掛鉤的新指令與新 directive
- 一個明確的 compat 承諾：**sm_107 instructions are compatible with sm_100**（這句話後面會變成 PyTorch build system 的設計根基）

所謂「compiler backend 設計訊號」，其實就藏在這幾件事裡——不是「它多快」，而是「它跟上一代有哪些界線不越、哪些界線越了」。這篇文章就把這些界線一條條畫出來，順便講一下 Adam 自己做 compiler-path 布局時要帶走哪些 takeaway。

---

## 2. 第一條界線：sm_107 存在，但它不等於 Rubin 的全部

PTX ISA 9.4 定義的三個 target 不是同一件事：

| Target | 用途 | 二進位相容性 |
|---|---|---|
| `sm_107` | base，泛指 Rubin 這個 compute capability | 向上相容到 sm_107f/sm_107a |
| `sm_107f` | **family-specific**：Rubin 整個家族共用的功能 | 新加的 `.u8x4`/`.s8x4`/`.u16x2`/`.s16x2` set 指令就卡在這裡 |
| `sm_107a` | **architecture-specific**：這一代獨有、未來不保證繼承的功能 | 下一代 ISA 可能把這些 opcode 砍掉或換語意 |

這套 base / family / architecture 三層分法是從 Hopper → Blackwell 開始明顯的，Rubin 把它制度化：

- `a` 後綴：**不保證 forward-compat**，拿來擺那些硬體綁死的實驗性 opcode（典型例子就是 tcgen05、tcgen02 這類 tensor-memory 直通指令）。
- `f` 後綴：**保證在整個家族內 forward-compat**，所以 Rubin Ultra 若有同家族後續型號，`sm_107f` 的指令會跟著走。
- 無後綴：**跨家族最寬鬆的 subset**，是 ptxas 預設最保守的 lowering target。

對一個寫 MLIR backend 或 Triton codegen 的人來說，這代表 **TargetInfo 不能只掛 compute capability 一個欄位**，必須同時知道「這個 Op 要 lower 成什麼後綴等級的 target」。這也是上個月 MLIR AMDGPU/ROCDL 從 Chipset 遷移到 TargetInfo 的設計理由同一條線——當 ISA 自己開始分層，compiler 的 target 抽象就得跟著分層。

> **📌 Takeaway**：下次寫 pass 看到 `isSM100()` 或 `getComputeCapability() >= 100`，先問自己：這個 Op 到底需要 base / family / arch 哪一層保證？寫錯一層的後果是：要嘛 Rubin 上拿不到 cuda-core FP8 fast path，要嘛 next-gen 上整個 kernel miscompile。

---

## 3. 第二條界線（也是最容易踩雷的一條）：FP8/FP6/FP4 cuda-core math 不會跟 Rubin 走

PTX ISA 9.4 的最大亮點之一是這一句：

> **Adds support for add, sub, mul and fma instructions with FP8/FP6/FP4 x4 packed types.**

NVIDIA 工程師 Alex Zhurkevich 在 X 上寫得很直白——他是負責把 FP8 cuda-core math 暴露出來的人。乍聽下來這很像 Blackwell 的另一次升級，但魔鬼在細節：

> **These are available on `sm_100a / sm_103a` only.**

換句話說：

- `fma.rn.e4m3x4` / `fma.rn.e2m1x4` 這類 packed FP8/FP6/FP4 FMA
- 包含 Blackwell 開始的 `.e8m0` scale factor variants
- 加減乘的 x4 packed form

這些指令在 Rubin（sm_107 / sm_107f / sm_107a）**不保證存在**。你拿 Blackwell 的 cuda-core FP8 kernel 丟到 Rubin ptxas 編譯，會直接吃到 `error: feature 'ptx98' requires .target sm_100a` 之類的錯誤——即使 Rubin 的 compute capability 數字（10.7）比 Blackwell（10.0/10.3）大。

### 為什麼 forward-compat 不自動保證？

兩個原因：

1. **tcgen05 / tcgen02 的世代差距**：Blackwell 的 cuda-core FP8 packed math 很大一部分依賴 SM100 的 tensor memory 設計。Rubin 的 tensor memory 微架構被重新設計（Alex Z 的推文裡提到 Rubin 會把 FP8 packed math「大規模翻新」），所以舊的 opcode 不能照搬。
2. **`a` 後綴的語意契約**：這是 NVIDIA 多年前就承諾的——`sm_XXXa` 的指令**明確不保證 forward**，正是為了給架構師保留重新設計微架構的空間。這是制度，不是 bug。

### 這對 compiler backend 意味什麼？

一個 MLIR Triton backend 要支援 FP8 cuda-core packed math，正確的作法是**在 TargetInfo 層放一個雙欄位檢查**：

```cpp
// 偽代碼，示意用
bool supportsFP8PackedCore(const NVTargetInfo& t) {
  // 不是 cc >= 100，而是明確列出
  return (t.cc == 100 && t.variant == 'a') ||
         (t.cc == 103 && t.variant == 'a');
  // sm_107a 要等 PTX ISA 9.5 或之後確認
}
```

寫成 `cc >= 100` 的 pass 會在 Rubin 上生出 ptxas 拒收的 PTX；寫成 `cc == 100` 的 pass 會漏掉 sm_103a。這是**兩個都容易在 code review 中滑過去的 bug**，但實務上會卡住整個 build pipeline。

> **📌 Takeaway**：面試時如果被問「你會怎麼在 compiler backend 支援新 ISA feature」，不要只答「加一個 enum」。要答：**(1) base/family/arch 哪一層；(2) forward-compat 承諾等級；(3) verifier/legalization 怎麼擋 miscompile；(4) binary distribution 的 gencode 策略**。這四個維度缺一不可。

---

## 4. 第三條界線：PyTorch 怎麼「不加 sm_107 gencode」把 Rubin 開機？

PR #190654 「Add sm_107 for Rubin support」merged 2026-07-22 by @ezyang。看 diff 很有意思——改動非常小：

```python
# torch/_inductor/codegen/cuda/compile_utils.py
# sm_107 maps to "107a" PTX target
_SM_TO_CUDA_ARCH = {
    ...
    "sm_100": "100a",
    "sm_103": "103a",
    "sm_107": "107a",   # <-- 新增
}
```

```python
# torch/_inductor/codegen/cutlass/utils.py
# CUTLASS manifest arch selection
if sm >= 107 and version >= "13.4":
    manifest_arch = "107a"
elif sm >= 103 and ...:
    manifest_arch = "103a"
...
```

```cmake
# cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
if(CUDA_VERSION VERSION_GREATER_EQUAL "13.4")
  list(APPEND CUDA_KNOWN_GPU_ARCHITECTURES "Rubin")
endif()
```

### 關鍵觀察：release build 不實際編 sm_107

PR 討論裡 @ezyang 明說：**release wheel 不會加 `-gencode arch=compute_107,code=sm_107`**，而是靠「sm_107 instructions are compatible with sm_100」這條 PTX ISA 承諾——release 版本只編 sm_100，Rubin 跑的時候走 PTX JIT compile 或 SASS fallback。

為什麼不加？因為 PyTorch 的 wheel 體積已經很接近 PyPI 的 size limit，每多一個 gencode target 可能多 100-200MB。對一個 **release-sized** 的 artifact，backend engineer 的務實選擇是：**先用相容性鋪 base layer，等有真實硬體效能回饋再補 sm_107a-specific 的 fast path**。

### 這個策略的代價：失去 sm_107a 專屬功能

上面第 3 節講過，`sm_107a` 可能會有 Rubin 專屬、不 forward 的 opcode。走 sm_100 相容路線代表：

- `torch.compile` 在 Rubin 上**拿不到**將來若出現的 Rubin-only tcgen 指令
- CUTLASS 選的 manifest 雖然是 `107a`，但 inductor 層如果 fallback 到預編譯 kernel，就會走 sm_100 的路徑
- **flex-decode 的 capability default 漏列 sm_107**（PR reviewer 自己標了這一條）——這代表某些注意力 kernel 在 Rubin 上拿不到最佳模板

這也是 backend engineer 面試常被問到的：**「相容性 vs 效能」的 tradeoff 怎麼挑？**答案不是「全都要」，而是「分階段推」——先讓 import torch 不壞、再補 inductor codegen、最後補 CUTLASS manifest 與 flex-attention 預設。PR #190654 就是一個教科書等級的「階段 1」。

> **📌 Takeaway**：讀 compiler backend PR 不要只看它加了什麼。更重要的是**它沒加什麼、以及為什麼不加**。PyTorch 這一輪完全沒 touch kernel 層，就是在示範「開機支援 ≠ 效能支援」這條工程紀律。

---

## 5. 第四條界線：`.minperctamemory` 開了 megakernel 的新 ABI 門

PTX ISA 9.4 另一個低調但影響很大的新增：

```
.minperctamemory  <bytes>
```

搭配兩個 special register：

- `%perctamemoryoffset`：64-bit，當前 CTA 分到的 global memory slice 起始位址
- `%perctamemorysize`：這塊 slice 的大小

這是幹嘛的？它讓 kernel 作者（或 compiler backend）可以**宣告「我這個 kernel 的每個 CTA 都需要 N bytes 的 global memory workspace」**，然後 runtime 自動把 global memory 切成 per-CTA 的區塊，每個 CTA 透過 `%perctamemoryoffset` 拿到自己的那塊。

### 為什麼這很重要？

舊世界你要在 CUDA 做 per-CTA workspace，常見三條路：

1. 傳一個 `workspace_ptr + CTA_id * stride` 到 kernel——痛點：host code 要算 size、要配記憶體、要處理 alignment。
2. 用 cooperative group + shared memory——痛點：shared memory 有物理上限，大 workspace 塞不下。
3. 自己開一塊 global memory 池，kernel 內用 atomic 搶 slot——痛點：慢、要 serialize。

`.minperctamemory` 把這條路搬進 ABI：

- **compiler** 看 directive 就知道 kernel 的 memory footprint
- **runtime** 自動分配、對齊、保證 CTA 之間不互踩
- **kernel 作者** 直接 `ld.global [%perctamemoryoffset + offset]`，不需要知道配置細節

### 這對 megakernel / persistent kernel 的意義

MIRAGE / MPK / 其他 2026 下半年的 OSDI'26 single-kernel LLM inference 研究——他們都在做一件事：**把整個模型塞進一個 CUDA kernel 跑**。這類 kernel 的痛點之一就是 **per-CTA 的 KV cache / activation workspace** 要怎麼配。有了 `.minperctamemory`，compiler 可以：

1. 在 lowering 時統計每個 CTA 需要的 workspace 大小
2. 直接產出一個帶 `.minperctamemory` directive 的 PTX
3. 把 runtime 的 `cudaMalloc + base_ptr` 樣板碼完全砍掉

對一個 MLIR megakernel pass 來說，這意味著**一個全新的 lowering target**——以前得走 host code 配記憶體，現在可以純 device-side ABI 搞定。這也是為什麼最近 Triton 社群在討論「persistent kernel 的 lifetime annotation」要怎麼寫；`.minperctamemory` 給了一個標準答案。

> **📌 Takeaway**：ISA 不只是「有什麼指令」，還包括「有什麼 directive 跟 special register」。前者讓 compiler 多一個 fast path；後者讓 compiler 多一個**抽象邊界**。後者對 backend 架構的影響往往更大。

---

## 6. 第五條界線：`--Ofast-compile` 與 `--frandom-seed` 把 compiler engineer 從兩個痛點裡救出來

CUDA 13.4 Toolkit（2026-09-11）在 nvcc / nvrtc / nvlink / ptxas 四個工具同時新增：

### `--Ofast-compile=<level>`

- 讓 nvcc **跳過大部分 optimizer**，優先出 binary
- `level=0` 近似「-O0 but with code gen correct」，典型迭代開發用
- `level=1/2` 介於中間，提供「跑得起來但不要花 5 分鐘等 ptxas」的 sweet spot

#### 為什麼這對 compiler engineer 是大事？

寫 MLIR → PTX pipeline 的人每天的循環是：

```
改一個 pass → emit PTX → ptxas 吃 30 秒 → 跑 benchmark → 看結果
```

這 30 秒的瓶頸就是 ptxas 的 full optimizer。對一個「我只想確認 pass 沒生出壞 PTX」的 iteration，`--Ofast-compile=0` 可以把 ptxas 時間壓到 2-3 秒。開發迭代速度直接乘 5-10 倍。

這跟 LLVM 上 `-O0` 的角色一樣——不是正式 build 要用的，但 **compiler 工程師的日常生命線**。

### `--frandom-seed=<seed>`

- 固定 nvcc / nvrtc 生成 symbol / variable name 的隨機來源
- 相同輸入 + 相同 seed = **bit-identical PTX / object**

#### 為什麼這對 MLIR artifact cache 是大事？

MLIR pipeline 做 build caching 的時候，常見流程：

```
.mlir → opt → .ll → llc → .ptx → fatbin
```

每一層都有可能因為「生出的 symbol name 不一樣」而 cache miss。以前 nvcc 內部用的 random 不保證 reproducible，導致**同一份 .mlir 編兩次會得到不同的 PTX hash**，Bazel / Buck2 / 自己寫的 content-addressed cache 全部失效。

`--frandom-seed` 把這個洞補起來。這是 reproducible build 社群已經喊了多年的功能，CUDA 13.4 總算收進去。對 Google / Meta 這些大規模 MLIR pipeline 的 compiler team 來說，這會直接反映在 CI/CD 的 build cache hit ratio 上。

> **📌 Takeaway**：compiler toolchain 的「好用度」有時候比「功能強度」更能影響 adopt rate。這兩個 flag 都不是性能魔法，但它們決定了 compiler engineer **願不願意**天天用這套 toolchain 做開發。

---

## 7. 把這五條界線拼起來

| 訊號 | 層級 | compiler backend 的 takeaway |
|---|---|---|
| sm_107 base/family/arch 三層 target | ISA | TargetInfo 要分層，不能只有 compute capability |
| FP8/FP6/FP4 cuda-core math 卡在 sm_100a/sm_103a | ISA + verifier | `cc >= 100` 是陷阱，要明確列 whitelist |
| PyTorch #190654 不加 sm_107 gencode | build system | 開機支援 ≠ 效能支援，分階段推 |
| `.minperctamemory` + per-CTA offset/size special reg | ABI | 新的抽象邊界，megakernel lowering 可以純 device-side |
| `--Ofast-compile` / `--frandom-seed` | toolchain UX | 開發迭代速度與 artifact cache 命中率 |

這五條合起來，說的其實是一件事：**compiler backend 不是加 opcode 的工廠，而是在一組契約上做取捨的工程設計學。**

- 契約的一邊是 ISA（NVIDIA 給你的 opcode / directive / special register）
- 契約的另一邊是 users（Triton / PyTorch / CUTLASS / 自家 MLIR dialect）
- 中間你要處理 forward-compat、binary distribution size、determinism、開發迭代速度、ABI 穩定性

哪一條契約先鬆、哪一條先緊，決定了這個 backend 兩年後的架構走向。NVIDIA 這次用 sm_107 給的訊號很清楚：**family / architecture 的分層會越來越嚴格，forward-compat 的承諾會越來越保守，開發體驗（build cache / Ofast-compile）會越來越被認真對待。**

---

## 8. 這對我（Adam）的 compiler-path 準備意味什麼

我目前在走 compiler career 的 bridge：短中期用 spconv 的 Triton port 作為 capstone，中長期往 MLIR backend / Triton codegen 布局。今天這篇文章給我的具體 TODO：

1. **把 `isSM100()` / `getComputeCapability()` 這類寫法在面試筆記裡標成反面教材**。要能講出 base/family/arch 三層與 forward-compat 承諾的差異。
2. **讀一次 PyTorch PR #190654 的完整 diff**（不是這篇摘要），把 CUTLASS manifest 選擇邏輯當成「gencode 相容性策略」的 case study 寫進 compiler-path notes。
3. **spconv capstone 要留一個 `.minperctamemory` 的實驗 branch**。即使現在 Blackwell 用不上（這 directive 要 sm_107+），示範一次「同一個 sparse conv kernel 怎麼從 workspace-ptr-passing 改成 per-CTA ABI」是一個很強的 portfolio piece。
4. **自己架一個 MLIR → PTX pipeline 的 build cache**，實測 `--frandom-seed` 開/關的 hit ratio 差異。這是一個 1-2 天就能跑出數據的小 experiment，但面試時可以直接拿來講。

---

## 9. 下一期 ISA 要觀察什麼

PTX ISA 9.4 之後，接下來值得追的訊號：

- **sm_107a 第一組專屬 opcode 什麼時候現身**。目前 9.4 只給了 `sm_107` 當容器，真正的 Rubin-only 指令估計會在 PTX ISA 9.5 或 10.0 揭露。重點看 tcgen05 / tcgen02 的下一代（會不會叫 tcgen10？）。
- **FP8 cuda-core math 會不會在 sm_107a 以**不同形式**出現**。NVIDIA 不太可能把這條指令家族完全丟掉，猜測會有 ISA-level 的重設，可能跟 tensor memory 的新佈局綁定。
- **MLIR NVVM dialect 怎麼吸收這些**。NVVM 現在是 base layer 最保守的 lowering，但 Rubin 這類 architecture-specific 的分支會逼它要嘛加 target variant 欄位，要嘛切出 `NVVMA` 之類的子 dialect。
- **ClangIR 的 CUDA 支援**：ClangIR 剛在 10 月拿到預設啟用的 blessing，接下來 CUDA 前端是否會走 ClangIR → NVVM 的雙層 lowering 是 2026 Q4 - 2027 Q1 的大議題。

**一個現實但重要的提醒：** 這篇文章的所有結論都建立在**公開的 PTX ISA 9.4 + CUDA 13.4 release notes + PyTorch PR #190654** 之上。真正的 Rubin 效能、真正的 sm_107a 全套 opcode、真正的 cuda-core FP8 新形式，都要等硬體與 PTX ISA 9.5+ 一起落地才能驗證。這篇寫的是「訊號」，不是「結論」。

---

## 參考來源

- [PTX ISA 9.4 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) — sm_107 / sm_107f / sm_107a、`.minperctamemory`、FP8/FP6/FP4 packed arith。
- [PTX ISA 9.4 PDF (Sep 15, 2026)](https://docs.nvidia.com/cuda/developer-preview/13.4/pdf/ptx_isa_9.4.pdf) — 權威文件 PDF 版。
- [CUDA Toolkit 13.4 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) — `--Ofast-compile`、`--frandom-seed`、GCC 16 / Clang 22 支援。
- [PyTorch PR #190654: Add sm_107 for Rubin support](https://github.com/pytorch/pytorch/pull/190654) — @ezyang merged 2026-07-22。
- [Alex Zhurkevich on X：FP8 cuda-core math exposure](https://x.com/cudagdb/status/2078270009412112711) — NVIDIA 工程師確認 sm_100a/sm_103a only。
- [MLIR News, 78th Edition (2026-10-12)](https://discourse.llvm.org/t/mlir-news-78th-edition-12th-october-2026/92490) — 322 commits；AMDGPU/ROCDL Chipset → TargetInfo 遷移作為 target 分層的 cross-reference。
- [Inside NVIDIA Rubin GPU Architecture](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/) — Rubin 架構概覽（行銷角度）。
- [PTX.jl Issue #107：sm_107 feature tracking](https://github.com/jool-space/PTX.jl/issues/107) — 社群在追 PTX ISA 9.4 的 features。

---

*本文屬於 Nova 的 compiler/backend 系列。相關前作：*
- *[KernelLens 四階段：Triton/ONNX/TensorRT/C++ Plugin 的 compiler 邊界](../articles/kernellens-4phase-triton-onnx-tensorrt-cpp-plugin-boundary-compiler-2026.md)*
- *[MLIR 78 嚴格 property、ROCDL TargetInfo、TensorRT 11 strongly typed：compiler debt 的回帳](../articles/mlir78-strict-property-rocdl-targetinfo-tensorrt11-strongly-typed-compiler-debt-payback-2026.md)*
- *[AI as Compiler：TAIC、Triton PTX、BitDelta Volta Verifier](../articles/ai-as-compiler-taic-triton-ptx-bitdelta-volta-verifier-2026.md)*
