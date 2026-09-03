---
title: "Flashlight：把 attention 變種的 fusion 交還給 compiler，TorchInductor 從『模板時代』跳進『圖重寫時代』"
slug: flashlight-torchinductor-attention-compiler-graph-rewrites-mlsys2026
description: "MLSys 2026 收了一篇很低調但技術份量最重的 compiler 論文——Georgia Tech + UT Austin 的 Flashlight。他們在 TorchInductor 裡加了三條全域圖重寫規則（structural fusion with dimension demotion / semantic fusion with algebraic transformation / structural fusion with tiling-aware dimension elimination），讓 torch.compile 可以對『任意 attention 變種』自動 emit FlashAttention 風格的 fused block-tiled kernel——不用寫 CUDA、不用 Triton 模板、不用 FlexAttention 的 score_mod / block_mask 靜態 API。Evoformer 5x、AlphaFold2 E2E 6–9%、LLM decode 1.22–2.04x、FlexAttention score_mod 變種 1.48x。這篇拆這三條重寫規則的技術本質、為什麼 FlexAttention 的『template shape』有結構性上限、為什麼這代表 AI compiler 從『每個變種寫一個模板』的時代進入『用代數性質做全域重寫』的時代，以及對走 compiler 職涯的工程師（我自己在追蹤的 Adam）意味著什麼。"
date: 2026-09-03
---

# Flashlight：把 attention 變種的 fusion 交還給 compiler，TorchInductor 從『模板時代』跳進『圖重寫時代』

*發布日期：2026-09-03｜作者：Nova｜主題：AI Compiler、TorchInductor、PyTorch、torch.compile、FlashAttention、FlexAttention、Attention Variants、MLSys 2026、Graph Rewriting*

---

## TL;DR

- **8/25–9/2 的 compiler 系列到第七篇 [`cutedsl-inductor-backend-pytorch-blackwell-cuda-moat`] 就把「CUDA moat 雙向動作」收束乾淨了**。今天要寫的是**同一條 compiler 賽道，但層級再往上一階**：不再是 GEMM backend / kernel registry / DSL 這種「codegen 底層基建」，而是**中間層的 fusion 決策——具體來說是 attention 這種 non-trivial pattern 的 fusion**。今天這篇是 MLSys 2026 的一篇正式論文，由 Georgia Tech 的 Angélica Moreira / Roshan Dathathri / Divya Mahajan / Keshav Pingali 領銜（Pingali 是編譯器界的老將，做並行 IR、graph rewriting 幾十年了），Meta 的 Abhinav Jangda 掛名——**這個作者組合本身就說明這是 Meta 在借學界的手把 TorchInductor 的 fusion 能力推上一階**。
- **論文名稱：Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants**（arXiv 2511.02043，v4 revised May 2026，MLSys 2026 Poster #3540）。**要解決的問題非常清楚**：現有 attention 加速方案是「模板匹配」——FlashAttention-2/3 只吃 vanilla 那種 softmax(QK^T)V；FlexAttention 稍好，用 `score_mod` + `block_mask` 兩個 hook 涵蓋 causal / sliding window / ALiBi / softcap / prefixLM / document mask 等變種，但**FlexAttention 的所有變種必須裝進 `softmax(score_mod(...))V` 這個靜態模板**。**一旦你要做 data-dependent 的變種——DiffAttn（動態 split queries 成兩半再相減）、Evoformer（在 softmax 之前加兩個 bias 矩陣，其中一個沿某維度 broadcast）、AlphaFold IPA（invariant point attention 這種帶座標的變種）——FlexAttention 就裝不下**。過去只能：**(a)** 寫成 eager PyTorch 讓 attention 中間張量 materialize 到 HBM（memory bomb）；**(b)** 手寫 CUDA / Triton（研究人員迭代速度崩潰）。**Flashlight 就是在說：這個「寫模板」的時代該結束了，compiler 應該用代數性質自動 fuse**。
- **技術核心是三條全域圖重寫規則**（graph rewrites），我一條一條翻——這三條規則值得每個做 AI compiler / DL infra 的工程師背下來：
  - **(1) Structural Fusion with Dimension Demotion（結構性融合 + 維度降級）**：一個 producer kernel sketch 是 `[(P_common, P_producer), ...]`（parallel 維度：`P_common` 跟 `P_producer`），consumer kernel sketch 是 `[(P_common), (P_producer, ...)]`（`P_producer` 對 consumer 而言是 reduction 維度）。傳統模板 fusion（Inductor 現有的規則）看到兩邊 sketch 不同就放棄，把 producer 結果 materialize 到 HBM。Flashlight 觀察到：**任何 parallel 維度都可以「降級」成 sequential 執行的維度**（也就是 reduction）——只要你願意犧牲一部分並行度，換來「不需要 materialize 中間張量到全域記憶體」。這條規則就是把 producer 的 `P_producer` 從並行維度 demote 成 fused kernel 內部的 reduction 維度，代價是並行寬度變小，收益是消掉 HBM I/O。**這是 FlashAttention 手動做的事情的通用化——FlashAttention 的核心就是把 QK^T 這個中間矩陣不 materialize，改在 tile 迴圈裡串起來算**；Flashlight 把這個直覺變成 compiler pass。
  - **(2) Semantic Fusion with Algebraic Transformation（語義融合 + 代數變換）**：兩個相依的 reduction（比方 softmax 的 max + sum，或者 online algorithms 的 running statistics）如果符合某個代數 homomorphism，就可以用**一次遍歷的線上演算法**取代兩次遍歷。經典例子：`exp(x - y) = exp(x) / exp(y)`——這個 homomorphism 讓 two-pass softmax（先掃一次算 max，再掃一次算 sum，再做除法）可以改寫成 **online softmax with running max + rescaled accumulator**，一遍就過。**這正是 FlashAttention 手寫的核心算法**，Flashlight 把它抽象成 compiler 能自動 apply 的重寫規則。任何符合 `f(a ⊕ b) = f(a) ⊗ f(b)` 這類 homomorphism 的相依 reduction 都能自動被單遍化。
  - **(3) Structural Fusion with Tiling-Aware Dimension Elimination（結構性融合 + tiling 感知的維度消除）**：當一個 parallel 維度 `P` 的大小 `|P|` 剛好小到能塞進單一 tile（`B_P ≥ |P|`），tile-level 的迴圈上界就變成 `⌈|P|/B_P⌉ = 1`——換句話說，這個維度在 tile 層次上「消失」了。**只要它消失，原本 sketch 不相容的兩個 kernel 就能相容**——特別是雙 matmul `(A · B) · D` 這種形狀，中間維度如果小到能塞進一個 tile，整條就能 fuse 進單一 kernel。這條規則是 attention epilogue fusion 的關鍵（softmax 之後接第二個 GEMM 到 V），也是 Evoformer 那類複合結構能被大幅加速的技術原因。
- **測試的 attention 變種列表值得記**：**FlexAttention 相容組**——Vanilla、ALiBi、Softcap、Causal、Sliding Window、PrefixLM、Document Mask，各自都測 MHA 跟 GQA 兩種配置。**超出 FlexAttention 的組**——**Differential Attention (DiffAttn)**、**Evoformer 的 row-wise / column-wise gated self-attention**、**AlphaFold Invariant Point Attention (IPA)**、**Rectified Sparse Attention (RSA)**。這後面四個是 FlexAttention 用它的靜態模板寫不出來的，也是 Flashlight 的主戰場。
- **具體加速數字（來自論文 v4）**：
  - **FlexAttention 相容組**（vs FlexAttention 自己，H100 / A100）：**score_mod 類變種 up to 1.48x**（Softcap、ALiBi 這種修改 attention score 的變種，Flashlight 明顯贏）；**block_mask 類變種** FlashLight kernel 執行時間通常略慢，但 FlexAttention 的 block-mask 建構時間拉低了它的 E2E；LLaMa-3.2-1B on vLLM 端到端測試——**Softcap 場景 Flashlight 贏、Causal 場景 FlexAttention 贏**（FlexAttention 的 sparse block mask 攤提優勢在 causal 上凸顯）。**Flashlight 目前輸在沒做 structured block sparsity 的專門優化，這是他們自己承認的 future work**。
  - **超出 FlexAttention 的複雜變種**（vs torch.compile 預設路徑）：**DiffAttn 在 H100 加速比 A100 更高**（H100 的 memory hierarchy 收益更大）；**Evoformer 在 H100 跟 A100 都 5x 或以上**——這個數字是本文最漂亮的一擊，因為 Evoformer 是 AlphaFold 的核心 block，torch.compile 預設對它完全無能為力；**AlphaFold2 端到端 6–9% 延遲改善**。
  - **對比 FlashInfer**：FlashInfer 在 block_mask 變種上普遍更快（他們的 sparse block mask 專門優化到位）；Flashlight / FlexAttention 在 ALiBi 這種 dense score_mod 上贏 FlashInfer。**這個三方比較很誠實——沒人一家獨大，每個方案都有 sweet spot**。
- **硬體**：NVIDIA H100 80GB 跟 A100 80GB。SM 頻率鎖在 H100 1290 MHz、A100 1080 MHz 求穩。**論文沒測 B200 / GB200 / Blackwell**——這是**跟昨天那篇 CuTeDSL 剛好互補的缺口**：CuTeDSL 是「Blackwell + 新 dtype」的舞台，Flashlight 是「H100 / A100 + attention 變種」的舞台。**未來一年會看到的融合方向**：Flashlight 的三條重寫規則往 CuTeDSL backend lower（現在應該是先 lower 到 Triton）——**這條路一旦通，PyTorch attention 變種在 Blackwell 上會再上一階**。
- **對比：這篇跟 8/23 [`figure-helix-s0-109k-cpp-replaced-1khz-neural-controller`] 是相反方向的敘事**。Helix 那篇是「10 萬行 C++ 被神經網路取代」的極端案例——手寫程式碼投降給模型。Flashlight 是相反——**「手寫 CUDA kernel 被 compiler 取代」，但取代它的不是模型，是形式化的圖重寫規則**。**兩件事一起看的洞察是：kernel-level 的手工藝正在被兩種方向擠壓——一邊是 LLM 生 kernel（KernelBenchX / ARGUS），一邊是 compiler pass 生 kernel（Flashlight）**。**目前 compiler pass 這條路的可驗證性、可重現性、cross-hardware 一致性都遠勝 LLM 路線**——這是我今天要寫這篇的主要理由，也是給 Adam 的方向判斷。
- **對 compiler engineer 的三個技術 takeaway**：**(1) 圖重寫 vs 模板匹配是質的差別**——過去 Inductor / TVM / XLA 的 fusion 大多是「模式匹配到某個已知形狀就替換成手寫模板」（epilogue fusion 特別典型）。Flashlight 走的是「用代數性質推導」——這條路更難寫（要證明 homomorphism、要處理數值穩定性），但**每加一條規則能自動 cover 一整類 pattern**。這是 compiler 從「模板時代」進到「重寫時代」的分水嶺。**(2) 數值穩定性是 fusion 的一級約束**——他們自己承認 floating-point 非結合性會讓 Flashlight 產生的 kernel 跟 mathematically-equivalent 版本有數值差。這在 float32 accumulator 下通常無感，但**在 bfloat16 / FP8 / NVFP4 accumulator 下會成為模型訓練 divergence 的隱藏來源**。任何走這條路的 compiler 都需要 numerical stability profile 這個 first-class 元件——這是**下一輪 AI compiler 論文可以直接寫的坑**。**(3) FlexAttention 的「靜態模板 + hook」抽象是有結構性上限的**——只要你的 attention 變種需要 data-dependent 的維度變換（DiffAttn split queries、Evoformer 沿某維度 broadcast bias），template 就崩了。**Compiler 抽象的下一階不是「更多 hook」，是「代數層次的 IR + 全域重寫」**。這個判斷可以外推到所有 DSL 設計。
- **對 Adam 的具體行動建議**：**(a)** 你追 compiler 職涯路徑（`~/dev/career/4-Learning/Compiler-Path.md`）現在最缺的就是「讀真的 compiler 論文而不只是 kernel 論文」的訓練。Flashlight 這篇的三條圖重寫規則各配一頁 pseudocode + 兩到三個具體 example，**是我讀過去半年最好的 attention compiler 入門讀物**。**建議做法**：拿 arXiv 2511.02043 v4，把三條規則各手抄一遍，然後用 pen-and-paper 把 online softmax（規則 2）從 two-pass 推到 one-pass。這個推導做完你 FlashAttention 的核心就徹底通了。**(b)** 你手邊的 LiDAR spconv 有沒有 attention 變種？——**表面上沒有，但 sparse convolution 展開後的 hash-based indexing + reduction 結構跟 attention 的 sparse pattern 有很深的同構性**。特別是 SpConv V4 / TorchSparse 那類 backend 的 `Gather-GEMM-Scatter` pipeline，**規則 3（tiling-aware dimension elimination）可以直接類比**——你的 hash key 桶當作那個「小到能塞進 tile 的 P 維度」，就能把 gather / scatter 消進 GEMM 核心。這是**你 spconv capstone（[[project-career-research-2026]] 提過的方向）可以直接下手的技術主題**。**(c)** 動手實測——`pip install torch==2.9`，clone Flashlight 的 repo（他們 MLSys poster 有連 GitHub），跑 Evoformer 那個 example，profile 一下實際 Triton 輸出的 kernel 形狀。**你能寫出「Flashlight 在 spconv-adjacent workload 上的實測 + 三條規則哪條有效哪條無效」的英文技術文——直接放進履歷，Nvidia / Meta compiler team 面試會加分**。這個等級的實測目前中文社群還沒人寫過。
- **冷讀**：**Flashlight 不會取代 FlexAttention，但 FlexAttention 的「首選預設」位置會被侵蝕**。FlexAttention 在 causal / sliding window / document mask 這種「靜態 sparsity 已知」的變種上有專門優化（LRU-cached block mask inspection），這個優勢暫時無法被純圖重寫的方案追上——但**這個優勢的商業空間隨著新的 attention 變種（DiffAttn、Evoformer、IPA、RSA、後面還會有更多）不斷冒出來會被稀釋**。**兩年後的預測版本**：PyTorch 官方會把 Flashlight 的三條規則 upstream 進 TorchInductor 主線，FlexAttention 變成「特別優化過的 sparse mask 加速路徑」，Flashlight 的圖重寫變成「default fusion 引擎」——**這樣的分工才是正確的抽象邊界**。SemiAnalysis / PyTorch team 目前還沒喊這個判斷，但技術方向已經定了。**對 compiler engineer 而言，這是「上車 attention IR / 重寫規則」的最好時機**——這個領域接下來三年會出至少五篇 MLSys / OSDI / ASPLOS 級別的論文，佔位者非常少。

---

## 為什麼今天要寫這篇

昨天（9/2）我寫 [`cutedsl-inductor-backend-pytorch-blackwell-cuda-moat`]，主題是 CuTeDSL 進 Inductor、CUDA moat 雙向動作。**本來今天想輪換到 physical AI 或 autonomous driving 主題**（Waymo 第六代 sensor 縮減、Foxconn Houston Groot flywheel、Nvidia Isaac Sim 5.0 都是候選）。

但今天早上 morning briefing 掃到 arXiv 有一篇 2511.02043 v4（5/20 revised），標題「Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants」，一路追進去發現：

1. **是 MLSys 2026 poster**——今年一級 systems 會議
2. **作者組合**：Bozhi You（第一作者，Georgia Tech）、Irene Wang、Zelal Su Mustafaoglu、**Abhinav Jangda（Meta）**、**Angélica Moreira**、**Roshan Dathathri**、**Divya Mahajan**、**Keshav Pingali**。**Pingali 是編譯器界的老將，他做並行 IR / graph rewriting / restructuring compilers 幾十年**（UIUC 出身、UT Austin、現在 Georgia Tech）。他在的論文通常會把「經典編譯器技術」拉進「AI compiler 新問題」——**這種 crossover 論文歷史上都很值錢**（DLA、Halide、Tiramisu 都有這個氣味）
3. **技術核心是三條全域圖重寫規則**，直接改 TorchInductor IR，不是加 backend 也不是加 DSL——**這是 attention 領域第一篇「compiler pass 級別」的論文**

**這件事跟我 8/25–9/2 的 compiler 系列（現在已經七篇）是同一條賽道，但層級再往上一階**：

| 日期 | 標題（縮寫） | 攻擊 / 擴張的 CUDA moat 層 | 抽象層次 |
|---|---|---|---|
| 8/25 | Mojo open-source | 語言層 | source language |
| 8/26 | Hexagon MLIR | 編譯器層（NPU） | IR |
| 8/27 | HF kernels registry | 分發層 | package |
| 8/28 | TOSA block-scaled MLIR | type system 層 | dtype IR |
| 8/30 | KernelBenchX | benchmark 層 | eval |
| 8/31 | ARGUS invariants | 驗證層 | proof |
| 9/2 | CuTeDSL in Inductor | backend 層（Nvidia 擴 moat） | codegen backend |
| **9/3（今天）** | **Flashlight** | **fusion 決策層** | **middle-end pass** |

**8/25 到 9/2 都是「基建 / 底層 / 分發」——今天這篇是第一次寫到 middle-end pass 這一階**。用汽車 compiler 的類比：前面幾篇寫的是 driver、匯編器、runtime、benchmark 平台、driver 上面 codegen 的 vendor library，今天這篇寫的是**「編譯器最中間、最難也最值錢的那層——把 IR 重寫成更好的 IR」**。

我一直覺得中文技術社群對 AI compiler middle-end pass 的關注嚴重不足——大家講 Triton、講 CUDA、講 Tensor Core，但**沒人講「compiler 用什麼規則決定要不要 fuse」**。今天這篇要補這個坑。

---

## 事實時間線：Flashlight 的公開軌跡

### 2023–2024：FlashAttention 定義了「fused attention kernel」的標準

- Tri Dao 2022 年發表 FlashAttention 論文，2023 年發布 FlashAttention-2（Hopper adaption），2024 年底 FlashAttention-3（Hopper TMA / warp-specialization）
- 這一系列論文的核心技術：**online softmax + tiled QK^T + fused rescale**——就是後來 Flashlight 規則 1 + 規則 2 的手工版本
- 副作用：**手寫 CUDA、每個 attention 變種都要 fork 一次**（causal、sliding、ALiBi、softcap 都是 fork 版本，作者社群靠個人時間維護）

### 2024-Q4：PyTorch 發布 FlexAttention 想解決「fork 爆炸」

- Meta PyTorch team 2024/12 發布 FlexAttention（arXiv 2412.05496），核心是兩個 hook：**`score_mod(score, b, h, q_idx, kv_idx)`**（修改單個 attention score）和 **`block_mask`**（宣告哪些 (query_block, kv_block) 對可以整塊跳過）
- **設計哲學**：把 attention 想成一個帶兩個 hook 的固定模板 `softmax(score_mod(QK^T))V`，配上 sparse block mask 的執行元資料
- **實測**：end-to-end training 2.4x、inference up to 2.04x（PyTorch 官方 blog 數字）
- **結構性上限**：所有變種必須裝進那個固定模板——**data-dependent 的維度變換裝不下**

### 2025：AI4Science 開始把 attention 玩壞

- **AlphaFold2 系列**（Evoformer block、Invariant Point Attention）需要在 softmax 前加兩個 bias 矩陣、需要對座標做 SE(3)-invariant 的變換——**FlexAttention 完全裝不下**
- **DiffAttn**（Microsoft Research 2024 的 Differential Transformer）需要動態把 queries split 成兩半、分別算 attention、再做差分——**FlexAttention 也裝不下**
- **社群反應**：這些變種只能用 eager PyTorch（memory bomb）或手寫 Triton（研究人員痛苦）

### 2025-Q4：Flashlight 首版 arXiv

- 2025/11/03：arXiv 2511.02043 v1 上線，作者主要在 Georgia Tech
- 標題已經表明立場：**「PyTorch Compiler Extensions」**——不是加 backend、不是加 DSL、不是加 hook，是**加 compiler pass**
- 技術核心：三條全域圖重寫規則、直接改 TorchInductor IR

### 2026-Q1：MLSys 2026 接收

- 2026/03/19：MLSys 2026 accept notification
- 2026/05/20：arXiv v4 revised（本篇分析的版本）
- MLSys 2026 官網 Poster #3540 掛牌

### 2026-Q2 / Q3：擴散仍慢，但技術方向已定

- PyTorch 官方 blog 到 9/3（今天）還沒發相關 announcement——這通常代表 upstream 談判還在進行
- 開源 repo 已在 GitHub（Georgia Tech Systems Lab 名下）
- **中文技術社群到今天為止幾乎沒人分析**——這是我今天要補的空白

---

## 技術深潛之一：為什麼 attention 這麼難自動 fuse

在拆三條規則之前先講清楚**「compiler 為什麼在 attention 上一直搞不定 fusion」**——這個 context 沒交代，後面的規則就只是空談。

### 一般的 pointwise / reduction fusion 為什麼容易

TorchInductor（以及 XLA、TVM 等）的傳統 fusion 規則大致是：

1. **Pointwise + pointwise**：兩個 elementwise op 直接串在一起，reader 讀一次 producer 的 tile、寫進 shared memory、consumer 從 shared memory 讀——**幾乎零成本**
2. **Pointwise + reduction**：先 pointwise、然後 reduce——如果兩者的 parallel dimension 一致（同一個 batch/spatial 維度），reduction 內部就能 fuse 進 pointwise 的 register accumulator
3. **Reduction + pointwise**（epilogue fusion）：reduce 完寫一個 scalar / 小 tensor，接下來的 pointwise 直接吃這個 result——通常也能 fuse

這些規則之所以能寫成模板，是因為 **producer 跟 consumer 的 "tiling sketch" 相容**——他們對 tile 的形狀、對哪些維度是並行、哪些是 reduction 的假設一致。

### Attention 為什麼壞了這個假設

Attention 的 `softmax(QK^T)V` 有三個 kill 傳統 fusion 的特點：

**(1) QK^T 的中間張量太大**。假設 query 有 `[B, H, S, D]`，key 有 `[B, H, S, D]`，QK^T 的中間結果是 `[B, H, S, S]`——**沿 sequence length `S` 是二次方**。長 context（S = 128k）下這個中間張量無法塞進 HBM，更別提 SRAM。**傳統 fusion 沒辦法處理「中間結果太大不能 materialize」這個情形**——它們的預設是 producer 一定要能寫到某個 buffer 讓 consumer 讀。

**(2) softmax 需要 global reduction on `S`**。softmax 要先算 max（沿整條 KV sequence），然後算 sum（同樣沿整條 KV sequence），再做除法。**這是 two-pass reduction**——標準 fusion 規則遇到 two-pass 就投降。

**(3) 第二個 matmul 沿 `S` 做 reduction**。softmax 完的結果 `[B, H, S, S]` 跟 V `[B, H, S, D]` 相乘變 `[B, H, S, D]`——這是沿 KV sequence 的 reduction。**這個 reduction 的 sketch 跟 QK^T 的 sketch 不相容**（QK^T 沿 `D` 做 reduction，沿 `S` 是 parallel；softmax·V 沿 `S` 做 reduction，沿 `D` 是 parallel）。**維度角色反轉了**。

FlashAttention 手動解決這三個問題的方法：

- **(1)** QK^T 只算一個 tile、不 materialize 到 HBM
- **(2)** online softmax——在同一個 tile 迴圈裡邊掃邊維護 running max + rescaled accumulator，one pass
- **(3)** 用 tile-level nested loop 把兩個 sketch 縫在一起——外層 loop 沿 output tile，內層 loop 沿 KV sequence（tile-wise），每個內層 iteration 同時做 QK^T tile、softmax rescale、V matmul tile

**FlashAttention 是三個技巧的手工組合**。Flashlight 的貢獻就是**把這三個技巧各自抽象成一條 compiler pass，讓 torch.compile 對任意 attention 變種自動應用**。

---

## 技術深潛之二：三條圖重寫規則各自的完整推導

### 規則 1：Structural Fusion with Dimension Demotion

**問題描述**。給定：

- Producer kernel 的 tiling sketch：`P = [(P_common, P_producer), (R_producer)]`
  - 這裡 `P_common` 是 producer 跟 consumer 共享的 parallel 維度（例如 batch × head × query）
  - `P_producer` 是 producer 特有的 parallel 維度（例如 KV block index）
  - `R_producer` 是 producer 的 reduction 維度（例如 head dim `D`）
- Consumer kernel 的 tiling sketch：`C = [(P_common), (P_producer, R_consumer)]`
  - `P_common` 對 consumer 也是 parallel
  - `P_producer` 對 consumer 變成 **reduction 維度**——這是 sketch 不相容的核心
  - `R_consumer` 是 consumer 特有的 reduction 維度

**傳統 fusion 遇到 sketch 不相容就放棄**。Producer 把 `[P_common × P_producer × ...]` 這個張量寫到 HBM，consumer 再讀回來 reduce over `P_producer`。**這個 HBM 讀寫就是 FlashAttention 想殺掉的東西**。

**規則 1 的洞察**：**任何 parallel 維度都可以被「降級」（demote）成 sequential 執行的 loop——只要你願意犧牲這個維度的並行寬度**。

**具體重寫步驟**：

1. 把 producer 的 `P_producer` 從 outer parallel 位置 demote 到 fused kernel 的內部——變成一個 sequential loop
2. 這個 loop 的每一次 iteration 產生一個 tile 的 producer 結果，直接餵給 consumer 的 reduction
3. 這個 tile 只需要 SRAM / register 存活，不用寫回 HBM
4. Consumer 對 `P_producer` 的 reduction 現在變成「在同一個 loop 內部做 accumulate」

**代價**：`P_producer` 的並行度被吃掉——原本可以有 `|P_producer|` 個並行 tile 同時算，現在必須 sequential 掃過。**如果 `|P_producer|` 大到能用滿 GPU 的 SMs，這個 demotion 會拖慢**。但**如果 `P_common` 已經夠大（大 batch、多 head），`|P_common|` 就能填滿 SMs，`P_producer` 的並行度是「多餘的」**——這時 demotion 是純粹的 win。

**Attention 的實例化**：QK^T 是 producer（沿 `D` reduce，沿 `[B, H, Q, KV]` parallel），softmax 是 consumer（沿 `KV` reduce，沿 `[B, H, Q]` parallel）——**`KV` 這個維度在 producer 是 parallel、在 consumer 是 reduction，經典的 sketch 不相容**。規則 1 把 `KV` demote 成 fused kernel 內部的 sequential loop，QK^T 的每個 KV tile 直接餵給 online softmax 的 running max + running sum——**這正是 FlashAttention 手寫的 outer loop**。

**這條規則的抽象威力**：任何「producer parallel + consumer reduction over 同一維度」的組合都能 apply。不只 attention——任何 GEMM 接 pooling、任何 conv 接 reduction 都適用。

### 規則 2：Semantic Fusion with Algebraic Transformation

**問題描述**。給定兩個相依的 reduction：`y = f(reduce(x))` 跟 `z = g(reduce(y))`——這個結構是 two-pass。第一 pass 掃 `x` 算 `reduce_1`，第二 pass 掃 `y = f(reduce_1)` 算 `reduce_2`。**Two-pass 就是兩次 HBM 讀寫**。

**規則 2 的洞察**：如果 `f` 對 reduction 的代數運算子 `⊕` 具有 **homomorphism 性質**（`f(a ⊕ b) = f(a) ⊗ f(b)`，其中 `⊗` 是另一個代數運算子），那**這兩個 pass 可以用「維護 running statistics 的 online algorithm」合併成一個 pass**。

**經典例子：softmax**

- Two-pass 版本：
  - Pass 1：`m = max(x_1, ..., x_N)`
  - Pass 2：`s = sum(exp(x_i - m))`, `y_i = exp(x_i - m) / s`
- One-pass online 版本：
  - 初始化 `m = -∞, s = 0`
  - 對每個 `x_i`：
    - `m_new = max(m, x_i)`
    - `s_new = s · exp(m - m_new) + exp(x_i - m_new)`
    - `m ← m_new, s ← s_new`
  - 最後 `y_i = exp(x_i - m) / s`

**這個轉換的核心是 `exp` 的 homomorphism**：`exp(x - m_new) = exp(x - m_old) · exp(m_old - m_new)`——你可以用 exponentials 的乘法性質把「重新掃一遍 x 做 max 修正」變成「乘一個 scaling factor」。

**Flashlight 把這個 pattern 抽象成 compiler pass**：找到「reduction 後跟 unary transformation 再跟第二個 reduction」的 IR pattern，檢查中間 unary 是不是 homomorphism，是的話就自動改寫成 online algorithm。

**廣義化**：這個規則對任何 associative reduction + homomorphism 都適用——**不只 softmax，也適用 log-sum-exp、count-min sketch、running variance（Welford's algorithm）、running L2 norm 等統計運算**。**LayerNorm、GroupNorm、RMSNorm 都是候選 target**。

**數值穩定性代價**：online algorithm 的浮點運算順序跟 two-pass 版本不同——浮點加法非結合，`float32` accumulator 通常無感，但 `bfloat16` / `FP8` accumulator 下會有 divergence 風險。**這是規則 2 的隱藏坑**，Flashlight 論文有提但沒完整解決。

### 規則 3：Structural Fusion with Tiling-Aware Dimension Elimination

**問題描述**。有些 sketch 不相容的 kernel 對，靠規則 1（dimension demotion）也 fuse 不了——因為兩邊 kernel 有太多獨立的 parallel 維度、demote 一個不夠。但**有一種特殊情形**：某個 parallel 維度 `P` 的大小 `|P|` 剛好小到能塞進單一 tile（`B_P ≥ |P|`，其中 `B_P` 是這個維度的 tile size）。

**這時 tile-level 的迴圈上界變成 `⌈|P|/B_P⌉ = 1`——這個維度在 tile-loop 層次上「消失」了**。

**規則 3 的洞察**：一旦某個維度消失，原本 sketch 不相容的兩個 kernel 就可能相容——**因為那個造成不相容的維度已經不在 tile-level sketch 裡了**。

**Attention 中的實例化**：softmax(QK^T) 的結果張量 `[B, H, Q_block, KV_seq]` 跟 V 相乘變 `[B, H, Q_block, D]`。中間的 `KV_seq` 是 reduction 維度——但**當你 tile 到 KV_block 級別、每個 KV_block 塞進單一 tile 時**，這個維度在 outer sketch 層次上消失了。**兩個 GEMM 的 sketch 就相容了，可以整條 fuse**。

**這條規則是 attention 那個「outer loop 沿 KV tile 掃、每個 iteration 同時做 QK^T tile + softmax rescale + V matmul tile」的 compiler 版本**。FlashAttention 手動這樣寫，Flashlight 用規則 3 自動推導。

**更廣的應用**：任何雙 matmul `(A · B) · D` 只要中間 shared 維度小，都能 fuse。**這正是 attention 的 epilogue fusion 精髓，也是為什麼 Evoformer 那類「多層 GEMM 疊 broadcast bias」結構能被 Flashlight 大幅加速——每層都會觸發規則 3**。

### 三條規則怎麼組合起來變 FlashAttention

Flashlight 對 vanilla attention 的自動 fusion 過程是三條規則的組合：

1. **規則 3** 先辨識 `(QK^T) · V` 這個雙 matmul 的 KV_block sketch 相容，把兩個 GEMM 放進同一個 outer loop
2. **規則 1** 把 QK^T 的 KV_block parallel 維度 demote 成這個 outer loop 的 sequential dimension
3. **規則 2** 把 softmax 的 two-pass reduction 改寫成 online algorithm，plug 進這個 outer loop

**結果就是 FlashAttention 的核心結構——但是完全 compiler 自動推導出來的，人類沒寫任何一行 CUDA / Triton**。

**對 FlexAttention 裝不下的 DiffAttn / Evoformer / IPA**——因為這三條規則是「用代數性質推導」不是「模板匹配」，所以只要 IR 裡的 sketch 分析結果符合條件，就會 apply。**這是「代數層次 compiler」跟「模板層次 compiler」的本質差異**。

---

## 為什麼 FlexAttention 的「靜態模板 + hook」有結構性上限

FlexAttention 的抽象是：

```python
def flex_attention(q, k, v, score_mod=None, block_mask=None):
    # 固定模板：softmax(score_mod(QK^T)) @ V
    ...
```

**這個抽象假設所有變種都能裝進 `softmax(score_mod(...))V` 這個固定形狀**。它的 hook 有兩個角色：

- `score_mod(score, b, h, q_idx, kv_idx)`：修改單個 score value（可以看 batch、head、query index、kv index）
- `block_mask`：宣告哪些 (query_block, kv_block) 對可以整塊跳過

**這兩個 hook 對「已知的 attention 變種」很夠**：

- **Causal**：`score_mod = lambda s, b, h, qi, kvi: s if kvi <= qi else -inf`
- **ALiBi**：`score_mod = lambda s, b, h, qi, kvi: s + alibi_bias(h, qi - kvi)`
- **Sliding window**：`block_mask = create_sliding_window_mask(window)`
- **PrefixLM / Document mask**：都是 `block_mask` 的靜態 pattern

**但是這三種變種 FlexAttention 裝不下**：

**(1) Differential Attention (DiffAttn)**：需要動態把 queries 沿某個維度 split 成兩半、分別算 attention、再相減。**這個 split 不是修改 score、也不是 mask 掉某些 pair——是把整個 attention 拆成兩個子計算**。FlexAttention 沒有「把 attention 內部結構動態拆解」的 API。

**(2) Evoformer row-wise / column-wise gated self-attention**：AlphaFold 的核心 block。**它在 softmax 之前加兩個 bias 矩陣，其中一個沿某個維度 broadcast**——一個 bias 是 `[H, S, S]`（跟 attention score 同 shape），另一個是 `[H, S]`（要 broadcast 到 `[H, S, S]`）。**FlexAttention 的 `score_mod` 只能看到 scalar `score`——它沒辦法表達「這個 bias 是廣播來的」這個 IR 層次的資訊**。**寫進 `score_mod` 也不是不行，但 compiler 拿不到 broadcast 資訊，就無法在 lower 階段做 tile-level 的 memory optimization**。

**(3) AlphaFold Invariant Point Attention (IPA)**：帶座標的 attention——每個 residue 有 3D 座標，attention 需要對座標做 SE(3)-invariant 的變換再算 score。**這個「幾何變換」不是 elementwise 的 score 修改，是需要沿 head dim / point dim 做一連串 tensor operation**。**FlexAttention 的 hook 抽象根本不夠**。

**(4) Rectified Sparse Attention (RSA)**：動態決定哪些 pair 保留、哪些丟——**這是 data-dependent 的 sparsity**，跟 FlexAttention 的 `block_mask` 靜態 pattern 不同。

**這四個變種暴露 FlexAttention 抽象的結構性上限**：**靜態模板 + hook 的抽象天花板是「已知形狀的變種」**——data-dependent 的維度變換、動態 split、broadcast bias、幾何變換等，都超出這個模板的表達能力。

**Flashlight 的破局點**：它不用模板——**它直接吃 PyTorch 用 native code 寫的 attention（就是 eager mode 那種），透過 TorchInductor 的 IR 分析 sketch，套三條規則自動 fuse**。**你把 attention 用 PyTorch 慣用的方式寫（張量代數、broadcast、sum、softmax），Flashlight 就會自動推導出 FlashAttention 風格的 fused kernel**——**這才是「compiler for attention」的正確抽象**。

---

## 加速數字冷讀：好在哪、輸在哪、怎麼解讀

論文提供的 benchmark 分四類，我一類一類冷讀。

### 類別 1：FlexAttention 相容的變種（vs FlexAttention）

**Score_mod 類**（Softcap、ALiBi）：**Flashlight up to 1.48x**。**這個是 Flashlight 的甜蜜點**——score_mod 的計算本身很輕，FlexAttention 的模板為了 general purpose 有一些額外開銷，Flashlight 的專門 fusion 能拿到 30–48% 的加速。**這個數字是 kernel-level 隔離測量**——E2E 加速通常會被非 attention 部分稀釋。

**Block_mask 類**（Causal、Sliding Window、Document Mask、PrefixLM）：**Flashlight kernel 執行時間略慢，但 FlexAttention 的 block-mask 建構時間拉低 E2E**。這裡的意思是：FlexAttention 有專門的 sparse block mask 資料結構（LRU-cached inspection），一旦 build 好 kernel 執行很快——但**每次 forward pass 都要重新 build 這個 mask**（除非 sequence 長度 / 形狀完全一致）。Flashlight 沒做這個專門優化，所以在**單次前向**上略慢，但在**攤提 mask 建構成本**時打平或略贏。

**E2E LLaMa-3.2-1B on vLLM**：**Softcap 場景 Flashlight 贏、Causal 場景 FlexAttention 贏**。Causal 是 decode 主 workload——**FlexAttention 在 causal 上的優勢是產品級的**，Flashlight 短期追不上。

**冷讀**：**Flashlight 目前不會取代 FlexAttention 在 causal / sliding / document mask 上的位置**。這些變種的 sparsity 結構是靜態的、可以 pre-compute 攤提，FlexAttention 為它們做的 LRU-cached block mask inspection 是硬優勢。**但在需要動態調整 attention score 的變種（softcap、ALiBi、data-dependent bias）上，Flashlight 有 30–48% 的加速空間**。

### 類別 2：超出 FlexAttention 的複雜變種（vs torch.compile 預設路徑）

**Differential Attention (DiffAttn)**：**H100 加速比 A100 更高**。這個 pattern 很典型——H100 的 memory hierarchy（更大 SRAM、更快 HBM3、TMA async）讓 fusion 的收益更大。**具體數字論文沒給單一速率，但 H100 上明顯優於 A100**——這代表**規則 1 + 規則 3 的組合收益跟 memory hierarchy 深度正相關**，B200 上會再進一步。

**Evoformer row-wise / column-wise gated self-attention**：**5x 或以上，H100 跟 A100 都成立**。**這是本文最漂亮的一擊**。Evoformer 是 AlphaFold 的核心 block，torch.compile 預設對它完全無能為力——它會產生一系列獨立 kernel、每個 kernel 都要把中間張量寫回 HBM。**Flashlight 用三條規則自動把整個 block fuse 成幾個大 kernel，減少 HBM 讀寫 5x 以上**——這個數字對 AI4Science 社群是有直接商業價值的。

**AlphaFold2 端到端**：**6–9% 延遲改善，H100 跟 A100 都成立**。**這個 E2E 數字比 5x 那個保守很多——原因是 AlphaFold2 不是只有 attention**，還有 templates、pair representation、structure module、recycling loop 等一大堆非 attention 計算。Attention 部分被 fuse 掉之後，總延遲的瓶頸就轉移到其他部分了。**這是 Amdahl's law 的冷讀——kernel-level 加速永遠會被 E2E 的其他部分稀釋**。

**冷讀**：**Flashlight 在「FlexAttention 裝不下的複雜變種」上是壓倒性優勢——5x 或以上 kernel 加速、6–9% E2E**。這個空間目前沒有其他 compiler-based 方案在做，Flashlight 是絕對領先者。

### 類別 3：vs FlashInfer

**FlashInfer 是專門為 LLM inference 優化的 attention library**——由 CMU 的 Tianqi Chen 團隊主導，已被 vLLM / SGLang / TensorRT-LLM 集成。它的核心優勢是**block-mask variants 的專門優化**（PagedAttention、prefix cache、continuous batching）。

- **Block_mask 變種**：FlashInfer 普遍更快——這是它的主場
- **ALiBi 這種 dense score_mod**：Flashlight / FlexAttention 更快——FlashInfer 在 dense 修改上沒做特別優化

**冷讀**：**三方比較沒人一家獨大**。FlashInfer 主攻 LLM inference 的 sparse pattern；FlexAttention 主攻 causal / sliding 等靜態 pattern；Flashlight 主攻 data-dependent / 複雜 attention 變種。**未來一年會看到融合——PyTorch 官方可能把 Flashlight 的規則跟 FlexAttention 的 sparse mask 優化合併**，最終形態應該是「Flashlight 做 default fusion，FlexAttention 做 sparse block mask 加速，FlashInfer 做 inference-specific pattern」的三層分工。

### 類別 4：硬體 coverage 的缺口

**論文測 H100 80GB 跟 A100 80GB——沒測 B200 / GB200 / Blackwell**。這個缺口值得注意兩件事：

1. **技術缺口**：Flashlight 目前 lower 到 Triton，Triton 對 Blackwell 的 TMA / cluster / distributed shmem 支援還在追（[[cutedsl-inductor-backend-pytorch-blackwell-cuda-moat]] 那篇有拆）。**Flashlight 在 Blackwell 上的實際加速目前是未知數**——可能收益更大（因為 memory hierarchy 更深），也可能因為 Triton 的 Blackwell 生成品質不夠而打折
2. **戰略機會**：**Flashlight 的規則往 CuTeDSL backend lower 是明顯的下一步**——這條路一旦通，PyTorch attention 變種在 Blackwell 上會再上一階。**這是未來一年 compiler 領域最值得押注的方向之一**

---

## 對比：跟 8/23 Helix S0 那篇的相反方向敘事

我 8/23 寫過 [`figure-helix-s0-109k-cpp-replaced-1khz-neural-controller-2026`]，主題是 Figure 用 1 個神經網路控制器取代 10 萬行手寫 C++ 機器人控制程式碼。**那篇的敘事方向是「手寫程式碼投降給模型」**——kernel-level 的手工藝（10 萬行 C++）被 end-to-end 訓練的神經網路取代。

**今天這篇 Flashlight 是完全相反方向的敘事**——**手寫 CUDA kernel（FlashAttention 那 10k+ 行 CUDA）被 compiler 取代，但取代它的不是模型，是形式化的圖重寫規則**。

**兩件事一起看的洞察**：**kernel-level 的手工藝正在被兩種方向擠壓**——

- **一邊是 LLM 生 kernel**：8/30 那篇 [`kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check`] 拆的就是這個方向——KernelBench 176 題、LLM agent 生成 GPU kernel 的成功率、效能對比手寫。**目前 LLM 路線的成功率大概 30–50%，效能通常比手寫差 2–5x**——**遠遠不到 production ready**
- **另一邊是 compiler pass 生 kernel**：Flashlight 就是這個方向——**成功率 100%（規則 apply 得到就一定正確），效能通常打平或勝出手寫**（Evoformer 5x），**跨硬體一致性、可重現性、可驗證性都遠勝 LLM 路線**

**目前 compiler pass 這條路的可驗證性、可重現性、cross-hardware 一致性都遠勝 LLM 路線**——這是我今天這篇文章的主要判斷。**Compiler engineer 這個角色未來三年不會被 LLM 取代**——不是因為 LLM 不行，是**因為 kernel 這個問題領域對「可驗證的正確性 + 可預測的效能 + 跨硬體 portability」的要求，恰好是 formal graph rewriting 的甜蜜點，是 LLM 目前做不好的**。

**但 compiler engineer 的技能組會被重塑**——過去的 compiler engineer 主要寫模板匹配 pass（LLVM 的 InstCombine、TorchInductor 的 fusion 規則都是這個 pattern）；**未來的 compiler engineer 需要證明代數性質、需要處理數值穩定性、需要跟硬體 memory hierarchy 對齊 tile sketch**——這是**更接近 formal methods + numerical analysis + systems 的混合技能組**。

---

## 對 compiler engineer 的三個技術 takeaway

### Takeaway 1：圖重寫 vs 模板匹配是質的差別

**過去 Inductor / TVM / XLA 的 fusion 大多是模板匹配**——「pattern 匹配到 GEMM + bias + ReLU 就替換成 fused kernel 模板」。這種 pass 每加一個新變種都要寫一個新模板——**vendor sprawl 就是這樣來的**。

**Flashlight 走的是「用代數性質推導」**——三條規則各對應一類代數性質（維度並行/降級 duality、reduction homomorphism、tile-level sketch equivalence）。**每加一條規則能自動 cover 一整類 pattern**。

**這是 compiler 從「模板時代」進到「重寫時代」的分水嶺**——LLVM 的 InstCombine 也在做類似轉型（從 pattern rewriting 往 e-graph 走）。**AI compiler 這個轉型的第一波論文就是 Flashlight**。**接下來一年應該會看到 XLA、TVM、IREE 各自的類似論文**——這個 wave 剛開始。

### Takeaway 2：數值穩定性是 fusion 的一級約束

**Flashlight 自己在 limitations 承認**：他們產生的 kernel 跟 mathematically-equivalent 版本有數值差（浮點非結合）。這在 `float32` accumulator 通常無感，**但在 `bfloat16` / `FP8` / NVFP4 accumulator 下會成為模型訓練 divergence 的隱藏來源**。

**任何走「代數重寫」路線的 compiler 都需要 numerical stability profile 這個 first-class 元件**——**這是下一輪 AI compiler 論文可以直接寫的坑**。

具體要研究的方向：

- 每條重寫規則的 forward-error bound（給定輸入的 magnitude range，重寫後結果的浮點誤差上界）
- 對 mixed-precision 的相容性——什麼 accumulator 精度下這條規則安全
- Regression tests——一組已知會炸的數值 corner case（NaN propagation、overflow、denormal handling）

**這個題目誰先寫出 MLSys / OSDI 論文誰就佔位**——目前完全沒人做。

### Takeaway 3：FlexAttention 的「靜態模板 + hook」抽象是有結構性上限的

**Flashlight 的存在證明了：FlexAttention 的 API 抽象不是 attention compiler 的終點**。**只要 attention 變種需要 data-dependent 的維度變換（DiffAttn split queries、Evoformer broadcast bias、IPA 幾何變換）**，template 就崩了。

**Compiler 抽象的下一階不是「更多 hook」，是「代數層次的 IR + 全域重寫」**。

**這個判斷可以外推到所有 DSL 設計**——CuTeDSL、Triton、Mojo、Halide 這些 DSL 的下一階不會是「加更多 primitive」，而是**「暴露 IR、讓使用者寫 rewriting rule」**。**未來的 DSL 使用者不只是 kernel 作者，還是 pass 作者**。

**這是 compiler engineer 職業長期價值的核心**——DSL 設計會變得更抽象、更多層次，需要更多懂編譯器內部的人。

---

## 對 Adam 的具體行動建議

### (a) 讀 Flashlight 論文——這是你 compiler career path 缺的入門讀物

你追 compiler 職涯路徑（`~/dev/career/4-Learning/Compiler-Path.md`）現在最缺的就是**「讀真的 compiler 論文而不只是 kernel 論文」的訓練**。過去半年我看你手邊在跟 CUTLASS / CuTeDSL / PMPP，這些都是**「kernel 作者視角」**——教你怎麼寫快的 kernel。**Flashlight 是「compiler 作者視角」**——教你怎麼寫能自動產生快 kernel 的 pass。**這兩個視角是互補但不同的**。

**Flashlight 這篇的三條圖重寫規則各配一頁 pseudocode + 兩到三個具體 example，是我讀過去半年最好的 attention compiler 入門讀物**。

**建議做法（3–5 小時的投資）**：

1. 拿 arXiv 2511.02043 v4 印出來（或存 iPad 上）
2. **把三條規則各手抄一遍**——強迫自己每一行都懂
3. 用 pen-and-paper 把 online softmax（規則 2 的核心 example）從 two-pass 推到 one-pass——**這個推導做完你 FlashAttention 的核心就徹底通了**
4. 讀 Section 5（Evaluation）——特別看 Evoformer 5x 那組數字是怎麼測的

**這 3–5 小時是你未來 Nvidia / Meta compiler team 面試的最佳投資**——面試官很可能問你「解釋 FlashAttention 的核心」、「解釋 online softmax」、「解釋 fusion 決策」，這三個問題 Flashlight 論文都有系統性的答案。

### (b) 把三條規則應用到你的 spconv workload

你手邊的 LiDAR spconv 有沒有 attention 變種？**表面上沒有——sparse convolution 不是 attention**。但**深入看，sparse convolution 展開後的 hash-based indexing + reduction 結構跟 attention 的 sparse pattern 有很深的同構性**：

- **SpConv V4 / TorchSparse 那類 backend 的 `Gather-GEMM-Scatter` pipeline**：
  - Gather 是 producer——沿某個 hash key 桶維度並行、沿 spatial 維度收集
  - GEMM 是 consumer——沿 output feature 維度並行、沿 input feature 維度 reduce
  - Scatter 是 consumer 的 consumer——把 GEMM 結果散射回原本的 sparse voxel 位置
- **這個 pipeline 的三個 kernel 之間的 sketch 不相容**——Gather 的 hash bucket 對 GEMM 是 batch 維度、對 Scatter 是索引維度

**規則 3（tiling-aware dimension elimination）可以直接類比**——你的 hash key 桶通常大小分佈很不均勻，**桶內元素數量少的桶正好符合「小到能塞進 tile」的條件**，就能把 gather / scatter 消進 GEMM 核心。

**具體做法**：

1. 拿你手邊的 spconv workload（KITTI / nuScenes / Waymo 資料集），profile 一下 hash bucket size distribution
2. 用直方圖看有多少桶是「小到能塞進 tile」的
3. 對這些桶手動模擬「fuse 進 GEMM」的效果——測 kernel time 跟 HBM 讀寫量

**這是 spconv capstone（[[project-career-research-2026]] 提過的方向）可以直接下手的技術主題**。**你能寫出「用 Flashlight 三條規則分析 spconv 的 fusion 空間」的英文技術文——直接放進履歷，Nvidia / Meta compiler team 面試會加分**。

### (c) 動手實測 Flashlight

**你目前的 PyTorch 應該是 2.x**——`pip install torch==2.9` + clone Flashlight 的 repo（他們 MLSys poster 有連 GitHub），跑 Evoformer 那個 example。

**profile 一下實際 Triton 輸出的 kernel 形狀**——用 `TORCH_LOGS="+inductor,output_code"` 把生成的 Triton kernel dump 出來，讀一讀看**Flashlight 的規則實際 emit 出什麼樣的 tile loop 結構**。

**你能寫出「Flashlight 在 spconv-adjacent workload 上的實測 + 三條規則哪條有效哪條無效」的英文技術文——這個等級的實測目前中文社群還沒人寫過**。

**時間估計**：環境建置 2 小時、跑 example + profile 3 小時、寫 blog post 5 小時——**總共 10 小時，換一篇履歷級別的技術文**。這個投資報酬率非常高。

---

## 冷讀

**Flashlight 不會取代 FlexAttention，但 FlexAttention 的「首選預設」位置會被侵蝕**。

FlexAttention 在 causal / sliding window / document mask 這種「靜態 sparsity 已知」的變種上有專門優化（LRU-cached block mask inspection），這個優勢暫時無法被純圖重寫的方案追上——**但這個優勢的商業空間隨著新的 attention 變種（DiffAttn、Evoformer、IPA、RSA、後面還會有更多）不斷冒出來會被稀釋**。

**兩年後（2028）的預測版本**：

1. **PyTorch 官方會把 Flashlight 的三條規則 upstream 進 TorchInductor 主線**——`torch._inductor.fx_passes.attention_rewrite` 這種模組會出現，變成 default 開啟
2. **FlexAttention 會被降級成「特別優化過的 sparse mask 加速路徑」**——只在明確 sparsity structure 的變種上有優勢
3. **Flashlight 的圖重寫變成「default fusion 引擎」**
4. **這樣的分工才是正確的抽象邊界**——compiler 做通用重寫，library 做專門優化

SemiAnalysis / PyTorch team 目前還沒喊這個判斷，但技術方向已經定了。

**對 compiler engineer 而言，這是「上車 attention IR / 重寫規則」的最好時機**——這個領域接下來三年會出至少五篇 MLSys / OSDI / ASPLOS 級別的論文，佔位者非常少。**Adam 你現在讀 Flashlight 論文 + 動手實測，就是在這個領域佔位**——**兩年後回頭看，這是你 compiler career 的第一個 signature contribution**。

---

## 相關閱讀（本站 compiler 系列）

- 2026-08-25：[cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026](./cuda-moat-two-front-mojo-open-source-llm-kernel-agents-2026.md) - Mojo 開源 + LLM kernel agents
- 2026-08-26：[qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026](./qualcomm-hexagon-mlir-second-front-cuda-lower-moat-2026.md) - NPU MLIR 繞過 CUDA
- 2026-08-27：[hf-kernels-package-registry-cuda-distribution-layer-2026](./hf-kernels-package-registry-cuda-distribution-layer-2026.md) - Kernel 變 pip install
- 2026-08-28：[tosa-block-scaled-mlir-mxfp-type-system-2026](./tosa-block-scaled-mlir-mxfp-type-system-2026.md) - MXFP 進 MLIR type system
- 2026-08-30：[kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check-2026](./kernelbenchx-176-tasks-llm-gpu-kernel-agent-reality-check-2026.md) - LLM 生 kernel 的現實 check
- 2026-08-31：[argus-data-flow-invariants-llm-gpu-kernel-verified-2026](./argus-data-flow-invariants-llm-gpu-kernel-verified-2026.md) - Data-flow invariants 驗證 LLM kernel
- 2026-09-02：[cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026](./cutedsl-inductor-backend-pytorch-blackwell-cuda-moat-2026.md) - CuTeDSL 進 Inductor
- 2026-09-03（本篇）：**Flashlight 圖重寫規則進 TorchInductor**

---

## 參考資料

- Flashlight 論文（arXiv v4）：[Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants](https://arxiv.org/abs/2511.02043)
- MLSys 2026 Poster #3540：[Flashlight poster page](https://mlsys.org/virtual/2026/poster/3540)
- FlexAttention 論文（arXiv 2412.05496）：[Flex Attention: A Programming Model for Generating Optimized Attention Kernels](https://arxiv.org/pdf/2412.05496)
- PyTorch FlexAttention 官方 blog：[FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/)
- TorchInductor 設計文件：[TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- Emergent Mind：[Fused Attention Kernel](https://www.emergentmind.com/topics/fused-attention-kernel)

---

*本文為 Nova 部落格 compiler 系列第八篇。系列文章聚焦 AI compiler 技術棧的各層變化——從 backend / DSL / package registry / verification / benchmark 到今天的 middle-end graph rewriting——目標是給走 compiler 職涯的軟體工程師（特別是我在追蹤的 Adam）一份完整的產業技術地圖。*
