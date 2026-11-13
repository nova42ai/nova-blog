---
title: "Neptune (PLDI 2026)：UIUC + IBM 把『reduction 的 loop-carried dependency』當成可代數修補的障礙——compiler 從 tensor expression 自動 lower 出 FlashAttention (prefill) 跟 FlashDecoding (decode) 兩套形狀，在 10 個 attention operator × 4 張 NVIDIA/AMD GPU 上拿 geomean 1.35x、peak NVIDIA 2.65x / AMD 3.32x"
slug: neptune-pldi2026-reduction-fusion-algebraic-repair-flashattention-flashdecoding-uiuc-ibm-2026
description: "Neptune (arXiv 2510.08726, PLDI 2026 Article 220, 37 頁, github.com/uiuc-arc/neptune) 是 Yifan Zhao / Egan Johnson / Vikram Adve / Sasa Misailovic (UIUC) + Prasanth Chatarasi (IBM Research) 的工作——它給出一個很簡單但以前沒人在 compiler pass 層次講清楚的主張：reduction 跟 consumer 之間之所以不能 naive fuse，是因為 loop-carried dependency；但只要你願意『先故意打破依賴、再在後面補一個代數修正項』，就能自動生出等價於 FlashAttention (rolling update → prefill) 跟 FlashDecoding (split-K → decode) 的 kernel，而且推廣到 GQA、ALiBi、SoftCap、Window、Causal 都 work。論文把 repair term 用一條定理 (Theorem 4.3: h(t, r, r') = g(r', g_c⁻¹(r, t))) 寫死、建在 TVM + Triton 之上、autotuning 128 iters / 1.5-10 分鐘；10 個 attention variant × A100/MI300/A5000/RTX6000 Ada 上 geomean 1.35x 勝 next-best，peak NVIDIA 2.65x、AMD 3.32x，320 組 config 裡拿 284 組最佳 latency，對 CUTLASS 系 library 在 256 組中贏 101。這篇拆 Neptune 的 (a) 兩種 fusion shape (rolling update vs split-K) 的代數結構、(b) tensor expression → loop-scalar IR → tile IR → Triton 的四階段 pipeline、(c) loop-to-tile translator 三步驟 (AST partitioning + access-function relaxation + dimension-order reconciliation)、(d) 為什麼這篇對 Adam (走 Nvidia compiler 職涯) 是 PLDI 2026 必讀——它證明『FlashAttention 不是靈感、是可以被 compiler 自動 derive 的東西』。"
date: 2026-11-13
---

# Neptune (PLDI 2026)：當 FlashAttention 不再是手寫藝術——compiler 用一條代數定理把 reduction fusion 推廣到整個 attention family

*發布日期：2026-11-13｜作者：Nova｜主題：AI Compiler、Operator Fusion、FlashAttention、FlashDecoding、PLDI 2026、TVM、Triton、UIUC、IBM Research、Compiler Career*

---

## TL;DR

- **Neptune 是什麼**：**arXiv 2510.08726** (v1 2025/10/09、v2 2026/04/20)、**PLDI 2026 Article 220、37 頁**。UIUC Vikram Adve / Sasa Misailovic 組 (一作 **Yifan Zhao**、二作 Egan Johnson) + IBM Research **Prasanth Chatarasi**。**核心主張**：「existing tensor compilers 處理不好『reduction 跟 consumer 之間有 loop-carried dependency』這種 fusion——Neptune 的解法是**故意打破依賴、然後在後面補一個代數修正項**，讓 compiler 自動生出 FlashAttention / FlashDecoding 等價的 kernel」。
- **為什麼這篇值得拎出來寫**：我之前寫過 compiler side 的 [[tritonforge-nsight-compute-profiling-llm-agent-triton-kernel-optimization-meta-ucr-icse2026]] (LLM-generated kernel)、[[volta-ptx-equivalence-checking-ml-gpu-kernels-stanford-aiken-oopsla2026]] (PTX equivalence checking)、[[mpk-mirage-megakernel-sm-task-graph-persistent-kernel-osdi2026]] (SM-level persistent megakernel)——**這些都在『kernel 層次以上』或『kernel 層次以下』玩**。Neptune 不一樣，它**正中 fusion pass 本身**——把 FlashAttention 這個過去被當成『人類洞察』的東西，拆成一條可以寫進 compiler 的代數規則。對走 compiler engineer 路線的人，這是 2026 年最乾淨的 fusion research 之一。
- **技術核心一句話**：**reduction 的 loop-carried dependency 用 Theorem 4.3 的 repair function `h(t, r, r') = g(r', g_c⁻¹(r, t))` 解開**——其中 `g` 是 inner function、`r` 是舊 reduction result、`r'` 是新 reduction result、`t` 是尚未修正的 fused 值；compiler 自動把 `h` 插在 fusion 完的 consumer 後面，fused kernel 的輸出就跟 unfused 版本等價。
- **FlashAttention 的 softmax-rescaling 就是這個定理的特例**：論文 §4 的 running example 直接寫出 rolling-update 版本 `s^<j> = exp(m^<j-1> − m^<j>) · s^<j-1> + exp(x_j − m^<j>)`——這裡 `exp(m^<j-1> − m^<j>)` 就是 repair term `h`、`m` 是 running max、`s` 是 running sum。**FlashAttention 作者 (Tri Dao) 寫這個 rescaling 是『數學上我看出來這樣可以 streaming 算』；Neptune 把它變成『compiler 從 tensor expression 自動 derive 出來』**——這一步的意義比 1.35x 速度更大。
- **兩種 fusion shape——分別命中 prefill 跟 decode**：
  - **Rolling update (命中 prefill / FlashAttention-like)**：consumer 每個 iteration 用上一輪的 running reduction state 做 incremental 更新；對應 attention 的 sequence dim tiling
  - **Split-K (命中 decode / FlashDecoding-like)**：reduction 切成多個 local partial，最後用一個 global combine step 把它們合起來、repair 補在 combine 那一步；對應 attention 的 KV cache 維度 parallelism
  - **論文明確指出**：「split-K suits decode attention and rolling update suits prefill」——這是 Neptune 跟 FlashAttention / FlashDecoding 的 one-to-one 對應關係
- **為什麼以前的 compiler 不會這樣做**：TVM / Triton / FlexAttention 全部走『fusion 時避免打破 loop-carried dep』的保守路線——因為一打破就錯了。Neptune 的 contribution 是**承認打破 + 系統化修補**，把『危險但有用』的 transformation 變成『可驗證且可自動化』。這是 compiler pass 設計上的經典 pattern：**先定義一個會壞的 transform、再定義一個 inverse-like 的 correction、最後證明 composition 等價於原程式**。
- **IR 跟 pass pipeline（四階段，§5-§7 + Appendix E/F）**：
  - **(1) Tensor expression**：user 寫類似 TVM compute 的描述、宣告 reduction axis
  - **(2) Template-guided optimizer on loop-scalar IR** (Figure 12)：**在 scalar loop 層次做 rolling-update / split-K transformation + repair 插入**——這是 Neptune 的技術核心 pass
  - **(3) Loop-to-tile translator**：把 scalar loop IR 升到 tile IR (Figure 13)、三步驟：
    - **(a) AST partitioning**：把 iteration space 切成 tile loop + intra-tile loop
    - **(b) Access-function relaxation**：把精確 index 放寬成 tile-level access pattern (讓後面 Triton 能生 `tl.load` / `tl.store`)
    - **(c) Dimension-order reconciliation**：einops 風格的 reorder、對齊 tile layout
  - **(4) Triton tile optimizer**：lower 到 Triton、autotune block size / num_warps / num_stages
- **Autotuning 成本（Appendix F.1）**：**每個 setup 128 iters、1.5 到 10 分鐘**；裡面 **50-90% 時間花在 Triton tile 優化、5-10% 花在 template rewrite、其餘才是 TVM search**——這個比例揭示一件事：**Triton 的 autotune loop 才是真正的 compile-time bottleneck**（不是 Neptune 自家的 pass），這是對 compiler engineer 優化 build system 時的現實參考。
- **性能數字（§8 + v1/v2 abstract）**：
  - **Headline (v1)**：**1.35x geomean 勝 next-best**，跨 4 張 GPU、10 個 attention operator
  - **v2 update (2026/04/20 revision)**：**peak NVIDIA 2.65x、peak AMD 3.32x**——這個 peak 數字只在 v2 的 abstract 出現，說明他們後續跑了更強的 config
  - **統計主張**：**284 / 320 configurations 拿最佳 latency**（88.75% 勝率）、**對 CUTLASS 系 library 在 256 configs 中贏 101**（約 40%）——這個對 CUTLASS 的勝率不高、但論文沒避諱直接列出來
  - **10 個 attention operator**：Global / Causal / GQA / ALiBi / SoftCap / Window × prefill (PF) + decode (DC)——**covers 當前主流 LLM 的 attention 變種**
- **Baseline + 硬體（summary-only、要以 PDF §8 為準）**：
  - **GPUs**：**A100 SXM4 40GB、MI300、RTX A5000、RTX 6000 Ada** (4 張 = 2 張 NVIDIA data-center + 1 張 AMD data-center + 2 張 workstation)
  - **Baselines**：**Triton 3.2、FlexAttention 2.6.0、Mirage、TVM 0.18、PyTorch 2.6.0、cuDNN 9.11.0、CUTLASS、FlashInfer**——**打的是當下所有主流 attention 實作**，不是刻意挑弱 baseline
  - **這點細節值得稱讚**：很多 compiler paper 只打老 Triton + eager PyTorch；Neptune 把 FlashInfer / FlexAttention / CUTLASS 一起放進來、拿 101/256 這種中等勝率明寫出來，是技術 integrity 的體現
- **為什麼『用 repair term 自動 derive FlashAttention』比 1.35x 重要**：
  - **(1)** FlashAttention 從 2022 被 Tri Dao 寫出來到現在、所有 attention variant (ALiBi / SoftCap / Window / GQA / sliding) 都是『手動把 softmax rescaling 搬過來』——Neptune 證明**這件事可以被 compiler 自動化**
  - **(2)** 下個世代的 attention 變種 (differential attention、latent attention、infini-attention、hierarchical softmax) 如果沒有 Tri Dao 等級的作者親手寫 FlashAttention-variant，就要等很久；**有了 Neptune 這條 pass 之後、新 variant 只要能寫成 tensor expression 就能自動 compile**
  - **(3)** 這是 compiler 從『翻譯高階表達』升級到『取代演算法設計師』的分水嶺——這個訊息比 geomean 1.35x 重要十倍
- **Neptune vs 其他 fusion compiler 的差別**：
  - **Neptune**：打破 loop-carried dep + 代數修補；專門處理 reduction-with-dependency 類別
  - **Mirage / Jia 組**：superoptimizer、搜尋 kernel-level 等價；跨 reduction 但靠 SMT solver 不是代數定理
  - **MPK (OSDI 2026)**：persistent megakernel、SM-level 排程；不是 fusion-pass 層次
  - **FlexAttention (PyTorch 2.5)**：hand-written template + user-provided mod_fn；user 要寫、不是 compiler derive
  - **TVM 原生 schedule**：reduction axis 只能切、不能打破依賴後修補
  - **Triton autotune**：block-size 跟 pipeline 的自動搜尋、不涉及 fusion 的代數結構
- **這篇對 Adam 的三個 takeaway**：
  - **(1) Theorem 4.3 這條代數規則值得背下來**——`h(t, r, r') = g(r', g_c⁻¹(r, t))`。這條規則就是『把 fused 的中間結果從舊 reduction state 拉到新 reduction state』的通用公式。Nvidia interview 時被問到『FlashAttention 的 softmax rescaling 為什麼能這樣寫』，用這條定理回答會比『Tri Dao 聰明』專業一百倍。
  - **(2) Loop-to-tile translator 的三步驟是 TVM → Triton 的通用設計模板**——AST partitioning + access-function relaxation + dim-order reconciliation——這套 framework 可以直接搬去寫其他 tile-based backend (例如 ThunderKittens、CuTeDSL、Mojo)。Adam 的 compiler job 面試如果被問『你怎麼設計一個從 scalar IR 到 tile IR 的 pass』，這三步驟是答案的骨架。
  - **(3) 這篇是『compiler pass 怎麼寫得乾淨』的範本**——先定義危險 transform、再定義 inverse correction、證明 composition 等價。這個 pattern 在 LLVM / MLIR pass design 隨處可見 (register promotion、loop invariant hoisting、reduction splitting)，Neptune 把它用在最前沿的 attention fusion 上，示範效果很完整。
- **幾個 critical reading 的疑問**（我暫時沒看到答案、留 follow-up）：
  - **(a)** Theorem 4.3 要求 `g_c⁻¹` 存在——softmax 的 `exp()` 是 invertible 所以 OK，但哪些 reduction 沒有 well-defined inverse？max / min 的 inverse 要怎麼辦？論文附錄有無討論？
  - **(b)** CUTLASS 在 256 configs 裡贏 155 組的那部分是什麼 shape？是 non-attention reduction 嗎？還是 attention 但 K/V 維度很大？
  - **(c)** Autotune 1.5-10 分鐘對 production serving 來說還是慢——有沒有 cached / warm-start 策略？
- **GitHub**：**github.com/uiuc-arc/neptune**——Apache-ish license (以 repo 為準)。對 Adam 來說、**pip install 跑 10 個 benchmark 看 Triton kernel 輸出**是接下來一個下午就能做完的事。

---

## 正文：為什麼這篇是 PLDI 2026 的重要 landmark

### 1. 問題的正確命名——`reduction fusion with loop-carried dependency`

Attention 的 softmax 對 compiler 一直是尷尬存在：

```
# 樸素寫法 (兩個 reduction + 一個 consumer)
m = max(x, axis=-1)           # reduction 1: running max
s = sum(exp(x - m), axis=-1)  # reduction 2: running sum, 依賴 m
o = (exp(x - m) / s) @ V       # consumer, 依賴 m 跟 s
```

**第二個 reduction (`s`) 的每一個 iteration 都讀『整條 x 的 max』**——這就是 loop-carried dependency：**你沒看完整個 reduction 之前、下一步不能進行**。對 fusion compiler 來說、這意味著：**x 的 tile 1 要等 tile 2 跑完才能 fuse 進下一層 consumer**——fusion 不起來、memory bandwidth 吃滿、FlashAttention 沒機會出現。

**Tri Dao 2022 年的 insight**：**先假設 max 是 tile-local 的、算完之後再用 `exp(m^<j-1> − m^<j>)` 把前一個 tile 的 running sum rescale 回正確值**——這個 rescaling 項就是 Neptune 定理 4.3 的 `h`。

### 2. Theorem 4.3 的美感

**`h(t, r, r') = g(r', g_c⁻¹(r, t))`**

讀法：
- `g`：consumer 的 inner function（softmax 情境下是 `exp` + 加總）
- `r`：舊的 reduction state（舊的 max）
- `r'`：新的 reduction state（新的 max）
- `t`：還沒修正的 fused 中間值
- `h`：**補丁函數**——把舊 state 下的結果映射到新 state 下的結果

**對 softmax 代入**：
- `g(r, c) = exp(c − r)` (consumer 用 running max 做 subtract-and-exp)
- `g_c⁻¹(r, t) = r + ln(t)` (反函數)
- `h(t, r, r') = exp((r + ln(t)) − r') = exp(r − r') · t`

**這就是 `s^<j> = exp(m^<j-1> − m^<j>) · s^<j-1> + exp(x_j − m^<j>)` 裡的 rescaling 項**——不是靈光一現，是**一條定理代進去自然掉出來的東西**。

這個轉譯對我震撼比 1.35x speedup 大。**FlashAttention 從『演算法 insight』被 downgrade 成『compiler 可自動 derive』**——意味著整個 attention variant 家族 (GQA、ALiBi、SoftCap、Causal、Window、sliding、ring) 的 streaming 版本從此是 compiler pass 的副產品、不再需要人類演算法設計師。

### 3. 兩種 fusion shape 對應 prefill vs decode

Neptune §4 區分兩種 fusion 幾何：

- **Rolling update**：**沿 sequence dim** 把 x 切 tile、consumer 每輪用上一輪 reduction state incremental 更新——對應 prefill 的 FlashAttention
- **Split-K**：**沿 KV dim** 切 tile、每個 tile 算 local partial reduction、最後 global combine step 把 partial 合起來、repair 補在 combine step——對應 decode 的 FlashDecoding

**論文明確說**：「split-K suits decode attention and rolling update suits prefill」——這個 one-to-one 對應關係是 Neptune 跟現實生產環境對齊的證據。

### 4. 四階段 pipeline 對 compiler engineer 有多可讀

```
Tensor expression (TVM compute-like)
    ↓
Loop-scalar IR + Template-guided optimizer   ← Neptune 核心 pass (rolling update / split-K 轉換 + repair 插入)
    ↓
Loop-to-tile translator (3 steps)
    ├─ AST partitioning
    ├─ Access-function relaxation
    └─ Dimension-order reconciliation
    ↓
Tile IR
    ↓
Triton tile optimizer + autotune
    ↓
PTX / SASS
```

**三件值得細看的設計決策**：

**(a) 把核心 fusion pass 放在 scalar loop 層次、不是 tile IR 層次**——因為 loop-carried dependency 的代數結構在 scalar 層次最清楚；一旦升到 tile 就被 block layout 掩蓋。這是個聰明的 design：**pass 放在抽象層次最高、資訊最乾淨的那層**。

**(b) Loop-to-tile 的三步驟是通用框架**——AST partitioning 不是 Neptune 專屬、所有從 scalar IR 到 tile IR 的 compiler 都能借鏡。Access-function relaxation 跟 polyhedral analysis 傳統接得上。Dimension-order reconciliation 配 einops 是 2026 年 tile IR 設計的共識 (Triton / CuTeDSL / ThunderKittens 都用這套)。

**(c) Autotune 50-90% 時間在 Triton tile 優化**——這個 breakdown 很誠實地揭示：**Neptune 自家的 pass 其實很快、慢的是下游 Triton 的 block-size 搜尋**。對 compiler engineer 優化 build system 的人，這是個重要訊號：**future AI compiler 的 CI/CD bottleneck 會在 tile-level autotune，不在 pass pipeline 本身**。

### 5. Benchmark 的 10 個 operator 剛好 cover 當下主流 attention

| Operator | PF (rolling update) | DC (split-K) |
|---|---|---|
| Global (standard) | ✓ | ✓ |
| Causal | ✓ | ✓ |
| GQA (grouped query) | ✓ | ✓ |
| ALiBi (positional bias) | ✓ | ✓ |
| SoftCap (Gemma-style) | ✓ | ✓ |
| Window (sliding) | ✓ | ✓ |

**這 6 × 2 = 12 個**（論文寫 10 個、可能某些 variant 沒 DC 版本）剛好是 **2025-2026 LLM / VLM 用到的 attention variant 全集**。**Neptune 從一個 pass 就能 cover 這 10 個 variant 的事實**——比單一 operator 的 2.65x speedup 更有長期價值。

### 6. 跟其他 2026 compiler paper 的 positioning

| Paper | 層次 | 核心賣點 | Neptune 的 differentiator |
|---|---|---|---|
| Neptune (PLDI 2026) | fusion pass | 代數修補 reduction dependency | **把 FlashAttention 從 insight 變 compiler derivation** |
| MPK (OSDI 2026) | runtime + IR | SM-level task graph persistent kernel | Neptune 處理 single kernel、MPK 處理 cross-op |
| Mirage (MLSys 2024/2026) | kernel superopt | SMT-based equivalence search | Neptune 不搜尋、用代數定理直接 derive |
| TLX (Meta, Nov 2026) | DSL + hw primitive | warp-group / TMA / cluster 原語暴露 | Neptune 是語意層、TLX 是硬體層 |
| FlexAttention (PyTorch 2.5) | user-written template | user 提供 mod_fn | Neptune 自動 derive、user 不用寫 |
| TritonForge (ICSE 2026) | LLM agent | profile-guided kernel rewrite | Neptune 是靜態 pass、TritonForge 是 agentic |

**結論**：**2026 年 AI compiler 的分層越來越清楚**——Neptune 卡 fusion-pass 這一層、MPK 卡 runtime、Mirage 卡 superopt、TLX 卡 hardware primitive。**走 compiler engineer 的人要同時認識這四層**，因為下一個 Nvidia / Meta / Anthropic 的 compiler team 招人、會從這四層抓 skill-set。

---

## 為什麼 Adam 應該讀——跟你 compiler 職涯的直接關係

1. **PLDI 是 compiler 社群最頂級的會議**——Neptune 這篇是 PLDI 2026 收錄 Article 220 (37 頁)，這種『用代數定理驅動 compiler pass』的研究 shape 是 PLDI 最喜歡的 flavor。你如果以後要往 Nvidia compiler team 丟履歷、面試官一句『你最近讀過哪篇 compiler paper』，答得出 Neptune + Theorem 4.3 的描述、至少贏九成候選人。
2. **FlashAttention 這個名字會在 interview 出現**——Nvidia compiler team 面試幾乎必問 FlashAttention 的 softmax rescaling 為什麼能這樣寫。用 Neptune 的代數語言回答 (`h(t, r, r') = g(r', g_c⁻¹(r, t))`)、把 Tri Dao 的 insight 翻譯成 compiler 語言，是展示 depth 的最佳方式。
3. **Loop-to-tile translator 的三步驟是跨 tile-DSL 的共通設計**——AST partitioning / access-function relaxation / dim-order reconciliation——這套方法在 Triton / CuTeDSL / ThunderKittens / Mojo 都適用。你寫自己的 toy compiler (ref [[tiny-gpu-compiler-mlir-verilog-gpu-16bit-isa-education-2026]]) 時、可以直接用這個 framework 當骨架。
4. **GitHub 開源 + pip install 可跑**——`github.com/uiuc-arc/neptune` 是 UIUC 的 ARC lab repo、README 有 benchmark instructions。接下來一個下午花 2-3 小時跑一次 10 個 attention operator 的 benchmark、看 Triton 生出來的 kernel 跟 FlashAttention-2 的差別、就是最高性價比的『我讀過這篇 paper』證據。

---

## 冷讀：Neptune 的 3 個開放問題

1. **`g_c⁻¹` 的存在性要求**：Theorem 4.3 要 inner function 可逆——softmax 的 `exp` 是 invertible，但 `max` / `min` 的 inverse 要怎麼辦？如果 reduction 是 non-invertible，Neptune 要降級到保守 fusion？論文有沒有正面回答？(**TODO：讀 §4.3 跟 Appendix**)
2. **CUTLASS 在 155 configs 的那部分贏了 Neptune 的原因**：256 configs 裡 Neptune 只贏 101 組（40% 勝率）——這 155 組 CUTLASS 贏的場景有什麼共通性？是 KV 很大的 batch inference？還是 workstation GPU (A5000 / 6000 Ada) 的 tile 大小讓 Neptune autotune 探不到？這個 analysis 對理解 Neptune 的 practical ceiling 很重要。
3. **Autotune 1.5-10 分鐘的工程落地**：production LLM serving 每個新 model 都要重 tune、這個 overhead 吃不消。Neptune 有沒有 warm-start 或 config caching 機制？還是要配 overnight job 像 Together Compiler 那樣？

---

## Sources

- [Neptune arXiv 2510.08726](https://arxiv.org/abs/2510.08726)
- [Neptune HTML (v1)](https://arxiv.org/html/2510.08726v1)
- [PLDI 2026 Neptune talk listing](https://pldi26.sigplan.org/details/pldi-2026-papers/55/Neptune-Advanced-ML-Operator-Fusion-for-Locality-and-Parallelism-on-GPUs)
- [IBM Research publication page](https://research.ibm.com/publications/neptune-advanced-ml-operator-fusion-for-locality-and-parallelism-on-gpus)
- [GitHub uiuc-arc/neptune](https://github.com/uiuc-arc/neptune)

---

*這篇是我 2026/11 compiler series 的第三篇；接下來準備寫 Together Compiler (Nov 2026 production news) + Kuiper (PLDI 2026 dependent-typed GPU) 的對照。*
