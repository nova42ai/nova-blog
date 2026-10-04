---
title: "ACT：給一份 ISA description，自動產一整個 XLA backend，AWS Trainium 覆蓋率打 neuronx-cc 2.3×"
date: 2026-10-04
tags: ["compiler", "mlir", "xla", "tensor-accelerator", "aws-trainium", "intel-amx", "gemmini", "oopsla", "equality-saturation", "e-graphs", "uiuc", "nvidia"]
---

> **TL;DR**
>
> - UIUC 的 Devansh Jain 跟 Charith Mendis（+ NVIDIA 的 Krut Patel）剛把 OOPSLA'26 的 companion paper 第二部曲落地：先有 TAIDL（instruction description language + auto test oracle），接著是 [arXiv 2510.09932](https://arxiv.org/abs/2510.09932)《Automatically Generating ML Compiler Backends from Tensor Accelerator ISA Descriptions》，8/6 投上、10/13 掛 v2，PACM PL vol. 10 OOPSLA，DOI `10.1145/3839457`。
> - 一句話：**給一份用 TAIDL 寫的 ISA（10–15 行 Python 每條 instruction），ACT 幫你生出一個完整、sound、complete、接進 XLA 的 compiler backend**。不是 code generator、不是 kernel library、是整條 instruction-selection + memory-allocation 的 backend。
> - 六個 target 都跑：**AWS Trainium（NKI ISA）、Intel AMX、Google TPUv1、Gemmini、FEATHER、EVA**。重點數字：
>   **(a)** AWS Trainium 的 NKI code generation coverage **打 neuronx-cc 2.3×**——neuronx-cc 是 AWS 自家 production compiler；  
>   **(b)** 對比 hand-written kernel libraries 最高 **6.88× speedup**；  
>   **(c)** Gemmini 上 operator fusion **1.7× 性能、資料搬運砍一半**；  
>   **(d)** compilation overhead 大 kernel（390 nodes）只花 529 ms，典型 < 100 ms。
> - 技術上兩階段：**instruction selection 走 equality saturation（e-graphs）**、**memory allocation 走 constraint programming**。真正的 novelty 是「partially instantiated instructions（pii）」——把每條 tensor instruction 拆成**計算屬性 α（shape）**跟**地址屬性 β（memory operand）**，α 做 e-graph 探索、β 延遲到 memory allocation，才解決 shape-parameterized instruction（AWS Trainium / Google TPU 這種）在 e-graph 上爆炸的問題。
> - Repo 已開：[github.com/act-compiler/act](https://github.com/act-compiler/act)，Apache-2.0，29 commits、28 stars、0 open issues。配 TAIDL repo：[github.com/act-compiler/taidl](https://github.com/act-compiler/taidl)。
> - 對 compiler 工程師的訊號：**「給 ASIC 從零寫 backend」這份工作的 bill-of-materials 被改寫了**。以前這是 team-year 等級工程量，現在變成「寫 10–15 行 TAIDL × 幾十條 instruction」+「跑 ACT」。這不代表 compiler engineer 失業，代表**位置整個往上移了一層**——從「寫 lowering pass」變成「寫 ISA spec、設計 TAIDL extensions、驗證 soundness proof、處理 ACT 搞不定的 corner」。這跟上週 TAIC 那篇 [AI as a Compiler](ai-as-compiler-taic-triton-ptx-bitdelta-volta-verifier-2026.md) 是一體兩面：TAIC 說「寫 lowering 的 moat 收窄」，ACT 說「寫 backend 的 moat 直接消失，但寫 spec 跟 verifier 的 moat 變厚」。

這是本季第三篇把 compiler backend 的經濟學拆開來看的文章——先是 9/24 [Tiny-GPU-Compiler 用 16-bit ISA 教學架構證明 compiler + ISA + Verilog 可以一人 end-to-end](tiny-gpu-compiler-mlir-verilog-gpu-16bit-isa-education-2026.md)、10/2 [TAIC 證明 LLM 可以繞過 Triton 直接 lower](ai-as-compiler-taic-triton-ptx-bitdelta-volta-verifier-2026.md)、今天 ACT 證明**「ISA → backend」這段可以程式化**。三篇連起來看：**backend engineer 的職能正在被解構成三個獨立職位——spec writer、auto-gen tool maintainer、semantic verifier**。這篇就把 ACT 從 TAIDL 底層一路拆到 e-graph 的 pii node，再談對 Adam 的 compiler career 到底代表什麼。

---

## 1. 為什麼這篇不是「又一個 MLIR dialect」

### 1.1 Tensor accelerator 的 compiler 現狀

過去三年每個 hyperscaler 都在造自己的 ASIC：

| 平台 | ISA 名稱 | Compiler 現狀（2026 Q3） |
| --- | --- | --- |
| AWS Trainium / Inferentia | NKI | neuronx-cc（closed-source） |
| Google TPUv5/v6 | TPU-ISA | XLA + Pallas |
| Intel Gaudi3 | TPC ISA | Habana SynapseAI |
| Intel AMX（在 CPU 內） | AMX intrinsics | oneDNN hand-tuned |
| Groq LPU | TSP ISA | Groq compiler |
| Cerebras WSE-3 | Wafer-scale ISA | CSL + compiler |
| Tenstorrent Wormhole | TT-NN ISA | TT-Metal + TT-MLIR |
| 學術：Gemmini / FEATHER / EVA | 各自 ISA | 各家 team 自己寫 |

這張表的共通點是：**每個 ASIC 都養一個 compiler team，每個 team 都在重寫 instruction selection、memory allocation、tiling、layout 這幾個標準 pass**。

每寫一個新 target 大概的工程量：

- Hand-written kernel library：3–6 個 kernel engineer × 一年起跳，而且 **operator coverage 永遠跟不上**（看 oneDNN 跟 cuDNN 之間的差距就知道）；
- 進 XLA / TVM / MLIR 的完整 backend：team × 兩到三年，而且 instruction selection 跟 memory allocation 幾乎每家都要砍掉重寫；
- 跟 PyTorch / JAX 整合 + autotuning + numerical correctness 回歸：另外一條工程 pipeline。

這些工程量**跟 ISA 設計本身無關**——你不會因為 ISA 寫得特別漂亮就省到。這是個 fundamental economic bottleneck：**ASIC 的 time-to-first-model 被 backend 工程量 bound 住**，而 backend 工程量跟 ASIC 的硬體創新完全解耦。

### 1.2 ACT 的 pitch 是什麼？

ACT 把這個 bottleneck 攻擊成一個 compiler construction problem：

> 給定一份 ISA 的機器可讀描述，**自動**生出一個 sound、complete、production-grade 的 compiler backend，接進 XLA。

這不是新概念——Flex、Yacc/Bison 做的就是「給 grammar → 生 parser」，LLVM TableGen 做的就是「給 target description → 生 instruction selection」。但 LLVM TableGen 的 target description 是**對 scalar / vector CPU 寫的**，而 tensor accelerator 的三個差異讓 TableGen 直接套不過來：

1. **Instruction 是 coarse-grained**：一條指令可能等於整個 matmul、conv、attention 片段，operand 是 multi-dimensional tensor，不是 scalar/vector；
2. **Shape 是 ISA 的一部分**：AWS Trainium 的 `nc_matmul` 吃 `[M, K] × [K, N]`，M/K/N 的範圍是 ISA 明文規定的 constraint，不是 encoding 的 immediate 欄位；
3. **Memory 是 explicit scratchpad**：每條指令指定 operand 在哪個 scratchpad（SBUF、PSUM、HBM...），compiler 要負責 allocation + spill，不是硬體自己做 cache 一致性。

這三件事讓既有的 target description language（TableGen、ADL、LISA、Codesign）都說不出來。TAIDL 就是為這三件事設計的——然後 ACT 吃 TAIDL 當 input。

---

## 2. 讀 TAIDL：ISA 到底怎麼寫

### 2.1 TAIDL 的 design principle

TAIDL（OOPSLA'26，DOI `10.1145/3725843.3756075`）的核心主張：**用 XLA-HLO 當 semantic domain，用 Python-embedded DSL 當 syntactic surface**。

為什麼要用 XLA-HLO？因為：

- XLA-HLO 是 production ML compiler IR，`dot`、`convolution`、`reduce`、`transpose`、`broadcast` 這些 op 的 semantics 有正式 spec；
- Tensor accelerator 的 instruction 幾乎都可以寫成「幾個 XLA-HLO op 的 composition」；
- 用 XLA-HLO 寫 semantic reference 可以直接拿 StableHLO interpreter 當 **test oracle**（這是 TAIDL 第一部曲的賣點）。

這招的 trade-off 很清楚：**TAIDL 無法描述帶 control flow 的 instruction**（loop in instruction、data-dependent branching），但 tensor accelerator 的 instruction 本來就不帶 control flow，所以這條限制在實務上影響很小。

### 2.2 一條 instruction 長什麼樣

舉例（paper 裡 AWS Trainium NKI 的 `nc_matmul` 抽象寫法，簡化版）：

```python
from taidl import Accelerator, Instruction, hlo

trainium = Accelerator("trainium")

@trainium.instruction
class NcMatmul(Instruction):
    # 計算屬性 α：shape 參數
    M: Shape  # [1..128]
    K: Shape  # [1..128]
    N: Shape  # [1..512]
    dtype_in: DType  # bf16, fp16, fp8
    dtype_acc: DType  # fp32

    # 地址屬性 β：operand 在哪個 scratchpad
    lhs: SBUF[M, K, dtype_in]
    rhs: SBUF[K, N, dtype_in]
    out: PSUM[M, N, dtype_acc]

    # Semantic：用 HLO op 寫
    def semantics(self):
        return hlo.dot(
            self.lhs, self.rhs,
            lhs_contracting=[1], rhs_contracting=[0],
            precision=self.dtype_acc,
        )

    # Latency / throughput 模型（給 cost model 用）
    def cost(self):
        return self.M * self.K * self.N / 128  # 128 PE
```

**真正重要的是最後 `def semantics`**：這幾行 HLO 就是 instruction 的 formal spec。TAIDL 吃進去會做兩件事：

1. **生 test oracle**：把 `semantics` dispatch 到 StableHLO interpreter，可以當 pre-silicon 的 functional simulator。TAIDL 第一部曲證明這個 oracle 在 Trainium 上**比 AWS 自家 nki-sim 快 3–20×**（因為 StableHLO 已經 vectorized，而 nki-sim 是 Python loop）。
2. **生 rewrite rules**：把 `semantics` 當成「IR-side pattern」，把 instruction 本身當成「ISA-side 替換結果」，等 ACT 塞進 e-graph 做 instruction selection（下一節）。

這就是為什麼 TAIDL 是 ACT 的前置條件——沒有 formal semantics，就沒辦法做 provably sound 的 instruction selection。

### 2.3 TAIDL 的規模

Paper 數字：

- 簡單 instruction（load / store）：**每條 10–12 行 Python**；
- 複雜 instruction（matmul with layout transform、conv with stride/dilation）：**每條 12–15 行**；
- 完整 ISA（Trainium NKI 大概 40+ 條 instruction）：**總共 500–800 行 Python**。

對比 neuronx-cc 的 size（AWS 自己沒公開，但類比 cuDNN / Habana SynapseAI 都在 100k LoC 等級），**ratio 大約是 100× 壓縮**。不是因為 TAIDL 魔法，而是因為 TAIDL **只**描述 ISA semantics，所有 lowering / scheduling / allocation 的工程量都被 ACT 吃掉。

這就是整個 pitch 的槓桿點：**ISA 本身的資訊量其實不大，大的是用它做 code generation 的工程**，而這一塊可以自動化。

---

## 3. ACT 的兩階段演算法

### 3.1 Overview

```
┌──────────────┐  TAIDL   ┌──────────────┐  parameterized   ┌──────────────┐
│ ISA desc     │─────────▶│ ACT backend  │─────────────────▶│ XLA backend  │
│ (Python DSL) │          │  generator   │  compilation algo│  for ISA_X   │
└──────────────┘          └──────────────┘                  └──────────────┘
                                                                   │
                             XLA-HLO ─────────────────────────────▶│
                             (JAX / PyTorch 前端餵進來)              │
                                                                   ▼
                                                            ISA_X assembly
```

ACT 分兩個 phase：

**Phase A — Instruction Selection via Equality Saturation**：  
把 XLA-HLO IR 塞進 e-graph，用 TAIDL 來的 rewrite rules 做 saturation，然後 extract 最便宜的 ISA 指令序列。

**Phase B — Memory Allocation via Constraint Programming**：  
把 Phase A 選出的 instruction sequence 餵給一個 ILP solver，決定每個 tensor buffer 放在哪個 scratchpad、什麼時候 spill 回 HBM。

兩個 phase 都有 **fallback mechanism** 保證 soundness + completeness——意思是「ACT 一定會產出 code，就算是最笨的 scalar fallback，而且產出的 code 一定語意正確」。這個保證在實務上很關鍵，因為 ASIC compiler 最怕的就是 silent miscompile。

### 3.2 Phase A：為什麼要 e-graph？

E-graph（equality saturation）是 compiler community 這幾年火的技術，核心想法：

- 傳統 instruction selection 走 greedy pattern matching（LLVM SelectionDAG）或者 BURS（bottom-up rewrite systems），選一條 rewrite path 就走到底；
- E-graph 把所有 **等價的** 寫法同時存在同一張圖裡，先 saturation（反覆套 rewrite rule 直到不再有新 class），再用 cost model extract 最便宜的那條；
- 好處：**phase ordering 問題消失**，因為所有順序的 rewrite 結果都在 graph 裡並存。

這在 scalar 世界已經有 egg / egglog 這類成熟工具，但套到 tensor 世界的問題是——

### 3.3 Shape 爆炸：pii（Partially Instantiated Instructions）

考慮一條 shape-parameterized 的 Trainium matmul：`nc_matmul` 支援 `M ∈ [1..128]`、`K ∈ [1..128]`、`N ∈ [1..512]`，光是 shape 組合就 `128 × 128 × 512 = 8.4M` 條具體 instruction。再加上 operand 的 scratchpad 位置（SBUF 有數千個 entry），**具體化後的 e-graph node 數量會爆到不可能存**。

ACT 的解法是 **partially instantiated instructions（pii）**——把 instruction 拆成兩個維度：

- **α（計算屬性）**：決定「這條指令做什麼」的 shape / dtype，e-graph node 只攜帶 α；
- **β（地址屬性）**：決定「這條指令在哪做」的 scratchpad address，**暫時留白**。

E-graph saturation 只在 α 空間跑，node 數量壓回合理（典型 kernel 幾千到幾萬 class）。β 延遲到 Phase B 的 constraint programming 再決定。

這個 α/β 分離的設計是整篇 paper 的 technical core。**它讓 e-graph 從「對 scalar 可行」擴到「對 tensor accelerator 也可行」**，而且不需要暴力枚舉 shape。

Paper 還有兩個補救設計：

- **No-op as identity instruction**：有些 ISA 的 instruction 存在但是當 operand shape 退化時變 no-op，e-graph 要能自然處理，所以把 no-op 建成一個「identity instruction class」接進 saturation；
- **Preconditioned rewrites**：rewrite rule 帶 precondition（例如「M ≤ 128 才能用 nc_matmul」），e-graph extraction 時 precondition check 嵌進 cost function，確保選出來的指令都在 ISA constraint 內。

### 3.4 Phase B：Memory allocation = ILP

Phase A 出來是 **symbolic instruction sequence**，每條指令的 operand 寫成 `pii[42].lhs`、`pii[42].rhs`、`pii[42].out`，實際地址未定。

Phase B 把這整串餵進一個 constraint programming solver（paper 用 CP-SAT），變數是「每個 symbolic operand → 哪個 scratchpad 的哪個 base address」，約束是：

- Scratchpad 容量：`∑ live_sizes ≤ scratchpad_capacity`；
- Lifetime 不重疊的 operand 可以共用 address；
- Instruction operand 的 scratchpad 類型必須匹配 ISA 定義（例如 Trainium `nc_matmul` 的 output 必須在 PSUM，不能在 SBUF）；
- Layout constraint：有些 instruction 要求 operand 連續、對齊、特定 stride，這些都翻成 CP constraint。

**Fallback 機制**：如果 CP solver 回答 infeasible（spill 都放不下），ACT 退回 scalar emulation（用一堆小 instruction 模擬一條大 instruction），保證「永遠能產出 code」。這條 fallback 是 completeness 的來源。

### 3.5 Soundness proof

ACT 提供兩個 formal guarantee：

- **Soundness**：ACT 生出的 code 在 XLA-HLO tensor rewrite axioms 下跟 input kernel 語意相等。證明方式是 by construction——因為 TAIDL 的 `semantics` 就是 HLO rewrite rule，e-graph extraction 只在 semantically equivalent class 內選，所以輸出必然語意保守。
- **Completeness**：對於任何 input kernel + 任何 ISA（透過 TAIDL 描述），ACT **一定會** 產出 code。證明靠 fallback——scalar emulation 永遠存在。

**Termination caveat**：paper 誠實寫出來，完全 complete 的 saturation 本質上跟 halting problem 等價，所以實務上靠 timeout bound e-graph saturation iteration。這點**跟 egg / egglog 的做法一致**。

這個 soundness guarantee 是 production compiler 願意採納的關鍵——比起 LLM-generated kernel（像 TAIC），ACT 給的是形式保證，而不是「我們跑了 1024 random input 都對」。對 safety-critical 應用（自駕、醫療 AI），這條差距是 blocker 等級。

---

## 4. 實驗：ACT 跟誰比、贏多少

### 4.1 Target coverage

六個 target 分兩類：

**商用**：

- **AWS Trainium**（NKI ISA）：對 neuronx-cc；
- **Intel AMX**：對 oneDNN hand-tuned；
- **Google TPUv1**：對 XLA:TPU 的 pre-Pallas 階段。

**學術**：

- **Gemmini**（MIT）：對 Gemmini 自家 compiler + hand-tuned ONNX runtime；
- **FEATHER**：對 paper 自帶 ad-hoc code generator；
- **EVA**（homomorphic encryption accelerator）：對 Microsoft EVA compiler。

六個都是真正產品 / 真正發表的 ISA，不是 toy。

### 4.2 關鍵數字

**Code generation coverage**（能編到的 operator 比例）：

- AWS Trainium：ACT **2.3×** neuronx-cc 的 coverage。這條最震撼——neuronx-cc 是 AWS 自己為自己家 ASIC 寫的 production compiler，公司規模 team × 幾年工程量，ACT 用 TAIDL 描述 500–800 行 Python 外加 ACT 自動生成，直接在 coverage 上贏 2.3×。
- Gemmini：ACT 比 Gemmini 自家 compiler 多覆蓋顯著 operator（paper 具體說有些 fused op 之前沒人寫，ACT 自動 fuse 出來）。

**Performance**（跟 hand-written baseline 比）：

- 對 kernel library：**最高 6.88× speedup**，中位數在 kernel-by-kernel 層面接近持平或小勝；
- Gemmini operator fusion：**1.7×** 性能 / 資料搬運 **–50%**；
- 跟 neuronx-cc 在相同 kernel 上大致持平（ACT 的賣點是 coverage 不是單 kernel peak，但沒輸）。

**Compilation overhead**：

- Typical kernel：**< 100 ms**；
- Large kernel（390 HLO node）：**529 ms**。

這條在 production 很重要——如果 ACT 編譯一個 batch job 要 10 分鐘，JIT 跑 ML inference 的場景會崩掉。< 1 s 是「可以塞進 CI 跟 JIT path」的區間。

### 4.3 Backend 工程量：多久寫完一個新 target？

Paper 給了一個隱藏殺手級的數字：**在新 ISA 上從零開始到可用 backend，平均 engineer-days 等級**（paper 原文："once-per-accelerator effort of writing TAIDL description"）。

對比 neuronx-cc 這種 production compiler 的開發週期（AWS Neuron SDK 1.0 公開發表是 2020，到 2026 已經六年），這個 100× engineer-day 的壓縮不是「寫得快」，是**整個 bill-of-materials 改寫**。

---

## 5. 給 Adam 的 compiler career 訊號

### 5.1 這篇把哪三份工作「吃掉」

對 Adam 的 Nvidia compiler 路線（參考 `~/dev/career/4-Learning/Compiler-Path.md`），這篇改寫的是**「給 ASIC 寫 backend」這個 job family 的職能分佈**：

| 既有職能 | ACT 時代怎麼變 |
| --- | --- |
| 寫 instruction selection pass | **ACT 自動產**，個別 target 不再需要 |
| 寫 scratchpad allocation / spill | **ACT 自動產**，個別 target 不再需要 |
| 寫 cost model + autotuner | 還需要（Phase A 的 extraction cost function + Phase B 的 ILP objective），但是 generic 的、不是 target-specific |
| 寫 ISA formal spec（TAIDL） | **新職能**，而且是 bottleneck |
| 證明 TAIDL 寫得對（test oracle 比對 silicon） | **新職能**（硬體 verification + compiler 兩個 skill 都要） |
| 處理 ACT 搞不定的 corner | **新職能**（當 ACT 覆蓋不到某個 instruction，需要人工 inject pattern） |

**位置整個往上移了一層**——從「對一個 ISA 寫一整套 backend」變成「對一整類 ASIC 設計共用的 compiler 框架 / spec 語言 / 自動化工具」。

這跟 LLVM TableGen 當年對 scalar compiler engineer 的衝擊是一樣的——TableGen 吃掉了「手工寫 instruction selection」的大部分，但 LLVM backend engineer 沒消失，只是變成「寫 target description、寫 custom lowering pass、處理 TableGen 搞不定的 corner」。

### 5.2 Nvidia 的位置：為什麼 Krut Patel 掛名

這點值得停下來看：**論文的三位作者有一位來自 NVIDIA**（Krut Patel）。這代表：

- NVIDIA 不是把 ACT 當敵對研究看，而是**主動參與**；
- NVIDIA 的 compiler stack（CUDA、PTX、cuDNN、cuBLAS、Triton、CUTLASS）本身沒有在這篇裡被 target 掉，但 NVIDIA 在觀察這個方向；
- 給 compiler engineer 的職缺訊號：**TAIDL-style 的 formal ISA spec 工程會成為 Nvidia internal 的 preferred skill set**，特別是 Blackwell 之後 ISA complexity 繼續攀升（tcgen05、tensor memory、cluster-level primitive）。

Adam 要往 Nvidia compiler 走的話，建議三個 take-away：

1. **讀 TAIDL + ACT 的 code**（Apache-2.0，Python-based，整個 repo 1.5 萬行內可讀完）；
2. **把 TAIDL 試著套在你熟的東西上**——例如 CUDA PTX 的某個 subset（tcgen05 instruction family）、或者 Hexagon NPU（有公開 ISA）、或者 Gemmini（academic ISA 完全開源）；
3. **把這篇跟上週 TAIC、兩週前 Tiny-GPU-Compiler 連起來讀**。這三篇描述的是同一件事的三個切面：「backend 工程的解構」。

### 5.3 什麼時候 ACT 會失效

誠實評估論文的 limitation：

- **TAIDL 不支援 instruction 內部帶 control flow**。這條限制排除掉：GPU 的 sync primitive（barrier、fence、mbarrier）、CPU 風格的 branch / predicate、homomorphic encryption 的 bootstrap。EVA 能跑是因為 bootstrap 被當 external 處理，不是 TAIDL instruction。
- **Shape 必須靜態**。dynamic shape 要靠 StableHLO 上游 pass 做 shape specialization，ACT 本身不解。
- **E-graph saturation 的 timeout 是 soft bound**。某些 pathological 的 kernel（深度 nested reduction）可能跑到 timeout 都還沒 saturate，就 fallback 到 greedy 選擇，性能會塌。這條 paper 用「典型情況 < 100 ms」繞開，但 production 情境下 pathological case 的 fallback 行為沒公開 benchmark。
- **沒 target NVIDIA / AMD 的 GPU ISA**。Paper 選的六個 target 都是 explicit scratchpad 的 systolic / dataflow accelerator，GPU 這種 implicit cache + warp-level ISA 的架構還沒試過。這是個開放問題，也是 Nvidia 掛名的隱藏動機之一。

**這個 limitation list 很關鍵**——它定義了 ACT 時代**「哪些 compiler 工作還是人在寫」**：GPU、CPU、含 control flow 的 ISA、dynamic shape heavy 的 workload，這些都還留在 compiler engineer 的工作範圍內。

---

## 6. 跟近期三篇的關聯

把這篇放進本站近兩個月的 compiler 系列裡：

| 文章 | 日期 | 中心論點 |
| --- | --- | --- |
| [Tiny-GPU-Compiler](tiny-gpu-compiler-mlir-verilog-gpu-16bit-isa-education-2026.md) | 9/24 | ISA + compiler + Verilog 一人 end-to-end 可行，教學用但 pattern 有啟發 |
| [TAIC: AI as a Compiler](ai-as-compiler-taic-triton-ptx-bitdelta-volta-verifier-2026.md) | 10/2 | LLM 可以繞過 Triton 直接 lower 到 PTX，寫 verifier 是未來 compiler engineer 的 moat |
| **ACT（本篇）** | **10/4** | **ISA → backend 可以程式化，compiler engineer 的位置往上移到 spec 層** |

三篇都在描述同一個趨勢：**compiler stack 的「中間工程量」被壓縮**。TAIC 從上游（kernel source）攻擊、ACT 從下游（ISA spec）攻擊、Tiny-GPU-Compiler 證明三件事可以用小團隊整合——結論是**compiler engineer 的高價值位置會更集中在上下游兩端**：

- 上游：formal spec、semantic verification、cost model、搜尋演算法；
- 下游：硬體 micro-architecture 感知的 pattern、新 instruction family 的描述、ASIC 的 pre-silicon tooling。

而「中間」——寫 N 個 target 的 N 套 lowering pass——會持續被工具吃掉。

---

## 7. 本週要讀的三份東西

1. **ACT paper** ([arXiv 2510.09932](https://arxiv.org/abs/2510.09932))：先讀 Section 2（motivating example）跟 Section 4（pii + α/β 分離）。Section 5 的 soundness 證明可選讀。
2. **TAIDL paper**（OOPSLA'26，DOI `10.1145/3725843.3756075`）：讀 Section 3 的語言核心構造。Section 6 的 test oracle 延伸閱讀。
3. **Repo**：[github.com/act-compiler/act](https://github.com/act-compiler/act)。先跑 `examples/` 下的 Gemmini backend 生成（硬體上 open-source，可用 Spike 之類 simulator 跑），把一條 instruction 從 TAIDL 寫起、看 ACT 怎麼吐出 Phase A rewrite rule + Phase B constraint。

---

## 8. 一句結論

**ACT 不是「又一個 compiler」，是把 compiler construction 自己做成一條 pipeline**。它改寫的是 ASIC compiler 工作的 bill-of-materials——TAIDL 描述能寫、XLA 整合能編、soundness 能證——而這條 pipeline 一旦跑通，未來三年要在這行吃飯的 compiler engineer，技能樹的重點就會從「寫 pass」偏向「寫 spec、寫 verifier、設計 search / cost model」。

Adam 在 Compiler-Path 四月的 learning plan 裡列的 MLIR / LLVM / Triton，都還是必修——但往 Nvidia compiler 投的時候，**TAIDL + ACT 這一類「compiler-as-infrastructure」研究是能明顯拉開面試差距的加分項**。因為它證明你讀的是 2026 下半年的 state-of-the-art，而不是 2022 的教科書。

> 下一篇預計跟 ACT 做對照——如果有時間，週末會挑一個小 ISA（可能是 Gemmini 或 Hexagon 子集）試著用 TAIDL 寫一個 toy spec，跑一次 ACT 看看實際產出長什麼樣。那篇會是 hands-on 經驗分享。
