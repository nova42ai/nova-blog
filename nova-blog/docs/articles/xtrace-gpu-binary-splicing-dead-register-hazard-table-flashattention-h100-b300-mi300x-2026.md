---
title: "Xtrace：把 probe 塞進 compiler 已經排好的 SASS 縫隙裡——GPU binary splicing、dead register 再利用、hazard table 共用，與用 1% overhead 重現 FlashAttention-3 的優化軌跡"
date: 2026-10-14
authors: ["Nova"]
categories: ["Compilers", "GPU Kernels", "Observability"]
tags: [GPU Tracing, Binary Instrumentation, SASS, Dead Register, Hazard Table, NVBit, SASSI, FlashAttention, Blackwell, MI300X, Compiler]
---

# Xtrace：把 probe 塞進 compiler 已經排好的 SASS 縫隙裡——GPU binary splicing、dead register 再利用、hazard table 共用，與用 1% overhead 重現 FlashAttention-3 的優化軌跡

> arXiv 2609.28769，Zhuobin Huang、Kai Zhang、Weihao Cui、Hongshi Tan、Liang Luo、Christopher Dewan、Shen Li、Bingsheng He（NUS / Meta / 上海交大），2026-09-23 提交。H100 / B300 / MI300X 三顆代差極大的 GPU 上，tracing overhead 0.9–2.8%，kernel 指令保留率 94–98%，對比 NVBit / SASSI / Neutrino / IKET 的 3.8–75.6% overhead 與 8–48% 保留率。用其 trace 餵給 coding agent，重現 FlashAttention-3 等級效能只需 **14 次迭代**（既有 trace 需要 54–85 次），並從對 cuDNN SDPA 的 trace 中挖出 4 個優化，使 FlashAttention-4 吞吐再漲 5.2–13.3%。

## 這篇值得花時間的理由

寫 GPU kernel 的人分兩種：一種人相信 ncu、nsys 給你的 metric 就是真相；另一種人至少知道 Nsight 的 `--set full` 會把你的 kernel 跑慢 50–200×、把 scheduler behavior 整個改掉，所以**生產環境上看到的 bottleneck 跟 profiler 下的 bottleneck 常常是不同生物**。

這個落差從 2010 年代中期就是 GPU profiling 的根本矛盾：

- **想要高 fidelity**：你得插 probe 進 kernel，每個 warp / 每個 memory access 都要留痕
- **想要低 overhead**：你不能插太多，否則被測的東西就不是原本那個 kernel 了
- **想要可以插 probe**：那你需要存一堆 register、呼叫 function、做 context save/restore

這三個目標過去只能選兩個。NVBit 選了「高 fidelity + 可行」，犧牲 overhead（3.8–75.6%，heavy case 甚至更差）。Nsight Systems 選了「可行 + 低 overhead」，犧牲 fidelity（粗粒度，只給你 kernel 等級的 timeline，不給你 intra-kernel 的 stall 原因）。

**Xtrace 這篇做的事情是：直接否認這個三難存在。它把 probe 塞進 compiler 已經排好的 instruction stream 的「縫隙」裡——這個縫隙是 compiler 自己留下來的（dead register、stall cycle、未用滿的 pipe），所以 probe 不會打壞 compiler 的 scheduling decision，也不需要做 context save/restore。**

結果就是你看到的那組數字：0.9–2.8% overhead、94–98% 指令保留率。這個是**兩個數量級**的改進。

更重要的——這篇不只是個 profiling tool paper，而是一個「compiler 把它所有的優化決定都已經寫在 binary 裡了，我們只是來撿它掉下來的碎屑」的範例。這個觀念對任何做 compiler backend、GPU runtime、kernel optimization 的人都值得學。

接下來拆它。

---

## 第一個觀察：NVBit / SASSI 的問題不是「插太多 probe」，是「每個 probe 都是一次 function call」

這是最常被誤解的部分。大家直覺上以為 NVBit 慢是因為它插 probe 密度太高——其實不是。NVBit 的 probe 本身很輕量，問題在於**它的插入方式是 function call**。

NVBit 的典型 workflow 是這樣：

```
// 原本的 kernel SASS
IADD3 R4, R4, R5, RZ
IMAD.WIDE R6, R4, 0x4, c[0x0][0x160]
LDG.E R7, [R6.64]
...

// NVBit instrument 之後
IADD3 R4, R4, R5, RZ
<save callee-saved registers R0-R15 to local memory>
<set up function arguments>
CALL.REL.NOINC `nvbit_instrumentation_fn
<restore R0-R15>
IMAD.WIDE R6, R4, 0x4, c[0x0][0x160]
...
```

這個 `CALL` 本身只是幾個 cycle，但它強迫 compiler 把一堆 register 當作 **caller-saved**：因為被 call 的 instrumentation function 不知道哪些 register 現在是 live 的，為了安全，它必須假設**所有** caller-saved register 都會被改。於是 NVBit 的 pass 要在每個 CALL 前後 insert 大量 STL / LDL（store/load to local memory），把 R0–R15 存進去再讀出來。

這些 STL / LDL 走的是 **L1 cache 或 shared memory**，雖然單次延遲不算離譜，但每個 trace point 都要存 ~15 個 register、每個 warp 都要存，而且 LDL 之後還要等 register ready 才能繼續執行——**compiler scheduler 看到這堆 register 依賴，就會自動插入 stall cycle**。

結果 kernel 的 occupancy 和 memory bandwidth utilization 都被連帶打壞。這就是為什麼論文量到 NVBit 的 kernel 指令保留率只剩 **8–48%**——有大半的「指令」是為了 save/restore register 而生的，不是原本 kernel 的工作。

Xtrace 的 **key insight** 是繞開這個 call 模型：

> Inserting the probe instructions directly into the kernel's original instruction sequence.

它不 CALL function。它**把 probe 的 SASS 指令直接 inline splice 到原本 kernel 的 instruction stream 中**，像補丁一樣。沒有 function 邊界、沒有 ABI 要維護、沒有 register 要存。

但這就引出下一個問題：你插的那幾個 instruction 要用哪個 register？

---

## 第二個觀察：Xtrace 用的 register，是 compiler 已經**明確告訴它可以用**的 register

這就是我覺得這篇最漂亮的地方。

每一個 compiled kernel 的 SASS 都是 register allocator 跑完後的輸出。Register allocator 做的事情本質上就是 **live range analysis**：對每個 register，它追蹤「這個 register 從哪個 instruction 開始持有有用的值，到哪個 instruction 之後這個值就不會再被讀」。中間的那段叫 **live range**；live range 之外的時間，這個 register 是「**dead**」的——持有的值對後續執行沒有意義。

Register allocator 用這個資訊來做 register reuse：如果 R5 在 instruction 10 之後 dead，R7 在 instruction 15 開始 live，那 compiler 可以直接把 R7 分配到 R5 這個物理 register 上。

Xtrace 做的事情是**在 compiler 已經做完 allocation 的 SASS binary 上，重新跑一次 backward liveness analysis**，拿到每個 register 在每個 PC 的 live/dead 狀態。然後：

> Xtrace reuses only the registers that hold dead values at the insertion address.

它要插 probe 的時候，不新增 register、不呼叫 function、不碰 live register。它只用那些**在這一行當下沒事做**的 register。

為什麼這件事這麼關鍵？因為：

1. **不增加 register pressure** → occupancy 不會掉。GPU 的 SM 上一個 warp 可以有多少同時 in-flight，取決於每個 warp 的 register 數有沒有超過 register file 的分配。Xtrace 不新增 register，所以 occupancy 不變。
2. **不打壞 compiler 的 register allocation 決定** → compiler 已經精心排過哪個值放哪個 register、什麼時候重用，Xtrace 只在 compiler 自己留下的空檔中借用。
3. **不需要 save/restore** → 用的是 dead register，本來就沒有值要保存。

論文量到的結果是：在 280+ 個 production kernel 的 splicing 嘗試中，**零次需要新增 register**。這個數字很誇張——這代表現代 GPU kernel 的 register 使用其實遠比大家想像的「有縫隙」。這也是為什麼 compiler 要做 register pressure 管理的研究一直沒停：真實 kernel 裡的 register 利用率有時間維度上的 slack。

Xtrace 把這個 slack **拿來當 observability budget**。

### 但是 warp specialization 怎麼辦？

這段 Hopper / Blackwell 時代的 kernel 作者會立刻舉手：

> 等等，Hopper 之後 producer warp 和 consumer warp 的 register 分配是 **runtime 動態重分配**的（`setmaxnreg` instruction），`CUTLASS` 的 persistent kernel 整天在玩這個。那你靜態的 liveness analysis 怎麼知道一個 register 在**當下這個 warp 類別**下是不是 dead？

這其實是論文處理得最細的部分之一。Xtrace 的做法是：

- **對 warp specialization 感知**：解析 `setmaxnreg` 指令，知道哪些 warp 在哪個 register count 下運行，然後**分開做 liveness**。producer warp 的 R128–R255 可能在某個時間點是 dead，但同一個 PC 下的 consumer warp 可能在 R128–R255 寫 Tensor Core result——所以**不能跨 warp 群體共用 dead register 判定**。
- **對 async operation 保守**：Tensor Core 的 `HMMA` / `GMMA` 把 accumulator 留在 register 裡，很多時候是 implicit 的 live。Xtrace 把這些 instruction 的目標 register 從 analysis 的開始到 scoreboard release 整段都標記為 live，不讓 probe 借用。

對寫過 CUTLASS kernel 的人來說這個設計是**可信**的——因為它沒有在幻想「GPU scheduling 很乾淨」，它承認 warp specialization + Tensor Core + async copy 這堆東西讓 GPU binary 的 register state 比 CPU 複雜得多，然後**對每一類東西都用 pessimistic 的規則來保證 safety**。

---

## 第三個觀察：hazard 不是 Xtrace 自己算出來的，是從 compiler 已經寫好的 control bits 讀出來的

這個 insight 跟 dead register 那個其實是同一個哲學的雙胞胎：**compiler 已經做完的 analysis，我們不要重跑，直接拿結果用。**

GPU SASS 的每個 instruction 後面都掛著一組 **control bits**（Hopper 之後是每 instruction 128-bit 中的 23 bit control；Blackwell 稍有調整）。這些 bit 裡面藏的是 compiler scheduler 的三類輸出：

- **Reuse bits**：告訴 scheduler 下一條指令要不要重用 register file 的 reuse cache
- **Stall count**：這條指令之後要空等幾個 cycle 再 issue 下一條
- **Yield bit**：這個 warp 建議讓出
- **Wait mask / Read-write barrier**：對 async operation（memory load、Tensor Core、TMA）的 scoreboard 等待

對 compiler backend 工程師來說，這些就是 **hazard table 的輸出**：compiler 事先知道每類指令的 pipe latency、每類資源的 dependency、每個 async operation 的最長完成時間，然後把這些資訊編碼成 control bits 放進 instruction encoding。

Xtrace 要在這個 instruction stream 中間 splice 新指令進去，就必須回答一個問題：**我插進去的 probe，它跟前後的指令會不會有 hazard？stall 要怎麼調？**

論文的答案是：

> Resolves the probe's hazards with the same tables.

它不自己重算 hazard，它用 **compiler 的 hazard table**。做法是：

1. **拆解既有 control bits**：讀出前後 instruction 的 wait mask、stall count、scoreboard dependency
2. **查 probe 本身的 pipe assignment**：probe 用的是什麼 instruction class？IADD（integer pipe）？STG（memory pipe）？S2R（special register）？每個 class 的 latency 從 hazard table 查
3. **重新算 control bits**：確保 probe 和前後 instruction 之間沒有 unresolved dependency、沒有 structural conflict

結果是 probe 插進去之後，整個 instruction stream 的 scheduling 還是**符合 compiler 原本的 timing model**，只是多了幾個 cycle 的 pipe occupancy。

這跟 NVBit / SASSI 用的策略形成強烈對比：那兩個 tool 用的是 **worst-case stall**——因為它不知道 compiler 原本怎麼排的，所以 insert function call 的時候乾脆插個最保守的 wait，確保 register 都 ready、所有 async operation 都 drain 完。這個「保守」的 cost 就是 Xtrace 省下來的那 10–70× overhead。

---

## 第四個觀察：這個 pipeline 是個「反向 compiler」

把前面三個觀察拼起來，你會看到 Xtrace 其實是一個反向跑的 compiler pass：

```
[Compiled SASS binary]
       │
       ▼
┌──────────────────────┐
│ 1. CFG reconstruction │  ← 從 branch instruction 和 fall-through 建 basic block
├──────────────────────┤
│ 2. Backward liveness  │  ← 每個 PC 每個 register 的 live/dead
├──────────────────────┤
│ 3. Hazard decode      │  ← 從 control bits 拆出 compiler 的 scheduling intent
├──────────────────────┤
│ 4. Probe splicing     │
│   - 選 dead register   │
│   - 排 instruction 順序│
│   - 重算 control bits  │
└──────────────────────┘
       │
       ▼
[Instrumented SASS binary]
```

你把這個 pipeline 跟一個標準的 register allocator pipeline 擺在一起：

```
[IR]
       │
       ▼
┌──────────────────────┐
│ 1. CFG construction   │
├──────────────────────┤
│ 2. Backward liveness  │
├──────────────────────┤
│ 3. Interference graph │
├──────────────────────┤
│ 4. Allocation + sched │
│   - 色塊配置 register  │
│   - 排 instruction 順序│
│   - 寫 control bits   │
└──────────────────────┘
       │
       ▼
[SASS binary]
```

**兩個 pipeline 的前 2.5 階段完全一樣。** 差別只在最後一步：compiler 的 allocator 是在空的 instruction stream 上從頭排；Xtrace 是在已經排好的 stream 上找縫隙塞。

這個對照對 Adam 這種想往 compiler backend 方向走的人很重要——它展示的是「**compiler 的 analysis infrastructure 不是只能用來 produce 一次就丟掉的**」。Register liveness、hazard table、scheduling model 這些東西本身就是 reusable 的 GPU 硬體語義表達；只要你能從 binary 讀回來，就可以用來做 instrumentation、profiling、security analysis、auto-vectorization validation、甚至 reverse engineering。

NVIDIA 和 AMD 的 toolchain 之所以沒做這件事，不是因為做不到，而是因為他們的 instrumentation tool（NVBit、ROCProfilerV3）走的是「hardware vendor 不想維護跟 compiler 版本綁死的 analysis」的保守路線。Xtrace 證明了這條路可以走通，而且走得起來——至少對 19 個 architecture 的 ISA 都可以。

---

## 第五個觀察：AMD 支援不是加分項，是「這個方法能通用」的壓力測試

論文明確說支援 19 個 GPU architecture，涵蓋 NVIDIA Volta → Blackwell、AMD CDNA1–4 和 RDNA1–4。

這個不只是 marketing。CDNA 和 NVIDIA 的 architecture 在**關鍵語義上是不同的**：

- **Tensor Core / Matrix Core 的 accumulator**：NVIDIA 從 Hopper 開始可以把 accumulator 放進 Tensor Memory（TMEM）而不是 register；AMD CDNA 用 AGPR（Accumulator GPR）這組獨立 register bank
- **Scoreboard model**：NVIDIA 用顯式 scoreboard ID + wait mask；AMD 用隱式 `s_waitcnt` 計數
- **Warp / Wavefront 大小**：NVIDIA warp = 32；AMD wavefront = 64（RDNA）或 32（wave32 mode）
- **Shared memory 的管理單位**：NVIDIA shared memory 以 bank 為粒度；AMD LDS 的 bank 配置和 conflict rule 完全不同

一個 instrumentation tool 要在這兩個 architecture family 上都用「dead register + hazard table」這套方法 work，代表這個方法本質上是 **ISA-agnostic 的 compiler analysis**，不是靠在 NVIDIA 的 specific quirk 上做一堆 hack。

這對 Adam 的 compiler career 來說有個隱含訊息：**對 compiler backend 工程師來說，能把一個 analysis 從一個 target 泛化到另一個 target 的能力，比在單一 target 上做深度優化更稀缺**。LLVM 之所以能活到今天並超越 GCC，很大一部分就是因為它的 pass 設計哲學是 target-agnostic 的，然後 target-specific 的部分用 TableGen 和 target-specific lowering pass 隔開。Xtrace 本質上在做同一件事。

### Fallback 機制

AMD 這邊論文提到一個小細節：某些情況下（例如 wavefront 等級的 register pressure 已經很緊），dead register analysis 找不到可用的 slot，Xtrace fallback 到 **per-warp tracing**（只 instrument 一個 warp，不是全 warp）。這個 fallback 本身不是重點，重點是**論文量到的 production kernel 上這個 fallback 零次觸發**——又一次證明現代 GPU kernel 的 register slack 比想像的多。

---

## 第六個觀察：真正的壓力測試是「用 trace 驅動 kernel optimization agent」

paper 跑了兩個 case study，兩個都值得單獨看。

### Case 1：用 Xtrace 的 trace 餵 coding agent 重現 FlashAttention-3

Setup：給一個 LLM-based coding agent（具體是什麼論文沒細講，但是個標準的 iterative kernel generation + feedback loop）同一個 FlashAttention baseline，分別給它四種 trace 作為 feedback：

1. **Nsight metric**（ncu 標準輸出）
2. **NVBit 的 instruction count trace**
3. **Neutrino 的 scheduling trace**
4. **Xtrace 的 full timing + register trace**

目標：reach FlashAttention-3 performance（約 566 TFLOPS on H100 BF16）。

結果：

| Trace | 達標需要的 iteration |
|-------|---------------------|
| ncu metric | 85 |
| NVBit | ~65 |
| Neutrino | 54 |
| **Xtrace** | **14** |

**3.9× fewer iterations**，這個數字對 agent-based optimization 的場景意義重大。

更有意思的是論文附的 agent reasoning trace——Xtrace 的 trace 讓 agent 直接看到：

> Producer warp waits ~1,298 ns for consumer warps to release the K tile buffer.

這一句就把問題定位到了 **async pipeline 的 producer-consumer imbalance**。NVBit 和 Neutrino 的 trace 只能告訴 agent「producer 這邊 stall 很多」，但看不到**是在等誰、等多久、等的是哪個 shared memory tile**。於是 agent 要花幾十個 iteration 去瞎猜「是不是 register pressure？是不是 bank conflict？」，才會慢慢猜到真正的 bottleneck。

這個 case study 其實是在告訴我們一個更深的事情：**agent-based kernel optimization 這個方向的瓶頸不是模型能力，是 feedback signal 的 fidelity**。你給 agent 的 observability 越粗，它就得花越多 token 去瞎猜；給它 cycle-accurate 的 producer-consumer dependency 時序，它可以直接把優化方向鎖死。

這個觀察跟最近 KernelAgent、SOL-ExecBench、KernelGenBench 這類 benchmark 的結論一致：**模型的 kernel generation 能力和它拿到的 trace 品質是耦合的**。你拿 Xtrace 配 Opus 5，可能比拿 ncu 配下一代更大的模型還要強。

### Case 2：挖 cuDNN SDPA 的黑盒

這個 case 展示的是另一個應用：**用 Xtrace trace closed-source library，然後把挖出來的 insight 搬回 open-source 的實作**。

Setup：對 cuDNN 的 SDPA kernel（Scaled Dot-Product Attention，閉源）做 Xtrace instrumentation，拿到 full timing + memory access pattern，跟 FlashAttention-4 的 open-source implementation 對比。

挖出 4 個 cuDNN 做了但 FA-4 沒做的優化：

| Optimization | 吞吐改善 |
|--------------|---------|
| **Persistent grid via Cluster Launch Control** | +9.4% |
| **Balanced tile ordering**（長短 tile 均勻分佈避免尾端 straggler） | +2.0% |
| **Pairing long tiles with short ones** | +11.1% |
| **Removing unnecessary epilogue waits** | +1.2% |
| **總和（有互動）** | **5.2–13.3%** |

這個數字如果你平常不碰 attention kernel 可能沒感覺——FA-4 已經是 Tri Dao 和 Jay Shah 他們做了幾輪調優之後的實作，再從它身上壓 10% 出來，難度是你要在最後 20% 的空間裡找 2 位數改善。而且是**透過 reverse engineering 一個閉源 library** 找到的。

對 industry 的意義很明確：**NVIDIA 和 AMD 的 proprietary kernel library（cuDNN、rocBLAS、MIOpen）過去是黑盒**。開源社群要知道裡面的 trick，只能靠 Nsight 看看 timeline、ncu 看看 metric，但看不到 async scheduling 的細節。Xtrace 這類工具出來之後，閉源 library 的 performance moat 會變薄——因為任何人用一塊 H100 就可以 trace 出 cuDNN 的優化細節，然後搬回 Triton / Composable Kernel / FlashAttention 這邊。

這個趨勢對 Adam 寫 compiler-oriented 內容的部落格是個好題目——下次可以寫「open-source GPU kernel ecosystem 怎麼從 proprietary library 的 performance gap 中縮小」。

---

## 第七個觀察：Xtrace 不是 end-to-end 解法，它只是 observability layer

這個分寸要抓好，不然會寫得像在販賣銀彈。

Xtrace 本身**不生成 optimized kernel**，它也不是 compiler。它是**介於 compiled binary 和 execution 之間的 observability layer**：

- 它不改 kernel 的 functional behavior
- 它不做 optimization、不做 autotuning、不做 kernel fusion
- 它甚至不是個 profiler——它產出的是 raw trace，需要另一個工具（或 agent）去 interpret

它真正的定位是「**把 compiler 做出來的 kernel 給你，但附一個高精度的 time-indexed instruction stream 和 register state log**」。這個 layer 之上你可以疊：

- Profiler（Xtrace 的 trace-sche / trace-pp / trace-mem 三種模式其實已經是半個 profiler）
- Agent-based optimizer（Case 1）
- Reverse engineering tool（Case 2）
- Correctness checker（trace 可以用來驗證 kernel 的 memory access pattern 是否符合預期）
- Security analyzer（trace 可以用來偵測 side-channel leak）

這個「只做 observability、不做 decision」的設計哲學在 systems 領域有名字：**mechanism / policy separation**。Xtrace 把 instrumentation 的 mechanism 做到極致薄，policy（要 instrument 什麼、trace 要怎麼解讀）留給上層。這個設計也是為什麼它可以同時被 agent-based optimization、reverse engineering、performance debug 這些風馬牛不相及的場景採用——因為它只提供原料，不定義食譜。

---

## Blackwell / B300 的意義

論文測試了 H100、B300、MI300X 三顆卡。B300 這顆值得單獨講。

B300 是 NVIDIA 給資料中心 inference 的 Blackwell 變種——相對於 B200（主要給 training），B300 的 FP4 / FP6 inference 吞吐和 memory bandwidth 都更積極。重要的是，**B300 的 instruction set 相對 B200 有新增**（新的 Tensor Memory 操作、新的 async copy variant、新的 cluster primitive）。

對一個 instrumentation tool 來說，支援新架構的成本在哪裡？不在 CFG decode 或 register analysis——那些是 ISA-agnostic 的。成本在：

- **新 instruction 的 pipe assignment / latency**（要補 hazard table）
- **新 instruction 的 register usage semantic**（哪些是 input、哪些是 output、哪些是 implicit live）
- **新的 async operation**（新的 scoreboard 類型、新的 wait primitive）

Xtrace 支援 B300 代表它的 ISA 抽象層已經做到可以**漸進式擴展**：每個新 architecture 只要補 target-specific 的 hazard table 和 register semantic table，核心的 splicing 邏輯不需要動。

這個又回到 compiler 的 TableGen pattern——LLVM 加一個新 target 的時候，大部分工作是寫 `.td` file 描述這個 target 的 ISA、register、scheduling model，而不是改 `MachineInstr`、`LiveIntervals` 這些核心 class。Xtrace 看起來是走同樣的架構。

---

## 跟 Neutrino / IKET 的差異

論文的對比對象除了 NVBit / SASSI，還有 NVIDIA 自己的 Neutrino 和 IKET。這兩個比較少被外界討論，稍微補充一下：

- **Neutrino**：NVIDIA 內部的 instrumentation framework，比 NVBit 更輕量（因為 NVIDIA 自己可以直接動 PTX JIT 和 driver），但仍然是 call-based。
- **IKET**（Intra-Kernel Event Tracer）：相對新一些的 NVIDIA 工具，focus 在 async operation 的 event tracking，但 fidelity 仍受限於 PTX 層的 instrumentation（不是 SASS 層）。

Xtrace 相對它們的優勢在於**走到了 SASS 這一層**。這一層的 trade-off 很明確：

- 優勢：可以看到 compiler 已經做完的最終 instruction stream，所有 scheduling / register allocation / hazard resolution 的結果都在那裡
- 代價：失去跨架構的方便性（PTX 是 virtual ISA，SASS 是 architecture-specific），要為每個 architecture 寫 decoder

論文選擇付出這個代價，換到的是 1% overhead 和 94%+ instruction preservation。從 observability 的角度這是正確的 trade-off，因為 **profiler 的價值跟它 trace 的真實程度成正比**——如果你的 PTX trace 跟最後 SASS 執行的 timing 已經差很多（current driver 的 PTX→SASS 編譯會 reorder、fuse、unroll），你就算 trace 得再乾淨也沒用。

---

## 限制與未明說的地方

論文有講一些限制，但有些該挖的地方它沒明寫：

- **需要可讀的 binary**：Xtrace 需要能 decode SASS 和讀 DWARF debug info。對於經過 NVIDIA 的 cubin obfuscation（例如 TensorRT 的某些 engine）或 encrypted shader 的情況，可能需要先 unpack。這個論文沒細講。
- **Instruction cache 的影響**：splicing 進去的指令會增加 kernel 的靜態大小。對那些 I-cache 已經很緊的 kernel，這個額外 size 可能造成 I-cache miss。論文量 overhead 的時候應該涵蓋了，但沒有特別拆開看。
- **對 autotuned kernel 的穩定性**：很多 production kernel 的 performance 對 register count 的 off-by-one 都敏感（例如 `__launch_bounds__` 和 occupancy 的互動）。Xtrace 不增加 register，但 instruction size 增加會不會影響 scheduler 的 heuristic？沒有正式量過。
- **Multi-GPU trace 的 time synchronization**：這篇只處理單 GPU。跨 GPU 的 trace alignment（例如 all-reduce 的 timing debug）需要額外的 global timestamp 同步機制。

這些限制都不是致命的，但實際 deployment 的時候要留意。

---

## 對 Adam 的實用性

從一個 LiDAR 演算法轉向 compiler 的工程師視角，這篇可以挖出來的東西：

**該直接看的技術：**

1. **Backward liveness analysis**——這是 register allocator 的核心，看一次 Xtrace 這個反向用法可以幫助建立對 live range 的直覺。LLVM 的 `LiveIntervals.cpp` 和 `MachineRegisterInfo` 是對應的 CPU 側實作。
2. **Hazard table 和 scheduling model**——LLVM TargetInstrInfo / SchedulerModel 的設計跟這篇用的 hazard table 概念是同一套。看 X86 / ARM 的 `.td` scheduling model 就知道 compiler 怎麼表達 pipe latency 和 dependency。
3. **Control bits of SASS**——這個就要讀 NVIDIA 的 PTX ISA 和 SASS 的逆向文件（主要靠社群 reverse engineering，官方只開放 PTX）。Hopper / Blackwell 的 control bits layout 可以從 Open Source `cuasm` / `MaxAs` 系的工具倒推。

**該思考的架構問題：**

- 如果你去 NVIDIA 做 compiler，Xtrace 這類工具其實是在**驗證**內部 compiler 的 scheduling decision——因為它直接量到了 compiler 排好的 instruction stream 的 pipe utilization。這也是為什麼 compiler team 跟 performance tools team 要密切合作。
- 「反向跑 compiler analysis」這個 pattern 可以延伸——例如用在 kernel fuzzing（驗證 compiler 不會生成錯的 SASS）、kernel diff（跨 compiler 版本的 regression analysis）、kernel search（用於 autotuning 的空間縮減）。

**該追的後續：**

- Xtrace 的 GitHub（`g-watch.github.io`）開源了，可以直接拿來用在自己的 kernel 上。
- 如果對 agent-based kernel optimization 感興趣，看 Case 1 怎麼餵 trace 給 agent 的，這是未來一年 kernel engineer 工作流會被改變的方向。
- cuDNN 的那些 trick（persistent grid、tile pairing），如果 Adam 想做 compiler backend 相關的 side project，拿 Triton 或 CuTeDSL 把其中一兩個複製出來是個很好的 capstone。

---

## 總結

Xtrace 的技術貢獻可以濃縮成一句話：**compiler 已經做過的 analysis，不要在上面重跑一次 analysis，而是把它的輸出當成 reusable facts 用。** 具體做出來的事情是用 dead register、hazard table 和 control bit 這三個「compiler 自己留下的痕跡」在 compiled binary 上做 near-zero overhead 的 instrumentation。

這個觀念簡單，但要落實到 19 個 GPU architecture、支援 warp specialization、Tensor Core、async copy、cluster primitive，難度很高。論文沒有在吹什麼新概念，它是把一個「大家都知道該這麼做但沒人做」的事情做到產品等級。

對 compiler 這個領域來說，這類「後處理 compiled binary」的工具這幾年會變成一個重要的方向——因為 LLM-based kernel optimization、GPU reverse engineering、production profiling 都開始要求比 PTX 更深的 observability。Xtrace 可能不會是最終的那個工具，但它定義了一個 baseline：**你的 instrumentation overhead 應該在 3% 以下，fidelity 應該在 95% 以上。** 達不到這個標準的工具，從此之後就會被打上「老一代」的標籤。

---

## 參考來源

- [Xtrace arXiv 2609.28769（paper）](https://arxiv.org/abs/2609.28769)
- [Xtrace HTML 版本](https://arxiv.org/html/2609.28769v1)
- [MLIR News 78th Edition](https://discourse.llvm.org/t/mlir-news-78th-edition-12th-october-2026/92490)
- [PTXBench arXiv 2608.17379](https://arxiv.org/abs/2608.17379)
- [HIERA Workload-Aware GPU Kernel Planning](https://arxiv.org/abs/2608.21157)
- 相關前作：NVBit（PLDI 2019）、SASSI（ISCA 2015）、Neutrino / IKET（NVIDIA 內部工具）
