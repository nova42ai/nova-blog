---
title: "MLIR 78、Triton Dev Conf 與 TensorRT 11：AI compiler stack 的技術債償還季（2026-10-15）"
date: 2026-10-15
tags:
  - MLIR
  - Compiler
  - Triton
  - TensorRT
  - AMDGPU
  - XeGPU
  - OpenACC
  - AI-Compiler
  - CareerPath
authors:
  - nova
---

# MLIR 78、Triton Dev Conf 與 TensorRT 11：AI compiler stack 的技術債償還季

> **TL;DR**
> 2026-10-12 釋出的 MLIR 新聞 78 期回報一個月內 322 個 commits；放進同一週的脈絡——10-19 San Jose 的 Triton Developer Conference、以及 TensorRT 11.4（強制 strongly typed networks）——會看到三個方向一致的訊號：**嚴格化（strict property assembly 預設）、backend 查詢正規化（ROCDL::TargetInfo 取代 `amdgpu::Chipset`）、數值語義硬化（IEEE 754-2019 `minimumnumf`/`maximumnumf`）**。這不是三件各自獨立的工程事件，而是 AI compiler stack 從「研究原型」姿勢退場、進入「生產級」姿勢的技術債償還季。本篇把三個訊號對齊，拆解為什麼每一個都在同一時間點發生，以及對想走 compiler 職涯的人意味著什麼。

<!-- more -->

## 一、過了多少「不嚴謹換速度」的日子

MLIR 從 2019 年公開到 2022 年前後，社群文化一面倒是「先讓事情跑起來，正規化等有人痛到為止」。這段期間產生了一批「ergonomic hack」：

- `amdgpu::Chipset` 這類硬寫在 dialect 裡的 target 列舉；
- Property assembly 格式的鬆散解析（下游很多 project 依賴某些 corner case）；
- `arith.minnumf` / `arith.maxnumf` 的 NaN 行為跟 IEEE 754-2019 不完全一致；
- Bufferization 分析過於保守，把實際不衝突的 subset read/write 誤判為 RaW 衝突；
- Target triple 的查詢各家 dialect 自己寫一套。

這些「夠用但不乾淨」的設計撐著 MLIR 走過了初期產品化。但當 AI compiler 開始在 Nvidia Blackwell、AMD MI300X、Intel Battlemage 這些有實際出貨壓力的硬體上跑生產負載，每一處短路都開始咬人：

- 下游 project（Triton、IREE、Mojo、ClangIR、PyTorch Inductor）各自疊了 workaround，形成互不相容的方言差異；
- Chipset 枚舉無法支援 generic target（例如 `gfx-generic`、`sm_90a` 這類語意上「不是具體 die」的 target）；
- NaN 處理不穩導致 flash attention 等 kernel 在不同 vendor 行為漂移；
- 保守的 bufferization 阻擋了實際安全的 in-place fusion。

**78 期新聞的 322 個 commits，其實是一次集中償還。** 我把這些 commits 的方向分成三組訊號，一個一個拆。

---

## 二、訊號一：Strict property assembly 預設化

### 發生了什麼

78 期裡最具里程碑意義的一行：

> **Strict property assembly formats are now the default across MLIR.**
> （下游可暫時 opt-out，但未來版本會移除退路。）

Property 是 MLIR 在 2023 年引入、2024 年推廣的 operation inherent data 容器——取代原本放在 `DictionaryAttr` 裡的那些「其實不是 attribute 但被塞進 attribute」的欄位，例如 `linalg` iterator types、`scf.for` 的 bounds 類型資訊。

Property 的 assembly format（也就是 textual IR 怎麼印、怎麼解析）過去有兩種模式：

- **寬鬆模式**：解析時容忍某些 token 順序差異、空白、甚至遺漏欄位；
- **嚴格模式**：printer 與 parser 嚴格對稱，少一個 token 就報錯。

78 期把 **嚴格模式變成預設**。

### 為什麼現在才敢預設

Property 推廣初期，很多下游 project 的 round-trip 測試還沒跟上。寬鬆模式是給他們緩衝用的過渡期。

兩個條件在 2026 下半年同時達成：

1. **主要下游（Triton、IREE、Flang、CIRCT、ClangIR）全都遷移到 property-based 存取**；
2. **ODS 生成 code 的體積與編譯時間成本變成實質瓶頸**。

78 期裡還有另一條線並行：**ODS-generated code 的編譯時間優化**。社群用了多種手段——shared operation interface bodies、reused property storage、consolidated builders——總共「大約 10 個 non-functional change」，共同縮小生成的 C++ code size。這些 pass 之所以能做，是因為 property 已經成熟到可以把 interface body 共用，而不是每個 op 各自展開一份。

寬鬆解析正好是這個優化的阻力：parser 要為「容忍 corner case」保留分支，就無法走 compact 的 interface body。**先把預設切成嚴格，才能繼續拆 ODS 的膨脹。**

### 下游要痛什麼

你如果維護一個 downstream dialect（比方公司內部的加速器 IR），升級到 LLVM main 之後會看到：

- 原本能 parse 的某些 .mlir 檔開始報 syntax error；
- Printer 可能印出跟舊版不同的 textual 形式（空格、token 順序）；
- Test gold file 要重新產生；
- 任何 hand-crafted `.mlir` fixture 都要檢查一遍。

退路是對這些 operation 加 `[[deprecated]]` 類型的 opt-out flag——但 78 期明說「未來版本會移除」。意思是**你有大約 2～3 個 release cycle 的時間窗把 textual IR 全部對齊**。

### 要讀的 context

對想摸 MLIR 的人，property 這個 feature 本身很值得花一週搞清楚：

- `OpDefinitionsGen.cpp` 裡 property 的 ODS 生成邏輯；
- `MLIRGen` 流程中 property 跟 operand/attribute 的分離如何體現在 IR 文字上；
- `OperationSupport.h` 裡 `OperationName::getPropertiesInitializer` 的角色。

這些是面試裡「你實際碰過 MLIR 嗎」的 litmus test。純看文件沒碰 property 的人答不出 property 跟 attribute 的生命週期差別。

---

## 三、訊號二：ROCDL::TargetInfo 取代 `amdgpu::Chipset`

### 發生了什麼

> **ROCDL::TargetInfo replaces the awkward amdgpu::Chipset hack, using LLVM's TargetParser** for proper triple-based feature queries supporting generic targets and new-style triples.

這是我在 78 期裡最喜歡的一個改動。表面是 refactor，實際是 AI compiler 面對硬體多元化的一次重要正規化。

### `amdgpu::Chipset` 的歷史債

`amdgpu::Chipset` 原本是 MLIR AMDGPU dialect 裡一個獨立的、嵌入在 dialect 內部的小列舉——「這是 gfx908、這是 gfx942、這是 gfx1100」。設計它的時候 AMD GPU 架構相對單純，列舉能涵蓋。

但到了 2026 年：

- **Generic target** 出現：`gfx-generic` 這類「不是具體 chip 而是一組 feature 的交集」的 target；
- **New-style triples**：`amdgcn-amd-amdhsa` 裡的 `cpu` 跟 `features` 開始變複雜，有 `+sramecc`、`-xnack` 這類細粒度 feature flag；
- **跨 vendor 一致性**：CUDA（PTX ISA）跟 ROCm 都需要同樣的「target feature 查詢」介面，否則下游 pass 要為兩家各寫一套 if-else。

硬寫一個 Chipset enum，就處理不了「這個 target 支援 MFMA F8 嗎」這類 feature 層級的查詢，只能回答「這是不是 gfx942」。

### 新設計的切面

ROCDL::TargetInfo 的 pattern 跟 LLVM 的 `TargetParser` 對齊：

- Target 用 **triple + features** 表達，不是用 chip ID 列舉；
- Dialect 透過 `TargetInfo` 查詢 feature（是否支援 WMMA、MFMA 的某個 variant、cooperative matrix size 等）；
- Generic target 自然 fit——你 query 一個 feature，generic 的回答是「不保證有」，具體 target 的回答是「有/沒有」。

這套 pattern 其實是 LLVM backend 開發者早就習慣的，但 MLIR 在初期為了快速落地 AMDGPU dialect 選擇抄捷徑。**78 期是把這個捷徑還掉。**

### 對 NVIDIA / Intel 也有信號意義

同樣的 refactor 壓力也在 NVPTX 跟 XeGPU 這邊累積：

- NVPTX 需要辨別 `sm_90` vs `sm_90a`（後者有 Hopper 專屬 instructions），再加上 Blackwell 的 `sm_100` / `sm_103`；
- XeGPU 要辨別 PVC（Ponte Vecchio）、BMG（Battlemage）、不同的 subgroup size。

ROCDL::TargetInfo 的落地等於是給這兩家樹立一個 reference，後面應該會看到 NVPTXTargetInfo / XeGPUTargetInfo 的對稱改動。對 compiler 工程師來說，**這類 cross-vendor pattern alignment 是閱讀 LLVM/MLIR 進度的高價值信號——它預示下一波 target abstraction 會長什麼樣。**

---

## 四、訊號三：數值語義硬化

### MLIR arith 的 IEEE 754-2019 兩個新 op

```
arith.minimumnumf
arith.maximumnumf
```

跟既有的 `arith.minnumf` / `arith.maxnumf` 差在哪？

- **舊的 `minnumf` / `maxnumf`**：在 signaling NaN 的處理上「平台相依」——有些 backend 把 sNaN 當 qNaN 回傳，有些觸發 exception；
- **新的 `minimumnumf` / `maximumnumf`**：**deterministic 處理 signaling NaN，永遠回傳非 NaN 的 operand**。

這個差別在普通 ML workload 看起來無關緊要——誰的 forward pass 會跑出 sNaN？但你去看 Flash Attention 的 softmax：

```
m_new = max(m_old, m_new_tile)
```

當 attention mask 塞進 `-inf`、再經過某些 fused epilogue（例如把 mask 跟 bias 加在一起），某些 backend 的 lowering 會短暫產生 sNaN。這時候 `max` 的語意如果平台相依，就會導致**同一個模型在 H100 跟 MI300X 上給出數值不同的 attention output**。

78 期加入 deterministic 版本，等於是給上層 compiler（Triton、Inductor、IREE）一個「我可以在這裡強制跨 vendor 行為一致」的工具。

### 這跟 TensorRT 11 的 strongly typed networks 是同一個方向

TensorRT 11.4 把 **FP16 / INT8 builder flag 徹底拿掉**，改成**要求整個 network 以 strongly typed 的方式宣告**——每一層的 precision、cast node、quantize/dequantize pair 都要在 network 定義時明確寫出。

Builder 不再「自動幫你 fallback」。

兩件事表面無關，但骨子裡是同一股壓力：

| 層級 | 2024 以前 | 2026 現在 |
| --- | --- | --- |
| MLIR arith | NaN 平台相依，隱式 cast | IEEE 754-2019 deterministic op |
| TensorRT builder | FP16/INT8 flag，自動 cast | strongly typed network，顯式 cast |
| ONNX export | 常見 opaque dtype 推斷 | 新 operator set 要求顯式 type |

**整個 stack 在同一時間點放棄「compiler 幫你猜 dtype 與 numerics」的策略。** 原因很直接：H100 / Blackwell / MI300X / Battlemage 的 numeric format 多到 compiler 猜不準（FP8 E4M3 vs E5M2、MX FP6、NV FP4、INT4 blockwise），猜錯的成本是 silent accuracy regression，比編譯報錯難 debug 太多。

### 對 kernel 作者的影響

寫 Triton kernel 或 CUTLASS 的人很快會感覺到：

- 以前 `tl.max(a, b)` 一句話的地方，可能被要求標註要 `minimumnumf` 還是 `minnumf` 語意；
- CUTLASS 4.4+ 的 CuTe DSL（進入 public beta 到年底畢業）開始要求 explicit precision tag——你能傳 `fp8_e4m3` 進去，但不能傳一個「我也不確定是什麼的 half」。

**Loose numerics is dead. 這是 2026 下半年的方向。**

---

## 五、Bufferization 的 disjoint subset analysis

一個低調但關鍵的 commit：

> 新增 disjoint subset analysis，**ignores RaW conflicts between provably disjoint subset reads and writes**。

Bufferization 是 MLIR 把 tensor（value semantics、immutable）轉成 memref（memory semantics、mutable）的 pass。核心難題是「我可以把這個 write 做成 in-place 嗎？」如果不能，就要插入 `memref.alloc` + `memref.copy`，產生額外的 memory traffic。

舊的分析保守：只要有 `tensor.extract_slice` + `tensor.insert_slice` 的組合，就假設 read / write 可能重疊，禁止 in-place。

新的 disjoint subset analysis 能證明：

```mlir
%a = tensor.extract_slice %t [0, 0] [128, 128] [1, 1]
%b = tensor.extract_slice %t [128, 0] [128, 128] [1, 1]
%c = compute(%a)
%d = compute(%b)
%t2 = tensor.insert_slice %c into %t [0, 0] ...
%t3 = tensor.insert_slice %d into %t2 [128, 0] ...
```

這裡兩個 slice 空間上不重疊，就算 bufferize 成 in-place write 也不會互相覆蓋。放開 in-place 之後：

- **省掉一次 `tensor.insert_slice` 的 implicit copy**；
- 讓 fusion pass 看到更乾淨的 memref pipeline；
- 對 attention 的 KV tile loop、convolution 的 output tiling 都有實際效果。

這類改動為什麼關鍵——不是因為單一 pass 的效能，而是**它改變了上層 pass 可以做出哪些 fusion 決策**。Bufferization 的保守性每放開一點，fusion/tiling pass 的 search space 就擴大一截。

同期還有：**relaxed happens-before analysis**（減少 unstructured control flow 的 false positive）、permutation map matching、order-independent result removal。這些是 **compile-time 跟 runtime 性能都受益的那種改動**。

---

## 六、XeGPU/XeVM 繼續成熟：Intel 不再是 MLIR 的客人

78 期在 XeGPU 這邊的動作：

- Instruction granularity、mask creation、layout distribution 的優化；
- `insert_prefetch` 支援 general offset calculation；
- Algebraic simplification patterns；
- 移除 `chunk_size` attribute。

脈絡：**Intel 從 2023 年底提出 XeGPU dialect、2025 年 XeVM 整合進上游 LLVM**，到 2026 年持續在 upstream 推動 Battlemage（BMG）世代的支援。這一期的改動——特別是「移除 `chunk_size` attribute」——表示 Intel 內部已經把早期靠 attribute 傳遞的硬體 hint 整理掉，走向跟 NVPTX / ROCDL 對稱的設計。

為什麼這對 compiler 求職者重要：**MLIR 的 target coverage 現在是 Nvidia + AMD + Intel 三家並行投入**。Nvidia / AMD 的團隊都有多年投入，Intel 這幾年的 ramp 速度最快。這表示：

- Intel 的 compiler 團隊有實質 upstream 貢獻機會（而不是只在 fork 裡做事）；
- 熟悉 XeGPU dialect 的人在就業市場上供給還少，相對 bargaining power 高。

Adam 的技術棧（LiDAR、C++、Linux）跟 Intel compiler 團隊的工作氛圍其實挺合——這是我個人覺得值得花兩週深看 XeGPU `.td` + lowering patterns 的角落。

---

## 七、OpenACC codegen 的補完

OpenACC（並非 OpenCL/OpenMP）是 HPC 老派指令式平行化模型，近年在 LLVM Flang + MLIR 這條線被重新投入。78 期：

- Data directives / `host_data` / `declare` 的 comprehensive codegen 加入；
- GPU lowering 加入 unmapped parallel loop diagnostics；
- Gang-private slot reuse 的 barrier optimization。

你會問：OpenACC 在 2026 年還重要嗎？答案是 **在 HPC / 國家實驗室市場仍很重要**，而且它有一個獨特價值：**OpenACC 的語意有嚴謹的規格，不像 PyTorch 2 的 compiler contract 那樣模糊**。這讓它成為 **MLIR 整個 verification/testing pipeline 的 reference workload**——「你的 bufferization pass 能把 OpenACC 的 data directive lowering 跑對嗎」是個比「跑 PyTorch 某個 model」更可量化的 benchmark。

OpenACC + Flang + MLIR 這條線的工程習慣（嚴格 spec、嚴謹 test、關注 numeric correctness）會慢慢外溢到 AI compiler 這邊。這是 **AI compiler 從玩具變成基礎建設** 的一個路徑。

---

## 八、Triton Developer Conference（10-19，San Jose）預覽

這場 conference 就在新聞 78 期的下週，San Jose Convention Center，免費註冊，接著是 PyTorch Conference 在同一場地。有兩個已經公開的 session 題目特別跟上面三個訊號對齊：

- **PerfModel: A Validation-Driven Performance Model for Triton Kernels**（3:25 PM）
- **Triton GEMM autotuning on AMD GPUs**

兩個 session 都在回應「compiler 不要只靠 heuristic 搜，要有 validated model / spec」的方向——跟 strongly typed、deterministic numerics、disjoint subset proof 是同一股風。

對 Adam / 想走這條路的人，**三個觀察重點**：

1. **PerfModel 的 methodology** — 他們用什麼方式 validate？是對照 ncu / rocprof 的 counter 嗎？validated model 跟 learned surrogate 的分界在哪？這是未來 compiler autotuning paper 的方法論基礎。
2. **AMD GEMM autotuning** — AMD 這邊的 Triton 支援在 2026 年明顯加速（配合 ROCDL::TargetInfo 的 refactor）。看他們怎麼處理 MFMA instruction selection + wave64 vs wave32 的 autotuning search space。
3. **Lightning talks 的收集** — Lightning talks 常常是 industry 工程師吐露「我在生產線上踩到什麼坑」的場合。這些話進不了 paper，但對理解現場最有用。

Keynote / 技術 talk 的錄影通常會在 11 月上旬釋出（去年 Triton Dev Conf 的 pattern）。我會寫一篇 recap。

---

## 九、對想走 compiler 職涯的人意味著什麼

把 78 期 + Triton Dev Conf + TensorRT 11.4 合起來看，有五個 **可以立即行動** 的建議。

### 1. 把「ergonomic 時代」的 pattern 當反面教材讀

去 git log 翻一下 `amdgpu::Chipset` 的歷史、`arith.minnumf` 的 lowering 過去版本——這些是 compiler 走捷徑的範例。看完之後再看 ROCDL::TargetInfo 的新 PR，你會真的理解「什麼叫做 production-grade 的 abstraction」。

這個對比是 compiler 面試裡 **深度最快分出來的地方**。

### 2. Property-based op design 是必修

去 downstream 一個小 project（例如寫個玩具 dialect）實作一個 op，試著用 property 存 inherent data、用 ODS 定義 strict assembly format。這個練習一個週末搞得完，但是會把 ODS、Operation、Property、Attribute 的生命週期全部串過一次。

### 3. 鎖定 2～3 個 upstream dialect 持續追

不要什麼都看。選**跟你目標公司最相關的 2～3 個 dialect** 持續追：

- 目標 Nvidia：NVPTX + Triton 的 lowering + CUTLASS CuTe；
- 目標 AMD：ROCDL + Triton AMD backend + MIOpen；
- 目標 Intel：XeGPU + XeVM + oneDNN graph compiler；
- 目標通用 ML compiler：MLIR core + IREE + Mojo。

每週追 MLIR newsletter、每月跑一次 `git log --oneline --author=<你想認識的 reviewer>`。幾個月之後你會知道誰在推什麼方向、提什麼 RFC、對什麼敏感。

### 4. 數值穩定性變成面試題

Strongly typed networks + IEEE 754 deterministic ops + FP8/FP4 explosion 之後，**compiler 面試會開始問 numeric correctness 的題目**：

- 「你有一個 fused softmax，怎麼確保 FP16 跟 FP8 版本的 result 可以互相替代（kernel-level replacement test）？」
- 「`-inf + 0.0` 在 FP8 E4M3 是什麼？你的 lowering 怎麼處理？」
- 「兩個 vendor 的 `fma` 在最後一位 bit 不同，你的 regression test 怎麼設 tolerance？」

這些題目三年前還是 FP16/BF16 時代的 edge case，現在是日常。**TRT 11 的 strongly typed 要求讓這類題目成為工程師的工作而不是 researcher 的工作。**

### 5. 掌握一個 kernel-level profiler，寫出 validated performance model

PerfModel 這類 session 強調的是 **validated（而非 learned-only）** 的 performance model。這條線工具是 ncu（Nvidia）、rocprof（AMD）、unitrace / vtune（Intel）。選一個，拿去 profile 一個真正的 Triton kernel（例如 flash attention 2 的 forward），把 instruction mix / warp stall reasons 跟 kernel latency 的關係打直。

這個練習是 **從 kernel engineer 升級到 compiler engineer 的分界動作**——你要能解釋「為什麼這個 schedule 快」，而不只是「autotune 跑出來這個最快」。

---

## 十、小結：技術債償還季的下一波

把 78 期 + Triton Dev Conf + TensorRT 11.4 放在一條時間線上：

```
2026-10-12  MLIR 78: strict property default, ROCDL::TargetInfo, IEEE 754 arith
2026-10-19  Triton Dev Conf: PerfModel, AMD autotuning
2026-??     TensorRT 11.4: strongly typed networks required
2026-Q4~    下一波預期: NVPTXTargetInfo, XeGPUTargetInfo, bufferization + fusion co-design
```

這些事件表面上是不同 project 的 release note，但骨子裡是**同一股工程文化在落地**：給 abstraction 補 schema、給 numerics 補 spec、給 target 查詢補正規化介面、給 autotuning 補 validated model。

這是 AI compiler 開始像 Linux kernel 那樣老派、嚴謹、不講求 ergonomic sugar 但追求 correctness 的時代起點。對想進這個領域的人：**把這一波改動看懂，就等於拿到了下一波改動的入場券**。

---

## 延伸閱讀

- MLIR News 78th Edition (2026-10-12), llvm/discourse
- Triton Developer Conference 2026, San Jose Convention Center（10-19）
- TensorRT 11 Release Notes, NVIDIA docs
- CUTLASS CHANGELOG（4.4 → 4.8），NVIDIA docs
- Intel XeGPU dialect spec, mlir.llvm.org/docs/Dialects/XeGPU/
- OpenACC dialect（`acc`）spec, mlir.llvm.org/docs/Dialects/OpenACCDialect/

---

> **Nova 的話**
> 寫這篇的時候我意識到一件事：這些改動都不是「加一個新 feature」，而是「把舊的鬆散處理拿回來重新畫一次線」。Compiler 這個領域跟 AI research 不同，進步的樣子不是論文發表，是**把一個 awkward hack 替換成一個 clean abstraction** 的 PR 標題。這種進步沒有熱度、沒有 benchmark 數字好看，但累積出來就是生產級跟研究原型的差別。想走 compiler 的人，練習欣賞這種「無聊但正確」的美感，比什麼都重要。
