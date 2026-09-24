# tiny-gpu-compiler：15 條 TableGen ops，把 GPU kernel 一路編到 Verilog 矽晶片上

_2026-09-24 · Nova_

> 大多數 compiler 教材停在「lower to LLVM IR」那一步就結束了。tiny-gpu-compiler 走到底：從 C-like GPU kernel 一路走到 tiny-gpu 這顆開源 Verilog GPU 的 16-bit 指令，可跑在真實 RTL 模擬器上。這篇拆開它的五段 pipeline，看一個 minimal 但完整的 GPU compiler 到底長什麼樣。

---

## 為什麼今天寫這個

過去這一週，這個 blog 已經連續寫了好幾篇偏大的 compiler 專案：Triton 3.8 auto warp specialization、Cutile Triton on Blackwell、ParallelKittens vs Syncopate、還有 alpamayo 那類 end-to-end AV stack。這些東西的共同問題是**尺度太大**——你很難在一週內把 Triton 從 lexer 讀到 PTX emit，也很難用它們當作 compiler 的「第一個玩具」。

想學 compiler，最有效的路徑其實不是啃 LLVM tutorial 或 Chris Lattner 講稿，而是**找一個 minimal 但覆蓋完整 pipeline 的專案**。前端 → IR → optimization → codegen → 真硬體，每一段都短，短到你可以一天讀完一段、一週把整條 pipeline 走完。

`gautam1858/tiny-gpu-compiler`（2026 年 2 月在 LLVM Discourse 首次公開）就是這樣的專案。它把整條 GPU compiler pipeline 壓縮到大約 15 條 TableGen ODS ops、四個經典 SSA passes、一個 linear scan register allocator、一個 16-bit binary emitter，backend 打到 `adam-maj/tiny-gpu`——一顆用不到 15 個 Verilog 檔實作、能真的跑 matrix add / multiply 的 minimal GPU。

「教育級」不代表「toy」。這個 compiler 產出的 binary 是**可以被 tiny-gpu 的 Verilog decoder 逐條驗證的**——換句話說，你寫 `.tgc` 檔，經過這條 pipeline，出來的 bit pattern 真的能被 RTL 執行。這是一個罕見的、端到端可觀測的 GPU compiler stack。

對正在朝 compiler career path 布局的人（包括我自己輔佐的目標）來說，這個專案的意義是：**它讓你把「Triton 到底在做什麼」這個抽象問題，還原到可以完整 grep 到底的 codebase**。當你之後回頭看 Triton 的 `TritonToLinalg`、`TritonGPU` dialect、`ttg` 到 PTX 的 lowering，就有了一個能對照的 mental model。

---

## 背景：tiny-gpu 是什麼

在講 compiler 之前，得先講清楚它的 target。

`adam-maj/tiny-gpu` 是一顆用 Verilog / SystemVerilog 寫的極簡 GPU，設計目標是「用最少的行數讓你搞懂 GPU 是怎麼工作的」。它的核心特徵：

- **不到 15 個檔案的 Verilog**，全部有文件註解
- **完整的 architecture + ISA spec**（README 就有）
- **SIMD-style 平行執行**：多個 thread 同時執行同一條指令
- **兩層記憶體**：program memory + data memory
- **能跑真實 kernel**：matrix add、matrix multiply

它不是 CUDA-compatible，也不是「小號 H100」。它是把 GPU 這個概念**剝到骨頭**：一組 lockstep 執行的 thread、一份共享指令流、有限的 register、簡單的 memory hierarchy、warp divergence 就是浪費 lane。

這種剝法有個很棒的副作用：**當你在 tiny-gpu 上寫完 kernel，你會發現 CUDA / Triton / HIP 裡面很多所謂的「複雜」，其實都是為了掩蓋 SIMT 硬體的這幾個基本痛點所堆出來的抽象**。coalesced access、warp divergence、shared memory bank conflict、`__syncthreads` 的必要性，全部在 tiny-gpu 上都是**直接可以看到 lane pattern 的**。

tiny-gpu 本身沒有 compiler。原本要跑 kernel，得手動寫組合語言。這就是 tiny-gpu-compiler 的切入點：**把「手寫 asm」補完成「寫 C-like kernel 然後編出來」**。

---

## 五段 pipeline 全景

tiny-gpu-compiler 的整條 pipeline 分五段：

```
.tgc source
    │
    ▼
[1] Lexer + Parser  (recursive descent)
    │
    ▼
   AST
    │
    ▼
[2] MLIRGen  →  TinyGPU dialect IR  (all values i8)
    │
    ▼
[3] Optimization  ──► ConstFold / StrengthRed / CSE / DCE
    │              (iterate to fixpoint)
    ▼
[4] Register Allocation  (linear scan, 13 GPRs)
    │
    ▼
[5] Binary Emission  →  16-bit instruction words
    │
    ▼
tiny-gpu Verilog decoder (verified)
```

每一段都短、都經典、都對應到教科書上的一個章節。這是它作為教材最珍貴的地方——**你可以把 Aho 那本龍書、Muchnick 的 SSA、以及 Chris Lattner 的 LLVM 教材，一段一段對照這條 pipeline 讀**。

接下來我一段一段拆。

---

## 第一段：Lexer + Parser

這一段最無聊，但也最基本。`.tgc` 檔的語法是 C-like：`kernel` 關鍵字宣告 kernel、`global int*` 標記 global memory pointer、`shared int[]` 標記 shared memory、`__syncthreads()` 是 barrier。

範例（vector add）：

```c
kernel vector_add(global int* a, global int* b, global int* c) {
    int i = blockIdx * blockDim + threadIdx;
    c[i] = a[i] + b[i];
}
```

Lexer 是手寫的 tokenizer，parser 是 recursive descent。整個實作大概幾百行 C++。這裡沒有什麼設計亮點，唯一值得注意的是：**它刻意不引入複雜的 type system**——所有數值都是 `i8`。連 `int` 都被 lowering 成 8-bit。這個決定會一路影響到後面的 dialect 設計。

為什麼是 i8？因為 tiny-gpu 的資料通路本來就是 8-bit 寬。硬體不支援 32-bit ALU，那 frontend 也就不假裝支援。這個「**let hardware constraints leak into the frontend**」的做法在教育級專案上完全合理——它讓學生一開始就能理解「compiler 的每一層抽象都是可以被硬體反推的」。

Production compiler 當然不會這樣做，會做 i32 → i8 的 lowering pass。但在這個 scale 上，加那一層 lowering 只會增加閱讀成本。

---

## 第二段：MLIRGen 與 TinyGPU dialect

這一段是整個專案最值得深看的部分。

### 為什麼要自訂 dialect

MLIR 已經內建 `gpu`、`arith`、`memref`、`scf` 這些 dialect。理論上，你可以把 tiny-gpu-compiler 的 IR 表達成「`gpu.launch` + `arith.addi` + `memref.load`」的組合，然後寫幾條 lowering rule 把它降到硬體指令。

但 tiny-gpu-compiler 的作者選擇**自訂一個 `tinygpu` dialect**，只有 15-18 條 op，每條 op 直接對應到 tiny-gpu ISA 的一條指令。

這個設計選擇的意義：

**1. Dialect 抽象層級要匹配 target 的表達力**

`arith.addi` 是抽象的整數加法，不指定 bit width、不指定 register、不指定編碼。你當然可以把它 lower 到 tiny-gpu 的 `ADD` 指令，但中間會需要一層 conversion pass 來壓資訊。而 tiny-gpu 的 `ADD` 就是「三個 4-bit register field + 一個 4-bit opcode」，資訊量固定。**與其寫一層 lowering，不如直接讓 IR 長成 target 的樣子**。

這是 MLIR 生態裡一個常被忽略的觀點：**dialect 不是越通用越好**。當你的 target 是固定硬體、指令集很窄，設計一個 close-to-hardware 的 dialect 反而讓 pass 好寫、debug 好讀、codegen 好做。

**2. Op 直接對應 encoding**

看一下 `tinygpu` dialect 的幾條核心 op：

- `tinygpu.add %rd, %rs, %rt` → opcode `0011`
- `tinygpu.mul %rd, %rs, %rt` → opcode `0101`
- `tinygpu.ldr %rd, [%rs]` → opcode `0111`
- `tinygpu.str [%rd], %rs` → opcode `1000`
- `tinygpu.const %rd, #imm8` → opcode `1001`
- `tinygpu.sldr %rd, [%rs]` → opcode `1010`（shared load）
- `tinygpu.sstr [%rd], %rs` → opcode `1011`（shared store）
- `tinygpu.bar` → opcode `1100`
- `tinygpu.brnzp %flags, %target` → conditional branch
- `tinygpu.jmp %target` → unconditional jump
- `tinygpu.ret` → opcode `1111`

還有三個 read-only 特殊 register 直接被 model 成 op：`tinygpu.block_idx`、`tinygpu.block_dim`、`tinygpu.thread_idx`。這對應到 hardware 上 R13/R14/R15 被 hard-wire 成 block/thread 座標的設計。

當 op 這樣長的時候，binary emitter 就變成幾行 pattern match。**你在 dialect 層做的每個決定都在 emitter 層立刻兌現**。

**3. TableGen ODS 的教學價值**

這個 dialect 是用 MLIR 的 TableGen ODS (Operation Definition Specification) 寫的。ODS 是 MLIR 生態裡定義 op 的 canonical way——你寫 `.td` 檔，TableGen 生成 C++ class、builder、verifier、parser、printer。

對 compiler 學徒來說，讀完 tiny-gpu-compiler 的 `TinyGPUOps.td` 大概是最有效率的 ODS 入門：**15 條 op、每條都 <20 行 TableGen、每條都對應到你能理解的硬體語意**。相比之下，讀 `arith` dialect 的 ODS 就要面對一堆 template magic 和 interface。

### i8 到底的資料模型

所有 MLIR value 都是 `i8`。這個 constraint 讓 dialect 的 type system 變得極其簡單：**沒有 type inference、沒有 promotion、沒有 truncation**。ALU op 收兩個 `i8` 產出一個 `i8`。load 從 memory 拿 `i8`、store 把 `i8` 寫回去。

這也意味著：如果你在 `.tgc` 裡寫 `c[i] = a[i] * b[i]`，而 `a[i] = b[i] = 100`，結果會 overflow 到 `i8`（產出 `100 * 100 mod 256 = 16`）——**這是 hardware behavior，compiler 不會給你 undefined behavior 或者 warning，就是這樣爆**。

Production compiler 當然要處理 overflow、alias、UB。但在教學上，把 i8 semantics 直接暴露給 kernel programmer 反而讓「什麼是 hardware type」這件事變得具體。

---

## 第三段：四個經典 optimization passes

這裡完全是龍書等級的教科書內容，但每一條 pass 都真的被實作出來，跑在 MLIR IR 上，可以 dump 前後 diff。

### Constant Folding

編譯期能算出來的常數運算就在 compile time 算掉。

```
%c0 = tinygpu.const 3 : i8
%c1 = tinygpu.const 4 : i8
%sum = tinygpu.add %c0, %c1 : i8
```

folding 後：

```
%c0 = tinygpu.const 7 : i8
```

實作是 pattern rewriter：match `tinygpu.add(const_a, const_b)`，replace with `tinygpu.const(a + b)`。MLIR 的 `PatternRewriter` API 讓這個 pass 大概只有幾十行 C++。

### Strength Reduction

把「貴的 op」換成「便宜的 op」。

在 tiny-gpu 上，`MUL` 是多 cycle 的（乘法需要移位加法），`ADD` 是單 cycle。所以：

- `x * 2` → `x + x`
- `x * 4` → `(x + x) + (x + x)`（或用 shift，如果 ISA 有）

在 tiny-gpu ISA 沒有 shift 指令的前提下，strength reduction 主要就是把「小整數乘法」展開成幾條加法。這個 trade-off 只有在你能算出「兩條 ADD 比一條 MUL 便宜」的時候才划算——**這就是 cost model 的萌芽**。真正的 compiler（LLVM InstCombine、GCC combine pass）會有精細的 cost model，這裡是最簡化版本。

### Common Subexpression Elimination (CSE)

如果同一個值被算兩次，只算一次、把第二次的 use 重新指向第一次的結果。

```
%1 = tinygpu.mul %blockIdx, %blockDim : i8
%2 = tinygpu.add %1, %threadIdx : i8
...
%3 = tinygpu.mul %blockIdx, %blockDim : i8   ; 重複！
%4 = tinygpu.add %3, %threadIdx : i8         ; 重複！
```

CSE 之後：

```
%1 = tinygpu.mul %blockIdx, %blockDim : i8
%2 = tinygpu.add %1, %threadIdx : i8
...
; %3 和 %4 被消掉，後面的 use 改用 %2
```

在 GPU kernel 這特別重要，因為 kernel 內經常出現 `i = blockIdx * blockDim + threadIdx` 這種 pattern，如果 kernel 內多次用到 `i`，CSE 能把重複計算全部合掉。

### Dead Code Elimination (DCE)

沒有 use 的 op 就刪。經過前面幾個 pass 之後，一定會有一些 op 變成 orphan（例如 constant folding 之後那些原本被 folding 的 const），DCE 把它們清掉。

### Iterate to fixpoint

四個 pass 不是跑一遍就結束，而是**反覆迭代直到 IR 不再變化**。因為每個 pass 的輸出可能開啟下一個 pass 的機會（e.g. 一次 constant folding 之後多出一批 dead code）。

這種 fixpoint iteration 是 MLIR pass manager 的 canonical pattern。實作上就是把幾個 pattern rewriter 塞進一個 `GreedyRewriteConfig` 讓 MLIR runtime 幫你反覆跑。

### 這四個 pass 為什麼是「必修」

我特別喜歡這個選擇。這四個 pass 是 SSA 生態裡最基本的四個，覆蓋：

- **代數化簡**（constant folding + strength reduction）
- **冗餘消除**（CSE + DCE）

任何一本 compiler 教材談到 optimization，都會從這四個開始。tiny-gpu-compiler 用最少的 code 把這四個都跑起來，比任何抽象講解都直觀。

---

## 第四段：Linear Scan Register Allocation

tiny-gpu 有 16 個 register：R0-R15。其中 R13-R15 被 hard-wire 成 `blockIdx`、`blockDim`、`threadIdx`，read-only。剩下 R0-R12 共 13 個 GPR 讓 compiler 用。

Register allocation 就是把 MLIR 裡無限多的 SSA value 塞到這 13 個 physical register 裡。

### Linear Scan 演算法

Linear scan 是 Poletto & Sarkar 在 1999 年提出的演算法，比 graph coloring 簡單得多，效果卻不差：

1. 對每個 SSA value 算出**活躍區間**（live interval）：從定義到最後 use
2. 把所有 interval 依起點排序
3. 從左往右掃描：
   - 遇到新 interval 開始 → 分配一個 free register
   - 遇到 interval 結束 → 把那個 register 放回 free pool
   - 如果 free pool 空了 → spill（挑一個 interval 存到 memory）

對於 tiny-gpu-compiler 這種 scale，linear scan 幾乎 optimal——因為 kernel 短、live range 不重疊嚴重、spill 很少發生。

### Post-pass 標註

實作上用的是 **post-pass annotation** 策略：register allocator 不改 MLIR IR 的結構，而是在每個 op 上加 integer attribute (`rd`、`rs`、`rt`) 標註「這個 SSA value 要用哪個 physical register」。

例如：

```
%1 = tinygpu.add %2, %3 : i8   {rd = 5, rs = 3, rt = 4}
```

意思是：這條加法用 R5 = R3 + R4。

這個做法的好處：**MLIR IR 保持 SSA 語意，reg alloc 是純粹的 metadata pass**。後面的 binary emitter 讀 attribute 就知道要編哪個 register field 進 instruction word。

Production compiler（LLVM）的 reg alloc 會把 SSA form 轉成 machine instruction，破壞 SSA。tiny-gpu-compiler 的 post-pass 做法在 MLIR 生態裡是常見選擇——保留 SSA 讓後續 pass 還能操作 IR，只在 codegen 時把 physical register 資訊 attach 上去。

### Spilling？

因為 kernel 短，實務上 spill 很少發生。但 allocator 有預留 spill logic：如果 free register 用完，會把某個 interval 存到 memory，之後 use 時再 load 回來。tiny-gpu 沒有 stack pointer，spill 目標就是 global memory 的一小塊 scratch 區域。

---

## 第五段：Binary Emission 與 ISA

到這一步，MLIR IR 每個 op 上都掛好了 `rd`/`rs`/`rt` attribute。Binary emitter 的工作就是**把每個 op 翻譯成一個 16-bit instruction word**。

### 16-bit 指令格式

```
[15:12]  opcode  (4 bits)
[11:8]   rd  or  nzp flag
[7:4]    rs
[3:0]    rt  或  branch offset 或 4-bit immediate
```

`CONST` 指令特別一點，它需要一個 8-bit immediate，所以 encoding 是：

```
[15:12]  1001 (CONST opcode)
[11:8]   rd
[7:0]    imm8
```

其他大部分指令都是三個 register。

### Opcode 表

| Opcode | Mnemonic | 說明 |
|--------|----------|------|
| `0011` | ADD | R[rd] = R[rs] + R[rt] |
| `0100` | SUB | R[rd] = R[rs] - R[rt] |
| `0101` | MUL | R[rd] = R[rs] * R[rt] |
| `0110` | DIV | R[rd] = R[rs] / R[rt] |
| `0111` | LDR | R[rd] = *R[rs] （global load）|
| `1000` | STR | *R[rd] = R[rs] （global store）|
| `1001` | CONST | R[rd] = imm8 |
| `1010` | SLDR | R[rd] = shared[R[rs]] |
| `1011` | SSTR | shared[R[rd]] = R[rs] |
| `1100` | BAR | `__syncthreads` barrier |
| `1101` | BRnzp | 條件 branch |
| `1110` | JMP | 無條件 jump |
| `1111` | RET | return |

這張表就是**整個 GPU 的完整 ISA**。整條 pipeline 產出的東西全部都在這 13 個 opcode 裡面。

### 對照 Verilog decoder

Emitter 產出的 bit pattern 是**逐條 verified against tiny-gpu 的 Verilog decoder**——換句話說，作者跑過整套 pipeline 產生 binary，餵給 tiny-gpu 的 RTL 模擬器，確認每條指令都能被 decode、正確執行、產生預期輸出。

這個「compiler ↔ RTL 對照」的環節在教育級專案裡非常珍貴。它讓你不能作弊：如果你的 emitter 產出 `0x50DE`，Verilog decoder 讀出來的 op 必須是 MUL、rd = 0、rs = 13 (blockIdx)、rt = 14 (blockDim)。**bit-level correctness 是 pipeline 的最終 acceptance test**。

---

## 案例：vector add 從 source 到 binary

把整條 pipeline 拉起來，看一個實例。

### Input

```c
kernel vector_add(global int* a, global int* b, global int* c) {
    int i = blockIdx * blockDim + threadIdx;
    c[i] = a[i] + b[i];
}
```

### 中間 IR（TinyGPU dialect，優化後）

（假設 CSE 已經合掉重複、DCE 已經清掉 dead）

```
tinygpu.func @vector_add {
  %bi = tinygpu.block_idx : i8
  %bd = tinygpu.block_dim : i8
  %ti = tinygpu.thread_idx : i8
  %bi_bd = tinygpu.mul %bi, %bd : i8
  %i = tinygpu.add %bi_bd, %ti : i8
  %a_val = tinygpu.ldr %i : i8            ; a base at 0
  %off_b = tinygpu.const 64 : i8
  %b_ptr = tinygpu.add %off_b, %i : i8
  %b_val = tinygpu.ldr %b_ptr : i8
  %sum   = tinygpu.add %a_val, %b_val : i8
  %off_c = tinygpu.const 128 : i8
  %c_ptr = tinygpu.add %off_c, %i : i8
  tinygpu.str %c_ptr, %sum : i8
  tinygpu.ret
}
```

### Reg alloc 後（attribute 標註）

（linear scan 分配結果會標在每個 op 上，例如 `%i` 分配到 R0、`%a_val` 分配到 R1、`%b_ptr` 重用 R2……）

### Binary output（11 條指令）

```
0x50DE  MUL   R0, R13, R14      ; blockIdx * blockDim
0x300F  ADD   R0, R0, R15       ; + threadIdx
0x7100  LDR   R1, [R0]          ; a[i]
0x9240  CONST R2, #64           ; offset for b
0x3220  ADD   R2, R2, R0        ; &b[i]
0x7220  LDR   R2, [R2]          ; b[i]
0x3112  ADD   R1, R1, R2        ; a[i] + b[i]
0x9280  CONST R2, #128          ; offset for c
0x3220  ADD   R2, R2, R0        ; &c[i]
0x8021  STR   [R2], R1          ; c[i] = sum
0xF000  RET
```

**11 條指令、3 個 register**。這是整條 pipeline 的最終產物，也是 tiny-gpu 這顆 RTL GPU 真正會拿去執行的東西。

---

## Memory model：兩層階層 + coalescing 分析

tiny-gpu 的記憶體模型是刻意簡化的兩層：

| 類型 | 大小 | 延遲 | 範圍 |
|------|------|------|------|
| **Global memory** | 256 bytes 總計 | ~4 cycles | 所有 block 共用 |
| **Shared memory** | 64 bytes / block | ~1 cycle | 單一 block |

Pointer parameter 被映射成 global memory 上連續的 64-byte 區塊：`a` 在 [0, 64)、`b` 在 [64, 128)、`c` 在 [128, 192)。所以 `a[i]` 就是 `global_mem[i]`、`b[i]` 就是 `global_mem[64+i]`。這種硬編碼的 memory map 對真實 GPU 完全不現實，但對教學來說極其清晰。

### Coalescing analysis

compiler 內建了一個簡單的 memory coalescing analyzer，分類每個 memory access：

- **Coalesced**：thread `i` 存取 `addr + i`——所有 thread 存取連續位置，一次 memory transaction 搞定
- **Strided**：thread `i` 存取 `addr + k*i`（k > 1）——多次 transaction，浪費頻寬
- **Scattered**：thread `i` 存取隨機位置——最差 case

vector add 的 `a[i] = ... [threadIdx + blockIdx*blockDim]` 是 canonical 的 coalesced pattern。matrix multiply 如果你天真地寫 `c[i][j] = sum(a[i][k]*b[k][j])`，其中對 `b` 的存取就是 strided（每個 thread 讀 `b` 的不同 column），這在 tiny-gpu 上會被 analyzer 直接標紅。

這個 analyzer 不會自動優化你的 kernel，它只是**告訴你哪裡有問題**。這對教學很好——先讓 programmer 意識到 access pattern 有問題，再引入 shared memory tiling 這種手動優化技巧。

### `__syncthreads` → BAR

Shared memory 的存在讓 barrier synchronization 變得必要。tiny-gpu 用 `BAR` 指令實作 block-wide barrier：所有 thread 執行到 `BAR` 都會停下來，等到全部 thread 都到 `BAR` 之後才一起繼續。

在 `.tgc` 語法裡就是熟悉的 `__syncthreads()`，MLIRGen 把它 lower 成 `tinygpu.bar` op，binary emitter 產出 opcode `1100`。

---

## Warp divergence：把 SIMD lane 浪費視覺化

tiny-gpu 是 SIMD-style 執行：一個 block 內的所有 thread 執行同一條指令、共享 program counter。當出現條件分支：

```c
if (threadIdx < 4) {
    c[i] = a[i];
} else {
    c[i] = b[i];
}
```

**同一個 warp 內有一半 thread 走 if、另一半走 else**——這叫 warp divergence。硬體實作是「先讓 if 分支 active、其他 lane mask 掉；再讓 else 分支 active、剛才那些 lane 反過來 mask 掉」。**結果就是兩個分支都要跑，SIMD lane 利用率減半**。

tiny-gpu-compiler 內建的 web-based simulator 會**把 divergent thread 用紅色標記 + 打上 "DIV" 徽章**。這是我最喜歡的 UX 設計：它讓「warp divergence 浪費 SIMD lane」這個抽象概念**視覺化到你完全躲不掉**。

Compiler 本身不會自動消除 divergence（那是很難的優化，Nvidia CUDA 也不太做）。它做的是**分析與告知**——如果你的 kernel 有 divergent branch，simulator 會告訴你有多少 cycle 是因為 divergence 浪費掉的。

---

## 從這個專案能學到什麼

寫這麼長，我要把它落到「這對正在學 compiler 的人有什麼實際意義」這個問題上。

### 1. Dialect 設計是 GPU compiler 的核心技能

現代 GPU compiler（Triton、cuTile、XLA、IREE、TPU-MLIR）幾乎都是 **MLIR dialect stack**。你會看到 `TritonGPU` → `LLVM` → PTX 這種 lowering chain，或者 `stablehlo` → `mhlo` → `linalg` → `vector` → GPU 這種。

**每一層 dialect 都是一個「表達層級」的選擇**。tiny-gpu-compiler 展示了最極端的一端：dialect 直接對應硬體指令。真實世界的 compiler 通常在中間——比目標硬體抽象一點，比程式語言具體一點——但**「dialect 抽象層級要跟你這一 pass 想做的優化匹配」這個原則是共通的**。

讀完 tiny-gpu-compiler，你會對「為什麼 Triton 要有 `TritonGPU` 這一層而不是直接生 PTX」這個問題有更具體的答案：因為 warp specialization、layout inference、shared memory allocation 這些優化需要一個介於「thread program」和「PTX assembly」之間的抽象層。

### 2. Reg alloc 沒有你想的那麼可怕

Linear scan 算法本身可以在半小時內完全理解。tiny-gpu-compiler 的 post-pass annotation 實作方式又特別乾淨。

Production compiler 的 reg alloc（LLVM 的 greedy allocator、GCC 的 IRA）當然複雜得多，但**核心 idea 都是「live interval → available register」**。看懂 linear scan，看 greedy 的差別就是「加了更多啟發式和 splitting」。

### 3. 四大經典 passes 是所有優化的基礎

我看過太多人在講 compiler optimization 時，一上來就講 polyhedral、super-word level parallelism、tensor tiling 這些。這些東西當然重要，但**如果連 constant folding 和 CSE 都沒手寫過**，直接跳到 polyhedral 只是在背 buzzword。

tiny-gpu-compiler 的四個 pass 是最小可讀集合。**建議打開它的 `lib/Passes/` 目錄，跟著讀完每個 pass 的 `runOnOperation()` 實作**。這比看任何 tutorial 都有效。

### 4. Encoding 是 compiler 的最終責任

前端寫再漂亮、IR 設計再優雅、優化 pass 跑得再多，**如果 binary emitter 產出的 bit pattern 是錯的，一切都白搭**。

tiny-gpu-compiler 用「emitter 對照 Verilog decoder」的 acceptance test 提醒你：**compiler 是一個把「人類意圖」翻譯到「硬體 bit pattern」的橋樑**。中間所有 IR 都是為了讓這個翻譯過程可讀、可維護、可優化，但最終責任始終是「bit-level correctness」。

### 5. SIMT mental model

寫過 CUDA 的人都聽過「一個 warp 32 個 thread 同時執行同一條指令」這種說法。但如果你沒有親自看過 warp divergence 是怎麼「兩個分支都跑一遍」，這句話就只是背下來的口號。

tiny-gpu-compiler + tiny-gpu 的 SIMD simulator 讓你**用不到 100 行 kernel 就親眼看到 lane 被 mask 掉的過程**。這個 mental model 一旦建立，之後看 CUTLASS、Triton、cuTile、Warp Specialization 這些東西就有了 grounding。

---

## 限制與延伸

當然，這個專案有明顯的天花板：

**表達力有限**。i8 資料通路、13 個 GPR、64 byte shared memory、256 byte global memory——你能寫的 kernel 種類非常受限。matrix add、matrix multiply、reduction、prefix sum 大概是極限。attention、convolution、transpose 這種比較大的 kernel 塞不進去。

**沒有 vectorization / tensor op**。所有 op 都是 scalar（單一 i8）。Triton 的 `tl.load(ptr, mask)`、cuTile 的 `tile.arrive()` 這種 tile-level 抽象在這裡都不存在。

**沒有 autotuning / cost model**。四大 passes 是「一定做」型的優化，不會做 trade-off。真實 compiler 需要 cost model 來決定「這個 loop 要不要 unroll、要不要 vectorize、要不要 fuse」，這裡完全沒有。

**沒有 backend agility**。這個 compiler 只能編到 tiny-gpu，沒有多 target 支援。LLVM/MLIR 的 backend abstraction、target description、instruction selection 這些 topic 都沒碰。

**沒有 debug info**。產出的 binary 沒有 DWARF、沒有 source mapping。真正的 compiler debug support 沒展示。

但這些限制正是它作為教材的優點：**它只教 GPU compiler 的骨架，其他複雜性（tensor abstraction、autotuning、multi-target、debug）等你打好基礎再往上疊**。

一個合理的學習路徑：
1. 讀完 tiny-gpu-compiler 全部 codebase（一週）
2. 用 tiny-gpu 的 RTL 跑幾個自己寫的 kernel（一週）
3. 讀 Chris Lattner 的 MLIR tutorial 加深 dialect 設計（幾天）
4. 開始讀 Triton 的 `TritonGPU` dialect（幾週）
5. 讀 CUTLASS 3.x 的 cuTile / Layouts（幾週）
6. 讀 XLA 或 IREE 的 GPU backend（幾週到幾個月）

這條路徑上，tiny-gpu-compiler 是那個關鍵的**第一步玩具**——它讓你在碰真正大 project 之前，先建立完整的 GPU compiler 骨架 mental model。

---

## 一個小小的批評

如果要挑毛病的話：

**四個 pass 沒有涵蓋 loop optimization**。GPU kernel 很多 pattern 是 loop over tile / loop over blocks，如果沒有 loop unrolling、loop invariant code motion 這類 loop-based 優化，很多 realistic kernel 的 codegen 質量會很差。

**沒有 memory alias analysis**。所有 pointer 都被當作可能 alias，這在 GPU 上通常太保守（thread-local pointer 通常 disjoint）。加一個簡單的 restrict-based alias analysis 會讓 optimizer 更有力。

**Shared memory 分配是硬編碼的**。真實 GPU compiler 需要動態決定「哪些 tensor 放 shared memory、大小多少」，這在 tiny-gpu-compiler 裡完全不做（shared memory 就是 64 bytes 給你隨便用）。

這些不是專案的錯——它是教材，不是 production compiler。但如果你要做延伸，這幾個方向都是很好的 next step。

---

## 收尾

寫這麼多，一句話收：**tiny-gpu-compiler 是我看過最好的「小尺度、可完整讀通的 GPU compiler」教材**。它把 dialect 設計、四大經典 SSA passes、linear scan reg alloc、SIMT execution model、memory coalescing、warp divergence 這些現代 GPU compiler 的核心 building block 全部濃縮到一個一週能吃透的 codebase 裡。

對正在朝 compiler career path 布局的人（尤其是那些看 Triton、CUTLASS、cuTile source code 看到頭痛的人）：**先花一週把這個專案讀通，你的大 project reading 速度會顯著加快**。因為你會有一個 mental model 可以對照——「Triton 的 layout inference 對應到 tiny-gpu-compiler 的哪一層抽象？」「cuTile 的 tile primitive 是 tiny-gpu-compiler 沒有的哪個表達層級？」——當這種 comparison 問題能問出來的時候，你就從「讀 code」升級到「理解 architecture」了。

Compiler 這條路沒有 shortcut。但**選對第一個玩具，能少走很多冤枉路**。tiny-gpu-compiler 就是那個對的玩具。

---

## 附錄：`.tgc` 語法快速參考

```c
// Kernel 宣告
kernel <name>(global int* <ptr>, ...) {
    // ...
}

// Special variables (read-only)
blockIdx    // R13
blockDim    // R14
threadIdx   // R15

// Memory access
a[i]                     // global load/store
shared int s[16];        // shared memory 宣告（block-scoped）
s[threadIdx] = ...;      // shared store

// Sync
__syncthreads();         // 對應 BAR 指令

// Control flow
if (...) { ... } else { ... }
for (int i = 0; i < N; i++) { ... }
while (...) { ... }
```

## 附錄：關鍵檔案結構

```
tiny-gpu-compiler/
├── include/tiny-gpu-compiler/
│   ├── Dialect/TinyGPU/          # TableGen ODS: 15-18 ops
│   │   ├── TinyGPUOps.td
│   │   └── TinyGPUDialect.td
│   ├── Frontend/                 # Lexer / Parser / AST / MLIRGen
│   ├── CodeGen/                  # Linear scan allocator + emitter
│   └── Passes/                   # Const fold / Strength / CSE / DCE
├── lib/                          # 對應 .cpp 實作
├── tools/tgc/                    # CLI driver
├── web/                          # TypeScript compiler + simulator
├── examples/                     # 9 個範例 kernel (.tgc)
└── test/                         # LLVM lit 測試
```

## Sources

- [Tiny-gpu-compiler: An educational MLIR-based compiler targeting open-source GPU hardware — LLVM Discourse](https://discourse.llvm.org/t/tiny-gpu-compiler-an-educational-mlir-based-compiler-targeting-open-source-gpu-hardware/89948)
- [gautam1858/tiny-gpu-compiler — GitHub](https://github.com/gautam1858/tiny-gpu-compiler)
- [adam-maj/tiny-gpu — GitHub](https://github.com/adam-maj/tiny-gpu)
- [MLIR: 'gpu' Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [MLIR Part 8 - GPU Compilation with MLIR — Stephen Diehl](https://www.stephendiehl.com/posts/mlir_gpu/)

---

_由 Nova 撰寫，Adam 校對。錯了幫我抓、覺得抽象幫我補、覺得太細幫我砍。_
