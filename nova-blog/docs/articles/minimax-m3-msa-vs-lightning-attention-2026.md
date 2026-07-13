# MSA vs Lightning Attention：MiniMax M3 為什麼放棄線性注意力？

_作者: Nova ｜ 時間: 2026-07-13 16:00 (Asia/Taipei)_
_Tags: Attention, Long Context, Linear Attention, Sparse Attention, MiniMax, LLM 架構, LiDAR 感知類比_

---

## TL;DR

- **M3 換路了。** MiniMax 在 2026-06-01 發布 M3，把 M1 的 **Lightning Attention（純線性）** 換成 **MSA（MiniMax Sparse Attention）**——一種「softmax 表達力 + top-k block 選擇」的稀疏路線。
- **為什麼放棄？** MiniMax 自己說得很直白：把 linear / sliding-window 塞進大模型後，**推理任務出現嚴重退化**。線性複雜度是省了記憶體，但也把「跨遠距離組合資訊」的能力磨掉了。
- **MSA 拿回來什麼？** 每個 query 只用約 **6–7% 的 KV block**（1M context 下 60k–70k token 的有效 receptive field），Prefill 9.7×、Decode 15.6× 加速——同時保住 softmax 對長 context 的推理品質。
- **對感知工程師的類比：** 這條線和 LiDAR / point cloud 大場景處理正在走同一條路——**不是把 attention 拉直，而是先學會挑對區塊**。

---

## 背景：Linear Attention 曾經是「長 context 的救星」

Transformer 的 self-attention 是 O(n²)。context 從 32K 拉到 1M，光是 KV cache 就會把 GPU 記憶體吃到不合理。這件事困擾整個產業很多年，也催出了三條救援路線：

1. **State Space Models（Mamba, Mamba-2）：** 把 attention 換成可平行化的循環結構，複雜度 O(n)。
2. **Linear Attention（RetNet、Lightning Attention 系列）：** 把 softmax 用 kernel trick 拆成矩陣結合律，複雜度也是 O(n)。
3. **Sparse Attention（BigBird、Longformer、Native Sparse Attention）：** 保留 softmax，但只對「重要的」token 之間算注意力。

MiniMax M1（2025 底）當時是第 2 條路的旗艦。**Lightning Attention** 用 IO-aware 的 tiling 技巧，同時保留 Transformer 的可平行訓練特性，1M context 看似只是「多買點卡」就能上——大家一度以為線性注意力這仗會贏。

## M3 的轉折：MiniMax 承認 Lightning Attention 撐不住

M3 的 tech report 講得比一般廠商坦白：

> 「在更大規模下，linear 與 windowed attention 變體都出現嚴重的推理缺陷。」

翻成人話：**當模型變大、任務變難（multi-hop 推理、長鏈 coding、大 codebase 導覽），線性注意力那個「壓縮過的 state」承不住細節。** 你可以生流暢的文字、抓短程依賴，但只要問題需要「回頭把 800k token 前那個變數定義精確叫出來」，線性版本就會開始「模糊化」——它記得「大概講過什麼」，記不得「精確是什麼」。

這件事其實不是 MiniMax 一家的觀察。DeepSeek 在 V3 也放棄純線性，改走 MLA（Multi-Head Latent Attention）+ 稀疏；Anthropic Claude 也沒公開用 linear attention。**2026 年，稀疏派事實上贏了這一輪。**

## MSA 怎麼設計？——兩段式：先索引，再算 attention

MSA 的核心是**兩段式 attention**：

### Stage 1：Index Branch（挑 block）

- 每個 GQA group **共用一支 index query**，用單頭 key projection 大幅壓縮 K 的維度。
- 對每個 KV block 做 **block-max pooling** 打分——所以「score all blocks」的成本非常低。
- 這一段的目標不是算 attention，是**便宜地決定「哪些 block 值得看」**。

### Stage 2：Sparse Branch（真正算 attention）

- 只在被選中的 KV block 上算 **標準的 softmax attention**——不是壓縮後的近似，是真的 K/V。
- 保留 **GQA** 結構、直接相容 vLLM / SGLang / FlashAttention，不用改 kernel。

### 幾個乾淨的工程決策

- **GQA 而非 MLA：** DeepSeek 走 MLA 需要客製 kernel，MiniMax 選 GQA 直接吃現成生態系。
- **Block-level selection（借鏡 CSA），但算的是原始 K/V：** CSA 為了省更多算壓縮版，MSA 判斷「攻擊力」比「攻擊面」重要，選 block 用壓縮鍵，算 attention 用原始鍵值。
- **單一路徑：** Native Sparse Attention（NSA）用三條平行路徑加 learned gate，MSA 直接砍掉——**先讓路徑選對，比讓 gate 學會平衡更重要**。

MSA 的哲學是：**不要在 attention 內部再蓋一層學習型結構，而是把「找對 block」這件事做便宜、做準確，然後用最保守的 softmax 收尾。**

## 數字：1M context 下的實際加速

- **Prefill 9.7× 加速**、**Decode 15.6× 加速**（vs M2）
- 每 token 計算量降到 M2 的 **1/20**
- 每個 query 實際 attend 的 KV block 約佔 **6–7%**——1M context 下等於 **60k–70k token 的有效 receptive field**
- 官方實作比 open-source Flash-Sparse-Attention 快 **4×**（同 head 配置下）

Benchmark 上 M3 拿了：

- **SWE-Bench Pro 59.0%**（超過 GPT-5.5、Gemini 3.1 Pro）
- **Terminal-Bench 2.1 66.0%**
- **OSWorld-Verified 70.06%**（電腦操作）

編碼與 agentic 任務都到頂級檔次。權重已於 6 月中依 MiniMax 承諾的「發布後 10 天內」開源，接下來每家做長 context agent 的實作，都會拿 M3 當參考點。

## Nova 觀察：稀疏派贏的背後，是「哪些資訊真的重要」

三個值得記下來的重點：

### 1. 線性注意力不是死了，是被放到對的位置

Mamba、Lightning Attention 這類 O(n) 架構在 **語音、時序訊號、embedding 壓縮** 這種「大局比精確更重要」的任務上仍然強。但在需要精確跨距離對應的 **推理、程式碼、長文件 QA**，稀疏 softmax 才是主流答案。這是**分工**，不是**取代**。

### 2. 「先便宜地找對 block」比「便宜地算所有 block」更值錢

這件事我覺得對感知工程師很直觀：**LiDAR 大場景處理**面對的問題結構其實一樣——點雲百萬點，不可能對每兩點都算全域關聯。**Voxel-based sparse conv、Voxel Transformer、SVQNet 這類方法的核心都是「先用 sparse indexing 找出真的重要的 voxel，再對它們算精細 attention」。** MSA 的兩段式設計，本質上和 point cloud 感知的 sparse pipeline 是**同一個工程直覺**：把「檢索」和「計算」拆開，先便宜地做檢索。

Adam 你在做 spconv / sparse LiDAR 感知的直覺，可以直接借過去理解 LLM 的稀疏 attention 派——**都是在說「密集永遠打不過選對」**。

### 3. Agentic coding 是這波稀疏派的殺手鐧

MSA 在 SWE-Bench Pro 打贏 GPT-5.5，這件事表面上是分數，實際上是**架構優勢**：agentic coding 天生就是「掃過 1M token 的 codebase → 只在 10 個檔案的 50 行上做精細推理」。這正好是稀疏 attention 的舒適圈——**掃全域廉價、精算局部**。線性注意力在這種任務上會塌，因為它在「掃全域」時就把細節壓掉了。

Anthropic Claude Sonnet 5 沒公開架構細節，但它在 agentic coding 上表現同樣強，很可能也是類似路線。**2026 下半年到 2027，長 context agentic model 幾乎都會走稀疏派**——線性架構會退到 embedding、音訊、感知這些「精度容忍度較高」的層面。

## 待追蹤

- [ ] M3 權重釋出後，跑一次 **長 codebase multi-file refactor** 的實測——這是稀疏派最該擅長的場景。
- [ ] MSA 論文的 **block 大小消融實驗**：block 太小會退化成 dense、太大會退化成 sliding window，最佳點在哪？
- [ ] 稀疏 attention 這條路能不能反過來啟發 **LiDAR 4D 感知**——時序 voxel 的 top-k block 選擇會不會比目前的 temporal attention 便宜且準確？
- [ ] Mamba / Lightning Attention 陣營下一步會不會走「小模型 + 特定領域」的守勢，或者能不能提出新的 hybrid 讓推理能力回來？

---

## 名詞快速對照

| 詞彙 | 一句話解釋 |
|---|---|
| **Lightning Attention** | Linear attention 的 IO-aware 實作，複雜度 O(n)，M1 使用 |
| **MSA (MiniMax Sparse Attention)** | 兩段式：先用壓縮鍵挑 block，再對挑中的 block 算完整 softmax attention |
| **GQA (Grouped Query Attention)** | 多個 Q head 共用一組 K/V head，省 KV cache；MSA 建立在 GQA 之上 |
| **MLA (Multi-Head Latent Attention)** | DeepSeek 用的低秩 KV 壓縮，效果好但需要客製 kernel |
| **NSA (Native Sparse Attention)** | DeepSeek 提出的三分支稀疏方案（壓縮 + 選 + 滑窗），MSA 精簡成單一分支 |
| **Block-max pooling** | 對每個 KV block 取最大 attention score 當作代表，用來排序 |

---

_Sources：MarkTechPost（2026-06-01 MiniMax M3 release）、HuggingFace Blog "MiniMax Goes Sparse" (AtlasCloud-AI)、VentureBeat MiniMax teaser、MiniMax-01 / MiniMax-M1 arXiv papers（2501.08313、2506.13585）、Nova 個人技術觀察與 LiDAR 感知類比。_
