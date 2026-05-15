---
title: "1200 萬 Token 一口氣吃完：SubQ 把 Attention 從 O(n²) 拉回線性的代價與意義"
slug: subq-12m-context-subquadratic-attention
description: "Subquadratic 在 5 月發表的 SubQ 模型，宣稱把 attention compute 在 12M context 下壓低 1000 倍。本文拆解 SSA 稀疏注意力架構、RULER/MRCR 基準成績，以及它對 RAG、多 agent 設計的潛在衝擊。"
date: 2026-05-15
tags: [AI, LLM, Attention, 系統架構, 效能優化, Long Context]
category: AI Architecture
---

## 前言：Context Window 的天花板，從來不是「記不住」，而是「算不完」

每次有人吹噓「我們的模型支援 1M context」，我都想反問一句：**那實際丟 1M tokens 進去，你願意付的單價是多少？**

Transformer 的標準注意力機制是 O(n²)。Context 翻倍，attention compute 翻四倍。這就是為什麼長 context 模型在帳單上總是貴得離譜——不是因為廠商貪心，是因為數學就是這樣。

於是大家想了一堆替代方案：RAG、Sliding Window、Sparse Local Attention、各種 retrieval-augmented 設計。本質上都是「不要真的把整個 context 都塞進 attention」。

直到 5 月 5 日，一家叫 **Subquadratic** 的公司發表了 **SubQ**——他們宣稱這是業界第一個「fully subquadratic」的商用 LLM，原生支援 12M tokens context window。

值得拆開來看嗎？我認為值得。

---

## 核心宣稱：1000 倍 attention compute 縮減

SubQ 的賣點寫得很大膽：

- **12M tokens** 原生 context window
- 在 12M context 下，attention compute 比現有 frontier model **少約 1000 倍**
- 比 FlashAttention 快 **52 倍**（architecture-level 比較），少 **63%** compute
- 成本約是 Claude Opus 或 GPT-5.5 的 **1/5**
- 已募資 **2900 萬美元**，估值約 5 億美元

這些數字單看會以為又是另一個過度行銷的 startup pitch。但他們的基準成績其實有點看頭：

| Benchmark | SubQ | Claude Opus 4.6 / 4.7 |
|---|---|---|
| RULER 128K | 95.6% | 94.8% |
| MRCR v2 | 65.9 | 32.2 |
| SWE-Bench Verified | 81.8 | 80.8 |

RULER 是長 context 的 needle-in-haystack 標配；MRCR v2 是更難的多文件檢索-推理混合任務。SubQ 在這兩項都打贏 Claude Opus 4.7 的 production 版本——尤其 MRCR v2 的 65.9 vs 32.2，差距大到不像同一個量級的模型。

---

## 技術核心：SSA 是什麼？

SubQ 的架構名稱是 **SSA (Subquadratic Sparse Attention)**。從公開資訊整理，它的關鍵設計是：

> **For each query token, the model selects a small subset of positions to attend to based on content rather than fixed patterns, then computes exact attention only over those.**

這句話的重點在「**based on content**」。

過去的 sparse attention（Longformer、BigBird 等）多半是 **fixed pattern**——sliding window + global tokens，pattern 是預定義的。問題是：你怎麼知道某個 query 真正需要的資訊在哪？固定窗口很可能根本沒涵蓋到。

SSA 的作法是讓模型 **動態地** 為每個 query token 挑選少量需要 attend 的位置，然後只在那個小子集上計算 exact attention。這就把總計算量從 O(n²) 拉到接近線性——當 n 是 12M 時，1000 倍的差距並不誇張。

當然，這帶來一個顯而易見的工程挑戰：**如何在不掃過所有位置的情況下，挑出「正確」的子集**？這通常涉及某種 hash、clustering，或學習式的 retrieval。SubQ 沒有公開全部細節，但能在 RULER 95.6% 的成績下做到這個壓縮比，代表他們的 selection 機制是可用的，不只是 attend 到鄰近 tokens 就喊收工。

---

## 為什麼這件事重要？因為它打到了三個現存設計的根

### 1. RAG 的存在意義被部分稀釋

RAG（Retrieval-Augmented Generation）這幾年是長文本處理的標準解法：把外部知識切成 chunk，用 embedding 撈出 top-k，再塞進 context。但 RAG 有個本質問題——**chunking 一定會破壞跨段落的長距推理**。

如果 12M tokens 可以一次塞進去，且 attention compute 還比現有方案少 1000 倍，那「整個 codebase 一次餵進去」就變成可行選項。SubQ 自己也直接點名這個 use case：

> entire codebases, large document collections, large spreadsheets, database tables, or long-running interaction histories in a single pass

對程式碼分析、長合約解讀、跨文件研究這類任務，RAG 從「必須」變成「可選」。

### 2. Multi-agent 協作的部分理由消失

很多 multi-agent 系統的存在，本質上是因為單一 agent 的 context 不夠用，所以才把任務切成多個子 agent，各自處理一部分再彙整。

如果 context 一次能裝 12M tokens，那「為了繞過 context 限制而切 agent」就沒必要了。Multi-agent 仍然有專業化分工的價值，但「context 限制」這個推力會消失一塊。

### 3. Long-context 的單價結構會被壓平

目前長 context 的 pricing 通常是線性甚至超線性的——你用 100K context 不只付 100 倍 1K 的錢，還可能更多。SubQ 把架構推向線性，意味著 **1M tokens 的單價可以跟 100K 接近一個量級**，而不是 100 倍。

這對「需要長 context 但不能燒錢」的應用（個人助理、長期對話、私人知識庫）是真的解放。

---

## 我的保留意見：別急著歡呼

SubQ 的成績很漂亮，但有幾件事我會持保留態度：

**第一**，benchmark 跟 production reliability 是兩回事。RULER 95.6% 很好看，但 RULER 是合成基準，跟真實長文檔的雜訊、結構、語義密度都不同。要看實際表現，還是得 dogfood。

**第二**，SubQ 在 SWE-Bench Verified 上是 81.8，僅僅微幅超越 Claude Opus 同級的 80.8。這代表它的 **核心推理能力沒有顯著突破**——優勢主要在 long context 的處理效率，而不是「智商」變強。對需要深度推理而非長文吸收的任務，這個模型未必更好。

**第三**，SSA 的 selection 機制是它的核心 IP，公開資訊很有限。一個你看不到實作細節的「魔法盒子」，在真實場景遇到 edge case 時要怎麼除錯、怎麼優化？這對 enterprise 採用是真實阻力。

**第四**，目標明年 Q4 衝到 50M context window。我不懷疑技術可以擴展，但 50M tokens 的應用場景到底有多少？這可能是「能做」勝過「值得做」的典型範例。

---

## 對工程師的實際意義

如果你是在做以下事情，SubQ 這類架構值得認真看：

- **Code analysis tools**：把整個 monorepo（不誇張，是整個）塞進去做跨檔案推理
- **Legal/Research**：跨數百份文件的引用追蹤與一致性檢查
- **個人 AI Agent**：累積數年對話歷史，不需要外掛 memory store
- **Database query helper**：直接塞 schema + sample data，不用 retrieval

如果你是在做：

- **Real-time low-latency inference**（chat、code completion）
- **Math/reasoning-heavy tasks**（純推理深度仍是 frontier model 強項）
- **多模態為主的應用**（SubQ 目前是純文字）

那 SubQ 暫時還不會是首選。

---

## 結語：架構創新終於回來了

過去兩年的 LLM 進展，老實說大多是 **scale + post-training tricks**——更多參數、更好的 RLHF、更多 reasoning data。架構層面真正的突破不多，多數人乾脆假設 transformer 就是終局。

SubQ 證明了一件事：**Attention 的 O(n²) 不是物理定律，是設計選擇**。當有人願意從第一性原理重做時，可能省下的不只是錢，是整個應用模式的天花板。

不是說 SubQ 一定會成功——startup 高淘汰率、競爭對手會快速跟進——但這個方向（content-based 動態 sparse attention）會是接下來 1-2 年很多大公司也會走的路。如果 SubQ 的數字能在 production 環境驗證，這可能是繼 FlashAttention 之後，attention 機制最重要的工程突破。

值得追蹤。

---

## 延伸閱讀

- [Subquadratic 官方介紹](https://subq.ai/introducing-subq)
- [The New Stack: Subquadratic debuts 12M context window](https://thenewstack.io/subquadratic-12-million-context-window/)
- [SiliconANGLE: Subquadratic launches with $29M](https://siliconangle.com/2026/05/05/subquadratic-launches-29m-bring-12m-token-context-windows-ai/)
- [DataCamp: SubQ AI Explained](https://www.datacamp.com/blog/subq-ai-explained)
- [explainx.ai: SSA sparse attention deep dive](https://explainx.ai/blog/subq-ssa-sparse-attention-12m-context-2026)
