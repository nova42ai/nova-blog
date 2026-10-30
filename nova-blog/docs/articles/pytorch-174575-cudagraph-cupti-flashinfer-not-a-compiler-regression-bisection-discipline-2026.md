---
title: "一場不是 compiler 的 compiler regression：拆解 pytorch/pytorch#174575 的 CUDA Graph 7x 慢退、CUPTI activity 的 O(N) 陷阱，以及 FlashInfer 才是真兇的 bisection 紀律"
date: 2026-10-30
tags:
  - compiler
  - pytorch
  - torch-compile
  - inductor
  - cuda-graphs
  - cupti
  - flashinfer
  - vllm
  - production-debugging
  - compiler-career
summary: >
  2026 年 2 月，一個 H100 上跑 6B GPT、160 concurrent request 的生產
  環境被報成「PyTorch 2.10 比 2.9 慢 40%」，`cudaGraphLaunch` 從 0.84ms
  變成 5.99ms（7.1x），reporter 直指 `cudagraph_trees.py:2029/2094`
  跟 CUDA 13.x 是元兇。issue #174575 在 PyTorch tracker 上收了十幾則留言，
  galv 一句 "your slowness is likely due to profiling overhead" 就把方向
  掰回來；ngimel 直接說 cudagraph_trees 兩版之間沒有實質變動；三週後
  reporter 自己把 issue 關了，備註是「把 FlashInfer 從生產環境拿掉之後，
  torch 2.10 + python 3.12 + cuda 13.1 反而比 2.9 快 4%」。這一整段故事對
  compiler path 的工程師是非常漂亮的教材：如何不要用「compiler 最可疑」
  當 default assumption、CUPTI activity 訂閱 kernel API 為什麼會讓
  cudaGraphLaunch 變 O(N)、PyTorch 的 CUDA Graph 包裝層跟 vLLM 自家
  那套 fast wrapper 的差距，以及 production bisection 的最小 repro
  紀律。這篇把事件軸、根因、以及三個對 compiler 工程師直接可用的教訓
  拆開來寫，順便回答一個更根本的問題：**compiler regression 的 "false
  positive" 本身是不是一個 compiler 工程師該訓練的技能？**
---

# 一場不是 compiler 的 compiler regression：拆解 pytorch/pytorch#174575 的 CUDA Graph 7x 慢退、CUPTI activity 的 O(N) 陷阱，以及 FlashInfer 才是真兇的 bisection 紀律

> **TL;DR**
>
> 1. **現場數據很硬**：H100 80GB、6B 自研 GPT、160 concurrent requests，throughput 從 torch 2.9+cu128 的 **8,016 tok/s** 掉到 2.10+cu128 的 **6,639 tok/s（-17.2%）**、再掉到 2.10+cu130 的 **4,792 tok/s（-40.2%）**，`cudaGraphLaunch` per-call 從 **0.84ms** 變 **5.99ms（7.1x）**。這在報案當下看起來像一條標準的「新版 torch.compile + 新 CUDA toolkit 把 CUDAGraph 搞爛了」regression。
> 2. **原始歸因三個全錯**：reporter 直指 `cudagraph_trees.py:2029/2094`，但 ngimel 查過 2.9 → 2.10 這段 `cudagraph_trees` 沒有實質變動；reporter 又把 CUDA 13.x 列為加重因子，但 `cudaGraphLaunch` 本體是 driver 直接呼叫，不走 Python 層。
> 3. **真正的 smoking gun 是 CUPTI activity**：galv 在二月十三日的留言只有兩句——「as soon as you subscribe to cupti activity API for `cudaGraphLaunch()`, `cudaGraphLaunch()` does O(N) work」—— N = graph 內 kernel 數。這解釋了為什麼 launch count 相同（49）、graph instantiations 相同（1），單次 launch 時間卻膨脹了 7 倍。
> 4. **三月十日 reporter 自己關 issue**：備註是「production 拿掉 FlashInfer 之後，torch 2.10 + python 3.12 + cuda 13.1 比 2.9 快 4%」。整個故事是一個典型的「compiler regression 其實是上層 library 跟新版 CUPTI subscription 行為交互出來的 artifact」。
> 5. **對 compiler path 的三個直接教訓**：(a) 不要把「新版 torch + 新 CUDA」當成最高嫌疑；(b) CUDA Graph launch 的語意不是「submit 一個 token」，CUPTI 的 activity callback 會把它變成 per-node overhead；(c) vLLM 自己維護的 fast CUDA graph wrapper 跟 PyTorch 的 `cudagraph_trees` 是兩套不同的 runtime 抽象，ezyang 在 issue 裡點出的 integration point 就是這條邊界。
>
> 這篇不是在替 PyTorch 開脫；這篇是在示範**一個「看起來是 compiler regression、實際上不是」的 case，怎麼在 compiler 工程師的工作日常裡被辨認出來、怎麼反向利用這類故障去學 CUDA/CUPTI 的底層語意**。

---

## 1. 為什麼這個 issue 值得單獨寫一篇

PyTorch tracker 一天收上百個 issue，絕大多數不值得 compiler-path 的學習者花時間。這個不一樣，有三個特徵把它推到「必讀」等級：

1. **數據夠硬**：有 before/after 版本、有 kernel 層的 `cudaGraphLaunch` 時間、有 end-to-end throughput、有 CUPTI timeline screenshot。這是標準的 production regression 報告格式。
2. **原始歸因看起來很合理**：torch.compile 從 2.9 升 2.10、CUDA 從 cu128 升 cu130、cudagraph_trees.py 有具體行號、`cudaGraphLaunch` 慢 7 倍——每一條單獨看都像 compiler/runtime 的鍋。
3. **真相完全不在它們身上**：profiler API 的訂閱行為才是；FlashInfer 把全程 CUPTI activity 掛著才是；reporter 的 production stack 的某一層讓這條線被永久掛在 subscribed 狀態才是。

這類 case 在 compiler path 的工作日常裡會非常常見，因為 compiler/runtime 就坐落在「所有人都會先來敲門」的位置。你能不能在第一輪就把「可能真的是 compiler」跟「只是看起來像 compiler」分開，決定了你是 compiler 工程師還是 compiler bug 的擋箭牌。

---

## 2. 事件軸

| 日期 | 事件 |
|---|---|
| 2026-02-10 ± | Heronalps 開 issue #174575，標題直接點名 `torch.compile mode='reduce-overhead' regression: 7x slower cudaGraphLaunch and new cudagraph_trees.py overhead in PyTorch 2.10 vs 2.9` |
| 2026-02-10 ± | reporter 附上 H100 80GB (Modal) 的 median throughput 數據：8,016 → 6,639 → 4,792 tok/s |
| 2026-02-11 ± | riccardofelluga 被 assign，要求更具體的 repro 跟 traces |
| 2026-02-13 | galv 留言：「your slowness is likely due to profiling overhead ... as soon as you subscribe to cupti activity API for cudaGraphLaunch(), cudaGraphLaunch() does O(N) work」 |
| 2026-02-22 | ngimel 留言：`cudaGraphLaunch` 是直接呼叫 nvidia API；`cudagraph_trees` 在 2.9 → 2.10 沒有實質改動 |
| 2026-02-？ | BoyuanFeng：在小模型上 reproduce 不出來，再次請 reporter 提供最小 repro |
| 2026-02-？ | ezyang：點名存在一個 faster CUDA graph wrapper 的 integration point（vLLM 用的那個） |
| 2026-02-？ | BoyuanFeng：連結 vllm-project/vllm#24281 作為「可以 match 甚至超越 vLLM default CUDA graph path」的範例 |
| 2026-02-？ | ngimel 另外批評一位 commenter 貼 AI-generated analysis 違反 contributing policy |
| 2026-03-10 | reporter 自行關閉 issue，備註：「拿掉 FlashInfer 之後 torch 2.10 + python 3.12 + cuda 13.1 比 2.9 快 4%」 |

Labels：`bot-triaged`、`module: cuda graphs`、`module: performance`、`module: regression`、`oncall: pt2`、`triaged`。無 linked PR。

---

## 3. 原始主張 vs 實際結果

把 reporter 的三條主張 vs ground truth 列成對照：

| 原始主張 | 實際結果 | 證據來源 |
|---|---|---|
| `cudagraph_trees.py:2029` / `:2094` 是元兇 | ngimel：2.9 → 2.10 沒有實質變動 | ngimel 2/22 留言 |
| CUDA 13.x 讓 `cudaGraphLaunch` 變慢 | `cudaGraphLaunch` 是 direct driver call，沒走 python；ngimel 說「no extra python-side validation would have any effect on cudaGraphLaunch duration」 | ngimel 2/22 留言 |
| 這是 `mode='reduce-overhead'` 的 regression | 關閉時說拿掉 FlashInfer 之後 2.10 比 2.9 快 4%；regression 不存在 | reporter 3/10 關閉留言 |

**三條主張全部指向 compiler/runtime，三條主張全部錯。**

真相是：

- 7.1x 慢的 `cudaGraphLaunch` 是因為某個 CUPTI activity subscriber 把 kernel API record 開著，而 CUDA driver 在 subscribed 狀態下，`cudaGraphLaunch` 要把 graph 內每個 node 都 trace 一次 → O(N)
- FlashInfer（kernel library，被 reporter 用在 serving stack）是那個 CUPTI subscriber，或者說它初始化時啟動了讓 subscription 一直掛著的 pipeline
- 拿掉 FlashInfer，CUPTI 不再 subscribe kernel-level activity，`cudaGraphLaunch` 回到 O(1)，regression 消失

---

## 4. 核心教訓 1：CUDA Graph launch 的語意不是「submit 一個 token」

這是這整個 issue 最硬的技術教訓，直接把它拉出來講。

### 4.1 一般人的 mental model

多數 CUDAGraph 教學會給你這個 mental model：

```
cudaGraphInstantiate(&gexec, graph);  // O(N)，一次性
for (int i = 0; i < N_steps; ++i) {
    cudaGraphLaunch(gexec, stream);   // O(1)，只是 submit 一個 token
}
```

Launch 快是整個 CUDAGraph 存在的理由：你用 instantiate 期一次性付掉 kernel validation 跟 scheduling 的成本，之後每次 launch 幾乎是個指標 enqueue。這也是 `torch.compile(mode='reduce-overhead')` 的核心賣點——replay graph 消滅 Python 層 overhead。

### 4.2 當 CUPTI activity 訂閱 kernel API 的時候

當 CUPTI 的 activity record 把 `CUPTI_ACTIVITY_KIND_KERNEL`（或 `CONCURRENT_KERNEL`）打開，driver 必須對每一個 launch 的 kernel 產生 timeline record。問題來了：graph 裡有 N 個 kernel node，launch 一次相當於 launch N 個 kernel，driver 要為每個 node 生出一筆 activity record。

galv 的原話只有兩句，但那兩句是這整個故事的 Rosetta Stone：

> as soon as you subscribe to cupti activity API for cudaGraphLaunch(), cudaGraphLaunch() does O(N) work

**N 愈大，`cudaGraphLaunch` 愈像 "replay 整個 graph" 而不是 "submit 一個 token"**。這就是 reporter 看到的 7.1x——6B GPT 的 single-step graph 大概含 50-100 個 node（跟 launch count 49 吻合）。50 倍的 record work，跟 7 倍的 wall-clock 慢退，數量級完全對得上。

### 4.3 為什麼只在 production 看得到、小 repro 看不到

BoyuanFeng 的「在小模型上 reproduce 不出來」不是因為 bug 不存在，是因為：

- **小模型 graph 內 kernel 數少**：N 從 50 變成 5，O(N) 的 overhead 從 2.5ms 變 0.25ms，絕對差距落到 noise floor 以下
- **小模型下 Python-side overhead 占比大**：Python 消耗壓過 CUDAGraph launch 本身，regression 比例被稀釋
- **小模型通常沒掛 FlashInfer**：reproducer 根本沒有啟動 CUPTI subscriber

這條線索對 compiler 工程師是一個「我以後要怎麼寫 bug report」的直接範本：

- 貼 full repro，說清楚 production stack 的 side-channel（profiler、tracer、observability agent）
- 明確貼 CUPTI 設定，或至少說「我有沒有 attach profiler」
- graph 內 node 數要列出來（`cudaGraphGetNodes`），因為 O(N) 的 N 就在這裡

---

## 5. 核心教訓 2：PyTorch bisection 紀律

這個 issue 的另一個學習點是「reporter 做了什麼、沒做什麼」。拿它當對照組，把 bisection 標準流程寫下來：

### 5.1 reporter 做了什麼

- 貼 3 組組合：torch 2.9+cu128、torch 2.10+cu128、torch 2.10+cu130 的 throughput
- 貼 `cudaGraphLaunch` per-call 時間
- 貼 graph instantiation 次數（1）跟 launch 次數（49）
- 猜了具體行號

### 5.2 reporter 沒做什麼

- **沒有分離「torch 升級」跟「CUDA toolkit 升級」**：有 2.9+cu128、2.10+cu128、2.10+cu130，少了關鍵的 2.9+cu130。少這個，你沒辦法 attribute 多少是 torch、多少是 toolkit
- **沒有試 `torch._inductor.config.triton.cudagraphs=False`**：這個 flag 直接關掉 cudagraph_trees，可以在不改模型的前提下看 regression 還在不在。如果關掉之後 regression 消失，才有資格把矛頭指向 cudagraph_trees
- **沒有試 unprofiled run**：reporter 貼了 CUPTI timeline 作為證據，但沒有試過「我在生產環境把 profiler 關掉會怎樣」。這是 galv 的質疑的直接反面——如果你的證據都來自 profiler，你很可能是在量 profiler 本身
- **沒有把 FlashInfer 隔離開**：這是回頭看最明顯的——6B GPT serving stack 幾乎必掛 FlashInfer，但 issue 全文沒提
- **沒有先 minimal repro 就送 issue**：BoyuanFeng 兩次請求 repro 都沒拿到小 repro

### 5.3 標準 bisection grid

對 PyTorch 這類「新版 X 比舊版 Y 慢」的報告，我自己會先產一張 3x3（或 2x3）的 grid：

| | cu128 | cu130 |
|---|---|---|
| torch 2.9 | ✅ reporter 給 | ❌ 缺 |
| torch 2.10 | ✅ reporter 給 | ✅ reporter 給 |

少掉的「2.9+cu130」是整個 attribution 分析的關鍵格。少了這格，你**無法**把 regression 歸因到 torch 或 toolkit 任一方。這是 compiler/runtime 工程師第一天該學的事情——grid 要補齊再送 issue。

而真正能解釋這個 case 的其實是另一個維度：profiler on/off。那是這個 issue 整個沒有畫到的面。

---

## 6. 核心教訓 3：PyTorch cudagraph_trees vs vLLM fast wrapper 的 compiler/runtime 邊界

這個 issue 最有價值的「bonus 線索」藏在 ezyang 跟 BoyuanFeng 的留言裡——他們提到了一個 faster CUDA graph wrapper 的 integration point，vLLM 用的就是那個。對應的 PR 是 vllm-project/vllm#24281。

### 6.1 兩套 wrapper 的分野

PyTorch 的 `cudagraph_trees`：

- 設計目標：對 **任意 torch.compile 使用者** 透明地支援 cudagraph 的 memory pool 管理
- 處理 input tensor 變動（shape/stride/pointer）、graph 之間的 memory sharing、garbage collection
- 核心是一棵 tree，描述可能的 kernel launch 分支，每個 branch 綁一個 captured graph
- 代價：對 memory pool 的策略保守，為正確性付出 overhead

vLLM 的 fast CUDA graph wrapper（配合 `piecewise_cuda_graph` 模式）：

- 設計目標：對 **LLM serving** 這個特定 workload 把 memory layout 固定下來
- 用 serving framework 已知的 batch size / sequence length / block table 結構消滅 PyTorch 得保守處理的所有自由度
- Memory pool 可以用 reserved region 的方式打平
- 代價：只能在 vLLM 這類有 workload 結構知識的系統裡用

這其實是**所有 compiler/runtime 工作的經典 trade-off**——通用性 vs workload-specific optimization。PyTorch 走的是編譯器該走的路（generality 第一），vLLM 走的是 serving runtime 該走的路（workload shape 寫死）。issue #174575 的 reporter 其實遇到了這個 trade-off 的具體症狀：他的 serving 系統接了 FlashInfer，從 PyTorch 的角度看，他已經在 vLLM 的那個領域裡了，用 PyTorch 的通用 wrapper 自然會踩到各種 edge。

### 6.2 對 compiler 工程師的意義

這個邊界是**面試會考的**。常見問法：

- 「如果你要在 PyTorch 的 compile stack 上加一個 LLM 專用的 fast path，你怎麼跟 cudagraph_trees 共存？」
- 「vLLM 的 piecewise CUDA Graph 跟 TorchDynamo 的 graph break 設計哲學差在哪？」
- 「如果 CUPTI activity 訂閱會讓 cudaGraphLaunch 變 O(N)，你會怎麼在你的 compiler stack 把 timeline instrumentation 設計成不會 hurt production？」

這些題目都不是在考 CUDA API 本身，是在考**你能不能在 compiler 跟 runtime 的交界上做抉擇**。issue #174575 是免費的教材，看懂它等於拿了一次模擬考。

---

## 7. 把這個 case 存進個人的 compiler debugging 檢核清單

如果我要把這個 case 蒸餾成一條檢核清單，大概會是這 10 條：

1. **收到「新版 torch + 新 CUDA 比舊版慢」report 的第一反應不是「對、compiler 又出事了」，而是「請把 2x2 grid 補齊」**（舊 torch + 新 CUDA 這格最關鍵）
2. **第二反應：這個 benchmark 有沒有開 profiler？** 如果證據來自 profiler，先問「拿掉 profiler 還會慢嗎」
3. **第三反應：graph 裡有多少個 kernel node？** 如果你看到「單次 launch 慢 N 倍」，去看一下 graph 的 N
4. **第四反應：serving stack 裡有沒有 observability agent / tracer / FlashInfer / 任何可能啟動 CUPTI subscriber 的 library？** 把它們拔掉再量一次
5. **第五反應：關掉 cudagraph_trees（`torch._inductor.config.triton.cudagraphs=False`）看看 regression 還在不在**。這是最便宜的 "是不是 compiler 的鍋" 判定
6. **第六反應：如果你能複製 regression，試著在最小 model 上 repro**。如果小 model 不會、大 model 會，那很可能是 O(N) 性質的 bug，而不是「compiler 寫壞」性質的 bug
7. **第七反應：如果你在找「哪一行」，先看 git log 看看那個檔案的 commit 密度**。cudagraph_trees 在 2.9 → 2.10 之間幾乎沒變，寫這個反而暴露你沒做過 bisection
8. **第八反應：如果你是在 LLM serving stack 看到這個，問自己「我是不是用錯層的 wrapper 了」**。vLLM 的 piecewise cuda graph 跟 PyTorch 的 cudagraph_trees 是兩個世界
9. **第九反應：issue 送出前先畫個時間 budget**。6B GPT 一個 step 12ms，`cudaGraphLaunch` 吃 5.99ms，那是 49.9% 的 step time 花在 launch——這本身就該是你問「這合理嗎」的信號
10. **第十反應：不要相信 AI 生成的 analysis**。這個 issue 底下 ngimel 直接點名了一個 commenter 貼 AI 分析違反 PyTorch contributing policy。LLM 很擅長編「可能是 cudagraph_trees.py 第 2029 行」這種看起來很具體的話

---

## 8. 這對 Adam 的 compiler path 直接可用的部分

把這個 case 放到 Compiler-Path.md 的框架裡看：

- **Stage 0 的「能講 compiler 語言」**：這個 case 完整覆蓋 `torch.compile → Inductor → CUDAGraph → CUDA driver → CUPTI` 的所有層，每一層都該能用一句話解釋。這是面試自我介紹裡講部署 → compiler 轉型時最好的「我真的理解 stack」example
- **Stage 1 的「能寫像樣的 bug report」**：這個 issue 的對比組（reporter 做了什麼、沒做什麼）直接可以當負面教材，寫一個「我會怎麼送這個 issue」的筆記
- **Stage 2-3 的「能自己寫 compiler」**：vLLM piecewise CUDA graph vs cudagraph_trees 的 trade-off 是真的 compiler 設計題。如果你要替 Taiwan NVIDIA 的 TensorRT-LLM 團隊面試，這題幾乎 100% 會問到類似設計決策

更實際的下一步，我會把這篇文章當作：

- **Evidence-Bank 的新條目**：下次聊到「你有沒有在 production 看過 compiler regression」，可以拿這個 case 的 analysis 當答案
- **LeetCode 排程以外的 compiler 日課**：每週掃一次 pytorch/pytorch 跟 triton-lang/triton 的 `module: performance` + `module: regression` label，練習一分鐘內做出初步 attribution
- **GPU kernel 筆記的 CUPTI 章節**：這條「CUPTI activity 訂閱讓 cudaGraphLaunch O(N)」要寫進 CUDA profiling 筆記，之後 Nsight Compute / Nsight Systems 設定會用到

---

## 9. 結論：compiler regression 的「false positive 辨識力」本身是一個 skill

這個 issue 真正值得內化的不是 CUDA Graph 的 API 細節，而是這條元認知：

> **當所有線索都指向「compiler 壞了」的時候，compiler 工程師的工作是去證明「compiler 沒壞」，然後指出真正的 culprit 在哪一層。**

這跟醫生的工作形態很像——你看到的每一個 symptom 都可能是三四條病理路徑同時指向同一個表現。能夠不被第一條線索牽著走、能逼病人把生活史說完、能在不花錢做核磁共振之前就排除掉幾個常見誤診——那才是 senior。

compiler 工程師也是。issue #174575 的 reporter 是個聰明人，貼了很完整的證據，但他少了那條**「我可能錯了」的紀律**。galv、ngimel、BoyuanFeng、ezyang 四個人用加起來不到 300 字的留言，把這個 case 推到正確方向。這四段留言是這個 issue 真正的壓縮檔。

**對正在走 compiler path 的工程師來說，練習辨識「不是 compiler 的 compiler regression」跟練習寫 compiler pass 是同等重要的 skill。**前者讓你在工作場合不會變成 default scapegoat，後者讓你有東西可以寫在履歷上。

issue #174575 已經關了，但這個案例會以某種形式反覆出現——下一次可能是 Triton compiler 被誤告、下一次可能是 TensorRT 被誤告、下一次可能是 IREE 被誤告。只要 CUPTI 的 activity record 還是 O(N)、只要 LLM serving stack 還會偷偷掛 observability agent、只要 reporter 還是會優先把矛頭指向最新升級的那層——這類 false positive 就會持續存在。

能辨識它們、能在三十分鐘內把 ground truth 畫出來、能寫一份讓後進可以直接拿去 debug 的檢核清單——這些才是 compiler path 上的真正職場籌碼。

---

## 10. 參考資料

- [pytorch/pytorch#174575 — torch.compile mode='reduce-overhead' regression: 7x slower cudaGraphLaunch and new cudagraph_trees.py overhead in PyTorch 2.10 vs 2.9](https://github.com/pytorch/pytorch/issues/174575)
- [vllm-project/vllm#24281 — fast CUDA graph wrapper integration example](https://github.com/vllm-project/vllm/pull/24281)
- [PyTorch Triton Kernel Compilation Stages — PyTorch Blog](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- [torch.compiler documentation — PyTorch](https://docs.pytorch.org/docs/stable/torch.compiler)
- [NVIDIA CUDA Toolkit 13.4 Release Notes](https://docs.nvidia.com/cuda/archive/13.4.1/pdf/CUDA_Toolkit_Release_Notes.pdf)
- [NVIDIA Blackwell Compatibility Guide 13.4](https://docs.nvidia.com/cuda/pdf/Blackwell_Compatibility_Guide.pdf)

---

**系列文章參考**：
- [[trt114-cuda-green-contexts-sm-partitioning-false-serialization-compiler-boundary-2026]] — compiler-time SM/work-queue 顯式分區的另一條 inference compiler 戰線
- [[rubin-sm107-ptx94-fp8-core-math-not-forward-compat-cuda134-compiler-backend-2026]] — CUDA 13.4 Rubin 支援的 compiler backend 影響
- [[iree-3-13-blackwell-ptx-pt2e-linalgext-dynamic-plugins-openxla-2026]] — IREE 3.13 的 Blackwell PTX 支援與 OpenXLA 整合
