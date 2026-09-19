---
title: "Physical AI 把 GPU 拖回作業系統核心：SOSP 2026 Linux AGX、Kairos、gpu_ext 三重奏——當 Jetson Thor 已經給了 MIG／Green Contexts，policy 由誰來寫？"
date: 2026-09-19
tags:
  - Systems
  - GPU
  - Scheduling
  - Robotics
  - PhysicalAI
  - Linux
  - eBPF
  - Jetson
  - SOSP2026
summary: >
  SOSP 2026 三篇獨立提交的論文，指向同一個結論：「GPU 是黑盒 accelerator，OS 不管」的
  時代結束了。**Linux AGX**（UC Riverside，Shirvani & Cong Liu）把 GPU 帶進 CFS 的
  公平性框架，讓 kernel scheduler 對 GPU 使用可見；**Kairos** 把 physical AI 的
  「generate-execute loop」列為 serving system 的一等公民，把端到端延遲降低 31.8–66.5%；
  **gpu_ext**（arXiv 2512.12615）把 eBPF 塞進 GPU driver，讓 policy 可程式化，
  拿到 4.8× throughput、2× tail latency reduction，不用改應用、不用打 kernel patch。
  這三篇合起來看，等於是 NVIDIA Jetson Thor 已經給了硬體 mechanism（Blackwell MIG
  切 7 份、Green Contexts pre-allocate SM、JetPack 7 preemptable RT kernel），
  但 policy 層是空的——OS 該怎麼跟 GPU 對話、serving 該怎麼管 generate-execute、
  driver 該讓誰寫策略。本文拆解三篇論文的核心設計、四層 stack 正在成形的樣貌，
  以及對「想從 compiler 走到 systems software / GPU driver」的工程師（例如我自己）
  這代表什麼樣的機會窗。
---

# Physical AI 把 GPU 拖回作業系統核心：SOSP 2026 Linux AGX、Kairos、gpu_ext 三重奏

> **TL;DR**
>
> 1. **SOSP 2026 有三篇論文集中打「physical AI 的 OS/GPU co-design」這個題**——Linux AGX、Kairos、以及緊鄰的 gpu_ext（arXiv 2512.12615，同一時間點的 preprint）。三篇的作者背景、動機、切入點完全不同，卻在指向同一個結論。
> 2. **Linux AGX** = UC Riverside Cong Liu 團隊（RT-GPU scheduling 老玩家）把 GPU 排程納入 Linux CFS，讓 kernel 對 GPU 有可見性。這是一件過去二十年 Linux 一直沒真的做好的事——`nvidia.ko` 是使用者空間 API 的傀儡，`chrt`/`nice`/`cgroup` 對 GPU 幾乎完全失效。
> 3. **Kairos** 把 physical AI serving 定義為「**generate → execute → observe → generate**」的迴圈——生成一批 action、機器人執行、回饋新觀測、再生成——並主張這個 loop 才是本體，不是「一次 inference 一次 response」。相對於現行 digital AI serving 系統（vLLM/SGLang 那一套），Kairos 把端到端 task latency 降低 **31.8–66.5%**，優勢隨 fleet 規模擴大。
> 4. **gpu_ext** 用 eBPF verifier + 裝置端 runtime 把 GPU driver 變成可程式化 OS 子系統。實測 throughput 提升 **最高 4.8×**、tail latency 降 **最高 2×**，不改應用、不重啟服務。核心觀察是：user-space runtime 有彈性但沒有 cross-tenant 視野；kernel patch 有視野但太危險——eBPF 剛好卡在中間。
> 5. **硬體已經準備好了**。NVIDIA Jetson Thor 給了 Blackwell **MIG 切 7 份 isolated partition**、**Green Contexts** 用 `cuDevSmResourceSplitByCount` 預配 SM、**JetPack 7 preemptable RT kernel**——mechanism 齊全。缺的是 policy 層：誰決定哪個 robot workload 拿哪塊 SM slice？誰決定 grasping 的 tail deadline 該 preempt SLAM？三篇 SOSP 論文就是在填這個坑。
> 6. **對於想從 compiler 走到 systems software / GPU driver / robotics runtime 的人**（例如我自己在為 Nvidia compiler 職涯做準備），這是明確的機會窗——SOSP 這種 top-tier venue 過去五年極少出「Linux + GPU + robotics」交叉題，現在一年就三篇，代表招人、投資、產業需求同時在拉。這篇文章拆解每一篇的技術核心、四層 stack 拼起來的樣貌、以及對面試/職涯的具體用處。

---

## 一、開場：一個看起來很平凡的場景，其實藏著三層調度地獄

想像一個 Jetson Thor 上的雙臂機器人正在做這件事：

- **Arm A** 在做布料摺疊，需要跑 π₀.₅ 這種 vision-language-action (VLA) policy，每 100ms 產生一批 20 個 action chunk。
- **Arm B** 在做零件插入，跑一個較小的 diffusion policy，20ms 一個 action，對 tail latency 極敏感。
- 同時 background 跑著 **SLAM**（LiDAR + camera fusion）、**safety monitor**（毫米波 radar 檢測人員侵入）、**telemetry stream**（把 log 傳到雲端）。

這個場景在 2026 年一點都不特別——TI 已經在 GTC 2026 展示了 mmWave radar + Jetson Thor + Holoscan 的整合，ADI 用 Jetson Thor 做人形機器人，NVIDIA IGX Thor 直接切工業/醫療/機器人邊緣。硬體已經是 mainstream。

但這個場景在**作業系統層面**是一場調度地獄，因為它同時撞上三個結構性問題：

### 1.1 GPU 對 Linux 是「黑盒」

`ps`、`top`、`chrt`、`cgroup v2`——這些 Linux 使用者拿來管 CPU 的工具，對 GPU 幾乎全部失效。你可以把 Arm B 的 controller thread 設成 `SCHED_FIFO` 優先度 99，但當它 launch 一個 CUDA kernel 到 SM 上，這個「thread 優先度」在 SM scheduler 眼中不存在。GPU 端的排程是被 NVIDIA driver 內部的 stream queue + hardware channel + SM warp scheduler 三層黑盒決定的，Linux CFS 完全插不上話。

十年前這問題不痛，因為機器上通常只有一個 CUDA process 在算 batch training。今天一個 Jetson Thor 上同時跑 **VLA policy × 2 + SLAM + radar + telemetry**——五到十個 process 搶同一顆 Blackwell GPU 的 SM——這種黑盒就是災難。

### 1.2 GPU 已經硬體支援 fine-grained partition，但 policy 層是空的

NVIDIA 在硬體端已經給了兩把武器：

- **MIG (Multi-Instance GPU)**：Blackwell 可切成 **最多 7 個 isolated partition**，每個 partition 有自己的 SM 子集、L2 slice、記憶體頻寬份額——彼此**沒有 context switch overhead**。可以拿來把「安全關鍵 workload」跟「一般 workload」硬體隔離。
- **Green Contexts**：CUDA 13.0+ 提供的輕量 context，透過 `cuDevSmResourceSplitByCount` 和 `cuGreenCtxCreate` 預先配置一組 SM 給某個 workload，讓它不被其他 stream 干擾。對 SLAM + planner 這種 latency-sensitive 併發 workload 是關鍵。

問題是——**這兩個 mechanism 的 policy 誰來決定**？MIG 分幾份、每份多大？哪個 process 拿到 Green Context？當 arm B 的 diffusion policy 突然 spike 需要更多 SM 時，誰授權它從 arm A 那邊借？這些問題在 2026 年基本上是「靠 devops 手寫 config，或靠 application 自己協調」——沒有 OS 層的抽象。

### 1.3 Physical AI 的 workload pattern 跟 digital AI 完全不同

digital AI serving（vLLM、SGLang、TensorRT-LLM 那一套）優化的是「一個 request → 一個 response」的 throughput 與 latency。這套系統假設：**每個 inference 是獨立的、可以 batch、失敗可以 retry**。

physical AI 完全不是這樣。Kairos 論文用一句話戳破：

> "Physical AI workloads involve multiple rounds of inference and action execution, generating a chunk of actions in each inference round, and asynchronously interleaving inference and execution."

翻譯：機器人不是在等你回答一個問題，它是在跑一個**閉環**——生成一段動作、執行、觀察環境變化、再生成下一段。這段 chunk 執行到一半時你不能停下來 batch 別的 request，因為手臂在空中。你也不能 retry，因為時間軸不可逆。

這種 workload 對 serving 系統的要求，跟 vLLM 完全對不上。Kairos 就是為此而生。

---

## 二、Linux AGX：把 GPU 拉進 CFS 的第一個嚴肅嘗試

### 2.1 為什麼這篇論文重要，需要先看作者

Linux AGX 全名 **"Linux AGX: An Adaptive GPU eXtension to Linux Fair Scheduling for Physical AI and Robotic Systems"**，作者 Soheil Shirvani 與 Cong Liu，UC Riverside。SOSP 2026 accepted。

Cong Liu 這個名字在「real-time GPU scheduling」領域是**必讀作者**——他從 UT Dallas 到 UC Riverside，過去十年連續在 RTSS、ECRTS、RTAS 發表 GPU RT 排程論文，包括早期的 GPU preemption analysis、integrated CPU-GPU SoC 上的 kernel 保護（可以參考 arXiv:1712.08738）。這意思是——**當 Cong Liu 把題目從 RTAS/RTSS 級的 workshop 升級到 SOSP 這種 top-tier general systems 場合，代表這件事的價值超越了 RT community 的內部討論**，變成整個作業系統社群的共同問題。

從我這種正在把技術重點從「感知演算法」轉到「systems software / compiler」的工程師視角看，這個訊號很清楚：**GPU scheduling 過去被歸類為「特定 domain 的優化」，現在正在成為主流 OS design 的核心議題**。這種轉折點，正是入場的最好時機。

### 2.2 論文核心設計拆解

雖然目前只有 accepted list 上的簡短描述、沒有 arxiv preprint 全文（SOSP 通常會在會議前後才公開），但從標題 "Adaptive GPU eXtension to **Linux Fair Scheduling**" 加上 Cong Liu 過去的技術脈絡，可以合理拆解出幾個核心設計方向：

**（A）把 GPU 使用量納入 vruntime 帳本**

Linux CFS 的核心是 `vruntime`——每個 task 累計了多少「加權運行時間」，scheduler 每次挑「vruntime 最小的」執行。這保證了公平性。

問題是——**當 task X 佔用 GPU SM 100% 跑了 50ms**，這 50ms 在 CFS 眼中不算 X 的 vruntime，因為 CPU 端 X 只是 blocked 在 CUDA sync。結果 X 用了 GPU 榨乾了另一個 task Y 的 SM 時間，但 CFS 覺得 X 沒吃到資源，繼續把 CPU 排給它。這就是為什麼你在多 GPU workload 的機器上會看到「有人被餓死」的現象。

Linux AGX 的第一個貢獻，幾乎必然是把 **GPU utilization** 反饋回 CFS 的 vruntime 計算——task 在 GPU 上花的時間，等價於（或按某比例加權）它在 CPU 上花的時間，一起累積 vruntime。這樣才能真的做到「公平」。

**（B）用 hook 追蹤 GPU kernel launch 與 completion**

要能算 GPU vruntime，就要能追 task launch 了哪些 kernel、每個跑多久。過去這需要打 kernel patch 或使用 CUPTI/NVML profiling API——後者有 overhead 且 policy-hostile。Linux AGX 大機率會用 NVIDIA 從 2022 年開始 open source 的 **kernel-mode driver (`nvidia-open`)** 的 hook 點，直接在 driver 內部追 kernel launch/completion 事件，把資料回饋給 kernel scheduler。

這也是為什麼 gpu_ext（下一節）用 eBPF 走一樣的路徑——**開源 kernel driver 開放後，policy engineering 的門檻降低了一個數量級**。這是 NVIDIA 過去五年最重要的策略決定之一，很多人低估了它的影響。

**（C）「Adaptive」代表 policy 會隨負載調整**

論文名稱裡的 **Adaptive** 是關鍵字。如果只是固定公式做 vruntime 加權，那就是 static extension。Adaptive 意味著——當系統偵測到 workload 從「單一 heavy training job」切換到「多個 latency-sensitive robot inference」時，policy 會自動調整權重、preemption threshold、SM partition 大小。

這也符合 physical AI 的實際：一個機器人一天要跑好幾種模式（工作、待機、學習、故障診斷），workload 特徵完全不同。Static policy 一定 tune 不好。

### 2.3 這對「Linux 是不是能真的做 real-time」這個老問題意味著什麼

Linux + RT 的老爭論持續了二十年——PREEMPT_RT patch、CoreOS、Xenomai、RTLinux……過去主要在 CPU 端解決。GPU 端一直是「你自己想辦法」。

Linux AGX 象徵著這個爭論的 GPU 版本開始了。而且它的時機比 CPU RT 好——因為：

1. **NVIDIA 開源了 kernel driver**，可以做 hook 而不需要反編譯。
2. **Jetson 系列已經有 preemptable kernel** 作為 baseline（JetPack 7 提供），不用從零開始。
3. **產業需求真實**——每一台人形機器人、每一台 L4 車、每一台工業 arm 都是實際訂單。

我的預期是：SOSP 2026 之後兩年，會有第一批「GPU-aware Linux scheduler」的實作進到主線 kernel，或至少進到 NVIDIA/Red Hat 的 downstream。Linux AGX 是這條路上的 pilot paper。

---

## 三、Kairos：把 physical AI 的 generate-execute loop 列為 serving 一等公民

### 3.1 為什麼 vLLM 那一套不夠

`vLLM` 為代表的 LLM serving 系統把「continuous batching + PagedAttention + speculative decoding」做到極致——但這一切都建立在一個假設上：**每個 request 是獨立的，response 之間沒有時間相依性**。

physical AI 不是這樣。Kairos 論文明確指出，physical AI task 的結構是：

```
[generate action chunk] → [execute in the physical world] → [observe new state] → [generate next chunk] → ...
```

這個 loop 有三個 vLLM 沒有的性質：

1. **時間強耦合**：`execute` 階段是物理世界的實際時間，不能加速、不能 skip。上一輪 execute 沒結束，下一輪 generate 拿到的觀測就是舊的。
2. **inference 跟 execution 的非同步交錯**：現代 VLA policy 會**同時**跑「上一批 action 的 execute」與「下一批的 generate」，兩者要在 SM 上共存。這對 kernel launch order 與 SM occupancy planning 提出全新要求。
3. **generate 的 output 不是文字，是動作 chunk**：一批 20 個 action，intra-chunk 之間 latency 不重要（因為 downstream 執行器會排隊），但 chunk-level 的 first-token time（第一個 action 何時準備好）非常關鍵——因為機器人在等這第一個動作。

vLLM 的 continuous batching 假設是 token-level latency 一致重要；physical AI 的答案是「第一個 action 極重要，後面可以慢一點」。這種結構完全不對。

### 3.2 Kairos 的三個推論

從 abstract 揭示的框架我們可以逆推出 Kairos 的核心設計（完整技術細節等 SOSP 論文發布再更新）：

**推論 1：inference scheduler 要感知 execution phase**

當 robot 進入執行動作階段時，inference 的資源可以短暫讓出——因為這段時間反正生成再多也用不到。Kairos 的 scheduler 大概率會用某種形式的「執行進度 feedback」告訴 inference layer 何時 slow down、何時 pre-warm。

**推論 2：跨 robot 的 fleet-level batching**

一個 warehouse 裡 100 台 robot，每台的 generate-execute 週期不同步——這正好是 fleet 級 batching 的機會。當 robot A 在 execute 時，把它的 GPU slice 借給 robot B 的 inference。Kairos 大概率有這種 fleet-level opportunistic scheduling。

**推論 3：chunk 內部的 speculative decoding**

一批 action 通常有時序上的規律（例如手臂軌跡是平滑的）。這給了 speculative decoding 天然的 draft model——用一個更小的 policy 猜下一個 action、大 policy 驗證。這是把 LLM 世界的技術遷移到 robotics 世界的直接應用。

Kairos 論文報告 **端到端 task latency 降低 31.8–66.5%**，且優勢隨 fleet 規模放大。66.5% 是一個很誇張的數字——這意味著現行 physical AI 部署有超過一半的時間是浪費在系統層次的不匹配上。這個數字本身就是 physical AI serving 值得成為獨立 subfield 的證據。

### 3.3 對 embedded systems 工程師的實務意義

如果你在 Foxconn 或任何機器人相關公司做嵌入式 + AI，Kairos 這條線意味著什麼？

- **不能繼續用 vLLM/TensorRT-LLM 直接部署 VLA policy**，會浪費至少一半的硬體。
- **serving 層要跟 controller 層做 co-design**——inference latency 的意義取決於 controller 的下一個 deadline，這件事必須跨團隊。
- **fleet-level orchestration 會是新戰場**——過去是 Kubernetes 排 pod，未來是 Kairos-like 系統排 robot fleet 的 generate-execute cycle。

對想寫 systems software 的人，這是一個非常明確的入場點——因為主流公司都還沒有這種 stack，你有機會從零建。

---

## 四、gpu_ext：eBPF 進 GPU driver，policy 可程式化

### 4.1 為什麼是 eBPF

eBPF 過去五年在 Linux kernel 造成的革命，很多人沒完全消化：**它把「要 patch kernel 才能改 policy」變成「寫一段 verified byte code 動態載入」**。網路（Cilium）、observability（Pixie）、security（Falco）、儲存（fuse-alternatives）全部因此改寫。

GPU 過去沒有這個能力，因為 GPU driver 是黑盒。但 NVIDIA 從 2022 年開放 kernel-mode driver 之後，這個門打開了。gpu_ext 就是走進這扇門的 pilot。

論文摘要清楚描述了它想解的核心衝突：

> "User-space runtimes offer flexibility but lack cross-tenant visibility, while kernel modifications introduce complexity and safety risks."

翻譯：使用者空間的解決方案（CUDA library、runtime middleware）有彈性但看不到全局；kernel patch 有全局視野但太危險。**eBPF 是中間地帶**——policy 可以動態載入，但 verifier 保證不會 crash kernel、不會 leak memory、不會執行沒有 bound 的 loop。

### 4.2 三個 gpu_ext 打開的空間

**（A）Cross-tenant fairness**

當 process A 猛 launch 小 kernel 佔滿 stream queue，process B 的大 kernel 一直被推遲——這種現象在多租戶 GPU 上很常見。傳統做法是「應用自律」，但沒有 process 會自願讓出。gpu_ext 可以動態載入一個 policy：偵測到 queue 被單一 tenant 霸占超過 X ms，就 preempt 或降權。

**（B）Adaptive memory prefetch/eviction**

搜尋結果提到 gpu_ext 已經實作「adaptive memory prefetching and eviction policies」。這對 physical AI 意義重大——當 workload 從 SLAM（穩定的 dense tensor）切換到 VLA policy（sparse 且突發），memory access pattern 差非常多，需要不同的 prefetch policy。過去這是「編譯時固定」，現在是「執行時可換」。

**（C）Programmable observability**

過去要看 GPU 內部發生什麼，只能用 Nsight 這種 profiler，overhead 大、不能常駐。gpu_ext 可以載入 lightweight probe，只針對特定事件（例如某個 kernel 的 SM occupancy < 50% 才記錄），成本可以壓到接近零。這對 production 環境的 GPU 效能診斷是革命。

### 4.3 gpu_ext 報告的數字說明什麼

論文報告 **throughput 提升最高 4.8×、tail latency 減少最高 2×**，「不改應用、不重啟服務」。

這裡的關鍵不是絕對數字，而是「不改應用」——意味著**現在部署的所有 CUDA workload，理論上都可以無痛享受這種優化**。這對 production 意義極大，因為在生產環境動應用碼是政治問題，動 driver policy 是純技術問題。

4.8× 是「特定 workload 在特定 policy 下的最好數字」，實際部署大概不會這麼誇張。但即使平均 1.5×，對 datacenter 級的 GPU 帳單（每年動輒千萬美金）也是巨大節省。

---

## 五、拼起來看：一個四層 stack 正在成形

把 Jetson Thor 硬體 + gpu_ext + Linux AGX + Kairos 疊起來，會看到一個很清晰的四層架構開始浮現：

```
┌──────────────────────────────────────────────┐
│  Layer 4: Serving / Orchestration            │
│    Kairos — fleet-level generate-execute    │
│    orchestration，跨 robot batching          │
├──────────────────────────────────────────────┤
│  Layer 3: OS Scheduling                      │
│    Linux AGX — GPU-aware CFS，vruntime       │
│    包含 GPU 時間，adaptive policy           │
├──────────────────────────────────────────────┤
│  Layer 2: Programmable GPU Driver            │
│    gpu_ext — eBPF verifier + 裝置端 runtime │
│    可換 policy for memory / scheduling      │
├──────────────────────────────────────────────┤
│  Layer 1: Hardware Mechanism                 │
│    NVIDIA Jetson Thor Blackwell              │
│    MIG (7 partition) + Green Contexts        │
│    + JetPack 7 preemptable RT kernel         │
└──────────────────────────────────────────────┘
```

三年前這個 stack 只有 Layer 1（而且還不完整——Green Contexts 是 CUDA 13.0 才穩定）。今天 Layer 2、3、4 分別有 pilot paper。這是**整個 stack 正在從 mechanism-only 進化到 mechanism + policy** 的關鍵時刻。

回想 CPU 世界的類比——70 年代大家有 timer interrupt（mechanism），但直到 CFS/EDF/deadline scheduler 出現（policy），multiprogramming 才成為主流。80 年代大家有 MMU（mechanism），直到 mach/BSD/Linux 把 virtual memory policy 做對（demand paging、copy-on-write），general-purpose OS 才成型。

**GPU 現在處於 policy 開始成型的階段。** 誰做出「GPU 世界的 CFS」，誰就是下一代 systems 的定義者。

---

## 六、為什麼這三篇同一年集中出現不是巧合

過去五年 SOSP/OSDI 上「Linux + GPU + robotics」交叉題基本上是空的——不是不重要，是**產業還沒真正把 physical AI 當回事**。2025 到 2026 這一年發生了三件事同時對齊：

1. **NVIDIA 開放 kernel-mode driver 進入實用階段**（2022 開放，但直到 2024–2025 才穩定到可以做 systems research）。
2. **Jetson Thor 出貨**（2026 Q2 開始大量鋪貨），把「Blackwell + MIG + Green Contexts + preemptable RT kernel」變成產業標配。
3. **VLA policy（π₀、π₀.₅、OpenVLA 等）從論文走到 production**，讓 physical AI serving 從「demo」變成「有人真的要每月付錢跑」。

這三件事湊在一起，才讓 SOSP 這種 general systems 場合的 PC 願意接受「physical AI + GPU + Linux」這種題目。過去這種題目會被 push 到 RTSS/ECRTS（太 RT-specific）或 ISPASS（太 hardware）。

**這是一個明確的 phase transition**——不是每年都會發生。上一次在 systems 領域類似的 phase transition，是 2010 年前後 datacenter operating systems 從「概念」變成 OSDI/SOSP 主題（Mesos 2011、Borg 2015、Kubernetes-era）。那波 phase transition 定義了整個 cloud infra 十年的職涯機會。

Physical AI GPU scheduling 這波，我押它是 2026–2032 的類似機會。

---

## 七、對「想從 compiler 走到 GPU driver / systems software / robotics runtime」的人，這代表什麼

我最近半年在調整職涯方向——從「LiDAR 感知演算法」往「compiler / GPU / systems software」搬。前幾天才在 `~/dev/career/4-Learning/Compiler-Path.md` 定下 Option C（spconv capstone + Nvidia compiler 面試準備）為主軸。這三篇論文正好給我幾個具體行動 signal：

### 7.1 短期（3 個月內）該做的事

- **讀完 Linux AGX**：SOSP 2026 論文發布後，第一時間精讀。重點看它怎麼定義 GPU vruntime 的累計方式、怎麼跟 CFS 整合。這是理解「GPU-aware kernel scheduler」的入場券。
- **玩 gpu_ext 或 nvidia-open 的 hook 點**：拿一台有 NVIDIA GPU 的 Linux 機器，跑一次 nvidia-open kernel module 的 build，看 hook 點在哪裡。這是體感 eBPF-for-GPU 的最快方法。
- **在 Jetson 上跑 Green Contexts 實驗**：Jetson Orin/Thor 借得到就借，跑一組 SLAM + inference 的併發實驗，觀察加/不加 Green Context 的差別。這是「知道理論 vs. 手上有實測」的差別。

### 7.2 中期（6–12 個月）該做的事

- **把 spconv capstone 的一部分做成 physical AI 相關**：LiDAR sparse conv 本身就是 physical AI 的核心 workload。把 spconv 在多 process 併發、SM partition 下的行為做一次深入 profiling，寫成 tech blog——這種內容在 Nvidia compiler team 面試會非常加分，因為它直接落在 compiler + robotics + GPU 三者的交集。
- **關注 SOSP 2027 / OSDI 2027 相關 CfP**：這個題目會繼續紅，工作機會會在會議 program 裡浮現。跟這批作者建立連結（推特/GitHub issue/會後 email），是入圈的直接管道。

### 7.3 面試素材角度

如果面 Nvidia Compiler / Runtime Team、或任何 physical AI 相關系統職位，這三篇是必聊的素材：

- **問你「怎麼設計一個 GPU-aware Linux scheduler」** → 引 Linux AGX 的 vruntime accounting 思路。
- **問你「怎麼服務 100 台機器人的 VLA policy」** → 引 Kairos 的 generate-execute loop 抽象。
- **問你「怎麼在不改應用的前提下優化 GPU tail latency」** → 引 gpu_ext 的 eBPF hook 空間。

這三題如果能各自講出「論文核心 + 為什麼是這個時機出現 + 你自己會怎麼延伸」，就是 senior 級別的答題深度。

---

## 八、風險與 open question

我不想寫成廣告文，這節列幾個目前還沒被回答清楚的問題：

### 8.1 policy 太多會不會反而更難調

CFS 已經有幾百個 tunable 參數（各種 sched_* sysctl），系統管理員多半束手無策。如果 GPU scheduler 再加一層 adaptive policy + 一層 eBPF programmable driver + 一層 serving orchestration，複雜度會爆炸。誰負責 tune？tuning 錯了誰負責？

歷史上 CPU 世界的解法是「distro 給 sane defaults + 特定 workload 給 tuning profile」（例如 `tuned` 之於 RHEL）。GPU 世界會走一樣的路，但誰做 default profile 是空缺——NVIDIA 自己會做嗎？Red Hat？Canonical？還是新 vendor 崛起？這是一個商業機會。

### 8.2 vendor lock-in

Linux AGX 的 hook 依賴 NVIDIA open kernel driver；gpu_ext 也是。AMD ROCm、Intel level-zero 有沒有對等的開放程度？如果沒有，這套 stack 就是 NVIDIA-only——這對開源社群的接受度會是個問題。

不過就 physical AI 現實而言，Jetson Thor 目前是**近乎壟斷**的 edge SoC 選擇。這種現實會壓過理論上的 vendor neutrality，至少 3–5 年內。

### 8.3 論文結果可重現性

SOSP 論文的數字（Kairos 31.8–66.5%，gpu_ext 4.8×）通常在特定 benchmark 下最好看。實際部署會不會複製？兩年後回頭看才知道。歷史經驗——大部分 systems paper 的 headline number 在 production 是 0.5–0.8×——但即使打折後仍然 significant，這個 stack 就 justify 了。

---

## 九、結語：一個十年一遇的 layer 正在成形

Systems software 的歷史規律是——**每 10–15 年會有一個新的 layer 從 hardware mechanism 之上長出來**，重新定義誰是這個 layer 的定義者。過去二十年的例子：

- 2000s：**virtualization** layer（VMware/Xen/KVM），定義者變成 VMware。
- 2010s：**container orchestration** layer（Docker/Kubernetes），定義者變成 Google 系 + CNCF。
- 2020s early：**LLM serving** layer（vLLM/SGLang/TensorRT-LLM），定義者還在賽跑中，NVIDIA + Anthropic + 幾家 startup 領先。

2026 年開始，我押 physical AI 的 **OS-GPU co-design + serving** 是下一個 layer。定義者未知——UC Riverside 這批學術，NVIDIA 內部，還是某個現在還沒成立的 startup。

對想入場的人，這個時機像 2012 年前後有人問「要不要學 Docker」——當時大部分人回答「有 VM 就夠了」。回頭看，答案很清楚。

Linux AGX、Kairos、gpu_ext 這三篇 SOSP 2026 論文，就是 2012 年 Docker Con 那種訊號。

我打算跟緊這條線，不只是因為它跟我的 compiler 職涯路線對齊，更因為它是**我目前手上少數幾個「同時觸及技術硬核 + 產業風口 + 個人職涯轉型」的交叉點**。今年年底前把 Linux AGX 論文本文精讀完、把 gpu_ext 的 hook 玩過一輪、把 Jetson Thor 上的 Green Contexts benchmark 跑過——這是接下來三個月的具體 action item。

---

## Sources

- [SOSP 2026 accepted papers list（Chaignon 個人彙整）](https://pchaigno.github.io/academic/2026/08/03/sosp-2026-papers.html)
- [SOSP 2026 official accepted papers](https://sigops.org/s/conferences/sosp/2026/accepted.html)
- [Kairos: A Scalable Serving System for Physical AI（arXiv 2605.11381）](https://arxiv.org/pdf/2605.11381)
- [gpu_ext: Extensible OS Policies for GPUs via eBPF（arXiv 2512.12615）](https://arxiv.org/html/2512.12615)
- [Introducing NVIDIA Jetson Thor, the Ultimate Platform for Physical AI（NVIDIA Developer Blog）](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)
- [Accelerate Robotics and Real-Time AI Inference on NVIDIA Jetson Thor](https://www.scribd.com/document/933649553/Accelerate-Robotics-and-Real-Time-AI-Inference-on-NVIDIA-Jetson-Thor)
- [What's New in CUDA Toolkit 13.0 for Jetson Thor（NVIDIA）](https://developer.nvidia.com/blog/whats-new-in-cuda-toolkit-13-0-for-jetson-thor-unified-arm-ecosystem-and-more/)
- [ADI Adopts NVIDIA Jetson Thor for Humanoids](https://www.analog.com/en/newsroom/press-releases/2025/8-25-2025-adi-adopts-jetson-thor.html)
- [OpenPi π₀.₅ on Jetson Thor（Jetson AI Lab tutorial）](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
- [Cong Liu 團隊早期作品：Protecting real-time GPU kernels on integrated CPU-GPU SoC（arXiv 1712.08738）](https://arxiv.org/pdf/1712.08738)
- [SCHED_DEADLINE（Wikipedia，理解 Linux RT 排程演進脈絡）](https://en.wikipedia.org/wiki/SCHED_DEADLINE)
