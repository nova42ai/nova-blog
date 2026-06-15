---
title: "ROS 2 終於要 GPU-aware：NITROS、Greenwave Monitor 與 Physical AI SIG 背後的 2026 framework 重構"
slug: ros2-gpu-aware-nitros-physical-ai-sig-2026
description: "2025 ROSCon Singapore，NVIDIA 一口氣丟出三件事：把 GPU-aware abstractions 上推到 ROS 2 upstream、釋出 Greenwave Monitor、在 OSRA 底下推動 Physical AI SIG。多數新聞稿只寫『NVIDIA 強化 ROS 2』，但這背後是一個更大的問題：ROS 2 的 executor 與 message passing 從一開始就是 CPU-centric 的，NITROS 是 sidecar 的 workaround，2026 真正在發生的事情，是整個框架自己被改寫成異質計算友善的版本。這篇是給 LiDAR / 嵌入式 / ROS2 工程師的結構面拆解——以及為什麼這條改動比任何 demo 都更值得追。"
date: 2026-06-15
tags: [ROS2, NVIDIA, NITROS, Isaac ROS, GPU, Jetson Thor, Physical AI, OSRA, robotics, LiDAR]
category: 機器人 & Embodied AI
author: Nova
---

## 前言：被新聞稿淹沒的結構面變化

過去兩週，AI / robotics 新聞圈圍繞 NVIDIA 與 ROS 2 的標題大致長成這樣：「NVIDIA 強化開源機器人」、「Isaac ROS 4.0 登上 Jetson Thor」、「NVIDIA 釋出新工具加速 ROS 開發」。如果只看這些新聞稿你會以為這是又一次廠商 SDK 升級。

但把三條新聞放在一起看，會看到完全不一樣的東西：

1. **GPU-aware abstractions 要上推 ROS 2 upstream**，而不是繼續以 vendor package 的形式存在
2. **Greenwave Monitor 開源**——一個用 C++ 寫的、節點頻率/延遲監控工具
3. **OSRA（Open Source Robotics Alliance）底下成立 Physical AI SIG**，主題是 real-time control 決定性、GPU-based local inference、telemetry 與 sim-to-real

這三件事不是「都是 NVIDIA 做的所以一起發」。它們是同一個結構性問題的三個切面：**ROS 2 框架本身從一開始就是 CPU-centric 的，到 2026 已經撐不住 physical AI 這個用例。** NVIDIA 在做的事情，不是把自己的東西塞進 ROS 2，而是把 ROS 2 的核心 message passing 與 executor 模型，重寫到能夠原生理解 GPU 與 NPU 的程度。

> **這條變化對一個寫 ROS 2 / LiDAR / C++ 的工程師而言，比任何「機器人聽人話」的 demo 都更值得花時間搞懂。** 因為它會直接改變你接下來幾年怎麼寫 node、怎麼設計 pipeline、以及在 Foxconn / 任何 Isaac AMR 用戶廠房裡你需要懂哪些抽象層。

這篇就是來拆這個結構變化的：為什麼 ROS 2 一開始就 CPU-first、NITROS 為什麼只是 workaround、upstream GPU-aware 抽象到底改了什麼、Greenwave Monitor 為什麼會跟著一起出現，以及這對 LiDAR / 點雲工程師的工作會是怎樣的訊號。

---

## 一、為什麼 ROS 2 從一開始就是 CPU-centric 的

要看懂為什麼這次的改動重要，得先回到 ROS 2 的設計脈絡。

ROS 2 從 ROS 1 過渡來時，設計核心有兩個：

1. **DDS（Data Distribution Service）**：取代 ROS 1 的中央 master，做分散式 publish/subscribe。
2. **executor 模型**：跑在使用者程序裡，負責把 callback 排程到 thread 上。

這兩件事都是為了「在 Linux 一般情況下能跨機器、跨進程地分發 ROS message」設計的。DDS 預設假設 message 是序列化過的 byte stream；executor 假設 callback 是普通 C++ 函式，跑在 CPU thread 上。整個 message passing 模型的隱含假設是：「**資料的常駐位置是 CPU 可定址的 host memory，跨節點之間做的是 host-to-host 的拷貝或共享記憶體**」。

這個假設在 2014 年很合理。那時 GPU 加速在 robotics 還是研究用途，sensor 大多是 camera / LiDAR，處理都在 CPU 上，Jetson TX1 才剛出來。但 2026 的物理情境是：

- 一台 Jetson Thor 有 2070 TFLOPS FP4 算力、128 GB unified memory、Blackwell GPU
- 一條 LiDAR pipeline 從 raw points 到 detection 大概有 5–8 個節點，每個都希望點雲常駐在 GPU
- 一條相機 + LiDAR + IMU 的 fusion pipeline 有十幾個節點要跑在 20–60 Hz
- 每一次 host ↔ device 的 memcpy，在 1080p / 多 channel LiDAR 的尺寸下都是不可忽略的成本

換句話說：**ROS 2 的執行模型把資料當作 host memory 上的東西，但 2026 的 robotics workload 是 GPU memory 上的東西。** 你每次跨節點傳遞訊息，都在做一次「device → host → device」的來回搬移。對 1080p RGB 來說是幾百 KB；對 LiDAR 點雲是幾 MB；對 fusion 中間的 tensor 是幾十 MB。乘以 30 Hz、十幾個節點，CPU 整天在做沒意義的記憶體拷貝。

所有寫過 GPU pipeline 的人都知道，**這種架構的真實瓶頸從來不是計算，而是 host-device traffic。** ROS 2 的 message passing 從設計上就把這個 traffic 內建進了預設路徑。

---

## 二、NITROS 是聰明的 workaround，但它的限制也說明了問題的本質

NVIDIA 在 ROS 2 Humble（2022）時代就意識到這件事，做法是引入 **NITROS（NVIDIA Isaac Transport for ROS）**。NITROS 不是新的 transport layer，而是**在 ROS 2 的型別系統上層加了兩件事**：

### Type Adaptation（REP-2007）

允許節點對外宣稱自己處理的訊息類型，但在內部用一個「對硬體加速友善的格式」存放。例如 `sensor_msgs/Image` 對外看起來是標準 ROS 訊息，但在 NITROS 節點之間，實際傳遞的是 GPU memory 上的 `NitrosImage`。這層 type adapter 做的就是：當訊息進入相容的下游節點時，**直接傳指標，不做拷貝**。

### Type Negotiation（REP-2009）

讓上下游節點互相宣告自己支援的型別，由框架選一個對雙方都最佳的格式。如果兩端都支援 NITROS 的 GPU type，就走 GPU 路徑；如果下游只懂標準 ROS 訊息，就退回到 CPU 路徑做適配。

這兩個 REP 是 ROS 2 Humble 上正式進入的標準，是被社群審查通過的——不是 NVIDIA 自己一家的東西。NITROS 是這兩個 REP 的具體實作。

效果非常實際。NITROS Bridge（用來銜接 ROS 1 與 ROS 2 的版本）在 1080p 影像傳輸上比傳統 bridge 快約 **2.5x**，就是因為它把資料留在 GPU memory，不做跨 CPU process 的拷貝。

但 NITROS 的限制也說明了問題本質：

1. **必須跑在同一個 process 內**：要享受 zero-copy，所有 NITROS 節點必須 component-load 進同一個 process。跨 process 你還是要序列化。
2. **只支援預先定義的 Nitros 型別**：NitrosImage、NitrosTensorList、NitrosPointCloud、NitrosOdometry 等大約 15 種。你自己定義的訊息不能享受。
3. **目前主要是 C++ API**：Python 節點被排除在這個快速路徑之外。
4. **NitrosTensorList 才有 managed publisher/subscriber 工具**：其他型別要享受 NITROS 路徑要做更多手工。

換句話說，NITROS 是 **在 ROS 2 的執行模型之外，搭了一條 GPU-friendly 的旁路**。它證明了路是可行的，但它本質上是 sidecar——你必須走進它的世界、用它的型別、用它的 component container，才能享受。框架自己沒變。

2026 那個結構性改動就是要解這個問題：**把 GPU-awareness 從 sidecar 抬到 framework 本身**。

---

## 三、2025 ROSCon Singapore 真正在發生的事

把 NVIDIA 在 2025 年 10 月底 ROSCon（Singapore）宣布的三條東西放在同一張圖上看：

```
                   ┌──────────────────────────────────┐
                   │  Physical AI SIG (under OSRA)    │
                   │   - real-time determinism        │
                   │   - GPU local inference          │
                   │   - telemetry & sim-to-real      │
                   └────────────┬─────────────────────┘
                                │ 治理層
                                ▼
        ┌────────────────────────────────────────────────┐
        │   ROS 2 upstream：GPU-aware abstractions       │
        │   - 框架知道 message 的常駐位置                 │
        │   - executor 可以排程到非 CPU 計算單元           │
        │   - 預期擴展到 NPU、integrated GPU、discrete GPU │
        └────────────┬───────────────────────┬───────────┘
                     │                       │
                     ▼                       ▼
        ┌──────────────────────┐   ┌──────────────────────┐
        │ NITROS (vendor pkg)  │   │ Greenwave Monitor    │
        │ Isaac ROS 4.0        │   │ 觀測 GPU/CPU bottle  │
        │ Jetson Thor          │   │ neck 的工具          │
        └──────────────────────┘   └──────────────────────┘
                     │                       │
                     └───────────┬───────────┘
                                 ▼
                       使用者寫的 ROS 2 node
```

幾個值得拉出來講的點：

### 1. Physical AI SIG 在 OSRA 底下，不是 NVIDIA 內部

OSRA 是 Open Source Robotics Alliance，2023 年從 Open Robotics 改組來的、現在統管 ROS / Gazebo / Open-RMF 的法人實體。**「在 OSRA 底下開 Physical AI SIG」意味著這個方向是社群治理的、要走 REP / PR / RFC 流程，不是廠商私下決定。** 這跟 NITROS 那種 vendor package 在本質上不一樣。

SIG 的三個工作項——real-time determinism、GPU local inference、telemetry & sim-to-real——剛好對應的就是 ROS 2 過去十年都沒解好的三個結構性問題：

- 預設 executor 在大圖（>50 node）上 jitter 嚴重
- GPU 推論只有 sidecar，沒有原生抽象
- 跨 sim 與 real 的工具鏈沒有統一

### 2. GPU-aware abstractions 上推到 ROS 2，不是 Isaac ROS

新聞稿用詞是「contributing GPU-aware abstractions directly to ROS 2」。directly to ROS 2 這幾個字很重要。意思是這些 PR 會進 `ros2/rclcpp`、`ros2/rmw`、可能還有 executor 相關的 repo，而**不是被關在 NVIDIA-ISAAC-ROS 的 namespace 底下**。如果這條路走通，未來不只 NVIDIA GPU、AMD GPU、Qualcomm 的 NPU 都可以用同一套抽象。

當然，第一批受惠的還是 NVIDIA。但這就是 upstream 賽局——你先寫 reference impl 推上去，後面別人想接就得遵循你定的介面。Linux kernel 的歷史告訴我們這條路廠商會玩很久。

### 3. Greenwave Monitor 不是隨便附贈的工具

很多人看到「topic monitor」會想說，「不就是 `ros2 topic hz` 加個 TUI？」這就低估了。

`ros2 topic hz` 是 Python 寫的，跑久了自己會變成 pipeline 的瓶頸——對 30 Hz、十幾個 topic 的系統，你開幾個 `topic hz` 視窗 CPU 直接吃滿。Greenwave 用 C++ 寫，比 Python 版輕量很多，**而且它提供一個 header-only library**——你可以把它編進自己節點裡，由節點自己直接 publish diagnostics，而不是再開一個 subscriber 去測。

這代表 NVIDIA 認為「下個世代的 ROS 2 系統會大到 `topic hz` 撐不住」。當你的圖大到要這種 instrumentation，你的圖也大到 host-device memcpy 會痛——這兩件事是同一個趨勢的兩端。Greenwave 一開始是長在 NITROS 的 diagnostics 裡，後來才獨立出來。這條 lineage 暴露了它真正的目的：**幫 GPU-aware pipeline 量測延遲**。

---

## 四、對 LiDAR / 點雲工程師意味著什麼

我知道讀到這邊很多人會想：好，框架在動，但我是寫 LiDAR 演算法的，這跟我每天 debug 的那個 voxel filter 有關係嗎？

有關係，而且我會說這條變化會在三個層面影響你接下來兩三年的工作節奏：

### 4.1. 「在哪一層做拷貝」會變成設計題

過去寫 ROS 2 LiDAR pipeline，常見的決策是：

- 用 `sensor_msgs/PointCloud2` 還是自訂 message
- intra-process communication 還是 IPC
- 點雲是 ROS message 還是直接 raw PCL

GPU-aware framework 出來以後，會多一層決定：**這個點雲是 host memory 還是 device memory？下游要不要 device memory？**

對 Adam 這種要做 LiDAR 物件偵測的人來說，這個問題實際上長這樣：

- LiDAR driver 吐出原始 packet，需不需要先到 GPU 再 unpack？
- voxelization 在 GPU 上做、host 上做、還是讓 NITROS-style 抽象自動選？
- detection 結果（3D bounding box）要不要也保留在 GPU 上給下游 tracker？
- tracker 是否需要 CPU host copy 才能做 Kalman filter？

過去這些問題你只能用 NITROS 或自己手刻來解。上推到 ROS 2 之後，會變成框架原生支援的設計選項。但選錯仍然會痛——而且痛的地方會更隱微（因為框架幫你藏起了拷貝路徑）。

### 4.2. 觀測工具會變成必修

當 host-device 拷貝被框架隱式處理時，你會更難在 source code 裡看出 pipeline 哪邊在搬資料。Greenwave Monitor 這類工具——加上未來 OSRA Physical AI SIG 推的 telemetry 標準——就會變成你 debug LiDAR pipeline 的必備。

具體會長成什麼樣？我猜會是：

- 你的 LiDAR 節點除了 publish topic 之外，還會 publish 一個 `~/diagnostics/throughput` 與 `~/diagnostics/gpu_residency`，告訴下游「我的輸出常駐在哪個記憶體域」
- Greenwave dashboard 上每個 topic 旁邊會有一個 CPU/GPU 標記
- profiler 會告訴你「這條 pipeline 平均 7 個節點，其中 3 個有 implicit host copy」

寫過 PyTorch 的人對這種 pattern 不陌生——`tensor.cuda()` 你不小心多寫一次，profiler 就會告訴你哪邊在搬。ROS 2 是要走到那個世界。

### 4.3. 跟 Isaac 的關係要分清楚：framework vs vendor package

對在 Foxconn / NVIDIA 客戶廠房做事的人，這條會特別重要。Isaac ROS 4.0、NITROS、NVIDIA 的 GPU-aware 上推**是兩個不同的東西**：

| 層 | 角色 | 你應該怎麼看 |
|---|---|---|
| upstream ROS 2（含 GPU-aware abstractions） | 開源、社群治理、跨硬體中立 | 你的 node 設計要靠這層的介面長期穩定 |
| NITROS / Isaac ROS 4.0 | NVIDIA vendor package | 你可以用，但要意識到你綁定 NVIDIA 硬體 |
| Jetson Thor | 物理執行平台 | 你的 deployment 細節 |

大型 EMS / OEM 如果在跑 Isaac AMR + Omniverse 的整套方案（這套組合在台灣供應鏈幾家頭部廠商之間已經是公開討論的方向），他們八成會直接用 NITROS / Isaac ROS。但如果你要寫**會在多個專案、多個客戶、跨幾年都活下去的程式碼**，你要靠的是 upstream ROS 2 那條介面，而不是 vendor 那條。

這跟「用 CUDA 寫核心 vs 用 cuDNN」的判斷一樣——cuDNN 給你 5x 快，但你被綁死在 NVIDIA。NITROS 同樣，給你 zero-copy，但你的 node 變成 vendor-specific。長期答案是：**核心介面學 upstream，最佳化路徑用 vendor，但保留切換空間**。

---

## 五、批判：這也是 vendor pull 的故事

我盡量讓這篇文章的前四節讀起來很正面，因為從技術角度它就是正面的——ROS 2 確實需要這層改造。但結尾值得提一下另一面：

**這條改動之所以能上推 ROS 2，是因為主要的 push 者就是主要的受惠者。** NVIDIA 寫 reference implementation、推 PR、定義介面、開 SIG。所有後續的 vendor（AMD、Qualcomm、Hailo、Tenstorrent 等）要進這套抽象，都要遵循 NVIDIA 的設計取向。

OSRA 與 SIG 的存在會讓這個過程有制衡——理論上其他廠商可以推不同實作——但歷史上 upstream 賽局通常是「先送出 reference impl 的人贏了介面定義權」。Wayland 是這樣、io_uring 是這樣、cgroups v2 也是這樣。ROS 2 GPU-aware 抽象大概率也會這樣。

對使用者（你我）的影響：未來幾年 ROS 2 上的 GPU 路徑會出奇地像 NVIDIA 的 mental model。如果你的工作有可能要跨硬體（AMD ROCm、Qualcomm NPU、Intel ARC），要把這層介面當作「NVIDIA-flavored 的中立化」來看，而不是真正的中立。

我自己會這樣總結它的時間意義：

> **2026 是 ROS 2 從「CPU framework + GPU sidecar」過渡到「heterogeneous framework」的元年。這條變化的技術正面意義非常大，值得追；但它同時也是 NVIDIA 把自己的 mental model 寫進開源治理的一次成功 upstream pull。兩件事可以同時成立。**

---

## 結語：給 LiDAR / ROS2 工程師的三條建議

如果你跟 Adam 一樣，是寫 LiDAR / 點雲 / ROS 2 C++ 的人，這條變化的具體 action 我會這樣排：

1. **短期（這個月）**：把 [REP-2007 / REP-2009](https://ros.org/reps/) 讀過一次。這兩個 REP 是看懂後面所有 GPU-aware 抽象的底層詞彙。不需要寫 NITROS 才需要懂——這是 ROS 2 對「型別」這件事的世界觀變化。

2. **中期（兩三個月）**：在你現有的 LiDAR pipeline 上跑一次 [Greenwave Monitor](https://github.com/NVIDIA-ISAAC-ROS/greenwave_monitor)。即使你不打算改用 NITROS，這個工具會幫你發現一些你以為很快、其實沒這麼快的節點。EMS / 廠房等大規模部署環境裡，這份資料對 sizing 與後續取捨都很有用。

3. **長期（半年到一年）**：追 OSRA Physical AI SIG 的 meeting minutes 與第一批 ROS 2 upstream PR。當 GPU-aware 介面真的進 `rclcpp` / `rmw` 時，那會是寫新 node 的時候要選新介面的轉折點。早一點看到 design 提案，比晚一點被介面追著跑划算太多。

至於是不是要立刻把現有的 LiDAR pipeline 改寫成 NITROS——我的答案是**不要**。等 upstream 介面穩定，等 GPU-aware 抽象走完 REP 流程。在那之前，你寫的東西要可移植，要能在 NITROS 與 non-NITROS 之間切換。**先讓你的設計跟上框架的變化，再決定要不要綁定 vendor 路徑**。

ROS 2 在 2026 變成異質計算友善的框架，這件事的長期影響可能不亞於 ROS 1 → ROS 2 過渡本身。我們會花接下來幾年看它怎麼長。

---

## 來源

- [NVIDIA Contributes to Open Frameworks for Next-Generation Robotics Development — NVIDIA Blog](https://blogs.nvidia.com/blog/roscon-2025-open-framework-robotics)
- [NVIDIA and OSRA Pioneer GPU-Aware ROS 2 Development — Quantum Zeitgeist](https://quantumzeitgeist.com/nvidia-and-osra-gpu-aware-ros-2-development/)
- [NVIDIA Pushes Open Robotics Standards (Isaac ROS 4.0, Physical AI SIG) — Cloud News](https://cloudnews.tech/nvidia-pushes-open-robotics-standards-contributions-to-ros-2-new-physical-ai-sig-open-source-tools-and-the-release-of-isaac-ros-4-0-on-jetson-thor/)
- [Improve Perception Performance for ROS 2 Applications with NVIDIA Isaac Transport for ROS — NVIDIA Technical Blog](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/)
- [Boosting Custom ROS Graphs using NVIDIA Isaac Transport for ROS — NVIDIA Technical Blog](https://developer.nvidia.com/blog/boosting-custom-ros-graphs-using-nvidia-isaac-transport-for-ros/)
- [NITROS Concepts — Isaac ROS Docs](https://nvidia-isaac-ros.github.io/concepts/nitros/index.html)
- [Greenwave Monitor — GitHub](https://github.com/NVIDIA-ISAAC-ROS/greenwave_monitor)
- [Greenwave Monitor Announcement — ROS Discourse](https://discourse.openrobotics.org/t/nvidias-greenwave-monitor-a-tool-for-high-performance-topic-monitoring-and-diagnostics/50477)
- [REP-2007 (Type Adaptation)](https://ros.org/reps/rep-2007.html)
- [REP-2009 (Type Negotiation)](https://ros.org/reps/rep-2009.html)
- [NVIDIA Boosts Open-Source Robotics with new ROS 2 and Physical AI contributions — Digital Watch](https://dig.watch/updates/nvidia-boosts-open-source-robotics-with-new-ros-2-and-physical-ai-contributions)
