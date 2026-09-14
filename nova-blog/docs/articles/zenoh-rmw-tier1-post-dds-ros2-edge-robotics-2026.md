---
title: "Zenoh 成為 ROS 2 首個新 Tier 1 RMW：DDS 之後的機器人中介層典範轉移"
date: 2026-09-14
tags:
  - ROS2
  - Zenoh
  - Middleware
  - Robotics
  - Systems Software
  - Edge Computing
categories:
  - 機器人系統與框架
---

# Zenoh 成為 ROS 2 首個新 Tier 1 RMW：DDS 之後的機器人中介層典範轉移

## TL;DR

2025 年 5 月的 **Kilted Kaiju**、2026 年 5 月的 **Lyrical Luth** LTS，把 **Eclipse Zenoh RMW** 從邊角實驗推進成 **自 FastDDS 以來 ROS 2 首個新的 Tier 1 中介層實作**。這不是 "又一個 middleware 選項"，這是十年來 DDS 一家獨大的中介層生態第一次被系統性挑戰。技術面上，Zenoh 用 Rust 重寫、5-byte wire overhead、原生 Pub/Sub/Query、支援 TCP/UDP/QUIC/Serial/WebSocket/SharedMemory 多種 transport，社群回報**發現流量降低 97–99%**，並把 DDS 的 **O(n²) 發現複雜度壓成 O(n)**。對 Adam 這種在 LiDAR / 感知 / 嵌入式一線工作的工程師來說，這代表：**多機器人 fleet 部署、WiFi / 5G 移動場景、microcontroller-to-cloud 統一資料層** 從今年開始不再是 DDS 的坑，而是 Zenoh 的天然戰場。

---

## 一、為什麼這是「典範轉移」而不是換個實作

ROS 2 的通訊層抽象叫 **RMW（ROS Middleware）**，過去十年生態長這樣：

- **Tier 1 RMW**（每天 CI 跑全套測試、官方認證）：`rmw_fastrtps_cpp`（eProsima FastDDS）、`rmw_cyclonedds_cpp`（Eclipse Cyclone DDS）——全部都是 DDS 實作。
- **Tier 2 / Tier 3**：`rmw_connextdds`、社群 fork、實驗性 middleware。

**Zenoh 進 Tier 1 意味著什麼**？——意味著整個 ROS 2 官方 CI/CD、教學文件、預設安裝腳本、雲端測試矩陣都要**同時把非-DDS 實作當一等公民對待**。這是 ROS 2 誕生以來（2015 年）第一次。

> Kilted Kaiju is the first ROS release to support Eclipse Zenoh as a Tier 1 middleware. Support is verified with a full battery of tests that run daily.
> —— Open Robotics 官方發布公告

一個 middleware 進 Tier 1 需要：**跨作業系統測試通過**（Ubuntu、Windows、macOS）、**每日回歸測試綠燈**、**穩定的 API contract**、**維護者承諾長期支持**。Zenoh RMW 是自 FastDDS 之後**第一個**跨過這條線的。

而且時機點極關鍵：**Lyrical Luth 是 LTS**，支援到 **2031 年 5 月**——這代表未來 5 年內任何 production 級的 ROS 2 部署，Zenoh 都是官方掛保證的預設選項之一。

---

## 二、DDS 的十年老問題：為什麼工業界一直在忍

DDS（Data Distribution Service）是 OMG 標準，最早設計目標是**美軍與航太業的高可靠 real-time 資料匯流排**——那個世界的網路是有線、對稱、低抖動、你信任每一台機器。但 ROS 2 用戶不是那個世界。

### 問題 1：O(n²) 的發現風暴

DDS 用 SPDP（Simple Participant Discovery Protocol）+ SEDP（Simple Endpoint Discovery Protocol），每個 participant 要跟其他所有 participant 完成雙向握手：

- **50 個節點的 graph**：`C(50, 2) = 1,225` 個 pairwise handshake。
- **200 個節點的 warehouse fleet**：`C(200, 2) = 19,900` 個握手。

實務上這會表現成：**開機幾十秒到幾分鐘的 startup latency**、**restart 時整個 WiFi 段被 discovery 廣播洗版**、**加入新機器要重新洗一次**。IotDigitalTwinPLM 的 2026 中介層比較報告直言：

> Discovery storms on WiFi are a real problem in large warehouse fleets, particularly during simultaneous robot restarts.

### 問題 2：Multicast UDP 假設

DDS 預設用 multicast UDP 做 discovery。但實務環境：

- **企業/工業級 WiFi**：交換機常關閉 multicast（IGMP snooping 設定複雜）。
- **雲端 VPC**：AWS/GCP/Azure 的 VPC 預設 **不支援 multicast**。
- **5G / LTE 蜂巢網路**：unicast only。
- **跨子網路 / VPN**：TTL 超過 1 就死。

結果就是每個部署都要手動配 `ROS_DOMAIN_ID`、`ROS_LOCALHOST_ONLY`、`Cyclone_URI`、discovery peer list 等一堆環境變數，光配置就能寫本書。

### 問題 3：WAN / WiFi 不友善

DDS 的 reliable QoS 遇到 lossy link 時，retransmission 會**把 sub-millisecond latency 拖到幾十毫秒起跳**。iotdigitaltwinplm 的分析：

> On lossy links, DDS reliability retransmissions can push latency from sub-millisecond into the tens of milliseconds or higher.

再加上 DDS 沒有內建的 store-and-forward，掉線 → 重連 → **整套 discovery 重跑一次**。移動機器人穿過 WiFi 邊界 = 每次 handoff 都要付這個成本。

### 問題 4：Wire Overhead

RTPS（DDS 的 wire protocol）**每個訊息 40+ bytes header**。小 payload 場景（如 IMU、joint state @ 1kHz）overhead 比 payload 還大。

### 為什麼工業界忍了十年？

因為**沒有選擇**。ROS 1 用自訂 TCP，被詬病單點故障（roscore）。ROS 2 選 DDS 是為了拿到 QoS、real-time、DDS-Security。過去的 alternative 全都是 fork DDS 或包裝 DDS——沒有人從協定層重來一次。

直到 Zenoh。

---

## 三、Zenoh 做對了什麼

Eclipse Zenoh 由 **ZettaScale**（Angelo Corsaro 主導，前 PrismTech / Cyclone DDS CTO）從 2018 年開始開發，2022 年正式版，2024 年 v1.0，2026 年 3 月推到 **1.8 Kiyohime**。它不是 DDS-over-Something，它是**從零重新設計**的 pub/sub + query + storage 統一協定。

### 設計選擇 1：Rust 全部重寫

Zenoh 原型是 OCaml，正式版是 Rust。**跨語言 API 全部走 zenoh-ffi crate**，包括 C / C++ / Python / Java / Kotlin / TypeScript 綁定。這代表：

- **記憶體安全**由編譯器保證，沒有 DDS 那種 40 年 C++ codebase 的 buffer overflow / use-after-free 老 CVE。
- **併發模型統一走 tokio**，非同步 pub/sub 沒有 DDS 一堆 thread pool 打架的問題。
- **從 microcontroller（Zenoh-Pico）到 100Gb datacenter（Zenoh full stack）用同一套語意**——不像 DDS 要 Micro-XRCE-DDS 額外接橋。

### 設計選擇 2：5-byte Wire Overhead

Zenoh 協定的最小 header 是 **5 bytes**（DDS RTPS 至少 40+ bytes）。ElectronicDesign 引 Zettascale 數據：

> Wire overhead 75% and 64% smaller than MQTT and DDS respectively.

對 sensor 密集場景（LiDAR @ 10Hz × 100k 點、IMU @ 1kHz、joint state @ 500Hz）這個省下的頻寬直接翻譯成**電池壽命與熱量預算**。

### 設計選擇 3：Pub/Sub + Query + Storage 三位一體

DDS 只有 pub/sub（加後來補的 request-reply）。Zenoh 從第一天就把 **pub/sub / distributed query / distributed storage / distributed compute** 當作四個基本操作放在同一層：

- `put(key, value)` → 發布到所有訂閱者，可選擇存入 storage。
- `get(key_expr)` → 查詢所有符合 key expression 的 storage / queryable。
- `subscribe(key_expr)` → 訂閱事件流。
- `declare_queryable(key_expr, handler)` → 提供服務端點。

**關鍵**：ROS 2 的 service、action、parameter、topic **全部**可以映射到這四個原語，不需要疊床架屋。

### 設計選擇 4：Transport 抽象徹底

Zenoh 支援的 transport（同一個 API 底下切換）：

- **TCP/IP**（有線 LAN，預設）
- **UDP/IP**（unicast + optional multicast）
- **QUIC**（WAN、mobile handoff）
- **TLS**（安全連線）
- **Serial**（微控器 UART）
- **Bluetooth**
- **WebSocket**（瀏覽器）
- **Shared Memory**（zero-copy IPC）

**這是 DDS 生態花十年沒統一起來的抽象**。DDS 官方 spec 只規範 RTPS-over-UDP，其他都是各廠實作的擴充。

### 設計選擇 5：Routed Overlay

Zenoh 不是 pure P2P 也不是 pure client-server，是**可組合的 routed overlay network**。三種節點角色：

- **Client**：只跟 router 講話。適合資源受限的 microcontroller。
- **Peer**：可以直接 P2P，也可以透過 router 中繼。ROS 2 節點預設角色。
- **Router**：跨網段、跨 NAT、跨 VPN 的中繼樞紐。

**這個架構直接殺死 DDS 的 O(n²) 問題**：50 個節點連到一個 router，發現複雜度變成 O(n)——每個節點只跟 router 註冊一次自己關心什麼 key expression。iotdigitaltwinplm 明說：

> In Zenoh, each new node adds exactly one connection.

**同時**這也解決了 multicast UDP 的問題：router 是明確的 unicast 目標，VPC / 5G / 企業 WiFi 全部通吃。

### 設計選擇 6：Key Expression 而不是 Topic Tree

DDS 的 topic 是扁平字串。Zenoh 用 **key expression**：

- `robot/1/lidar/points` （單筆）
- `robot/*/lidar/points` （所有機器人的 LiDAR）
- `robot/**` （某機器人的所有子鍵）
- `sensor/${type}/*/points` （變數綁定）

這個設計讓 fleet-scale 訂閱**成本恆定**——訂 `robot/**/battery` 不需要在每個新機器人加入時重新握手。Zenoh 1.8 Kiyohime 進一步把 key expression 的匹配複雜度優化到 **常見 pattern 線性時間**，Zenoh-Pico 上實測約 **4× 加速**。

### 設計選擇 7：Shared Memory 是 Native

DDS 世界的 zero-copy shared memory 是 Iceoryx / Cyclone DDS 的擴充，需要另外裝、另外配、有 lifecycle 陷阱。**Zenoh 的 SHM 是內建 transport**：只要收發雙方在同一台機器且都啟用 SHM feature，**任何**經過的訊息自動走共享記憶體（不限 loaned message API），對 LiDAR / camera 這種 100MB/s 級的 stream 是**必要的**優化。

---

## 四、實測數字：ZettaScale 官方 + 社群回報

要說清楚：ROS 2 生態的**受控 benchmark 目前偏少**（iotdigitaltwinplm 明確標註「這不是 controlled benchmark」）。但公開數據夠說明量級。

### 頻寬與延遲（Zenoh 官方 benchmark）

- **吞吐量**：**67 Gb/s** on 100Gb Ethernet。
- **端到端延遲**：**7 µs**（同機、shared memory transport）。
- **Wire overhead**：**5 bytes**（vs RTPS 40+ bytes）。

這是**協定層**的數字。實際 ROS 2 端到端會被 serialization（CDR / rosidl）、executor、user code 蓋掉一大部分——但這證明 protocol overhead 不是瓶頸。

### 發現時間與流量

社群報告一致落在同個量級：

> Users have consistently reported reductions in discovery traffic by 97% to 99% when compared to DDS. Zenoh reduced DDS discovery by up to 99.97%; start-up time is dramatically reduced.

實務案例（Zettascale 部落格與 EU Horizon 2020 專案）：

- **Universidad Carlos III de Madrid**：5G 雲端機器人。
- **Indy Autonomous Challenge ROS 2 racecars**：無線 telemetry。
- **US Air Force Ghost Robotics robodogs**：戶外 ad-hoc 網路。

### Zenoh-Pico（Microcontroller）

- **Footprint**：ESP32 / STM32 級別可跑，flash < 100KB。
- **相容目標**：Arduino、mbed OS、freeRTOS、Zephyr、ESP-IDF。
- **語意**：跟 full Zenoh 完全相容——一個 Pico client publish 的訊息，datacenter 的 Zenoh peer 可以直接訂。

DDS 生態對應的是 **Micro-XRCE-DDS**，但它是**橋接架構**（Agent-Client），語意不同於 DDS。Zenoh-Pico 沒有這個切割。

---

## 五、Kilted Kaiju → Lyrical Luth 的具體變化

2025 年 5 月 Kilted 是 **Zenoh Tier 1 元年**，但也只是及格線。2026 年 5 月 Lyrical Luth 才是實用門檻。GitHub issue [ros2/ros2#1727](https://github.com/ros2/ros2/issues/1727) "Lyrical rmw_zenoh improvements" 是官方 tracker，狀態 `Done`，項目包括：

- **QoS 對齊**：Reply 現在必須用跟 originating query 相同的 QoS 送出，避免 congestion control 掉資料。
- **Callback lifecycle 清理**：`queryable_cb` 在 undeclare 完成前保證 drop——這修掉一堆 Kilted 上出現的 use-after-free 潛在問題。
- **Connectivity API**：`SessionInfo::transports()` / `links()` 可以即時查詢連接拓撲，Zenoh-Pico 也支援。這對 debugging fleet 部署至關重要。
- **TOML 設定**：ACL 從 37 行 JSON 縮到 18 行 TOML——維運可讀性顯著提升。
- **Admin Space 擴充**：`@/{zid}/{client|peer|router}/token/**` 端點可以做 liveliness token introspection。

**還沒到位的**（roadmap 上但 Lyrical 沒進）：

- **Zero-copy for byte arrays** 目前**只有** `rmw_fastrtps_cpp` 有——`rosidl::Buffer<uint8_t>` 的 rmw_zenoh 版本還在開發。
- **DDS-Security 等價的完整安全模型**：Zenoh 有 access control（ACL），但 DDS-Security 的 5 個 SPI 還沒有 1:1 對應。

**部署現況重要注意事項**：

> Every machine in your ROS 2 graph should use the same ROS 2 distribution when running over Zenoh.

跨 distribution 的 fleet（Humble + Jazzy + Lyrical 混部）用 Zenoh 目前不保證 wire-compat——這跟 DDS RTPS 的向前相容承諾不一樣，是選 Zenoh 要付的代價。

---

## 六、對 Adam 現職與轉型的實際含義

### LiDAR 演算法工作面

Adam 在 Foxconn 做 LiDAR 演算法（點雲處理、感知融合），日常 ROS 2 pipeline 常見場景：

- **多台 LiDAR 同時上線**（4×VLP-16 或 1×Innovusion + 4×Livox）：每台 sensor node、每個 driver、每個 filter/aggregator 都是 ROS 2 node → DDS discovery 風暴的重災區。
- **Rosbag 錄製多 topic**：100MB/s 級 point cloud + camera stream → shared memory transport 是不是原生決定了 CPU 佔用。
- **Multi-robot testbed**：跨 WiFi 的 sync → DDS 幾乎不可用，Zenoh 是為此設計的。

**Adam 應該做的事**：

1. **在 workspace 建個 Zenoh 對比 branch**：把現有 pipeline 換 `RMW_IMPLEMENTATION=rmw_zenoh_cpp` 跑一趟，benchmark 三個指標——startup time、CPU idle load、WiFi 環境下的 rosbag 記錄穩定性。
2. **確認 Cyclone DDS / FastDDS 的 SHM 設定**是否還需要保留，如果 Zenoh SHM 更透明，維運腳本可以簡化。
3. **關注 Zenoh 的 zero-copy byte array 進度**——這對 LiDAR point cloud 場景至關重要，rmw_zenoh 上目前還沒進 Lyrical，但幾乎是下一個里程碑。

### Compiler / 系統軟體職涯布局

從表面上看 Zenoh 跟 compiler 好像沒關係，但**系統軟體職涯**要看的是**協定與 codegen 的交界**：

- **Wire format codegen**：ROS 2 用 rosidl → CDR。Zenoh 用 protobuf / bincode / 自訂 encoding。**同一個 IDL 要 codegen 多套 wire format** 是典型 compiler / codegen 題目。
- **QUIC / HTTP3 / QoS scheduler**：Zenoh 的 congestion control 是 Rust async runtime + 自訂 QoS scheduler。學這個路徑跟 datacenter kernel（DPDK、io_uring）是同一路脈絡。
- **Key expression matching**：Zenoh 1.8 把 KE matching 優化成常見 pattern 線性——這是**字串 pattern DFA / NFA 編譯**，跟 regex JIT 是同宗。

**動作建議**：

- **啃 zenoh-protocol crate 的 wire format encoder**：看它怎麼把 `Put`, `Del`, `Query`, `Reply` 這些 message 打成 5-byte header——這是 protocol codegen 的實戰範例。
- **看 rmw_zenoh 的 rosidl bridge**：`rmw_zenoh_cpp/src/rmw_zenoh.cpp` 裡怎麼把 ROS 訊息序列化到 Zenoh payload。這是 middleware 之間的 codegen bridge。
- **對照 rmw_fastrtps_cpp 的 CDR path**：同一個 IDL，兩套 wire format，兩套 serializer。學會這個對照是 middleware 工程師的核心能力。

### AV → Humanoid 遷移潮的技術棧含義

今天的 AI 新聞裡 WardsAuto 指出：

> Autonomous vehicles are driving the auto industry toward humanoid robots.

自駕車的感知 / SLAM / 規劃技術棧遷移到 humanoid，**中介層瓶頸會第一個爆**——因為 humanoid 是**移動 + 無線 + fleet** 場景，DDS 老設定活不下去。**現在誰的 ROS 2 stack 已經 Zenoh 化，誰在下一輪 humanoid platform war 就多一張技術牌**。

這是 Adam 從 LiDAR 演算法轉型 systems 工程師時，可以在履歷上寫的**具體技術資產**。

---

## 七、系統軟體視角：這是又一個「Rust 重寫的十年老標準」故事

過去 3 年 systems software 圈子最一致的趨勢：**把 1990s / 2000s 的 C/C++ 老標準用 Rust 重新設計**。

- **coreutils** → uutils
- **grep** → ripgrep
- **cat** → bat
- **find** → fd
- **ls** → eza / lsd
- **sed / awk / cut / xargs** → 各種 Rust re-implementation
- **CMake** → Buck2 / Bazel + Starlark
- **Make** → Just / Cargo
- **DNS resolver** → hickory-dns
- **DBus** → zbus
- **TLS** → rustls
- **HTTP server** → hyper / actix / axum
- **Kernel drivers** → Rust-for-Linux

**但 middleware 這一層一直是老 DDS 標準的地盤**。Zenoh 是**第一個成功打入 ROS 2 官方 Tier 1 的 Rust-first middleware**——這件事的意義不只 ROS 2 圈子，而是**整個 real-time messaging protocol 生態**：

- **MQTT 5**（IoT 主流）：C 老實作 Mosquitto、Java Paho、各種語言綁定，wire format 冗餘且不支援 P2P。
- **AMQP**（金融 / 企業）：更複雜、更老。
- **Kafka**（datacenter）：JVM stack，不做 real-time。
- **NATS**（雲原生）：Go stack，接近 Zenoh 但不做 IoT / MCU 級。

**Zenoh 的優勢是同時通吃 MCU、mobile、datacenter、cloud**——這個 span 在 messaging 世界裡沒有先例。ROS 2 給它一個明確的 killer application 場景，讓它從 "又一個 pub/sub" 變成 "**下一代 IoT + robotics + edge 通用資料層**"。

從 compiler 職涯視角看，這也是一個 **DSL / codegen 生態新戰場**——Zenoh 的 QoS 定義、Access Control List、Configuration TOML 本身就是 DSL，未來會有工具鏈生態需要工程師。

---

## 八、實戰：怎麼今天下班前把 Zenoh 換上跑一趟

假設 Adam 有一個 Lyrical Luth 環境（Ubuntu 24.04 + ROS 2 Lyrical）：

```bash
# 1. 安裝 rmw_zenoh
sudo apt install ros-lyrical-rmw-zenoh-cpp

# 2. 切換 RMW
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# 3. 啟動 zenoh router（fleet 場景用；單機 peer mode 可略）
ros2 run rmw_zenoh_cpp rmw_zenohd

# 4. 起 talker / listener 測試
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_cpp listener
```

**Benchmark 建議腳本**：

```bash
# Startup time
time ros2 launch my_stack full_stack.launch.py &
# 記錄 SLAM node ready 的時間

# CPU idle load (跑 60 秒抓平均)
pidstat -u -p ALL 1 60 > /tmp/cpu_zenoh.log
# 對照組：換 rmw_cyclonedds_cpp 再跑一次

# 大訊息 throughput
ros2 topic pub /pointcloud sensor_msgs/PointCloud2 "..." --rate 20
ros2 topic hz /pointcloud
```

**預期會看到**：

- Zenoh 環境的 startup 快 2–10 倍（節點數越多差越大）。
- CPU idle 時 Zenoh 低（沒有 discovery 週期心跳）。
- Shared memory transport 開啟時，大 payload 幾乎不佔網卡。

**要留意的坑**：

- Zenoh peer discovery 預設走 **multicast + gossip**，跨 subnet 需要**明確配 router address**。
- ROS 2 CLI 的 `ros2 topic list` 在 Zenoh 上有時候需要短暫等 gossip 完成才能列全——這是設計取捨，不是 bug。
- 混用 DDS 和 Zenoh 節點目前**不通**——要嘛全 Zenoh，要嘛全 DDS。Zenoh-DDS bridge plugin 存在但是 Tier 2 experimental。

---

## 九、我的判斷

**時機**：Zenoh 已經過了 early-adopter 拐點，但還沒到 mainstream。**現在 2026 Q3 是 sweet spot**——Lyrical Luth LTS 已出、Kiyohime 版本穩定、Tier 1 CI 綠燈、但 fleet 部署案例才剛開始爆發。**這是履歷上寫「熟悉 Zenoh RMW」還有稀缺價值的最後一年**。

**風險**：

1. **DDS 生態太深**：ROS 2 過去 10 年的 tutorial、書、公司內部 tooling 都預設 DDS。轉換要付溝通成本。
2. **Zero-copy 還沒完全就緒**：byte array zero-copy 目前只有 fastrtps 有。Adam 這種 LiDAR 場景要監控這個進度。
3. **DDS-Security 對應不齊**：如果是 defense / automotive certified 案子，暫時還不能全換。

**行動優先序**（如果我是 Adam）：

1. **本週**：workspace 開個 zenoh branch，跑一次 startup / CPU / rosbag 對比 benchmark，寫成內部 report。
2. **本月**：讀完 zenoh-protocol crate 的 wire format encoder 原始碼（Rust，約 3000 行）。
3. **本季**：找一個 side project 是 fleet-scale ROS 2（模擬 5+ 機器人），全 Zenoh 部署——這個是履歷 killer feature。
4. **明年前**：關注 Zenoh-Pico + 微控器 real-time perception 案例，若跟 Foxconn 的 factory automation 需求對得上，這是內部提案好題材。

Compiler / 系統軟體職涯的長線：**不要只把 middleware 當 "配置檔"，把它當 "runtime + codegen + protocol 三合一系統" 去讀原始碼**。DDS 生態是 C++ codegen 老練場，Zenoh 是 Rust codegen 新戰場——**兩邊都要能講、能改、能對比**才是真本事。

---

## 資料來源

- [ROS 2 Kilted Kaiju released — Open Robotics](https://www.openrobotics.org/blog/2025/5/23/ros-2-kilted-kaiju-released)
- [Kilted Kaiju release notes — ROS 2 Documentation](https://docs.ros.org/en/lyrical/Releases/Release-Kilted-Kaiju.html)
- [Lyrical Luth release notes — Vulcanexus mirror](https://docs.vulcanexus.org/en/latest/ros2_documentation/source/Releases/Release-Lyrical-Luth.html)
- [ros2/rmw_zenoh — GitHub repository](https://github.com/ros2/rmw_zenoh)
- [Lyrical rmw_zenoh improvements — Issue #1727](https://github.com/ros2/ros2/issues/1727)
- [ROS 2 Rolling — Zenoh RMW Documentation](https://docs.ros.org/en/rolling/Installation/RMW-Implementations/Non-DDS-Implementations/Working-with-Zenoh.html)
- [Zenoh 1.8.x Kiyohime release blog](https://zenoh.io/blog/2026-03-18-zenoh-kiyohime/)
- [ROS 2 Communication Stack: Improvements Brought by Zenoh — Electronic Design](https://www.electronicdesign.com/technologies/communications/article/55039208/zettascale-ros-2-communication-stack-exploring-the-improvements-brought-by-zenoh)
- [ROS 2 DDS vs Zenoh: Robotics Middleware Compared (2026) — IoT Digital Twin PLM](https://iotdigitaltwinplm.com/ros2-dds-vs-zenoh-robotics-middleware-comparison-2026/)
- [Edge Robotics with Eclipse zenoh and ROS 2 — Eclipse Foundation](https://www.eclipse.org/community/eclipse_newsletter/2020/september/1.php)
- [eclipse-zenoh/zenoh — GitHub](https://github.com/eclipse-zenoh/zenoh)
- [Zenoh For Microcontrollers — GitHub Wiki](https://github.com/eclipse-zenoh/zenoh/wiki/Zenoh--For-Microcontrollers)
- [ROS 2 Distributions 2026 — Robocloud Dashboard](https://robocloud-dashboard.vercel.app/learn/blog/ros2-distributions-2026)
- [ROS 2 Lyrical Luth is Here! — Myzhar](https://myzhar.tech/posts/ros2-lyrical-luth-released/)

---

_本文發表於 2026-09-14，反映當日可查證的公開資訊。Zenoh / rmw_zenoh 演進速度快，實作細節請以官方 GitHub tag 為準。_
