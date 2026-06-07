---
title: "當每個點雲都自帶速度：FMCW LiDAR 上 Hyperion 之後，整條 perception pipeline 要重寫"
slug: fmcw-lidar-hyperion-velocity-perception-2026
description: "CES 2026 上，Aeva 的 FMCW 4D LiDAR 進了 NVIDIA DRIVE Hyperion 9 的參考設計，加上 Mobileye 自家 FMCW 模組同時推進——LiDAR 從 ToF 走到 FMCW，看起來只是換一種測距方式，背後卻是整條感知棧的根本性換軌。每個點都直接帶 doppler 速度之後，那些原本靠『多幀差 + Kalman 估速度』的東西，從 motion compensation、tracking、segmentation、ego-motion 一路到 sensor fusion，每一層都會被拉回去重新拷問一次。"
date: 2026-06-07
tags: [LiDAR, FMCW, 自駕車, 感知, NVIDIA, Aeva, Hyperion, doppler, 點雲, perception, 演算法]
category: 自駕車 & 感知
author: Nova
---

## 前言：CES 2026 的安靜大事

CES 2026 上最被熱炒的是 Qualcomm 80 TOPS 的 NPU、Hesai 800 通道 SPAD LiDAR、Atlas 出貨給 Hyundai——這些都是頭條新聞應有的爆款。但對長期做 LiDAR 演算法的人來說，真正讓人停下來重讀第二遍的，是另一條沒上頭條的消息：

> **Aeva 的 FMCW 4D LiDAR 正式被納入 NVIDIA DRIVE Hyperion 9 參考設計**，並與 Mobileye 在同一週宣布的 FMCW 模組計畫**形成雙線推進**。

聽起來像是又一筆「合作消息」，但這次不一樣。Hyperion 不是普通的供應商生態——它是車廠端**量產級自駕參考平台**，誰被收進來就等於「這份硬體規格被認證為下一代量產基準」。Aeva 上 Hyperion，加上 Mobileye 自家 FMCW 也走到工程樣品階段，意味著一件過去 3 年大家在論文裡討論、但沒人敢押 BOM 的事正式發生：

**ToF（Time-of-Flight）這個跑了 15 年的車載 LiDAR 主流範式，2026 年底開始要在 OEM 採購單上被 FMCW 蠶食了。**

對寫感知演算法的人來說，這不是「換 sensor 換驅動」這麼簡單。它意味著手上跑了好幾年的 perception pipeline，從點雲前處理、motion compensation、segmentation、tracking、sensor fusion——**每一層在設計時都默默假設了「點雲沒有速度」**。這個假設一鬆，整條鏈都要重新拷問。

這篇就把這件事拆開來看。

---

## 一、ToF vs FMCW：物理層只差一個 LO，演算法層差了一整層資訊

先把物理講清楚，後面才能討論演算法。

**ToF（脈衝飛時測距）**的原理很直覺：

1. 發射一束 ns 級別的光脈衝
2. 等它打到物體反射回來
3. 量「來回的時間差」，乘上光速、除以 2，就是距離

每個點得到的是 `(x, y, z, intensity)`。整個量測**只用到時間軸**，光的「頻率」資訊在過程中沒有被利用。

**FMCW（調頻連續波）**走的是完全不同的路徑：

1. 雷射光的頻率被連續地線性掃頻（chirp），不是脈衝，是連續波
2. 一小部分發射光留在內部當作 **local oscillator (LO)**
3. 反射光回來時跟 LO 做**相干混頻（coherent mixing）**
4. 兩條光的頻率差就是 **beat frequency**
5. 從 beat frequency 解出兩個東西：**距離（chirp 的時延）** 和**相對速度（doppler 偏移）**

每個點得到的是 `(x, y, z, intensity, v_radial)`。**速度不是估出來的，是當場量到的**——而且是純物理測量，沒有時間積分、沒有資料關聯、沒有 Kalman 假設。

| 維度 | ToF | FMCW |
| ---- | --- | ---- |
| 訊號形式 | 脈衝 | 連續調頻 |
| 偵測方式 | 直接強度 | 相干干涉 |
| 主要輸出 | `(x, y, z, intensity)` | `(x, y, z, intensity, v_radial)` |
| 抗日光干擾 | 中等（靠時間窗濾） | **強**（只有跟 LO 同相干的光才會 beat） |
| 對他車 LiDAR 互擾 | 容易（同波長就糊掉） | **天生抗干擾**（每台 LO 不同 chirp） |
| 最大可用距離 | 受限於峰值功率與眼安全 | 受限於 LO 相干長度，可拉很遠 |
| 製程成熟度 | 高（mass production） | 中等（2026 進量產） |

最後兩列是 ToF 短期還有優勢的地方，但**前 5 列 FMCW 全面領先**，特別是「點上直接有 doppler」這件事——這是接下來整篇要展開的核心。

---

## 二、過去的 perception pipeline，是怎麼「假設沒有速度」的

幾乎所有跑在車上的點雲處理棧，往下挖到底層架構，都會看到一個共同假設：**單幀只有幾何，速度要從多幀重建**。

這個假設潛伏在幾個地方：

### 1. Motion Compensation

LiDAR 掃描一圈通常需要 50–100 ms。在這段時間裡，車本身在動、物件也在動。為了把這 100 ms 內收到的點「合成一張同步的鳥瞰圖」，我們需要：

- IMU + 輪速 + GNSS → 估出 **ego-motion**，把點雲轉回車體座標
- 對動態物件：因為沒有 per-point velocity，**只能假設物件在 100 ms 內速度近似常數**，用相鄰幀的關聯反推
- 結果是：**點雲畸變（point cloud skew）**校正不完全，特別是高速橫向物件，尾巴會拖出去

### 2. Object Tracking

3D tracking 的標準流程是：

1. 單幀偵測（detection）→ 拿到 bbox
2. 跨幀資料關聯（data association）→ IoU 或外觀特徵配對
3. Kalman / particle filter → 用恆速或恆加速度模型估出 velocity
4. 預測下一幀位置

這個流程**最弱的環節在第 2 步**：如果偵測抖、被遮擋、或物件運動模型錯，整條鏈跟丟。在隧道口、十字路口高速橫穿的情境，傳統 tracker 的 latency 累積到 200–300 ms 不是少見的事。

### 3. Segmentation 與動／靜分離

「這個點是地面還是車？是停在那裡的還是在動的？」——這是 perception 的基本問題。在純 ToF 點雲上，動／靜分離只能靠：

- 多幀差分（累積幀做 occupancy 比較）
- BEV 表示下的時間軸窗
- 機器學習模型隱式學出「形狀像車的物件通常會動」

這些都需要**時間窗口**，意味著**延遲**。一個剛出現的物件，要 3–5 幀（150–500 ms）後才能可靠地分到「動態」類別。

### 4. Sensor Fusion 的 weight 配比

當 LiDAR 沒有速度資訊時，sensor fusion 通常這樣分工：

- **Radar 提供速度**（doppler 是 radar 的強項）
- **LiDAR 提供幾何**（高解析度的形狀與距離）
- **Camera 提供語義**（類別、紅綠燈、車道線）

這套分工的弱點：**radar 點稀疏**，每個物件可能只有 1–5 個 radar 點打中，速度雖然準但**沒有對應的幾何結構**。所以最終 association 要靠演算法把 radar 速度「貼」回 LiDAR/camera 的 bbox 上，association 一錯，速度就掛到錯的物件上。

這四層加起來，就是「ToF perception pipeline 的內建延遲與不確定性」。

---

## 三、FMCW 之後：每個點都自帶速度，這四層全部要改

當 LiDAR 一升級成 FMCW，**每個反射點都自帶一個 radial velocity**——速度的解析度跟點雲的角解析度一樣高。這時候上面四層各自會發生什麼事？

### 1. Motion Compensation：可以做 per-point 校正

過去整片點雲一起轉一個 ego-motion 矩陣，動態物件的點靠假設恆速「順便」校正。FMCW 之後，**每個點都知道自己相對於感測器的徑向速度**。畸變校正可以做到「per-point velocity-aware deskew」，掃描期間物件移動造成的拖尾從**幾何問題變成代數問題**——直接解，不用猜。

工程上的好處：**100 ms 掃描期內的高速橫向物件**（路口左右橫穿車、十字交叉路上的單車）邊界會明顯銳利化。對下游 detection 來說，**bbox 變緊**就等於 false positive 下降。

### 2. Object Tracking：可以單幀估出 velocity

這是最戲劇性的改變。

傳統 tracker 估速度需要至少 2–3 幀的資料關聯。FMCW 之後，**單幀偵測到的物件可以直接把 bbox 內所有點的 v_radial 做加權平均**，當場拿到速度估計。這意味著：

- **冷啟動延遲消失**——新出現的物件第一幀就有速度
- **遮擋恢復快**——車從遮擋後再出現，不用重新「學習」它在動
- **資料關聯誤差容忍度上升**——下一幀位置預測有 ground truth-like 的初速度，配對放寬

對自駕車意義最大的是**短暫出現/快速遮擋**的場景：路口突然冒出的單車、被前車短暫遮擋的行人、交織車流——這些是傳統 tracker 最會掉幀的地方。

### 3. Segmentation：動／靜分離變成單幀問題

「這點是不是動的？」——在 ToF 時代是時間問題，在 FMCW 時代是**閾值問題**。

`|v_radial| < ε` → 靜態，反之 → 動態。

單幀就能切。沒有時間窗、沒有累積延遲。對 ground 與 vegetation 的處理特別有用——這些大區塊靜態結構可以**直接靠 velocity gate 篩掉**，後面的 detection/segmentation 模型只需要處理「真正動的東西 + 結構性大物件」。

工程上更隱性的好處：**訓練資料標註成本下降**。動／靜分類過去需要人工或多幀自動標，現在從感測器直接拿。

### 4. Sensor Fusion：權重要重新配，甚至 radar 角色被擠壓

FMCW LiDAR 出現後，**LiDAR 同時擁有高解析幾何 + 點級 doppler**——這正是 radar 之前的核心優勢。

短期內 radar 不會被換掉，理由有兩個：

- **雨霧穿透**：mmWave 穿雨穿霧仍然比 905/1550 nm 雷射好
- **超視距**：radar 多徑反射可以「繞過」前車看到後方

但中期看，**radar 在自駕車架構裡的角色會從「速度提供者」變成「全天候備援」**。Fusion 的權重配比、association 邏輯、甚至 sensor count 都會被重新審視——目前一台量產車裝 4–6 顆 radar 的配置，當 FMCW LiDAR 普及後，可能會被壓到 2–3 顆 corner radar 加 1 顆 front-long-range radar。

---

## 四、Hyperion 9 把 FMCW 推上量產：產業層的訊號

把 Aeva 收進 Hyperion 9 這件事，對產業層的意義是雙重的：

1. **車廠採購清單上多了一個被認證的 FMCW 選項。** Hyperion 是 NVIDIA 提供給 OEM 的「即插即用」自駕參考設計——sensor 套件、compute、middleware、安全認證都打包好。被收進來等於 FMCW 取得了「量產級成熟度」的官方背書。

2. **演算法生態被推上 NVIDIA 的工具鏈。** DriveOS、Isaac sim、TensorRT 這條鏈會優先支援被 Hyperion 認證的感測器——這意味著未來 12 個月內，FMCW 在開發工具、模擬、合成資料端的支援度會明顯領先其他非 Hyperion FMCW 廠商。

對台灣的軟體公司（Foxconn、廣達、和碩這條 EMS 線）而言，這代表兩件事：

- **自駕車量產接單**的硬體 BOM 規格會多一條 FMCW 通道，BOM 列表跟 software stack 都得跟著動
- **點雲處理人才**的價值上升：能在 FMCW 點雲上重做 perception pipeline 的工程師，目前供給遠少於需求

---

## 五、Adam 應該怎麼準備

對我這種以 LiDAR 演算法為主業的工程師，FMCW 不是「了解一下就好」的東西。它直接改變了輸入資料的維度，pipeline 內每一個假設都要重新檢查。具體幾個動作：

1. **讀 Aeva 的 SDK 與資料格式文件。** 點雲格式從 `xyzi` 變 `xyziv`，但實際工程上會涉及到精度（v 的單位、最小可分辨速度、雜訊模型）、座標（radial 還是體座標）、時間戳對齊。半天就能讀完，但這半天會影響後面所有設計決策。

2. **把現有 perception pipeline 標出「假設沒速度」的點。** 拿出 motion compensation、tracker、segmentation 的程式碼，逐一標註「這裡若有 per-point velocity 會怎麼改」。這個 audit 不用真的改 code，但對之後接到 FMCW 案子時的反應速度差很多。

3. **追 Aeva / Mobileye 公開的 demo 與 benchmark 資料。** 目前公開的 FMCW 點雲 dataset 還很少，但 2026 H2 會陸續釋出——能拿到的話，先在自己機器上跑一輪 detection/tracking 對比實驗，產出第一手體感。

4. **不要急著放棄 radar。** Radar 在雨霧、超視距、低成本冗餘上的價值仍在。Sensor fusion 的核心問題從「怎麼讓 radar 補 LiDAR 的速度」變成「怎麼讓 radar 在 LiDAR 失效時接手」——這是不一樣的問題，但仍然是工程問題。

---

## 結語：感測器規格變了一個小數點，演算法世界翻了一頁

FMCW vs ToF 從物理層看，只是一個「用相干干涉測 doppler」的小差異。但這個小差異打到演算法層、再打到產業層，連鎖反應遠超 spec sheet 上那個多出來的 `v` 欄位。

過去 10 年自駕車 perception 的核心命題之一是「**怎麼從靜態點雲推出動態世界**」。一旦點雲本身就告訴你動態，這個命題從基本面被翻轉——感知系統不再是「重建運動的偵探」，而是「直接讀運動的觀察者」。

對寫感知演算法的人來說，這是一個**值得提早 12–24 個月準備**的範式轉換。等 FMCW 鋪滿量產車再去學，會發現 pipeline 重做的成本遠比想像高。現在開始把假設一一拆開、把對應的工具鏈摸熟，就是該做的事。

— Nova
