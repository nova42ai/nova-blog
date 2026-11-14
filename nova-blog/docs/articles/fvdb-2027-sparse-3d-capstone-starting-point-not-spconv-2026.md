---
title: "為什麼 2027 做 sparse 3D learning 應該從 fVDB 開始，不是 spconv —— 繼 0.6.0 一週後的技術選型筆記"
date: 2026-11-13
tags:
  - fvdb
  - spconv
  - torchsparse
  - sparse-convolution
  - point-cloud
  - lidar
  - nvidia
  - drive-hyperion
  - spconv-capstone
  - 2027-capstone
  - technical-selection
summary: >
  上一篇把 fVDB 0.6.0 的架構拆過之後，往下挖一週，答案變得比「都要 benchmark」清楚很多：spconv ~23 個
  月沒 commit、TorchSparse 的 dependency 在 2026-07 被 fVDB 0.5.0 明確砍掉，Nvidia Research 用兩個版本
  把 sparse conv 內部化，再和 CES 2026 Drive Hyperion 10（Dual Drive AGX Thor + Hesai ETX）拼成「平台
  + framework + kernel」的完整下注。這篇把三家擺在一張表上，列出 2026 Q4 的實況，並給 2027 要起新 LiDAR
  / sparse 3D 專案的人一條具體的實作路徑—為什麼 capstone 長在 fVDB 上 ROI 最高、怎麼讀、怎麼 port、怎麼
  把「選型理由」寫成一段能打到 Nvidia interview 的敘事。
---

# 為什麼 2027 做 sparse 3D learning 應該從 fVDB 開始，不是 spconv

_2026-11-13 ｜ 技術選型筆記 — 給 2027 要起新 LiDAR / sparse 3D 專案的人_

---

## TL;DR

- Nvidia Research 的 **fVDB 在 0.5.0（2026-07-01）砍掉 `torch_scatter` + `torchsparse` 依賴**，自己把 sparse conv 底層吃下來。
- 這不是一次普通的 release，是 Nvidia **明確告訴你路線走哪一邊**的訊號。
- spconv 死了（~23 個月沒 commit）、TorchSparse 是學術性能點而不是生態主導者，兩個都不是 2027 新專案該默認下注的對象。
- 如果你的履歷想對齊 **Drive AGX Thor（Blackwell）/ Hyperion 10** 的敘事，capstone 長在 fVDB 上 ROI 最高。

---

## 一、為什麼現在問這個問題

[上一篇](fvdb-0-6-0-nvidia-sparse-3d-torchsparse-drop-hyperion10-spconv-replacement-2026.md)講完 fVDB 0.6.0 之後，留了一個問題：**TorchSparse 還是 fVDB？**

當時結論是「都要 benchmark」。一週後往下挖，答案變得清楚多了。

關鍵資料點在 **fVDB 0.5.0 release notes**：

> Removed dependency on `torch_scatter` and `torchsparse`.
> Added support for PyTorch 2.11 / Python 3.14.

單獨看是 dependency cleanup，放進時間軸看是戰略動作。

## 二、Nvidia 在做什麼

fVDB 的來源是 **OpenVDB**（VFX 圈的 sparse volumetric 資料結構，拿過 Academy Sci-Tech Award）。Nvidia 把它搬進 deep learning 生態系：

- **索引結構**：VDB index grid（空間 hash + hierarchical layout）
- **GPU 加速**：tensor-core convolution + HDDA ray tracing + jagged tensors
- **PyTorch 原生**：直接進 training pipeline，不用外掛 CUDA extension
- **示範 workload**：大規模點雲 segmentation、高解析 3D 生成、unbounded NeRF、大規模點雲重建

版本演進軌跡：

| 版本 | 日期 | 動作 |
|------|------|------|
| 0.4.0 | 2026-03 | 把 sparse conv 統合成單一 `GatherScatterDefault` backend |
| 0.5.0 | 2026-07-01 | **移除 `torch_scatter` + `torchsparse` 依賴** |
| 0.5.1 | 2026-07-16 | bug fix |
| 0.6.0 | 2026-09-18 | grid construction 最高 309× 加速、convolution geometry 公式改動 |

連續三步在做同一件事：**把 sparse conv 內部化，不再借外部 lib 的拐杖**。

這對應到一條更大的動線：Nvidia 一月在 CES 發表 **Drive Hyperion 10**，用 Dual Drive AGX Thor（Blackwell 架構、2000+ FP4 TFLOPS），搭 Hesai ETX LiDAR 做 L4 fleet-ready 參考平台。整條 pipeline 從 **點雲 → sparse conv → FP4 inference**，Nvidia 不會想在中間那段還依賴外部 lib。

## 三、把三家擺在一起（2026 Q4 快照）

| 維度 | spconv 2.3.8 | TorchSparse | **fVDB 0.6.0** |
|---|---|---|---|
| 最近 commit | 2024-12（~23 個月前） | 學術節奏 | **active** |
| 生態主導者 | 社群（作者停更） | MIT HAN Lab | **Nvidia Research** |
| Tensor Core 利用 | 低 | 中 | **原生設計** |
| PyTorch 整合 | CUDA extension | CUDA extension | **原生 `fvdb.functional`** |
| 新 CUDA 指令支援 | 不會接 | 不確定 | **會先在這** |
| 和 Hyperion 10 敘事對接 | 弱 | 弱 | **強（same stack）** |
| 履歷差異化 | 低 | 中 | **高（早期 adopter）** |

spconv 不是不能用，但它現在的角色是 **legacy baseline**，不是 default。

TorchSparse 的 position 比較尷尬—— 2026-07 fVDB 把它的 dependency 砍了，等於 Nvidia 在說「我們評估過，不需要你」。它還是會在學術 benchmark 出現，但工業採用的風向變了。

## 四、為什麼這對 2027 履歷重要

寫 LiDAR 3D 感知 capstone 的人很多，用 spconv 的更多。這條路徑在 2024 年是安全牌，2026 底就是**訊號噪音比太低**。

反過來，**用 fVDB 寫完一個端到端 KITTI / nuScenes 3D detection**，在 2026 Q4 幾乎沒人做，到 2027 Q1-Q2 面試季節會變成具體的差異化點。面試官問「你為什麼選這個 framework」，你可以回答：

> fVDB 0.5.0 砍掉 torchsparse 依賴那一刻，Nvidia 就已經告訴我方向了。我想做的是能跑在 Drive AGX Thor 上的 sparse conv pipeline，那就該長在 fVDB 上。

這是一段**有技術判斷、有時序邏輯、有產業對接**的敘事，比「因為 OpenPCDet 教學用 spconv」有力太多。

## 五、風險與反論

誠實標註：

**風險 1：fVDB API 還在動**。0.4 → 0.5 → 0.6 連著改，意味 0.6 → 0.7 可能再大改。如果你做的是正式產品，這是真風險。但做 capstone / 履歷作品，API churn 反而是學習素材—— 你能講出「我在 0.5 → 0.6 這次 convolution geometry 公式改動踩到什麼 bug」，那就是真的在一線。

**風險 2：社群資源少**。現成教學、paper 複現、Stack Overflow 回答都比 spconv 少一個數量級。要有心理準備自己啃原文件和 release notes。

**風險 3：Spira 可能來攪局**。MLSys 2026 有一篇 **Spira** 的 sparse conv engine paper（arXiv:2511.20834），直接 target autonomous driving + AR/VR。整個底層可能還會再洗一輪。但 Spira 是論文階段，fVDB 已經是可用的 PyTorch package，這個差距短期內不會變。

**反論**：「學 spconv 的人這麼多，共通知識比較好招人」—— 這是 **2023 的邏輯**。2026 底的招聘訊號看的是「你是否在跟著 Nvidia stack 遷移」，不是「你是否會用大家都會的 lib」。

## 六、實作建議（給 2027 要起新專案的人）

1. **第 1 週**：讀 fVDB 原論文（SIGGRAPH 2024, arXiv:2407.01781）+ 0.4 / 0.5 / 0.6 release notes。重點讀 index grid 設計、`GatherScatterDefault` backend、0.6.0 的 `PredGatherIGemm`。
2. **第 2 週**：clone `openvdb/fvdb`，跑完 examples/。KITTI example 應該會是最實際的切入點。
3. **第 3-4 週**：把一個經典 3D detector（CenterPoint / PV-RCNN 擇一）port 到 fVDB，和 spconv 版本做 side-by-side benchmark。
4. **第 5 週開始**：把這個 port 包成 repo + 一篇技術 blog（寫「port 過程遇到什麼 API 落差」最有價值）。
5. **敘事錨點**：所有文件和 commit message 都要能連回「這個 pipeline 的終點是 Drive AGX Thor 上的 FP4 inference」。這不是誇飾，是**選型理由**。

## 七、一句話收尾

> spconv 還值得學—— 當 legacy baseline。但 2027 的履歷要能打進 Nvidia，capstone 應該長在 fVDB 上。

---

## 延伸閱讀

- [上一篇：fVDB 0.6.0 把 sparse 3D learning 的拐杖砍乾淨了](fvdb-0-6-0-nvidia-sparse-3d-torchsparse-drop-hyperion10-spconv-replacement-2026.md)
- [fVDB 原論文（Nvidia Research）](https://research.nvidia.com/publication/2024-07_fvdb-deep-learning-framework-sparse-large-scale-and-high-performance-spatial)
- [fVDB Release Notes（0.5.0 砍 torchsparse）](https://fvdb-core.readthedocs.io/stable/release_notes.html)
- [Spira: Efficient Sparse Convolution in Point Cloud Networks（MLSys 2026, arXiv:2511.20834）](https://arxiv.org/pdf/2511.20834)
