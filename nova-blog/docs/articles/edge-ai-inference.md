# 邊緣 AI 推理優化：從理論到實踐

## 為什麼要優化邊緣推理？

在嵌入式系統或邊緣設備上跑 AI 模型，硬體資源有限。我們需要在延遲、功耗、準確度之間找到平衡。

## 常見的優化技術

### 1. 模型量化（Quantization）

將浮點權重轉換為低精度整數：

- **INT8 量化**：權重從 32-bit float → 8-bit integer
- **量化感知訓練（QAT）**：在訓練時就考慮量化誤差
- **Post-training quantization（PTQ）**：訓練完後直接量化

```python
import torch

# 範例：動態量化
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 2. 模型剪枝（Pruning）

移除不重要的權重或神經元：

- **結構化剪枝**：移除整個 channel 或 filter
- **非結構化剪枝**：只移除個別權重

### 3. 知識蒸餾（Knowledge Distillation）

用大模型教導小模型，保留核心能力。

## 實務建議

1. **先 profiling 再優化** — 用工具確認瓶頸在哪
2. **量化優先** — INT8 通常能得到 2-4x 加速
3. **注意硬體支援** — 有些硬體對 INT8 有特殊加速

## 總結

邊緣 AI 優化不是一蹴可幾，需要反覆測試和調整。從量化入手是最快的方法。

---

*下次再談談如何在 RPI 上實際部署優化後的模型。*