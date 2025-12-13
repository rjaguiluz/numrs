# 03. Time Series Forecasting (Python)

## 1D CNN for Sequence Data

```python
import numrs
from numrs.nn import Conv1d, ReLU, Linear, Flatten, Sequential

model = Sequential()
# Input: (Batch, 1, 100)
model.add(Conv1d(1, 16, kernel_size=3, stride=1, padding=1))
model.add(ReLU())
model.add(Flatten(start_dim=1))
model.add(Linear(16 * 100, 1))
```

## Data Input
Requires 3D input tensor.
```python
x = numrs.Tensor(data_3d, requires_grad=True)
# shape: (N, 1, L)
```
