# 03. Time Series Forecasting (Rust)

## 1D Convolution
Use `Conv1d` for sequence data.

```rust
use numrs::nn::{Conv1d, ReLU, Flatten, Linear};

let mut model = Sequential::new();

// Input: [Batch, 1, Length]
// Conv1d: in=1, out=16, k=3, s=1, p=1
model.add(Box::new(Conv1d::new(1, 16, 3, 1, 1)?));
model.add(Box::new(ReLU::new()));
model.add(Box::new(Flatten::new(1, -1)));
model.add(Box::new(Linear::new(16 * len, 1)?));
```

## Data Input
Construct 3D arrays.
```rust
let arr = Array::new(vec![batch_size, 1, seq_len], raw_data);
let t = Tensor::new(arr, true);
```
