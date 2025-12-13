# NumRs Core Documentation

**high-performance numerical compution and deep learning engine**

## 1. Introduction
**numrs-core** is the heart of the ecosystem. It is*   **Ops**: Supports standard operator overloading (`&a + &b`, `&a * &b`) for references, plus method chaining (`a.add(&b)`)., Autograd (Define-by-Run), and high-level Deep Learning primitives. It is designed to be the "Numpy + PyTorch" of Rust.

---

## 2. Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
numrs = "0.1"
```

---

## 3. Quick Start

```rust
use numrs::{Array, Tensor};

fn main() {
    // 1. Array
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let arr = Array::new(vec![2, 2], data);

    // 2. Tensor (Autograd)
    let t = Tensor::new(arr, true); // requires_grad = true

    // 3. Op
    let res = &t + &t; // TensorRef arithmetic
    
    // 4. Backward
    res.sum(None).backward();
    println!("Grad: {:?}", t.grad().unwrap());
}
```

---

## 4. Core API

### Arrays & Tensors
*   `Array<T>`: The fundamental data structure (Row-major, contiguous).
*   `Tensor`: Wraps an `Array` with Autograd history and Gradient storage.

---

## 5. Deep Learning

### Model Building
Use `Sequential` for stacking layers.

```rust
use numrs::nn::{Linear, ReLU, Sequential, Module};

let model = Sequential::new(vec![
    Box::new(Linear::new(10, 32).unwrap()),
    Box::new(ReLU::new()),
    Box::new(Linear::new(32, 1).unwrap())
]);
```

### Training Loop
Use `Trainer` for managed optimization.

```rust
use numrs::autograd::train::{Trainer, TrainerBuilder, MSELoss};

let builder = TrainerBuilder::new(model)
    .learning_rate(0.01);

let mut trainer = builder.build_adam(Box::new(MSELoss));
trainer.fit(&dataset, None, 10, true);
```

---

## 6. Optimizers & Loss

### Supported Optimizers
| Struct    | Description                           |
| :-------- | :------------------------------------ |
| `SGD`     | Stochastic Gradient Descent           |
| `Adam`    | Adaptive Moment Estimation            |
| `AdamW`   | Adam with Weight Decay                |
| `RMSprop` | Root Mean Square Propagation          |
| `LBFGS`   | Limited-memory BFGS                   |
| `LAMB`    | Layer-wise Adaptive Moments           |
| ...       | (See `docs/03_1_numrs_optimizers.md`) |

### Supported Losses
*   `MSELoss` (Regression)
*   `CrossEntropyLoss` (Classification)

---

## 7. Advanced Topics

### Time Series (1D CNN)
Use `Conv1d` for sequence processing.
*   `numrs::nn::Conv1d`
*   Input shape: `[Batch, Channels, Length]`

### ONNX
*   **Export**: `numrs::ops::save_onnx(&model, &dummy_input, "path.onnx")`
*   **Inference**: `numrs::ops::load_onnx("path.onnx")` (Feature gated).

---

## 8. Detailed Reference
See the `docs/` folder for in-depth guides:
*   [Ops Reference](docs/00_numrs_ops.md)
*   [Arrays Guide](docs/01_numrs_arrays.md)
*   [Deep Learning](docs/02_numrs_tensors_dl.md)
*   [Optimizers](docs/03_1_numrs_optimizers.md)
