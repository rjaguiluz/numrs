# NumRs Python Documentation

**high-performance numerical compution and deep learning engine**

## 1. Introduction
The **NumRs Python Binding** offers a PyTorch-like API with the performance and safety of Rust. It serves as a drop-in replacement for research and production workloads requiring strict typing and speed.

---

## 2. Installation

```bash
pip install .
```

---

## 3. Quick Start

```python
import numrs
from numrs import Tensor

# 1. Array/Tensor
t = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# 2. Ops
res = t + t

# 3. Autograd
res.sum().backward()
print(t.grad)
```

---

## 4. Core API

### Arrays & Tensors
*   **Interface**: Mimics PyTorch (`.shape`, `.T`, `.item()`).
*   **Type Safety**: Strict checking on dtype (Float32 default).
*   **Interoperability**: `.numpy()` converts to NumPy array.

---

## 5. Deep Learning

### Model Building
Use `Sequential` as the container.

```python
from numrs import Sequential
from numrs.nn import Linear, ReLU

model = Sequential()
model.add(Linear(10, 32))
model.add(ReLU())
model.add(Linear(32, 1))
```

### Training Loop
Use `Trainer` for high-performance loops.

```python
trainer = numrs.Trainer(model, optimizer="adam", loss="mse", lr=0.01)
trainer.fit(dataset, epochs=10)
```

---

## 6. Optimizers & Loss

### Supported Optimizers
| Key          | Algorithm                   |
| :----------- | :-------------------------- |
| `"sgd"`      | Stochastic Gradient Descent |
| `"adam"`     | Adam                        |
| `"adamw"`    | AdamW                       |
| `"nadam"`    | NAdam                       |
| `"radam"`    | RAdam                       |
| `"rmsprop"`  | RMSProp                     |
| `"adagrad"`  | AdaGrad                     |
| `"adadelta"` | AdaDelta                    |
| `"lamb"`     | LAMB                        |
| `"adabound"` | AdaBound                    |
| `"lbfgs"`    | L-BFGS                      |
| `"rprop"`    | Rprop                       |

### Supported Losses
*   `"mse"` (Regression)
*   `"cross_entropy"` (Classification)

---

## 7. Advanced Topics

### Time Series (1D CNN)
Use `Conv1d` for sequence processing.
*   **Input**: `[Batch, Channels, Length]`.
*   **Layers**: `Conv1d` -> `ReLU` -> `Flatten` -> `Linear`.

### ONNX
*   **Export**: `numrs.save_onnx(model, dummy, "path.onnx")`
*   **Inference**: `model = numrs.load_onnx("path.onnx")`

---

## 8. Detailed Reference
See the `docs/` folder for in-depth guides:
*   [Ops Reference](docs/00_numrs_ops.md)
*   [Arrays Guide](docs/01_numrs_arrays.md)
*   [Deep Learning](docs/02_numrs_tensors_dl.md)
*   [Optimizers](docs/03_1_numrs_optimizers.md)
