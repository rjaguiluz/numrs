# NumRs R Documentation

**high-performance numerical computation and deep learning engine**

## 1. Introduction
The **NumRs R Binding** brings systems-level deep learning capabilities to R. It interfaces via the C-ABI to offer high-speed tensor operations and model training within the R statistical environment.

---

## 2. Installation

```bash
R CMD INSTALL .
```

---

## 3. Quick Start

```r
library(numrs)

# 1. Array (Note: Transpose for Row-Major)
m <- matrix(c(1,2,3,4), 2, 2)
arr <- nr_array(as.numeric(t(m)), c(2,2))

# 2. Tensor
t <- nr_tensor(arr)

# 3. Ops
res <- nr_add(t, t)
print(as.numeric(res))
```

---

## 4. Core API

### Arrays & Tensors
*   **Layout**: R is Col-Major; NumRs is Row-Major. Use `t()` before flat conversion.
*   **S3 Classes**: `NumRsTensor`, `NumRsArray`.
*   **Dispatch**: Standard operators (`+`, `-`, `*`) overloaded.

---

## 5. Deep Learning

### Model Building
Use `nr_sequential` as the container.

```r
model <- nr_sequential(
    nr_linear(10, 32),
    nr_relu_layer(),
    nr_linear(32, 1)
)
```

### Training Loop
Use `nr_trainer` builder pattern.

```r
builder <- nr_trainer_builder(model)
builder <- nr_with_lr(builder, 0.01)

trainer <- nr_build(builder, "adam", "mse")
nr_fit(trainer, dataset, epochs=10)
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
Use `nr_conv1d` for sequence processing.
*   **Input**: `[Batch, Channels, Length]`.
*   **Layers**: `Conv1d` -> `ReLU` -> `Flatten` -> `Linear`.

### ONNX
*   **Export**: `nr_save_onnx(model, dummy, "path.onnx")`
*   **Inference**: `nr_load_onnx("path.onnx")`

---

## 8. Detailed Reference
See the `docs/` folder for in-depth guides:
*   [Ops Reference](docs/00_numrs_ops.md)
*   [Arrays Guide](docs/01_numrs_arrays.md)
*   [Deep Learning](docs/02_numrs_tensors_dl.md)
*   [Optimizers](docs/03_1_numrs_optimizers.md)
