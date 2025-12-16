# NumRs C Documentation

**high-performance numerical computation and deep learning engine**

## 1. Introduction
The **NumRs C ABI** is the foundational layer of the ecosystem. It exposes the Rust core to any language capable of FFI, providing a stable, zero-cost interface for high-performance computing.

---

## 2. Installation

Link against the shared library manually or via CMake/Make.

```bash
# Compilation Example
gcc main.c -I./include -L./target/release -lnumrs_c -o app
```

---

## 3. Quick Start

```c
#include "numrs.h"

int main() {
    numrs_print_startup_log();

    // 1. Array
    float data[] = {1.0, 2.0, 3.0, 4.0};
    uint32_t shape[] = {2, 2};
    NumRsArray *arr = numrs_array_new(data, shape, 2);

    // 2. Tensor (Autograd)
    NumRsTensor *t = numrs_tensor_new(arr, true);

    // 3. Op
    NumRsTensor *res = numrs_add(t, t);
    
    // 4. Cleanup
    numrs_tensor_free(t);
    numrs_tensor_free(res);
    numrs_array_free(arr);
    return 0;
}
```

---

## 4. Core API

### Arrays & Tensors
*   **Opaque Pointers**: All types (`NumRsArray`, `NumRsTensor`) are opaque.
*   **Memory**: You must manually call `*_free` for every `*_new` or operation result.
*   **Autograd**: Pass `true` to `numrs_tensor_new` to enable gradient tracking.

---

## 5. Deep Learning

### Model Building
Use `NumRsSequential` as the container.

```c
NumRsSequential *model = numrs_sequential_new();
numrs_sequential_add_linear(model, numrs_linear_new(10, 32));
numrs_sequential_add_relu(model, numrs_relu_layer_new());
numrs_sequential_add_linear(model, numrs_linear_new(32, 1));
```

### Training Loop
Use `NumRsTrainer` for managed training loops.

```c
NumRsTrainerBuilder *builder = numrs_trainer_builder_new(model);
numrs_trainer_builder_learning_rate(builder, 0.01);

NumRsTrainer *trainer = numrs_trainer_build(builder, "adam", "mse");
numrs_trainer_fit(trainer, dataset, 10);
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
Use `NumRsConv1d` for sequence processing.
*   **Input**: `[Batch, Channels, Length]`.
*   **Layers**: `Conv1d` -> `ReLU` -> `Flatten` -> `Linear`.

### ONNX
*   **Export/Import**: Currently experimental or via host language bindings.

---

## 8. Detailed Reference
See the `docs/` folder for in-depth guides:
*   [Ops Reference](docs/00_numrs_ops.md)
*   [Arrays Guide](docs/01_numrs_arrays.md)
*   [Deep Learning](docs/02_numrs_tensors_dl.md)
*   [Optimizers](docs/03_1_numrs_optimizers.md)
