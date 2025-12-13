# NumRs Node.js Documentation

**high-performance numerical compution and deep learning engine**

## 1. Introduction
The **NumRs Node Binding** provides native performance for AI tasks in JavaScript. Powered by N-API, it minimizes overhead and enables true Zero-Copy data sharing between JS and Rust.

---

## 2. Installation

```bash
npm install numrs-node
```

---

## 3. Quick Start

```javascript
const { Tensor, NumRsArray } = require('numrs-node');

// 1. Array
const data = new Float32Array([1, 2, 3, 4]);
const arr = new NumRsArray(data, [2, 2]);

// 2. Tensor (Autograd)
const t = new Tensor(arr, [2, 2], true);

// 3. Op
const res = t.add(t);
console.log(res.data.data);
```

---

## 4. Core API

### Arrays & Tensors
*   **Zero-Copy**: `NumRsArray` wraps `Float32Array` directly.
*   **Memory**: Managed by JS Garbage Collector (but holds Rust resources).
*   **Autograd**: Access `.grad` property on leaf tensors after `.backward()`.

---

## 5. Deep Learning

### Model Building
Use `Sequential` as the container.

```javascript
const { Sequential, nn } = require('numrs-node');

const model = new Sequential();
model.addLinear(new nn.Linear(10, 32));
model.addReLU(new nn.ReLU());
model.addLinear(new nn.Linear(32, 1));
```

### Training Loop
Use `Trainer` for native-speed loops.

```javascript
const { TrainerBuilder } = require('numrs-node');

const builder = new TrainerBuilder(model);
builder.learningRate(0.01);

const trainer = builder.build("adam", "mse");
trainer.fit(dataset, dataset, 10);
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
Use `nn.Conv1d` for sequence processing.
*   **Input**: `[Batch, Channels, Length]`.
*   **Layers**: `Conv1d` -> `ReLU` -> `Flatten` -> `Linear`.

### ONNX
*   **Export**: `model.saveOnnx(dummyInput, "model.onnx")`
*   **Inference**: `Sequential.loadOnnx("model.onnx")`

---

## 8. Detailed Reference
See the `docs/` folder for in-depth guides:
*   [Ops Reference](docs/00_numrs_ops.md)
*   [Arrays Guide](docs/01_numrs_arrays.md)
*   [Deep Learning](docs/02_numrs_tensors_dl.md)
*   [Optimizers](docs/03_1_numrs_optimizers.md)
