# NumRs WASM Documentation

**high-performance numerical computation and deep learning engine**

## 1. Introduction
The **NumRs WASM Binding** brings professional AI training and inference to the browser. It supports WebGPU acceleration and runs entirely client-side, ensuring data privacy and low latency.

---

## 2. Installation

```bash
npm install @numrs/wasm


---

## 3. Quick Start

```javascript
import init, { NumRsArray, Tensor } from '@numrs/wasm';
await init();

// 1. Array
const data = new Float32Array([1, 2, 3, 4]);
const arr = new NumRsArray(data, new Uint32Array([2, 2]));

// 2. Tensor (Autograd)
const t = new Tensor(arr, true);

// 3. Op
const res = t.add(t);
console.log(res.data().to_string());
```

---

## 4. Core API

### Arrays & Tensors
*   **WebGPU**: Call `init_webgpu()` to enable hardware acceleration.
*   **Shapes**: Must use `Uint32Array` for shape arguments.
*   **Sync/Async**: Most ops are synchronous; training can be chunked.

---

## 5. Deep Learning

### Model Building
Use `Sequential` as the container.

```javascript
import { Sequential, Linear, ReLU } from '@numrs/wasm';

const model = new Sequential();
model.add_linear(new Linear(10, 32));
model.add_relu(new ReLU());
model.add_linear(new Linear(32, 1));
```

### Training Loop
Use `Trainer` for client-side training.

```javascript
import { Trainer } from '@numrs/wasm';

const trainer = new Trainer(model, "adam", "mse", 0.01);
trainer.fit(xTrain, yTrain, 10);
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
*   **Export**: `OnnxModelWrapper.export_model_to_json(...)` (JSON format for web).
*   **Inference**: `OnnxModelWrapper.load_from_json(...)`.

---

## 8. Detailed Reference
See the `docs/` folder for in-depth guides:
*   [Ops Reference](docs/00_numrs_ops.md)
*   [Arrays Guide](docs/01_numrs_arrays.md)
*   [Deep Learning](docs/02_numrs_tensors_dl.md)
*   [Optimizers](docs/03_1_numrs_optimizers.md)
