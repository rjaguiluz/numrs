# @numrs/node

**NumRs Node** is the native Node.js binding for the [NumRs](https://github.com/rjaguiluz/numrs) numerical engine. It provides high-performance, hardware-accelerated (Metal/Accelerate on macOS, MKL/OpenBLAS elsewhere) tensor operations and deep learning capabilities.

## 🚀 Features

- **Hardware Acceleration**: Uses Metal on Apple Silicon for GPU-like performance.
- **Multithreading**: Native parallelization via Rayon.
- **Zero-Copy**: Efficient memory sharing between Node.js and Rust.
- **Deep Learning**: Full Autograd engine and Neural Network modules.
- **Type Safe**: Full TypeScript support.

## 📦 Installation

```bash
npm install @numrs/node
```

> **Note**: Requires Rust to be installed on your system to compile the native extension.
> ```bash
> curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
> ```

## 🎯 Usage

### Quick Start: Tensors & Autograd

```typescript
import { Tensor, nn } from '@numrs/node';

// Create tensors with autograd enabled
const x = Tensor.randn([2, 5], true); 
const w = Tensor.randn([5, 10], true);

// Fast matrix multiplication
const y = x.matmul(w);

// Backward pass
y.backward();
console.log("Gradient:", x.grad());
```

### Deep Learning Model

```typescript
import { Sequential, nn } from '@numrs/node';

const model = new Sequential();
model.add_linear(new nn.Linear(10, 32));
model.add_relu(new nn.ReLU());
model.add_linear(new nn.Linear(32, 2));

// Forward pass
const input = Tensor.randn([100, 10]);
const output = model.forward(input);
```

### Training Loop

Use the `TrainerBuilder` for a highly optimized, native training loop.

```typescript
import { TrainerBuilder, Dataset } from '@numrs/node';

// 1. Prepare Data
const input = Tensor.randn([100, 10]);
const target = Tensor.randn([100, 2]);
const dataset = new Dataset(input, target);

// 2. Train
const trainer = new TrainerBuilder(model, dataset)
    .batch_size(16)
    .learning_rate(0.01)
    .optimizer("adam") // Supports: adam, sgd, rmsprop, etc.
    .loss("mse")       // Supports: mse, cross_entropy
    .max_epochs(50)
    .build();

const history = trainer.fit();
console.log("Final Loss:", history.final_loss);
```

## 🧠 Optimizers & Loss Functions

| Category           | Supported Algorithms                                                                                                                     |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **Optimizers**     | `"sgd"`, `"adam"`, `"adamw"`, `"nadam"`, `"radam"`, `"rmsprop"`, `"adagrad"`, `"adadelta"`, `"lamb"`, `"adabound"`, `"lbfgs"`, `"rprop"` |
| **Loss Functions** | `"mse"` (Regression), `"cross_entropy"` (Classification)                                                                                 |

## 🔍 Advanced Features

### Zero-Copy Float32Array Ops
For raw numerical computing, you can operate directly on `Float32Array` buffers without Tensor overhead.

```javascript
const numrs = require('@numrs/node');
const a = new Float32Array([1, 2, 3, 4]);
const b = new Float32Array([5, 6, 7, 8]);
// Element-wise addition
const c = numrs.add(a, [2, 2], b, [2, 2]); 
```

### ONNX Export
Export your trained models to ONNX for verifying graphs or interoperability.
```javascript
model.save_onnx(dummy_input, "model.onnx");
```

## 📚 Documentation

For full API documentation, please refer to the main [NumRs Repository](https://github.com/rjaguiluz/numrs).

## License

AGPL-3.0-only
