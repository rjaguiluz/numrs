# @numrs/node

**NumRs Node** is the native Node.js binding for the [NumRs](https://github.com/rjaguiluz/numrs) numerical engine. It provides high-performance, hardware-accelerated (Metal/Accelerate on macOS, MKL/OpenBLAS elsewhere) tensor operations.

## 🚀 Features

- **Hardware Acceleration**: Uses Metal on Apple Silicon for GPU-like performance.
- **Multithreading**: Native parallelization via Rayon.
- **Zero-Copy**: Efficient memory sharing between Node.js and Rust.
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

### Quick Start

```typescript
import { Tensor, nn, TrainerBuilder, Dataset } from '@numrs/node';

// Create tensors
const a = Tensor.randn([100, 100]);
const b = Tensor.randn([100, 100]);

// Fast operations
const c = a.matmul(b);

console.log("Result shape:", c.shape());
```

### Training Example

```typescript
import { Tensor, nn, TrainerBuilder, Dataset } from '@numrs/node';

// 1. Prepare Data
const input = Tensor.randn([100, 10]);
const target = Tensor.randn([100, 2]);
const dataset = new Dataset(input, target);

// 2. Define Model
const model = new nn.Sequential();
model.add_linear(new nn.Linear(10, 32));
model.add_relu(new nn.ReLU());
model.add_linear(new nn.Linear(32, 2));

// 3. Train
const trainer = new TrainerBuilder(model, dataset)
    .batch_size(16)
    .learning_rate(0.01)
    .max_epochs(50)
    .build();

const history = trainer.fit();
console.log("Final Loss:", history.final_loss);
```

## 📚 Documentation

For full API documentation, please refer to the main [NumRs Repository](https://github.com/rjaguiluz/numrs).

## License

AGPL-3.0-only
