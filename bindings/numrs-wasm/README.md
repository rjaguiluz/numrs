# NumRs WebAssembly Bindings

High-performance numerical computing for JavaScript via WebAssembly.

## 🚀 Features

- **Zero FFI overhead**: Direct WASM calls, no Node.js native addon layer
- **SIMD acceleration**: Uses WebAssembly SIMD when available
- **Works everywhere**: Browser, Node.js, Deno, Bun
- **TypeScript support**: Auto-generated TypeScript definitions
- **No native dependencies**: Pure WASM, no compilation needed

## 📦 Installation

```bash
npm install @numrs/wasm
```

## 🎯 Usage

### Node.js

```javascript
const numrs = require('@numrs/wasm');

// Create arrays (using Float32Array for best performance)
const a = new Float32Array([1, 2, 3, 4]);
const b = new Float32Array([5, 6, 7, 8]);

// Element-wise operations
const result = numrs.add_f32(a, [4], b, [4]);
console.log(result); // [6, 8, 10, 12]

// Matrix multiplication
const m1 = new Float32Array([1, 2, 3, 4]);
const m2 = new Float32Array([5, 6, 7, 8]);
const product = numrs.matmul_f32(m1, [2, 2], m2, [2, 2]);
```

### Browser (ES Modules)

```javascript
import init, * as numrs from '@numrs/wasm/pkg-web/numrs_wasm.js';

await init(); // Initialize WASM module

const a = new Float32Array([1, 2, 3, 4]);
const result = numrs.add_f32(a, [4], a, [4]);
```

## 🔬 API

All functions work with `Float32Array` for optimal performance:

### Binary Operations
- `add_f32(a, shape_a, b, shape_b)` - Element-wise addition
- `sub_f32(a, shape_a, b, shape_b)` - Element-wise subtraction
- `mul_f32(a, shape_a, b, shape_b)` - Element-wise multiplication
- `div_f32(a, shape_a, b, shape_b)` - Element-wise division
- `matmul_f32(a, shape_a, b, shape_b)` - Matrix multiplication

### Unary Operations
- `sin_f32(data, shape)` - Sine
- `cos_f32(data, shape)` - Cosine
- `exp_f32(data, shape)` - Exponential
- `sqrt_f32(data, shape)` - Square root

### Reductions
- `sum_f32(data, shape)` - Sum all elements
- `mean_f32(data, shape)` - Mean of all elements

### Backend Info
- `startup_log()` - Print backend information
- `backend_info()` - Get backend details as JSON string

## ⚡ Performance

WASM bindings have **zero FFI overhead** compared to native Node.js addons. However, they don't have access to optimized BLAS libraries like MKL.

**Best for:**
- Small to medium-sized operations (<10K elements)
- Browser-based ML/data viz
- Cross-platform deployment
- Serverless/edge computing

**Use native addons for:**
- Large matrix operations (>100K elements)
- Heavy linear algebra with BLAS/LAPACK
- Maximum CPU performance

## 🏗️ Building from Source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for Node.js
npm run build

# Build for web
npm run build:web

# Build all targets
npm run build:all
```

## 📊 Benchmarks

Run benchmarks:
```bash
npm run bench
```

See `BENCHMARK_WASM.md` for detailed results.

## 🤝 Comparison with Native Bindings

| Feature | WASM (`@numrs/wasm`) | Native (`@numrs/native`) |
|---------|---------------------|--------------------------|
| FFI Overhead | None ✅ | ~30μs per call |
| BLAS/MKL | ❌ | ✅ |
| Browser Support | ✅ | ❌ |
| Installation | npm only | Requires compilation |
| Small ops (<1K) | Faster | Slower (FFI) |
| Large ops (>100K) | Slower | Faster (MKL) |

## 📄 License

MIT
