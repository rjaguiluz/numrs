# NumRs JavaScript Bindings

High-performance Node.js bindings for NumRs with zero-copy Float32 operations powered by Intel MKL.

## ✨ Features

- **🚀 Zero-Copy Operations**: Direct Float32Array access with no conversions (5-7x faster than standard API)
- **⚡ Intel MKL Backend**: Optimized BLAS operations with multi-threading support
- **📊 25+ Operations**: Binary, unary, reduction, linear algebra, and shape operations
- **💾 Memory Efficient**: 50% memory reduction using Float32 instead of Float64
- **🔧 Type Safe**: Full TypeScript definitions included

## Performance

Benchmark results on 12th Gen Intel Core i9-12900HK:

| Operation Category | Average Throughput | Best Performance |
|-------------------|-------------------|------------------|
| Matrix Multiplication | 40.44 Gops/s | 114.13 Gops/s (512×512) |
| Reduction Operations | 513.54 Mops/s | 803.79 Mops/s (min) |
| Binary Operations | 164.59 Mops/s | 278 Mops/s (sub) |
| Unary Operations | 161.75 Mops/s | 577 Mops/s (neg) |

See [BENCHMARK_JS_12th_Gen_IntelR_CoreTM_i9-12900HK.md](./BENCHMARK_JS_12th_Gen_IntelR_CoreTM_i9-12900HK.md) for complete benchmark results.

## Installation

```bash
npm install
npm run build
```

## Quick Start

```javascript
const numrs = require('./index.node');

// Create Float32Arrays
const size = 10000;
const a = new Float32Array(size);
const b = new Float32Array(size);

// Initialize with random data
for (let i = 0; i < size; i++) {
  a[i] = Math.random();
  b[i] = Math.random();
}

// Zero-copy operations with clean API!
const result = numrs.add(a, [size], b, [size]);
console.log('Result:', result.slice(0, 5)); // Float32Array
```

## API Reference

All operations work directly with Float32Arrays - no type conversions!

### Binary Operations

Operations on two arrays of the same shape:

```javascript
// Syntax: operation(data1, shape1, data2, shape2) -> Float32Array
numrs.add(a, [rows, cols], b, [rows, cols]);
numrs.sub(a, [rows, cols], b, [rows, cols]);
numrs.mul(a, [rows, cols], b, [rows, cols]);
numrs.div(a, [rows, cols], b, [rows, cols]);
numrs.pow(a, [rows, cols], b, [rows, cols]);
```

### Unary Operations

Operations on a single array:

```javascript
// Syntax: operation(data, shape) -> Float32Array
numrs.sin(data, [size]);
numrs.cos(data, [size]);
numrs.tan(data, [size]);
numrs.exp(data, [size]);
numrs.log(data, [size]);
numrs.sqrt(data, [size]);
numrs.abs(data, [size]);
numrs.relu(data, [size]);
numrs.sigmoid(data, [size]);
numrs.tanh(data, [size]);
```

**Special case - Negation:**
```javascript
// Returns Array<number> instead of Float32Array
const result = numrs.neg(data, [size]); // Array<number>
const typedResult = new Float32Array(result);
```

### Reduction Operations

Operations that reduce an array to a single scalar:

```javascript
// Syntax: operation(data, shape) -> number
const total = numrs.sum(data, [rows, cols]);
const average = numrs.mean(data, [rows, cols]);
const maximum = numrs.max(data, [rows, cols]);
const minimum = numrs.min(data, [rows, cols]);
const variance = numrs.variance(data, [rows, cols]);
```

### Linear Algebra

```javascript
// Matrix multiplication: (M×K) @ (K×N) -> (M×N)
const result = numrs.matmul(
  matA, [M, K],  // First matrix
  matB, [K, N]   // Second matrix
); // Returns Float32Array of size M*N

// Dot product: returns scalar
const dotResult = numrs.dot(
  vecA, [size],
  vecB, [size]
); // Returns number
```

### Shape Operations

```javascript
// Transpose: (M×N) -> (N×M)
const transposed = numrs.transpose(data, [rows, cols]);

// Reshape: change dimensions without copying data
const reshaped = numrs.reshape(data, [oldShape], [newShape]);
```

## Complete Example

See [example.js](./example.js) for a comprehensive demo of all operations.

```javascript
const numrs = require('./index.node');

// Binary operations
const a = new Float32Array([1, 2, 3, 4]);
const b = new Float32Array([5, 6, 7, 8]);
const sum = numrs.add(a, [4], b, [4]);
console.log('Sum:', sum); // Float32Array [6, 8, 10, 12]

// Unary operations
const angles = new Float32Array([0, Math.PI/4, Math.PI/2]);
const sines = numrs.sin(angles, [3]);
console.log('Sine:', sines);

// Reductions
const data = new Float32Array([1, 2, 3, 4, 5]);
console.log('Sum:', numrs.sum(data, [5])); // 15
console.log('Mean:', numrs.mean(data, [5])); // 3

// Matrix multiplication
const matA = new Float32Array([1, 2, 3, 4]); // 2×2
const matB = new Float32Array([5, 6, 7, 8]); // 2×2
const product = numrs.matmul(matA, [2, 2], matB, [2, 2]);
console.log('Matrix product:', product); // Float32Array [19, 22, 43, 50]
```

## TypeScript Support

Full TypeScript definitions are available:

```typescript
import type { Float32Array } from './index';

declare function add(
  data1: Float32Array,
  shape1: number[],
  data2: Float32Array,
  shape2: number[]
): Float32Array;

// ... all other operations
```

## Benchmarking

Run the comprehensive benchmark suite:

```bash
node benchmark_gen.js
```

This generates a detailed markdown report with:
- System configuration
- Performance metrics for all 77 operation/size combinations
- Throughput statistics (Mops/s and Gops/s)
- Operation categories and summaries

## Performance Tips

1. **Use Float32Arrays directly** - Avoid conversions from/to regular JavaScript arrays
2. **Batch operations** - Process larger arrays to amortize overhead
3. **Reuse arrays** - Create Float32Arrays once and reuse them
4. **Matrix sizes** - MKL performs best with matrices ≥256×256

## Legacy API

The original JsArray-based API has been deprecated in favor of the zero-copy Float32Array API:

```javascript
// Old (deprecated) - 5-7x slower
const a = numrs.array([2, 2], [1, 2, 3, 4], 'float32');
const b = numrs.array([2, 2], [5, 6, 7, 8], 'float32');
const result = numrs.add(a, b);

// New (recommended) - Zero-copy, ultra fast!
const aData = new Float32Array([1, 2, 3, 4]);
const bData = new Float32Array([5, 6, 7, 8]);
const result = numrs.add(aData, [2, 2], bData, [2, 2]);
```

## Building from Source

```bash
# Debug build
cargo build

# Release build with optimizations
npm run build

# CPU-only (no WebGPU)
npm run build:cpu
```

## License

See [LICENSE](../LICENSE) for details.

## See Also

- [Complete Benchmark Results](./BENCHMARK_JS_12th_Gen_IntelR_CoreTM_i9-12900HK.md)
- [Main NumRs Documentation](../README.md)
- [TypeScript Definitions](./index.d.ts)
