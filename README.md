# numrs — core library

**NumRs** is an experimental, high-performance numerical & Deep Learning Engine framework for Rust, inspired by NumPy and PyTorch.

This project is structured as a collection of specialized modules, each with its own specific responsibility. This README serves as a high-level index to the detailed documentation for each component.

## Architecture Overview

NumRs is built on two main layers:
1.  **Numerical Engine**: Handles arrays, types, and raw execution (SIMD/BLAS).
2.  **Machine Learning Framework**: Built on top of the engine, providing autograd and neural networks.

```mermaid
graph TD
    NumRs[NumRs Core]
    
    subgraph Engine["Numerical Engine"]
        Array[Link: src/array]
        Ops[Link: src/ops]
        Backend[Link: src/backend]
    end
    
    subgraph ML["ML Framework"]
        Autograd[Link: src/autograd]
        NN[NN Modules]
    end
    
    NumRs --> Engine
    NumRs --> ML
    Autograd --> Ops
    Ops --> Backend
    Ops --> Array
    Backend --> Array
```

## 📚 Ecosystem Documentation

Select your preferred language to view the specific documentation:

| Component      | Language          | Documentation                                 |
| :------------- | :---------------- | :-------------------------------------------- |
| **NumRs Core** | 🦀 **Rust**        | [View Rust Docs](numrs-core/DOCS.md)          |
| **NumRs C**    | 🇨 **C / C++**     | [View C ABI Docs](numrs-c/DOCS.md)            |
| **NumRs Node** | 🟢 **Node.js**     | [View JS Docs](bindings/numrs-js/DOCS.md)     |
| **NumRs Wasm** | 🕸️ **WebAssembly** | [View Wasm Docs](bindings/numrs-wasm/DOCS.md) |
| **NumRs Py**   | 🐍 **Python**      | [View Python Docs](bindings/numrs-py/DOCS.md) |
| **NumRs R**    | 📐 **R**           | [View R Docs](bindings/numrs-r/DOCS.md)       |

---

## Module Documentation (Internal internals)

Detailed architecture documentation for `numrs-core` developers:

### 📦 [src/array](src/array/README.md)
**The Data Layer**. Defines `Array<T>`, `DynArray` (dynamic typing), and the Type Promotion system.

### 🧮 [src/ops](src/ops/README.md)
**The User API**. Contains the definitions for all mathematical operations (`add`, `matmul`, `sum`, etc.).

### ⚙️ [src/backend](src/backend/README.md)
**The Execution Engine**. Manages the **Zero-Cost Dispatch System** and interfaces with hardware accelerators (MKL, BLIS, Accelerate, SIMD).

### 🧠 [src/autograd](src/autograd/README.md)
**The ML Engine**. Implements `Tensor` for Reverse Mode Automatic Differentiation, Neural Network layers, and Optimizers.

## Quick Start

```bash
# Build with auto-detected optimizations (ASICS/BLAS)
cargo build --release
```

For detailed examples, see the `examples/` directory.

## License
AGPL-3.0-only
