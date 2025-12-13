# specialized cpu backend

This directory contains pure Rust basic implementations, including both scalar and vector (SIMD) kernels.

## Contents

- **`mod.rs`**: Entry point for CPU operations.
- **`scalar.rs`**: Fallback implementations using standard Rust loops. Guaranteed to compile on any architecture.
- **`simd.rs`**: Single Instruction Multiple Data (SIMD) kernels. Uses Rust's specific intrinsics (like AVX2 on x86, NEON on ARM) to process multiple data points per cycle.
- **`parallel.rs`**: Multi-threading logic using `rayon`. It chunks large arrays and processes them in parallel across available CPU cores.

## Design Rationale

- **Universality**: Not all platforms have MKL or BLAS available. The Scalar backend ensures `numrs` runs everywhere Rust runs.
- **Performance**: SIMD is critical for element-wise operations (like `add`, `sin`, `exp`) where the overhead of calling external BLAS libraries might outperform the gain.
- **Parallelism**: For very large reductions or element-wise ops, multi-threading provides significant speedups.

## Interaction

1.  **Dispatch**: The `dispatch` module might select `simd::add` over `scalar::add` if CPU features are detected.
2.  **Validation**: At startup, `simd.rs` functions are tested (via `std::panic::catch_unwind` or check flags) to ensure they don't crash on the host CPU.
