# specialized backend module

This directory contains the execution engines for `numrs`.

## Contents

- **`dispatch.rs`**: The core of the zero-cost dispatch system. It manages the `DispatchTable` and `OnceCell` initialization to direct operations to the best available backend at runtime with minimal overhead (< 1ns).
- **`blas/`**: Integration with BLAS libraries (MKL, BLIS, Accelerate). Used primarily for matrix multiplication and reductions.
- **`cpu/`**: Contains pure Rust implementations, including both scalar fallbacks and SIMD-optimized kernels (AVX2/NEON) for element-wise operations.
- **`webgpu/`**: (Experimental) Backend for GPU execution using `wgpu`.
- **`cuda/`** & **`metal/`**: Placeholder directories for future native GPU backends.
- **`capabilities.rs`**: Logic to probe the host system's capabilities (e.g., "Has AVX2?", "Is MKL available?").

## Design Rationale

NumRs is designed to be **backend-agnostic** but **performance-critical**.

- **Dispatch System**: Instead of deciding how to run an operation every time it's called (expensive), we verify the system capabilities *once* at startup and populate a table of function pointers. This allows the hot-path to be a simple indirect function call.
- **Static Linking**: The BLAS backend is designed to be statically linked to ensure the resulting binary is portable and "just works" without complex environment setup.

## Interaction

1.  **Usage**: The `src/ops` module does not call `blas` or `cpu` functions directly. Instead, it calls `get_dispatch_table()`, which returns the pointers to the optimal implementation.
2.  **Selection**: At startup, `heuristics.rs` and `capabilities.rs` run micro-benchmarks or checks to decide which backend to put in the dispatch table (e.g., prefer MKL for dgemm, prefer SIMD for f32 add).
3.  **Data**: Backends receive `Array<T>` references. They are responsible for executing the math efficiently on the underlying data slice.
