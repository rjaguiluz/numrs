# linear algebra operations module

This directory contains operations for linear algebra, such as matrix multiplication.

## Contents

- **`matmul.rs`**: General Matrix Multiplication (GEMM). This is the heavy lifter of deep learning.
- **`dot.rs`**: Dot product for vectors.

## Design Rationale

- **Performance Critical**: `matmul` is typically the most expensive operation in neural networks.
- **Dispatch**: We heavily prioritize dispatching this to BLAS (MKL/BLIS) if available.
- **Batched Support**: Future versions aim to support batched matmul for higher throughput.

## Interaction

1.  **Call**: `numrs::ops::matmul(&A, &B)`.
2.  **Dispatch**:
    - **BLAS**: If available, calls `cblas_dgemm` / `cblas_sgemm`.
    - **SIMD**: If no BLAS, uses a tiled/blocked implementation in Rust with AVX2.
    - **Scalar**: Fallback (very slow, O(n³)).
