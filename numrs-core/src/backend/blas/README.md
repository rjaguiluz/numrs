# specialized blas backend

This directory contains the integration with Basic Linear Algebra Subprograms (BLAS).

## Contents

- **`mod.rs`**: Implements the `BlasBackend` struct. This file contains the logic to call BLAS functions (like `cblas_dgemm`, `cblas_sasum`) via FFI (Foreign Function Interface). It uses `cblas-sys` or similar bindings depending on the architecture.

## Design Rationale

BLAS is the industry standard for dense linear algebra.

- **Static Linking**: Crucially, NumRs links these libraries *statically*. This means the `mod.rs` here relies on build-time configuration (in `build.rs` and `Cargo.toml`) to ensure symbols like `cblas_dgemm` are available in the final binary without needing external DLLs.
- **Performance**: For large matrices (>500x500), BLAS implementations (MKL, BLIS) are orders of magnitude faster than naive loops.

## Interaction

1.  **Dispatch**: The `dispatch.rs` system detects if BLAS is available.
2.  **Call**: If selected, operations like `matmul` call functions in this module.
3.  **Data**: This module takes raw pointers from `Array<T>` and passes them to C-compatible BLAS routines.
