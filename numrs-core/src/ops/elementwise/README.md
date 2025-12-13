# elementwise operations module

This directory contains operations that apply a function to each element of an input array independently.

## Contents

- **`binary/`**: Operations taking two inputs (e.g., `add(&a, &b)`, `mul`, `div`). Broadly supports broadcasting (expanding smaller arrays to match larger ones).
- **`unary/`**: Operations taking one input (e.g., `sin(&a)`, `cos`, `exp`, `neg`).

## Design Rationale

- **Parallelism**: Because each output element depends only on the corresponding input element(s), these operations are embarrassingly parallel.
- **SIMD**: These maps perfectly to SIMD instructions, allowing us to process 8 `f32`s or 4 `f64`s at once on modern CPUs.
- **Zero-Copy**: Where possible, we aim for zero-copy views, but elementwise ops usually allocate a new result array.

## Interaction

1.  **Call**: User calls `numrs::ops::add(a, b)`.
2.  **Dispatch**: The system checks input types. If they differ (e.g., `i32` + `f32`), the integer array is promoted to float.
3.  **Kernel**: The generic backend kernel iterates over the data (potentially using SIMD/Threads) and writes to the output.
