# specialized array module

This directory contains the core data structures for `numrs`.

## Contents

- **`array.rs`**: Defines the generic `Array<T>` struct. This is the fundamental building block of the library, holding contiguous memory for data and shape information.
- **`dyn_array.rs`**: Defines `DynArray`, an enum wrapper that erases the generic type parameter. This is crucial for Python/JS bindings and providing a NumPy-like dynamic API (e.g., `ops::add(&dyn_a, &dyn_b)` could return an `Array<f32>` or `Array<f64>` depending on inputs).
- **`dtype.rs`**: Defines the `DType` enum and trait system, enumerating supported types (F32, F64, I32, I8, U8, Bool).
- **`promotion.rs`**: Implements type promotion logic (e.g., `f32` + `f64` -> `f64`). This ensures mathematical correctness when operating on mixed types.
- **`ops_traits.rs`**: Implements Rust operator overloading (`Add`, `Sub`, `Mul`, `Div`) for `&Array<T>` and `&DynArray`, allowing natural syntax like `&a + &b`.

## Design Rationale

The `Array` system is designed to balance **performance** (via generics and contiguous memory) with **usability** (via `DynArray` and type promotion).

- `Array<T>` is zero-cost and used internally by kernels.
- `DynArray` adds a small dispatch overhead but enables dynamic interfaces required by interpreted languages.
- `promoted_dtype` ensures NumRs behaves intuitively like NumPy.

## Interaction

1.  **Input**: Users typically create `Array<T>` or `DynArray`.
2.  **Ops**: Operations in `src/ops` take these arrays as input.
3.  **Backend**: The raw data slice from `Array<T>` is passed to backend kernels (SIMD/BLAS) for processing.
