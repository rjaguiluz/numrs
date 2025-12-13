# specialized operations module

This directory contains the public API for all mathematical operations in `numrs`.

## Contents

- **`mod.rs`**: Re-exports all operations to the top-level API.
- **`elementwise/`**: Operations that apply a function to every element independently (e.g., `add`, `mul`, `sin`, `exp`). Supports broadcasting (limited).
- **`reduction/`**: Operations that reduce the dimensionality of an array (e.g., `sum`, `mean`, `max`, `min`). Supports `axis` arguments.
- **`linalg/`**: Linear algebra operations (e.g., `matmul`, `dot`).
- **`shape/`**: Operations that change the array structure without changing data (e.g., `reshape`, `transpose`).
- **`stats/`**: Statistical operations (e.g., `var`, `std`).
- **`promotion_wrappers.rs`**: Helper layer that handles type promotion before dispatching. For example, if adding `f32` and `f64`, this layer promotes the `f32` input to `f64` before calling the core execution kernel.

## Design Rationale

The `ops` module is the **user-facing API**. It abstracts away the complexity of type promotion and backend selection.

- **Modularity**: New operations can be added by creating a new file in the appropriate subdirectory and implementing the generic interface.
- **Consistency**: All operations follow a similar signature `op(&a, &b) -> Result<DynArray>`.
- **Inline Dispatch**: Critical operations are marked `#[inline(always)]` to ensure the compiler can optimize the dispatch call site.

## Interaction

1.  **Call**: The user calls `numrs::ops::add(&a, &b)`.
2.  **Promotion**: Use `promotion_wrappers.rs` to ensure `a` and `b` have compatible types.
3.  **Dispatch**: The code fetches the global dispatch table from `src/backend/dispatch.rs`.
4.  **Execution**: The selected kernel (from `src/backend`) is executed.
