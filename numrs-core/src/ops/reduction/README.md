# reduction operations module

This directory contains operations that reduce the dimensionality of an array (e.g., collapse a vector to a scalar).

## Contents

- **`sum.rs`**: Computes the sum of elements.
- **`mean.rs`**: Computes the arithmetic mean.
- **`max.rs` / `min.rs`**: Finds the maximum or minimum value.
- **`argmax.rs`**: Returns the indices of maximum values (useful for classification).
- **`variance.rs`**: Computes variance and standard deviation.

## Design Rationale

- **Axis Support**: Reductions can be applied globally (returning 1 element) or along a specific dimension (returning a smaller array).
- **Accumulation Precision**: Sums on `f32` arrays might internally use `f64` accumulators to prevent precision loss on very large datasets.
- **Parallelism**: Efficient reduction requires a "reduce-combine" strategy (MapReduce style) to utilize multiple cores.

## Interaction

1.  **Call**: `numrs::ops::sum(&a, Some(0))` (sum along axis 0).
2.  **Algorithm**:
    - If `a` is contiguous and we reduce all axes -> optimized SIMD loop.
    - If reducing a specific axis -> strided iteration.
3.  **Result**: Returns a new `Array` with rank `N-1` (or scalar).
