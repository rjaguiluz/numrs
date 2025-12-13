# 00. Operations (Rust)

## Overview
NumRs implements a wide range of tensor operations found in `numrs::ops`. Most are exposed as methods on `Tensor` or via trait overloading.

## Binary Operations
Standard arithmetic is supported on `&Tensor`.

```rust
let c = &a + &b;
let c = &a - &b;
let c = &a * &b; // Element-wise
let c = &a / &b;
let c = a.matmul(&b); // Matrix Multiplication
```

## Unary Operations
```rust
let b = a.log();
let b = a.exp();
let b = a.relu();
let b = a.sigmoid();
let b = a.tanh();
```

## Reduction
```rust
let s = a.sum(None);       // Sum all
let s = a.sum(Some(0));    // Sum axis 0
let m = a.mean(Some(1));   // Mean axis 1
```

## Shape Manipulation
```rust
let b = a.reshape(vec![4, 5]);
let b = a.transpose(0, 1);
let b = a.flatten(1, -1);
```
