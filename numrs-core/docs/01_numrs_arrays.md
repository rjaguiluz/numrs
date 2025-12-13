# 01. Arrays (Rust)

## The `Array<T>` Struct
`numrs::Array` is the backing store for Tensors. It holds data in a contiguous `Vec<T>` and metadata about shape and strides.

### Creation
```rust
use numrs::Array;

// From data and shape
let data = vec![1.0, 2.0, 3.0, 4.0];
let arr = Array::new(vec![2, 2], data);

// Zeros / Ones
let z = Array::<f32>::zeros(vec![3, 3]);
let o = Array::<f32>::ones(vec![3, 3]);
```

### Access
```rust
let val = arr.item(); // If scalar
let vec = arr.data;   // Move out Vec
```

### Memory Layout
NumRs uses **Row-Major** layout (C-style). Use `transpose()` if interacting with Column-Major libraries inside Rust manually, though `Tensor` handles this often abstractly.
