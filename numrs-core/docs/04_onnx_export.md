# 04. ONNX Export (Rust)

## Functionality
Export any `Tensor` graph to ONNX.

```rust
use numrs::ops::export::export_to_onnx;

// 1. Run forward pass
let output = model.forward(&input)?;

// 2. Export
// 'output' holds the graph trace
export_to_onnx(&output, "model.onnx")?;
```

> **Note**: This traverses the graph backwards from `output`. All nodes required to compute `output` will be saved. `requires_grad=true` nodes become Initializers (weights), others become Inputs.
