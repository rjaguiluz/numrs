# 05. ONNX Inference (Rust)

## Implementation (Experimental)
NumRs can load ONNX graphs and execute them if the Ops are supported.

```rust
// Pseudo-code (Feature dependent)
use numrs::ops::import::load_onnx;

let model = load_onnx("model.onnx")?;
let res = model.forward(&input)?;
```
