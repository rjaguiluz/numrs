# 04. ONNX Export (C)

> **Note**: Native C ABI for ONNX export is currently experimental or available via host language bindings.

To export a model trained in C, you currently need to rely on the Rust core re-exports or future ABI expansions. 

*Planned API:*
```c
// Future
numrs_save_onnx(model, dummy_input, "model.onnx");
```
