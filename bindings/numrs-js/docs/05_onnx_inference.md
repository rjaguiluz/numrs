# 05. ONNX Inference (JS)

## Logic
Load an ONNX file (proto) and run inference via NumRs engine.

> **Note**: This feature is exposed via `load_onnx` (if compiled with `onnx` feature). Ensure your NumRs build includes it.

```javascript
// Pseudo-code (Feature conditional)
const model = nn.Sequential.loadOnnx("./my_model.onnx");
const output = model.forward(input);
```
