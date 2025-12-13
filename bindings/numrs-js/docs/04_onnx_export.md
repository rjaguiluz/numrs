# 04. ONNX Export (JS)

## Usage
Export any `Sequential` model to a standard `.onnx` file.

```javascript
// 1. Create a dummy input with correct shape
// (Needed to trace the graph)
const dummyInput = new Tensor(new NumRsArray(data, [1, n_features]), [1, n_features], false);

// 2. Save
model.saveOnnx(dummyInput, "./my_model.onnx");
```
