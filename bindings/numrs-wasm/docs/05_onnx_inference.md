# 05. ONNX Inference (WASM)

## Load from JSON
Import a previously exported JSON model string.

```javascript
const wrapper = OnnxModelWrapper.load_from_json(jsonString);

// Run Inference
// Requires inputs as Array<NumRsArray>
const result = wrapper.infer_simple(["input"], [inputArray]);
console.log(result.get("output"));
```
