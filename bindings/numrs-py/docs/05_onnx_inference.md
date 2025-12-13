# 05. ONNX Inference (Python)

## Loading
Load a generic `.onnx` file as a NumRs model.

```python
import numrs

model = numrs.load_onnx("model.onnx")
output = model(input_tensor)
```
