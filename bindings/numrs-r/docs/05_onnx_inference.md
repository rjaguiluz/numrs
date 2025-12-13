# 05. ONNX Inference (R)

## Loading
```r
model <- nr_load_onnx("model.onnx")
out <- nr_forward(model, input_t)
```
