# 04. ONNX Export (R)

## Experimental Support
Exports a `nr_sequential` model to ONNX.

```r
# Create dummy input with shape
dummy_t <- nr_tensor(nr_array(runif(N*D), shape=c(1, D)))

nr_save_onnx(model, dummy_t, "model.onnx")
```
