# 04. ONNX Export (Python)

## API
Save a trained `Sequential` model.

```python
# 1. Create Dummy Input
dummy_input = numrs.Tensor(nested_list_data, requires_grad=False)

# 2. Export
numrs.save_onnx(model, dummy_input, "model.onnx")
```
