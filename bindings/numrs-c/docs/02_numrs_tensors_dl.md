# 02. NumRs Tensors & DL (C)

## Tensors vs Arrays
*   `NumRsArray`: Raw data.
*   `NumRsTensor`: Wrapper for Autograd.

```c
// Promote Array to Tensor
NumRsTensor *t = numrs_tensor_new(arr, true); // true = requires_grad
```

## Deep Learning Modules
NumRs C ABI exposes standard layers as opaque structs.

### Layers
*   `NumRsLinear* numrs_linear_new(size_t in, size_t out)`
*   `NumRsReLU* numrs_relu_layer_new()`
*   `NumRsSigmoid* numrs_sigmoid_new()`
*   `NumRsConv1d* numrs_conv1d_new(...)`

### Sequential Containers
```c
NumRsSequential *model = numrs_sequential_new();
numrs_sequential_add_linear(model, numrs_linear_new(10, 32));
numrs_sequential_add_relu(model, numrs_relu_layer_new());
```

### Forward Pass
```c
NumRsTensor *output = numrs_sequential_forward(model, input_tensor);
```

### Autograd
```c
// 1. Forward
NumRsTensor *loss = numrs_mse_loss(output, target);

// 2. Backward
numrs_tensor_backward(loss);

// 3. Access Gradients (if needed manually)
// Gradients are stored internally for the optimizer step.
```
