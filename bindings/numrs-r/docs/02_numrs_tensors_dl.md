# 02. NumRs Tensors & DL (R)

## Tensor (`NumRsTensor`)
Wraps array for autograd. S3 class.

```r
t <- nr_tensor(arr, requires_grad=TRUE)
t2 <- t + t
nr_backward(t2)
print(nr_grad(t))
```

## Deep Learning

### Layers
*   `nr_linear(in, out)`
*   `nr_relu_layer()`
*   `nr_sigmoid_layer()`

### Sequential Model
```r
model <- nr_sequential(
    nr_linear(10, 32),
    nr_relu_layer()
)
```

### Training
```r
dataset <- nr_dataset(...)
builder <- nr_trainer_builder(model)
trainer <- nr_build(builder, "adam", "mse")

nr_fit(trainer, dataset, epochs=10)
```
