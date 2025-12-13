# 06. Hands On: R Fraud Detection

Source: `examples/fraud_detection.R`

```r
library(numrs)

# 1. Dataset
# ... (Data prep, remember transpose!) ...
dataset <- nr_dataset(x_tensor, c(N,3), y_tensor, c(N,1), 32)

# 2. Model
model <- nr_sequential(
  nr_linear(3, 16),
  nr_relu_layer(),
  nr_linear(16, 1),
  nr_sigmoid_layer()
)

# 3. Train
trainer <- nr_build(nr_trainer_builder(model), "adam", "mse")
nr_fit(trainer, dataset, epochs=20)
```
