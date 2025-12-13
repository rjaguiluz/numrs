# 03. Optimizers (R)

## Configuration
Use `nr_build` to finalize the trainer.

```r
builder <- nr_trainer_builder(model)
builder <- nr_with_lr(builder, 0.001)

trainer <- nr_build(builder, "adamw", "mse")
```

## Available Optimizers
| Key          | Name                        |
| :----------- | :-------------------------- |
| `"sgd"`      | Stochastic Gradient Descent |
| `"adam"`     | Adam                        |
| `"adamw"`    | AdamW                       |
| `"nadam"`    | NAdam                       |
| `"radam"`    | RAdam                       |
| `"lamb"`     | LAMB                        |
| `"adabound"` | AdaBound                    |
| `"rmsprop"`  | RMSProp                     |
| `"adagrad"`  | AdaGrad                     |
| `"adadelta"` | AdaDelta                    |
| `"lbfgs"`    | L-BFGS                      |
| `"rprop"`    | Rprop                       |

## Available Losses
*   `"mse"`
*   `"cross_entropy"`
