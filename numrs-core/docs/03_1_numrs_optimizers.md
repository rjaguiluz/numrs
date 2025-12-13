# 03. Optimizers (Rust)

## Configuration
The `TrainerBuilder` simplifies optimizer selection, or you can use them directly if implementing a custom loop.

```rust
// Builder usage
let trainer = TrainerBuilder::new(model)
    .learning_rate(1e-3)
    .build_adam("cross_entropy");
```

## Supported Optimizers (Structs)
Located in `numrs::autograd::optim`.

| Struct     | Algorithm                   |
| :--------- | :-------------------------- |
| `SGD`      | Stochastic Gradient Descent |
| `Adam`     | Adam                        |
| `AdamW`    | AdamW                       |
| `NAdam`    | NAdam                       |
| `RAdam`    | RAdam                       |
| `RMSprop`  | RMSProp                     |
| `AdaGrad`  | AdaGrad                     |
| `AdaDelta` | AdaDelta                    |
| `LAMB`     | LAMB                        |
| `AdaBound` | AdaBound                    |
| `LBFGS`    | L-BFGS                      |
| `Rprop`    | Rprop                       |

## Supported Losses (Structs)
Located in `numrs::autograd::train`.
*   `MSELoss`
*   `CrossEntropyLoss`
