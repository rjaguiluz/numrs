# 03. Optimizers (Python)

## Usage
Pass strings to `Trainer`.

```python
import numrs

trainer = numrs.Trainer(model, optimizer="nadam", loss="cross_entropy", lr=0.01)
```

## Supported Optimizers
You can select any update rule supported by the core engine:

| Optimizer    | String Key   | Best For                                   |
| :----------- | :----------- | :----------------------------------------- |
| **SGD**      | `"sgd"`      | Simple, robust baselines.                  |
| **Adam**     | `"adam"`     | General purpose.                           |
| **AdamW**    | `"adamw"`    | Models with weight decay (Regularization). |
| **NAdam**    | `"nadam"`    | Adam + Nesterov momentum.                  |
| **RAdam**    | `"radam"`    | Self-stabilizing start.                    |
| **RMSProp**  | `"rmsprop"`  | Recurrent networks.                        |
| **AdaGrad**  | `"adagrad"`  | Sparse gradients.                          |
| **AdaDelta** | `"adadelta"` | Adaptive LR.                               |
| **LAMB**     | `"lamb"`     | Large batch training (BERT-like).          |
| **AdaBound** | `"adabound"` | Adam speed + SGD generalization.           |
| **L-BFGS**   | `"lbfgs"`    | Small datasets, high precision needs.      |
| **Rprop**    | `"rprop"`    | Full-batch resilient propagation.          |

## Supported Losses
*   `"mse"`
*   `"cross_entropy"`
