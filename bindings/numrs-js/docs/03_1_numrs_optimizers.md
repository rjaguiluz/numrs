# 03. Optimizers (JS)

## Configuration
Use `TrainerBuilder` to configure the training run.

```javascript
const { TrainerBuilder } = require('numrs-node');

const builder = new TrainerBuilder(model);
builder.learningRate(0.001);

// Build with string identifier
const trainer = builder.build("adamw", "cross_entropy");
```

## Supported Optimizers
Pass these string keys to `.build()`:

| Key          | Algorithm    | Notes                                                  |
| :----------- | :----------- | :----------------------------------------------------- |
| `"sgd"`      | **SGD**      | Standard stochastic gradient descent.                  |
| `"adam"`     | **Adam**     | Good default for most problems.                        |
| `"adamw"`    | **AdamW**    | Recommended for Transformers/Modern architectures.     |
| `"nadam"`    | **NAdam**    | Nesterov + Adam.                                       |
| `"radam"`    | **RAdam**    | More stable start than Adam.                           |
| `"rmsprop"`  | **RMSProp**  | Common in RNNs.                                        |
| `"adagrad"`  | **AdaGrad**  | Sparse data friendly.                                  |
| `"adadelta"` | **AdaDelta** | No learning rate tuning usually needed.                |
| `"lamb"`     | **LAMB**     | For large batch training.                              |
| `"adabound"` | **AdaBound** | Transitions from Adam to SGD behavior.                 |
| `"lbfgs"`    | **L-BFGS**   | Second-order method (slow per step, fast convergence). |
| `"rprop"`    | **Rprop**    | Good for full-batch training.                          |

## Supported Losses
1.  **"mse"**: Mean Squared Error.
2.  **"cross_entropy"**: Cross Entropy (Logits input).
