# 03. Optimizers (WASM)

## Usage
Pass the optimizer name string to the `Trainer` constructor.

```javascript
import { Trainer } from '@numrs/wasm';

const trainer = new Trainer(model, "radam", "mse", 0.001);
```

## Supported Optimizers
The WASM binding supports the full NumRs optimizer suite:

*   `"sgd"`
*   `"adam"`
*   `"adamw"`
*   `"nadam"`
*   `"radam"`
*   `"lamb"`
*   `"adabound"`
*   `"rmsprop"`
*   `"adagrad"`
*   `"adadelta"`
*   `"lbfgs"`
*   `"rprop"`

## Supported Losses
*   `"mse"`
*   `"cross_entropy"`
