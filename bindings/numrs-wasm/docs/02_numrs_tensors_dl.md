# 02. NumRs Tensors & DL (WASM)

## Tensor
Wraps `NumRsArray` for autograd.

```javascript
import { Tensor } from 'numrs-wasm';
const t = new Tensor(arr, true); // requires_grad=true
```

## Deep Learning (`Sequential`)

```javascript
import { Sequential, Linear, ReLU, Sigmoid } from 'numrs-wasm';

const model = new Sequential();
model.add_linear(new Linear(10, 32));
model.add_relu(new ReLU());
```

## Training (`Trainer`)

```javascript
import { Trainer } from 'numrs-wasm';

const trainer = new Trainer(model, "adam", "mse", 0.01);
const history = trainer.fit(xTrain, yTrain, 10);
```
