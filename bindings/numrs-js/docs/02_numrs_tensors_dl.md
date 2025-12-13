# 02. NumRs Tensors & DL (JS)

## Tensor
Autograd enabled wrapper.

```javascript
const { Tensor } = require('numrs-node');
const t = new Tensor(arr, [2, 2], true); // requires_grad=true
t.backward();
console.log(t.grad);
```

## Deep Learning (`nn`)

### Sequential API
```javascript
const { Sequential, nn } = require('numrs-node');

const model = new Sequential();
model.addLinear(new nn.Linear(10, 32));
model.addReLU(new nn.ReLU());

const out = model.forward(inputTensor);
```
