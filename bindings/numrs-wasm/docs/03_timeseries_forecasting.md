# 03. Time Series Forecasting (WASM)

## Client-Side Time Series
Perform forecasting directly in the user's browser without sending data to a server.

## Model Setup
```javascript
const model = new Sequential();
// Input: [Batch, 1, Sequence]
model.add_conv1d(new Conv1d(1, 16, 3, 1, 1));
model.add_relu(new ReLU());
model.add_flatten(new Flatten(1, -1));
model.add_linear(new Linear(16 * seqLen, 1));
```
