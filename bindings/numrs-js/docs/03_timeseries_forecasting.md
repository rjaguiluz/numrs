# 03. Time Series Forecasting (JS)

## Overview
Predicting future values using a 1D Convolutional Network.

## Model
```javascript
const model = new nn.Sequential();

// Input: [Batch, 1, 50] (Sequence length 50)
model.addConv1d(new nn.Conv1d(1, 16, 3, 1, 1)); // -> [Batch, 16, 50]
model.addReLU(new nn.ReLU());
model.addFlatten(new nn.Flatten(1, -1));         // -> [Batch, 800]
model.addLinear(new nn.Linear(16 * 50, 1));      // -> [Batch, 1]
```

## Data Prep
Ensure inputs are 3D: `[Batch, Channels, Length]`.
```javascript
const x = new Float32Array(batchSize * 1 * seqLen);
const shape = [batchSize, 1, seqLen];
const tensor = new Tensor(new NumRsArray(x, shape), shape, false);
```
