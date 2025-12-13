# 06. Hands On: Browser Fraud Detection (WASM)

Source: `examples/fraud_detection/index.js`

Runs entirely in the browser using WebGPU/WASM.

```javascript
// 1. Data
const xTrain = new Tensor(xArr, false);
const yTrain = new Tensor(yArr, false);

// 2. Model
const model = new Sequential();
model.add_linear(new Linear(29, 64));
model.add_relu(new ReLU());
model.add_linear(new Linear(64, 1));
model.add_sigmoid(new Sigmoid());

// 3. Train
const trainer = new Trainer(model, "adam", "cross_entropy", 0.01);
trainer.fit(xTrain, yTrain, 5);
```
