# 06. Hands On: Fraud Detection (JS)

Source: `examples/fraud_detection.ts`

```typescript
import { Tensor, nn, TrainerBuilder, Dataset } from 'numrs-node';

// 1. Dataset
const dataset = new Dataset(xData, yData, 32);

// 2. Model
const model = new nn.Sequential();
model.addLinear(new nn.Linear(3, 16));
model.addReLU(new nn.ReLU());
model.addLinear(new nn.Linear(16, 1));
model.addSigmoid(new nn.Sigmoid());

// 3. Train
const builder = new TrainerBuilder(model);
const trainer = builder.build("adam", "mse");
trainer.fit(dataset, dataset, 10, true);
```
