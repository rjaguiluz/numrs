# 06. Hands On: Fraud MLP (Python)

Source: `examples/fraud_detection.py`

```python
import numrs

# 1. Dataset
inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] # XOR
targets = [[0.0], [1.0], [1.0], [0.0]]
dataset = numrs.Dataset(inputs, targets, batch_size=2)

# 2. Model
model = numrs.Sequential()
model.add(numrs.Linear(2, 16))
model.add(numrs.ReLU())
model.add(numrs.Linear(16, 1))
model.add(numrs.Sigmoid())

# 3. Train
trainer = numrs.Trainer(model, "adam", "mse", lr=0.01)
trainer.fit(dataset, epochs=100)
```
