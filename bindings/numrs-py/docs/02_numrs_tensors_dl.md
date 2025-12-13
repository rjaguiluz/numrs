# 02. NumRs DL (Python)

## Autograd
```python
x = Tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad) # [4.0]
```

## Neural Networks (`numrs.nn`)

### Layers
*   `numrs.Linear(in, out)`
*   `numrs.ReLU()`
*   `numrs.Sigmoid()`
*   `numrs.Conv1d(...)`

### Sequential
```python
import numrs

model = numrs.Sequential()
model.add(numrs.Linear(10, 32))
model.add(numrs.ReLU())
res = model(input_tensor)
```

## Training (`numrs.Trainer`)
```python
trainer = numrs.Trainer(model, "adam", "mse", 0.01)
trainer.fit(dataset, epochs=5)
```
