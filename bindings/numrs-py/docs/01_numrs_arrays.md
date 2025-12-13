# 01. NumRs Arrays & Tensors (Python)

In Python, `Array` usage is mostly hidden behind `Tensor`, which is the primary user-facing class.

## Tensor Creation
```python
from numrs import Tensor

# From List
t = Tensor([1.0, 2.0, 3.0])

# From Nested List (2D)
t2 = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
```

## Data Access
*   `t.item()`: Returns scalar (float).
*   `t.numpy()`: Returns NumPy array copy (if available).
