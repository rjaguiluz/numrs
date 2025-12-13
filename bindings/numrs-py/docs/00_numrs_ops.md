# 00. NumRs Python Ops

## Overview
Maps `numrs::ops` to Python bindings.

## 1. Element-wise

### Binary
Overloaded operators: `+`, `-`, `*`, `/`, `**` (pow), `@` (matmul).
Broadcasting is supported.

### Unary
| Operation   | Method       | Usage         |
| :---------- | :----------- | :------------ |
| **Log**     | `.log()`     | `t.log()`     |
| **Exp**     | `.exp()`     | `t.exp()`     |
| **Sigmoid** | `.sigmoid()` | `t.sigmoid()` |
| **ReLU**    | `.relu()`    | `t.relu()`    |

## 2. Reduction
| Operation | Method    |
| :-------- | :-------- |
| **Sum**   | `.sum()`  |
| **Mean**  | `.mean()` |

## 3. Shape
| Operation     | Method                 |
| :------------ | :--------------------- |
| **Reshape**   | `.reshape(shape)`      |
| **Flatten**   | `.flatten()`           |
| **Transpose** | `.T` or `.transpose()` |
