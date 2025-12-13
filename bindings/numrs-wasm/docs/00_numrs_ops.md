# 00. NumRs WASM Ops

## Overview
WASM bindings mirror the core `ops` module but exposed to JavaScript.

## 1. Element-wise

### Binary
| Operation | JS Function | Usage       |
| :-------- | :---------- | :---------- |
| **Add**   | `add`       | `add(a, b)` |
| **Sub**   | `sub`       | `sub(a, b)` |
| **Mul**   | `mul`       | `mul(a, b)` |
| **Div**   | `div`       | `div(a, b)` |

### Unary
| Operation   | JS Function |
| :---------- | :---------- |
| **Log**     | `log`       |
| **Exp**     | `exp`       |
| **Sigmoid** | `sigmoid`   |

## 2. Reduction
| Operation | JS Function |
| :-------- | :---------- |
| **Sum**   | `sum`       |
| **Mean**  | `mean`      |

## 3. Linear Algebra
| Operation  | JS Function |
| :--------- | :---------- |
| **MatMul** | `matmul`    |
| **Dot**    | `dot`       |
