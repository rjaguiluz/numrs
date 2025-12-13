# 00. NumRs R Ops

## Overview
R operations using C ABI via S3 dispatch.

## 1. Element-wise

### Binary
Standard R operators are overloaded for `NumRsTensor`.
| Operation | Method | Usage     |
| :-------- | :----- | :-------- |
| **Add**   | `+`    | `t1 + t2` |
| **Sub**   | `-`    | `t1 - t2` |
| **Mul**   | `*`    | `t1 * t2` |
| **Div**   | `/`    | `t1 / t2` |

Functions:
*   `nr_add(a, b)`
*   `nr_sub(a, b)`

### Unary
| Operation | Method      |
| :-------- | :---------- |
| **Log**   | `nr_log(t)` |
| **Exp**   | `nr_exp(t)` |

## 2. Reduction and Stats
| Operation | Method                        |
| :-------- | :---------------------------- |
| **Mean**  | `nr_mean(t)`                  |
| **Sum**   | `nr_sum(t)`                   |
| **MSE**   | `nr_mse_loss(preds, targets)` |

## 3. Linear Algebra
| Operation    | Method      | Usage               |
| :----------- | :---------- | :------------------ |
| **MatMul**   | `%*%`       | `t1 %*% t2`         |
| **Function** | `nr_matmul` | `nr_matmul(t1, t2)` |
