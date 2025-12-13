# 00. NumRs JS Ops

## Overview
Operations on `Tensor` or `NumRsArray` in Node.js.

## 1. Element-wise

### Binary
| Operation | Method        | Usage        |
| :-------- | :------------ | :----------- |
| **Add**   | `.add(other)` | `t1.add(t2)` |
| **Sub**   | `.sub(other)` | `t1.sub(t2)` |
| **Mul**   | `.mul(other)` | `t1.mul(t2)` |
| **Div**   | `.div(other)` | `t1.div(t2)` |

### Unary
| Operation | Method   |
| :-------- | :------- |
| **Log**   | `.log()` |
| **Exp**   | `.exp()` |

## 2. Reduction
| Operation | Method        |
| :-------- | :------------ |
| **Sum**   | `.sum(axis)`  |
| **Mean**  | `.mean(axis)` |

## 3. Linear Algebra
| Operation  | Method           |
| :--------- | :--------------- |
| **MatMul** | `.matmul(other)` |
