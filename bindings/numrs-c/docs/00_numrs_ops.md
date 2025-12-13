# 00. NumRs C Ops

## Overview
This document maps `numrs::ops` to the **C ABI** exposed in `include/numrs.h`. All operations return a new `NumRsTensor*` that **must be freed**.

## 1. Element-wise Operations

### Binary
| Operation | C Function  | Signature                                                |
| :-------- | :---------- | :------------------------------------------------------- |
| **Add**   | `numrs_add` | `NumRsTensor* numrs_add(NumRsTensor *a, NumRsTensor *b)` |
| **Sub**   | `numrs_sub` | `NumRsTensor* numrs_sub(NumRsTensor *a, NumRsTensor *b)` |
| **Mul**   | `numrs_mul` | `NumRsTensor* numrs_mul(NumRsTensor *a, NumRsTensor *b)` |
| **Div**   | `numrs_div` | `NumRsTensor* numrs_div(NumRsTensor *a, NumRsTensor *b)` |
| **Pow**   | `numrs_pow` | `NumRsTensor* numrs_pow(NumRsTensor *a, NumRsTensor *b)` |

### Unary (Math)
| Operation   | C Function      | Signature                                                  |
| :---------- | :-------------- | :--------------------------------------------------------- |
| **Log**     | `numrs_log`     | `NumRsTensor* numrs_log(NumRsTensor *t)`                   |
| **Exp**     | `numrs_exp`     | `NumRsTensor* numrs_exp(NumRsTensor *t)`                   |
| **ReLU**    | `numrs_relu`    | `NumRsTensor* numrs_relu(NumRsTensor *t)`                  |
| **Sigmoid** | `numrs_sigmoid` | `NumRsTensor* numrs_sigmoid(NumRsTensor *t)`               |
| **Softmax** | `numrs_softmax` | `NumRsTensor* numrs_softmax(NumRsTensor *t, int64_t axis)` |

## 2. Reduction
| Operation    | C Function       | Signature                                                                                     |
| :----------- | :--------------- | :-------------------------------------------------------------------------------------------- |
| **Sum**      | `numrs_sum`      | `NumRsTensor* numrs_sum(NumRsTensor *t, const int64_t *axes, size_t n_axes, bool keep_dims)`  |
| **Mean**     | `numrs_mean`     | `NumRsTensor* numrs_mean(NumRsTensor *t, const int64_t *axes, size_t n_axes, bool keep_dims)` |
| **MSE Loss** | `numrs_mse_loss` | `NumRsTensor* numrs_mse_loss(NumRsTensor *preds, NumRsTensor *targets)`                       |

## 3. Linear Algebra
| Operation  | C Function     | Signature                                                   |
| :--------- | :------------- | :---------------------------------------------------------- |
| **MatMul** | `numrs_matmul` | `NumRsTensor* numrs_matmul(NumRsTensor *a, NumRsTensor *b)` |

## 4. Shape Manipulation
| Operation   | C Function      | Signature                                                                        |
| :---------- | :-------------- | :------------------------------------------------------------------------------- |
| **Reshape** | `numrs_reshape` | `NumRsTensor* numrs_reshape(NumRsTensor *t, const uint32_t *shape, size_t ndim)` |
| **Flatten** | `numrs_flatten` | `NumRsTensor* numrs_flatten(NumRsTensor *t, int64_t start_dim, int64_t end_dim)` |
