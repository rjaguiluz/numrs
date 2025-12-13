# 03. Optimizers (C)

## Overview
NumRs C ABI supports optimizing models using the `NumRsTrainer`.

## Usage
The primary entry point is `numrs_trainer_build`, which takes string identifiers for the optimizer and loss function.

```c
NumRsTrainerBuilder *builder = numrs_trainer_builder_new(model);
numrs_trainer_builder_learning_rate(builder, 0.01);

// Build with specific Optimizer and Loss
NumRsTrainer *trainer = numrs_trainer_build(builder, "adam", "mse");
```

## Supported Optimizers
The following strings are valid for the `optimizer` argument:

| Identifier   | Algorithm                       | Description                                      |
| :----------- | :------------------------------ | :----------------------------------------------- |
| `"sgd"`      | **Stochastic Gradient Descent** | Classic gradient descent.                        |
| `"adam"`     | **Adam**                        | Adaptive Moment Estimation (Standard).           |
| `"adamw"`    | **AdamW**                       | Adam with Weight Decay fix.                      |
| `"nadam"`    | **NAdam**                       | Nesterov-accelerated Adam.                       |
| `"radam"`    | **RAdam**                       | Rectified Adam.                                  |
| `"rmsprop"`  | **RMSProp**                     | Root Mean Square Propagation.                    |
| `"adagrad"`  | **AdaGrad**                     | Adaptive Gradient Algorithm.                     |
| `"adadelta"` | **AdaDelta**                    | Adaptive learning rate method.                   |
| `"lamb"`     | **LAMB**                        | Layer-wise Adaptive Moments for Batch training.  |
| `"adabound"` | **AdaBound**                    | Adaptive Gradient Methods with Dynamic Bound.    |
| `"lbfgs"`    | **L-BFGS**                      | Limited-memory Broyden–Fletcher–Goldfarb–Shanno. |
| `"rprop"`    | **Rprop**                       | Resilient Backpropagation.                       |

## Supported Losses
| Identifier        | Description                          |
| :---------------- | :----------------------------------- |
| `"mse"`           | Mean Squared Error (Regression).     |
| `"cross_entropy"` | Cross Entropy Loss (Classification). |

## Training Loop
```c
numrs_trainer_fit(trainer, dataset, epochs);
```
