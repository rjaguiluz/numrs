# specialized optimizers module

This directory contains implementations of various optimization algorithms for training neural networks.

## Contents

- **`base.rs`**: Defines the `Optimizer` trait, which all optimizers must implement. This ensures a consistent API for `step()`, `zero_grad()`, and `learning_rate()` management.
- **`sgd.rs`**: Stochastic Gradient Descent with optional Momentum and Nesterov.
- **`adam.rs`**: Adaptive Moment Estimation (Adam), the most common optimizer for general deep learning.
- **`adamw.rs`**: Adam with decoupled weight decay, often offering better generalization.
- **`rmsprop.rs`**: Root Mean Square Propagation.
- **`schedulers.rs`**: Learning rate schedulers (e.g., `StepLR`, `CosineAnnealingLR`) to adjust the learning rate during training.
- **Other variants**: `Adagrad`, `Adadelta`, `NAdam`, `RAdam`, `LAMB`, `LBFGS`, `AdaBound`, `Lookahead`.

## Design Rationale

- **Extensibility**: The `Optimizer` trait makes it easy to add new algorithms or research experimental ones.
- **State Management**: Each optimizer struct holds its own internal state (e.g., momentum buffers, previous gradients) separate from the model parameters.

## Interaction

1.  **Creation**: Users instantiate an optimizer by passing a reference to the model's parameters (usually a `Vec<Arc<RwLock<Tensor>>>`).
2.  **Training Loop**:
    - `opt.zero_grad()`: Clears old gradients.
    - `loss.backward()`: Computes new gradients in the Tensors.
    - `opt.step()`: Reads the gradients and updates the Tensors' data arrays.
