# specialized autograd module

This directory contains the machine learning and automatic differentiation frameowrk.

## Contents

- **`tensor.rs`**: Defines the `Tensor` struct, which wraps `Array<T>` and adds gradient tracking capabilities (`requires_grad`, `grad`). This is the main object for ML workflows.
- **`backward.rs`**: Implements the backward propagation engine. It traverses the compute graph in reverse order to calculate gradients using the chain rule.
- **`ops.rs`**: Defines differentiable operations. These are wrappers around core `ops` that also record the operation in the compute graph for backward passes.
- **`nn.rs`**: Provides higher-level Neural Network modules like `Linear`, `ReLU`, `Sigmoid`, and `Sequential` containers.
- **`optim/`**: Directory containing optimization algorithms (e.g., `SGD`, `Adam`, `RMSprop`) to update model parameters based on gradients.
- **`train.rs`**: High-level training APIs (`Trainer`, `TrainerBuilder`) to simplify the training loop, metrics, and validation.

## Design Rationale

The `autograd` module transforms `numrs` from a simple math library into a Deep Learning framework.

- **Reverse Mode AD**: Chosen for its efficiency in training neural networks (many inputs, scalar loss).
- **Module System**: Inspired by PyTorch, allowing users to compose complex models from simple building blocks.
- **Separation of Concerns**: `autograd` handles the *what* (graph, gradients) while `src/backend` handles the *how* (execution).

## Interaction

1.  **Forward Pass**: Users invoke methods on `Tensor` (e.g., `x.matmul(&w)`). The `autograd` module calls `src/ops` to compute the result and records the node in the graph.
2.  **Backward Pass**: Calling `.backward()` on a scalar `Tensor` (loss) triggers the gradient computation logic in `backward.rs`.
3.  **Update**: Optimizers in `optim/` read the `.grad` field of Tensors and update the `.data` (which are `Array<T>`) in-place.
