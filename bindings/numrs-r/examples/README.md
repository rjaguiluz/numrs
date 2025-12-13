# NumRs R Binding Examples

This directory contains examples demonstrating how to use the `numrs` R package.

## Examples

1.  **`basic_ops.R`**: Demonstrates tensor creation, basic arithmetic (`+`, `*`), and autograd (backpropagation).
2.  **`neural_net.R`**: Demonstrates the High-Level API. Defines a sequential model, creates a dataset, and trains using the `Trainer` API (SGD).
3.  **`linear_regression.R`**: Demonstrates a manual training loop concept (though `Trainer` is recommended).

## Usage

Ensure the package is installed:

```R
devtools::install("path/to/bindings/numrs-r")
```

Then run an example:

```bash
Rscript examples/basic_ops.R
```
