# 03. Time Series Forecasting (C)

## Overview
Example of setting up a 1D Convolutional Neural Network (CNN) for time series data using the C ABI.

## Architecture
`Input (N, 1, L) -> Conv1d -> ReLU -> Flatten -> Linear -> Output`

## Implementation

```c
// 1. Data Shape: [Batch, Channels=1, Length=SequenceLength]
// Note: Requires 3D input tensor

// 2. Model
NumRsSequential *model = numrs_sequential_new();

// Conv1d: in_channels=1, out_channels=16, kernel=3, stride=1, padding=1
numrs_sequential_add_conv1d(model, numrs_conv1d_new(1, 16, 3, 1, 1));
numrs_sequential_add_relu(model, numrs_relu_layer_new());

// Flatten: Collapse [Batch, 16, L] -> [Batch, 16*L]
// Assuming L is known or fixed after conv
numrs_sequential_add_flatten(model, numrs_flatten_new(1, -1));

// Linear Head
numrs_sequential_add_linear(model, numrs_linear_new(16 * L, 1));
```
