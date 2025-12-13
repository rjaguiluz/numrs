# 06. Hands On: Fraud Detection (Rust)

## Overview
A complete binary classification example.

## Code
```rust
use numrs::{Array, Tensor};
use numrs::nn::{Sequential, Linear, ReLU, Sigmoid};
use numrs::autograd::train::{Dataset, TrainerBuilder};

fn main() -> anyhow::Result<()> {
    // 1. Data
    // (Assume loaded somehow)
    let b = 32;
    let ds = Dataset::new(x_train, y_train, b);

    // 2. Model
    let mut model = Sequential::new();
    model.add(Box::new(Linear::new(30, 16)?));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(Linear::new(16, 1)?));
    model.add(Box::new(Sigmoid::new()));

    // 3. Train
    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(0.01)
        .build_adam("mse"); // Using MSE for simplicity (or BCELoss)

    trainer.fit(&ds, 20);
    
    Ok(())
}
```
