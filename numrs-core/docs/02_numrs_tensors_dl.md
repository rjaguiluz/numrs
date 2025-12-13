# 02. Tensors & Deep Learning (Rust)

## Tensors (`Tensor`)
Wraps an `Array` + `Autograd` context.

```rust
let t = Tensor::new(arr, true); // Track gradients
```

### Backward Pass
```rust
y.backward(); // Populates .grad() on leaves
```

## Neural Networks (`numrs::nn`)

### Sequential API
Stacks `Box<dyn Module>`.

```rust
use numrs::nn::*;

let mut model = Sequential::new();
model.add(Box::new(Linear::new(784, 128)?)); // Returns Result
model.add(Box::new(ReLU::new()));
model.add(Box::new(Linear::new(128, 10)?));
```

### Custom Modules
Implement the `Module` trait.

```rust
struct MyLayer { ... }

impl Module for MyLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> { ... }
    fn parameters(&self) -> Vec<Tensor> { ... }
}
```

## Training
Use `numrs::autograd::train::Trainer`.

```rust
let builder = TrainerBuilder::new(model);
let mut trainer = builder.build_sgd("mse");
trainer.fit(&dataset, 10);
```
