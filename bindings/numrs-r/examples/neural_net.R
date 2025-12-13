# Neural Network Example (Trainer API)
library(numrs)

cat("=== NumRs R Binding - Neural Network ===\n")

# 1. Define Model
# A simple MLP: Linear(2->4) -> ReLU -> Linear(4->1)
model <- nr_sequential()
nr_add_linear(model, nr_linear(2, 4))
nr_add_relu(model, nr_relu_layer())
nr_add_linear(model, nr_linear(4, 1))

cat("Model defined.\n")

# 2. Data (XOR-like problem)
inputs <- c(0,0, 0,1, 1,0, 1,1) # Flattened 4x2
targets <- c(0, 1, 1, 0)        # 4x1

ds <- nr_dataset(
  inputs = inputs, inputs_shape = c(4, 2),
  targets = targets, targets_shape = c(4, 1),
  batch_size = 4
)

# 3. Trainer
builder <- nr_trainer_builder(model)
builder <- nr_with_lr(builder, 0.1)
trainer <- nr_build_sgd(builder)

cat("Starting training (10 epochs)...\n")
nr_fit(trainer, ds, 10)

cat("Training complete.\n")
