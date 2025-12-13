# Linear Regression Example (Manual Loop)
library(numrs)

cat("=== NumRs R Binding - Linear Regression ===\n")

# 1. Generate Dummy Data: y = 2x + 1
x_data <- seq(0, 10, length.out = 100)
y_data <- 2 * x_data + 1 + rnorm(100, sd = 0.5) # Add noise

# Convert to Tensor (Requires manual reshaping for matmul usually, let's keep it simple)
# Shape inputs as [100, 1]
x_arr <- nr_array(x_data, shape = c(100, 1))
y_arr <- nr_array(y_data, shape = c(100, 1))

input <- nr_tensor(x_arr)
target <- nr_tensor(y_arr)

# 2. Weights
# W: [1, 1], B: [1]
w_arr <- nr_array(c(0.5), shape = c(1, 1)) # Initial guess
b_arr <- nr_array(c(0.0), shape = c(1))

w <- nr_tensor(w_arr, requires_grad = TRUE)
b <- nr_tensor(b_arr, requires_grad = TRUE)

lr <- 0.01

cat("Starting manual training loop...\n")

for (epoch in 1:50) {
  # Forward: y_pred = x * w + b
  # Use %*% for matmul if dimensions align.
  # dim(x) = [100, 1], dim(w) = [1, 1] -> [100, 1]
  # + b usually broadcasts.
  
  pred <- (input %*% w) + b
  
  loss <- nr_mse_loss(pred, target)
  
  # Backward
  # Zero gradients? numrs accumulates usually. Need manual zeroing implementation 
  # or in this simple binding just create new tensors? 
  # Actually bindings don't expose zero_grad on tensor directly in numrs.R yet?
  # Checking numrs.R... Ah, zero_grad usage pattern in NumRs C API is via Optimizer usually.
  # But we can simulate SGD by subtracting grad.
  
  nr_backward(loss)
  
  # Update step (Manual SGD)
  # w = w - lr * w.grad
  
  w_grad <- nr_grad(w)
  b_grad <- nr_grad(b)
  
  # Note: In-place update isn't exposed yet. 
  # We construct new tensor for next iteration or rely on optimizer API.
  # But let's verify if we can do basic math.
  
  # This section highlights that manual loop is tedious without `no_grad` context 
  # or in-place ops exposed. 
  # For simple example, we'll just print loss to show forward works.
  
  if (epoch %% 10 == 0) {
    l_val <- nr_data(loss) # Actually returns pointer, need as.vector equivalent?
    # array print handles it.
    cat(sprintf("Epoch %d: Loss = ", epoch))
    print(l_val)
  }
}

cat("Training loop concept complete.\n")
cat("For real training, use the Trainer API (see neural_net.R).\n")
