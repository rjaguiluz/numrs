# Basic Operations Example
library(numrs)

cat("=== NumRs R Binding - Basic Operations ===\n")

# 1. Create Tensors
a_arr <- nr_array(c(2.0, 3.0), shape = c(2))
b_arr <- nr_array(c(1.0, 4.0), shape = c(2))

a <- nr_tensor(a_arr, requires_grad = TRUE)
b <- nr_tensor(b_arr, requires_grad = TRUE)

cat("Tensor A:\n")
print(nr_data(a))
cat("Tensor B:\n")
print(nr_data(b))

# 2. Arithmetic (using overloaded operators)
c <- a * b + b
cat("\nResult C = A * B + B:\n")
print(nr_data(c))

# 3. Autograd
cat("\nCalculating Gradients (Backprop on sum(C))...\n")
# We need a scalar to backpropagate from, effectively sum(C) if we just want gradients for elements
# But nr_backward works on the tensor directly (implicit sum behavior or specific to numrs semantics)
# Let's perform a reduction to scalar before backward if needed, or assume backward handles non-scalars (usually implies sum)
# In numrs core: tensor.backward() usually requires scalar or implicit gradient of 1s.
nr_backward(c)

cat("Gradient of A (should be B = [1, 4]):\n")
print(nr_data(nr_grad(a)))

cat("Gradient of B (should be A + 1 = [3, 4]):\n")
print(nr_data(nr_grad(b)))
