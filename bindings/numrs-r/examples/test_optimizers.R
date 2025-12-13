library(numrs)

cat("Verifying Optimizer Dispatch...\n")

# Synthetic data
set.seed(42)
N <- 100
D <- 10
X <- nr_array(as.numeric(matrix(rnorm(N*D), N, D)), shape=c(N,D))
Y <- nr_array(as.numeric(matrix(rnorm(N), N, 1)), shape=c(N,1))
dataset <- nr_dataset(X, c(N,D), Y, c(N,1), batch_size=10)

train_with_opt <- function(opt_name, lr) {
  cat(paste("\nTesting optimizer:", opt_name, "\n"))
  model <- nr_sequential()
  nr_add_linear(model, nr_linear(D, 1))
  
  builder <- nr_trainer_builder(model)
  builder <- nr_with_lr(builder, lr)
  
  # Build generic trainer
  trainer <- tryCatch({
    nr_build(builder, optimizer=opt_name, loss="mse")
  }, error = function(e) {
    cat(paste("Failed to build with", opt_name, ":", e$message, "\n"))
    return(NULL)
  })
  
  if (is.null(trainer)) return(NULL)
  
  cat("Trainer built successfully. Running fit...\n")
  nr_fit(trainer, dataset, epochs=2)
  cat("Fit complete.\n")
}

optimizers_to_test <- c(
  "sgd", 
  "adam", "adamw", 
  "rmsprop", 
  "adagrad", "adadelta", 
  "nadam", "radam", 
  "lamb", "adabound",
  "lbfgs", "rprop"
)

for (opt in optimizers_to_test) {
  train_with_opt(opt, 0.01)
}

train_with_opt("invalid_opt", 0.01) # Should fail gracefully

cat("\nDone verifying all optimizers.\n")
