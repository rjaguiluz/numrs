library(numrs)

# 1. Synthetic Time Series Data (Sine Wave Forecasting)
set.seed(42)
n_samples <- 100
seq_length <- 50 # Sequence length
in_channels <- 1 # Uniform channels

# Create simple sine waves with random phase/freq
input_series <- array(0, dim=c(n_samples, in_channels, seq_length))
labels <- array(0, dim=c(n_samples, 1)) # Scalar target (next value prediction or classification)

for (i in 1:n_samples) {
  t <- seq(0, 4*pi, length.out=seq_length)
  phase <- runif(1, 0, pi)
  freq <- runif(1, 1, 2)
  signal <- sin(freq*t + phase)
  input_series[i, 1, ] <- signal
  
  # Target: Is mean positive? (Binary classification)
  labels[i, 1] <- as.numeric(mean(signal) > 0)
}

# Convert to NumRs
# IMPORTANT: R arrays are Column-Major. C expects Row-Major.
# For 3D tensor [N, C, L], C expects L fastest, then C, then N.
# Permute to [L, C, N] effectively transposes the flattened sequence to row-major [N, C, L].
input_tensor <- nr_array(as.numeric(aperm(input_series, c(3, 2, 1))), shape = c(n_samples, in_channels, seq_length))

# One-Hot Encoding for CrossEntropy
# Class 0: Negative Mean -> [1, 0]
# Class 1: Positive Mean -> [0, 1]
targets_onehot <- matrix(0, nrow = n_samples, ncol = 2)
targets_onehot[labels == 0, 1] <- 1
targets_onehot[labels == 1, 2] <- 1
# Targets need transposition t() for row-major layout
target_tensor <- nr_array(as.numeric(t(targets_onehot)), shape = c(n_samples, 2))

dataset <- nr_dataset(input_tensor, c(n_samples, in_channels, seq_length), target_tensor, c(n_samples, 2), batch_size=16)

# 2. Build CNN Model
# Input: [n_samples, 1, 50]
model <- nr_sequential(
  nr_conv1d(in_channels=1, out_channels=8, kernel_size=3), # -> [N, 8, 48]
  nr_batchnorm1d(8),
  nr_relu_layer(),
  nr_flatten_layer(start_dim=1, end_dim=-1), # -> [N, 8*48 = 384]
  nr_dropout(0.2),
  nr_linear(384, 2) # 2 Logits for CrossEntropy
  # No Sigmoid
)

cat("CNN Model built.\n")

# 3. Build Trainer
builder <- nr_trainer_builder(model)
builder <- nr_with_lr(builder, 0.01) # Slightly higher LR for CE
trainer <- nr_build(builder, optimizer="adam", loss="cross_entropy")

cat("Trainer built. Starting training...\n")

# 4. Train
nr_fit(trainer, dataset, epochs=50)

cat("Training complete.\n")

# 5. Verify
cat("\nVerification (First 20 Samples):\n")
val_indices <- 1:20
# Permute validation input FIRST
val_in_data <- input_series[val_indices,,,drop=FALSE]
val_in_permuted <- aperm(val_in_data, c(3, 2, 1))
val_in <- nr_tensor(nr_array(as.numeric(val_in_permuted), shape=c(length(val_indices), in_channels, seq_length)))

out_tensor <- nr_forward(model, val_in)
logits <- as.numeric(out_tensor)
logits_mat <- matrix(logits, nrow=length(val_indices), ncol=2, byrow=TRUE)

# Softmax
probs_mat <- t(apply(logits_mat, 1, function(x) exp(x) / sum(exp(x))))
pos_mean_probs <- probs_mat[,2]
pred_labels <- as.integer(pos_mean_probs > 0.5)
expected_labels <- labels[val_indices]

results <- data.frame(
  Sample = val_indices,
  Pos_Mean_Prob = round(pos_mean_probs, 4),
  Predicted_Label = pred_labels,
  Expected_Label = expected_labels,
  Error = round(abs(pos_mean_probs - expected_labels), 4),
  Correct = ifelse(pred_labels == expected_labels, "YES", "NO")
)
print(results)

accuracy <- mean(results$Predicted_Label == results$Expected_Label)
cat(sprintf("\nVal Accuracy: %.2f%%\n", accuracy * 100))
