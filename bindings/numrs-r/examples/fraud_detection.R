library(numrs)

# 1. Synthetic Data Generation (Matching numrs-core logic)
# Features: [Amount, Hour, RiskScore]
# Logic: Fraud if (Risk > 0.8 && Amount > 0.5) || (Risk > 0.95)
cat("Step 1: Generating Dataset (Synthetic Fraud Data)...\n")
set.seed(42)
n_samples <- 2000

cat("Step 1: Generating Dataset (Synthetic Fraud Data - Balanced)...\n")
set.seed(42)
n_samples <- 2000

# Generate balanced data
# Half Legit (Low risk), Half Potential Fraud (High risk)
n_legit <- n_samples / 2
n_fraud <- n_samples / 2

# Group 1: Safe (Risk < 0.8)
amount1 <- runif(n_legit, 0.0, 1.0)
hour1 <- runif(n_legit, 0.0, 1.0)
risk1 <- runif(n_legit, 0.0, 0.8)

# Group 2: Risky (Risk > 0.8)
amount2 <- runif(n_fraud, 0.4, 1.0) # Bias amount higher to trigger fraud logic
hour2 <- runif(n_fraud, 0.0, 1.0)
risk2 <- runif(n_fraud, 0.8, 1.0)

amount <- c(amount1, amount2)
hour <- c(hour1, hour2)
risk <- c(risk1, risk2)

# Shuffle
perm <- sample(n_samples)
amount <- amount[perm]
hour <- hour[perm]
risk <- risk[perm]

# Apply ground truth logic
is_fraud <- (risk > 0.8 & amount > 0.5) | (risk > 0.95)
labels <- as.numeric(is_fraud)

cat(sprintf("   Fraud Rate: %.2f%%\n", mean(labels) * 100))

# Combine into input matrix [N, 3]
input_data <- matrix(c(amount, hour, risk), nrow = n_samples, ncol = 3, byrow = FALSE)

# One-Hot Encoding for CrossEntropyLoss
# Class 0: Legit -> [1, 0]
# Class 1: Fraud -> [0, 1]
# Targets must be [N, 2]
targets_onehot <- matrix(0, nrow = n_samples, ncol = 2)
targets_onehot[labels == 0, 1] <- 1 # Col 1 = Legit
targets_onehot[labels == 1, 2] <- 1 # Col 2 = Fraud

# Convert to NumRs format
# IMPORTANT: R stores matrices Column-Major. NumRs C implementation expects Row-Major.
# We must transpose (t()) before flattening (as.numeric()) to get Row-Major memory layout.
input_tensor <- nr_array(as.numeric(t(input_data)), shape = c(n_samples, 3))
target_tensor <- nr_array(as.numeric(t(targets_onehot)), shape = c(n_samples, 2))

dataset <- nr_dataset(input_tensor, c(n_samples, 3), target_tensor, c(n_samples, 2), batch_size=32)
cat(sprintf("   Train samples: %d\n", n_samples))

# 2. Build Model (Matching Architecture but adapted for CrossEntropy)
# Architecture: Linear(3->16) -> ReLU -> Linear(16->8) -> ReLU -> Linear(8->2)
cat("\nStep 2: Defining MLP Architecture...\n")
model <- nr_sequential(
  nr_linear(3, 16),
  nr_relu_layer(),
  nr_linear(16, 8),
  nr_relu_layer(),
  nr_linear(8, 2)
)
cat("   Architecture: Linear(3->16) -> ReLU -> Linear(16->8) -> ReLU -> Linear(8->2) (Logits)\n")

# 3. Build Trainer
cat("\nStep 3: Training...\n")
builder <- nr_trainer_builder(model)
builder <- nr_with_lr(builder, 0.01)

# Using Adam + CrossEntropy (Stable)
trainer <- nr_build(builder, optimizer="adam", loss="cross_entropy")
cat("Trainer built. Starting training...\n")

# 4. Train
nr_fit(trainer, dataset, epochs=20) # Total 20 epochs
cat("   Training finished.\n")

# 5. Verification (Simulating Production Inference)
cat("\nStep 5: Simulating Production Inference...\n")

# Test Cases: [Amount, Hour, Risk]
# 1. Low risk, low amount -> Legit (0)
# 2. High risk (>0.95) -> Fraud (1)
# 3. High risk (>0.8) & High amount (>0.5) -> Fraud (1)
# 4. High risk (>0.8) BUT Low amount (<0.5) -> Legit (0)
test_cases_matrix <- matrix(c(
  0.1, 0.5, 0.1,  # Legit
  0.9, 0.2, 0.96, # Fraud
  0.6, 0.8, 0.85, # Fraud
  0.2, 0.9, 0.85  # Legit
), ncol=3, byrow=TRUE)

expected_labels <- c(0, 1, 1, 0)
expected_class <- c("Legit", "Fraud", "Fraud", "Legit")

# IMPORTANT: Transpose test input too!
test_in <- nr_tensor(nr_array(as.numeric(t(test_cases_matrix)), shape=c(4, 3)))
logits_tensor <- nr_forward(model, test_in)
logits <- as.numeric(logits_tensor) 
# Logits come as flat vector, reshape to [4, 2] in R logic
logits_mat <- matrix(logits, nrow=4, ncol=2, byrow=TRUE)

# Softmax in R for probability display (manual)
probs_mat <- t(apply(logits_mat, 1, function(x) exp(x) / sum(exp(x))))
fraud_probs <- probs_mat[,2]

preds_class <- ifelse(fraud_probs > 0.5, "Fraud", "Legit")

results <- data.frame(
  Amount = test_cases_matrix[,1],
  Hour = test_cases_matrix[,2],
  Risk = test_cases_matrix[,3],
  Fraud_Prob = round(fraud_probs, 4),
  Pred = preds_class,
  Match = ifelse(preds_class == expected_class, "OK", "FAIL")
)
print(results)
