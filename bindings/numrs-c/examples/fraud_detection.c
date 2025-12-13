#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// --- NumRs C Headers (Forward Decls) ---
typedef struct NumRsArray NumRsArray;
typedef struct NumRsTensor NumRsTensor;
typedef struct NumRsSequential NumRsSequential;
typedef struct NumRsLinear NumRsLinear;
typedef struct NumRsReLU NumRsReLU;
typedef struct NumRsDataset NumRsDataset;
typedef struct NumRsTrainerBuilder NumRsTrainerBuilder;
typedef struct NumRsTrainer NumRsTrainer;

// Library Functions (from lib.rs)
extern const char *numrs_version();
extern void numrs_print_startup_log();

// Array
extern NumRsArray *numrs_array_new(const float *data, const uint32_t *shape,
                                   size_t ndim);
extern void numrs_array_free(NumRsArray *p);

// Tensor
extern NumRsTensor *numrs_tensor_new(NumRsArray *arr, bool requires_grad);
extern void numrs_tensor_free(NumRsTensor *p);

// NN
extern NumRsSequential *numrs_sequential_new();
extern NumRsLinear *numrs_linear_new(size_t in, size_t out);
extern NumRsReLU *numrs_relu_layer_new(); // Rename to distinguish from op
extern void numrs_sequential_add_linear(NumRsSequential *seq,
                                        NumRsLinear *layer);
extern void numrs_sequential_add_relu(NumRsSequential *seq, NumRsReLU *layer);

// Training
extern NumRsDataset *numrs_dataset_new(const float *inputs,
                                       const uint32_t *in_shape, size_t in_ndim,
                                       const float *targets,
                                       const uint32_t *out_max_shape,
                                       size_t out_ndim, size_t batch_size);
extern NumRsTrainerBuilder *numrs_trainer_builder_new(NumRsSequential *model);
extern NumRsTrainerBuilder *
numrs_trainer_builder_learning_rate(NumRsTrainerBuilder *b, double lr);
extern NumRsTrainer *numrs_trainer_build_sgd_mse(NumRsTrainerBuilder *b);
extern void numrs_trainer_fit(NumRsTrainer *t, NumRsDataset *ds, size_t epochs);

int main() {
  printf("NumRs C Example: Fraud Detection\n");
  printf("Version: %s\n", numrs_version());
  numrs_print_startup_log();

  // 1. Data Generation (Dummy)
  // 1000 samples, 4 features
  const int N = 1000;
  const int D = 4;
  float *inputs = (float *)malloc(N * D * sizeof(float));
  float *targets = (float *)malloc(N * 1 * sizeof(float));

  for (int i = 0; i < N; i++) {
    float sum = 0;
    for (int j = 0; j < D; j++) {
      float val = (float)rand() / RAND_MAX;
      inputs[i * D + j] = val;
      sum += val;
    }
    targets[i] = (sum > 2.0) ? 1.0f : 0.0f;
  }

  uint32_t in_shape[] = {N, D};
  uint32_t out_shape[] = {N, 1};

  NumRsDataset *dataset =
      numrs_dataset_new(inputs, in_shape, 2, targets, out_shape, 2, 32);
  printf("Dataset created.\n");

  // 2. Model: Linear(4->16) -> ReLU -> Linear(16->1)
  NumRsSequential *model = numrs_sequential_new();

  numrs_sequential_add_linear(model, numrs_linear_new(4, 16));
  numrs_sequential_add_relu(model, numrs_relu_layer_new());
  numrs_sequential_add_linear(model, numrs_linear_new(16, 1));

  printf("Model constructed.\n");

  // 3. Trainer
  NumRsTrainerBuilder *builder = numrs_trainer_builder_new(model);
  builder = numrs_trainer_builder_learning_rate(builder, 0.01);
  NumRsTrainer *trainer = numrs_trainer_build_sgd_mse(builder);

  printf("Starting training...\n");
  numrs_trainer_fit(trainer, dataset, 5); // 5 epochs
  printf("Training finished.\n");

  // Cleanup (Simplified, real app would free everything)
  free(inputs);
  free(targets);

  return 0;
}
