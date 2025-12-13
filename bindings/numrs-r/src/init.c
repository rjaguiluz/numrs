#include "numrs_r.h"

// ============================================================================
// Global Tag Definitions
// ============================================================================
SEXP tag_array;
SEXP tag_tensor;
SEXP tag_seq;
SEXP tag_linear;
SEXP tag_relu;
SEXP tag_sigmoid;
SEXP tag_softmax;
SEXP tag_dropout;
SEXP tag_flatten;
SEXP tag_batchnorm;
SEXP tag_conv1d;
SEXP tag_dataset;
SEXP tag_trainer;
SEXP tag_trainer_builder;

// ============================================================================
// Finalizers
// ============================================================================

void finalizer_array(SEXP ptr) {
  if (R_ExternalPtrAddr(ptr)) {
    numrs_array_free((struct NumRsArray *)R_ExternalPtrAddr(ptr));
    R_ClearExternalPtr(ptr);
  }
}

void finalizer_tensor(SEXP ptr) {
  if (R_ExternalPtrAddr(ptr)) {
    numrs_tensor_free((struct NumRsTensor *)R_ExternalPtrAddr(ptr));
    R_ClearExternalPtr(ptr);
  }
}

void finalizer_seq(SEXP ptr) {
  if (R_ExternalPtrAddr(ptr)) {
    numrs_sequential_free((struct NumRsSequential *)R_ExternalPtrAddr(ptr));
    R_ClearExternalPtr(ptr);
  }
}

void finalizer_dataset(SEXP ptr) {
  if (R_ExternalPtrAddr(ptr)) {
    numrs_dataset_free((struct NumRsDataset *)R_ExternalPtrAddr(ptr));
    R_ClearExternalPtr(ptr);
  }
}

void finalizer_trainer(SEXP ptr) {
  if (R_ExternalPtrAddr(ptr)) {
    numrs_trainer_free((struct NumRsTrainer *)R_ExternalPtrAddr(ptr));
    R_ClearExternalPtr(ptr);
  }
}

// ============================================================================
// Utils
// ============================================================================

SEXP r_numrs_version() { return Rf_mkString((const char *)numrs_version()); }

SEXP r_numrs_print_startup_log() {
  numrs_print_startup_log();
  return R_NilValue;
}

SEXP create_s3_ptr(void *ptr, SEXP tag, const char *class_name,
                   void (*finalizer)(SEXP)) {
  SEXP res = PROTECT(R_MakeExternalPtr(ptr, tag, R_NilValue));
  if (finalizer) {
    R_RegisterCFinalizerEx(res, finalizer, TRUE);
  }
  Rf_setAttrib(res, R_ClassSymbol, Rf_mkString(class_name));
  UNPROTECT(1);
  return res;
}

// ============================================================================
// Registration (Forward Declarations)
// ============================================================================

// Array
extern SEXP r_numrs_array_new(SEXP data, SEXP shape);
extern SEXP r_numrs_array_print(SEXP ptr);
extern SEXP r_numrs_array_shape(SEXP ptr);
extern SEXP r_numrs_array_to_r(SEXP ptr);

// Tensor
extern SEXP r_numrs_tensor_new(SEXP arr_ptr, SEXP requires_grad);
extern SEXP r_numrs_tensor_backward(SEXP ptr);
extern SEXP r_numrs_tensor_data(SEXP ptr);
extern SEXP r_numrs_tensor_grad(SEXP ptr);
extern SEXP r_numrs_tensor_shape(SEXP ptr);
extern SEXP r_numrs_add(SEXP a_ptr, SEXP b_ptr);
extern SEXP r_numrs_sub(SEXP a_ptr, SEXP b_ptr);
extern SEXP r_numrs_mul(SEXP a_ptr, SEXP b_ptr);
extern SEXP r_numrs_div(SEXP a_ptr, SEXP b_ptr);
extern SEXP r_numrs_matmul(SEXP a_ptr, SEXP b_ptr);
extern SEXP r_numrs_relu(SEXP a_ptr);
extern SEXP r_numrs_sigmoid(SEXP a_ptr);
extern SEXP r_numrs_mse_loss(SEXP a_ptr, SEXP b_ptr);

// NN
extern SEXP r_numrs_sequential_new();
extern SEXP r_numrs_linear_new(SEXP in_f, SEXP out_f);
extern SEXP r_numrs_relu_layer_new();
extern SEXP r_numrs_sequential_add_linear(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_relu(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_forward(SEXP seq_ptr, SEXP input_ptr);

// Optim
extern SEXP r_numrs_dataset_new(SEXP inputs, SEXP inputs_shape, SEXP targets,
                                SEXP targets_shape, SEXP batch_size);
extern SEXP r_numrs_trainer_builder_new(SEXP seq_ptr);
extern SEXP r_numrs_trainer_builder_learning_rate(SEXP b_ptr, SEXP lr);
extern SEXP r_numrs_trainer_build_sgd_mse(SEXP b_ptr);
extern SEXP r_numrs_trainer_fit_run(SEXP t_ptr, SEXP d_ptr, SEXP epochs);

// Extended Tensor Ops
extern SEXP r_numrs_log(SEXP a_ptr);
extern SEXP r_numrs_mean(SEXP a_ptr);
extern SEXP r_numrs_reshape(SEXP a_ptr, SEXP shape);
extern SEXP r_numrs_flatten(SEXP a_ptr, SEXP start_dim, SEXP end_dim);

// Extended NN
extern SEXP r_numrs_conv1d_new(SEXP in_c, SEXP out_c, SEXP k_size, SEXP stride,
                               SEXP pad);
extern SEXP r_numrs_batchnorm1d_new(SEXP num_features);
extern SEXP r_numrs_dropout_new(SEXP p);
extern SEXP r_numrs_flatten_layer_new(SEXP start_dim, SEXP end_dim);
extern SEXP r_numrs_sigmoid_layer_new();
extern SEXP r_numrs_softmax_layer_new();
extern SEXP r_numrs_sequential_add_conv1d(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_batchnorm1d(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_dropout(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_flatten(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_sigmoid(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_softmax(SEXP seq_ptr, SEXP layer_ptr);

// Extended Optim
extern SEXP r_numrs_trainer_build(SEXP b_ptr, SEXP optimizer_name,
                                  SEXP loss_name);

static const R_CallMethodDef CallEntries[] = {
    {"numrs_version", (DL_FUNC)&r_numrs_version, 0},
    {"numrs_print_startup_log", (DL_FUNC)&r_numrs_print_startup_log, 0},
    {"numrs_array_new", (DL_FUNC)&r_numrs_array_new, 2},
    {"numrs_array_print", (DL_FUNC)&r_numrs_array_print, 1},
    {"numrs_array_shape", (DL_FUNC)&r_numrs_array_shape, 1},
    {"numrs_array_to_r", (DL_FUNC)&r_numrs_array_to_r, 1},
    {"numrs_tensor_new", (DL_FUNC)&r_numrs_tensor_new, 2},
    {"numrs_tensor_backward", (DL_FUNC)&r_numrs_tensor_backward, 1},
    {"numrs_tensor_data", (DL_FUNC)&r_numrs_tensor_data, 1},
    {"numrs_tensor_grad", (DL_FUNC)&r_numrs_tensor_grad, 1},
    {"numrs_tensor_shape", (DL_FUNC)&r_numrs_tensor_shape, 1},
    {"numrs_add", (DL_FUNC)&r_numrs_add, 2},
    {"numrs_sub", (DL_FUNC)&r_numrs_sub, 2},
    {"numrs_mul", (DL_FUNC)&r_numrs_mul, 2},
    {"numrs_div", (DL_FUNC)&r_numrs_div, 2},
    {"numrs_matmul", (DL_FUNC)&r_numrs_matmul, 2},
    {"numrs_relu", (DL_FUNC)&r_numrs_relu, 1},
    {"numrs_sigmoid", (DL_FUNC)&r_numrs_sigmoid, 1},
    {"numrs_mse_loss", (DL_FUNC)&r_numrs_mse_loss, 2},
    {"numrs_sequential_new", (DL_FUNC)&r_numrs_sequential_new, 0},
    {"numrs_linear_new", (DL_FUNC)&r_numrs_linear_new, 2},
    {"numrs_relu_layer_new", (DL_FUNC)&r_numrs_relu_layer_new, 0},
    {"numrs_sequential_add_linear", (DL_FUNC)&r_numrs_sequential_add_linear, 2},
    {"numrs_sequential_add_relu", (DL_FUNC)&r_numrs_sequential_add_relu, 2},
    {"numrs_sequential_forward", (DL_FUNC)&r_numrs_sequential_forward, 2},
    {"numrs_dataset_new", (DL_FUNC)&r_numrs_dataset_new, 5},
    {"numrs_trainer_builder_new", (DL_FUNC)&r_numrs_trainer_builder_new, 1},
    {"numrs_trainer_builder_learning_rate",
     (DL_FUNC)&r_numrs_trainer_builder_learning_rate, 2},
    {"numrs_trainer_build_sgd_mse", (DL_FUNC)&r_numrs_trainer_build_sgd_mse, 1},
    {"numrs_trainer_fit_run", (DL_FUNC)&r_numrs_trainer_fit_run, 3},
    // New Ops
    {"numrs_log", (DL_FUNC)&r_numrs_log, 1},
    {"numrs_mean", (DL_FUNC)&r_numrs_mean, 1},
    {"numrs_reshape", (DL_FUNC)&r_numrs_reshape, 2},
    {"numrs_flatten", (DL_FUNC)&r_numrs_flatten, 3},
    // New NN
    {"numrs_conv1d_new", (DL_FUNC)&r_numrs_conv1d_new, 5},
    {"numrs_batchnorm1d_new", (DL_FUNC)&r_numrs_batchnorm1d_new, 1},
    {"numrs_dropout_new", (DL_FUNC)&r_numrs_dropout_new, 1},
    {"numrs_flatten_layer_new", (DL_FUNC)&r_numrs_flatten_layer_new, 2},
    {"numrs_sigmoid_layer_new", (DL_FUNC)&r_numrs_sigmoid_layer_new, 0},
    {"numrs_softmax_layer_new", (DL_FUNC)&r_numrs_softmax_layer_new, 0},
    {"numrs_sequential_add_conv1d", (DL_FUNC)&r_numrs_sequential_add_conv1d, 2},
    {"numrs_sequential_add_batchnorm1d",
     (DL_FUNC)&r_numrs_sequential_add_batchnorm1d, 2},
    {"numrs_sequential_add_dropout", (DL_FUNC)&r_numrs_sequential_add_dropout,
     2},
    {"numrs_sequential_add_flatten", (DL_FUNC)&r_numrs_sequential_add_flatten,
     2},
    {"numrs_sequential_add_sigmoid", (DL_FUNC)&r_numrs_sequential_add_sigmoid,
     2},
    {"numrs_sequential_add_softmax", (DL_FUNC)&r_numrs_sequential_add_softmax,
     2},
    // New Optim
    {"numrs_trainer_build", (DL_FUNC)&r_numrs_trainer_build, 3},
    {NULL, NULL, 0}};

void R_init_numrs(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

  tag_array = Rf_install("NumRsArray");
  tag_tensor = Rf_install("NumRsTensor");
  tag_seq = Rf_install("NumRsSequential");
  tag_linear = Rf_install("NumRsLinear");
  tag_relu = Rf_install("NumRsReLU");
  tag_dataset = Rf_install("NumRsDataset");
  tag_trainer = Rf_install("NumRsTrainer");
  tag_trainer_builder = Rf_install("NumRsTrainerBuilder");

  tag_linear = Rf_install("NumRsLinear"); // Re-check if repeated
  // Initialize missing tags
  tag_conv1d = Rf_install("NumRsConv1d");
  tag_batchnorm = Rf_install("NumRsBatchNorm1d");
  tag_dropout = Rf_install("NumRsDropout");
  tag_flatten = Rf_install("NumRsFlatten");
  tag_sigmoid = Rf_install("NumRsSigmoid");
  tag_softmax = Rf_install("NumRsSoftmax");
}
