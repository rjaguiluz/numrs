#ifndef NUMRS_R_H
#define NUMRS_R_H

#define R_NO_REMAP
#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>

#include "numrs.h"

// Macros
#define PROTECT_COUNT(n)                                                       \
  int protect_count = n;                                                       \
  PROTECT_depth += n
#define UNPROTECT_ALL() UNPROTECT(protect_count)

// Tensor Ops (Extended)
extern SEXP r_numrs_log(SEXP a_ptr);
extern SEXP r_numrs_mean(SEXP a_ptr);
extern SEXP r_numrs_reshape(SEXP a_ptr, SEXP shape);
extern SEXP r_numrs_flatten(SEXP a_ptr, SEXP start_dim, SEXP end_dim);

// NN Layers (Extended)
extern SEXP r_numrs_conv1d_new(SEXP in_c, SEXP out_c, SEXP k_size, SEXP stride,
                               SEXP pad);
extern SEXP r_numrs_batchnorm1d_new(SEXP num_features);
extern SEXP r_numrs_dropout_new(SEXP p);
extern SEXP r_numrs_flatten_layer_new(SEXP start_dim, SEXP end_dim);
extern SEXP r_numrs_sigmoid_layer_new(); // Layer version vs Op version
extern SEXP r_numrs_softmax_layer_new();

// NN Sequential Adders (Extended)
extern SEXP r_numrs_sequential_add_conv1d(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_batchnorm1d(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_dropout(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_flatten(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_sigmoid(SEXP seq_ptr, SEXP layer_ptr);
extern SEXP r_numrs_sequential_add_softmax(SEXP seq_ptr, SEXP layer_ptr);

// Optim (Generic)
extern SEXP r_numrs_trainer_build(SEXP b_ptr, SEXP optimizer_name,
                                  SEXP loss_name);

// Global Tags (defined in init.c)
extern SEXP tag_array;
extern SEXP tag_tensor;
extern SEXP tag_seq;
extern SEXP tag_linear;
extern SEXP tag_relu;
extern SEXP tag_sigmoid;
extern SEXP tag_softmax;
extern SEXP tag_dropout;
extern SEXP tag_flatten;
extern SEXP tag_batchnorm;
extern SEXP tag_conv1d;
extern SEXP tag_dataset;
extern SEXP tag_trainer;
extern SEXP tag_trainer_builder;

// Missing shape getters (manually added)
void numrs_array_shape(const struct NumRsArray *ptr, uint32_t *out_shape,
                       uintptr_t *out_ndim);
uintptr_t numrs_array_ndim(const struct NumRsArray *ptr);
const uintptr_t *numrs_tensor_shape(struct NumRsTensor *ptr,
                                    uintptr_t *out_ndim);
float *numrs_array_data(struct NumRsArray *ptr);

// Finalizers (defined in init.c)
void finalizer_array(SEXP ptr);
void finalizer_tensor(SEXP ptr);
void finalizer_seq(SEXP ptr);
void finalizer_dataset(SEXP ptr);
void finalizer_trainer(SEXP ptr);

// Helper to create S3 object
SEXP create_s3_ptr(void *ptr, SEXP tag, const char *class_name,
                   void (*finalizer)(SEXP));

#endif
