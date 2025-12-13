#include "numrs_r.h"

SEXP r_numrs_sequential_new() {
  struct NumRsSequential *seq = numrs_sequential_new();
  SEXP res = PROTECT(R_MakeExternalPtr(seq, tag_seq, R_NilValue));
  R_RegisterCFinalizerEx(res, finalizer_seq, TRUE);
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_linear_new(SEXP in_f, SEXP out_f) {
  uintptr_t i = (uintptr_t)Rf_asInteger(in_f);
  uintptr_t o = (uintptr_t)Rf_asInteger(out_f);
  struct NumRsLinear *l = numrs_linear_new(i, o);
  SEXP res = PROTECT(R_MakeExternalPtr(l, tag_linear, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_relu_layer_new() {
  struct NumRsReLU *l = numrs_relu_layer_new();
  SEXP res = PROTECT(R_MakeExternalPtr(l, tag_relu, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_sequential_add_linear(SEXP seq_ptr, SEXP layer_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsLinear *layer =
      (struct NumRsLinear *)R_ExternalPtrAddr(layer_ptr);
  numrs_sequential_add_linear(seq, layer);
  R_ClearExternalPtr(layer_ptr); // Ownership transferred
  return R_NilValue;
}

SEXP r_numrs_sequential_add_relu(SEXP seq_ptr, SEXP layer_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsReLU *layer = (struct NumRsReLU *)R_ExternalPtrAddr(layer_ptr);
  numrs_sequential_add_relu(seq, layer);
  R_ClearExternalPtr(layer_ptr);
  return R_NilValue;
}

SEXP r_numrs_sequential_forward(SEXP seq_ptr, SEXP input_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsTensor *input =
      (struct NumRsTensor *)R_ExternalPtrAddr(input_ptr);
  struct NumRsTensor *res = numrs_sequential_forward(seq, input);
  SEXP r = PROTECT(R_MakeExternalPtr(res, tag_tensor, R_NilValue));
  R_RegisterCFinalizerEx(r, finalizer_tensor, TRUE);
  UNPROTECT(1);
  return r;
}

// --- Extended Layers ---

SEXP r_numrs_conv1d_new(SEXP in_c, SEXP out_c, SEXP k_size, SEXP stride,
                        SEXP pad) {
  uintptr_t ic = (uintptr_t)Rf_asInteger(in_c);
  uintptr_t oc = (uintptr_t)Rf_asInteger(out_c);
  uintptr_t k = (uintptr_t)Rf_asInteger(k_size);
  uintptr_t s = (uintptr_t)Rf_asInteger(stride);
  uintptr_t p = (uintptr_t)Rf_asInteger(pad);

  struct NumRsConv1d *l = numrs_conv1d_new(ic, oc, k, s, p);
  SEXP res = PROTECT(R_MakeExternalPtr(l, tag_conv1d, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_batchnorm1d_new(SEXP num_features) {
  uintptr_t nf = (uintptr_t)Rf_asInteger(num_features);
  struct NumRsBatchNorm1d *l = numrs_batchnorm1d_new(nf);
  SEXP res = PROTECT(R_MakeExternalPtr(l, tag_batchnorm, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_dropout_new(SEXP p) {
  double prob = Rf_asReal(p);
  struct NumRsDropout *l = numrs_dropout_new((float)prob);
  SEXP res = PROTECT(R_MakeExternalPtr(l, tag_dropout, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_flatten_layer_new(SEXP start_dim, SEXP end_dim) {
  uintptr_t s = (uintptr_t)Rf_asInteger(start_dim);
  uintptr_t e = (uintptr_t)Rf_asInteger(end_dim);
  struct NumRsFlatten *l = numrs_flatten_new(s, e);
  SEXP res = PROTECT(R_MakeExternalPtr(l, tag_flatten, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_sigmoid_layer_new() {
  struct NumRsSigmoid *l = numrs_sigmoid_new();
  SEXP res = PROTECT(R_MakeExternalPtr(l, tag_sigmoid, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_softmax_layer_new() {
  struct NumRsSoftmax *l = numrs_softmax_new();
  SEXP res = PROTECT(R_MakeExternalPtr(l, tag_softmax, R_NilValue));
  UNPROTECT(1);
  return res;
}

// --- Extended Sequential Adders ---

SEXP r_numrs_sequential_add_conv1d(SEXP seq_ptr, SEXP layer_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsConv1d *layer =
      (struct NumRsConv1d *)R_ExternalPtrAddr(layer_ptr);
  numrs_sequential_add_conv1d(seq, layer);
  R_ClearExternalPtr(layer_ptr);
  return R_NilValue;
}

SEXP r_numrs_sequential_add_batchnorm1d(SEXP seq_ptr, SEXP layer_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsBatchNorm1d *layer =
      (struct NumRsBatchNorm1d *)R_ExternalPtrAddr(layer_ptr);
  numrs_sequential_add_batchnorm1d(seq, layer);
  R_ClearExternalPtr(layer_ptr);
  return R_NilValue;
}

SEXP r_numrs_sequential_add_dropout(SEXP seq_ptr, SEXP layer_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsDropout *layer =
      (struct NumRsDropout *)R_ExternalPtrAddr(layer_ptr);
  numrs_sequential_add_dropout(seq, layer);
  R_ClearExternalPtr(layer_ptr);
  return R_NilValue;
}

SEXP r_numrs_sequential_add_flatten(SEXP seq_ptr, SEXP layer_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsFlatten *layer =
      (struct NumRsFlatten *)R_ExternalPtrAddr(layer_ptr);
  numrs_sequential_add_flatten(seq, layer);
  R_ClearExternalPtr(layer_ptr);
  return R_NilValue;
}

SEXP r_numrs_sequential_add_sigmoid(SEXP seq_ptr, SEXP layer_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsSigmoid *layer =
      (struct NumRsSigmoid *)R_ExternalPtrAddr(layer_ptr);
  numrs_sequential_add_sigmoid(seq, layer);
  R_ClearExternalPtr(layer_ptr);
  return R_NilValue;
}

SEXP r_numrs_sequential_add_softmax(SEXP seq_ptr, SEXP layer_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsSoftmax *layer =
      (struct NumRsSoftmax *)R_ExternalPtrAddr(layer_ptr);
  numrs_sequential_add_softmax(seq, layer);
  R_ClearExternalPtr(layer_ptr); // Ownership transferred
  return R_NilValue;
}
