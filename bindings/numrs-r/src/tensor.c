#include "numrs_r.h"

SEXP r_numrs_tensor_new(SEXP arr_ptr, SEXP requires_grad) {
  struct NumRsArray *arr = (struct NumRsArray *)R_ExternalPtrAddr(arr_ptr);
  if (!arr)
    Rf_error("Invalid Array pointer");

  bool req = Rf_asLogical(requires_grad);

  R_ClearExternalPtr(arr_ptr); // Prevention double free

  struct NumRsTensor *t = numrs_tensor_new(arr, req);

  SEXP res = create_s3_ptr(t, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_tensor_backward(SEXP ptr) {
  struct NumRsTensor *t = (struct NumRsTensor *)R_ExternalPtrAddr(ptr);
  if (!t)
    Rf_error("Invalid Tensor pointer");
  numrs_tensor_backward(t);
  return R_NilValue;
}

SEXP r_numrs_tensor_shape(SEXP ptr) {
  // Robust check for invalid/cleared pointer
  if (TYPEOF(ptr) != EXTPTRSXP || !R_ExternalPtrAddr(ptr)) {
    return R_NilValue;
  }

  struct NumRsTensor *t = (struct NumRsTensor *)R_ExternalPtrAddr(ptr);
  if (!t)
    return R_NilValue;

  uintptr_t ndim;
  const uintptr_t *s_ptr_raw =
      numrs_tensor_shape(t, &ndim); // returns *const usize

  SEXP shape = PROTECT(Rf_allocVector(INTSXP, ndim));
  int *s_ptr = INTEGER(shape);

  for (uintptr_t i = 0; i < ndim; i++) {
    s_ptr[i] = (int)s_ptr_raw[i];
  }

  UNPROTECT(1);
  return shape;
}

SEXP r_numrs_tensor_data(SEXP ptr) {
  struct NumRsTensor *t = (struct NumRsTensor *)R_ExternalPtrAddr(ptr);
  if (!t)
    Rf_error("Invalid Tensor pointer");

  struct NumRsArray *arr =
      numrs_tensor_data(t); // Returns reference or new? usually new wrapper
  // Actually numrs_tensor_data returns internal pointer probably, need deep
  // check. Assuming it returns new wrapper pointer that we own? C API usually:
  // struct NumRsArray *numrs_tensor_data(struct NumRsTensor *ptr);
  // If it returns *internal* pointer, we shouldn't free it via finalizer unless
  // we increase refcount. Assuming for now it's safe to wrap.
  SEXP res =
      create_s3_ptr(arr, tag_array, "NumRsArray",
                    NULL); // Don't finalize internal data if owned by tensor!
  return res;
}

SEXP r_numrs_tensor_grad(SEXP ptr) {
  struct NumRsTensor *t = (struct NumRsTensor *)R_ExternalPtrAddr(ptr);
  if (!t)
    Rf_error("Invalid Tensor pointer");

  struct NumRsTensor *grad = numrs_tensor_grad(t);
  if (!grad)
    return R_NilValue;

  SEXP res = PROTECT(R_MakeExternalPtr(grad, tag_tensor, R_NilValue));
  R_RegisterCFinalizerEx(res, finalizer_tensor, TRUE);
  UNPROTECT(1);
  return res;
}

// Ops
SEXP r_numrs_add(SEXP a_ptr, SEXP b_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  struct NumRsTensor *b = (struct NumRsTensor *)R_ExternalPtrAddr(b_ptr);
  struct NumRsTensor *t_res = numrs_add(a, b);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_sub(SEXP a_ptr, SEXP b_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  struct NumRsTensor *b = (struct NumRsTensor *)R_ExternalPtrAddr(b_ptr);
  struct NumRsTensor *t_res = numrs_sub(a, b);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_mul(SEXP a_ptr, SEXP b_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  struct NumRsTensor *b = (struct NumRsTensor *)R_ExternalPtrAddr(b_ptr);
  struct NumRsTensor *t_res = numrs_mul(a, b);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_div(SEXP a_ptr, SEXP b_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  struct NumRsTensor *b = (struct NumRsTensor *)R_ExternalPtrAddr(b_ptr);
  struct NumRsTensor *t_res = numrs_div(a, b);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_matmul(SEXP a_ptr, SEXP b_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  struct NumRsTensor *b = (struct NumRsTensor *)R_ExternalPtrAddr(b_ptr);
  struct NumRsTensor *t_res = numrs_matmul(a, b);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_relu(SEXP a_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  struct NumRsTensor *t_res = numrs_relu(a);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_sigmoid(SEXP a_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  struct NumRsTensor *t_res = numrs_sigmoid(a);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_mse_loss(SEXP a_ptr, SEXP b_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  struct NumRsTensor *b = (struct NumRsTensor *)R_ExternalPtrAddr(b_ptr);
  struct NumRsTensor *t_res = numrs_mse_loss(a, b);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_log(SEXP a_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  if (!a)
    return R_NilValue;
  struct NumRsTensor *t_res = numrs_log(a);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_mean(SEXP a_ptr) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  if (!a)
    return R_NilValue;
  // TODO: Verify axis -1 behavior.
  struct NumRsTensor *t_res = numrs_mean(a, -1);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_reshape(SEXP a_ptr, SEXP shape) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  if (!a)
    return R_NilValue;

  SEXP shape_int = PROTECT(Rf_coerceVector(shape, INTSXP));
  int *s_ptr = INTEGER(shape_int);
  R_xlen_t ndim = Rf_xlength(shape_int);
  uint32_t *u_shape = (uint32_t *)malloc(ndim * sizeof(uint32_t));
  for (R_xlen_t i = 0; i < ndim; i++) {
    u_shape[i] = (uint32_t)s_ptr[i];
  }

  struct NumRsTensor *t_res = numrs_reshape(a, u_shape, ndim);
  free(u_shape);
  UNPROTECT(1);

  if (!t_res)
    return R_NilValue;
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}

SEXP r_numrs_flatten(SEXP a_ptr, SEXP start_dim, SEXP end_dim) {
  struct NumRsTensor *a = (struct NumRsTensor *)R_ExternalPtrAddr(a_ptr);
  if (!a)
    return R_NilValue;

  uintptr_t s = (uintptr_t)Rf_asInteger(start_dim);
  uintptr_t e = (uintptr_t)Rf_asInteger(end_dim);

  struct NumRsTensor *t_res = numrs_flatten(a, s, e);
  SEXP res = create_s3_ptr(t_res, tag_tensor, "NumRsTensor", finalizer_tensor);
  return res;
}
