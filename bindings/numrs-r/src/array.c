#include "numrs_r.h"

SEXP r_numrs_array_new(SEXP data, SEXP shape) {
  // Validate inputs
  if (!Rf_isReal(data))
    Rf_error("Data must be real vector");
  if (!Rf_isInteger(shape) && !Rf_isReal(shape))
    Rf_error("Shape must be integer vector");

  // Convert inputs
  const double *d_ptr = REAL(data);
  R_xlen_t len = Rf_xlength(data);

  // Create float buffer copy
  float *f_data = (float *)malloc(len * sizeof(float));
  for (R_xlen_t i = 0; i < len; i++) {
    f_data[i] = (float)d_ptr[i];
  }

  // Convert shape
  SEXP shape_int = PROTECT(Rf_coerceVector(shape, INTSXP));
  int *s_ptr = INTEGER(shape_int);
  R_xlen_t ndim = Rf_xlength(shape_int);

  uint32_t *u_shape = (uint32_t *)malloc(ndim * sizeof(uint32_t));
  for (R_xlen_t i = 0; i < ndim; i++) {
    u_shape[i] = (uint32_t)s_ptr[i];
  }

  // Call Rust
  struct NumRsArray *arr = numrs_array_new(f_data, u_shape, ndim);

  // Cleanup
  free(f_data);
  free(u_shape);
  UNPROTECT(1); // shape_int

  // Wrap in ExternalPtr
  SEXP res = create_s3_ptr(arr, tag_array, "NumRsArray", finalizer_array);
  return res;
}

SEXP r_numrs_array_print(SEXP ptr) {
  struct NumRsArray *arr = (struct NumRsArray *)R_ExternalPtrAddr(ptr);
  if (!arr)
    Rf_error("Invalid Array pointer");
  numrs_array_print(arr);
  return R_NilValue;
}

SEXP r_numrs_array_shape(SEXP ptr) {
  // Robust check for invalid/cleared pointer
  if (TYPEOF(ptr) != EXTPTRSXP || !R_ExternalPtrAddr(ptr)) {
    return R_NilValue;
  }

  struct NumRsArray *arr = (struct NumRsArray *)R_ExternalPtrAddr(ptr);
  // arr check is redundant with above but safe
  if (!arr)
    return R_NilValue;

  uintptr_t ndim = numrs_array_ndim(arr);
  SEXP shape = PROTECT(Rf_allocVector(INTSXP, ndim));
  int *s_ptr = INTEGER(shape);

  // numrs_array_shape writes to u32 buffer
  uint32_t *u_shape = (uint32_t *)malloc(ndim * sizeof(uint32_t));
  numrs_array_shape(arr, u_shape, NULL);

  for (uintptr_t i = 0; i < ndim; i++) {
    s_ptr[i] = (int)u_shape[i];
  }

  free(u_shape);
  UNPROTECT(1);
  return shape;
}

SEXP r_numrs_array_to_r(SEXP ptr) {
  struct NumRsArray *arr = (struct NumRsArray *)R_ExternalPtrAddr(ptr);
  if (!arr)
    Rf_error("Invalid Array pointer");

  uintptr_t ndim = numrs_array_ndim(arr);

  // Calculate total elements
  R_xlen_t numel = 1;
  uint32_t *u_shape = (uint32_t *)malloc(ndim * sizeof(uint32_t));
  numrs_array_shape(arr, u_shape, NULL);
  for (uintptr_t i = 0; i < ndim; i++) {
    numel *= u_shape[i];
  }

  // Allocate R vector
  SEXP res = PROTECT(Rf_allocVector(REALSXP, numel));
  double *r_ptr = REAL(res);

  // Get raw float data
  float *c_ptr = numrs_array_data(arr);

  // Copy cast
  for (R_xlen_t i = 0; i < numel; i++) {
    r_ptr[i] = (double)c_ptr[i];
  }

  free(u_shape);
  UNPROTECT(1);
  return res;
}
