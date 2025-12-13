#include "numrs_r.h"

SEXP r_numrs_dataset_new(SEXP inputs, SEXP inputs_shape, SEXP targets,
                         SEXP targets_shape, SEXP batch_size) {
  // Inputs (float array)
  const double *in_ptr = REAL(inputs);
  R_xlen_t in_len = Rf_xlength(inputs);
  float *f_in = (float *)malloc(in_len * sizeof(float));
  for (R_xlen_t i = 0; i < in_len; i++)
    f_in[i] = (float)in_ptr[i];

  // Inputs Shape
  SEXP in_shape_int = PROTECT(Rf_coerceVector(inputs_shape, INTSXP));
  int *is_ptr = INTEGER(in_shape_int);
  R_xlen_t in_ndim = Rf_xlength(in_shape_int);
  uint32_t *u_in_shape = (uint32_t *)malloc(in_ndim * sizeof(uint32_t));
  for (R_xlen_t i = 0; i < in_ndim; i++)
    u_in_shape[i] = (uint32_t)is_ptr[i];

  // Targets
  const double *t_ptr = REAL(targets);
  R_xlen_t t_len = Rf_xlength(targets);
  float *f_t = (float *)malloc(t_len * sizeof(float));
  for (R_xlen_t i = 0; i < t_len; i++)
    f_t[i] = (float)t_ptr[i];

  // Targets Shape
  SEXP t_shape_int = PROTECT(Rf_coerceVector(targets_shape, INTSXP));
  int *ts_ptr = INTEGER(t_shape_int);
  R_xlen_t t_ndim = Rf_xlength(t_shape_int);
  uint32_t *u_t_shape = (uint32_t *)malloc(t_ndim * sizeof(uint32_t));
  for (R_xlen_t i = 0; i < t_ndim; i++)
    u_t_shape[i] = (uint32_t)ts_ptr[i];

  uintptr_t bs = (uintptr_t)Rf_asInteger(batch_size);

  struct NumRsDataset *ds =
      numrs_dataset_new(f_in, u_in_shape, in_ndim, f_t, u_t_shape, t_ndim, bs);

  free(f_in);
  free(u_in_shape);
  free(f_t);
  free(u_t_shape);
  UNPROTECT(2); // shapes

  SEXP res = PROTECT(R_MakeExternalPtr(ds, tag_dataset, R_NilValue));
  R_RegisterCFinalizerEx(res, finalizer_dataset, TRUE);
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_trainer_builder_new(SEXP seq_ptr) {
  struct NumRsSequential *seq =
      (struct NumRsSequential *)R_ExternalPtrAddr(seq_ptr);
  struct NumRsTrainerBuilder *b = numrs_trainer_builder_new(seq);
  SEXP res = PROTECT(R_MakeExternalPtr(b, tag_trainer_builder, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_trainer_builder_learning_rate(SEXP b_ptr, SEXP lr) {
  struct NumRsTrainerBuilder *b =
      (struct NumRsTrainerBuilder *)R_ExternalPtrAddr(b_ptr);
  double l = Rf_asReal(lr);
  struct NumRsTrainerBuilder *b2 = numrs_trainer_builder_learning_rate(b, l);
  SEXP res = PROTECT(R_MakeExternalPtr(b2, tag_trainer_builder, R_NilValue));
  UNPROTECT(1);
  return res;
}

SEXP r_numrs_trainer_build_sgd_mse(SEXP b_ptr) {
  struct NumRsTrainerBuilder *b =
      (struct NumRsTrainerBuilder *)R_ExternalPtrAddr(b_ptr);
  struct NumRsTrainer *t = numrs_trainer_build(b, "sgd", "mse");
  SEXP res = create_s3_ptr(t, tag_trainer, "NumRsTrainer", finalizer_trainer);
  return res;
}

SEXP r_numrs_trainer_fit_run(SEXP t_ptr, SEXP d_ptr, SEXP epochs) {
  struct NumRsTrainer *t = (struct NumRsTrainer *)R_ExternalPtrAddr(t_ptr);
  struct NumRsDataset *d = (struct NumRsDataset *)R_ExternalPtrAddr(d_ptr);
  uintptr_t e = (uintptr_t)Rf_asInteger(epochs);
  numrs_trainer_fit(t, d, e);
  return R_NilValue;
}

SEXP r_numrs_trainer_build(SEXP b_ptr, SEXP optimizer_name, SEXP loss_name) {
  struct NumRsTrainerBuilder *b =
      (struct NumRsTrainerBuilder *)R_ExternalPtrAddr(b_ptr);

  const char *opt_str = CHAR(STRING_ELT(optimizer_name, 0));
  const char *loss_str = CHAR(STRING_ELT(loss_name, 0));

  struct NumRsTrainer *t = numrs_trainer_build(b, opt_str, loss_str);

  if (!t)
    return R_NilValue; // Build failed (unknown optimizer/loss)

  SEXP res = create_s3_ptr(t, tag_trainer, "NumRsTrainer", finalizer_trainer);
  return res;
}
