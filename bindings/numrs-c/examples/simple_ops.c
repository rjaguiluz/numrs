#include "../include/numrs.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  printf("NumRs C Example: Simple Ops (Numpy Style)\n");
  numrs_print_startup_log();

  // 1. Create Data (Numpy-style)
  // Array A: [1.0, 2.0, 3.0, 4.0] shape (2, 2)
  float data_a[] = {1.0, 2.0, 3.0, 4.0};
  uint32_t shape[] = {2, 2};
  size_t ndim = 2;

  printf("\nCreating Array A:\n");
  NumRsArray *arr_a = numrs_array_new(data_a, shape, ndim);
  numrs_array_print(arr_a);

  // Array B: [10.0, 20.0, 30.0, 40.0] shape (2, 2)
  float data_b[] = {10.0, 20.0, 30.0, 40.0};
  printf("\nCreating Array B:\n");
  NumRsArray *arr_b = numrs_array_new(data_b, shape, ndim);
  numrs_array_print(arr_b);

  // 2. Convert to Tensors for Operations
  NumRsTensor *t_a =
      numrs_tensor_new(arr_a, false); // false = no autograd needed
  NumRsTensor *t_b = numrs_tensor_new(arr_b, false);

  // 3. Perform Operations
  printf("\nCalculating C = A + B:\n");
  NumRsTensor *t_c = numrs_add(t_a, t_b);
  NumRsArray *arr_c = numrs_tensor_data(t_c);
  numrs_array_print(arr_c);

  printf("\nCalculating D = A * B:\n");
  NumRsTensor *t_d = numrs_mul(t_a, t_b);
  NumRsArray *arr_d = numrs_tensor_data(t_d);
  numrs_array_print(arr_d);

  // 4. Cleanup
  // Note: numrs_tensor_new in C bindings typically clones the array data into
  // the tensor, so we should free the original arrays if we are done with them.
  // However, looking at tensor.rs: `let tensor = Tensor::new(arr.clone(), ...)`
  // Yes, it clones. So we own arr_a, arr_b, and can free them anytime.

  numrs_array_free(arr_a);
  numrs_array_free(arr_b);

  numrs_tensor_free(t_a);
  numrs_tensor_free(t_b);

  numrs_array_free(
      arr_c); // Data extracted from tensor is a new copy? Check tensor.rs
  // tensor.rs: `Box::new(NumRsArray { inner: arr })` where `arr` is cloned from
  // tensor data. Yes, numrs_tensor_data returns a new opaque handle to cloned
  // data.
  numrs_tensor_free(t_c);

  numrs_array_free(arr_d);
  numrs_tensor_free(t_d);

  printf("\nDone!\n");
  return 0;
}
