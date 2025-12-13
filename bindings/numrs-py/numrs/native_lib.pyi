import ctypes
from typing import Any, List, Optional, Type, Union

# Base types
c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint32_p = ctypes.POINTER(ctypes.c_uint32)
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_bool = ctypes.c_bool
c_size_t = ctypes.c_size_t

# Opaque structs
class NumRsArray(ctypes.Structure): ...
class NumRsTensor(ctypes.Structure): ...
class NumRsSequential(ctypes.Structure): ...
class NumRsLinear(ctypes.Structure): ...
class NumRsReLU(ctypes.Structure): ...
class NumRsConv1d(ctypes.Structure): ...
class NumRsDataset(ctypes.Structure): ...
class NumRsTrainer(ctypes.Structure): ...
class NumRsTrainerBuilder(ctypes.Structure): ...

class Lib:
    numrs_version: ctypes._FuncPointer
    numrs_print_startup_log: ctypes._FuncPointer

    # Array
    numrs_array_new: ctypes._FuncPointer
    numrs_array_free: ctypes._FuncPointer
    numrs_array_print: ctypes._FuncPointer

    # Tensor
    numrs_tensor_new: ctypes._FuncPointer
    numrs_tensor_free: ctypes._FuncPointer
    numrs_tensor_data: ctypes._FuncPointer
    numrs_tensor_grad: ctypes._FuncPointer
    numrs_tensor_backward: ctypes._FuncPointer

    # Ops (Binary)
    numrs_add: ctypes._FuncPointer
    numrs_sub: ctypes._FuncPointer
    numrs_mul: ctypes._FuncPointer
    numrs_div: ctypes._FuncPointer
    numrs_pow: ctypes._FuncPointer
    numrs_matmul: ctypes._FuncPointer
    numrs_mse_loss: ctypes._FuncPointer

    # Ops (Unary)
    numrs_relu: ctypes._FuncPointer
    numrs_sigmoid: ctypes._FuncPointer
    numrs_log: ctypes._FuncPointer
    numrs_exp: ctypes._FuncPointer

    # Ops (Reduction)
    numrs_sum: ctypes._FuncPointer
    numrs_mean: ctypes._FuncPointer
    numrs_max: ctypes._FuncPointer
    numrs_min: ctypes._FuncPointer
    numrs_softmax: ctypes._FuncPointer

    # Ops (Shape)
    numrs_reshape: ctypes._FuncPointer
    numrs_flatten: ctypes._FuncPointer

    # NN
    numrs_linear_new: ctypes._FuncPointer
    numrs_relu_layer_new: ctypes._FuncPointer
    numrs_conv1d_new: ctypes._FuncPointer
    numrs_sequential_new: ctypes._FuncPointer
    numrs_sequential_free: ctypes._FuncPointer
    numrs_sequential_add_linear: ctypes._FuncPointer
    numrs_sequential_add_relu: ctypes._FuncPointer
    numrs_sequential_forward: ctypes._FuncPointer

    # Train
    numrs_dataset_new: ctypes._FuncPointer
    numrs_dataset_free: ctypes._FuncPointer
    numrs_trainer_builder_new: ctypes._FuncPointer
    numrs_trainer_builder_learning_rate: ctypes._FuncPointer
    numrs_trainer_build: ctypes._FuncPointer
    numrs_trainer_fit: ctypes._FuncPointer
    numrs_trainer_free: ctypes._FuncPointer

lib: Lib
