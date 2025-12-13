from .native_lib import (
    lib, NumRsSequential, NumRsLinear, NumRsReLU, NumRsConv1d,
    NumRsSigmoid, NumRsSoftmax, NumRsDropout, NumRsFlatten, NumRsBatchNorm1d
)
from .tensor import Tensor
import ctypes

class Module:
    pass

class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        self._ptr = lib.numrs_linear_new(in_features, out_features)

class ReLU(Module):
    def __init__(self) -> None:
        self._ptr = lib.numrs_relu_layer_new()

class Conv1d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        self._ptr = lib.numrs_conv1d_new(in_channels, out_channels, kernel_size, stride, padding)

class Sequential(Module):
    def __init__(self) -> None:
        self._ptr = lib.numrs_sequential_new()
        
    def add(self, layer: Module) -> None:
        if isinstance(layer, Linear):
            lib.numrs_sequential_add_linear(self._ptr, layer._ptr)
        elif isinstance(layer, ReLU):
            lib.numrs_sequential_add_relu(self._ptr, layer._ptr)
        elif isinstance(layer, Conv1d):
            lib.numrs_sequential_add_conv1d(self._ptr, layer._ptr)
        elif isinstance(layer, Sigmoid):
            lib.numrs_sequential_add_sigmoid(self._ptr, layer._ptr)
        elif isinstance(layer, Softmax):
            lib.numrs_sequential_add_softmax(self._ptr, layer._ptr)
        elif isinstance(layer, Dropout):
            lib.numrs_sequential_add_dropout(self._ptr, layer._ptr)
        elif isinstance(layer, Flatten):
            lib.numrs_sequential_add_flatten(self._ptr, layer._ptr)
        elif isinstance(layer, BatchNorm1d):
            lib.numrs_sequential_add_batchnorm1d(self._ptr, layer._ptr)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")
            
    def forward(self, x: Tensor) -> Tensor:
        out_ptr = lib.numrs_sequential_forward(self._ptr, x._ptr)
        return Tensor._from_ptr(out_ptr)
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
        
        if hasattr(self, '_ptr') and self._ptr:
            lib.numrs_sequential_free(self._ptr)

class Sigmoid(Module):
    def __init__(self) -> None:
        self._ptr = lib.numrs_sigmoid_new()

class Softmax(Module):
    def __init__(self) -> None:
        self._ptr = lib.numrs_softmax_new()

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        self._ptr = lib.numrs_dropout_new(p)

class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        # Resolve -1 to usize::MAX or similar logic handled in C? 
        # C binding expects usize. Python -1 usually means "end".
        # numrs-core/src/ops/export.rs line 213 check: *end_dim == usize::MAX
        # So pass -1 as usize::MAX? Or convert here.
        # But ctypes c_size_t is unsigned. -1 will overflow/wrap to MAX.
        # So passing -1 (as int) via c_size_t should work if struct.pack/ctypes handles it.
        # But explicitly:
        s = start_dim
        e = end_dim if end_dim != -1 else (2**64 - 1) # Assuming 64-bit usize? Or rely on ctypes cast.
        # ctypes size_t cast of -1 usually gives MAX.
        self._ptr = lib.numrs_flatten_new(start_dim, end_dim)

class BatchNorm1d(Module):
    def __init__(self, num_features: int) -> None:
        self._ptr = lib.numrs_batchnorm1d_new(num_features)

