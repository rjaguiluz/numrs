from ..native_lib import lib, c_uint32_p
from ..tensor import Tensor
import ctypes

def reshape(t, shape):
    c_shape = (ctypes.c_uint32 * len(shape))(*shape)
    return Tensor._from_ptr(lib.numrs_reshape(t._ptr, c_shape, len(shape)))

def flatten(t, start_dim=0, end_dim=-1):
    return Tensor._from_ptr(lib.numrs_flatten(t._ptr, start_dim, end_dim))

def concat(tensors, axis=0):
    # tensors: list of Tensor objects
    c_tensors = (ctypes.POINTER(NumRsTensor) * len(tensors))()
    for i, t in enumerate(tensors):
        c_tensors[i] = t._ptr
    return Tensor._from_ptr(lib.numrs_concat(c_tensors, len(tensors), axis))

def broadcast_to(t, shape):
    c_shape = (ctypes.c_uint32 * len(shape))(*shape)
    return Tensor._from_ptr(lib.numrs_broadcast_to(t._ptr, c_shape, len(shape)))
