from ..native_lib import lib
from ..tensor import Tensor

def sum(t, axis=-1):
    return Tensor._from_ptr(lib.numrs_sum(t._ptr, axis))

def mean(t, axis=-1):
    return Tensor._from_ptr(lib.numrs_mean(t._ptr, axis))
    
def mse_loss(pred, target):
    return Tensor._from_ptr(lib.numrs_mse_loss(pred._ptr, target._ptr))

def softmax(t, axis=-1):
    return Tensor._from_ptr(lib.numrs_softmax(t._ptr, axis))

def max(t, axis=-1):
    return Tensor._from_ptr(lib.numrs_max(t._ptr, axis))

def min(t, axis=-1):
    return Tensor._from_ptr(lib.numrs_min(t._ptr, axis))

def argmax(t, axis=-1):
    return Tensor._from_ptr(lib.numrs_argmax(t._ptr, axis))

def variance(t, axis=-1):
    return Tensor._from_ptr(lib.numrs_variance(t._ptr, axis))

def norm(t):
    return Tensor._from_ptr(lib.numrs_norm(t._ptr))
