from ..native_lib import lib
from ..tensor import Tensor

def relu(t):
    return Tensor._from_ptr(lib.numrs_relu(t._ptr))

def sigmoid(t):
    return Tensor._from_ptr(lib.numrs_sigmoid(t._ptr))

def log(t):
    return Tensor._from_ptr(lib.numrs_log(t._ptr))

def exp(t):
    return Tensor._from_ptr(lib.numrs_exp(t._ptr))

def sqrt(t):
    return Tensor._from_ptr(lib.numrs_sqrt(t._ptr))

def neg(t):
    return Tensor._from_ptr(lib.numrs_neg(t._ptr))

def abs(t):
    return Tensor._from_ptr(lib.numrs_abs(t._ptr))

def tanh(t):
    return Tensor._from_ptr(lib.numrs_tanh(t._ptr))

def sin(t):
    return Tensor._from_ptr(lib.numrs_sin(t._ptr))

def cos(t):
    return Tensor._from_ptr(lib.numrs_cos(t._ptr))

def tan(t):
    return Tensor._from_ptr(lib.numrs_tan(t._ptr))

def asin(t):
    return Tensor._from_ptr(lib.numrs_asin(t._ptr))

def acos(t):
    return Tensor._from_ptr(lib.numrs_acos(t._ptr))

def atan(t):
    return Tensor._from_ptr(lib.numrs_atan(t._ptr))

def softplus(t):
    return Tensor._from_ptr(lib.numrs_softplus(t._ptr))

def leaky_relu(t):
    return Tensor._from_ptr(lib.numrs_leaky_relu(t._ptr))
