from ..native_lib import lib
from ..tensor import Tensor

def dot(a, b):
    # a and b are Tensors
    return Tensor._from_ptr(lib.numrs_dot(a._ptr, b._ptr))
