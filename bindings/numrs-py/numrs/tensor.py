from .native_lib import lib, NumRsTensor, c_bool
from .types import DType
from .array import Array
import ctypes
import math

class Tensor:
    """Wrapper around NumRsTensor C opaque pointer."""
    def __init__(self, data: object, requires_grad: bool = False, dtype: DType = "float32", _ptr: ctypes.POINTER(NumRsTensor) = None) -> None:
        """
        Create a new Tensor.
        Args:
            data: List of floats/Arrays or Array object.
            requires_grad: bool, whether to track gradients.
            dtype: Data type of the tensor (currently only "float32").
            _ptr: ctypes pointer (internal use only).
        """
        if _ptr:
            self._ptr = _ptr
            return

        array = None
        if isinstance(data, Array):
            array = data
        else:
            # Assume list
            array = Array(data, dtype=dtype)
            
        # Note: Tensor consumes the Array in C logic usually (takes ownership),
        # but our specific C binding numrs_tensor_new clones the array data internally.
        # So it is safe to let `array` Python object die or persist.
        
        self._ptr = lib.numrs_tensor_new(array._ptr, requires_grad)
        
    def _from_ptr(ptr: ctypes.POINTER(NumRsTensor)) -> 'Tensor':
        if not ptr:
            return None
        obj = Tensor.__new__(Tensor)
        obj._ptr = ptr
        return obj

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.numrs_tensor_free(self._ptr)
            self._ptr = None

    def backward(self) -> None:
        lib.numrs_tensor_backward(self._ptr)
        
    def data(self) -> Array:
        """Return a copy of the underlying Array data."""
        arr_ptr = lib.numrs_tensor_data(self._ptr)
        return Array._from_ptr(arr_ptr)
    
    @property
    def grad(self) -> 'Tensor':
        """Return the gradient as a Tensor, or None."""
        grad_ptr = lib.numrs_tensor_grad(self._ptr)
        if not grad_ptr:
            return None
        return Tensor._from_ptr(grad_ptr)
        
    @property
    def shape(self) -> tuple[int]:
        """Return the shape of the tensor."""
        ndim = ctypes.c_size_t(0)
        shape_ptr = lib.numrs_tensor_shape(self._ptr, ctypes.byref(ndim))
        if not shape_ptr:
            return ()
        # Convert C array to tuple
        # shape_ptr is *const usize.
        # We need to cast it to array of correct size?
        # ctypes provides slicing for pointer?
        # No, cast to POINTER(c_size_t * ndim.value) then .contents?
        # Easier: shape_ptr[i] indexing if pointer type supports it?
        # shape_ptr return type was defined as POINTER(c_size_t) in native_lib.
        # So we can iterate/list conversion.
        return tuple(shape_ptr[i] for i in range(ndim.value))
        
    def item(self) -> float:
        """Return the value of this tensor as a standard Python number. Only works for 1-element tensors."""
        if math.prod(self.shape) != 1:
             # Or check if shape is empty or all 1s?
             # For now rely on C implementation or just return value.
             pass
        return lib.numrs_tensor_item(self._ptr)
        
    def detach(self) -> 'Tensor':
        """Returns a new Tensor, detached from the current graph."""
        ptr = lib.numrs_tensor_detach(self._ptr)
        return Tensor._from_ptr(ptr)
        
    def zero_grad(self) -> None:
        """Sets the gradient of the tensor to zero."""
        lib.numrs_tensor_zero_grad(self._ptr)

    def transpose(self) -> 'Tensor':
        """Returns a tensor that is a transposed version of input."""
        ptr = lib.numrs_transpose(self._ptr)
        return Tensor._from_ptr(ptr)

    def reshape(self, shape: tuple[int] | list[int]) -> 'Tensor':
        """Reshapes the tensor to the specified shape."""
        # Convert shape to C array
        shape_len = len(shape)
        # Use c_size_t for usize
        ShapeArray = ctypes.c_size_t * shape_len
        c_shape = ShapeArray(*shape)
        
        ptr = lib.numrs_tensor_reshape(self._ptr, c_shape, shape_len)
        return Tensor._from_ptr(ptr)

    def flatten(self) -> 'Tensor':
        """Flattens the tensor into a 1D tensor."""
        ptr = lib.numrs_tensor_flatten(self._ptr)
        return Tensor._from_ptr(ptr)
    
    # Alias
    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    # --- Operators ---

    def __add__(self, other: 'Tensor') -> 'Tensor':
        return self._binary_op(lib.numrs_add, other)

    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return self._binary_op(lib.numrs_sub, other)

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        return self._binary_op(lib.numrs_mul, other)

    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        return self._binary_op(lib.numrs_div, other)
        
    def __pow__(self, exponent: float) -> 'Tensor':
        # Specific binding for pow(tensor, scalar)
        if not isinstance(exponent, (int, float)):
             raise TypeError("Exponent must be a scalar")
        out_ptr = lib.numrs_pow(self._ptr, float(exponent))
        return Tensor._from_ptr(out_ptr)
        
    def matmul(self, other: 'Tensor') -> 'Tensor':
        return self._binary_op(lib.numrs_matmul, other)
        
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)

    def _binary_op(self, c_fn: object, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            # TODO: Support scalar broadcasting implicitly? 
            # For now require Tensor.
            raise TypeError("Operands must be Tensors")
            
        out_ptr = c_fn(self._ptr, other._ptr)
        return Tensor._from_ptr(out_ptr)
