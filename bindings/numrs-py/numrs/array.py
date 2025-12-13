from .native_lib import lib, NumRsArray, c_float_p, c_uint32_p, c_size_t
from .types import DType
import ctypes

class Array:
    """Wrapper around NumRsArray C opaque pointer."""
    def __init__(self, data: object = None, shape: object = None, dtype: DType = "float32") -> None:
        self._ptr: ctypes.POINTER(NumRsArray) = None # type: ignore
        if dtype != "float32":
            raise ValueError(f"Unsupported dtype: {dtype}. Only 'float32' is currently supported.")
        if data is not None:
            if shape is None:
                shape = [len(data)]
                
            # Flatten data if it's nested (simple implementation)
            # For now assume flat list of floats or single list
            # Ideally we'd use recursion or numpy flat
            
    def __init__(self, data: object = None, shape: object = None, dtype: DType = "float32") -> None:
        self._ptr: ctypes.POINTER(NumRsArray) = None # type: ignore
        if dtype != "float32":
            raise ValueError(f"Unsupported dtype: {dtype}. Only 'float32' is currently supported.")
        if data is not None:
            if shape is None:
                shape = [len(data)]
            
            flat_data = []
            is_int = True
            is_float = False
            
            def flatten(l):
                nonlocal is_int, is_float
                for el in l:
                    if isinstance(el, list):
                        flatten(el)
                    else:
                        if isinstance(el, float):
                            is_float = True
                            is_int = False
                        elif not isinstance(el, int):
                            # Treat unknown as float
                            is_float = True
                            is_int = False
                        flat_data.append(el)
                        
            flatten(data)
            
            c_shape = (ctypes.c_uint32 * len(shape))(*shape)
            
            if is_int:
               # Use i32 converter
               c_data = (ctypes.c_int * len(flat_data))(*[int(x) for x in flat_data])
               self._ptr = lib.numrs_array_from_i32(c_data, c_shape, len(shape))
            else:
               # Default to f64 source (standard python float) converting to f32
               c_data = (ctypes.c_double * len(flat_data))(*[float(x) for x in flat_data])
               self._ptr = lib.numrs_array_from_f64(c_data, c_shape, len(shape))
            
    def _from_ptr(ptr: ctypes.POINTER(NumRsArray)) -> 'Array':
        obj = Array.__new__(Array)
        obj._ptr = ptr
        return obj

    def __del__(self):
        if self._ptr:
            lib.numrs_array_free(self._ptr)
            self._ptr = None

    def __repr__(self):
        # We can implement a better repr by capturing stdout from c print 
        # or implementing a to_list C function. For now print to stdout via C.
        if self._ptr:
            print("C-Side Print:", end=" ")
            lib.numrs_array_print(self._ptr)
            return "<NumRs Array>"
        return "<NumRs Array (Valid)>"
    @classmethod
    def zeros(cls, shape: list[int]) -> 'Array':
        c_shape = (ctypes.c_uint32 * len(shape))(*shape)
        ptr = lib.numrs_array_zeros(c_shape, len(shape))
        return cls._from_ptr(ptr)

    @classmethod
    def ones(cls, shape: list[int]) -> 'Array':
        c_shape = (ctypes.c_uint32 * len(shape))(*shape)
        ptr = lib.numrs_array_ones(c_shape, len(shape))
        return cls._from_ptr(ptr)

    @property
    def shape(self) -> list[int]:
        if not self._ptr:
            return []
        ndim = lib.numrs_array_ndim(self._ptr)
        c_shape = (ctypes.c_uint32 * ndim)()
        c_ndim = c_size_t(0)
        lib.numrs_array_shape(self._ptr, c_shape, ctypes.byref(c_ndim))
        return list(c_shape)
    
    @property
    def data(self) -> list[float]:
        if not self._ptr:
            return []
        # Return copy of data for safety
        # We need size. Product of shape.
        shape = self.shape
        size = 1
        for dim in shape:
            size *= dim
            
        data_ptr = lib.numrs_array_data(self._ptr)
        if not data_ptr:
            return []
        # Cast to array and convert to list
        return list((ctypes.c_float * size).from_address(ctypes.addressof(data_ptr.contents)))

    def reshape(self, shape: list[int]) -> 'Array':
        if not self._ptr:
            raise RuntimeError("Array is null")
        c_shape = (ctypes.c_uint32 * len(shape))(*shape)
        new_ptr = lib.numrs_array_reshape(self._ptr, c_shape, len(shape))
        if not new_ptr:
            raise RuntimeError("Reshape failed (incompatible shape?)")
        return Array._from_ptr(new_ptr)

    def print(self):
        if self._ptr:
            lib.numrs_array_print(self._ptr)

    def __add__(self, other: 'Array') -> 'Array':
        if not isinstance(other, Array):
            raise TypeError("Operands must be Arrays")
        ptr = lib.numrs_array_add(self._ptr, other._ptr)
        if not ptr: raise RuntimeError("Add failed")
        return Array._from_ptr(ptr)
        
    def __sub__(self, other: 'Array') -> 'Array':
        if not isinstance(other, Array):
             raise TypeError("Operands must be Arrays")
        ptr = lib.numrs_array_sub(self._ptr, other._ptr)
        if not ptr: raise RuntimeError("Sub failed")
        return Array._from_ptr(ptr)
        
    def __mul__(self, other: 'Array') -> 'Array':
        if not isinstance(other, Array):
             raise TypeError("Operands must be Arrays")
        ptr = lib.numrs_array_mul(self._ptr, other._ptr)
        if not ptr: raise RuntimeError("Mul failed")
        return Array._from_ptr(ptr)
        
    def __truediv__(self, other: 'Array') -> 'Array':
        if not isinstance(other, Array):
             raise TypeError("Operands must be Arrays")
        ptr = lib.numrs_array_div(self._ptr, other._ptr)
        if not ptr: raise RuntimeError("Div failed")
        return Array._from_ptr(ptr)
        
    def __matmul__(self, other: 'Array') -> 'Array':
        if not isinstance(other, Array):
             raise TypeError("Operands must be Arrays")
        ptr = lib.numrs_array_matmul(self._ptr, other._ptr)
        if not ptr: raise RuntimeError("Matmul failed")
        return Array._from_ptr(ptr)
        
    def matmul(self, other: 'Array') -> 'Array':
        return self.__matmul__(other)
