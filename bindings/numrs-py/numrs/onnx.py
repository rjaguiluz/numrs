from .native_lib import lib, NumRsOnnxModel
from .tensor import Tensor
from .array import Array
import ctypes

def save_onnx(tensor: Tensor, path: str) -> None:
    """
    Exports the computation graph ending at `tensor` to an ONNX file.
    
    Args:
        tensor: The output tensor of the graph to export.
        path: Destination file path (e.g. "model.onnx").
    """
    if not isinstance(tensor, Tensor):
         raise TypeError("First argument must be a Tensor")
         
    c_path = path.encode('utf-8')
    res = lib.numrs_onnx_export(tensor._ptr, c_path)
    if res != 0:
        raise RuntimeError(f"Failed to export ONNX to {path} (Error code: {res})")

class OnnxModel:
    """Wrapper for a loaded ONNX model (inference only)."""
    def __init__(self, path: str):
        c_path = path.encode('utf-8')
        ptr = lib.numrs_onnx_load(c_path)
        if not ptr:
             raise RuntimeError(f"Failed to load ONNX model from {path}")
        self._ptr = ptr
        
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.numrs_onnx_free(self._ptr)
            
    def infer(self, input_data: Array, input_name: str = "input") -> Array:
        """
        Run inference. currently supports single input/output.
        """
        if not isinstance(input_data, Array):
            # Try to convert
            input_data = Array(input_data)
            
        c_name = input_name.encode('utf-8')
        out_ptr = lib.numrs_onnx_infer_simple(self._ptr, input_data._ptr, c_name)
        
        if not out_ptr:
             raise RuntimeError("Inference failed")
             
        return Array._from_ptr(out_ptr)
