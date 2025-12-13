import ctypes
import os
import sys
import platform

# Define base types
c_float_p = ctypes.POINTER(ctypes.c_float)
c_uint32_p = ctypes.POINTER(ctypes.c_uint32)
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_bool = ctypes.c_bool
c_size_t = ctypes.c_size_t

# Opaque structs
class NumRsArray(ctypes.Structure):
    pass

class NumRsTensor(ctypes.Structure):
    pass

class NumRsSequential(ctypes.Structure):
    pass

class NumRsLinear(ctypes.Structure):
    pass

class NumRsReLU(ctypes.Structure):
    pass

class NumRsConv1d(ctypes.Structure):
    pass

class NumRsDataset(ctypes.Structure):
    pass

class NumRsTrainer(ctypes.Structure):
    pass

class NumRsTrainerBuilder(ctypes.Structure):
    pass

class NumRsSigmoid(ctypes.Structure):
    pass

class NumRsSoftmax(ctypes.Structure):
    pass

class NumRsDropout(ctypes.Structure):
    pass

class NumRsFlatten(ctypes.Structure):
    pass

class NumRsBatchNorm1d(ctypes.Structure):
    pass

# Helper to load library
def load_lib():
    # Try to find the library in potential build paths
    # Assuming we are in numrs-cy/numrs and the lib is in numrs-c/target/debug/deps or similar
    
    # Common paths relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.dirname(base_dir) # dev/numrs/
    
    lib_name = "libnumrs_c.dylib" if platform.system() == "Darwin" else "libnumrs_c.so"
    if platform.system() == "Windows":
        lib_name = "numrs_c.dll"

    # Search paths
    paths = [
        # 1. Bundled in package (Wheel / pip install)
        os.path.join(base_dir, lib_name),
        # 2. Local Development (Repositories)
        os.path.join(root_dir, "bindings", "numrs-c", "target", "debug", lib_name),
        os.path.join(root_dir, "bindings", "numrs-c", "target", "release", lib_name),
        os.path.join(root_dir, "target", "debug", lib_name), # If workspace build
    ]
    
    lib_path = None
    for p in paths:
        if os.path.exists(p):
            lib_path = p
            print(f"DEBUG: Loading library from {lib_path}")
            break
            
    if not lib_path:
        raise FileNotFoundError(f"Could not find {lib_name} in {paths}. Please build numrs-c first.")
        
    return ctypes.CDLL(lib_path)

lib = load_lib()

# ============================================================================
# Function Signatures
# ============================================================================

# Utils
lib.numrs_version.restype = c_char_p
lib.numrs_print_startup_log.restype = None

# Array
lib.numrs_array_new.argtypes = [c_float_p, c_uint32_p, c_size_t]
lib.numrs_array_new.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_free.argtypes = [ctypes.POINTER(NumRsArray)]
lib.numrs_array_free.restype = None

lib.numrs_array_print.argtypes = [ctypes.POINTER(NumRsArray)]
lib.numrs_array_print.restype = None

# Tensor
lib.numrs_tensor_new.argtypes = [ctypes.POINTER(NumRsArray), c_bool]
lib.numrs_tensor_new.restype = ctypes.POINTER(NumRsTensor)

lib.numrs_tensor_free.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_tensor_free.restype = None

lib.numrs_tensor_data.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_tensor_data.restype = ctypes.POINTER(NumRsArray)

lib.numrs_tensor_grad.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_tensor_grad.restype = ctypes.POINTER(NumRsTensor)

lib.numrs_tensor_backward.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_tensor_backward.restype = None

lib.numrs_tensor_reshape.argtypes = [ctypes.POINTER(NumRsTensor), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
lib.numrs_tensor_reshape.restype = ctypes.POINTER(NumRsTensor)

lib.numrs_tensor_flatten.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_tensor_flatten.restype = ctypes.POINTER(NumRsTensor)

# Ops (Binary)
for op in ['add', 'sub', 'mul', 'div', 'pow', 'matmul', 'mse_loss']:
    fn_name = f"numrs_{op}"
    if hasattr(lib, fn_name):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.POINTER(NumRsTensor), ctypes.POINTER(NumRsTensor)]
        fn.restype = ctypes.POINTER(NumRsTensor)
        
# Manual override for pow (second arg is scalar in simple binding? No, C API says tensor)
# C API: struct NumRsTensor *numrs_pow(struct NumRsTensor *a, double exponent);
lib.numrs_pow.argtypes = [ctypes.POINTER(NumRsTensor), ctypes.c_double]
lib.numrs_pow.restype = ctypes.POINTER(NumRsTensor)


# Ops (Unary)
for op in ['relu', 'sigmoid', 'log', 'exp', 'sqrt', 'neg', 'abs', 'tanh', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'softplus']:
    fn_name = f"numrs_{op}"
    if hasattr(lib, fn_name):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ctypes.POINTER(NumRsTensor)]
        fn.restype = ctypes.POINTER(NumRsTensor)

lib.numrs_leaky_relu.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_leaky_relu.restype = ctypes.POINTER(NumRsTensor)

# Ops (Reduction) - axis is intptr_t
for op in ['sum', 'mean', 'max', 'min', 'argmax', 'variance']: 
    fn_name = f"numrs_{op}"
    if hasattr(lib, fn_name):
        fn = getattr(lib, fn_name)
        # Assuming axis argument. C API has (tensor, axis).
        fn.argtypes = [ctypes.POINTER(NumRsTensor), ctypes.c_longlong] 
        fn.restype = ctypes.POINTER(NumRsTensor)

lib.numrs_norm.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_norm.restype = ctypes.POINTER(NumRsTensor)

lib.numrs_softmax.argtypes = [ctypes.POINTER(NumRsTensor), ctypes.c_longlong]
lib.numrs_softmax.restype = ctypes.POINTER(NumRsTensor)

# Ops (Linalg)
lib.numrs_dot.argtypes = [ctypes.POINTER(NumRsTensor), ctypes.POINTER(NumRsTensor)]
lib.numrs_dot.restype = ctypes.POINTER(NumRsTensor)
        
# Ops (Shape)


lib.numrs_concat.argtypes = [ctypes.POINTER(ctypes.POINTER(NumRsTensor)), c_size_t, c_size_t]
lib.numrs_concat.restype = ctypes.POINTER(NumRsTensor)

lib.numrs_broadcast_to.argtypes = [ctypes.POINTER(NumRsTensor), c_uint32_p, c_size_t]
lib.numrs_broadcast_to.restype = ctypes.POINTER(NumRsTensor)



# NN
lib.numrs_linear_new.argtypes = [c_size_t, c_size_t]
lib.numrs_linear_new.restype = ctypes.POINTER(NumRsLinear)

lib.numrs_relu_layer_new.argtypes = []
lib.numrs_relu_layer_new.restype = ctypes.POINTER(NumRsReLU)

lib.numrs_conv1d_new.argtypes = [c_size_t, c_size_t, c_size_t, c_size_t, c_size_t]
lib.numrs_conv1d_new.restype = ctypes.POINTER(NumRsConv1d)

lib.numrs_sequential_new.argtypes = []
lib.numrs_sequential_new.restype = ctypes.POINTER(NumRsSequential)

lib.numrs_sequential_free.argtypes = [ctypes.POINTER(NumRsSequential)]
lib.numrs_sequential_free.restype = None

lib.numrs_sequential_add_linear.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsLinear)]
lib.numrs_sequential_add_linear.restype = None

lib.numrs_sequential_add_relu.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsReLU)]
lib.numrs_sequential_add_relu.restype = None

lib.numrs_sequential_forward.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsTensor)]
lib.numrs_sequential_forward.restype = ctypes.POINTER(NumRsTensor)

# Train
lib.numrs_dataset_new.argtypes = [c_float_p, c_uint32_p, c_size_t, c_float_p, c_uint32_p, c_size_t, c_size_t]
lib.numrs_dataset_new.restype = ctypes.POINTER(NumRsDataset)

lib.numrs_dataset_free.argtypes = [ctypes.POINTER(NumRsDataset)]
lib.numrs_dataset_free.restype = None

# --- Trainer ---
lib.numrs_trainer_builder_new.argtypes = [ctypes.POINTER(NumRsSequential)]
lib.numrs_trainer_builder_new.restype = ctypes.POINTER(NumRsTrainerBuilder)

lib.numrs_trainer_builder_learning_rate.argtypes = [ctypes.POINTER(NumRsTrainerBuilder), ctypes.c_double]
lib.numrs_trainer_builder_learning_rate.restype = ctypes.POINTER(NumRsTrainerBuilder)

lib.numrs_trainer_build.argtypes = [ctypes.POINTER(NumRsTrainerBuilder), ctypes.c_char_p, ctypes.c_char_p]
lib.numrs_trainer_build.restype = ctypes.POINTER(NumRsTrainer)

lib.numrs_trainer_fit.argtypes = [ctypes.POINTER(NumRsTrainer), ctypes.POINTER(NumRsDataset), ctypes.c_size_t]
lib.numrs_trainer_fit.restype = None

lib.numrs_trainer_free.argtypes = [ctypes.POINTER(NumRsTrainer)]
lib.numrs_trainer_free.restype = None

# --- ONNX ---
class NumRsOnnxModel(ctypes.Structure):
    pass

lib.numrs_onnx_export.argtypes = [ctypes.POINTER(NumRsTensor), ctypes.c_char_p]
lib.numrs_onnx_export.restype = ctypes.c_int

lib.numrs_onnx_load.argtypes = [ctypes.c_char_p]
lib.numrs_onnx_load.restype = ctypes.POINTER(NumRsOnnxModel)

lib.numrs_onnx_free.argtypes = [ctypes.POINTER(NumRsOnnxModel)]
lib.numrs_onnx_free.restype = None

lib.numrs_onnx_infer_simple.argtypes = [ctypes.POINTER(NumRsOnnxModel), ctypes.POINTER(NumRsArray), ctypes.c_char_p]
lib.numrs_onnx_infer_simple.restype = ctypes.POINTER(NumRsArray)

# --- New Bindings (Step 2 Implementation) ---

# NN - Sigmoid
lib.numrs_sigmoid_new.argtypes = []
lib.numrs_sigmoid_new.restype = ctypes.POINTER(NumRsSigmoid)

lib.numrs_sequential_add_sigmoid.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsSigmoid)]
lib.numrs_sequential_add_sigmoid.restype = None

# NN - Softmax
lib.numrs_softmax_new.argtypes = []
lib.numrs_softmax_new.restype = ctypes.POINTER(NumRsSoftmax)

lib.numrs_sequential_add_softmax.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsSoftmax)]
lib.numrs_sequential_add_softmax.restype = None

# NN - Dropout
lib.numrs_dropout_new.argtypes = [ctypes.c_float]
lib.numrs_dropout_new.restype = ctypes.POINTER(NumRsDropout)

lib.numrs_sequential_add_dropout.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsDropout)]
lib.numrs_sequential_add_dropout.restype = None

# NN - Flatten
lib.numrs_flatten_new.argtypes = [c_size_t, c_size_t]
lib.numrs_flatten_new.restype = ctypes.POINTER(NumRsFlatten)

lib.numrs_sequential_add_flatten.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsFlatten)]
lib.numrs_sequential_add_flatten.restype = None

# NN - BatchNorm1d
lib.numrs_batchnorm1d_new.argtypes = [c_size_t]
lib.numrs_batchnorm1d_new.restype = ctypes.POINTER(NumRsBatchNorm1d)

lib.numrs_sequential_add_batchnorm1d.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsBatchNorm1d)]
lib.numrs_sequential_add_batchnorm1d.restype = None

# NN - Conv1d (add to sequential missing in previous)
lib.numrs_sequential_add_conv1d.argtypes = [ctypes.POINTER(NumRsSequential), ctypes.POINTER(NumRsConv1d)]
lib.numrs_sequential_add_conv1d.restype = None

# Tensor Methods
lib.numrs_tensor_shape.argtypes = [ctypes.POINTER(NumRsTensor), ctypes.POINTER(c_size_t)]
lib.numrs_tensor_shape.restype = ctypes.POINTER(c_size_t) # Returns *const usize

lib.numrs_tensor_item.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_tensor_item.restype = ctypes.c_float

lib.numrs_tensor_detach.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_tensor_detach.restype = ctypes.POINTER(NumRsTensor)

lib.numrs_tensor_zero_grad.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_tensor_zero_grad.restype = None

# Ops - Transpose
lib.numrs_transpose.argtypes = [ctypes.POINTER(NumRsTensor)]
lib.numrs_transpose.restype = ctypes.POINTER(NumRsTensor)

# Array API
lib.numrs_array_zeros.argtypes = [c_uint32_p, c_size_t]
lib.numrs_array_zeros.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_ones.argtypes = [c_uint32_p, c_size_t]
lib.numrs_array_ones.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_shape.argtypes = [ctypes.POINTER(NumRsArray), c_uint32_p, ctypes.POINTER(c_size_t)]
lib.numrs_array_shape.restype = None

lib.numrs_array_ndim.argtypes = [ctypes.POINTER(NumRsArray)]
lib.numrs_array_ndim.restype = c_size_t

lib.numrs_array_data.argtypes = [ctypes.POINTER(NumRsArray)]
lib.numrs_array_data.restype = c_float_p

lib.numrs_array_reshape.argtypes = [ctypes.POINTER(NumRsArray), c_uint32_p, c_size_t]
lib.numrs_array_reshape.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_add.argtypes = [ctypes.POINTER(NumRsArray), ctypes.POINTER(NumRsArray)]
lib.numrs_array_add.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_sub.argtypes = [ctypes.POINTER(NumRsArray), ctypes.POINTER(NumRsArray)]
lib.numrs_array_sub.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_mul.argtypes = [ctypes.POINTER(NumRsArray), ctypes.POINTER(NumRsArray)]
lib.numrs_array_mul.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_div.argtypes = [ctypes.POINTER(NumRsArray), ctypes.POINTER(NumRsArray)]
lib.numrs_array_div.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_matmul.argtypes = [ctypes.POINTER(NumRsArray), ctypes.POINTER(NumRsArray)]
lib.numrs_array_matmul.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_from_f64.argtypes = [ctypes.POINTER(ctypes.c_double), c_uint32_p, c_size_t]
lib.numrs_array_from_f64.restype = ctypes.POINTER(NumRsArray)

lib.numrs_array_from_i32.argtypes = [ctypes.POINTER(ctypes.c_int), c_uint32_p, c_size_t]
lib.numrs_array_from_i32.restype = ctypes.POINTER(NumRsArray)
