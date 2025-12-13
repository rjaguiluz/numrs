from .native_lib import lib, NumRsDataset, NumRsTrainerBuilder, NumRsTrainer
import ctypes
from .types import OptimizerType, LossType

__all__ = ["Dataset", "Trainer"]

def _flatten_and_get_shape(data):
    # Expect data to be list of lists (2D)
    flat = []
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0
    for row in data:
        flat.extend(row)
    return flat, [rows, cols]

class Dataset:
    def __init__(self, inputs: object, targets: object, batch_size: int) -> None:
        # inputs, targets: List of lists of floats
        in_flat, in_shape = _flatten_and_get_shape(inputs)
        tg_flat, tg_shape = _flatten_and_get_shape(targets)
        
        c_in = (ctypes.c_float * len(in_flat))(*in_flat)
        c_in_shape = (ctypes.c_uint32 * len(in_shape))(*in_shape)
        
        c_tg = (ctypes.c_float * len(tg_flat))(*tg_flat)
        c_tg_shape = (ctypes.c_uint32 * len(tg_shape))(*tg_shape)
        
        self._ptr = lib.numrs_dataset_new(
            c_in, c_in_shape, len(in_shape),
            c_tg, c_tg_shape, len(tg_shape),
            batch_size
        )
        
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.numrs_dataset_free(self._ptr)

class Trainer:
    def __init__(self, model: object, optimizer: OptimizerType = "sgd", loss: LossType = "mse", lr: float = 0.01) -> None:
        # 1. Create Builder
        builder_ptr = lib.numrs_trainer_builder_new(model._ptr)
        
        # 2. Set LR
        builder_ptr = lib.numrs_trainer_builder_learning_rate(builder_ptr, lr)
        
        # 3. Build using dynamic C API
        c_optimizer = optimizer.encode('utf-8')
        c_loss = loss.encode('utf-8')
        
        self._ptr = lib.numrs_trainer_build(builder_ptr, c_optimizer, c_loss)
        
        if not self._ptr:
            raise ValueError(f"Failed to build trainer. Check optimizer='{optimizer}' and loss='{loss}' support.")

    def fit(self, dataset: Dataset, epochs: int) -> None:
        lib.numrs_trainer_fit(self._ptr, dataset._ptr, epochs)
        
    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            lib.numrs_trainer_free(self._ptr)
