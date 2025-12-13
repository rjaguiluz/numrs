use napi::bindgen_prelude::*;
use napi_derive::napi;
use numrs::array::{Array, DynArray};

use crate::ArrayOptions;

/// Helper macro for reduction operations that support multiple dtypes
macro_rules! reduction_op {
    ($op_name:ident, $op_fn:path, $doc:expr) => {
        #[doc = $doc]
        #[napi]
        pub fn $op_name(
            data: Float32Array,
            options: Option<ArrayOptions>,
        ) -> Result<f64> {
            let opts = options.unwrap_or(ArrayOptions {
                shape: None,
            });
            
            let data_len = data.len();
            
            let shape = opts.shape
                .map(|s| s.into_iter().map(|v| v as usize).collect())
                .unwrap_or_else(|| vec![data_len]);
            
            let arr = Array::new(shape, data.as_ref().to_vec());
            let res = $op_fn(&arr)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            
            let result = match res {
                DynArray::F32(arr) => arr.data[0] as f64,
                _ => return Err(Error::from_reason("Unexpected result type")),
            };
            
            Ok(result)
        }
    };
}

// Define all reduction operations using the macro
reduction_op!(sum, numrs::ops::sum, "Sum operation - sum all elements\n\nExample: `numrs.sum([1, 2, 3], { dtype: 'float32' })`");
reduction_op!(mean, numrs::ops::mean, "Mean operation - compute average\n\nExample: `numrs.mean([1, 2, 3], { dtype: 'float32' })`");
reduction_op!(max, numrs::ops::max, "Max operation - find maximum value\n\nExample: `numrs.max([1, 5, 3], { dtype: 'float32' })`");
reduction_op!(min, numrs::ops::min, "Min operation - find minimum value\n\nExample: `numrs.min([5, 2, 8], { dtype: 'float32' })`");
reduction_op!(variance, numrs::ops::variance, "Variance operation - compute variance\n\nExample: `numrs.variance([1, 2, 3, 4], { dtype: 'float32' })`");
