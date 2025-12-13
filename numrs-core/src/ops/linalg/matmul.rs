use crate::array::{Array, DTypeValue, DynArray};
use anyhow::Result;

/// Matrix multiplication: C = A @ B
///
/// # Arguments
/// * `a` - Left matrix (M x K)
/// * `b` - Right matrix (K x N)
///
/// # Returns
/// Result matrix (M x N)
///
/// # Errors
/// Returns error if dimensions are incompatible
///
/// # Type Promotion
/// Inputs are promoted to f32 before computation (via binary_promoted_with).
/// This ensures compatibility with BLAS/MKL backends which operate on f32.
/// The result is then converted back to the appropriate output type.
#[inline(always)]
pub fn matmul<T1: DTypeValue, T2: DTypeValue>(a: &Array<T1>, b: &Array<T2>) -> Result<Array> {
    // Promote inputs to f32 for BLAS computation
    let dyn_result = crate::ops::promotion_wrappers::binary_promoted_with(a, b, |a_f32, b_f32| {
        let table = crate::backend::dispatch::get_dispatch_table();
        let result = (table.matmul)(a_f32, b_f32)?;
        Ok(DynArray::from(result))
    }, "matmul")?;
    dyn_result.into_typed()
}

