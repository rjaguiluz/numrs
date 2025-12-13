use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Elementwise multiply with automatic type promotion
/// 
/// Returns an Array with the promoted dtype.
#[inline(always)]
pub fn mul<T1, T2>(a: &Array<T1>, b: &Array<T2>) -> Result<Array>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    let dyn_result = crate::ops::promotion_wrappers::binary_promoted(a, b, ElementwiseKind::Mul, "mul")?;
    dyn_result.into_typed()
}
