use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Elementwise power with automatic type promotion
/// 
/// Returns an Array with the promoted dtype.
#[inline(always)]
pub fn pow<T1, T2>(a: &Array<T1>, b: &Array<T2>) -> Result<Array>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    let dyn_result = crate::ops::promotion_wrappers::binary_promoted(a, b, ElementwiseKind::Pow, "pow")?;
    dyn_result.into_typed()
}
