use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Unary absolute value
#[inline(always)]
pub fn abs<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::unary_promoted(a, ElementwiseKind::Abs, "abs")?;
    dyn_result.into_typed()
}
