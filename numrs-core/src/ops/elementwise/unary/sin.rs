use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Unary sine
#[inline(always)]
pub fn sin<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::unary_promoted(a, ElementwiseKind::Sin, "sin")?;
    dyn_result.into_typed()
}
