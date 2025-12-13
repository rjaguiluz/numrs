use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Unary cosine
#[inline(always)]
pub fn cos<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::unary_promoted(a, ElementwiseKind::Cos, "cos")?;
    dyn_result.into_typed()
}
