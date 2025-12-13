use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use crate::ops::promotion_wrappers;
use anyhow::Result;

#[inline(always)]
pub fn atan<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result = promotion_wrappers::unary_promoted(a, ElementwiseKind::Atan, "atan")?;
    dyn_result.into_typed()
}
