use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use crate::ops::promotion_wrappers;
use anyhow::Result;

#[inline(always)]
pub fn leaky_relu<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result = promotion_wrappers::unary_promoted(a, ElementwiseKind::LeakyRelu, "leaky_relu")?;
    dyn_result.into_typed()
}
