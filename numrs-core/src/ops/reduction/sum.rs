use crate::array::{Array, DTypeValue};
use crate::llo::reduction::ReductionKind;
use anyhow::Result;

/// Reduce-sum over whole array (axis=None) or specific axis
#[inline(always)]
pub fn sum<T: DTypeValue>(a: &Array<T>, axis: Option<usize>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::reduction_promoted(a, axis, ReductionKind::Sum, "sum")?;
    dyn_result.into_typed()
}

