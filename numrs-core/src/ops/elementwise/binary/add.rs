use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Elementwise add with automatic type promotion
/// 
/// Returns an Array with the promoted dtype. The return type defaults to f32
/// unless type promotion determines otherwise.
/// 
/// # Examples
/// ```
/// use numrs::{Array, ops};
/// 
/// let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
/// let b = Array::new(vec![3], vec![4.0, 5.0, 6.0]);
/// let c = ops::add(&a, &b).unwrap();
/// assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
/// ```
#[inline(always)]
pub fn add<T1, T2>(a: &Array<T1>, b: &Array<T2>) -> Result<Array>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    // Internally uses DynArray for type promotion, then converts to Array
    let dyn_result = crate::ops::promotion_wrappers::binary_promoted(a, b, ElementwiseKind::Add, "add")?;
    dyn_result.into_typed()
}
