use crate::array::{Array, DTypeValue};
use crate::llo::reduction::ReductionKind;
use anyhow::Result;

/// Find the maximum value along an axis
///
/// # Arguments
/// * `a` - Input array
/// * `axis` - Optional axis to reduce along (None = reduce all)
///
/// # Returns
/// Array with maximum values
///
/// # Example
/// ```
/// use numrs::Array;
/// use numrs::ops::max;
///
/// let a = Array::new(vec![2, 3], vec![1.0, 5.0, 3.0, 2.0, 4.0, 1.0]);
/// let result = max(&a, None).unwrap();
/// // result = [5.0]
/// ```
#[inline(always)]
pub fn max<T: DTypeValue>(a: &Array<T>, axis: Option<usize>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::reduction_promoted(a, axis, ReductionKind::Max, "max")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_all() {
        let a = Array::new(vec![6], vec![1.0, 5.0, 3.0, 2.0, 4.0, 1.0]);
        let result = max(&a, None).unwrap();
        
        assert_eq!(result.shape(), &[1]);
        assert_eq!(result.data[0], 5.0);
    }

    #[test]
    fn test_max_negative() {
        let a = Array::new(vec![4], vec![-1.0, -5.0, -3.0, -2.0]);
        let result = max(&a, None).unwrap();
        
        assert_eq!(result.data[0], -1.0);
    }

    #[test]
    fn test_max_single_element() {
        let a = Array::new(vec![1], vec![42.0]);
        let result = max(&a, None).unwrap();
        
        assert_eq!(result.data[0], 42.0);
    }
}
