use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Negation (unary minus): -a
///
/// Returns a new array with all elements negated.
#[inline(always)]
pub fn neg<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result =
        crate::ops::promotion_wrappers::unary_promoted(a, ElementwiseKind::Neg, "neg")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neg_f32() {
        let a = Array::new(vec![3], vec![1.0f32, -2.0, 3.0]);
        let result = neg(&a).unwrap();
        assert_eq!(result.data, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_neg_f64() {
        let a = Array::new(vec![2], vec![5.0f64, -10.0]);
        // Result is always f32 currently
        let result = neg(&a).unwrap();
        assert_eq!(result.data, vec![-5.0, 10.0]);
    }

    #[test]
    fn test_neg_i32() {
        let a = Array::new(vec![4], vec![1i32, -2, 0, 5]);
        // Result is always f32 currently
        let result = neg(&a).unwrap();
        assert_eq!(result.data, vec![-1.0, 2.0, 0.0, -5.0]);
    }
}
