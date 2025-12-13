use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Natural logarithm (ln(x))
#[inline(always)]
pub fn log<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::unary_promoted(a, ElementwiseKind::Log, "log")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_basic() {
        let a = Array::new(vec![3], vec![1.0, 2.718281828, 7.389056099]);
        let result = log(&a).unwrap();
        assert!((result.data[0] - 0.0).abs() < 1e-5); // ln(1) = 0
        assert!((result.data[1] - 1.0).abs() < 1e-5); // ln(e) = 1
        assert!((result.data[2] - 2.0).abs() < 1e-5); // ln(e²) = 2
    }

    #[test]
    fn test_log_powers_of_10() {
        let a = Array::new(vec![3], vec![1.0, 10.0, 100.0]);
        let result = log(&a).unwrap();
        assert!((result.data[0] - 0.0).abs() < 1e-5); // ln(1) = 0
        assert!((result.data[1] - 2.302585).abs() < 1e-5); // ln(10) ≈ 2.303
        assert!((result.data[2] - 4.605170).abs() < 1e-5); // ln(100) ≈ 4.605
    }
}
