use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Hyperbolic tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Array with tanh applied element-wise
///
/// # Example
/// ```
/// use numrs::Array;
/// use numrs::ops::tanh;
///
/// let a = Array::new(vec![3], vec![0.0, 1.0, -1.0]);
/// let result = tanh(&a).unwrap();
/// // result ≈ [0.0, 0.762, -0.762]
/// ```
#[inline(always)]
pub fn tanh<T: DTypeValue>(a: &Array<T>) -> Result<Array>
where
    T: crate::array::DTypeValue,
{
    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    // or equivalently: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    let dyn_result = crate::ops::promotion_wrappers::unary_promoted(a, ElementwiseKind::Tanh, "tanh")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_basic() {
        let a = Array::new(vec![3], vec![0.0, 1.0, -1.0]);
        let result = tanh(&a).unwrap();
        
        // tanh(0) = 0
        // tanh(1) ≈ 0.7615941559557649
        // tanh(-1) ≈ -0.7615941559557649
        assert!((result.data[0] - 0.0).abs() < 1e-6);
        assert!((result.data[1] - 0.7615941).abs() < 1e-5);
        assert!((result.data[2] + 0.7615941).abs() < 1e-5);
    }

    #[test]
    fn test_tanh_large_values() {
        let a = Array::new(vec![3], vec![10.0, -10.0, 0.0]);
        let result = tanh(&a).unwrap();
        
        // tanh(10) ≈ 1.0
        // tanh(-10) ≈ -1.0
        assert!((result.data[0] - 1.0).abs() < 1e-4);
        assert!((result.data[1] + 1.0).abs() < 1e-4);
        assert!((result.data[2] - 0.0).abs() < 1e-6);
    }
}
