use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use crate::ops::promotion_wrappers;
use anyhow::Result;

/// Softplus activation function: softplus(x) = ln(1 + e^x)
///
/// This is a smooth approximation of ReLU.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Array with softplus applied element-wise
///
/// # Example
/// ```
/// use numrs::Array;
/// use numrs::ops::softplus;
///
/// let a = Array::new(vec![3], vec![0.0, 1.0, -1.0]);
/// let result = softplus(&a).unwrap();
/// // result ≈ [0.693, 1.313, 0.313]
/// ```
#[inline(always)]
pub fn softplus<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result = promotion_wrappers::unary_promoted(a, ElementwiseKind::Softplus, "softplus")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softplus_basic() {
        let a = Array::new(vec![3], vec![0.0, 1.0, -1.0]);
        let result = softplus(&a).unwrap();
        
        // softplus(0) = ln(2) ≈ 0.693147
        // softplus(1) = ln(1 + e) ≈ 1.313262
        // softplus(-1) = ln(1 + e^-1) ≈ 0.313262
        assert!((result.data[0] - 0.693147).abs() < 1e-5);
        assert!((result.data[1] - 1.313262).abs() < 1e-5);
        assert!((result.data[2] - 0.313262).abs() < 1e-5);
    }

    #[test]
    fn test_softplus_large_values() {
        let a = Array::new(vec![2], vec![10.0, -10.0]);
        let result = softplus(&a).unwrap();
        
        // softplus(10) ≈ 10 (for large positive x, softplus(x) ≈ x)
        // softplus(-10) ≈ 0
        assert!((result.data[0] - 10.0).abs() < 1e-4);
        assert!(result.data[1] < 1e-4);
    }
}
