use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Array with sigmoid applied element-wise
///
/// # Example
/// ```
/// use numrs::Array;
/// use numrs::ops::sigmoid;
///
/// let a = Array::new(vec![3], vec![0.0, 1.0, -1.0]);
/// let result = sigmoid(&a).unwrap();
/// // result ≈ [0.5, 0.731, 0.269]
/// ```
#[inline(always)]
pub fn sigmoid<T: DTypeValue>(a: &Array<T>) -> Result<Array>
where
    T: crate::array::DTypeValue,
{
    // σ(x) = 1 / (1 + exp(-x))
    let dyn_result = crate::ops::promotion_wrappers::unary_promoted(a, ElementwiseKind::Sigmoid, "sigmoid")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_basic() {
        let a = Array::new(vec![3], vec![0.0, 1.0, -1.0]);
        let result = sigmoid(&a).unwrap();
        
        // sigmoid(0) = 0.5
        // sigmoid(1) ≈ 0.731
        // sigmoid(-1) ≈ 0.269
        assert!((result.data[0] - 0.5).abs() < 1e-6);
        assert!((result.data[1] - 0.7310585).abs() < 1e-5);
        assert!((result.data[2] - 0.2689414).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_large_values() {
        let a = Array::new(vec![3], vec![10.0, -10.0, 0.0]);
        let result = sigmoid(&a).unwrap();
        
        // sigmoid(10) ≈ 1.0
        // sigmoid(-10) ≈ 0.0
        assert!((result.data[0] - 1.0).abs() < 1e-4);
        assert!((result.data[1] - 0.0).abs() < 1e-4);
        assert!((result.data[2] - 0.5).abs() < 1e-6);
    }
}
