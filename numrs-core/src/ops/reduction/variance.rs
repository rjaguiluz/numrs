use crate::array::{Array, DTypeValue};
use crate::llo::reduction::ReductionKind;
use anyhow::Result;

/// Computes the variance along an optional axis.
///
/// Variance is computed as: var(x) = mean((x - mean(x))²)
///
/// This implementation uses the dispatch system for optimal performance:
/// - Welford's algorithm for numerical stability (single-pass when axis=None)
/// - Two-pass algorithm for axis-based reduction
///
/// # Arguments
///
/// * `a` - Input array
/// * `axis` - Optional axis to compute variance along. If None, computes global variance.
///
/// # Returns
///
/// Array containing the variance. Shape depends on axis parameter:
/// - axis=None: shape [1] (scalar)
/// - axis=Some(k): shape with dimension k removed
///
/// # Examples
///
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
/// let var: Array<f32> = ops::variance(&a, None)?;
/// // Variance of [1,2,3,4] with mean=2.5 is 1.25
/// assert!((var.data[0] - 1.25).abs() < 1e-5);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn variance<T: DTypeValue>(a: &Array<T>, axis: Option<usize>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::reduction_promoted(a, axis, ReductionKind::Variance, "variance")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_1d() -> Result<()> {
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let var = variance(&a, None)?;
        
        // Mean = 2.5, variance = mean of [(1-2.5)², (2-2.5)², (3-2.5)², (4-2.5)²]
        // = mean of [2.25, 0.25, 0.25, 2.25] = 1.25
        assert!((var.data[0] - 1.25).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_variance_zeros() -> Result<()> {
        let a = Array::new(vec![4], vec![5.0, 5.0, 5.0, 5.0]);
        let var = variance(&a, None)?;
        
        assert!(var.data[0].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_variance_single_element() -> Result<()> {
        let a = Array::new(vec![1], vec![42.0]);
        let var = variance(&a, None)?;
        
        assert!(var.data[0].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_variance_negative_values() -> Result<()> {
        let a = Array::new(vec![4], vec![-2.0, -1.0, 1.0, 2.0]);
        let var = variance(&a, None)?;
        
        // Mean = 0, variance = mean([4, 1, 1, 4]) = 2.5
        assert!((var.data[0] - 2.5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_variance_axis0_2d() -> Result<()> {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        // mean(axis=0) = [2.5, 3.5, 4.5]
        // var(axis=0) = [(1-2.5)²+(4-2.5)²)/2, ...]
        //             = [2.25, 2.25, 2.25]
        let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let var = variance(&a, Some(0))?;
        assert_eq!(var.shape(), &[3]);
        assert!((var.data[0] - 2.25).abs() < 1e-5);
        assert!((var.data[1] - 2.25).abs() < 1e-5);
        assert!((var.data[2] - 2.25).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_variance_axis1_2d() -> Result<()> {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        // mean(axis=1) = [2.0, 5.0]
        // var(axis=1) = [var([1,2,3]), var([4,5,6])]
        //             = [0.666..., 0.666...]
        let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let var = variance(&a, Some(1))?;
        assert_eq!(var.shape(), &[2]);
        assert!((var.data[0] - 0.6666666).abs() < 1e-5);
        assert!((var.data[1] - 0.6666666).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_variance_1d_axis0() -> Result<()> {
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let var = variance(&a, Some(0))?;
        assert_eq!(var.shape(), &[1]);
        assert!((var.data[0] - 1.25).abs() < 1e-5);
        Ok(())
    }
}


