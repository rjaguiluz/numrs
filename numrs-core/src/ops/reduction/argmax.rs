use crate::array::{Array, DTypeValue};
use crate::llo::reduction::ReductionKind;
use crate::ops::promotion_wrappers;
use anyhow::Result;

/// Returns the index of the maximum value along an optional axis.
///
/// # Arguments
///
/// * `a` - Input array
/// * `axis` - Optional axis to find argmax along. If None, finds global argmax.
///
/// # Returns
///
/// Array containing the index of the maximum value (as f32):
/// - axis=None: shape [1] (scalar index)
/// - axis=Some(k): shape with dimension k removed, containing indices along that axis
///
/// # Examples
///
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
/// let idx = ops::argmax(&a, None)?;
/// assert_eq!(idx.data[0] as usize, 4); // Index of max value 5.0
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// With ties (returns first occurrence):
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![4], vec![1.0, 3.0, 3.0, 2.0]);
/// let idx = ops::argmax(&a, None)?;
/// assert_eq!(idx.data[0] as usize, 1); // First occurrence of max value 3.0
/// # Ok::<(), anyhow::Error>(())
/// ```
#[inline(always)]
pub fn argmax<T: DTypeValue>(a: &Array<T>, axis: Option<usize>) -> Result<Array> {
    if a.data.is_empty() {
        anyhow::bail!("cannot compute argmax of empty array");
    }
    
    let dyn_result = promotion_wrappers::reduction_promoted(a, axis, ReductionKind::ArgMax, "argmax")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_basic() -> Result<()> {
        let a = Array::new(vec![5], vec![3.0, 1.0, 4.0, 1.0, 5.0]);
        let idx = argmax(&a, None)?;
        
        assert_eq!(idx.shape, vec![1]);
        assert_eq!(idx.data[0] as usize, 4);
        Ok(())
    }

    #[test]
    fn test_argmax_first_element() -> Result<()> {
        let a = Array::new(vec![4], vec![10.0, 2.0, 3.0, 4.0]);
        let idx = argmax(&a, None)?;
        
        assert_eq!(idx.data[0] as usize, 0);
        Ok(())
    }

    #[test]
    fn test_argmax_last_element() -> Result<()> {
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let idx = argmax(&a, None)?;
        
        assert_eq!(idx.data[0] as usize, 3);
        Ok(())
    }

    #[test]
    fn test_argmax_ties() -> Result<()> {
        // Should return first occurrence
        let a = Array::new(vec![5], vec![1.0, 3.0, 2.0, 3.0, 1.0]);
        let idx = argmax(&a, None)?;
        
        assert_eq!(idx.data[0] as usize, 1); // First 3.0
        Ok(())
    }

    #[test]
    fn test_argmax_negative() -> Result<()> {
        let a = Array::new(vec![4], vec![-5.0, -2.0, -3.0, -4.0]);
        let idx = argmax(&a, None)?;
        
        assert_eq!(idx.data[0] as usize, 1); // -2.0 is maximum
        Ok(())
    }

    #[test]
    fn test_argmax_single_element() -> Result<()> {
        let a = Array::new(vec![1], vec![42.0]);
        let idx = argmax(&a, None)?;
        
        assert_eq!(idx.data[0] as usize, 0);
        Ok(())
    }

    #[test]
    fn test_argmax_2d_flattened() -> Result<()> {
        let a = Array::new(vec![2, 3], vec![
            1.0, 2.0, 3.0,
            6.0, 5.0, 4.0,
        ]);
        let idx = argmax(&a, None)?;
        
        // Flattened: [1, 2, 3, 6, 5, 4]
        // Max value 6.0 is at index 3
        assert_eq!(idx.data[0] as usize, 3);
        Ok(())
    }

    #[test]
    fn test_argmax_for_classification() -> Result<()> {
        // Common ML use case: finding predicted class from logits
        let logits = Array::new(vec![5], vec![0.1, 0.3, 0.8, 0.2, 0.1]);
        let predicted_class = argmax(&logits, None)?;
        
        assert_eq!(predicted_class.data[0] as usize, 2); // Class 2 has highest score
        Ok(())
    }

    #[test]
    fn test_argmax_empty_array() {
        let a: Array<f32> = Array::new(vec![0], vec![]);
        let result = argmax(&a, None);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_argmax_axis0_2d() -> Result<()> {
        // [[1, 5, 3],
        //  [4, 2, 6]]
        // argmax(axis=0) = [1, 0, 1] (indices along axis 0)
        let a = Array::new(vec![2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let idx = argmax(&a, Some(0))?;
        assert_eq!(idx.shape, vec![3]);
        assert_eq!(idx.data[0] as usize, 1); // 4 > 1
        assert_eq!(idx.data[1] as usize, 0); // 5 > 2
        assert_eq!(idx.data[2] as usize, 1); // 6 > 3
        Ok(())
    }

    #[test]
    fn test_argmax_axis1_2d() -> Result<()> {
        // [[1, 5, 3],
        //  [4, 2, 6]]
        // argmax(axis=1) = [1, 2] (indices along axis 1)
        let a = Array::new(vec![2, 3], vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let idx = argmax(&a, Some(1))?;
        assert_eq!(idx.shape, vec![2]);
        assert_eq!(idx.data[0] as usize, 1); // 5 is max in first row
        assert_eq!(idx.data[1] as usize, 2); // 6 is max in second row
        Ok(())
    }

    #[test]
    fn test_argmax_axis0_3d() -> Result<()> {
        // shape [2, 2, 2]
        // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        // argmax(axis=0) -> [2, 2] with values [[5,6],[7,8]] being maxes
        let a = Array::new(
            vec![2, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let idx = argmax(&a, Some(0))?;
        assert_eq!(idx.shape, vec![2, 2]);
        // All indices should be 1 (second element along axis 0)
        assert_eq!(idx.data[0] as usize, 1);
        assert_eq!(idx.data[1] as usize, 1);
        assert_eq!(idx.data[2] as usize, 1);
        assert_eq!(idx.data[3] as usize, 1);
        Ok(())
    }

    #[test]
    fn test_argmax_1d_axis0() -> Result<()> {
        let a = Array::new(vec![4], vec![1.0, 4.0, 2.0, 3.0]);
        let idx = argmax(&a, Some(0))?;
        assert_eq!(idx.shape, vec![1]);
        assert_eq!(idx.data[0] as usize, 1);
        Ok(())
    }
}

