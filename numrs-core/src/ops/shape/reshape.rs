use crate::array::{Array, DTypeValue};
use anyhow::{bail, Result};

/// Reshapes an array to a new shape without changing its data.
///
/// # Arguments
///
/// * `a` - Input array
/// * `new_shape` - Target shape. Use -1 for one dimension to infer it automatically.
///
/// # Returns
///
/// Array with the same data but new shape
///
/// # Errors
///
/// Returns error if:
/// - New shape has multiple -1 dimensions
/// - New shape is incompatible with total size
/// - Resulting size would be negative
///
/// # Examples
///
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let reshaped = ops::reshape(&a, &[2, 3])?;
/// assert_eq!(reshaped.shape(), &[2, 3]);
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// Using -1 to infer dimension:
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
/// let reshaped = ops::reshape(&a, &[-1, 2])?; // Infers first dim as 2
/// assert_eq!(reshaped.shape(), &[2, 2]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn reshape<T: DTypeValue>(a: &Array<T>, new_shape: &[isize]) -> Result<Array<T>> {
    // Calculate total size of input
    let total_size = a.shape().iter().product::<usize>();

    // Find -1 dimension if any
    let mut infer_dim: Option<usize> = None;
    let mut known_size: usize = 1;
    
    for (i, &dim) in new_shape.iter().enumerate() {
        if dim == -1 {
            if infer_dim.is_some() {
                bail!("Can only specify one unknown dimension with -1");
            }
            infer_dim = Some(i);
        } else if dim < 0 {
            bail!("Invalid dimension size: {}", dim);
        } else {
            known_size *= dim as usize;
        }
    }

    // Compute final shape
    let final_shape: Vec<usize> = if let Some(infer_idx) = infer_dim {
        if known_size == 0 {
            bail!("Cannot infer dimension when other dimensions are zero");
        }
        if total_size % known_size != 0 {
            bail!(
                "Cannot reshape array of size {} into shape {:?}",
                total_size,
                new_shape
            );
        }
        let inferred_size = total_size / known_size;
        
        new_shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                if i == infer_idx {
                    inferred_size
                } else {
                    dim as usize
                }
            })
            .collect()
    } else {
        // No inference needed, just convert
        let final_shape: Vec<usize> = new_shape.iter().map(|&d| d as usize).collect();
        let new_size: usize = final_shape.iter().product();
        
        if new_size != total_size {
            bail!(
                "Cannot reshape array of size {} into shape {:?}",
                total_size,
                new_shape
            );
        }
        
        final_shape
    };

    // Create new array with same data but new shape
    let mut result = Array::new(final_shape, a.data.clone());
    result.dtype = a.dtype;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape_basic() -> Result<()> {
        let a = Array::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = reshape(&a, &[3, 2])?;
        
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(&reshaped.data, &a.data);
        Ok(())
    }

    #[test]
    fn test_reshape_3d() -> Result<()> {
        let a = Array::new(vec![8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let reshaped = reshape(&a, &[2, 2, 2])?;
        
        assert_eq!(reshaped.shape(), &[2, 2, 2]);
        assert_eq!(&reshaped.data, &a.data);
        Ok(())
    }

    #[test]
    fn test_reshape_infer_dimension() -> Result<()> {
        let a = Array::new(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = reshape(&a, &[-1, 3])?;
        
        assert_eq!(reshaped.shape(), &[2, 3]);
        Ok(())
    }

    #[test]
    fn test_reshape_infer_first_dimension() -> Result<()> {
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let reshaped = reshape(&a, &[-1, 2])?;
        
        assert_eq!(reshaped.shape(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn test_reshape_flatten() -> Result<()> {
        let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let flattened = reshape(&a, &[-1])?;
        
        assert_eq!(flattened.shape(), &[4]);
        Ok(())
    }

    #[test]
    fn test_reshape_invalid_size() {
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let result = reshape(&a, &[2, 3]);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_multiple_infer() {
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let result = reshape(&a, &[-1, -1]);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_to_scalar_like() -> Result<()> {
        let a = Array::new(vec![1], vec![42.0]);
        let reshaped = reshape(&a, &[1, 1, 1])?;
        
        assert_eq!(reshaped.shape(), &[1, 1, 1]);
        assert_eq!(reshaped.data[0], 42.0);
        Ok(())
    }

    #[test]
    fn test_reshape_identity() -> Result<()> {
        let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let reshaped = reshape(&a, &[2, 2])?;
        
        assert_eq!(reshaped.shape(), a.shape());
        assert_eq!(&reshaped.data, &a.data);
        Ok(())
    }
}
