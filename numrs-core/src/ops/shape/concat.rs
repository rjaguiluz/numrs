use crate::array::{Array, DTypeValue};
use anyhow::{bail, Result};

/// Concatenates arrays along an existing axis.
///
/// # Arguments
///
/// * `arrays` - Slice of arrays to concatenate
/// * `axis` - Axis along which to concatenate (0-indexed)
///
/// # Returns
///
/// Concatenated array
///
/// # Errors
///
/// Returns error if:
/// - arrays is empty
/// - axis is out of bounds
/// - arrays have incompatible shapes (all dims except axis must match)
///
/// # Examples
///
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = Array::new(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
/// let concatenated = ops::concat(&[&a, &b], 0)?;
/// assert_eq!(concatenated.shape(), &[4, 3]);
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// Concatenating along axis 1:
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
/// let b = Array::new(vec![2, 1], vec![5.0, 6.0]);
/// let concatenated = ops::concat(&[&a, &b], 1)?;
/// assert_eq!(concatenated.shape(), &[2, 3]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn concat<T: DTypeValue>(arrays: &[&Array<T>], axis: usize) -> Result<Array<T>> {
    if arrays.is_empty() {
        bail!("need at least one array to concatenate");
    }
    
    // Get reference shape and validate
    let first = arrays[0];
    let ndim = first.shape.len();
    
    if axis >= ndim {
        bail!("axis {} is out of bounds for array of dimension {}", axis, ndim);
    }
    
    // Validate all arrays have compatible shapes
    let mut concat_size = 0;
    for arr in arrays {
        if arr.shape.len() != ndim {
            bail!(
                "all arrays must have the same number of dimensions (got {} and {})",
                ndim,
                arr.shape.len()
            );
        }
        
        for (i, (&dim1, &dim2)) in first.shape.iter().zip(arr.shape.iter()).enumerate() {
            if i != axis && dim1 != dim2 {
                bail!(
                    "all arrays must have the same shape except on concat axis (axis {})",
                    axis
                );
            }
        }
        
        concat_size += arr.shape[axis];
    }
    
    // Compute output shape
    let mut out_shape = first.shape.clone();
    out_shape[axis] = concat_size;
    let total_size: usize = out_shape.iter().product();
    
    // Allocate output
    let mut output = Vec::with_capacity(total_size);
    
    // Compute strides
    let mut strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * out_shape[i + 1];
    }
    
    // Copy data
    // Strategy: iterate through all positions, keeping track of which array we're in along concat axis
    
    if axis == ndim - 1 {
        // Concatenating along last axis - most common case, optimized
        // We can copy entire rows at once
        let total_rows = total_size / concat_size;
        
        for row_idx in 0..total_rows {
            for arr in arrays {
                let arr_row_size = arr.shape[axis];
                let start = row_idx * arr_row_size;
                let end = start + arr_row_size;
                output.extend_from_slice(&arr.data[start..end]);
            }
        }
    } else {
        // General case: iterate through outer dimensions
        let outer_size: usize = first.shape[..axis].iter().product();
        let inner_size: usize = first.shape[axis + 1..].iter().product();
        
        for outer_idx in 0..outer_size {
            for arr in arrays {
                let arr_axis_size = arr.shape[axis];
                for axis_idx in 0..arr_axis_size {
                    let arr_start = (outer_idx * arr.shape[axis] + axis_idx) * inner_size;
                    let arr_end = arr_start + inner_size;
                    output.extend_from_slice(&arr.data[arr_start..arr_end]);
                }
            }
        }
    }
    
    Ok(Array::new(out_shape, output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_axis0_2d() -> Result<()> {
        let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Array::new(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = concat(&[&a, &b], 0)?;
        
        assert_eq!(c.shape(), &[4, 3]);
        assert_eq!(c.data, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]);
        Ok(())
    }

    #[test]
    fn test_concat_axis1_2d() -> Result<()> {
        let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::new(vec![2, 1], vec![5.0, 6.0]);
        let c = concat(&[&a, &b], 1)?;
        
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data, vec![
            1.0, 2.0, 5.0,
            3.0, 4.0, 6.0,
        ]);
        Ok(())
    }

    #[test]
    fn test_concat_1d() -> Result<()> {
        let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
        let b = Array::new(vec![2], vec![4.0, 5.0]);
        let c = Array::new(vec![1], vec![6.0]);
        let result = concat(&[&a, &b, &c], 0)?;
        
        assert_eq!(result.shape(), &[6]);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_concat_multiple_arrays() -> Result<()> {
        let a = Array::new(vec![1, 2], vec![1.0, 2.0]);
        let b = Array::new(vec![1, 2], vec![3.0, 4.0]);
        let c = Array::new(vec![1, 2], vec![5.0, 6.0]);
        let result = concat(&[&a, &b, &c], 0)?;
        
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn test_concat_empty_arrays() {
        let result = concat::<f32>(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_incompatible_shapes() {
        let a = Array::new(vec![2, 3], vec![0.0; 6]);
        let b = Array::new(vec![2, 4], vec![0.0; 8]);
        let result = concat(&[&a, &b], 0);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_invalid_axis() {
        let a = Array::new(vec![2, 3], vec![0.0; 6]);
        let result = concat(&[&a], 5);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_single_array() -> Result<()> {
        let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = concat(&[&a], 0)?;
        
        assert_eq!(result.shape(), a.shape());
        assert_eq!(result.data, a.data);
        Ok(())
    }

    #[test]
    fn test_concat_batch_processing() -> Result<()> {
        // Common ML use case: concatenating batches
        let batch1 = Array::new(vec![2, 4], vec![0.0; 8]); // 2 samples, 4 features
        let batch2 = Array::new(vec![3, 4], vec![0.0; 12]); // 3 samples, 4 features
        let combined = concat(&[&batch1, &batch2], 0)?;
        
        assert_eq!(combined.shape(), &[5, 4]); // 5 samples total
        Ok(())
    }
}

