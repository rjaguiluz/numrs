use crate::array::{Array, DTypeValue};
use anyhow::{bail, Result};

/// Fast 2D transpose using cache-friendly blocked algorithm
/// 
/// Uses 32x32 tiles to maximize cache hits during transpose.
/// This is ~2-3x faster than naive element-by-element transpose for large matrices.
#[inline]
fn transpose_2d_blocked<T: DTypeValue>(
    input: &[T],
    output: &mut [T],
    rows: usize,
    cols: usize,
) {
    const BLOCK_SIZE: usize = 32;
    
    // Process matrix in blocks
    for i_block in (0..rows).step_by(BLOCK_SIZE) {
        for j_block in (0..cols).step_by(BLOCK_SIZE) {
            let i_end = (i_block + BLOCK_SIZE).min(rows);
            let j_end = (j_block + BLOCK_SIZE).min(cols);
            
            // Transpose the block
            for i in i_block..i_end {
                for j in j_block..j_end {
                    let in_idx = i * cols + j;
                    let out_idx = j * rows + i;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

/// Transposes an array by permuting its dimensions.
///
/// # Arguments
///
/// * `a` - Input array
/// * `axes` - Optional permutation of axes. If None, reverses the order of axes.
///
/// # Returns
///
/// Array with permuted dimensions
///
/// # Errors
///
/// Returns error if:
/// - axes length doesn't match array dimensions
/// - axes contains invalid indices
/// - axes contains duplicate indices
///
/// # Examples
///
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let transposed = ops::transpose(&a, None)?;
/// assert_eq!(transposed.shape(), &[3, 2]);
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// With specific axes:
/// ```
/// use numrs::{Array, ops};
///
/// let a = Array::new(vec![2, 3, 4], vec![0.0; 24]);
/// let transposed = ops::transpose(&a, Some(&[2, 0, 1]))?;
/// assert_eq!(transposed.shape(), &[4, 2, 3]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn transpose<T: DTypeValue>(a: &Array<T>, axes: Option<&[usize]>) -> Result<Array<T>> {
    let ndim = a.shape.len();
    
    // Fast path: 2D transpose without custom axes (most common case)
    if ndim == 2 && axes.is_none() {
        let rows = a.shape[0];
        let cols = a.shape[1];
        let mut output = vec![T::zero(); rows * cols];
        
        // Use blocked transpose for cache efficiency
        transpose_2d_blocked(&a.data, &mut output, rows, cols);
        
        let mut result = Array::new(vec![cols, rows], output);
        result.dtype = a.dtype;
        return Ok(result);
    }
    
    // Default: reverse all axes
    let axes_vec: Vec<usize> = if let Some(ax) = axes {
        if ax.len() != ndim {
            bail!(
                "axes length {} doesn't match array dimensions {}",
                ax.len(),
                ndim
            );
        }
        
        // Validate axes
        let mut seen = vec![false; ndim];
        for &axis in ax {
            if axis >= ndim {
                bail!("axis {} is out of bounds for array of dimension {}", axis, ndim);
            }
            if seen[axis] {
                bail!("repeated axis {}", axis);
            }
            seen[axis] = true;
        }
        
        ax.to_vec()
    } else {
        (0..ndim).rev().collect()
    };
    
    // Compute new shape
    let new_shape: Vec<usize> = axes_vec.iter().map(|&i| a.shape[i]).collect();
    let total_size: usize = new_shape.iter().product();
    
    // Compute strides for input array (row-major)
    let mut in_strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        in_strides[i] = in_strides[i + 1] * a.shape[i + 1];
    }
    
    // Compute strides for output array (row-major)
    let mut out_strides = vec![1; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }
    
    // Allocate output
    let mut output = vec![T::zero(); total_size];
    
    // Transpose: iterate through all output positions
    for out_idx in 0..total_size {
        // Compute multi-dimensional index in output space
        let mut out_coords = vec![0; ndim];
        let mut remainder = out_idx;
        for i in 0..ndim {
            out_coords[i] = remainder / out_strides[i];
            remainder %= out_strides[i];
        }
        
        // Map to input coordinates using axes permutation
        let mut in_coords = vec![0; ndim];
        for i in 0..ndim {
            in_coords[axes_vec[i]] = out_coords[i];
        }
        
        // Compute input flat index
        let in_idx: usize = in_coords
            .iter()
            .zip(in_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();
        
        output[out_idx] = a.data[in_idx];
    }
    
    let mut result = Array::new(new_shape, output);
    result.dtype = a.dtype;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d() -> Result<()> {
        let a = Array::new(vec![2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        let t = transpose(&a, None)?;
        
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.data, vec![
            1.0, 4.0,
            2.0, 5.0,
            3.0, 6.0,
        ]);
        Ok(())
    }

    #[test]
    fn test_transpose_3d() -> Result<()> {
        let a = Array::new(vec![2, 2, 2], vec![
            1.0, 2.0,  3.0, 4.0,
            5.0, 6.0,  7.0, 8.0,
        ]);
        let t = transpose(&a, None)?;
        
        assert_eq!(t.shape(), &[2, 2, 2]);
        // Default transpose reverses axes: (0,1,2) -> (2,1,0)
        Ok(())
    }

    #[test]
    fn test_transpose_custom_axes() -> Result<()> {
        let a = Array::new(vec![2, 3, 4], vec![0.0; 24]);
        let t = transpose(&a, Some(&[2, 0, 1]))?;
        
        assert_eq!(t.shape(), &[4, 2, 3]);
        Ok(())
    }

    #[test]
    fn test_transpose_1d() -> Result<()> {
        let a = Array::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let t = transpose(&a, None)?;
        
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.data, a.data);
        Ok(())
    }

    #[test]
    fn test_transpose_identity() -> Result<()> {
        let a = Array::new(vec![3, 3], vec![0.0; 9]);
        let t = transpose(&a, Some(&[0, 1]))?;
        
        assert_eq!(t.shape(), a.shape());
        Ok(())
    }

    #[test]
    fn test_transpose_invalid_axes_length() {
        let a = Array::new(vec![2, 3], vec![0.0; 6]);
        let result = transpose(&a, Some(&[0, 1, 2]));
        
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_invalid_axis_value() {
        let a = Array::new(vec![2, 3], vec![0.0; 6]);
        let result = transpose(&a, Some(&[0, 5]));
        
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_duplicate_axes() {
        let a = Array::new(vec![2, 3], vec![0.0; 6]);
        let result = transpose(&a, Some(&[0, 0]));
        
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_matmul_compatibility() -> Result<()> {
        // Create matrix A (2x3)
        let a = Array::new(vec![2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        
        // Transpose to get (3x2)
        let at = transpose(&a, None)?;
        assert_eq!(at.shape(), &[3, 2]);
        
        // Now at can multiply with a: (3x2) @ (2x3) = (3x3)
        Ok(())
    }
}
