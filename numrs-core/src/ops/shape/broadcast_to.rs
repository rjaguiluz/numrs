use crate::array::{Array, DTypeValue};
use anyhow::{Result, bail};

/// Broadcast an array to a new shape following NumPy broadcasting rules.
/// 
/// Broadcasting Rules:
/// 1. Dimensions are aligned from right to left
/// 2. Sizes in each dimension must either match or one must be 1
/// 3. Missing dimensions are treated as 1
///
/// **Optimization:**
/// This implementation now leverages `Array::broadcast_view` to calculate correct strides
/// and then uses `Array::to_contiguous` for efficient materialization.
/// This avoids expensive div/mod operations in the inner loop.
///
/// # Examples
/// ```
/// use numrs::{Array, ops};
/// 
/// let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
/// let b = ops::broadcast_to(&a, &[2, 3]).unwrap();
/// assert_eq!(b.shape, vec![2, 3]);
/// ```
pub fn broadcast_to<T>(array: &Array<T>, target_shape: &[usize]) -> Result<Array<T>>
where
    T: DTypeValue,
{
    // Si ya tiene el shape deseado, retornar copia
    if array.shape == target_shape {
        return Ok(array.clone());
    }
    
    // 1. Create a virtual view with 0-strides for broadcast dimensions
    // This validates the broadcast shape internally
    let view = array.broadcast_view(target_shape)?;
    
    // 2. Materialize efficiently using the view's strides
    // to_contiguous uses incremental index updates, much faster than div/mod
    Ok(view.to_contiguous())
}



/// Valida que el broadcast sea posible según reglas de NumPy (pública para Array)
pub fn validate_broadcast_public(src_shape: &[usize], target_shape: &[usize]) -> Result<()> {
    let src_ndim = src_shape.len();
    let target_ndim = target_shape.len();
    
    // El target debe tener igual o más dimensiones
    if target_ndim < src_ndim {
        bail!(
            "broadcast_to: target shape {:?} has fewer dimensions than source {:?}",
            target_shape,
            src_shape
        );
    }
    
    // Validar cada dimensión desde el final
    for i in 0..src_ndim {
        let src_dim = src_shape[src_ndim - 1 - i];
        let target_dim = target_shape[target_ndim - 1 - i];
        
        if src_dim != target_dim && src_dim != 1 {
            bail!(
                "broadcast_to: dimension mismatch at axis {}: {} cannot broadcast to {}",
                target_ndim - 1 - i,
                src_dim,
                target_dim
            );
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_broadcast_scalar_to_vector() {
        let a = Array::new(vec![1], vec![5.0f32]);
        let b = broadcast_to(&a, &[3]).unwrap();
        assert_eq!(b.shape, vec![3]);
        assert_eq!(b.data, vec![5.0, 5.0, 5.0]);
    }
    
    #[test]
    fn test_broadcast_vector_to_matrix() {
        let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
        let b = broadcast_to(&a, &[2, 3]).unwrap();
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(b.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_broadcast_column_vector() {
        let a = Array::new(vec![2, 1], vec![1.0, 2.0]);
        let b = broadcast_to(&a, &[2, 3]).unwrap();
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(b.data, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }
    
    #[test]
    fn test_broadcast_invalid() {
        let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
        let result = broadcast_to(&a, &[2, 4]);
        assert!(result.is_err());
    }
}
