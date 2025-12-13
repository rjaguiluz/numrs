use crate::array::{Array, DTypeValue, DynArray};
use anyhow::{Result, bail};

/// Compute dot product of two 1-D arrays
/// 
/// For 1-D arrays, this is the inner product: sum(a * b)
/// 
/// This implementation uses the dispatch system for maximum performance:
/// - BLAS: sdot (5-10x faster)
/// - SIMD: FMA instructions (2-3x faster)
/// - Scalar: fallback
#[inline(always)]
pub fn dot<T1: DTypeValue, T2: DTypeValue>(a: &Array<T1>, b: &Array<T2>) -> Result<Array> {
    // Validate both are 1-D
    if a.shape().len() != 1 || b.shape().len() != 1 {
        bail!("dot: both inputs must be 1-D arrays");
    }
    
    // Validate same length
    if a.shape()[0] != b.shape()[0] {
        bail!("dot: arrays must have same length");
    }
    
    let dyn_result = crate::ops::promotion_wrappers::binary_promoted_with(a, b, |a_f32, b_f32| {
        let table = crate::backend::dispatch::get_dispatch_table();
        let result_scalar = (table.dot)(a_f32, b_f32)?;
        let result = Array::new(vec![1], vec![result_scalar]);
        Ok(DynArray::from(result))
    }, "dot")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_basic() {
        let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
        let b = Array::new(vec![3], vec![4.0, 5.0, 6.0]);
        let result = dot(&a, &b).unwrap();
        
        assert_eq!(result.shape(), &[1]);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        println!("DEBUG: result value = {}", result.data[0]);
        assert!((result.data[0] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_orthogonal() {
        let a = Array::new(vec![2], vec![1.0, 0.0]);
        let b = Array::new(vec![2], vec![0.0, 1.0]);
        let result = dot(&a, &b).unwrap();
        
        assert_eq!(result.shape(), &[1]);
        assert!(result.data[0].abs() < 1e-5); // Should be 0
    }

    #[test]
    fn test_dot_negative() {
        let a = Array::new(vec![3], vec![1.0, -2.0, 3.0]);
        let b = Array::new(vec![3], vec![-1.0, 2.0, -3.0]);
        let result = dot(&a, &b).unwrap();
        
        assert_eq!(result.shape(), &[1]);
        // 1*(-1) + (-2)*2 + 3*(-3) = -1 - 4 - 9 = -14
        assert!((result.data[0] - (-14.0)).abs() < 1e-5);
    }
}


