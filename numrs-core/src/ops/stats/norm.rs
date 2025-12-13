use crate::array::{Array, DTypeValue, promotion};
use crate::ops::reduction::sum::sum;
use crate::ops::elementwise::unary::sqrt::sqrt;
use anyhow::Result;

/// Compute L2 (Euclidean) norm of an array
/// 
/// Returns sqrt(sum(x^2))
#[inline(always)]
pub fn norm<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    // Convert to f32 for numerical computation
    let a_f32: Array<f32> = promotion::cast_array(a);
    // Square all elements using iterators
    let squared_data: Vec<f32> = a_f32.data.iter().map(|&x| x * x).collect();
    let squared = Array::new(a_f32.shape.clone(), squared_data);
    
    // Sum all squared elements (now returns Array<f32>)
    let sum_squared: Array<f32> = sum(&squared, None)?;
    
    // Take square root (shape [1] -> [1])
    let result: Array<f32> = sqrt(&sum_squared)?;
    
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_basic() {
        // [3, 4] -> sqrt(9 + 16) = 5
        let a = Array::new(vec![2], vec![3.0, 4.0]);
        let result = norm(&a).unwrap();
        
        assert_eq!(result.shape, vec![1]);
        assert!((result.data[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_unit_vector() {
        // [1, 0, 0] -> sqrt(1) = 1
        let a = Array::new(vec![3], vec![1.0, 0.0, 0.0]);
        let result = norm(&a).unwrap();
        
        assert_eq!(result.shape, vec![1]);
        assert!((result.data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_negative() {
        // [-3, 4] -> sqrt(9 + 16) = 5
        let a = Array::new(vec![2], vec![-3.0, 4.0]);
        let result = norm(&a).unwrap();
        
        assert_eq!(result.shape, vec![1]);
        assert!((result.data[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_larger() {
        // [1, 2, 3, 4] -> sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let result = norm(&a).unwrap();
        
        assert_eq!(result.shape, vec![1]);
        let expected = 30.0_f32.sqrt();
        assert!((result.data[0] - expected).abs() < 1e-5);
    }
}


