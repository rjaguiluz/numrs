use crate::array::{Array, DTypeValue, promotion};
use crate::ops::reduction::{sum::sum, max::max};
use anyhow::Result;

/// Compute softmax function: exp(x - max(x)) / sum(exp(x - max(x)))
/// 
/// Uses the numerically stable version by subtracting the maximum value
/// before computing exponentials to prevent overflow.
#[inline(always)]
pub fn softmax<T: DTypeValue>(a: &Array<T>, axis: Option<usize>) -> Result<Array> {
    // Convert to f32 for numerical computation
    let a_f32: Array<f32> = promotion::cast_array(a);
    // Only support full reduction (axis=None) for now
    if axis.is_some() {
        anyhow::bail!("softmax: axis-specific reduction not yet supported");
    }
    
    // Find max for numerical stability (now returns Array<f32>)
    let max_result: Array<f32> = max(&a_f32, None)?;
    let max_val = max_result.data[0];
    
    // Subtract max and compute exp: exp(x - max(x))
    // Subtract max and compute exp: exp(x - max(x))
    let exp_data: Vec<f32> = a_f32.data.iter()
        .map(|&x| (x - max_val).exp())
        .collect();
    let exp_vals = Array::new(a_f32.shape.clone(), exp_data);
    
    // Sum the exponentials (now returns Array<f32>)
    let sum_result: Array<f32> = sum(&exp_vals, None)?;
    let sum_val = sum_result.data[0];
    
    // Divide by sum
    // Divide by sum
    let result_data: Vec<f32> = exp_vals.data.iter()
        .map(|&x| x / sum_val)
        .collect();
    
    Ok(Array::new(a.shape.clone(), result_data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
        let result = softmax(&a, None).unwrap();
        
        assert_eq!(result.shape, vec![3]);
        
        // Sum should be 1.0
        let sum: f32 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // All values should be positive
        for &val in &result.data {
            assert!(val > 0.0);
        }
        
        // Values should be in ascending order (since input is ascending)
        assert!(result.data[0] < result.data[1]);
        assert!(result.data[1] < result.data[2]);
    }

    #[test]
    fn test_softmax_uniform() {
        let a = Array::new(vec![3], vec![5.0, 5.0, 5.0]);
        let result = softmax(&a, None).unwrap();
        
        assert_eq!(result.shape, vec![3]);
        
        // All values should be equal to 1/3
        for &val in &result.data {
            assert!((val - 1.0/3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_large_values() {
        // Test numerical stability with large values
        let a = Array::new(vec![3], vec![1000.0, 1001.0, 1002.0]);
        let result = softmax(&a, None).unwrap();
        
        assert_eq!(result.shape, vec![3]);
        
        // Sum should still be 1.0
        let sum: f32 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // No NaN or Inf
        for &val in &result.data {
            assert!(val.is_finite());
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_softmax_negative() {
        let a = Array::new(vec![3], vec![-1.0, 0.0, 1.0]);
        let result = softmax(&a, None).unwrap();
        
        assert_eq!(result.shape, vec![3]);
        
        // Sum should be 1.0
        let sum: f32 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        
        // Middle value should be intermediate
        assert!(result.data[0] < result.data[1]);
        assert!(result.data[1] < result.data[2]);
    }
}


