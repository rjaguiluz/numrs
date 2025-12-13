use crate::array::{Array, DTypeValue};
use crate::ops::{log, sum};
use anyhow::{bail, Result};

/// Computes the cross-entropy loss between predictions and targets.
///
/// Cross-entropy is computed as: -sum(targets * log(predictions))
///
/// This function assumes:
/// - `predictions` contains probabilities (values in [0, 1], summing to 1)
/// - `targets` contains one-hot encoded labels or probability distributions
///
/// # Arguments
///
/// * `predictions` - Predicted probability distribution (should sum to 1)
/// * `targets` - Target probability distribution (typically one-hot encoded)
///
/// # Returns
///
/// Scalar array [1] containing the cross-entropy loss
///
/// # Errors
///
/// Returns error if shapes don't match
///
/// # Examples
///
/// ```
/// use numrs::{Array, ops};
///
/// // Prediction for 3 classes: [0.7, 0.2, 0.1]
/// let predictions = Array::new(vec![3], vec![0.7, 0.2, 0.1]);
/// // True label is class 0: [1, 0, 0]
/// let targets = Array::new(vec![3], vec![1.0, 0.0, 0.0]);
/// 
/// let loss = ops::cross_entropy(&predictions, &targets)?;
/// // Loss = -log(0.7) ≈ 0.357
/// assert!(loss.data[0] > 0.3 && loss.data[0] < 0.4);
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// Perfect prediction:
/// ```
/// use numrs::{Array, ops};
///
/// let predictions = Array::new(vec![3], vec![1.0, 0.0, 0.0]);
/// let targets = Array::new(vec![3], vec![1.0, 0.0, 0.0]);
/// let loss = ops::cross_entropy(&predictions, &targets)?;
/// // Loss should be near 0
/// assert!(loss.data[0] < 0.001);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn cross_entropy<T1: DTypeValue, T2: DTypeValue>(
    predictions: &Array<T1>,
    targets: &Array<T2>,
) -> Result<Array> {
    // Validate shapes match
    if predictions.shape != targets.shape {
        bail!(
            "predictions shape {:?} doesn't match targets shape {:?}",
            predictions.shape,
            targets.shape
        );
    }
    
    // Convert both to f32 for computation (cross_entropy always works in f32)
    let pred_f32: Vec<f32> = predictions.data.iter().map(|&x| x.to_f32()).collect();
    let target_f32: Vec<f32> = targets.data.iter().map(|&x| x.to_f32()).collect();
    
    // Compute log of predictions (with numerical stability)
    const EPSILON: f32 = 1e-7;
    let clamped_data: Vec<f32> = pred_f32
        .iter()
        .map(|&p| p.clamp(EPSILON, 1.0 - EPSILON))
        .collect();
    
    let clamped = Array::new(predictions.shape.clone(), clamped_data);
    let log_preds: Array<f32> = log(&clamped)?;
    
    // Compute element-wise product: targets * log(predictions)
    let products: Vec<f32> = target_f32
        .iter()
        .zip(log_preds.data.iter())
        .map(|(&t, &lp)| t * lp)
        .collect();
    
    let product_array = Array::new(targets.shape.clone(), products);
    
    // Sum and negate
    let sum_result: Array<f32> = sum(&product_array, None)?;
    let loss = -sum_result.data[0];
    
    Ok(Array::new(vec![1], vec![loss]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_basic() -> Result<()> {
        // Perfect prediction
        let predictions = Array::new(vec![3], vec![1.0, 0.0, 0.0]);
        let targets = Array::new(vec![3], vec![1.0, 0.0, 0.0]);
        let loss = cross_entropy(&predictions, &targets)?;
        
        // Loss should be near 0 (actually slightly above due to epsilon)
        assert!(loss.data[0] < 0.001);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_incorrect_prediction() -> Result<()> {
        // Predicted class 0, but true label is class 2
        let predictions = Array::new(vec![3], vec![0.7, 0.2, 0.1]);
        let targets = Array::new(vec![3], vec![0.0, 0.0, 1.0]);
        let loss = cross_entropy(&predictions, &targets)?;
        
        // Loss = -log(0.1) ≈ 2.303
        assert!(loss.data[0] > 2.0);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_uniform() -> Result<()> {
        // Uniform prediction across 4 classes
        let predictions = Array::new(vec![4], vec![0.25, 0.25, 0.25, 0.25]);
        let targets = Array::new(vec![4], vec![1.0, 0.0, 0.0, 0.0]);
        let loss = cross_entropy(&predictions, &targets)?;
        
        // Loss = -log(0.25) ≈ 1.386
        assert!((loss.data[0] - 1.386).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_binary() -> Result<()> {
        // Binary classification
        let predictions = Array::new(vec![2], vec![0.8, 0.2]);
        let targets = Array::new(vec![2], vec![1.0, 0.0]);
        let loss = cross_entropy(&predictions, &targets)?;
        
        // Loss = -log(0.8) ≈ 0.223
        assert!((loss.data[0] - 0.223).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_soft_targets() -> Result<()> {
        // Soft targets (not one-hot)
        let predictions = Array::new(vec![3], vec![0.6, 0.3, 0.1]);
        let targets = Array::new(vec![3], vec![0.7, 0.2, 0.1]);
        let loss = cross_entropy(&predictions, &targets)?;
        
        // Should compute properly with soft targets
        assert!(loss.data[0] > 0.0);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_batch() -> Result<()> {
        // Batch of 2 samples with 3 classes each
        let predictions = Array::new(
            vec![6],
            vec![0.7, 0.2, 0.1, 0.1, 0.8, 0.1],
        );
        let targets = Array::new(
            vec![6],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let loss = cross_entropy(&predictions, &targets)?;
        
        // Average loss across batch
        assert!(loss.data[0] > 0.0);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_numerical_stability() -> Result<()> {
        // Test with very small probabilities (shouldn't panic)
        let predictions = Array::new(vec![3], vec![0.001, 0.001, 0.998]);
        let targets = Array::new(vec![3], vec![1.0, 0.0, 0.0]);
        let loss = cross_entropy(&predictions, &targets)?;
        
        // Loss should be large but finite
        assert!(loss.data[0] > 5.0 && loss.data[0] < 10.0);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_shape_mismatch() {
        let predictions = Array::new(vec![3], vec![0.5, 0.3, 0.2]);
        let targets = Array::new(vec![4], vec![1.0, 0.0, 0.0, 0.0]);
        let result = cross_entropy(&predictions, &targets);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_entropy_for_training() -> Result<()> {
        // Typical ML training scenario
        let logits_after_softmax = Array::new(vec![5], vec![0.1, 0.2, 0.5, 0.15, 0.05]);
        let one_hot_label = Array::new(vec![5], vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let loss = cross_entropy(&logits_after_softmax, &one_hot_label)?;
        
        // Loss = -log(0.5) ≈ 0.693
        assert!((loss.data[0] - 0.693).abs() < 0.01);
        Ok(())
    }
}

