
use crate::array::Array;
use anyhow::{Result, anyhow};
use crate::backend::cpu::random;

/// CPU Dropout Implementation
/// 
/// Randomly zeroes some elements of the input tensor with probability p using samples from 
/// a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.
/// 
/// # Arguments
/// * `input` - Input tensor
/// * `p` - Probability of an element to be zeroed. (0.0 means no dropout, 1.0 means all dropped)
/// * `training` - If false, returns input as-is (identity).
pub fn dropout(input: &Array, p: f32, training: bool) -> Result<Array> {
    // 1. Inference Mode: Identity
    if !training || p == 0.0 {
        return Ok(input.clone());
    }
    
    // 2. Validation
    if p < 0.0 || p > 1.0 {
        return Err(anyhow!("Dropout prob p must be between 0 and 1, got {}", p));
    }
    
    // 3. Training Mode: Inverted Dropout
    // Mask = Bernoulli(1-p)
    // Output = Input * Mask * (1 / (1-p))
    // This scales the result at training time so no scaling is needed at test time.
    
    let scale = 1.0 / (1.0 - p);
    let n = input.data.len();
    
    // Create mask buffer
    let mut mask_data = vec![0.0f32; n];
    random::bernoulli_mask(&mut mask_data, p);
    
    let input_data = &input.data;
    let mut output_data = Vec::with_capacity(n);
    
    // Apply mask and scale
    for i in 0..n {
        output_data.push(input_data[i] * mask_data[i] * scale);
    }
    
    Ok(Array::new(input.shape.clone(), output_data))
}
