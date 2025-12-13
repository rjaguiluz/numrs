//! Dropout Operations

use crate::array::Array;
use anyhow::Result;

/// Dropout
///
/// During training, randomly zeroes some element of the input tensor with probability p 
/// using samples from a Bernoulli distribution.
/// 
/// # Arguments
/// * `input` - Input tensor
/// * `p` - Probability of an element to be zeroed. (0.0 means no dropout)
/// * `training` - If false, returns input as-is.
pub fn dropout(input: &Array, p: f32, training: bool) -> Result<Array> {
    // Dispatch
    
    // Check for GPU
    #[cfg(numrs_kernel_dropout_gpu)]
    {
        if crate::backend::webgpu::is_available_cached() {
            return crate::backend::webgpu::dropout::dropout_webgpu(input, p, training);
        }
    }
    
    // Fallback: CPU
    // Note: SIMD for dropout requires vectorizable RNG which is complex.
    // Standard CPU implementation is usually sufficient or can be parallelized with Rayon.
    crate::backend::cpu::dropout::dropout(input, p, training)
}
