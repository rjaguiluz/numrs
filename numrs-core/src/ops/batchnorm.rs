//! Batch Normalization Operations

use crate::array::Array;
use anyhow::Result;

/// Backward compatibility alias
pub use batch_norm as batch_norm_1d;

/// Batch Normalization 1D
///
/// Applies Batch Normalization over a 3D input (Batch, Channels, Length).
/// 
/// # Arguments
/// * `input` - Input tensor [B, C, L]
/// * `running_mean` - Running mean stats [C] (In-place update during training)
/// * `running_var` - Running variance stats [C] (In-place update during training)
/// * `weight` - Learnable gamma [C]
/// * `bias` - Learnable beta [C]
/// * `training` - If true, uses batch stats and updates running stats. If false, uses running stats.
/// * `momentum` - Momentum for running stats check (default use 0.1)
/// * `eps` - Epsilon for stability
pub fn batch_norm(
    input: &Array,
    running_mean: &mut Array,
    running_var: &mut Array,
    weight: &Array,
    bias: &Array,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Result<Array> {
    
    // Dispatch
    if training {
        // TODO: Dispatch to GPU/SIMD if available
        #[cfg(numrs_kernel_batchnorm_gpu)]
        {
             if crate::backend::webgpu::is_available_cached() {
                 return crate::backend::webgpu::batchnorm::batch_norm_1d_training_webgpu(
                     input, running_mean, running_var, weight, bias, momentum, eps
                 );
             }
        }
        
        crate::backend::cpu::batchnorm::batch_norm_1d_training(
            input, running_mean, running_var, weight, bias, momentum, eps
        )
    } else {
        // GPU dispatch for inference
        #[cfg(numrs_kernel_batchnorm_gpu)]
        {
             if crate::backend::webgpu::is_available_cached() {
                 return crate::backend::webgpu::batchnorm::batch_norm_1d_inference_webgpu(
                     input, running_mean, running_var, weight, bias, eps
                 );
             }
        }
        
        crate::backend::cpu::batchnorm::batch_norm_1d_inference(
            input, running_mean, running_var, weight, bias, eps
        )
    }
}
