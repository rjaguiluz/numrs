//! Convolution operations
//! 
//! Provides 1D, 2D, and 3D convolutions with multi-backend support.

use crate::array::Array;
use anyhow::{Result, anyhow};

/// 1D Convolution
/// 
/// Applies a 1D convolution over an input signal composed of several input planes.
/// 
/// # Arguments
/// * `input` - Input tensor of shape [Batch, InChannels, Length]
/// * `weight` - Filters of shape [OutChannels, InChannels, KernelSize]
/// * `bias` - Optional bias of shape [OutChannels]
/// * `stride` - Stride of the convolution. Default: 1
/// * `padding` - Zero-padding added to both sides of the input. Default: 0
/// 
/// # Returns
/// Output tensor of shape [Batch, OutChannels, OutLength]
pub fn conv1d(
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    stride: usize,
    padding: usize,
) -> Result<Array> {
    // 1. Validation
    if input.shape.len() != 3 {
        return Err(anyhow!("Conv1D input must be 3D [Batch, InChannels, Length]"));
    }
    if weight.shape.len() != 3 {
        return Err(anyhow!("Conv1D weight must be 3D [OutChannels, InChannels, KernelSize]"));
    }
    
    let in_channels = input.shape[1];
    let kernel_in_channels = weight.shape[1];
    
    if in_channels != kernel_in_channels {
        return Err(anyhow!("Input channels ({}) mismatch kernel channels ({})", in_channels, kernel_in_channels));
    }
    
    if let Some(b) = bias {
         if b.shape.len() != 1 {
             return Err(anyhow!("Bias must be 1D"));
         }
         if b.shape[0] != weight.shape[0] {
             return Err(anyhow!("Bias size ({}) mismatch output channels ({})", b.shape[0], weight.shape[0]));
         }
    }

    // 2. Dispatch
    // Check for GPU availability
    #[cfg(numrs_kernel_conv_gpu)]
    {
        if crate::backend::webgpu::is_available_cached() {
            return crate::backend::webgpu::conv::conv1d_webgpu(input, weight, bias, stride, padding);
        }
    }

    // Check for SIMD (AVX)
    #[cfg(numrs_kernel_conv_simd)]
    {
        // TODO: Validate SIMD support at runtime
        return crate::backend::cpu::simd::conv1d_simd(input, weight, bias, stride, padding);
    }

    // Fallback: CPU Naive
    crate::backend::cpu::conv::conv1d_naive(input, weight, bias, stride, padding)
}
