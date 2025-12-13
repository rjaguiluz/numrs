
use crate::array::Array;
use anyhow::Result;

/// Naive Conv1D Implementation for CPU Fallback
pub fn conv1d_naive(
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    stride: usize,
    padding: usize,
) -> Result<Array> {
    // 1. Validation (Already done in ops, but good for safety)
    let batch_size = input.shape[0];
    let in_channels = input.shape[1];
    let input_len = input.shape[2];
    
    let out_channels = weight.shape[0];
    let kernel_size = weight.shape[2];
    
    // 2. Output Shape
    // out_len = (in_len + 2*padding - kernel_size) / stride + 1
    let out_len = ((input_len + 2 * padding).saturating_sub(kernel_size)) / stride + 1;
    let output_shape = vec![batch_size, out_channels, out_len];
    
    // 3. Execution (Naive Nested Loops)
    let mut output_data = vec![0.0; batch_size * out_channels * out_len];
    
    let input_data = &input.data;
    let weight_data = &weight.data;
    
    // Optimization: Pre-fetch bias values
    let bias_data = bias.map(|b| &b.data);

    for b in 0..batch_size {
        for oc in 0..out_channels {
            let bias_val = if let Some(bd) = bias_data { bd[oc] } else { 0.0 };
            
            for ol in 0..out_len {
                let mut sum = bias_val;
                
                // Input window start index (in "virtual" padded array)
                // virtual_idx = ol * stride
                // real_idx = virtual_idx - padding
                let input_start_idx = (ol * stride) as isize - padding as isize;
                
                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                        let current_input_idx = input_start_idx + k as isize;
                        
                        if current_input_idx >= 0 && current_input_idx < input_len as isize {
                            let val_in = input_data[
                                b * (in_channels * input_len) + 
                                ic * input_len + 
                                current_input_idx as usize
                            ];
                            let val_w = weight_data[
                                oc * (in_channels * kernel_size) + 
                                ic * kernel_size + 
                                k
                            ];
                            sum += val_in * val_w;
                        }
                    }
                }
                
                output_data[
                    b * (out_channels * out_len) + 
                    oc * out_len + 
                    ol
                ] = sum;
            }
        }
    }
    
    Ok(Array::new(output_shape, output_data))
}

/// Naive Conv1D Backward Implementation
/// Returns (grad_input, grad_weight, grad_bias)
pub fn conv1d_backward_naive(
    grad_output: &Array,
    input: &Array,
    weight: &Array,
    stride: usize,
    padding: usize,
) -> Result<(Array, Array, Option<Array>)> {
    let batch_size = input.shape[0];
    let in_channels = input.shape[1];
    let input_len = input.shape[2];
    
    let out_channels = weight.shape[0];
    let kernel_size = weight.shape[2];
    let out_len = grad_output.shape[2];
    
    // Gradients
    let mut grad_input_data = vec![0.0; batch_size * in_channels * input_len];
    let mut grad_weight_data = vec![0.0; out_channels * in_channels * kernel_size];
    let mut grad_bias_data = vec![0.0; out_channels];
    
    let grad_out_data = &grad_output.data;
    let input_data = &input.data;
    let weight_data = &weight.data;

    // Loop over batch and output spatial
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for ol in 0..out_len {
                let grad_val = grad_out_data[
                    b * (out_channels * out_len) + oc * out_len + ol
                ];
                
                // Accumulate grad_bias
                grad_bias_data[oc] += grad_val;
                
                // Mapped input start
                let input_start_idx = (ol * stride) as isize - padding as isize;
                
                for ic in 0..in_channels {
                    for k in 0..kernel_size {
                         let current_input_idx = input_start_idx + k as isize;
                         
                         // Valid input position?
                         if current_input_idx >= 0 && current_input_idx < input_len as isize {
                             let idx_in = b * (in_channels * input_len) + ic * input_len + current_input_idx as usize;
                             let idx_w = oc * (in_channels * kernel_size) + ic * kernel_size + k;
                             
                             // dL/dInput += grad_output * weight
                             grad_input_data[idx_in] += grad_val * weight_data[idx_w];
                             
                             // dL/dWeight += grad_output * input
                             grad_weight_data[idx_w] += grad_val * input_data[idx_in];
                         }
                    }
                }
            }
        }
    }
    
    let grad_input = Array::new(input.shape.clone(), grad_input_data);
    let grad_weight = Array::new(weight.shape.clone(), grad_weight_data);
    let grad_bias = Some(Array::new(vec![out_channels], grad_bias_data));
    
    Ok((grad_input, grad_weight, grad_bias))
}
