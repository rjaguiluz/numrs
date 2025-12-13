
use crate::array::Array;
use anyhow::{Result, anyhow};

/// BatchNorm1D Training Implementation (CPU Naive)
/// 
/// Input: [Batch, Channels, Length]
/// Weight/Bias/RunningStats: [Channels]
/// Output: same as Input
pub fn batch_norm_1d_training(
    input: &Array,
    running_mean: &mut Array,
    running_var: &mut Array,
    weight: &Array,
    bias: &Array,
    momentum: f32,
    eps: f32
) -> Result<Array> {
    if input.shape.len() != 3 {
        return Err(anyhow!("BatchNorm1D input must be 3D [Batch, Channels, Length]"));
    }
    
    let batch_size = input.shape[0];
    let channels = input.shape[1];
    let length = input.shape[2];
    
    // Check shapes
    if weight.shape[0] != channels || bias.shape[0] != channels {
        return Err(anyhow!("Weight/Bias size mismatch channels"));
    }
    
    // 1. Calculate Batch Statistics [Channels]
    // Mean(c) = sum(x[b,c,l]) / (B*L)
    let mut batch_mean = vec![0.0; channels];
    let mut batch_var = vec![0.0; channels];
    let num_elements = (batch_size * length) as f32;
    
    let input_data = &input.data;
    
    // Pass 1: Mean
    for c in 0..channels {
        let mut sum = 0.0;
        for b in 0..batch_size {
            for l in 0..length {
                let idx = b * (channels * length) + c * length + l;
                sum += input_data[idx];
            }
        }
        batch_mean[c] = sum / num_elements;
    }
    
    // Pass 2: Variance
    for c in 0..channels {
        let mut sum_sq_diff = 0.0;
        let mean = batch_mean[c];
        for b in 0..batch_size {
            for l in 0..length {
                let idx = b * (channels * length) + c * length + l;
                let diff = input_data[idx] - mean;
                sum_sq_diff += diff * diff;
            }
        }
        batch_var[c] = sum_sq_diff / num_elements;
    }
    
    // 2. Update Running Stats (Momentum)
    // running = (1 - momentum) * running + momentum * batch
    // Note: PyTorch default momentum is 0.1, meaning 0.1 of new data.
    // Here we assume 'momentum' arg adheres to library convention.
    for c in 0..channels {
        running_mean.data[c] = (1.0 - momentum) * running_mean.data[c] + momentum * batch_mean[c];
        // Unbiased var for running stats typically, but let's stick to simple biased update for now
        // or check if Bessel correction is needed for running_var update (usually yes)
        let unbiased_var = batch_var[c] * num_elements / (num_elements - 1.0);  
        running_var.data[c] = (1.0 - momentum) * running_var.data[c] + momentum * unbiased_var;
    }
    
    // 3. Normalize and Scale/Shift
    // y = (x - mean) / sqrt(var + eps) * weight + bias
    let mut output_data = vec![0.0; input.data.len()];
    
    let weight_data = &weight.data;
    let bias_data = &bias.data;
    
    for c in 0..channels {
        let mean = batch_mean[c];
        let var = batch_var[c];
        let inv_std = 1.0 / (var + eps).sqrt();
        let w = weight_data[c];
        let b = bias_data[c];
        
        for i in 0..batch_size {
            for l in 0..length {
                let idx = i * (channels * length) + c * length + l;
                let val = input_data[idx];
                let normalized = (val - mean) * inv_std;
                output_data[idx] = normalized * w + b;
            }
        }
    }
    
    Ok(Array::new(input.shape.clone(), output_data))
}

/// BatchNorm1D Inference Implementation (CPU Naive)
/// Uses running stats instead of batch stats.
pub fn batch_norm_1d_inference(
    input: &Array,
    running_mean: &Array,
    running_var: &Array,
    weight: &Array,
    bias: &Array,
    eps: f32
) -> Result<Array> {
    if input.shape.len() != 3 {
        return Err(anyhow!("BatchNorm1D input must be 3D"));
    }
    let channels = input.shape[1];
    
    let mut output_data = vec![0.0; input.data.len()];
    
    let input_data = &input.data;
    let mean_data = &running_mean.data;
    let var_data = &running_var.data;
    let weight_data = &weight.data;
    let bias_data = &bias.data;
    
    let batch_size = input.shape[0];
    let length = input.shape[2];

    for c in 0..channels {
        let mean = mean_data[c];
        let var = var_data[c];
        let inv_std = 1.0 / (var + eps).sqrt();
        let w = weight_data[c];
        let b = bias_data[c];
        
        for i in 0..batch_size {
            for l in 0..length {
                let idx = i * (channels * length) + c * length + l;
                let val = input_data[idx];
                let normalized = (val - mean) * inv_std;
                output_data[idx] = normalized * w + b;
            }
        }
    }
    
    Ok(Array::new(input.shape.clone(), output_data))
}

/// BatchNorm Backward Naive (Training Mode)
/// Returns (grad_input, grad_weight, grad_bias)
pub fn batchnorm_backward_naive(
    grad_output: &Array,
    input: &Array,
    weight: &Array,
    bias: &Array,
    eps: f32,
) -> Result<(Array, Array, Array)> {
     if input.shape.len() != 3 {
        return Err(anyhow!("BatchNorm1D input must be 3D"));
    }
    
    let batch_size = input.shape[0];
    let channels = input.shape[1];
    let length = input.shape[2];
    let num_elements = (batch_size * length) as f32;
    
    // Grads
    let mut grad_input_data = vec![0.0; input.data.len()];
    let mut grad_weight_data = vec![0.0; channels];
    let mut grad_bias_data = vec![0.0; channels];
    
    let grad_out_data = &grad_output.data;
    let input_data = &input.data;
    let weight_data = &weight.data; // Gamma
    
    // Process per channel
    for c in 0..channels {
        // 1. Calculate Mean and Variance (of Input)
        let mut sum = 0.0;
        for b in 0..batch_size {
            for l in 0..length {
                let idx = b * (channels * length) + c * length + l;
                sum += input_data[idx];
            }
        }
        let mean = sum / num_elements;
        
        let mut sum_sq_diff = 0.0;
        for b in 0..batch_size {
            for l in 0..length {
                 let idx = b * (channels * length) + c * length + l;
                 let diff = input_data[idx] - mean;
                 sum_sq_diff += diff * diff;
            }
        }
        let var = sum_sq_diff / num_elements;
        let std = (var + eps).sqrt();
        let inv_std = 1.0 / std;
        
        // 2. Calculate intermediates
        let mut sum_grad_out = 0.0;
        let mut sum_grad_out_x_hat = 0.0;
        
        for b in 0..batch_size {
            for l in 0..length {
                let idx = b * (channels * length) + c * length + l;
                let val = input_data[idx];
                let go = grad_out_data[idx];
                let x_hat = (val - mean) * inv_std;
                
                sum_grad_out += go;
                sum_grad_out_x_hat += go * x_hat;
            }
        }
        
        // 3. Gradients for Params
        grad_bias_data[c] = sum_grad_out;
        grad_weight_data[c] = sum_grad_out_x_hat; // Before multiplying by gamma? No, dL/dGamma = sum(dL/dy * x_hat) = sum(grad_out * x_hat)
        
        // 4. Gradient for Input
        // dL/dx = (gamma / (N * std)) * (N * dL/dx_hat - sum(dL/dx_hat) - x_hat * sum(dL/dx_hat * x_hat))
        // But dL/dx_hat = grad_out * gamma? No dL/dy = grad_out. y = gamma * x_hat + beta.
        // So dL/dx_hat = grad_out * gamma.
        
        // Let's use clean formula:
        // dx_hat = grad_out * gamma
        // dvar = sum(dx_hat * (x - mean) * -0.5 * (var + eps)^-1.5)
        // dmean = sum(dx_hat * -inv_std) + dvar * sum(-2 * (x - mean)) / N
        // dx = dx_hat * inv_std + dvar * 2 * (x - mean) / N + dmean / N
        
        // Optimized formula:
        // dx = (gamma / (N * std)) * (N * grad_out - sum_grad_out - x_hat * sum_grad_out_x_hat)
        
        let gamma = weight_data[c];
        let factor = gamma / (num_elements * std);
        
        for b in 0..batch_size {
            for l in 0..length {
                 let idx = b * (channels * length) + c * length + l;
                 let val = input_data[idx];
                 let go = grad_out_data[idx];
                 let x_hat = (val - mean) * inv_std;
                 
                 let num = num_elements * go - sum_grad_out - x_hat * sum_grad_out_x_hat;
                 grad_input_data[idx] = factor * num;
            }
        }
    }
    
    Ok((
        Array::new(input.shape.clone(), grad_input_data),
        Array::new(weight.shape.clone(), grad_weight_data),
        Array::new(bias.shape.clone(), grad_bias_data)
    ))
}
