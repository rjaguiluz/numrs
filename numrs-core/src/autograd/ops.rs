//! Operaciones con soporte para autograd
//! 
//! Cada operación:
//! 1. Ejecuta forward pass usando ops de NumRs
//! 2. Si autograd está habilitado, registra backward function
//! 3. Crea nuevo Tensor con compute node

use crate::array::Array;
use crate::autograd::{Tensor, ComputeNode, OpKind, is_grad_enabled};
use crate::autograd::backward;
use crate::ops;
use anyhow::Result;

impl Tensor {
    /// Add: self + other
    pub fn add(&self, other: &Tensor) -> Result<Self> {
        // Forward pass
        let result = ops::add(&self.data, &other.data)?;
        
        // Si autograd deshabilitado o ninguno requiere grad
        if !is_grad_enabled() || (!self.requires_grad && !other.requires_grad) {
            return Ok(Tensor::new(result, false));
        }
        
        // Crear compute node con backward function
        let requires_grad = self.requires_grad || other.requires_grad;
        let backward_fn = Box::new(backward::add_backward);
        
        let node = ComputeNode::new(
            OpKind::Add,
            vec![self.clone(), other.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, requires_grad))
    }

    /// Sub: self - other
    pub fn sub(&self, other: &Tensor) -> Result<Self> {
        let result = ops::sub(&self.data, &other.data)?;
        
        if !is_grad_enabled() || (!self.requires_grad && !other.requires_grad) {
            return Ok(Tensor::new(result, false));
        }
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let backward_fn = Box::new(backward::sub_backward);
        
        let node = ComputeNode::new(
            OpKind::Sub,
            vec![self.clone(), other.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, requires_grad))
    }
    
    /// Mul (elementwise): self * other
    pub fn mul(&self, other: &Tensor) -> Result<Self> {
        // Forward pass (elementwise multiply)
        // Use ops::mul to handle broadcasting and promotion automatically
        let result = ops::mul(&self.data, &other.data)?;
        
        if !is_grad_enabled() || (!self.requires_grad && !other.requires_grad) {
            return Ok(Tensor::new(result, false));
        }
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let backward_fn = Box::new(backward::mul_backward);
        
        let node = ComputeNode::new(
            OpKind::Mul,
            vec![self.clone(), other.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, requires_grad))
    }

    /// Div: self / other
    pub fn div(&self, other: &Tensor) -> Result<Self> {
        let result = ops::div(&self.data, &other.data)?;
        
        if !is_grad_enabled() || (!self.requires_grad && !other.requires_grad) {
            return Ok(Tensor::new(result, false));
        }
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let backward_fn = Box::new(backward::div_backward);
        
        let node = ComputeNode::new(
            OpKind::Div,
            vec![self.clone(), other.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, requires_grad))
    }
    
    /// MatMul: self @ other
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        // Forward pass
        let result = ops::matmul(&self.data, &other.data)?;
        
        if !is_grad_enabled() || (!self.requires_grad && !other.requires_grad) {
            return Ok(Tensor::new(result, false));
        }
        
        let requires_grad = self.requires_grad || other.requires_grad;
        let backward_fn = Box::new(backward::matmul_backward);
        
        let node = ComputeNode::new(
            OpKind::MatMul,
            vec![self.clone(), other.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, requires_grad))
    }
    
    /// ReLU: max(0, self)
    pub fn relu(&self) -> Result<Self> {
        // Forward pass
        let result = Array::new(
            self.data.shape.clone(),
            self.data.data.iter().map(|&x| x.max(0.0)).collect()
        );
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::relu_backward);
        
        let node = ComputeNode::new(
            OpKind::ReLU,
            vec![self.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, true))
    }
    
    /// Sigmoid: 1 / (1 + exp(-self))
    pub fn sigmoid(&self) -> Result<Self> {
        // Forward pass
        let result = Array::new(
            self.data.shape.clone(),
            self.data.data.iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect()
        );
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::sigmoid_backward);
        
        let node = ComputeNode::new(
            OpKind::Sigmoid,
            vec![self.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, true))
    }
    
    /// Exp: exp(self)
    pub fn exp(&self) -> Result<Self> {
        let result = Array::new(
            self.data.shape.clone(),
            self.data.data.iter().map(|&x| x.exp()).collect()
        );
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::exp_backward);
        
        let node = ComputeNode::new(
            OpKind::Exp,
            vec![self.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, true))
    }
    
    /// Log: log(self)
    pub fn log(&self) -> Result<Self> {
        let result = Array::new(
            self.data.shape.clone(),
            self.data.data.iter().map(|&x| x.ln()).collect()
        );
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::log_backward);
        
        let node = ComputeNode::new(
            OpKind::Log,
            vec![self.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, true))
    }
    
    /// Sum: suma todos los elementos
    pub fn sum(&self) -> Result<Self> {
        // Forward pass
        let result = ops::sum(&self.data, None)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::sum_backward);
        
        let node = ComputeNode::new(
            OpKind::Sum { axis: None },
            vec![self.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, true))
    }
    
    /// Mean: promedio de todos los elementos
    pub fn mean(&self) -> Result<Self> {
        let sum_val: f32 = self.data.data.iter().sum();
        let n = self.data.data.len() as f32;
        let result = Array::new(vec![1], vec![sum_val / n]);
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::mean_backward);
        
        let node = ComputeNode::new(
            OpKind::Mean { axis: None },
            vec![self.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, true))
    }
    
    /// MSE Loss: mean((self - target)^2)
    pub fn mse_loss(&self, target: &Tensor) -> Result<Self> {
        // Forward pass
        let diff_squared: f32 = self.data.data.iter()
            .zip(target.data.data.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        let n = self.data.data.len() as f32;
        let result = Array::new(vec![1], vec![diff_squared / n]);
        
        if !is_grad_enabled() || (!self.requires_grad && !target.requires_grad) {
            return Ok(Tensor::new(result, false));
        }
        
        let requires_grad = self.requires_grad || target.requires_grad;
        let backward_fn = Box::new(backward::mse_backward);
        
        let node = ComputeNode::new(
            OpKind::MSE,
            vec![self.clone(), target.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, requires_grad))
    }
    
    /// Cross-entropy loss con softmax integrado (Batch-aware)
    pub fn cross_entropy_loss(&self, target: &Tensor) -> Result<Self> {
        // Validation
        if self.data.shape.len() != 2 || target.data.shape.len() != 2 {
            return Err(anyhow::anyhow!("CrossEntropy expect 2D tensors [batch, classes]"));
        }
        
        let batch_size = self.data.shape[0];
        let num_classes = self.data.shape[1];
        
        let mut total_loss = 0.0;
        
        // Iterar row-wise (sample por sample)
        for i in 0..batch_size {
            let start = i * num_classes;
            let end = start + num_classes;
            let logits = &self.data.data[start..end];
            let targets = &target.data.data[start..end];
            
            // Softmax per sample
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|x| (x - max_val).exp()).sum();
            
            // Cross-entropy per sample: -sum(target * log(softmax))
            // log(exp(x-max) / sum) = (x-max) - log(sum)
            let sample_loss: f32 = logits.iter()
                .zip(targets.iter())
                .map(|(&x, &t)| {
                    let log_softmax = (x - max_val) - exp_sum.ln();
                    -t * log_softmax
                })
                .sum();
                
            total_loss += sample_loss;
        }
        
        // Mean Loss over batch
        let mean_loss = total_loss / batch_size as f32;
        let result = Array::new(vec![1], vec![mean_loss]);
        
        if !is_grad_enabled() || (!self.requires_grad && !target.requires_grad) {
            return Ok(Tensor::new(result, false));
        }
        
        let requires_grad = self.requires_grad || target.requires_grad;
        let backward_fn = Box::new(backward::cross_entropy_backward);
        
        let node = ComputeNode::new(
            OpKind::CrossEntropy,
            vec![self.clone(), target.clone()],
            Some(backward_fn),
        );
        
        Ok(Tensor::from_operation(result, node, requires_grad))
    }

    /// Flatten: flatten(start, end)
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        let result = ops::flatten(&self.data, start_dim, end_dim)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::flatten_backward);
        let node = ComputeNode::new(
            OpKind::Flatten { start_dim, end_dim },
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Reshape: input.reshape([d1, d2, ...])
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        let shape_isize: Vec<isize> = shape.iter().map(|&x| x as isize).collect();
        let result = ops::reshape(&self.data, &shape_isize)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::reshape_backward);
        let node = ComputeNode::new(
            OpKind::Reshape { shape },
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Conv1D
    pub fn conv1d(&self, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Result<Self> {
        let bias_data = bias.map(|b| &b.data);
        let result = ops::conv::conv1d(&self.data, &weight.data, bias_data, stride, padding)?;
        
        let mut inputs = vec![self.clone(), weight.clone()];
        let mut requires_grad = self.requires_grad || weight.requires_grad;
        
        if let Some(b) = bias {
            inputs.push(b.clone());
            requires_grad = requires_grad || b.requires_grad;
        }
        
        if !is_grad_enabled() || !requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::conv1d_backward);
        let node = ComputeNode::new(
            OpKind::Conv1D { stride, padding },
            inputs,
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// BatchNorm
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm(&self, running_mean: &mut Tensor, running_var: &mut Tensor, weight: &Tensor, bias: &Tensor, training: bool, momentum: f32, eps: f32) -> Result<Self> {
        let result = ops::batchnorm::batch_norm(
            &self.data, 
            &mut running_mean.data, 
            &mut running_var.data, 
            &weight.data, 
            &bias.data, 
            training, 
            momentum, 
            eps
        )?;
        
        let requires_grad = self.requires_grad || weight.requires_grad || bias.requires_grad;
        
        if !is_grad_enabled() || !requires_grad {
             return Ok(Tensor::new(result, false));
        }
        
        // Note: Running stats are not part of gradient computation graph usually, but updated in-place.
        // However, for ONNX export, they MUST be present as inputs to the node.
        let inputs = vec![
            self.clone(), 
            weight.clone(), 
            bias.clone(),
            running_mean.clone(),
            running_var.clone()
        ];
        let backward_fn = Box::new(backward::batchnorm_backward);
        
        let node = ComputeNode::new(
            OpKind::BatchNorm { training, momentum, eps },
            inputs,
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Dropout
    pub fn dropout(&self, p: f32, training: bool) -> Result<Self> {
        let result = ops::dropout::dropout(&self.data, p, training)?;
        
        if !is_grad_enabled() || !self.requires_grad {
             return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::dropout_backward);
        let node = ComputeNode::new(
            OpKind::Dropout { p, training },
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Pow: self^exponent
    pub fn pow(&self, exponent: f32) -> Result<Self> {
        let exponent_arr = Array::new(vec![1], vec![exponent]);
        let result = ops::pow(&self.data, &exponent_arr)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::pow_backward);
        let node = ComputeNode::new(
            OpKind::Pow(exponent),
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Sqrt: sqrt(self)
    pub fn sqrt(&self) -> Result<Self> {
        let result = ops::sqrt(&self.data)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::sqrt_backward);
        let node = ComputeNode::new(
            OpKind::Sqrt,
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Sin: sin(self)
    pub fn sin(&self) -> Result<Self> {
        let result = ops::sin(&self.data)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::sin_backward);
        let node = ComputeNode::new(
            OpKind::Sin,
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Cos: cos(self)
    pub fn cos(&self) -> Result<Self> {
        let result = ops::cos(&self.data)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::cos_backward);
        let node = ComputeNode::new(
            OpKind::Cos,
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Tan: tan(self)
    pub fn tan(&self) -> Result<Self> {
        let result = ops::tan(&self.data)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::tan_backward);
        let node = ComputeNode::new(
            OpKind::Tan,
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    /// Tanh: tanh(self)
    pub fn tanh(&self) -> Result<Self> {
        let result = ops::tanh(&self.data)?;
        
        if !is_grad_enabled() || !self.requires_grad {
            return Ok(Tensor::new(result, false));
        }
        
        let backward_fn = Box::new(backward::tanh_backward);
        let node = ComputeNode::new(
            OpKind::Tanh,
            vec![self.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }
}
