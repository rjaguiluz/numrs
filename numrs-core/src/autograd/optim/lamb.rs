//! LAMB (Layer-wise Adaptive Moments) Optimizer
//! 
//! LAMB es una extensión de LARS (Layer-wise Adaptive Rate Scaling) para Adam,
//! diseñado específicamente para entrenar modelos con large batch sizes.
//! 
//! **Características principales:**
//! - Adaptación layer-wise del learning rate
//! - Permite usar batch sizes muy grandes (>32k) sin degradar convergencia
//! - Combina Adam con normalización layer-wise
//! - Usado en BERT large-batch training
//! 
//! **Cuándo usar LAMB:**
//! - ✅ Training distribuido con large batches
//! - ✅ Cuando quieres escalar batch size sin re-tuning hyperparameters
//! - ✅ Large-scale pre-training (BERT, GPT, etc.)
//! - ❌ Small batch training (usa Adam en su lugar)
//! 
//! **Hiperparámetros típicos:**
//! - `lr`: 0.001 (ajustar según batch size)
//! - `beta1`: 0.9
//! - `beta2`: 0.999
//! - `weight_decay`: 0.01
//! - `eps`: 1e-6

use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::{Tensor, Optimizer};
use crate::array::Array;
use anyhow::Result;

/// LAMB Optimizer
pub struct LAMB {
    /// Parámetros a optimizar
    params: Vec<Rc<RefCell<Tensor>>>,
    /// Learning rate
    lr: f32,
    /// Exponential decay rate para first moment
    beta1: f32,
    /// Exponential decay rate para second moment
    beta2: f32,
    /// Epsilon para estabilidad numérica
    eps: f32,
    /// Weight decay (L2 regularization)
    weight_decay: f32,
    /// First moment estimates (momentum)
    m: Vec<Array>,
    /// Second moment estimates (variance)
    v: Vec<Array>,
    /// Timestep counter
    t: usize,
}

impl LAMB {
    /// Crea un nuevo optimizer LAMB
    /// 
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate (típico: 0.001)
    /// - `beta1`: Decay rate para first moment (típico: 0.9)
    /// - `beta2`: Decay rate para second moment (típico: 0.999)
    /// - `eps`: Epsilon para estabilidad numérica (típico: 1e-6)
    /// - `weight_decay`: L2 regularization (típico: 0.01)
    pub fn new(
        params: Vec<Rc<RefCell<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let m = params.iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();
        
        let v = params.iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();
        
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }
    
    /// Constructor con parámetros por defecto
    pub fn default(params: Vec<Rc<RefCell<Tensor>>>, lr: f32) -> Self {
        Self::new(params, lr, 0.9, 0.999, 1e-6, 0.01)
    }
}

impl Optimizer for LAMB {
    fn step(&mut self) -> Result<()> {
        self.t += 1;
        
        for (i, param) in self.params.iter().enumerate() {
            let mut param_borrowed = param.borrow_mut();
            
            if let Some(grad_rc) = &param_borrowed.grad {
                // Extract gradient data first (drop immutable borrow)
                let grad_data: Vec<f32> = {
                    let grad = grad_rc.borrow();
                    grad.data.clone()
                };
                
                // Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
                // Update biased second moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                for j in 0..self.m[i].data.len() {
                    self.m[i].data[j] = self.beta1 * self.m[i].data[j] 
                        + (1.0 - self.beta1) * grad_data[j];
                    
                    self.v[i].data[j] = self.beta2 * self.v[i].data[j] 
                        + (1.0 - self.beta2) * grad_data[j] * grad_data[j];
                }
                
                // Compute bias-corrected first moment: m̂_t = m_t / (1 - β1^t)
                let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
                
                // Compute Adam update: r_t = m̂_t / (√v̂_t + ε)
                // We combine map+collect for efficiency
                let mut adam_update: Vec<f32> = self.m[i].data.iter()
                    .zip(self.v[i].data.iter())
                    .map(|(m_val, v_val)| {
                        let m_hat = m_val / bias_correction1;
                        let v_hat = v_val / bias_correction2;
                        m_hat / (v_hat.sqrt() + self.eps)
                    })
                    .collect();
                
                // Add weight decay: r_t = r_t + λ * w_t
                if self.weight_decay > 0.0 {
                    for (u, p) in adam_update.iter_mut().zip(param_borrowed.data.data.iter()) {
                        *u += self.weight_decay * p;
                    }
                }
                
                // Compute layer-wise norms
                let weight_norm: f32 = param_borrowed.data.data.iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                
                let update_norm: f32 = adam_update.iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                
                // Compute trust ratio: φ = ||w|| / ||r_t||
                let trust_ratio = if weight_norm > 0.0 && update_norm > 0.0 {
                    weight_norm / update_norm
                } else {
                    1.0
                };
                
                // Apply update: w_{t+1} = w_t - lr * φ * r_t
                for (p, u) in param_borrowed.data.data.iter_mut().zip(adam_update.iter()) {
                    *p -= self.lr * trust_ratio * u;
                }
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&self) {
        for param_ref in &self.params {
            param_ref.borrow().zero_grad();
        }
    }
    
    fn learning_rate(&self) -> f32 {
        self.lr
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
    
    fn num_params(&self) -> usize {
        self.params.len()
    }
}
