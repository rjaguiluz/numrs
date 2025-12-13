//! AdaBound Optimizer
//! 
//! AdaBound es una variante de Adam que añade bounds dinámicos al learning rate,
//! transicionando gradualmente de Adam (adaptive) a SGD (constante) durante training.
//! Combina la convergencia rápida de Adam con la mejor generalización de SGD.
//! 
//! **Características principales:**
//! - Bounds dinámicos en el learning rate: [lr * (1 - 1/(γt + 1)), lr * (1 + 1/(γt))]
//! - Transición suave de comportamiento adaptativo a constante
//! - Mejor generalización que Adam puro en muchos casos
//! - Menos sensible a hyperparameters
//! 
//! **Cuándo usar AdaBound:**
//! - ✅ Cuando Adam overfitta o generaliza mal
//! - ✅ Para obtener lo mejor de Adam y SGD
//! - ✅ Training de modelos profundos
//! - ✅ Cuando quieres convergencia rápida + buena generalización
//! 
//! **Hiperparámetros típicos:**
//! - `lr`: 0.001
//! - `final_lr`: 0.1 (learning rate final, como SGD)
//! - `beta1`: 0.9
//! - `beta2`: 0.999
//! - `gamma`: 0.001 (controla velocidad de transición)

use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::{Tensor, Optimizer};
use crate::array::Array;
use anyhow::Result;

/// AdaBound Optimizer
pub struct AdaBound {
    /// Parámetros a optimizar
    params: Vec<Rc<RefCell<Tensor>>>,
    /// Learning rate inicial
    lr: f32,
    /// Learning rate final (como SGD)
    final_lr: f32,
    /// Exponential decay rate para first moment
    beta1: f32,
    /// Exponential decay rate para second moment
    beta2: f32,
    /// Epsilon para estabilidad numérica
    eps: f32,
    /// Weight decay (L2 regularization)
    weight_decay: f32,
    /// Gamma (controla velocidad de transición)
    gamma: f32,
    /// First moment estimates (momentum)
    m: Vec<Array>,
    /// Second moment estimates (variance)
    v: Vec<Array>,
    /// Timestep counter
    t: usize,
}

impl AdaBound {
    /// Crea un nuevo optimizer AdaBound
    /// 
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate inicial (típico: 0.001)
    /// - `final_lr`: Learning rate final (típico: 0.1)
    /// - `beta1`: Decay rate para first moment (típico: 0.9)
    /// - `beta2`: Decay rate para second moment (típico: 0.999)
    /// - `eps`: Epsilon para estabilidad numérica (típico: 1e-8)
    /// - `weight_decay`: L2 regularization (típico: 0.0)
    /// - `gamma`: Velocidad de transición (típico: 0.001)
    pub fn new(
        params: Vec<Rc<RefCell<Tensor>>>,
        lr: f32,
        final_lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        gamma: f32,
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
            final_lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            gamma,
            m,
            v,
            t: 0,
        }
    }
    
    /// Constructor con parámetros por defecto
    pub fn default(params: Vec<Rc<RefCell<Tensor>>>, lr: f32) -> Self {
        Self::new(params, lr, 0.1, 0.9, 0.999, 1e-8, 0.0, 0.001)
    }
}

impl Optimizer for AdaBound {
    fn step(&mut self) -> Result<()> {
        self.t += 1;
        
        // Compute dynamic bounds
        // Lower bound: lr * (1 - 1/(γt + 1))
        // Upper bound: lr * (1 + 1/(γt))
        let gamma_t = self.gamma * (self.t as f32);
        let lower_bound = self.final_lr * (1.0 - 1.0 / (gamma_t + 1.0));
        let upper_bound = self.final_lr * (1.0 + 1.0 / gamma_t.max(1e-8));
        
        for (i, param) in self.params.iter().enumerate() {
            let mut param_borrowed = param.borrow_mut();
            
            if let Some(grad_rc) = &param_borrowed.grad {
                // Extract gradient data first (drop immutable borrow)
                let grad_data: Vec<f32> = {
                    let grad = grad_rc.borrow();
                    if self.weight_decay > 0.0 {
                        grad.data.iter()
                            .zip(param_borrowed.data.data.iter())
                            .map(|(&g, &p)| g + self.weight_decay * p)
                            .collect()
                    } else {
                        grad.data.clone()
                    }
                };
                
                // Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
                // Update biased second moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                for (j, &g) in grad_data.iter().enumerate() {
                     self.m[i].data[j] = self.beta1 * self.m[i].data[j] 
                        + (1.0 - self.beta1) * g;
                     
                     self.v[i].data[j] = self.beta2 * self.v[i].data[j] 
                        + (1.0 - self.beta2) * g * g;
                }
                
                // Compute bias-corrected moments
                let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
                let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
                
                // Compute step size and apply bounds
                let step_size = self.lr * (bias_correction2.sqrt() / bias_correction1);
                
                // Update parameters with bounded learning rate
                for (j, p) in param_borrowed.data.data.iter_mut().enumerate() {
                    let m_hat = self.m[i].data[j] / bias_correction1;
                    let v_hat = self.v[i].data[j] / bias_correction2;
                    
                    // Compute effective learning rate
                    let denom = v_hat.sqrt() + self.eps;
                    let lr_t = step_size / denom;
                    
                    // Clip learning rate to bounds
                    let lr_clipped = lr_t.max(lower_bound).min(upper_bound);
                    
                    // Apply update
                    *p -= lr_clipped * m_hat;
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
