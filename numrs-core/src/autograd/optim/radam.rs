//! RAdam optimizer (Rectified Adam)

use super::base::{Optimizer, apply_weight_decay, get_gradient};
use crate::autograd::Tensor;
use crate::array::Array;
use anyhow::Result;
use std::rc::Rc;
use std::cell::RefCell;

/// RAdam optimizer - Adam con adaptive learning rate rectification
/// 
/// Soluciona el problema de warm-up en Adam aplicando una rectificación automática
/// del learning rate durante las primeras iteraciones. Elimina la necesidad de
/// warm-up manual.
/// 
/// Ventajas:
/// - No requiere warm-up manual
/// - Converge de forma más estable en early training
/// - Resultados similares o mejores que Adam + warm-up
/// 
/// # Ejemplo
/// ```ignore
/// let optimizer = RAdam::new(model.parameters_mut(), 0.001, 0.9, 0.999, 1e-8, 0.0);
/// ```
pub struct RAdam {
    params: Vec<Rc<RefCell<Tensor>>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    
    /// First moment estimates (momentum)
    m: Vec<Array>,
    
    /// Second moment estimates (adaptive learning rates)
    v: Vec<Array>,
    
    /// Time step
    t: u32,
}

impl RAdam {
    /// Crea un nuevo optimizer RAdam
    /// 
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate (típico: 0.001)
    /// - `beta1`: Exponential decay rate para first moment (típico: 0.9)
    /// - `beta2`: Exponential decay rate para second moment (típico: 0.999)
    /// - `eps`: Epsilon para estabilidad numérica (típico: 1e-8)
    /// - `weight_decay`: L2 regularization (típico: 0.0)
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
        Self::new(params, lr, 0.9, 0.999, 1e-8, 0.0)
    }
}

impl Optimizer for RAdam {
    fn step(&mut self) -> Result<()> {
        self.t += 1;
        let t = self.t as f32;
        
        // Bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);
        
        // Compute rho_inf (maximum length of approximated SMA)
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        
        // Compute rho_t (length of approximated SMA at time t)
        let rho_t = rho_inf - 2.0 * t * self.beta2.powf(t) / bias_correction2;
        
        // Determine if variance is tractable
        let use_adaptive = rho_t > 5.0;
        
        for (i, param_ref) in self.params.iter().enumerate() {
            let mut param = param_ref.borrow_mut();
            
            if !param.requires_grad {
                continue;
            }
            
            let mut grad = get_gradient(&param)?;
            
            // Weight decay
            apply_weight_decay(&mut grad, &param.data.data, self.weight_decay);
            
            // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
            for (m_val, &g) in self.m[i].data.iter_mut().zip(grad.iter()) {
                *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * g;
            }
            
            // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
            for (v_val, &g) in self.v[i].data.iter_mut().zip(grad.iter()) {
                *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * g * g;
            }
            
            if use_adaptive {
                // Variance is tractable, use adaptive update with rectification
                let r = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf / 
                        ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)).sqrt();
                
                for (p, (&m_val, &v_val)) in param.data.data.iter_mut()
                    .zip(self.m[i].data.iter().zip(self.v[i].data.iter()))
                {
                    // Bias-corrected moments
                    let m_hat = m_val / bias_correction1;
                    let v_hat = v_val / bias_correction2;
                    
                    // Rectified update
                    *p -= self.lr * r * m_hat / (v_hat.sqrt() + self.eps);
                }
            } else {
                // Variance is not tractable, use un-adapted momentum (like SGD)
                for (p, &m_val) in param.data.data.iter_mut()
                    .zip(self.m[i].data.iter())
                {
                    let m_hat = m_val / bias_correction1;
                    *p -= self.lr * m_hat;
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
