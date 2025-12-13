//! AdamW (Adam with decoupled Weight decay) optimizer

use super::base::{Optimizer, get_gradient};
use crate::autograd::Tensor;
use crate::array::Array;
use anyhow::Result;
use std::rc::Rc;
use std::cell::RefCell;

/// AdamW optimizer - Adam con weight decay desacoplado
/// 
/// A diferencia de Adam regular, AdamW aplica weight decay directamente
/// a los parámetros, no al gradiente. Esto mejora la regularización.
/// 
/// Update rule:
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
/// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
/// ```
/// 
/// # Ejemplo
/// ```ignore
/// let optimizer = AdamW::new(model.parameters_mut(), 0.001, 0.9, 0.999, 1e-8, 0.01);
/// ```
pub struct AdamW {
    params: Vec<Rc<RefCell<Tensor>>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Array>,
    v: Vec<Array>,
    t: usize,
}

impl AdamW {
    /// Crea un nuevo optimizer AdamW
    /// 
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate (típico: 0.001)
    /// - `beta1`: Decay rate para first moment (típico: 0.9)
    /// - `beta2`: Decay rate para second moment (típico: 0.999)
    /// - `eps`: Epsilon para estabilidad numérica (típico: 1e-8)
    /// - `weight_decay`: Weight decay (típico: 0.01 para AdamW)
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
        Self::new(params, lr, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) -> Result<()> {
        self.t += 1;
        
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        for (i, param_ref) in self.params.iter().enumerate() {
            let mut param = param_ref.borrow_mut();
            
            if !param.requires_grad {
                continue;
            }
            
            let grad = get_gradient(&param)?;
            
            // Update moments (sin weight decay en grad)
            for (m_val, &g) in self.m[i].data.iter_mut().zip(grad.iter()) {
                *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * g;
            }
            
            for (v_val, &g) in self.v[i].data.iter_mut().zip(grad.iter()) {
                *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * g * g;
            }
            
            // Update con weight decay desacoplado
            for (p, (&m_val, &v_val)) in param.data.data.iter_mut()
                .zip(self.m[i].data.iter().zip(self.v[i].data.iter()))
            {
                let m_hat = m_val / bias_correction1;
                let v_hat = v_val / bias_correction2;
                
                // AdamW: weight decay aplicado directamente al parámetro
                *p -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * *p);
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
