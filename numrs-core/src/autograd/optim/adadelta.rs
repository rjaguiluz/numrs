//! AdaDelta optimizer

use super::base::{Optimizer, apply_weight_decay, get_gradient};
use crate::autograd::Tensor;
use crate::array::Array;
use anyhow::Result;
use std::rc::Rc;
use std::cell::RefCell;

/// AdaDelta optimizer - extensión de AdaGrad que no requiere learning rate
/// 
/// Resuelve el problema de AdaGrad donde el learning rate decrece monótonamente
/// al acumular gradientes squared. Usa ventana móvil en lugar de suma infinita.
/// 
/// Características:
/// - No requiere learning rate manual (se adapta automáticamente)
/// - Usa exponential moving average en lugar de suma acumulativa
/// - Robusto a la elección de hyperparámetros
/// 
/// Update rule:
/// ```text
/// E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * grad^2
/// Delta_t = sqrt(E[Delta^2]_{t-1} + eps) / sqrt(E[g^2]_t + eps) * grad
/// E[Delta^2]_t = rho * E[Delta^2]_{t-1} + (1 - rho) * Delta_t^2
/// param = param - Delta_t
/// ```
/// 
/// # Ejemplo
/// ```ignore
/// let optimizer = AdaDelta::new(model.parameters_mut(), 0.95, 1e-6, 0.0);
/// ```
pub struct AdaDelta {
    params: Vec<Rc<RefCell<Tensor>>>,
    rho: f32,
    eps: f32,
    weight_decay: f32,
    
    /// Accumulated squared gradients (exponential moving average)
    e_grad_sq: Vec<Array>,
    
    /// Accumulated squared parameter updates (exponential moving average)
    e_delta_sq: Vec<Array>,
}

impl AdaDelta {
    /// Crea un nuevo optimizer AdaDelta
    /// 
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `rho`: Decay rate para moving averages (típico: 0.95)
    /// - `eps`: Epsilon para estabilidad numérica (típico: 1e-6)
    /// - `weight_decay`: L2 regularization (típico: 0.0)
    pub fn new(
        params: Vec<Rc<RefCell<Tensor>>>,
        rho: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let e_grad_sq = params.iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();
        
        let e_delta_sq = params.iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();
        
        Self {
            params,
            rho,
            eps,
            weight_decay,
            e_grad_sq,
            e_delta_sq,
        }
    }
    
    /// Constructor con parámetros por defecto
    pub fn default(params: Vec<Rc<RefCell<Tensor>>>) -> Self {
        Self::new(params, 0.95, 1e-6, 0.0)
    }
}

impl Optimizer for AdaDelta {
    fn step(&mut self) -> Result<()> {
        for (i, param_ref) in self.params.iter().enumerate() {
            let mut param = param_ref.borrow_mut();
            
            if !param.requires_grad {
                continue;
            }
            
            let mut grad = get_gradient(&param)?;
            
            // Weight decay
            apply_weight_decay(&mut grad, &param.data.data, self.weight_decay);
            
            // Update accumulated squared gradients: E[g^2] = rho * E[g^2] + (1 - rho) * g^2
            for (e_g, &g) in self.e_grad_sq[i].data.iter_mut().zip(grad.iter()) {
                *e_g = self.rho * *e_g + (1.0 - self.rho) * g * g;
            }
            
            // Compute parameter update and apply
            let mut deltas = Vec::with_capacity(grad.len());
            for ((&e_d, &e_g), &g) in self.e_delta_sq[i].data.iter()
                .zip(self.e_grad_sq[i].data.iter())
                .zip(grad.iter())
            {
                // RMS[Delta]_t / RMS[g]_t * g
                let delta = ((e_d + self.eps).sqrt() / (e_g + self.eps).sqrt()) * g;
                deltas.push(delta);
            }
            
            // Update accumulated squared deltas: E[Delta^2] = rho * E[Delta^2] + (1 - rho) * Delta^2
            for (e_d, &delta) in self.e_delta_sq[i].data.iter_mut().zip(deltas.iter()) {
                *e_d = self.rho * *e_d + (1.0 - self.rho) * delta * delta;
            }
            
            // Apply update
            for (p, &delta) in param.data.data.iter_mut().zip(deltas.iter()) {
                *p -= delta;
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
        // AdaDelta no tiene learning rate explícito, retorna 1.0 como placeholder
        1.0
    }
    
    fn set_learning_rate(&mut self, _lr: f32) {
        // AdaDelta no usa learning rate, ignoramos el set
    }
    
    fn num_params(&self) -> usize {
        self.params.len()
    }
}
