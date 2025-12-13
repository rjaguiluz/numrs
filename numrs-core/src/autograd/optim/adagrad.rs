//! AdaGrad optimizer

use super::base::{Optimizer, apply_weight_decay, get_gradient};
use crate::autograd::Tensor;
use crate::array::Array;
use anyhow::Result;
use std::rc::Rc;
use std::cell::RefCell;

/// AdaGrad (Adaptive Gradient) optimizer
/// 
/// Adapta el learning rate para cada parámetro basado en el historial de gradientes.
/// Ideal para sparse data.
/// 
/// Update rule:
/// ```text
/// G_t = G_{t-1} + grad^2
/// param = param - lr / (sqrt(G_t) + eps) * grad
/// ```
/// 
/// # Ejemplo
/// ```ignore
/// let optimizer = AdaGrad::new(model.parameters_mut(), 0.01, 1e-10, 0.0);
/// ```
pub struct AdaGrad {
    params: Vec<Rc<RefCell<Tensor>>>,
    lr: f32,
    eps: f32,
    weight_decay: f32,
    
    /// Accumulated squared gradients
    sum_squared: Vec<Array>,
}

impl AdaGrad {
    /// Crea un nuevo optimizer AdaGrad
    /// 
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate (típico: 0.01)
    /// - `eps`: Epsilon para estabilidad numérica (típico: 1e-10)
    /// - `weight_decay`: L2 regularization (típico: 0.0)
    pub fn new(
        params: Vec<Rc<RefCell<Tensor>>>,
        lr: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let sum_squared = params.iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();
        
        Self {
            params,
            lr,
            eps,
            weight_decay,
            sum_squared,
        }
    }
    
    /// Constructor con parámetros por defecto
    pub fn default(params: Vec<Rc<RefCell<Tensor>>>, lr: f32) -> Self {
        Self::new(params, lr, 1e-10, 0.0)
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self) -> Result<()> {
        for (i, param_ref) in self.params.iter().enumerate() {
            let mut param = param_ref.borrow_mut();
            
            if !param.requires_grad {
                continue;
            }
            
            let mut grad = get_gradient(&param)?;
            
            // Weight decay
            apply_weight_decay(&mut grad, &param.data.data, self.weight_decay);
            
            // Accumulate squared gradients: G = G + grad^2
            for (acc, &g) in self.sum_squared[i].data.iter_mut().zip(grad.iter()) {
                *acc += g * g;
            }
            
            // Update parameters: param = param - lr * grad / (sqrt(G) + eps)
            for (p, (&g, &acc)) in param.data.data.iter_mut()
                .zip(grad.iter().zip(self.sum_squared[i].data.iter()))
            {
                *p -= self.lr * g / (acc.sqrt() + self.eps);
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
