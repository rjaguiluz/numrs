//! Rprop (Resilient Backpropagation) Optimizer
//! 
//! Rprop es un optimizer que solo usa el signo del gradiente, no su magnitud.
//! Cada parámetro tiene su propio learning rate adaptativo que aumenta o disminuye
//! según si el signo del gradiente es consistente o cambia.
//! 
//! **Características principales:**
//! - Solo usa signo del gradiente (sign-based)
//! - Learning rate individual por parámetro
//! - Aumenta LR si signo es consistente (convergencia rápida)
//! - Disminuye LR si signo cambia (paso por mínimo)
//! - No requiere tuning cuidadoso del learning rate global
//! 
//! **Cuándo usar Rprop:**
//! - ✅ Batch training (full dataset)
//! - ✅ Cuando gradientes tienen diferentes escalas
//! - ✅ Problemas con plateaus
//! - ✅ Cuando no quieres tuning de LR
//! - ❌ Mini-batch training (inestable con ruido)
//! - ❌ Online learning
//! 
//! **Hiperparámetros típicos:**
//! - `lr_init`: 0.01 (learning rate inicial por parámetro)
//! - `eta_plus`: 1.2 (factor de incremento)
//! - `eta_minus`: 0.5 (factor de decremento)
//! - `lr_min`: 1e-6
//! - `lr_max`: 50.0

use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::{Tensor, Optimizer};
use crate::array::Array;
use anyhow::Result;

/// Rprop (Resilient Backpropagation) Optimizer
pub struct Rprop {
    /// Parámetros a optimizar
    params: Vec<Rc<RefCell<Tensor>>>,
    /// Learning rates individuales por parámetro
    step_sizes: Vec<Array>,
    /// Gradientes previos (para detectar cambio de signo)
    prev_grads: Vec<Array>,
    /// Factor de incremento (típico: 1.2)
    eta_plus: f32,
    /// Factor de decremento (típico: 0.5)
    eta_minus: f32,
    /// Learning rate mínimo
    lr_min: f32,
    /// Learning rate máximo
    lr_max: f32,
}

impl Rprop {
    /// Crea un nuevo optimizer Rprop
    /// 
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr_init`: Learning rate inicial para todos los parámetros (típico: 0.01)
    /// - `eta_plus`: Factor de incremento cuando signo es consistente (típico: 1.2)
    /// - `eta_minus`: Factor de decremento cuando signo cambia (típico: 0.5)
    /// - `lr_min`: Learning rate mínimo (típico: 1e-6)
    /// - `lr_max`: Learning rate máximo (típico: 50.0)
    pub fn new(
        params: Vec<Rc<RefCell<Tensor>>>,
        lr_init: f32,
        eta_plus: f32,
        eta_minus: f32,
        lr_min: f32,
        lr_max: f32,
    ) -> Self {
        let step_sizes = params.iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![lr_init; tensor.data.data.len()])
            })
            .collect();
        
        let prev_grads = params.iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();
        
        Self {
            params,
            step_sizes,
            prev_grads,
            eta_plus,
            eta_minus,
            lr_min,
            lr_max,
        }
    }
    
    /// Constructor con parámetros por defecto
    pub fn default(params: Vec<Rc<RefCell<Tensor>>>) -> Self {
        Self::new(params, 0.01, 1.2, 0.5, 1e-6, 50.0)
    }
}

impl Optimizer for Rprop {
    fn step(&mut self) -> Result<()> {
        for (i, param) in self.params.iter().enumerate() {
            let mut param_borrowed = param.borrow_mut();
            
            if let Some(grad_rc) = &param_borrowed.grad {
                // Extract gradient data first (drop immutable borrow)
                let grad_data: Vec<f32> = {
                    let grad = grad_rc.borrow();
                    grad.data.clone()
                };
                
                // Use 4-way zip for efficient iteration without bounds checks
                for (_j, (((param_val, &grad_curr), step_size), prev_grad)) in param_borrowed.data.data.iter_mut()
                    .zip(grad_data.iter())
                    .zip(self.step_sizes[i].data.iter_mut())
                    .zip(self.prev_grads[i].data.iter_mut())
                    .enumerate() 
                {
                    // Compute sign product: grad_curr * grad_prev
                    let sign_product = grad_curr * *prev_grad;
                    
                    if sign_product > 0.0 {
                        // Same sign: increase learning rate
                        *step_size = (*step_size * self.eta_plus).min(self.lr_max);
                        
                        // Update parameter
                        let update = *step_size * grad_curr.signum();
                        *param_val -= update;
                        
                        // Store current gradient for next iteration
                        *prev_grad = grad_curr;
                        
                    } else if sign_product < 0.0 {
                        // Different sign: decrease learning rate and don't update
                        *step_size = (*step_size * self.eta_minus).max(self.lr_min);
                        
                        // Set gradient to 0 to prevent double update
                        *prev_grad = 0.0;
                        
                    } else {
                        // First step or zero gradient: just update
                        let update = *step_size * grad_curr.signum();
                        *param_val -= update;
                        
                        // Store current gradient
                        *prev_grad = grad_curr;
                    }
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
        // Return average learning rate
        let total: f32 = self.step_sizes.iter()
            .flat_map(|arr| arr.data.iter())
            .sum();
        let count = self.step_sizes.iter()
            .map(|arr| arr.data.len())
            .sum::<usize>();
        total / (count as f32)
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        // Reset all learning rates to the specified value
        for step_size in &mut self.step_sizes {
            for val in &mut step_size.data {
                *val = lr;
            }
        }
    }
    
    fn num_params(&self) -> usize {
        self.params.len()
    }
}
