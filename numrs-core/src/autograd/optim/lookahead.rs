//! Lookahead Optimizer
//! 
//! Lookahead es un meta-optimizer que envuelve otro optimizer y mantiene
//! dos conjuntos de pesos: "fast weights" (actualizados por el inner optimizer)
//! y "slow weights" (actualizados ocasionalmente hacia los fast weights).
//! 
//! **Características principales:**
//! - Reduce variance y mejora convergencia
//! - Funciona como wrapper de cualquier optimizer
//! - Parámetros k (steps) y α (interpolation)
//! - Mejora robustez a hyperparameter choice
//! 
//! **Cuándo usar Lookahead:**
//! - ✅ Cuando quieres hacer training más estable
//! - ✅ Para reducir overfitting
//! - ✅ Con cualquier optimizer base (SGD, Adam, etc.)
//! - ✅ Cuando hyperparameter tuning es costoso
//! 
//! **Hiperparámetros típicos:**
//! - `k`: 5-10 (número de inner optimizer steps)
//! - `alpha`: 0.5 (interpolation factor)
//! 
//! **Ejemplo:**
//! ```rust,ignore
//! let sgd = SGD::new(params.clone(), 0.1, 0.9, 0.0);
//! let mut lookahead = Lookahead::new(sgd, params, 5, 0.5);
//! ```

use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::{Tensor, Optimizer};
use crate::array::Array;
use anyhow::Result;

/// Lookahead Optimizer (meta-optimizer wrapper)
pub struct Lookahead<O: Optimizer> {
    /// Inner optimizer (SGD, Adam, etc.)
    inner_optimizer: O,
    /// Slow weights (parámetros principales)
    slow_weights: Vec<Rc<RefCell<Tensor>>>,
    /// Fast weights backup (copias de los parámetros)
    fast_weights_backup: Vec<Array>,
    /// Número de steps antes de actualizar slow weights
    k: usize,
    /// Interpolation factor (típico: 0.5)
    alpha: f32,
    /// Contador de steps desde último sync
    step_counter: usize,
}

impl<O: Optimizer> Lookahead<O> {
    /// Crea un nuevo Lookahead optimizer
    /// 
    /// # Argumentos
    /// - `inner_optimizer`: El optimizer base (SGD, Adam, etc.)
    /// - `params`: Referencias a los tensors a optimizar
    /// - `k`: Número de inner optimizer steps entre actualizaciones (típico: 5)
    /// - `alpha`: Factor de interpolación (típico: 0.5)
    pub fn new(
        inner_optimizer: O,
        params: Vec<Rc<RefCell<Tensor>>>,
        k: usize,
        alpha: f32,
    ) -> Self {
        // Inicializar backup de fast weights con los valores actuales
        let fast_weights_backup = params.iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(
                    tensor.data.shape.clone(),
                    tensor.data.data.clone(),
                )
            })
            .collect();
        
        Self {
            inner_optimizer,
            slow_weights: params,
            fast_weights_backup,
            k,
            alpha,
            step_counter: 0,
        }
    }
}

impl<O: Optimizer> Optimizer for Lookahead<O> {
    fn step(&mut self) -> Result<()> {
        // Step 1: Actualizar con inner optimizer (fast weights)
        self.inner_optimizer.step()?;
        
        self.step_counter += 1;
        
        // Step 2: Cada k steps, actualizar slow weights
        if self.step_counter >= self.k {
            self.step_counter = 0;
            
            // Actualizar slow weights: θ_slow = θ_slow + α * (θ_fast - θ_slow)
            // Equivalente a: θ_slow = (1 - α) * θ_slow + α * θ_fast
            for (i, param) in self.slow_weights.iter().enumerate() {
                let mut param_borrowed = param.borrow_mut();
                
                for j in 0..param_borrowed.data.data.len() {
                    let slow_val = self.fast_weights_backup[i].data[j];
                    let fast_val = param_borrowed.data.data[j];
                    
                    // Interpolate: slow = slow + alpha * (fast - slow)
                    let new_slow = slow_val + self.alpha * (fast_val - slow_val);
                    
                    // Update slow weights backup
                    self.fast_weights_backup[i].data[j] = new_slow;
                    
                    // Reset fast weights to slow weights
                    param_borrowed.data.data[j] = new_slow;
                }
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&self) {
        self.inner_optimizer.zero_grad();
    }
    
    fn learning_rate(&self) -> f32 {
        self.inner_optimizer.learning_rate()
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.inner_optimizer.set_learning_rate(lr);
    }
    
    fn num_params(&self) -> usize {
        self.slow_weights.len()
    }
}
