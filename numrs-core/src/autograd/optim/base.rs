//! Optimizer trait - Base para todos los optimizers

use crate::autograd::Tensor;
use anyhow::Result;


/// Trait base para todos los optimizers
pub trait Optimizer {
    /// Actualiza los parámetros usando sus gradientes
    fn step(&mut self) -> Result<()>;
    
    /// Limpia los gradientes de todos los parámetros
    fn zero_grad(&self);
    
    /// Obtiene el learning rate actual
    fn learning_rate(&self) -> f32;
    
    /// Actualiza el learning rate
    fn set_learning_rate(&mut self, lr: f32);
    
    /// Obtiene el número de parámetros
    fn num_params(&self) -> usize;
}

/// Helper para aplicar weight decay
pub(crate) fn apply_weight_decay(
    grad: &mut [f32],
    params: &[f32],
    weight_decay: f32,
) {
    if weight_decay != 0.0 {
        for (g, &p) in grad.iter_mut().zip(params.iter()) {
            *g += weight_decay * p;
        }
    }
}

/// Helper para obtener gradientes de un parámetro
pub(crate) fn get_gradient(param: &Tensor) -> Result<Vec<f32>> {
    param.gradient()
        .ok_or_else(|| anyhow::anyhow!("Parameter has no gradient"))
        .map(|grad| grad.data.clone())
}
