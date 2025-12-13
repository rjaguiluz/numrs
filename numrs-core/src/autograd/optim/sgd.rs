//! SGD (Stochastic Gradient Descent) optimizer

use super::base::{apply_weight_decay, get_gradient, Optimizer};
use crate::array::Array;
use crate::autograd::Tensor;
use anyhow::Result;
use std::cell::RefCell;
use std::rc::Rc;

/// SGD optimizer con momentum opcional
///
/// Update rule:
/// ```text
/// v_t = momentum * v_{t-1} + grad
/// param = param - lr * v_t
/// ```
///
/// # Ejemplo
/// ```ignore
/// let optimizer = SGD::new(model.parameters_mut(), 0.01, 0.9, 0.0);
/// ```
pub struct SGD {
    /// Parámetros a optimizar
    params: Vec<Rc<RefCell<Tensor>>>,

    /// Learning rate
    lr: f32,

    /// Momentum factor (0.0 = sin momentum, típico: 0.9)
    momentum: f32,

    /// Velocities para momentum (uno por parámetro)
    velocities: Vec<Array>,

    /// Weight decay (L2 regularization)
    weight_decay: f32,
}

impl SGD {
    /// Crea un nuevo optimizer SGD
    ///
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate
    /// - `momentum`: Factor de momentum (default: 0.0, típico: 0.9)
    /// - `weight_decay`: L2 regularization (default: 0.0, típico: 1e-4)
    pub fn new(
        params: Vec<Rc<RefCell<Tensor>>>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> Self {
        // Inicializar velocities a cero
        let velocities = params
            .iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();

        Self {
            params,
            lr,
            momentum,
            velocities,
            weight_decay,
        }
    }

    pub fn simple(params: Vec<Rc<RefCell<Tensor>>>, lr: f32) -> Self {
        Self::new(params, lr, 0.0, 0.0)
    }

    /// Añade un parámetro al optimizer
    pub fn add_param(&mut self, param: Rc<RefCell<Tensor>>) {
        let tensor = param.borrow();
        let velocity = Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()]);
        drop(tensor); // Release borrow

        self.params.push(param);
        self.velocities.push(velocity);
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        for (i, param_ref) in self.params.iter().enumerate() {
            let mut param = param_ref.borrow_mut();

            if !param.requires_grad {
                continue;
            }

            let mut grad = get_gradient(&param)?;

            // Weight decay (L2 regularization)
            apply_weight_decay(&mut grad, &param.data.data, self.weight_decay);

            // Momentum: v = momentum * v + grad
            if self.momentum != 0.0 {
                for (v, &g) in self.velocities[i].data.iter_mut().zip(grad.iter()) {
                    *v = self.momentum * *v + g;
                }

                // Update: param -= lr * v
                for (p, &v) in param
                    .data
                    .data
                    .iter_mut()
                    .zip(self.velocities[i].data.iter())
                {
                    *p -= self.lr * v;
                }
            } else {
                // Sin momentum: param -= lr * grad
                for (p, &g) in param.data.data.iter_mut().zip(grad.iter()) {
                    *p -= self.lr * g;
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
