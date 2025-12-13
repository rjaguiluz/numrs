//! RMSprop optimizer

use super::base::{apply_weight_decay, get_gradient, Optimizer};
use crate::array::Array;
use crate::autograd::Tensor;
use anyhow::Result;
use std::cell::RefCell;
use std::rc::Rc;

/// RMSprop (Root Mean Square Propagation) optimizer
///
/// Update rule:
/// ```text
/// v_t = alpha * v_{t-1} + (1 - alpha) * grad^2
/// param = param - lr * grad / (sqrt(v_t) + eps)
/// ```
///
/// # Ejemplo
/// ```ignore
/// let optimizer = RMSprop::new(model.parameters_mut(), 0.01, 0.99, 1e-8, 0.0, 0.0);
/// ```
pub struct RMSprop {
    params: Vec<Rc<RefCell<Tensor>>>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,

    /// Moving average of squared gradients
    v: Vec<Array>,

    /// Momentum buffer (opcional)
    buffer: Vec<Array>,
}

impl RMSprop {
    /// Crea un nuevo optimizer RMSprop
    ///
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate (típico: 0.01)
    /// - `alpha`: Smoothing constant (típico: 0.99)
    /// - `eps`: Epsilon para estabilidad numérica (típico: 1e-8)
    /// - `weight_decay`: L2 regularization (típico: 0.0)
    /// - `momentum`: Momentum factor (típico: 0.0)
    pub fn new(
        params: Vec<Rc<RefCell<Tensor>>>,
        lr: f32,
        alpha: f32,
        eps: f32,
        weight_decay: f32,
        momentum: f32,
    ) -> Self {
        let v = params
            .iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();

        let buffer = params
            .iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();

        Self {
            params,
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            v,
            buffer,
        }
    }

    /// Constructor con parámetros por defecto
    pub fn default(params: Vec<Rc<RefCell<Tensor>>>, lr: f32) -> Self {
        Self::new(params, lr, 0.99, 1e-8, 0.0, 0.0)
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) -> Result<()> {
        for (i, param_ref) in self.params.iter().enumerate() {
            let mut param = param_ref.borrow_mut();

            if !param.requires_grad {
                continue;
            }

            let mut grad = get_gradient(&param)?;

            // Weight decay
            apply_weight_decay(&mut grad, &param.data.data, self.weight_decay);

            // Update moving average: v = alpha * v + (1 - alpha) * grad^2
            for (v_val, &g) in self.v[i].data.iter_mut().zip(grad.iter()) {
                *v_val = self.alpha * *v_val + (1.0 - self.alpha) * g * g;
            }

            if self.momentum > 0.0 {
                // Con momentum
                for (buf, (&g, &v_val)) in self.buffer[i]
                    .data
                    .iter_mut()
                    .zip(grad.iter().zip(self.v[i].data.iter()))
                {
                    *buf = self.momentum * *buf + g / (v_val.sqrt() + self.eps);
                }

                for (p, &buf) in param.data.data.iter_mut().zip(self.buffer[i].data.iter()) {
                    *p -= self.lr * buf;
                }
            } else {
                // Sin momentum
                for (p, (&g, &v_val)) in param
                    .data
                    .data
                    .iter_mut()
                    .zip(grad.iter().zip(self.v[i].data.iter()))
                {
                    *p -= self.lr * g / (v_val.sqrt() + self.eps);
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
