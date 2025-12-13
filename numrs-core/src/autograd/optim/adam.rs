//! Adam (Adaptive Moment Estimation) optimizer

use super::base::{apply_weight_decay, get_gradient, Optimizer};
use crate::array::Array;
use crate::autograd::Tensor;
use anyhow::Result;
use std::cell::RefCell;
use std::rc::Rc;

/// Adam optimizer
///
/// Update rule:
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
/// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
/// m_hat = m_t / (1 - beta1^t)
/// v_hat = v_t / (1 - beta2^t)
/// param = param - lr * m_hat / (sqrt(v_hat) + eps)
/// ```
///
/// # Ejemplo
/// ```ignore
/// let optimizer = Adam::new(model.parameters_mut(), 0.001, 0.9, 0.999, 1e-8, 0.0);
/// ```
pub struct Adam {
    /// Parámetros a optimizar
    params: Vec<Rc<RefCell<Tensor>>>,

    /// Learning rate
    lr: f32,

    /// Exponential decay rate para first moment (típico: 0.9)
    beta1: f32,

    /// Exponential decay rate para second moment (típico: 0.999)
    beta2: f32,

    /// Small constant para estabilidad numérica (típico: 1e-8)
    eps: f32,

    /// Weight decay (L2 regularization)
    weight_decay: f32,

    /// First moment estimates (moving average of gradients)
    m: Vec<Array>,

    /// Second moment estimates (moving average of squared gradients)
    v: Vec<Array>,

    /// Timestep counter
    t: usize,
}

impl Adam {
    /// Crea un nuevo optimizer Adam
    ///
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate (típico: 0.001)
    /// - `beta1`: Decay rate para first moment (típico: 0.9)
    /// - `beta2`: Decay rate para second moment (típico: 0.999)
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
        let m = params
            .iter()
            .map(|p| {
                let tensor = p.borrow();
                Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()])
            })
            .collect();

        let v = params
            .iter()
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

    pub fn with_lr(params: Vec<Rc<RefCell<Tensor>>>, lr: f32) -> Self {
        Self::default(params, lr)
    }

    /// Añade un parámetro al optimizer
    pub fn add_param(&mut self, param: Rc<RefCell<Tensor>>) {
        let tensor = param.borrow();
        let m_zeros = Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()]);
        let v_zeros = Array::new(tensor.data.shape.clone(), vec![0.0; tensor.data.data.len()]);
        drop(tensor); // Release borrow

        self.params.push(param);
        self.m.push(m_zeros);
        self.v.push(v_zeros);
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.t += 1;

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

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

            // Bias-corrected estimates and update
            for (p, (&m_val, &v_val)) in param
                .data
                .data
                .iter_mut()
                .zip(self.m[i].data.iter().zip(self.v[i].data.iter()))
            {
                let m_hat = m_val / bias_correction1;
                let v_hat = v_val / bias_correction2;
                *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
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
