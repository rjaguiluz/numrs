//! L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) Optimizer
//! 
//! L-BFGS es un optimizer quasi-Newton de segunda orden que aproxima la matriz Hessiana
//! usando un historial limitado de gradientes y actualizaciones. Es muy eficiente para
//! optimización batch (no estocástica) con funciones suaves.
//! 
//! **Características principales:**
//! - Usa información de segunda orden (curvatura)
//! - Memoria limitada: solo guarda últimos m pares (s, y)
//! - Convergencia super-lineal en problemas bien condicionados
//! - Requiere line search para estabilidad
//! 
//! **Cuándo usar L-BFGS:**
//! - ✅ Optimización batch (full dataset per step)
//! - ✅ Problemas pequeños/medianos con gradientes determinísticos
//! - ✅ Fine-tuning con dataset completo
//! - ✅ Funciones suaves y bien condicionadas
//! - ❌ Mini-batch SGD (inestable con gradientes ruidosos)
//! - ❌ Very large scale problems (memoria)
//! 
//! **Hiperparámetros típicos:**
//! - `lr`: 1.0 (usa line search internamente)
//! - `history_size`: 10-20 (memoria de pares s,y)
//! - `max_iter`: 20 (iteraciones de line search)

use std::rc::Rc;
use std::cell::RefCell;
use crate::autograd::{Tensor, Optimizer};

use anyhow::Result;

/// L-BFGS Optimizer (quasi-Newton de segunda orden)
pub struct LBFGS {
    /// Parámetros a optimizar
    params: Vec<Rc<RefCell<Tensor>>>,
    /// Learning rate (típicamente 1.0 con line search)
    lr: f32,
    /// Tamaño del historial (número de pares s,y a recordar)
    history_size: usize,
    /// Máximo número de iteraciones de line search
    _max_iter: usize,
    /// Historial de diferencias de parámetros: s_k = x_{k+1} - x_k
    s_history: Vec<Vec<f32>>,
    /// Historial de diferencias de gradientes: y_k = ∇f_{k+1} - ∇f_k
    y_history: Vec<Vec<f32>>,
    /// Gradiente previo (para calcular y_k)
    prev_grad: Option<Vec<f32>>,
    /// Parámetros previos (para calcular s_k)
    prev_params: Option<Vec<f32>>,
}

impl LBFGS {
    /// Crea un nuevo optimizer L-BFGS
    /// 
    /// # Argumentos
    /// - `params`: Referencias a los tensors a optimizar
    /// - `lr`: Learning rate (típico: 1.0)
    /// - `history_size`: Número de pares (s,y) a recordar (típico: 10)
    /// - `max_iter`: Máximo de iteraciones para line search (típico: 20)
    pub fn new(
        params: Vec<Rc<RefCell<Tensor>>>,
        lr: f32,
        history_size: usize,
        max_iter: usize,
    ) -> Self {
        Self {
            params,
            lr,
            history_size,
            _max_iter: max_iter,
            s_history: Vec::new(),
            y_history: Vec::new(),
            prev_grad: None,
            prev_params: None,
        }
    }
    
    /// Constructor con parámetros por defecto
    pub fn default(params: Vec<Rc<RefCell<Tensor>>>) -> Self {
        Self::new(params, 1.0, 10, 20)
    }
    
    /// Flatten all parameters into a single vector
    fn flatten_params(&self) -> Vec<f32> {
        let mut flat = Vec::new();
        for param in &self.params {
            let p = param.borrow();
            flat.extend_from_slice(&p.data.data);
        }
        flat
    }
    
    /// Flatten all gradients into a single vector
    fn flatten_grads(&self) -> Vec<f32> {
        let mut flat = Vec::new();
        for param in &self.params {
            let p = param.borrow();
            if let Some(grad_rc) = &p.grad {
                let grad = grad_rc.borrow();
                flat.extend_from_slice(&grad.data);
            } else {
                flat.extend(vec![0.0; p.data.data.len()]);
            }
        }
        flat
    }
    
    /// Compute dot product
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    /// Two-loop recursion para computar dirección de búsqueda
    fn two_loop_recursion(&self, grad: &[f32]) -> Vec<f32> {
        if self.s_history.is_empty() {
            // Si no hay historial, usar gradiente negativo (steepest descent)
            return grad.iter().map(|&g| -g).collect();
        }
        
        let m = self.s_history.len();
        let mut q = grad.to_vec();
        let mut alpha = vec![0.0; m];
        let mut rho = vec![0.0; m];
        
        // Backward pass
        for i in (0..m).rev() {
            rho[i] = 1.0 / Self::dot(&self.y_history[i], &self.s_history[i]).max(1e-10);
            alpha[i] = rho[i] * Self::dot(&self.s_history[i], &q);
            
            for (q_val, y_val) in q.iter_mut().zip(self.y_history[i].iter()) {
                *q_val -= alpha[i] * y_val;
            }
        }
        
        // Compute initial Hessian approximation: H_0 = (s^T y / y^T y) * I
        let s_last = &self.s_history[m - 1];
        let y_last = &self.y_history[m - 1];
        let gamma = Self::dot(s_last, y_last) / Self::dot(y_last, y_last).max(1e-10);
        
        // Scale: r = γ * q
        let mut r: Vec<f32> = q.iter().map(|&x| gamma * x).collect();
        
        // Forward pass
        for i in 0..m {
            let beta = rho[i] * Self::dot(&self.y_history[i], &r);
            for (r_val, s_val) in r.iter_mut().zip(self.s_history[i].iter()) {
                *r_val += s_val * (alpha[i] - beta);
            }
        }
        
        // Return search direction: d = -H * grad
        r.iter().map(|&x| -x).collect()
    }
}

impl Optimizer for LBFGS {
    fn step(&mut self) -> Result<()> {
        // Get current parameters and gradients
        let current_params = self.flatten_params();
        let current_grad = self.flatten_grads();
        
        // Update history if we have previous step
        if let (Some(prev_params), Some(prev_grad)) = (&self.prev_params, &self.prev_grad) {
            // s_k = x_{k+1} - x_k
            let s: Vec<f32> = current_params.iter()
                .zip(prev_params.iter())
                .map(|(curr, prev)| curr - prev)
                .collect();
            
            // y_k = ∇f_{k+1} - ∇f_k
            let y: Vec<f32> = current_grad.iter()
                .zip(prev_grad.iter())
                .map(|(curr, prev)| curr - prev)
                .collect();
            
            // Check curvature condition: s^T y > 0
            let sy = Self::dot(&s, &y);
            if sy > 1e-10 {
                self.s_history.push(s);
                self.y_history.push(y);
                
                // Keep only last history_size pairs
                if self.s_history.len() > self.history_size {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                }
            }
        }
        
        // Compute search direction using two-loop recursion
        let direction = self.two_loop_recursion(&current_grad);
        
        // Simple line search: just use learning rate
        // (En producción, usar Wolfe conditions o backtracking)
        let mut offset = 0;
        for param in &self.params {
            let mut p = param.borrow_mut();
            let len = p.data.data.len();
            
            for (val, &dir) in p.data.data.iter_mut().zip(direction[offset..offset+len].iter()) {
                 *val += self.lr * dir;
            }
            
            offset += len;
        }
        
        // Store current state for next iteration
        self.prev_params = Some(current_params);
        self.prev_grad = Some(current_grad);
        
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
