//! Neural Network Modules
//!
//! PyTorch-like API para construir modelos:
//! - Module trait para forward pass
//! - Linear, Sequential layers
//! - Parameter management

use crate::{Array, Tensor};
use anyhow::{anyhow, Result};
use std::cell::RefCell;
use std::rc::Rc;

/// Trait para módulos de red neuronal
///
/// Similar a torch.nn.Module, define la interfaz para:
/// - forward() - propagación hacia adelante
/// - parameters() - obtener todos los parámetros entrenables
/// Trait para módulos de red neuronal
///
/// Similar a torch.nn.Module, define la interfaz para:
/// - forward() - propagación hacia adelante
/// - parameters() - obtener todos los parámetros entrenables
pub trait Module {
    /// Forward pass del módulo
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Devuelve referencias a todos los parámetros entrenables
    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>>;

    /// Pone el módulo en modo entrenamiento
    fn train(&mut self) {
        // Default: no hace nada (útil para Dropout, BatchNorm)
    }

    /// Pone el módulo en modo evaluación
    fn eval(&mut self) {
        // Default: no hace nada
    }

    /// Clone the module (helper for Box<dyn Module>)
    fn box_clone(&self) -> Box<dyn Module>;
}

impl Clone for Box<dyn Module> {
    fn clone(&self) -> Box<dyn Module> {
        self.box_clone()
    }
}

/// Capa Linear: y = xW^T + b
///
/// Ejemplo:
/// ```ignore
/// let linear = Linear::new(10, 5)?;  // 10 inputs → 5 outputs
/// let output = linear.forward(&input)?;
/// ```
#[derive(Clone)]
pub struct Linear {
    weight: Rc<RefCell<Tensor>>, // [out_features, in_features]
    bias: Rc<RefCell<Tensor>>,   // [out_features]
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Crea una capa Linear con inicialización Xavier/Glorot
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        // Xavier initialization: U(-sqrt(k), sqrt(k)) donde k = 1/in_features
        let k = (1.0 / in_features as f32).sqrt();

        // Inicializar pesos aleatoriamente (aquí usamos valores simples)
        // En producción, usar distribución uniforme U(-k, k)
        let mut weight_data = Vec::with_capacity(in_features * out_features);
        for i in 0..(in_features * out_features) {
            // Pseudo-random usando patrón simple centered at 0
            let val = (i as f32 * 0.123).sin() * k;
            weight_data.push(val);
        }

        let weight = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![out_features, in_features], weight_data),
            true, // requires_grad
        )));

        let bias = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![out_features], vec![0.0; out_features]),
            true,
        )));

        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn weight(&self) -> Rc<RefCell<Tensor>> {
        self.weight.clone()
    }

    pub fn bias(&self) -> Rc<RefCell<Tensor>> {
        self.bias.clone()
    }

    /// Crea Linear con pesos específicos (útil para testing)
    pub fn with_weights(
        in_features: usize,
        out_features: usize,
        weight_data: Vec<f32>,
        bias_data: Vec<f32>,
    ) -> Result<Self> {
        let weight = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![out_features, in_features], weight_data),
            true,
        )));

        let bias = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![out_features], bias_data),
            true,
        )));

        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Retorna referencias mutables a los parámetros (para optimizers)
    pub fn parameters_mut(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    /// Backward pass manual (calcula gradientes de weight y bias)
    pub fn backward(&self, input: &Tensor, grad_output: &Array) -> Result<()> {
        let batch_size = input.data.shape[0];

        // Calcular grad_weight += grad_output^T @ input
        {
            let weight_shape;
            let weight_data_len;
            {
                let weight = self.weight.borrow();
                weight_shape = weight.data.shape.clone();
                weight_data_len = weight.data.data.len();
            }

            let mut weight = self.weight.borrow_mut();

            if weight.grad.is_none() {
                weight.grad = Some(Rc::new(RefCell::new(Array::new(
                    weight_shape.clone(),
                    vec![0.0; weight_data_len],
                ))));
            }

            let grad_weight = weight
                .grad
                .as_ref()
                .ok_or_else(|| anyhow!("Failed to initialize weight gradient"))?;
            let mut grad_w = grad_weight.borrow_mut();

            for i in 0..self.out_features {
                for j in 0..self.in_features {
                    let mut sum = 0.0;
                    for b in 0..batch_size {
                        let grad_out_val = grad_output.data[b * self.out_features + i];
                        let input_val = input.data.data[b * self.in_features + j];
                        sum += grad_out_val * input_val;
                    }
                    grad_w.data[i * self.in_features + j] += sum;
                }
            }
        }

        // Calcular grad_bias = sum(grad_output, axis=0)
        {
            let bias_shape;
            let bias_data_len;
            {
                let bias = self.bias.borrow();
                bias_shape = bias.data.shape.clone();
                bias_data_len = bias.data.data.len();
            }

            let mut bias = self.bias.borrow_mut();

            if bias.grad.is_none() {
                bias.grad = Some(Rc::new(RefCell::new(Array::new(
                    bias_shape.clone(),
                    vec![0.0; bias_data_len],
                ))));
            }

            let grad_bias = bias
                .grad
                .as_ref()
                .ok_or_else(|| anyhow!("Failed to initialize bias gradient"))?;
            let mut grad_b = grad_bias.borrow_mut();

            for i in 0..self.out_features {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    sum += grad_output.data[b * self.out_features + i];
                }
                grad_b.data[i] += sum;
            }
        }

        Ok(())
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // y = xW^T + b
        // input: [batch_size, in_features]
        // weight: [out_features, in_features]
        // weight^T: [in_features, out_features]
        // output: [batch_size, out_features]

        let w = self.weight.borrow();
        let b = self.bias.borrow();

        // Transponer weight: [out, in] -> [in, out]
        let w_t = w.transpose()?;

        // x @ W^T: [batch, in] @ [in, out] = [batch, out]
        let output = input.matmul(&w_t)?;

        // + bias: [batch, out] + [out] = [batch, out]
        output.add(&*b)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Sequential: ejecuta módulos en secuencia
///
/// Ejemplo:
/// ```ignore
/// let model = Sequential::new(vec![
///     Box::new(Linear::new(784, 128)?),
///     Box::new(ReLU),
///     Box::new(Linear::new(128, 10)?),
/// ]);
/// ```
#[derive(Clone)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }

    pub fn add(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer);
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// ReLU activation: max(0, x)
#[derive(Clone)]
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![] // Sin parámetros
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[derive(Clone)]
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.sigmoid()
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Softmax activation: exp(x_i) / sum(exp(x_j))
///
/// Nota: Actualmente no implementado en autograd,
/// usar directamente en loss function
#[derive(Clone)]
pub struct Softmax;

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Forward via ops::stats::softmax (wraps Core op)
        // Default axis=1 (last dim usually)
        // Core's stats::softmax takes axis: Option<usize>
        let axis = input.data.shape.len() - 1;
        let result = crate::ops::stats::softmax::softmax(&input.data, Some(axis))?;

        if !crate::autograd::is_grad_enabled() || !input.requires_grad {
            return Ok(Tensor::new(result, false));
        }

        let backward_fn = Box::new(crate::autograd::backward::softmax_backward);
        let node = crate::autograd::ComputeNode::new(
            crate::autograd::OpKind::Softmax,
            vec![input.clone()],
            Some(backward_fn),
        );
        Ok(Tensor::from_operation(result, node, true))
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Dropout layer
#[derive(Clone)]
pub struct Dropout {
    p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        Dropout { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.dropout(self.p, self.training)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Flatten layer
#[derive(Clone)]
pub struct Flatten {
    start_dim: usize,
    end_dim: usize,
}

impl Flatten {
    pub fn new(start_dim: usize, end_dim: usize) -> Self {
        Flatten { start_dim, end_dim }
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.flatten(self.start_dim, self.end_dim)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![]
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// Conv1d layer
#[derive(Clone)]
pub struct Conv1d {
    weight: Rc<RefCell<Tensor>>, // [out_channels, in_channels, kernel_size]
    bias: Rc<RefCell<Tensor>>,   // [out_channels]
    stride: usize,
    padding: usize,
    _in_channels: usize,
    _out_channels: usize,
    _kernel_size: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        // Kaiming/He initialization for Conv
        let k = (1.0 / (in_channels * kernel_size) as f32).sqrt();

        let mut weight_data = Vec::with_capacity(out_channels * in_channels * kernel_size);
        for _ in 0..(out_channels * in_channels * kernel_size) {
            // Fn pointer or something else for random if needed in future
            // For now simple pattern to avoid full RNG dependency mess if randomly seeded differently
            // But let's reuse standard random pattern if available or just 0.01 scale normal approx
            weight_data.push(0.01); // PLaceholder init, ideally use random
        }

        // Better Init: Random uniform [-k, k]
        // Since we don't have easy access to robust RNG here, we'll implement a simple one inline or use fixed
        // Let's use a deterministic pseudo-random sequence for now to match Linear's style
        let weight_data: Vec<f32> = (0..(out_channels * in_channels * kernel_size))
            .map(|i| ((i as f32 * 0.123).sin()) * k)
            .collect();

        let weight = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![out_channels, in_channels, kernel_size], weight_data),
            true,
        )));

        let bias = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![out_channels], vec![0.0; out_channels]),
            true,
        )));

        Ok(Conv1d {
            weight,
            bias,
            stride,
            padding,
            _in_channels: in_channels,
            _out_channels: out_channels,
            _kernel_size: kernel_size,
        })
    }

    pub fn weight(&self) -> Rc<RefCell<Tensor>> {
        self.weight.clone()
    }

    pub fn bias(&self) -> Rc<RefCell<Tensor>> {
        self.bias.clone()
    }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w = self.weight.borrow();
        let b = self.bias.borrow();
        input.conv1d(&*w, Some(&*b), self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

/// BatchNorm1d layer
#[derive(Clone)]
pub struct BatchNorm1d {
    weight: Rc<RefCell<Tensor>>, // Gamma
    bias: Rc<RefCell<Tensor>>,   // Beta
    running_mean: Rc<RefCell<Tensor>>,
    running_var: Rc<RefCell<Tensor>>,
    momentum: f32,
    eps: f32,
    training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Result<Self> {
        // Gamma init = 1
        let weight = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![num_features], vec![1.0; num_features]),
            true,
        )));

        // Beta init = 0
        let bias = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![num_features], vec![0.0; num_features]),
            true,
        )));

        // Running stats init = 0, 1 (not trainable)
        let running_mean = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![num_features], vec![0.0; num_features]),
            true, // requires_grad=true to force Export as Initializer
        )));

        let running_var = Rc::new(RefCell::new(Tensor::new(
            Array::new(vec![num_features], vec![1.0; num_features]),
            true, // requires_grad=true to force Export as Initializer
        )));

        Ok(BatchNorm1d {
            weight,
            bias,
            running_mean,
            running_var,
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        })
    }

    pub fn running_mean(&self) -> Rc<RefCell<Tensor>> {
        self.running_mean.clone()
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w = self.weight.borrow();
        let b = self.bias.borrow();
        let mut rm = self.running_mean.borrow_mut();
        let mut rv = self.running_var.borrow_mut();

        input.batch_norm(
            &mut *rm,
            &mut *rv,
            &*w,
            &*b,
            self.training,
            self.momentum,
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Tensor>>> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() -> Result<()> {
        // Linear(2, 3) con pesos conocidos
        let linear = Linear::with_weights(
            2,
            3,
            vec![
                1.0, 2.0, // fila 1
                3.0, 4.0, // fila 2
                5.0, 6.0,
            ], // fila 3
            vec![0.1, 0.2, 0.3],
        )?;

        // Input [1, 2] con valores [1, 2]
        let input = Tensor::new(Array::new(vec![1, 2], vec![1.0, 2.0]), false);

        // Forward: [1,2] @ [2,3] + [3] = [1,3]
        // [1, 2] @ [[1, 3, 5],    = [5, 11, 17]
        //           [2, 4, 6]]
        // + [0.1, 0.2, 0.3] = [5.1, 11.2, 17.3]
        let output = linear.forward(&input)?;

        assert_eq!(output.shape(), &[1, 3]);
        let vals = output.values();
        assert!((vals[0] - 5.1).abs() < 1e-5);
        assert!((vals[1] - 11.2).abs() < 1e-5);
        assert!((vals[2] - 17.3).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_sequential() -> Result<()> {
        let model = Sequential::new(vec![
            Box::new(Linear::with_weights(
                2,
                2,
                vec![1.0, 0.0, 0.0, 1.0],
                vec![1.0, 2.0],
            )?),
            Box::new(ReLU),
        ]);

        let input = Tensor::new(Array::new(vec![1, 2], vec![1.0, 2.0]), false);
        let output = model.forward(&input)?;

        assert_eq!(output.shape(), &[1, 2]);

        Ok(())
    }
}
