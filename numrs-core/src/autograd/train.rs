//! High-level Training API
//! 
//! Simplifica el training loop:
//! - Trainer para ejecutar epochs
//! - Métricas de evaluación
//! - Early stopping

use crate::{Tensor, Array};
use crate::autograd::{Module, Optimizer};
use anyhow::Result;


/// Función de loss
pub trait LossFunction {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
}

/// Mean Squared Error Loss
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        predictions.mse_loss(targets)
    }
}

/// Cross Entropy Loss
pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        predictions.cross_entropy_loss(targets)
    }
}

/// Métricas de evaluación
pub struct Metrics {
    pub loss: f32,
    pub accuracy: Option<f32>,
}

impl Metrics {
    pub fn new(loss: f32) -> Self {
        Metrics { loss, accuracy: None }
    }
    
    pub fn with_accuracy(loss: f32, accuracy: f32) -> Self {
        Metrics { loss, accuracy: Some(accuracy) }
    }
}

/// Dataset simple para training
pub struct Dataset {
    pub inputs: Array<f32>,
    pub targets: Array<f32>,
    pub batch_size: usize,
    pub num_samples: usize,
}

impl Dataset {
    pub fn new(inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>, batch_size: usize) -> Self {
        assert_eq!(inputs.len(), targets.len(), "Inputs y targets deben tener mismo tamaño");
        let num_samples = inputs.len();
        let input_dim = if num_samples > 0 { inputs[0].len() } else { 0 };
        let target_dim = if num_samples > 0 { targets[0].len() } else { 0 };
        
        // Flatten inputs
        let mut flat_inputs = Vec::with_capacity(num_samples * input_dim);
        for row in &inputs {
            flat_inputs.extend_from_slice(row);
        }
        
        // Flatten targets
        let mut flat_targets = Vec::with_capacity(num_samples * target_dim);
        for row in &targets {
            flat_targets.extend_from_slice(row);
        }
        
        Dataset { 
            inputs: Array::new(vec![num_samples, input_dim], flat_inputs),
            targets: Array::new(vec![num_samples, target_dim], flat_targets),
            batch_size,
            num_samples
        }
    }
    
    /// Devuelve el número de batches
    pub fn num_batches(&self) -> usize {
        (self.num_samples + self.batch_size - 1) / self.batch_size
    }
    
    /// Devuelve un batch específico
    pub fn get_batch(&self, batch_idx: usize) -> Result<(Tensor, Tensor)> {
        let start = batch_idx * self.batch_size;
        let end = (start + self.batch_size).min(self.num_samples);
        
        if start >= self.num_samples {
            return Err(anyhow::anyhow!("Batch index fuera de rango"));
        }
        
        let actual_batch_size = end - start;
        let input_dim = self.inputs.shape[1];
        let target_dim = self.targets.shape[1];
        
        // Slice input data efficiently (contiguous slice)
        let input_start_idx = start * input_dim;
        let input_end_idx = end * input_dim;
        let batch_inputs_data = self.inputs.data[input_start_idx..input_end_idx].to_vec();
        
        // Slice target data efficiently
        let target_start_idx = start * target_dim;
        let target_end_idx = end * target_dim;
        let batch_targets_data = self.targets.data[target_start_idx..target_end_idx].to_vec();
        
        let inputs_tensor = Tensor::new(
            Array::new(vec![actual_batch_size, input_dim], batch_inputs_data),
            false
        );
        
        let targets_tensor = Tensor::new(
            Array::new(vec![actual_batch_size, target_dim], batch_targets_data),
            false
        );
        
        Ok((inputs_tensor, targets_tensor))
    }
}

/// Trainer de alto nivel
/// 
/// Ejemplo:
/// ```ignore
/// let trainer = Trainer::new(model, optimizer, MSELoss);
/// let metrics = trainer.train_epoch(&dataset)?;
/// println!("Loss: {:.4}", metrics.loss);
/// ```
pub struct Trainer<M: Module, O: Optimizer> {
    pub model: M,
    optimizer: O,
    loss_fn: Box<dyn LossFunction>,
}

impl<M: Module, O: Optimizer> Trainer<M, O> {
    pub fn new(model: M, optimizer: O, loss_fn: Box<dyn LossFunction>) -> Self {
        Trainer { model, optimizer, loss_fn }
    }
    
    /// Get reference to the model (for debugging)
    pub fn model(&self) -> &M {
        &self.model
    }
    
    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
    
    /// Entrena una época completa
    pub fn train_epoch(&mut self, dataset: &Dataset) -> Result<Metrics> {
        let mut total_loss = 0.0;
        let num_batches = dataset.num_batches();
        
        for batch_idx in 0..num_batches {
            let (inputs, targets) = dataset.get_batch(batch_idx)?;
            
            // Forward pass
            let predictions = self.model.forward(&inputs)?;
            
            // Compute loss
            let loss = self.loss_fn.compute(&predictions, &targets)?;
            total_loss += loss.values()[0];
            
            // Backward pass
            loss.backward()?;
            
            // Update weights
            self.optimizer.step()?;
            self.optimizer.zero_grad();
        }
        
        let avg_loss = total_loss / num_batches as f32;
        Ok(Metrics::new(avg_loss))
    }
    
    /// Evalúa el modelo sin actualizar pesos
    pub fn evaluate(&self, dataset: &Dataset) -> Result<Metrics> {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;
        let num_batches = dataset.num_batches();
        
        for batch_idx in 0..num_batches {
            let (inputs, targets) = dataset.get_batch(batch_idx)?;
            
            // Forward pass (sin gradientes)
            let predictions = self.model.forward(&inputs)?;
            
            // Compute loss
            let loss = self.loss_fn.compute(&predictions, &targets)?;
            total_loss += loss.values()[0];
            
            // Compute accuracy (para clasificación)
            let pred_vals = predictions.values();
            let target_vals = targets.values();
            
            let batch_size = predictions.shape()[0];
            let num_classes = predictions.shape()[1];
            
            for i in 0..batch_size {
                // Argmax de predicciones
                let pred_start = i * num_classes;
                let pred_end = pred_start + num_classes;
                let pred_class = pred_vals[pred_start..pred_end]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0); // Default to class 0 if empty (should not happen)
                
                // Argmax de targets
                let target_start = i * num_classes;
                let target_end = target_start + num_classes;
                let target_class = target_vals[target_start..target_end]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                
                if pred_class == target_class {
                    correct += 1;
                }
                total += 1;
            }
        }
        
        let avg_loss = total_loss / num_batches as f32;
        let accuracy = correct as f32 / total as f32;
        
        Ok(Metrics::with_accuracy(avg_loss, accuracy))
    }
    
    /// Training loop completo con múltiples epochs
    pub fn fit(
        &mut self,
        train_dataset: &Dataset,
        val_dataset: Option<&Dataset>,
        epochs: usize,
        verbose: bool,
    ) -> Result<Vec<(Metrics, Option<Metrics>)>> {
        let mut history = Vec::new();
        
        for epoch in 0..epochs {
            // Train
            let train_metrics = self.train_epoch(train_dataset)?;
            
            // Validate
            let val_metrics = if let Some(val_ds) = val_dataset {
                Some(self.evaluate(val_ds)?)
            } else {
                None
            };
            
            if verbose {
                print!("Epoch {}/{}: train_loss={:.4}", epoch + 1, epochs, train_metrics.loss);
                
                if let Some(ref vm) = val_metrics {
                    print!(", val_loss={:.4}", vm.loss);
                    if let Some(acc) = vm.accuracy {
                        print!(", val_acc={:.4}", acc);
                    }
                }
                println!();
            }
            
            history.push((train_metrics, val_metrics));
        }
        
        Ok(history)
    }
}

/// Builder para crear Trainer fácilmente
pub struct TrainerBuilder<M: Module> {
    model: M,
    learning_rate: f32,
}

impl<M: Module> TrainerBuilder<M> {
    pub fn new(model: M) -> Self {
        TrainerBuilder {
            model,
            learning_rate: 0.01,
        }
    }
    
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }
    
    pub fn build_sgd(self, loss_fn: Box<dyn LossFunction>) -> Trainer<M, crate::autograd::SGD> {
        let params = self.model.parameters();
        let optimizer = crate::autograd::SGD::new(params, self.learning_rate, 0.9, 0.0);
        Trainer::new(self.model, optimizer, loss_fn)
    }
    
    pub fn build_adam(self, loss_fn: Box<dyn LossFunction>) -> Trainer<M, crate::autograd::Adam> {
        let params = self.model.parameters();
        let optimizer = crate::autograd::Adam::with_lr(params, self.learning_rate);
        Trainer::new(self.model, optimizer, loss_fn)
    }

    /// Generic builder for any optimizer
    pub fn build_with<O, F>(self, optimizer_factory: F, loss_fn: Box<dyn LossFunction>) -> Trainer<M, O>
    where 
        O: Optimizer,
        F: FnOnce(Vec<std::rc::Rc<std::cell::RefCell<crate::Tensor>>>, f32) -> O,
    {
        let params = self.model.parameters();
        let optimizer = optimizer_factory(params, self.learning_rate);
        Trainer::new(self.model, optimizer, loss_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dataset() -> Result<()> {
        let inputs = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let targets = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ];
        
        let dataset = Dataset::new(inputs, targets, 2);
        assert_eq!(dataset.num_batches(), 2);
        
        let (batch_inputs, batch_targets) = dataset.get_batch(0)?;
        assert_eq!(batch_inputs.shape(), &[2, 2]);
        assert_eq!(batch_targets.shape(), &[2, 1]);
        
        Ok(())
    }
}
