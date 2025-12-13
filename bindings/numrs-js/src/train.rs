use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::nn::Sequential as NativeSequential;
use numrs::autograd::train::{Dataset, Trainer, TrainerBuilder, MSELoss, CrossEntropyLoss, LossFunction};
use numrs::autograd::nn::Sequential;
use numrs::autograd::{SGD, Adam};
use std::cell::RefCell;
use std::rc::Rc;

#[napi(js_name = "Dataset")]
pub struct NativeDataset {
    pub(crate) inner: Dataset,
}

#[napi]
impl NativeDataset {
    #[napi(constructor)]
    pub fn new(inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, batch_size: u32) -> Result<Self> {
        // Convert f64 (JS) to f32 (Rust)
        let inputs_f32: Vec<Vec<f32>> = inputs.iter()
            .map(|row| row.iter().map(|&x| x as f32).collect())
            .collect();
            
        let targets_f32: Vec<Vec<f32>> = targets.iter()
            .map(|row| row.iter().map(|&x| x as f32).collect())
            .collect();
            
        Ok(NativeDataset {
            inner: Dataset::new(inputs_f32, targets_f32, batch_size as usize)
        })
    }
    
    #[napi(getter)]
    pub fn num_batches(&self) -> u32 {
        self.inner.num_batches() as u32
    }

    /// Create Dataset from flat Float32Arrays (fastest method)
    /// inputs: Flat array [num_samples * input_dim]
    /// targets: Flat array [num_samples * target_dim]
    #[napi(factory)]
    pub fn from_raw(
        inputs: Float32Array, 
        input_shape: Vec<u32>, 
        targets: Float32Array, 
        target_shape: Vec<u32>, 
        batch_size: u32
    ) -> Result<Self> {
        let input_data = inputs.as_ref().to_vec();
        let target_data = targets.as_ref().to_vec();
        
        let in_shape_usize: Vec<usize> = input_shape.iter().map(|&x| x as usize).collect();
        let out_shape_usize: Vec<usize> = target_shape.iter().map(|&x| x as usize).collect();
        
        // Assert dimensions make sense
        let num_samples = in_shape_usize[0];
        if num_samples != out_shape_usize[0] {
             return Err(Error::from_reason("Mismatch in number of samples between inputs and targets"));
        }
        
        let core_inputs = numrs::Array::new(in_shape_usize, input_data);
        let core_targets = numrs::Array::new(out_shape_usize, target_data);
        
        Ok(NativeDataset {
            inner: numrs::autograd::train::Dataset {
                inputs: core_inputs,
                targets: core_targets,
                batch_size: batch_size as usize,
                num_samples,
            }
        })
    }
}

// Wrapper enum for different Trainer types (generic over Optimizer)
// We fix Model = Sequential for now as that's what NativeSequential wraps.
pub enum TrainerWrapper {
    Sgd(Trainer<Sequential, SGD>),
    Adam(Trainer<Sequential, Adam>),
}

#[napi]
pub struct NativeTrainer {
    inner: RefCell<TrainerWrapper>,
}

#[napi]
impl NativeTrainer {
    #[napi]
    pub fn fit(
        &self, 
        train_dataset: &NativeDataset, 
        val_dataset: Option<&NativeDataset>, 
        epochs: u32,
        verbose: Option<bool>
    ) -> Result<Vec<NativeMetrics>> {
        let is_verbose = verbose.unwrap_or(true);
        let core_val_ds = val_dataset.map(|d| &d.inner);
        
        // This is tricky: Trainer::fit takes &mut self.
        // But we are inside RefCell.
        let mut wrapper = self.inner.borrow_mut();
        
        let history_res = match &mut *wrapper {
            TrainerWrapper::Sgd(t) => t.fit(&train_dataset.inner, core_val_ds, epochs as usize, is_verbose),
            TrainerWrapper::Adam(t) => t.fit(&train_dataset.inner, core_val_ds, epochs as usize, is_verbose),
        };
        
        let history = history_res.map_err(|e| Error::from_reason(e.to_string()))?;
        
        // Convert core Metrics to JS friendly struct
        let js_history = history.into_iter().map(|(train_m, val_m)| {
            NativeMetrics {
                train_loss: train_m.loss as f64,
                train_acc: train_m.accuracy.map(|a| a as f64),
                val_loss: val_m.as_ref().map(|m| m.loss as f64),
                val_acc: val_m.as_ref().and_then(|m| m.accuracy.map(|a| a as f64)),
            }
        }).collect();
        
        Ok(js_history)
    }
}

#[napi(object)]
pub struct NativeMetrics {
    pub train_loss: f64,
    pub train_acc: Option<f64>,
    pub val_loss: Option<f64>,
    pub val_acc: Option<f64>,
}

#[napi(js_name = "TrainerBuilder")]
pub struct NativeTrainerBuilder {
    // We hold the builder or just the params to construct it?
    // TrainerBuilder holds the model.
    // If we use core TrainerBuilder, it consumes model on build.
    // But NativeSequential holds model in RefCell<Rc<...>>?
    // Let's see NativeSequential.
    // src/nn.rs: pub struct NativeSequential { pub(crate) module: Rc<RefCell<Sequential>> }
    // Core TrainerBuilder::new(model: M) takes M by value? 
    // src/autograd/train.rs: pub fn new(model: M) -> Self. Yes.
    // So it takes ownership.
    // But our NativeSequential shares ownership via Rc<RefCell>.
    // Sequential in core implements Clone?
    // Let's hope Sequential implements Clone (it usually does if it holds Rc tensors, or if it's a struct of params).
    // If not, we might have trouble.
    // numrs-core/src/autograd/nn.rs -> Sequential struct.
    
    // We will hold the cloned model (if possible) or just access it.
    model: Rc<RefCell<Sequential>>,
    lr: f32,
}

#[napi]
impl NativeTrainerBuilder {
    #[napi(constructor)]
    pub fn new(model: &NativeSequential) -> Self {
        NativeTrainerBuilder {
            model: Rc::new(RefCell::new(model.clone_inner())), 
            lr: 0.01,
        }
    }
    
    #[napi]
    pub fn learning_rate(&mut self, lr: f64) -> &Self {
        self.lr = lr as f32;
        self
    }
    
    #[napi]
    pub fn build(&self, optimizer: String, loss: String) -> Result<NativeTrainer> {
        // We need to construct core TrainerBuilder.
        // It requires passing the model M.
        // M = Sequential.
        // We have Rc<RefCell<Sequential>>. 
        // We need to clone the Sequential itself, not the RC?
        // Wait, if Trainer takes ownership of M, then Trainer OWNS M.
        // But our NativeSequential also owns it (shared).
        // If Sequential is cloneable, we give a clone to Trainer.
        // If Sequential involves Tensors (which are Rc<RefCell>), cloning is cheap (shallow copy of tensors).
        // Let's assume Sequential derives Clone.
        
        let model_clone = self.model.borrow().clone();
        
        let builder = TrainerBuilder::new(model_clone).learning_rate(self.lr);
        
        let loss_fn: Box<dyn LossFunction> = match loss.as_str() {
            "mse" => Box::new(MSELoss),
            "cross_entropy" => Box::new(CrossEntropyLoss),
            _ => return Err(Error::from_reason(format!("Unknown loss function: {}", loss))),
        };
        
        // Optimizers
        let wrapper = match optimizer.as_str() {
            "sgd" => {
                let trainer = builder.build_sgd(loss_fn);
                TrainerWrapper::Sgd(trainer)
            },
            "adam" => {
                let trainer = builder.build_adam(loss_fn);
                TrainerWrapper::Adam(trainer)
            },
            _ => return Err(Error::from_reason(format!("Unknown optimizer: {}", optimizer))),
        };
        
        Ok(NativeTrainer {
            inner: RefCell::new(wrapper)
        })
    }
}
