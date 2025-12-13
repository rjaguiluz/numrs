use crate::nn::Sequential;
use crate::tensor::Tensor;
use numrs::autograd::optim::{
    AdaBound, AdaDelta, AdaGrad, Adam, AdamW, NAdam, RAdam, RMSprop, Rprop, LAMB, LBFGS, SGD,
};
use numrs::autograd::train::{
    CrossEntropyLoss, LossFunction, MSELoss, Trainer as CoreTrainer, TrainerBuilder,
};
use numrs::autograd::Sequential as CoreSequential;
use wasm_bindgen::prelude::*;

// Wrapper enum similar to C binding
pub enum TrainerWrapper {
    Sgd(CoreTrainer<CoreSequential, SGD>),
    Adam(CoreTrainer<CoreSequential, Adam>),
    AdamW(CoreTrainer<CoreSequential, AdamW>),
    NAdam(CoreTrainer<CoreSequential, NAdam>),
    RAdam(CoreTrainer<CoreSequential, RAdam>),
    Lamb(CoreTrainer<CoreSequential, LAMB>),
    AdaBound(CoreTrainer<CoreSequential, AdaBound>),
    RmsProp(CoreTrainer<CoreSequential, RMSprop>),
    AdaGrad(CoreTrainer<CoreSequential, AdaGrad>),
    AdaDelta(CoreTrainer<CoreSequential, AdaDelta>),
    Lbfgs(CoreTrainer<CoreSequential, LBFGS>),
    Rprop(CoreTrainer<CoreSequential, Rprop>),
}

#[wasm_bindgen]
pub struct Trainer {
    inner: TrainerWrapper,
}

#[wasm_bindgen]
impl Trainer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        model: &Sequential,
        optimizer: String,
        loss: String,
        lr: Option<f32>,
    ) -> Result<Trainer, JsValue> {
        let core_model = model.inner.clone(); // Clone the sequential model (it's Rc-wrapped internally usually)
                                              // Wait, Sequential in WASM wraps CoreSequential directly.
                                              // We might need to make sure we're cloning the structure properly or references?
                                              // Sequential::inner is `pub(crate) inner: CoreSequential`. CoreSequential usually owns its layers.
                                              // If we clone it, do we share weights?
                                              // CoreSequential has `layers: Vec<Box<dyn Module>>`.
                                              // cloning CoreSequential performs a deep clone of structure but layers?
                                              // Core layers like Linear usually wrap params in Rc<RefCell<Tensor>>. So cloning them shares state. Correct.

        let mut builder = TrainerBuilder::new(core_model);
        if let Some(learning_rate) = lr {
            builder = builder.learning_rate(learning_rate);
        }

        let loss_fn: Box<dyn LossFunction> = match loss.as_str() {
            "mse" => Box::new(MSELoss),
            "cross_entropy" => Box::new(CrossEntropyLoss),
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown loss function: {}",
                    loss
                )))
            }
        };

        let wrapper = match optimizer.as_str() {
            "sgd" => TrainerWrapper::Sgd(builder.build_sgd(loss_fn)),
            "adam" => TrainerWrapper::Adam(builder.build_adam(loss_fn)),
            // Using default params for others as used in C binding
            "adamw" => TrainerWrapper::AdamW(
                builder.build_with(|p, lr| AdamW::new(p, lr, 0.9, 0.999, 1e-8, 0.01), loss_fn),
            ),
            "nadam" => TrainerWrapper::NAdam(
                builder.build_with(|p, lr| NAdam::new(p, lr, 0.9, 0.999, 1e-8, 0.004), loss_fn),
            ),
            "radam" => TrainerWrapper::RAdam(
                builder.build_with(|p, lr| RAdam::new(p, lr, 0.9, 0.999, 1e-8, 0.0), loss_fn),
            ),
            "lamb" => TrainerWrapper::Lamb(
                builder.build_with(|p, lr| LAMB::new(p, lr, 0.9, 0.999, 1e-8, 0.01), loss_fn),
            ),
            "adabound" => TrainerWrapper::AdaBound(builder.build_with(
                |p, lr| AdaBound::new(p, lr, 0.1, 0.9, 0.999, 1e-8, 0.0, 1e-3),
                loss_fn,
            )),
            "rmsprop" => TrainerWrapper::RmsProp(
                builder.build_with(|p, lr| RMSprop::new(p, lr, 0.99, 1e-8, 0.0, 0.0), loss_fn),
            ),
            "adagrad" => TrainerWrapper::AdaGrad(
                builder.build_with(|p, lr| AdaGrad::new(p, lr, 1e-10, 0.0), loss_fn),
            ),
            "adadelta" => TrainerWrapper::AdaDelta(
                builder.build_with(|p, _lr| AdaDelta::new(p, 0.9, 1e-6, 0.0), loss_fn),
            ),
            "lbfgs" => TrainerWrapper::Lbfgs(
                builder.build_with(|p, lr| LBFGS::new(p, lr, 100, 20), loss_fn),
            ),
            "rprop" => TrainerWrapper::Rprop(
                builder.build_with(|p, _lr| Rprop::new(p, 0.001, 1.2, 0.5, 1e-6, 50.0), loss_fn),
            ),
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown optimizer: {}",
                    optimizer
                )))
            }
        };

        Ok(Trainer { inner: wrapper })
    }

    pub fn fit(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        epochs: usize,
    ) -> Result<Vec<f32>, JsValue> {
        let inputs_arr = inputs.inner.borrow().data.clone();
        let targets_arr = targets.inner.borrow().data.clone();

        let dataset = numrs::autograd::train::Dataset {
            inputs: inputs_arr,
            targets: targets_arr,
            batch_size: 32, // Default batch size, maybe expose it?
            num_samples: inputs.inner.borrow().data.shape[0],
        };

        let history_result = match &mut self.inner {
            TrainerWrapper::Sgd(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::Adam(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::AdamW(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::NAdam(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::RAdam(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::Lamb(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::AdaBound(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::RmsProp(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::AdaGrad(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::AdaDelta(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::Lbfgs(t) => t.fit(&dataset, None, epochs, false),
            TrainerWrapper::Rprop(t) => t.fit(&dataset, None, epochs, false),
        };

        match history_result {
            Ok(history) => {
                let losses: Vec<f32> = history.into_iter().map(|(m, _)| m.loss).collect();
                Ok(losses)
            }
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    pub fn train_step(&mut self, inputs: &Tensor, targets: &Tensor) -> Result<f32, JsValue> {
        let inputs_arr = inputs.inner.borrow().data.clone();
        let targets_arr = targets.inner.borrow().data.clone();
        let num_samples = inputs_arr.shape[0];

        // Create a dataset that acts as a single batch
        let dataset = numrs::autograd::train::Dataset {
            inputs: inputs_arr,
            targets: targets_arr,
            batch_size: num_samples, // Full batch
            num_samples,
        };

        let result = match &mut self.inner {
            TrainerWrapper::Sgd(t) => t.train_epoch(&dataset),
            TrainerWrapper::Adam(t) => t.train_epoch(&dataset),
            TrainerWrapper::AdamW(t) => t.train_epoch(&dataset),
            TrainerWrapper::NAdam(t) => t.train_epoch(&dataset),
            TrainerWrapper::RAdam(t) => t.train_epoch(&dataset),
            TrainerWrapper::Lamb(t) => t.train_epoch(&dataset),
            TrainerWrapper::AdaBound(t) => t.train_epoch(&dataset),
            TrainerWrapper::RmsProp(t) => t.train_epoch(&dataset),
            TrainerWrapper::AdaGrad(t) => t.train_epoch(&dataset),
            TrainerWrapper::AdaDelta(t) => t.train_epoch(&dataset),
            TrainerWrapper::Lbfgs(t) => t.train_epoch(&dataset),
            TrainerWrapper::Rprop(t) => t.train_epoch(&dataset),
        };

        result
            .map(|metrics| metrics.loss)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn evaluate(&self, inputs: &Tensor, targets: &Tensor) -> Result<f32, JsValue> {
        let inputs_arr = inputs.inner.borrow().data.clone();
        let targets_arr = targets.inner.borrow().data.clone();
        let num_samples = inputs_arr.shape[0];

        let dataset = numrs::autograd::train::Dataset {
            inputs: inputs_arr,
            targets: targets_arr,
            batch_size: num_samples,
            num_samples,
        };

        let result = match &self.inner {
            TrainerWrapper::Sgd(t) => t.evaluate(&dataset),
            TrainerWrapper::Adam(t) => t.evaluate(&dataset),
            TrainerWrapper::AdamW(t) => t.evaluate(&dataset),
            TrainerWrapper::NAdam(t) => t.evaluate(&dataset),
            TrainerWrapper::RAdam(t) => t.evaluate(&dataset),
            TrainerWrapper::Lamb(t) => t.evaluate(&dataset),
            TrainerWrapper::AdaBound(t) => t.evaluate(&dataset),
            TrainerWrapper::RmsProp(t) => t.evaluate(&dataset),
            TrainerWrapper::AdaGrad(t) => t.evaluate(&dataset),
            TrainerWrapper::AdaDelta(t) => t.evaluate(&dataset),
            TrainerWrapper::Lbfgs(t) => t.evaluate(&dataset),
            TrainerWrapper::Rprop(t) => t.evaluate(&dataset),
        };

        result
            .map(|metrics| metrics.loss)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
