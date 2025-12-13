use crate::array::NumRsArray;
use crate::nn::NumRsSequential;
use crate::tensor::NumRsTensor;
use numrs::autograd::train::{Dataset, Trainer, TrainerBuilder};
use numrs::Array;
use std::ffi::c_void;
use std::slice;

pub struct NumRsDataset {
    pub(crate) inner: Dataset,
}

pub struct NumRsTrainerBuilder {
    pub(crate) inner: TrainerBuilder<numrs::autograd::Sequential>, 
}


// --- Dataset ---

#[no_mangle]
pub unsafe extern "C" fn numrs_dataset_new(
    inputs: *const f32,
    input_shape: *const u32,
    input_ndim: usize,
    targets: *const f32,
    target_shape: *const u32,
    target_ndim: usize,
    batch_size: usize,
) -> *mut NumRsDataset {
    // Read shapes
    let in_shape_slice = std::slice::from_raw_parts(input_shape, input_ndim);
    let in_shape: Vec<usize> = in_shape_slice.iter().map(|&x| x as usize).collect();
    
    let out_shape_slice = std::slice::from_raw_parts(target_shape, target_ndim);
    let out_shape: Vec<usize> = out_shape_slice.iter().map(|&x| x as usize).collect();
    
    let num_samples = in_shape[0];
    let in_elements: usize = in_shape.iter().product();
    let out_elements: usize = out_shape.iter().product();
    
    // Copy data
    let in_data = std::slice::from_raw_parts(inputs, in_elements).to_vec();
    let out_data = std::slice::from_raw_parts(targets, out_elements).to_vec();
    
    let core_inputs = numrs::Array::new(in_shape, in_data);
    let core_targets = numrs::Array::new(out_shape, out_data);
    
    let dataset = Dataset {
        inputs: core_inputs,
        targets: core_targets,
        batch_size,
        num_samples
    };
    
    Box::into_raw(Box::new(NumRsDataset { inner: dataset }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_dataset_free(ptr: *mut NumRsDataset) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

// --- Trainer Builder ---

#[no_mangle]
pub unsafe extern "C" fn numrs_trainer_builder_new(model: *mut NumRsSequential) -> *mut NumRsTrainerBuilder {
    if model.is_null() {
        return std::ptr::null_mut();
    }
    // Deep clone the Sequential model to pass to TrainerBuilder
    let model_val = (*model).inner.borrow().clone();
    
    let builder = TrainerBuilder::new(model_val);
    Box::into_raw(Box::new(NumRsTrainerBuilder { inner: builder }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_trainer_builder_learning_rate(builder: *mut NumRsTrainerBuilder, lr: f64) -> *mut NumRsTrainerBuilder {
    if !builder.is_null() {
        let b = Box::from_raw(builder); 
        let new_inner = b.inner.learning_rate(lr as f32);
        let b = Box::new(NumRsTrainerBuilder { inner: new_inner });
        Box::into_raw(b)
    } else {
        std::ptr::null_mut()
    }
}

use std::ffi::{CStr, c_char};
use numrs::autograd::train::{CrossEntropyLoss, MSELoss, LossFunction};
use numrs::autograd::optim::{
    SGD, Adam, AdamW, NAdam, RAdam, LAMB, AdaBound, 
    RMSprop, AdaGrad, AdaDelta,
    LBFGS, Rprop
};
use crate::optim::TrainerWrapper;

pub struct NumRsTrainer {
    pub(crate) inner: std::cell::RefCell<TrainerWrapper>,
}

#[no_mangle]
pub unsafe extern "C" fn numrs_trainer_build(
    builder: *mut NumRsTrainerBuilder,
    optimizer: *const c_char,
    loss: *const c_char,
) -> *mut NumRsTrainer {
    if builder.is_null() || optimizer.is_null() || loss.is_null() {
        return std::ptr::null_mut();
    }
    
    let b = Box::from_raw(builder);
    let opt_str = CStr::from_ptr(optimizer).to_string_lossy();
    let loss_str = CStr::from_ptr(loss).to_string_lossy();
    
    // Select Loss
    let loss_fn: Box<dyn LossFunction> = match loss_str.as_ref() {
        "mse" => Box::new(MSELoss),
        "cross_entropy" => Box::new(CrossEntropyLoss),
        _ => return std::ptr::null_mut(), // Unknown loss
    };
    
    // Select Optimizer and Build Wrapper (using default params for now)
    let wrapper = match opt_str.as_ref() {
        "sgd" => TrainerWrapper::Sgd(b.inner.build_sgd(loss_fn)),
        "adam" => TrainerWrapper::Adam(b.inner.build_adam(loss_fn)),
        "adamw" => TrainerWrapper::AdamW(b.inner.build_with(|p, lr| AdamW::new(p, lr, 0.9, 0.999, 1e-8, 0.01), loss_fn)),
        "nadam" => TrainerWrapper::NAdam(b.inner.build_with(|p, lr| NAdam::new(p, lr, 0.9, 0.999, 1e-8, 0.004), loss_fn)),
        "radam" => TrainerWrapper::RAdam(b.inner.build_with(|p, lr| RAdam::new(p, lr, 0.9, 0.999, 1e-8, 0.0), loss_fn)),
        "lamb" => TrainerWrapper::Lamb(b.inner.build_with(|p, lr| LAMB::new(p, lr, 0.9, 0.999, 1e-8, 0.01), loss_fn)),
        "adabound" => TrainerWrapper::AdaBound(b.inner.build_with(|p, lr| AdaBound::new(p, lr, 0.1, 0.9, 0.999, 1e-8, 0.0, 1e-3), loss_fn)),
        "rmsprop" => TrainerWrapper::RmsProp(b.inner.build_with(|p, lr| RMSprop::new(p, lr, 0.99, 1e-8, 0.0, 0.0), loss_fn)),
        "adagrad" => TrainerWrapper::AdaGrad(b.inner.build_with(|p, lr| AdaGrad::new(p, lr, 1e-10, 0.0), loss_fn)),
        "adadelta" => TrainerWrapper::AdaDelta(b.inner.build_with(|p, _lr| AdaDelta::new(p, 0.9, 1e-6, 0.0), loss_fn)), // AdaDelta often ignores global LR or treats it differently
        "lbfgs" => TrainerWrapper::Lbfgs(b.inner.build_with(|p, lr| LBFGS::new(p, lr, 100, 20), loss_fn)),
        "rprop" => TrainerWrapper::Rprop(b.inner.build_with(|p, _lr| Rprop::new(p, 0.001, 1.2, 0.5, 1e-6, 50.0), loss_fn)), // Rprop uses internal step sizes, not standard LR usually
        _ => return std::ptr::null_mut(), // Unknown optimizer
    };
    
    Box::into_raw(Box::new(NumRsTrainer { inner: std::cell::RefCell::new(wrapper) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_trainer_fit(
    trainer: *mut NumRsTrainer,
    train_data: *mut NumRsDataset,
    epochs: usize
) {
    if trainer.is_null() || train_data.is_null() {
        return;
    }
    let t_cell = &mut (*trainer).inner;
    let mut t = t_cell.borrow_mut();
    
    // Pass the wrapper (NumRsDataset) as expected by TrainerWrapper::fit
    let ds_wrapper = &*train_data;
    t.fit(ds_wrapper, epochs);
}


#[no_mangle]
pub unsafe extern "C" fn numrs_trainer_free(ptr: *mut NumRsTrainer) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}
