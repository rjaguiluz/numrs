use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use numrs::autograd::optim::{
    Adam as CoreAdam, AdamW as CoreAdamW, NAdam as CoreNAdam, RAdam as CoreRAdam,
    AdaBound as CoreAdaBound, LAMB as CoreLAMB, Optimizer
};
use std::rc::Rc;
use std::cell::RefCell;

fn extract_params(params: Vec<&Tensor>) -> Vec<Rc<RefCell<numrs::Tensor>>> {
    params.iter().map(|t| t.inner.clone()).collect()
}

#[napi]
pub struct Adam { inner: CoreAdam }
#[napi]
impl Adam {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Adam { inner: CoreAdam::new(extract_params(params), lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}

#[napi]
pub struct AdamW { inner: CoreAdamW }
#[napi]
impl AdamW {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        AdamW { inner: CoreAdamW::new(extract_params(params), lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}

#[napi]
pub struct NAdam { inner: CoreNAdam }
#[napi]
impl NAdam {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        NAdam { inner: CoreNAdam::new(extract_params(params), lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}

#[napi]
pub struct RAdam { inner: CoreRAdam }
#[napi]
impl RAdam {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        RAdam { inner: CoreRAdam::new(extract_params(params), lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}

#[napi]
pub struct LAMB { inner: CoreLAMB }
#[napi]
impl LAMB {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        LAMB { inner: CoreLAMB::new(extract_params(params), lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}

#[napi]
pub struct AdaBound { inner: CoreAdaBound }
#[napi]
impl AdaBound {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, final_lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64, gamma: f64) -> Self {
        AdaBound { inner: CoreAdaBound::new(extract_params(params), lr as f32, final_lr as f32, beta1 as f32, beta2 as f32, eps as f32, weight_decay as f32, gamma as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}
