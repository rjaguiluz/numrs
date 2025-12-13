use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use numrs::autograd::optim::{
    RMSprop as CoreRMSprop, AdaGrad as CoreAdaGrad, AdaDelta as CoreAdaDelta, Optimizer
};
use std::rc::Rc;
use std::cell::RefCell;

fn extract_params(params: Vec<&Tensor>) -> Vec<Rc<RefCell<numrs::Tensor>>> {
    params.iter().map(|t| t.inner.clone()).collect()
}

#[napi]
pub struct RMSprop { inner: CoreRMSprop }
#[napi]
impl RMSprop {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, alpha: f64, eps: f64, weight_decay: f64, momentum: f64) -> Self {
        RMSprop { inner: CoreRMSprop::new(extract_params(params), lr as f32, alpha as f32, eps as f32, weight_decay as f32, momentum as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}

#[napi]
pub struct AdaGrad { inner: CoreAdaGrad }
#[napi]
impl AdaGrad {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, eps: f64, weight_decay: f64) -> Self {
        AdaGrad { inner: CoreAdaGrad::new(extract_params(params), lr as f32, eps as f32, weight_decay as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}

#[napi]
pub struct AdaDelta { inner: CoreAdaDelta }
#[napi]
impl AdaDelta {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, rho: f64, eps: f64, weight_decay: f64) -> Self {
        AdaDelta { inner: CoreAdaDelta::new(extract_params(params), rho as f32, eps as f32, weight_decay as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}
