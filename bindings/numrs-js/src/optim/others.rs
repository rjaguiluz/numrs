use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use numrs::autograd::optim::{
    LBFGS as CoreLBFGS, Rprop as CoreRprop, Optimizer
};
use std::rc::Rc;
use std::cell::RefCell;

fn extract_params(params: Vec<&Tensor>) -> Vec<Rc<RefCell<numrs::Tensor>>> {
    params.iter().map(|t| t.inner.clone()).collect()
}

#[napi]
pub struct LBFGS { inner: CoreLBFGS }
#[napi]
impl LBFGS {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, history_size: u32, max_iter: u32) -> Self {
        LBFGS { inner: CoreLBFGS::new(extract_params(params), lr as f32, history_size as usize, max_iter as usize) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}

#[napi]
pub struct Rprop { inner: CoreRprop }
#[napi]
impl Rprop {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr_init: f64, eta_plus: f64, eta_minus: f64, lr_min: f64, lr_max: f64) -> Self {
        Rprop { inner: CoreRprop::new(extract_params(params), lr_init as f32, eta_plus as f32, eta_minus as f32, lr_min as f32, lr_max as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}
