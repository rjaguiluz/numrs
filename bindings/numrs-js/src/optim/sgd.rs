use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use numrs::autograd::optim::{SGD as CoreSGD, Optimizer};
use std::rc::Rc;
use std::cell::RefCell;

fn extract_params(params: Vec<&Tensor>) -> Vec<Rc<RefCell<numrs::Tensor>>> {
    params.iter().map(|t| t.inner.clone()).collect()
}

#[napi]
pub struct SGD { inner: CoreSGD }
#[napi]
impl SGD {
    #[napi(constructor)]
    pub fn new(params: Vec<&Tensor>, lr: f64, momentum: f64, weight_decay: f64) -> Self {
        SGD { inner: CoreSGD::new(extract_params(params), lr as f32, momentum as f32, weight_decay as f32) }
    }
    #[napi] pub fn step(&mut self) -> Result<()> { self.inner.step().map_err(|e| Error::from_reason(e.to_string())) }
    #[napi] pub fn zero_grad(&self) { self.inner.zero_grad(); }
}
