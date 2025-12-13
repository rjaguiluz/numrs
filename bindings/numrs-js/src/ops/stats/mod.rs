use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use numrs::ops::stats;
use std::rc::Rc;
use std::cell::RefCell;

#[napi]
pub fn cross_entropy(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    let res = input.inner.borrow().cross_entropy_loss(&*target.inner.borrow())
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

#[napi]
pub fn norm(input: &Tensor) -> Result<Tensor> {
    let inner_tensor = input.inner.borrow();
    let res_arr = stats::norm::norm(&inner_tensor.data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let res_tensor = numrs::Tensor::new(res_arr, false);
    Ok(Tensor { inner: Rc::new(RefCell::new(res_tensor)) })
}

#[napi]
pub fn softmax(input: &Tensor, dim: Option<i32>) -> Result<Tensor> {
    let inner_tensor = input.inner.borrow();
    let axis = dim.map(|d| d as usize);
    
    let res_arr = stats::softmax(&inner_tensor.data, axis)
        .map_err(|e| Error::from_reason(e.to_string()))?;
        
    let res_tensor = numrs::Tensor::new(res_arr, false);
    Ok(Tensor { inner: Rc::new(RefCell::new(res_tensor)) })
}
