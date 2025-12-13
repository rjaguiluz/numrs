use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use numrs::ops::reduction;

#[napi]
pub fn argmax(input: &Tensor, axis: Option<i32>) -> Result<Tensor> {
    let inner = input.inner.borrow();
    let ax = axis.map(|d| d as usize);
    let res_arr = reduction::argmax::argmax(&inner.data, ax)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
}

#[napi]
pub fn max(input: &Tensor, axis: Option<i32>) -> Result<Tensor> {
    let inner = input.inner.borrow();
    let ax = axis.map(|d| d as usize);
    let res_arr = reduction::max::max(&inner.data, ax)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
}

#[napi]
pub fn min(input: &Tensor, axis: Option<i32>) -> Result<Tensor> {
    let inner = input.inner.borrow();
    let ax = axis.map(|d| d as usize);
    let res_arr = reduction::min::min(&inner.data, ax)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
}

#[napi]
pub fn variance(input: &Tensor, axis: Option<i32>) -> Result<Tensor> {
    let inner = input.inner.borrow();
    let ax = axis.map(|d| d as usize);
    // Fixed: Pass 2 args only as per previous error diagnostic
    let res_arr = reduction::variance::variance(&inner.data, ax)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
}

#[napi]
pub fn sum(input: &Tensor) -> Result<Tensor> {
    let res = input.inner.borrow().sum()
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

#[napi]
pub fn mean(input: &Tensor) -> Result<Tensor> {
    let res = input.inner.borrow().mean()
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

// Tensor methods for convenience
#[napi]
impl Tensor {
    #[napi]
    pub fn sum(&self) -> Result<Tensor> {
        let res = self.inner.borrow().sum()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn mean(&self) -> Result<Tensor> {
        let res = self.inner.borrow().mean()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
    
    #[napi]
    pub fn mse_loss(&self, target: &Tensor) -> Result<Tensor> {
        let res = self.inner.borrow().mse_loss(&*target.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}
