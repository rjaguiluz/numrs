use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use std::rc::Rc;
use std::cell::RefCell;
use numrs::ops::shape;

// Standalone functions
#[napi]
pub fn broadcast_to(input: &Tensor, shape: Vec<u32>) -> Result<Tensor> {
    let inner = input.inner.borrow();
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let res_arr = shape::broadcast_to::broadcast_to(&inner.data, &shape_usize)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
}

#[napi]
pub fn concat(inputs: Vec<&Tensor>, axis: u32) -> Result<Tensor> {
    let guards: Vec<_> = inputs.iter().map(|t| t.inner.borrow()).collect();
    let arrays: Vec<&numrs::Array> = guards.iter().map(|g| &g.data).collect();
    
    let res_arr = shape::concat::concat(&arrays, axis as usize)
        .map_err(|e| Error::from_reason(e.to_string()))?;
        
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
}

#[napi]
pub fn reshape(input: &Tensor, shape: Vec<u32>) -> Result<Tensor> {
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let res = input.inner.borrow().reshape(shape_usize)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

#[napi]
pub fn flatten(input: &Tensor, start: Option<i32>, end: Option<i32>) -> Result<Tensor> {
    let s = start.unwrap_or(0) as usize;
    // Handle -1 or default end
    let ndim = input.inner.borrow().shape().len();
    let e = match end {
        Some(val) if val < 0 => ndim.saturating_sub(1),
        Some(val) => val as usize,
        None => ndim.saturating_sub(1)
    };
    
    let res = input.inner.borrow().flatten(s, e)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

#[napi]
pub fn transpose(input: &Tensor, dim0: Option<i32>, dim1: Option<i32>) -> Result<Tensor> {
    if dim0.is_some() || dim1.is_some() {
        return Err(Error::from_reason("Transpose with specific dimensions not yet supported by autograd Tensor"));
    }
    let res = input.inner.borrow().transpose()
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

// Tensor methods
#[napi]
impl Tensor {
    #[napi]
    pub fn reshape(&self, shape: Vec<u32>) -> Result<Tensor> {
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let res = self.inner.borrow().reshape(shape_usize)
             .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn flatten(&self, start_dim: Option<u32>, end_dim: Option<u32>) -> Result<Tensor> {
        let s = start_dim.unwrap_or(0) as usize;
        let e = end_dim.unwrap_or(self.inner.borrow().shape().len() as u32 - 1) as usize;
        let res = self.inner.borrow().flatten(s, e)
             .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
    
    #[napi]
    pub fn transpose(&self) -> Result<Tensor> {
        let res = self.inner.borrow().transpose()
             .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}
