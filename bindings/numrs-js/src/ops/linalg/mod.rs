use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

#[napi]
impl Tensor {
    #[napi]
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let res = self.inner.borrow().matmul(&*other.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}

// --- Standalone Agnostic Ops ---

use crate::NumRsArray;
use numrs::ops::linalg;

#[napi]
pub fn dot(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // dot operates on Array (1D). Detached.
    let inner_a = a.inner.borrow();
    let inner_b = b.inner.borrow();
    
    let res_arr = linalg::dot::dot(&inner_a.data, &inner_b.data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
        
    let res_tensor = numrs::Tensor::new(res_arr, false);
    Ok(Tensor { inner: Rc::new(RefCell::new(res_tensor)) })
}

#[napi]
pub fn matmul(
    a: Either<&NumRsArray, &Tensor>, 
    b: Either<&NumRsArray, &Tensor>
) -> Result<Either<NumRsArray, Tensor>> {
    match (a, b) {
        (Either::A(arr_a), Either::A(arr_b)) => {
            let res = numrs::ops::matmul(&arr_a.inner, &arr_b.inner)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(Either::A(NumRsArray { inner: res }))
        },
        (Either::B(t_a), Either::B(t_b)) => {
            let res = t_a.matmul(t_b)?;
            Ok(Either::B(res))
        },
        _ => Err(Error::from_reason("matmul: Operands must be both Array or both Tensor"))
    }
}
