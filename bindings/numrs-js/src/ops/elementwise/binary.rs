use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

#[napi]
impl Tensor {
    #[napi]
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let res = self.inner.borrow().add(&*other.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let res = self.inner.borrow().sub(&*other.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let res = self.inner.borrow().mul(&*other.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let res = self.inner.borrow().div(&*other.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}

// --- Standalone Agnostic Ops ---

use crate::NumRsArray;

#[napi]
pub fn add(
    a: Either<&NumRsArray, &Tensor>, 
    b: Either<&NumRsArray, &Tensor>
) -> Result<Either<NumRsArray, Tensor>> {
    match (a, b) {
        (Either::A(arr_a), Either::A(arr_b)) => {
            let res = numrs::ops::add(&arr_a.inner, &arr_b.inner)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(Either::A(NumRsArray { inner: res }))
        },
        (Either::B(t_a), Either::B(t_b)) => {
            let res = t_a.add(t_b)?;
            Ok(Either::B(res))
        },
        _ => Err(Error::from_reason("add: Operands must be both Array or both Tensor"))
    }
}

#[napi]
pub fn sub(
    a: Either<&NumRsArray, &Tensor>, 
    b: Either<&NumRsArray, &Tensor>
) -> Result<Either<NumRsArray, Tensor>> {
    match (a, b) {
        (Either::A(arr_a), Either::A(arr_b)) => {
            let res = numrs::ops::sub(&arr_a.inner, &arr_b.inner)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(Either::A(NumRsArray { inner: res }))
        },
        (Either::B(t_a), Either::B(t_b)) => {
            let res = t_a.sub(t_b)?;
            Ok(Either::B(res))
        },
        _ => Err(Error::from_reason("sub: Operands must be both Array or both Tensor"))
    }
}

#[napi]
pub fn mul(
    a: Either<&NumRsArray, &Tensor>, 
    b: Either<&NumRsArray, &Tensor>
) -> Result<Either<NumRsArray, Tensor>> {
    match (a, b) {
        (Either::A(arr_a), Either::A(arr_b)) => {
            let res = numrs::ops::mul(&arr_a.inner, &arr_b.inner)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(Either::A(NumRsArray { inner: res }))
        },
        (Either::B(t_a), Either::B(t_b)) => {
            let res = t_a.mul(t_b)?;
            Ok(Either::B(res))
        },
        _ => Err(Error::from_reason("mul: Operands must be both Array or both Tensor"))
    }
}

#[napi]
pub fn div(
    a: Either<&NumRsArray, &Tensor>, 
    b: Either<&NumRsArray, &Tensor>
) -> Result<Either<NumRsArray, Tensor>> {
    match (a, b) {
        (Either::A(arr_a), Either::A(arr_b)) => {
            let res = numrs::ops::div(&arr_a.inner, &arr_b.inner)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            Ok(Either::A(NumRsArray { inner: res }))
        },
        (Either::B(t_a), Either::B(t_b)) => {
            let res = t_a.div(t_b)?;
            Ok(Either::B(res))
        },
        _ => Err(Error::from_reason("div: Operands must be both Array or both Tensor"))
    }
}

