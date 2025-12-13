use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use crate::NumRsArray;
use std::rc::Rc;
use std::cell::RefCell;
use numrs::ops::elementwise::unary;

// Bindings for unary ops
// Mixed: Some are strictly Tensor-only (for now), others are Agnostic (Either).
// We remove simple &Tensor versions if we have Agnostic versions to avoid name collision.

// --- Tensor Only (for now) ---

#[napi]
pub fn abs(input: &Tensor) -> Result<Tensor> {
    let res = unary::abs::abs(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

#[napi]
pub fn cos(input: &Tensor) -> Result<Tensor> {
    let res = unary::cos::cos(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

#[napi]
pub fn sin(input: &Tensor) -> Result<Tensor> {
    let res = unary::sin::sin(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

#[napi]
pub fn tan(input: &Tensor) -> Result<Tensor> {
    let res = unary::tan::tan(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

#[napi]
pub fn sqrt(input: &Tensor) -> Result<Tensor> {
    let res = input.inner.borrow().sqrt()
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

#[napi]
pub fn neg(input: &Tensor) -> Result<Tensor> {
    let neg_one_arr = numrs::Array::new(vec![1], vec![-1.0]);
    let neg_one = numrs::Tensor::new(neg_one_arr, false);
    let res = input.inner.borrow().mul(&neg_one)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

#[napi]
pub fn acos(input: &Tensor) -> Result<Tensor> {
    let res = unary::acos::acos(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

#[napi]
pub fn asin(input: &Tensor) -> Result<Tensor> {
    let res = unary::asin::asin(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

#[napi]
pub fn atan(input: &Tensor) -> Result<Tensor> {
    let res = unary::atan::atan(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

// Missing in core: ceil, floor, round
/*
#[napi]
pub fn ceil(input: &Tensor) -> Result<Tensor> {
    let res = unary::ceil::ceil(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

#[napi]
pub fn floor(input: &Tensor) -> Result<Tensor> {
    let res = unary::floor::floor(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

#[napi]
pub fn round(input: &Tensor) -> Result<Tensor> {
    let res = unary::round::round(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}
*/

#[napi]
pub fn square(input: &Tensor) -> Result<Tensor> {
    let res = input.inner.borrow().pow(2.0)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
}

#[napi]
pub fn tanh(input: &Tensor) -> Result<Tensor> {
    let res = unary::tanh::tanh(&input.inner.borrow().data)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res, false))) })
}

// --- Agnostic Ops (Either) ---

#[napi]
pub fn relu(a: Either<&NumRsArray, &Tensor>) -> Result<Either<NumRsArray, Tensor>> {
    match a {
        Either::A(arr) => {
             // core::ops::relu usually exists. If not, map manually.
             let out = unary::relu::relu(&arr.inner).map_err(|e| Error::from_reason(e.to_string()))?;
             Ok(Either::A(NumRsArray { inner: out }))
        },
        Either::B(t) => {
            let res = t.relu()?; // Uses Tensor::relu from tensor.rs
            Ok(Either::B(res))
        }
    }
}

#[napi]
pub fn sigmoid(a: Either<&NumRsArray, &Tensor>) -> Result<Either<NumRsArray, Tensor>> {
    match a {
        Either::A(arr) => {
             let out = unary::sigmoid::sigmoid(&arr.inner).map_err(|e| Error::from_reason(e.to_string()))?;
             Ok(Either::A(NumRsArray { inner: out }))
        },
        Either::B(t) => {
            let res = t.sigmoid()?;
            Ok(Either::B(res))
        }
    }
}

#[napi]
pub fn exp(a: Either<&NumRsArray, &Tensor>) -> Result<Either<NumRsArray, Tensor>> {
    match a {
        Either::A(arr) => {
             let out = unary::exp::exp(&arr.inner).map_err(|e| Error::from_reason(e.to_string()))?;
             Ok(Either::A(NumRsArray { inner: out }))
        },
        Either::B(t) => {
            let res = t.exp()?;
            Ok(Either::B(res))
        }
    }
}

#[napi]
pub fn log(a: Either<&NumRsArray, &Tensor>) -> Result<Either<NumRsArray, Tensor>> {
    match a {
        Either::A(arr) => {
             let out = unary::log::log(&arr.inner).map_err(|e| Error::from_reason(e.to_string()))?;
             Ok(Either::A(NumRsArray { inner: out }))
        },
        Either::B(t) => {
            let res = t.log()?;
            Ok(Either::B(res))
        }
    }
}
