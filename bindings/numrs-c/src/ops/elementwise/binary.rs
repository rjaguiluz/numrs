use crate::tensor::NumRsTensor;
use numrs::Tensor;

// Use explicit method calls instead of operators, and unwrap Result
#[no_mangle]
pub unsafe extern "C" fn numrs_add(a: *mut NumRsTensor, b: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.add(&(*b).inner).expect("add failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sub(a: *mut NumRsTensor, b: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.sub(&(*b).inner).expect("sub failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_mul(a: *mut NumRsTensor, b: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.mul(&(*b).inner).expect("mul failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_div(a: *mut NumRsTensor, b: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.div(&(*b).inner).expect("div failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}
