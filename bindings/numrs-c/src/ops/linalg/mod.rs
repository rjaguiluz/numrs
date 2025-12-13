use crate::tensor::NumRsTensor;
use numrs::Tensor;

#[no_mangle]
pub unsafe extern "C" fn numrs_matmul(a: *mut NumRsTensor, b: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.matmul(&(*b).inner).expect("matmul failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}
#[no_mangle]
pub unsafe extern "C" fn numrs_dot(a: *mut NumRsTensor, b: *mut NumRsTensor) -> *mut NumRsTensor {
    // Detached op
    let res_arr = numrs::ops::linalg::dot(&(*a).inner.data, &(*b).inner.data).expect("dot failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}
