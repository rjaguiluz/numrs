use crate::tensor::NumRsTensor;
use numrs::Tensor;

#[no_mangle]
pub unsafe extern "C" fn numrs_sum(a: *mut NumRsTensor, _axis: isize) -> *mut NumRsTensor {
    // Current Tensor implementation only supports global sum/mean
    // We ignore axis for now or could panic if axis != -1
    // TODO: Support axis in core Tensor
    let res = (*a).inner.sum().expect("sum failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_mean(a: *mut NumRsTensor, _axis: isize) -> *mut NumRsTensor {
    let res = (*a).inner.mean().expect("mean failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_max(a: *mut NumRsTensor, axis: isize) -> *mut NumRsTensor {
    let axis_opt = if axis < 0 { None } else { Some(axis as usize) };
    // Detached op
    let res_arr = numrs::ops::reduction::max(&(*a).inner.data, axis_opt).expect("max failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_min(a: *mut NumRsTensor, axis: isize) -> *mut NumRsTensor {
    let axis_opt = if axis < 0 { None } else { Some(axis as usize) };
    let res_arr = numrs::ops::reduction::min(&(*a).inner.data, axis_opt).expect("min failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_argmax(a: *mut NumRsTensor, axis: isize) -> *mut NumRsTensor {
    let axis_opt = if axis < 0 { None } else { Some(axis as usize) };
    let res_arr = numrs::ops::reduction::argmax(&(*a).inner.data, axis_opt).expect("argmax failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}
