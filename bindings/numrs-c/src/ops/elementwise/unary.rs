use crate::tensor::NumRsTensor;
use numrs::Tensor;

#[no_mangle]
pub unsafe extern "C" fn numrs_relu(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.relu().expect("relu failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sigmoid(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.sigmoid().expect("sigmoid failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_log(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.log().expect("log failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_exp(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.exp().expect("exp failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_pow(a: *mut NumRsTensor, exponent: f64) -> *mut NumRsTensor {
    let res = (*a).inner.pow(exponent as f32).expect("pow failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sqrt(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res = (*a).inner.sqrt().expect("sqrt failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_neg(a: *mut NumRsTensor) -> *mut NumRsTensor {
    // Implement neg as self * -1 to preserve gradients
    let minus_one_arr = numrs::Array::new(vec![1], vec![-1.0]);
    let minus_one = Tensor::new(minus_one_arr, false);
    let res = (*a).inner.mul(&minus_one).expect("neg failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_abs(a: *mut NumRsTensor) -> *mut NumRsTensor {
    // Detached op (no autograd in core for abs yet)
    let res_arr = numrs::ops::elementwise::unary::abs(&((*a).inner.data)).expect("abs failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tanh(a: *mut NumRsTensor) -> *mut NumRsTensor {
    // Detached op
    let res_arr = numrs::ops::elementwise::unary::tanh(&((*a).inner.data)).expect("tanh failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sin(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res_arr = numrs::ops::elementwise::unary::sin(&((*a).inner.data)).expect("sin failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_cos(a: *mut NumRsTensor) -> *mut NumRsTensor {
    // Autograd op
    let res = (*a).inner.cos().expect("cos failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_asin(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res_arr = numrs::ops::elementwise::unary::asin(&((*a).inner.data)).expect("asin failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_acos(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res_arr = numrs::ops::elementwise::unary::acos(&((*a).inner.data)).expect("acos failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_atan(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res_arr = numrs::ops::elementwise::unary::atan(&((*a).inner.data)).expect("atan failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_softplus(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res_arr = numrs::ops::elementwise::unary::softplus(&((*a).inner.data)).expect("softplus failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_leaky_relu(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res_arr = numrs::ops::elementwise::unary::leaky_relu(&((*a).inner.data)).expect("leaky_relu failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}
#[no_mangle]
pub unsafe extern "C" fn numrs_tan(a: *mut NumRsTensor) -> *mut NumRsTensor {
    let res_arr = numrs::ops::elementwise::unary::tan(&((*a).inner.data)).expect("tan failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}
