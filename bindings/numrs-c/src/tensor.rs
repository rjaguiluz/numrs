use crate::array::NumRsArray;
use numrs::Tensor;
use std::ffi::c_void;

pub struct NumRsTensor {
    pub(crate) inner: Tensor,
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_new(
    array: *mut NumRsArray,
    requires_grad: bool,
) -> *mut NumRsTensor {
    if array.is_null() {
        return std::ptr::null_mut();
    }
    // Clone array because Tensor takes ownership, but here we might want to keep NumRsArray valid?
    // In numrs-js, Tensor::new takes NumRsArray.
    // If we want to move, we should invalidate array ptr. But C usage pattern usually implies "use available data".
    // Let's clone for safety or assume ownership transfer?
    // NumRs core Tensor::new takes Array.
    let arr = &(*array).inner;
    let tensor = Tensor::new(arr.clone(), requires_grad);

    Box::into_raw(Box::new(NumRsTensor { inner: tensor }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_free(ptr: *mut NumRsTensor) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_backward(ptr: *mut NumRsTensor) {
    if !ptr.is_null() {
        (*ptr).inner.backward();
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_data(ptr: *mut NumRsTensor) -> *mut NumRsArray {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    // Return a new NumRsArray wrapping the tensor's data (Array)
    let arr = (*ptr).inner.data.clone();
    Box::into_raw(Box::new(NumRsArray { inner: arr }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_grad(ptr: *mut NumRsTensor) -> *mut NumRsTensor {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    // Tensor.grad is Option<Rc<RefCell<Array>>>
    // We want to return a Tensor (detached gradient)
    match &(*ptr).inner.gradient() {
        Some(grad_array) => {
            // Create a new Tensor from the gradient array (no grad required for gradient itself)
            let g = Tensor::new(grad_array.clone(), false);
            Box::into_raw(Box::new(NumRsTensor { inner: g }))
        }
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_shape(
    ptr: *mut NumRsTensor,
    out_ndim: *mut usize,
) -> *const usize {
    if ptr.is_null() {
        if !out_ndim.is_null() {
            *out_ndim = 0;
        }
        return std::ptr::null();
    }
    let shape = (*ptr).inner.shape(); // returns slice &[usize]
    if !out_ndim.is_null() {
        *out_ndim = shape.len();
    }
    shape.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_item(ptr: *mut NumRsTensor) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    (*ptr).inner.item()
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_zero_grad(ptr: *mut NumRsTensor) {
    if !ptr.is_null() {
        (*ptr).inner.zero_grad();
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_detach(ptr: *mut NumRsTensor) -> *mut NumRsTensor {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let detached = (*ptr).inner.detach();
    Box::into_raw(Box::new(NumRsTensor { inner: detached }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_reshape(
    ptr: *mut NumRsTensor,
    shape_ptr: *const usize,
    ndim: usize,
) -> *mut NumRsTensor {
    if ptr.is_null() || (ndim > 0 && shape_ptr.is_null()) {
        return std::ptr::null_mut();
    }

    let shape_vec = if ndim > 0 {
        std::slice::from_raw_parts(shape_ptr, ndim).to_vec()
    } else {
        vec![]
    };

    match (*ptr).inner.reshape(shape_vec) {
        Ok(res) => Box::into_raw(Box::new(NumRsTensor { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_tensor_flatten(ptr: *mut NumRsTensor) -> *mut NumRsTensor {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    let numel: usize = (*ptr).inner.shape().iter().product();
    match (*ptr).inner.reshape(vec![numel]) {
        Ok(res) => Box::into_raw(Box::new(NumRsTensor { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}
