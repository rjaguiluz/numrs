use crate::tensor::NumRsTensor;
use numrs::Tensor;
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn numrs_reshape(a: *mut NumRsTensor, shape_ptr: *const u32, ndim: usize) -> *mut NumRsTensor {
    let shape_slice = slice::from_raw_parts(shape_ptr, ndim);
    let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    
    match (*a).inner.reshape(shape) {
        Ok(res) => Box::into_raw(Box::new(NumRsTensor { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_flatten(a: *mut NumRsTensor, start_dim: usize, end_dim: usize) -> *mut NumRsTensor {
    let res = (*a).inner.flatten(start_dim, end_dim).expect("flatten failed");
    Box::into_raw(Box::new(NumRsTensor { inner: res }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_transpose(a: *mut NumRsTensor) -> *mut NumRsTensor {
    // Tensor.transpose() exists in core and returns Result<Tensor>
    match (*a).inner.transpose() {
        Ok(res) => Box::into_raw(Box::new(NumRsTensor { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_concat(tensors: *const *mut NumRsTensor, len: usize, axis: usize) -> *mut NumRsTensor {
    let tensor_ptrs = slice::from_raw_parts(tensors, len);
    let mut arrays = Vec::new();
    for &ptr in tensor_ptrs {
        arrays.push(&(*ptr).inner.data);
    }
    
    // Detached op
    let res_arr = numrs::ops::shape::concat(&arrays, axis).expect("concat failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_broadcast_to(a: *mut NumRsTensor, shape_ptr: *const u32, ndim: usize) -> *mut NumRsTensor {
    let shape_slice = slice::from_raw_parts(shape_ptr, ndim);
    let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    
    let res_arr = numrs::ops::shape::broadcast_to(&(*a).inner.data, &shape).expect("broadcast_to failed");
    Box::into_raw(Box::new(NumRsTensor { inner: Tensor::new(res_arr, false) }))
}

