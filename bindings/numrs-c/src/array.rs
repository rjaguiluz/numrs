use numrs::Array;
use std::ffi::{c_void, c_float};
use std::slice;

/// Opaque handle for C
pub struct NumRsArray {
    pub(crate) inner: Array,
}

/// Create a new Array from a raw float buffer
/// data: pointer to float array
/// shape: pointer to shape dimensions (uint32_t)
/// ndim: number of dimensions
#[no_mangle]
pub unsafe extern "C" fn numrs_array_new(
    data: *const c_float,
    shape: *const u32,
    ndim: usize,
) -> *mut NumRsArray {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let shape_vec: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let num_elements: usize = shape_vec.iter().product();

    let data_slice = slice::from_raw_parts(data, num_elements);
    
    // Copy data into Rust-owned Vec
    let array = Array::new(shape_vec, data_slice.to_vec());
    
    Box::into_raw(Box::new(NumRsArray { inner: array }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_array_free(ptr: *mut NumRsArray) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

// Helpers for debug
#[no_mangle]
pub unsafe extern "C" fn numrs_array_print(ptr: *const NumRsArray) {
    if !ptr.is_null() {
        let arr = &(*ptr).inner;
        println!("{:?}", arr);
    }
}

/// Create an array of zeros
#[no_mangle]
pub unsafe extern "C" fn numrs_array_zeros(
    shape: *const u32,
    ndim: usize,
) -> *mut NumRsArray {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = slice::from_raw_parts(shape, ndim);
    let shape_vec: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    
    let array = Array::zeros(shape_vec);
    Box::into_raw(Box::new(NumRsArray { inner: array }))
}

/// Create an array of ones
#[no_mangle]
pub unsafe extern "C" fn numrs_array_ones(
    shape: *const u32,
    ndim: usize,
) -> *mut NumRsArray {
    if shape.is_null() {
        return std::ptr::null_mut();
    }
    let shape_slice = slice::from_raw_parts(shape, ndim);
    let shape_vec: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    
    let array = Array::ones(shape_vec);
    Box::into_raw(Box::new(NumRsArray { inner: array }))
}

/// Get array shape
/// If out_shape is null, returns pointer to internal shape data (unsafe if array freed)
/// Ideally we copy to user buffer. For simplicity/safety let's return a pointer 
/// but caller must copy if they want it persistence. 
/// Or better: allow caller to pass buffer.
/// C-style: passing buffer is safer. But Python ctypes likes returning pointers.
/// Let's stick to returning internal pointer to `Vec` data? `Vec` owns data.
/// `Array` stores shape as `Vec<usize>`. We need `Vec<u32>` for C compat usually
/// or just expose `usize`. Python ctypes handles `c_size_t` (usize).
/// Let's assume usize for shape dimensions in C API to match Rust.
/// But previous `numrs_array_new` used `u32`. Let's stick to `u32` for consistency?
/// The `numrs_array_new` takes `*const u32`. 
/// So we need to convert. We can't return pointer to internal `Vec<usize>`.
/// We have to either allocate a new buffer or write to user buffer.
/// Strategy: Write to user-provided buffer.
#[no_mangle]
pub unsafe extern "C" fn numrs_array_shape(
    ptr: *const NumRsArray,
    out_shape: *mut u32,
    out_ndim: *mut usize,
) {
    if ptr.is_null() { return; }
    let arr = &(*ptr).inner;
    let shape = arr.shape();
    
    if !out_ndim.is_null() {
        *out_ndim = shape.len();
    }
    
    if !out_shape.is_null() {
        let out_slice = slice::from_raw_parts_mut(out_shape, shape.len());
        for (i, &dim) in shape.iter().enumerate() {
            out_slice[i] = dim as u32;
        }
    }
}

/// Get number of dimensions
#[no_mangle]
pub unsafe extern "C" fn numrs_array_ndim(ptr: *const NumRsArray) -> usize {
    if ptr.is_null() { return 0; }
    (*ptr).inner.shape().len()
}

/// Get raw data pointer (f32)
#[no_mangle]
pub unsafe extern "C" fn numrs_array_data(ptr: *mut NumRsArray) -> *mut c_float {
    if ptr.is_null() { return std::ptr::null_mut(); }
    // We need mutable access to get mutable pointer? Or const?
    // Array data is `Vec<f32>`. as_mut_ptr() or as_ptr().
    // Let's allow modification if we have a mutable pointer to Array.
    (*ptr).inner.data.as_mut_ptr() as *mut c_float
}

/// Reshape array
#[no_mangle]
pub unsafe extern "C" fn numrs_array_reshape(
    ptr: *const NumRsArray,
    new_shape: *const u32,
    ndim: usize,
) -> *mut NumRsArray {
    if ptr.is_null() || new_shape.is_null() { return std::ptr::null_mut(); }
    
    let shape_slice = slice::from_raw_parts(new_shape, ndim);
    let shape_isize: Vec<isize> = shape_slice.iter().map(|&x| x as isize).collect();
    
    // Core reshape is in ops::shape::reshape
    match numrs::ops::shape::reshape(&(*ptr).inner, &shape_isize) {
        Ok(res) => Box::into_raw(Box::new(NumRsArray { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

// --- Math Operations ---

/// Elementwise addition
#[no_mangle]
pub unsafe extern "C" fn numrs_array_add(
    a: *const NumRsArray,
    b: *const NumRsArray,
) -> *mut NumRsArray {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    match numrs::ops::add(&(*a).inner, &(*b).inner) {
        Ok(res) => Box::into_raw(Box::new(NumRsArray { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Elementwise subtraction
#[no_mangle]
pub unsafe extern "C" fn numrs_array_sub(
    a: *const NumRsArray,
    b: *const NumRsArray,
) -> *mut NumRsArray {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    match numrs::ops::sub(&(*a).inner, &(*b).inner) {
        Ok(res) => Box::into_raw(Box::new(NumRsArray { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Elementwise multiplication
#[no_mangle]
pub unsafe extern "C" fn numrs_array_mul(
    a: *const NumRsArray,
    b: *const NumRsArray,
) -> *mut NumRsArray {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    match numrs::ops::mul(&(*a).inner, &(*b).inner) {
        Ok(res) => Box::into_raw(Box::new(NumRsArray { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Elementwise division
#[no_mangle]
pub unsafe extern "C" fn numrs_array_div(
    a: *const NumRsArray,
    b: *const NumRsArray,
) -> *mut NumRsArray {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    match numrs::ops::div(&(*a).inner, &(*b).inner) {
        Ok(res) => Box::into_raw(Box::new(NumRsArray { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Matrix multiplication
#[no_mangle]
pub unsafe extern "C" fn numrs_array_matmul(
    a: *const NumRsArray,
    b: *const NumRsArray,
) -> *mut NumRsArray {
    if a.is_null() || b.is_null() { return std::ptr::null_mut(); }
    match numrs::ops::matmul(&(*a).inner, &(*b).inner) {
        Ok(res) => Box::into_raw(Box::new(NumRsArray { inner: res })),
        Err(_) => std::ptr::null_mut(),
    }
}

// --- Casting / From Vec ---

/// Create Array from f64 data (converts to internal f32)
#[no_mangle]
pub unsafe extern "C" fn numrs_array_from_f64(
    data: *const f64,
    shape: *const u32,
    ndim: usize,
) -> *mut NumRsArray {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let shape_vec: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let num_elements: usize = shape_vec.iter().product();

    let data_slice = slice::from_raw_parts(data, num_elements);
    
    // Core's from_vec handles conversion. 
    // We specify Array::<f32>::from_vec which takes generic input and converts to f32.
    let array = Array::<f32>::from_vec(shape_vec, data_slice.to_vec());
    
    Box::into_raw(Box::new(NumRsArray { inner: array }))
}

/// Create Array from i32 data (converts to internal f32)
#[no_mangle]
pub unsafe extern "C" fn numrs_array_from_i32(
    data: *const i32,
    shape: *const u32,
    ndim: usize,
) -> *mut NumRsArray {
    if data.is_null() || shape.is_null() {
        return std::ptr::null_mut();
    }

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let shape_vec: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let num_elements: usize = shape_vec.iter().product();

    let data_slice = slice::from_raw_parts(data, num_elements);
    
    let array = Array::<f32>::from_vec(shape_vec, data_slice.to_vec());
    
    Box::into_raw(Box::new(NumRsArray { inner: array }))
}
