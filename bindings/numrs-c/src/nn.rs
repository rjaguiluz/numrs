use crate::tensor::NumRsTensor;
use numrs::autograd::{Module, Sequential, Linear, ReLU, Conv1d, Sigmoid, Softmax, Dropout, Flatten, BatchNorm1d};
use numrs::Tensor;
use std::rc::Rc;
use std::cell::RefCell;

// Opaque handles
pub struct NumRsSequential {
    pub(crate) inner: Rc<RefCell<Sequential>>,
}

pub struct NumRsLinear {
    pub(crate) inner: Rc<RefCell<Linear>>,
}

pub struct NumRsReLU {
    pub(crate) inner: Rc<RefCell<ReLU>>,
}

pub struct NumRsConv1d {
    pub(crate) inner: Rc<RefCell<Conv1d>>,
}

pub struct NumRsSigmoid {
    pub(crate) inner: Rc<RefCell<Sigmoid>>,
}

pub struct NumRsSoftmax {
    pub(crate) inner: Rc<RefCell<Softmax>>,
}

pub struct NumRsDropout {
    pub(crate) inner: Rc<RefCell<Dropout>>,
}

pub struct NumRsFlatten {
    pub(crate) inner: Rc<RefCell<Flatten>>,
}

pub struct NumRsBatchNorm1d {
    pub(crate) inner: Rc<RefCell<BatchNorm1d>>,
}

// --- Sequential ---
#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_new() -> *mut NumRsSequential {
    // Sequential::new expects Vec<Box<dyn Module>>
    let seq = Sequential::new(vec![]);
    Box::into_raw(Box::new(NumRsSequential { inner: Rc::new(RefCell::new(seq)) }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_free(ptr: *mut NumRsSequential) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_add_linear(seq: *mut NumRsSequential, layer: *mut NumRsLinear) {
    if !seq.is_null() && !layer.is_null() {
        // Must clone the inner Linear, NOT the Rc
        let linear_obj = (*layer).inner.borrow().clone();
        (*seq).inner.borrow_mut().add(Box::new(linear_obj));
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_add_relu(seq: *mut NumRsSequential, layer: *mut NumRsReLU) {
    if !seq.is_null() && !layer.is_null() {
        // ReLU is unit struct but might implement Clone. 
        // CoreReLU definition: pub struct ReLU; (usually derives Clone, Copy)
        // If it holds state, we need deep clone.
        // Assuming Clone.
        let relu_obj = (*layer).inner.borrow().clone();
        (*seq).inner.borrow_mut().add(Box::new(relu_obj));
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_forward(seq: *mut NumRsSequential, input: *mut NumRsTensor) -> *mut NumRsTensor {
    if seq.is_null() || input.is_null() { return std::ptr::null_mut(); }
    
    // forward expects &Tensor.
    let input_tensor = &(*input).inner;
    let res = (*seq).inner.borrow().forward(input_tensor);
    match res {
        Ok(t) => Box::into_raw(Box::new(NumRsTensor { inner: t })),
        Err(_) => std::ptr::null_mut(), // TODO: Error handling
    }
}

// --- Linear ---
#[no_mangle]
pub unsafe extern "C" fn numrs_linear_new(in_features: usize, out_features: usize) -> *mut NumRsLinear {
    // Linear::new returns Result
    match Linear::new(in_features, out_features) {
        Ok(linear) => Box::into_raw(Box::new(NumRsLinear { inner: Rc::new(RefCell::new(linear)) })),
        Err(_) => std::ptr::null_mut(),
    }
}

// --- ReLU ---
#[no_mangle]
pub unsafe extern "C" fn numrs_relu_layer_new() -> *mut NumRsReLU {
    // ReLU::new() doesn't exist? Check JS: ReLU { inner: CoreReLU } -> struct Literal.
    // Core ReLU is unit struct?
    // In JS bindings: CoreReLU (unit struct).
    // In Core: pub struct ReLU.
    // Let's rely on default? Or struct literal.
    // Error said "no function named new".
    // So: let relu = ReLU; 
    let relu = ReLU;
    Box::into_raw(Box::new(NumRsReLU { inner: Rc::new(RefCell::new(relu)) }))
}

// --- Conv1d ---
#[no_mangle]
pub unsafe extern "C" fn numrs_conv1d_new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> *mut NumRsConv1d {
    match Conv1d::new(in_channels, out_channels, kernel_size, stride, padding) {
        Ok(conv) => Box::into_raw(Box::new(NumRsConv1d { inner: Rc::new(RefCell::new(conv)) })),
        Err(_) => std::ptr::null_mut(),
    }
}

// --- Sigmoid ---
#[no_mangle]
pub unsafe extern "C" fn numrs_sigmoid_new() -> *mut NumRsSigmoid {
    let sigmoid = Sigmoid;
    Box::into_raw(Box::new(NumRsSigmoid { inner: Rc::new(RefCell::new(sigmoid)) }))
}

// --- Softmax ---
#[no_mangle]
pub unsafe extern "C" fn numrs_softmax_new() -> *mut NumRsSoftmax {
    let softmax = Softmax;
    Box::into_raw(Box::new(NumRsSoftmax { inner: Rc::new(RefCell::new(softmax)) }))
}

// --- Dropout ---
#[no_mangle]
pub unsafe extern "C" fn numrs_dropout_new(p: f32) -> *mut NumRsDropout {
    let dropout = Dropout::new(p);
    Box::into_raw(Box::new(NumRsDropout { inner: Rc::new(RefCell::new(dropout)) }))
}

// --- Flatten ---
#[no_mangle]
pub unsafe extern "C" fn numrs_flatten_new(start_dim: usize, end_dim: usize) -> *mut NumRsFlatten {
    let flatten = Flatten::new(start_dim, end_dim);
    Box::into_raw(Box::new(NumRsFlatten { inner: Rc::new(RefCell::new(flatten)) }))
}

// --- BatchNorm1d ---
#[no_mangle]
pub unsafe extern "C" fn numrs_batchnorm1d_new(num_features: usize) -> *mut NumRsBatchNorm1d {
    match BatchNorm1d::new(num_features) {
        Ok(bn) => Box::into_raw(Box::new(NumRsBatchNorm1d { inner: Rc::new(RefCell::new(bn)) })),
        Err(_) => std::ptr::null_mut(),
    }
}

// --- Sequential Adders ---
#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_add_conv1d(seq: *mut NumRsSequential, layer: *mut NumRsConv1d) {
    if !seq.is_null() && !layer.is_null() {
        let obj = (*layer).inner.borrow().clone();
        (*seq).inner.borrow_mut().add(Box::new(obj));
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_add_sigmoid(seq: *mut NumRsSequential, layer: *mut NumRsSigmoid) {
    if !seq.is_null() && !layer.is_null() {
        let obj = (*layer).inner.borrow().clone();
        (*seq).inner.borrow_mut().add(Box::new(obj));
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_add_softmax(seq: *mut NumRsSequential, layer: *mut NumRsSoftmax) {
    if !seq.is_null() && !layer.is_null() {
        let obj = (*layer).inner.borrow().clone();
        (*seq).inner.borrow_mut().add(Box::new(obj));
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_add_dropout(seq: *mut NumRsSequential, layer: *mut NumRsDropout) {
    if !seq.is_null() && !layer.is_null() {
        let obj = (*layer).inner.borrow().clone();
        (*seq).inner.borrow_mut().add(Box::new(obj));
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_add_flatten(seq: *mut NumRsSequential, layer: *mut NumRsFlatten) {
    if !seq.is_null() && !layer.is_null() {
        let obj = (*layer).inner.borrow().clone();
        (*seq).inner.borrow_mut().add(Box::new(obj));
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_sequential_add_batchnorm1d(seq: *mut NumRsSequential, layer: *mut NumRsBatchNorm1d) {
    if !seq.is_null() && !layer.is_null() {
        let obj = (*layer).inner.borrow().clone();
        (*seq).inner.borrow_mut().add(Box::new(obj));
    }
}
