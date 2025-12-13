use crate::tensor::NumRsTensor;
use numrs::Tensor;

#[no_mangle]
pub unsafe extern "C" fn numrs_softmax(a: *mut NumRsTensor, axis: isize) -> *mut NumRsTensor {
    // Handle axis if supported by core, else assume -1 default
    let dim = if axis < 0 { 1 } else { axis as usize }; // Default to last dim usually? Or 1?
    let res = numrs::ops::stats::softmax::softmax(&(*a).inner.data, Some(dim)).unwrap(); // This returns Array
                                                                                         // Softmax op usually returns Array, so we wrap in Tensor?
                                                                                         // Autograd support for softmax might be via `activations` or raw op.
                                                                                         // If Tensor has .softmax(), use it. Logic in JS bindings uses `numrs::ops::stats::softmax`.
                                                                                         // Let's assume we want just the forward pass for now if Tensor doesn't support it directly.
                                                                                         // BUT, `tensor.rs` in JS bindings implements methods that use Autograd compatible ops.
                                                                                         // If core doesn't have autograd softmax, we return detached tensor.

    // JS Binding: global `softmax` uses `numrs::ops::stats::softmax`.
    // It returns Array, so it returns "any" (NumRsArray or float32array in JS).
    // If we want Tensor return, we just wrap in Tensor(..., requires_grad=false).

    let arr = (*a).inner.data.clone(); // Get array
    let out = numrs::ops::stats::softmax::softmax(&arr, Some(dim)).expect("Softmax failed");
    Box::into_raw(Box::new(NumRsTensor {
        inner: Tensor::new(out, false),
    }))
}
#[no_mangle]
pub unsafe extern "C" fn numrs_norm(a: *mut NumRsTensor) -> *mut NumRsTensor {
    // Core `ops::stats::norm` computes global L2 norm.
    let res = numrs::ops::stats::norm::norm(&(*a).inner.data).expect("norm failed");
    Box::into_raw(Box::new(NumRsTensor {
        inner: Tensor::new(res, false),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn numrs_mse_loss(
    a: *mut NumRsTensor,
    b: *mut NumRsTensor,
) -> *mut NumRsTensor {
    let t = (*a).inner.mse_loss(&(*b).inner).expect("mse_loss failed");
    Box::into_raw(Box::new(NumRsTensor { inner: t }))
}
