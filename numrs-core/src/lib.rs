//! NumRs — core library
//!
//! This crate provides the core runtime, IR, lowering pipeline and basic
//! backends to run simple numeric kernels. It's intentionally small and
//! focused on proving the architecture end-to-end.
//!
//! Public surface will be stable: array creation + add/mul/sum

pub mod array;
pub mod array_view; // Zero-copy view for FFI
pub mod autograd;
pub mod backend;
pub mod codegen;
pub mod ir;
pub mod llo;
pub mod ops;
pub mod ops_inplace; // Zero-copy operations for FFI bindings
pub mod startup; // ← Automatic differentiation

pub use array::{cast_array, promoted_dtype, Array, DType, DTypeValue};
pub use array_view::ArrayView;
pub use autograd::{is_grad_enabled, set_grad_enabled, NoGrad, Tensor}; // ← Autograd exports
pub use autograd::{AdaGrad, Adam, Optimizer, RMSprop, SGD}; // ← Optimizer exports
pub use autograd::{
    BatchNorm1d, Conv1d, Dropout, Flatten, Linear, Module, ReLU, Sequential, Sigmoid,
}; // ← Neural network modules
pub use autograd::{CrossEntropyLoss, Dataset, MSELoss, Trainer, TrainerBuilder};
pub use backend::dispatch::{get_backend_override, set_backend_override};
pub use llo::reduction::ReductionKind;
pub use llo::ElementwiseKind;
pub use ops::{
    abs, acos, add, asin, atan, cos, div, exp, log, mul, pow, relu, sigmoid, sin, softmax, sqrt,
    sub, sum, tan, tanh,
};
pub use startup::print_startup_log; // ← Training API
                                    // Re-export the compile-time HLO macro for convenience

#[cfg(target_arch = "wasm32")]
pub use backend::webgpu::{init_webgpu_wasm, set_webgpu_available_wasm};

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::ops::{add, div, mul, sub, sum};

    #[test]
    fn test_add_mul_sum() {
        let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
        let b = Array::new(vec![3], vec![2.0, 2.0, 2.0]);

        let c = add(&a, &b).expect("add failed");
        assert_eq!(c.data, vec![3.0, 4.0, 5.0]);

        let d = mul(&a, &b).expect("mul failed");
        assert_eq!(d.data, vec![2.0, 4.0, 6.0]);

        let s = sum(&a, None).expect("sum failed");
        // a = [1,2,3] -> sum = 6
        assert_eq!(s.data, vec![6.0]);

        // Sub / Div tests
        let a2 = Array::new(vec![2], vec![1.0, 0.0]);
        let b2 = Array::new(vec![2], vec![2.0, 1.0]);
        let sub_res = sub(&b2, &a2).expect("sub failed");
        assert_eq!(sub_res.data, vec![1.0, 1.0]);
        let div_res = div(&b2, &b2).expect("div failed");
        assert_eq!(div_res.data, vec![1.0, 1.0]); // TODO: Agregar sqrt, sin, cos cuando tengamos fast-path para ops unarias
                                                  // let x = Array::new(vec![3], vec![4.0, 9.0, 16.0]);
                                                  // let sx = sqrt(&x).expect("sqrt failed");
                                                  // assert_eq!(sx.to_f32().data, vec![2.0, 3.0, 4.0]);
    }
}
