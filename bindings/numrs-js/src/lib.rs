use napi_derive::napi;

mod array;
mod tensor;
pub mod ops;
pub mod nn;
pub mod optim;
pub mod onnx;

pub use array::NumRsArray;
pub use tensor::Tensor;
pub use ops::elementwise::binary::{add, sub, mul, div};
pub use ops::elementwise::unary::{relu, sigmoid, exp, log};
pub use ops::linalg::matmul;
pub mod train;
pub use nn::{Linear, Sequential, ReLU, Sigmoid};
pub use optim::SGD;

#[napi]
pub fn version() -> String {
    "0.0.1".to_string()
}