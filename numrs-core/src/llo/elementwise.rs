/// Elementwise LLO operations and strategy hints
/// This module describes elementwise ops (add, mul, ...) and possible
/// execution strategies (vectorized, simple loop, GPU kernel, etc.)

use serde::{Serialize, Deserialize};

/// Binary elementwise kinds (add, mul, sub, div, pow, ...)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementwiseBinaryKind {
    Add,
    Mul,
    Sub,
    Div,
    Pow,
}

/// Unary elementwise kinds (sqrt, sin, cos, exp, log, ...)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementwiseUnaryKind {
    Sqrt,
    Abs,
    Neg,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Relu,
    LeakyRelu,
}

/// Flattened ElementwiseKind kept for backward compatibility; prefer
/// using `ElementwiseBinaryKind` / `ElementwiseUnaryKind` when possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementwiseKind {
    Add,
    Mul,
    Sub,
    Div,
    Pow,
    /// Unary sqrt
    Sqrt,
    /// Trigonometric
    Sin,
    Cos,
    Tan,
    /// Inverse trigonometric
    Asin,
    Acos,
    Atan,
    /// Unary absolute value
    Abs,
    /// Unary negation
    Neg,
    /// Exponential
    Exp,
    /// Logarithm
    Log,
    /// Activation functions
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    Softplus,
}

/// Execution strategy selection enum. Backends may choose different variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementwiseStrategy {
    /// Let the runtime resolver pick the best concrete implementation.
    Default,
    Scalar,
    Vectorized,
    GpuKernel,
}

// Conversion helpers
impl From<ElementwiseBinaryKind> for ElementwiseKind {
    fn from(b: ElementwiseBinaryKind) -> Self {
        match b {
            ElementwiseBinaryKind::Add => ElementwiseKind::Add,
            ElementwiseBinaryKind::Mul => ElementwiseKind::Mul,
            ElementwiseBinaryKind::Sub => ElementwiseKind::Sub,
            ElementwiseBinaryKind::Div => ElementwiseKind::Div,
            ElementwiseBinaryKind::Pow => ElementwiseKind::Pow,
        }
    }
}

impl From<ElementwiseUnaryKind> for ElementwiseKind {
    fn from(u: ElementwiseUnaryKind) -> Self {
        match u {
            ElementwiseUnaryKind::Sqrt => ElementwiseKind::Sqrt,
            ElementwiseUnaryKind::Sin => ElementwiseKind::Sin,
            ElementwiseUnaryKind::Cos => ElementwiseKind::Cos,
            ElementwiseUnaryKind::Abs => ElementwiseKind::Abs,
            ElementwiseUnaryKind::Neg => ElementwiseKind::Neg,
            ElementwiseUnaryKind::Exp => ElementwiseKind::Exp,
            ElementwiseUnaryKind::Log => ElementwiseKind::Log,
            ElementwiseUnaryKind::Tan => ElementwiseKind::Tan,
            ElementwiseUnaryKind::Asin => ElementwiseKind::Asin,
            ElementwiseUnaryKind::Acos => ElementwiseKind::Acos,
            ElementwiseUnaryKind::Atan => ElementwiseKind::Atan,
            ElementwiseUnaryKind::Relu => ElementwiseKind::Relu,
            ElementwiseUnaryKind::LeakyRelu => ElementwiseKind::LeakyRelu,
        }
    }
}
