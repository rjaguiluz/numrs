use serde::{Deserialize, Serialize};

/// High-level ops for the IR. Keep them small and expressive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HloOp {
    Add,
    Mul,
    /// Elementwise subtraction
    Sub,
    /// Elementwise division
    Div,
    /// Elementwise power (binary)
    Pow,
    /// Unary sqrt
    Sqrt,
    /// Unary absolute value
    Abs,
    /// Unary exponent (e^x)
    Exp,
    /// Trigonometric ops
    Sin,
    Cos,
    Tan,
    /// Inverse trigonometric ops
    Asin,
    Acos,
    Atan,
    /// Activation functions
    Relu,
    LeakyRelu,
    /// Matrix multiplication
    MatMul,
    Sum { axis: Option<usize> },
    /// Constant / input node — represented as an HLO constant value
    Const,
    // future: Transpose, Reshape, etc
}

/// A node in the HLO graph. This is a single op with references to inputs.
#[derive(Debug, Clone)]
pub struct HloNode {
    pub id: usize,
    pub op: HloOp,
    pub inputs: Vec<usize>,
    pub shape: Vec<usize>,
}

impl HloNode {
    pub fn new(id: usize, op: HloOp, inputs: Vec<usize>, shape: Vec<usize>) -> Self {
        Self { id, op, inputs, shape }
    }

    pub fn const_node(shape: Vec<usize>) -> Self {
        Self::new(0, HloOp::Const, vec![], shape)
    }
}
