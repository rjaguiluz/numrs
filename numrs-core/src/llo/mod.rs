//! Low-level operations (LLO) — target-aware operations ready for codegen
use serde::{Serialize, Deserialize};

pub mod elementwise;
pub mod reduction;
pub mod matmul;
pub mod linalg;
pub mod shape;
pub mod stats;
pub mod random;
pub mod model;

pub use elementwise::*;
pub use reduction::*;
pub use matmul::*;
pub use linalg::*;
pub use shape::*;
pub use stats::*;
pub use random::*;
pub use model::*;

/// Top-level LLO program container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLOProgram {
    pub ops: Vec<LloOp>,
}

impl LLOProgram {
    pub fn new() -> Self { Self { ops: vec![] } }
    pub fn add_op(&mut self, op: LloOp) { self.ops.push(op); }
}

/// Enumeration of LLO operations. Each variant holds a backend-ready shape and input references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LloOp {
    Elementwise { kind: ElementwiseKind, inputs: Vec<usize>, output_shape: Vec<usize>, strategy: ElementwiseStrategy },
    Reduction { axis: Option<usize>, inputs: Vec<usize>, output_shape: Vec<usize> },
    MatMul { a: usize, b: usize, output_shape: Vec<usize> },
    /// Shape-related operations (reshape, transpose, concat, etc.)
    Shape { kind: ShapeKind, inputs: Vec<usize>, output_shape: Vec<usize> },
    /// Higher-level stats ops (softmax, norm, cross-entropy, ...)
    Stats { kind: StatsKind, inputs: Vec<usize>, output_shape: Vec<usize> },
    /// Random number generation ops (rand, randn, randint, seed)
    Random { kind: RandomKind, inputs: Vec<usize>, output_shape: Vec<usize> },
    /// Model-related operations (save, load, training)
    Model { kind: ModelKind, inputs: Vec<usize>, output_shape: Vec<usize> },
    /// Training operations (forward, backward, update)
    Training { kind: TrainingKind, inputs: Vec<usize>, output_shape: Vec<usize> },
}

/// Convert Array to OnnxTensor
pub fn array_to_onnx_tensor(name: &str, array: &crate::array::Array) -> anyhow::Result<OnnxTensor> {
    // Flatten data to bytes (F32)
    let mut data_bytes = Vec::with_capacity(array.data.len() * 4);
    for &val in &array.data {
        data_bytes.extend_from_slice(&val.to_le_bytes());
    }
    
    Ok(OnnxTensor {
        name: name.to_string(),
        dtype: 1, // FLOAT
        shape: array.shape.clone(),
        data: data_bytes,
    })
}
