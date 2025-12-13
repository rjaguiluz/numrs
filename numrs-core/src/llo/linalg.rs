/// Low-level linear algebra operation representation

use serde::{Serialize, Deserialize};

/// Kind of linear algebra operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinalgKind {
    /// Dot product of two 1-D arrays: sum(a * b)
    Dot,
}

/// Selected linalg strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinalgStrategy {
    ScalarLoop,
    Vectorized,
    Blas,
}
