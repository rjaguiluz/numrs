/// Low-level reduction op representation

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reduction {
    pub axis: Option<usize>,
}

/// Kind of reduction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionKind {
    Sum,
    Max,
    Min,
    Mean,
    ArgMax,
    Variance,
}

/// Selected reduction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionStrategy {
    ScalarLoop,
    Parallel,
    GpuKernel,
}
