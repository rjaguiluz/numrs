use serde::{Serialize, Deserialize};

/// Stats-related LLO kinds (softmax, norm, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StatsKind {
    Norm,
    Softmax,
    LogSoftmax,
    CrossEntropy,
}

impl Default for StatsKind {
    fn default() -> Self { StatsKind::Softmax }
}

// Future: include reduction/composition details (e.g., axes, epsilon) as needed
