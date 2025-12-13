/// MatMul LLO placeholder. In the future this will include block sizes and
/// tiling information and backend-specific heuristics.

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatMul {
    pub a_rows: usize,
    pub a_cols: usize,
    pub b_cols: usize,
}
