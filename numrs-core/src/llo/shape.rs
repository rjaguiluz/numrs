use serde::{Serialize, Deserialize};

/// Shape-related kinds for LLO (reshape, transpose, concat, slice, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShapeKind {
    Reshape,
    Transpose,
    Permute,
    Flatten,
    Squeeze,
    Unsqueeze,
    BroadcastTo,
    ExpandDims,
    Concat,
    Stack,
    Tile,
    Repeat,
    Slice,
    Take,
    Gather,
}

impl Default for ShapeKind {
    fn default() -> Self { ShapeKind::Reshape }
}

// Future: Add small helpers or shape metadata structs if needed for codegen
