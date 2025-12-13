//! Array module public API

pub mod array;
pub mod dtype;
pub mod dyn_array;
pub mod promotion;
pub mod ops_traits;

pub use array::Array;
pub use dtype::{DType, DTypeValue};
pub use dyn_array::DynArray;
pub use promotion::{promoted_dtype, cast_array, promote_arrays, validate_binary_op};
