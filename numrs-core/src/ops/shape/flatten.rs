
use crate::array::Array;
use anyhow::{Result, anyhow};

/// Flattens a contiguous range of dims into a tensor.
/// 
/// For use with batches, typically `start_dim` = 1.
/// 
/// # Arguments
/// * `input` - Input tensor
/// * `start_dim` - First dim to flatten (0-indexed)
/// * `end_dim` - Last dim to flatten (inclusive). Use -1 (or huge number) for "until end".
pub fn flatten(input: &Array, start_dim: usize, end_dim: usize) -> Result<Array> {
    let ndim = input.shape.len();
    
    // Clamp end_dim
    let end_dim = if end_dim >= ndim { ndim - 1 } else { end_dim };
    
    if start_dim > end_dim {
        return Err(anyhow!("flatten: start_dim ({}) cannot come after end_dim ({})", start_dim, end_dim));
    }
    
    let mut new_shape = Vec::new();
    
    // 1. Dims before start_dim
    for i in 0..start_dim {
        new_shape.push(input.shape[i]);
    }
    
    // 2. Flattened dim
    let mut flat_size = 1;
    for i in start_dim..=end_dim {
        flat_size *= input.shape[i];
    }
    new_shape.push(flat_size);
    
    // 3. Dims after end_dim
    for i in (end_dim + 1)..ndim {
        new_shape.push(input.shape[i]);
    }
    
    // Delegate to reshape logic (requires isize shape)
    let new_shape_isize: Vec<isize> = new_shape.iter().map(|&x| x as isize).collect();
    crate::ops::reshape(input, &new_shape_isize)
}
