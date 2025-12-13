//! # NumRs Operations API
//! 
//! Este módulo proporciona operaciones de array con dispatch automático al mejor backend.
//!
//! ## Uso directo (RECOMENDADO)
//! ```no_run
//! use numrs::ops;
//! # use numrs::Array;
//! # let a = Array::new(vec![2], vec![1.0, 2.0]);
//! # let b = Array::new(vec![2], vec![3.0, 4.0]);
//! # let arr = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
//! let result = ops::add(&a, &b)?;        // Dispatch automático (SIMD/BLAS/GPU)
//! let sum = ops::sum(&arr, None)?;       // Zero-overhead kernel call
//! let c = ops::matmul(&a, &b)?;          // Usa MKL/OpenBLAS/WebGPU según disponibilidad
//! # Ok::<(), anyhow::Error>(())
//! ```
//! - ✅ Zero overhead (inline + function pointers)
//! - ✅ Validación funcional de backends
//! - ✅ Selección automática del mejor kernel
//! - ✅ Perfecto para wrappers JS/Python
//!
//! ## Organización modular
//! Las operaciones están organizadas por categoría:
//! - `ops::elementwise::binary` - Operaciones binarias elemento por elemento (add, mul, etc.)
//! - `ops::elementwise::unary` - Operaciones unarias (sin, cos, sqrt, relu, etc.)
//! - `ops::reduction` - Reducciones (sum, mean, etc.)
//! - `ops::linalg` - Álgebra lineal (matmul, dot, etc.)
//!
//! Todas las operaciones usan el sistema de dispatch y son zero-cost.

mod promotion_wrappers;

pub mod elementwise {
	pub mod binary {
		pub mod add;
		pub mod mul;
		pub mod div;
		pub mod sub;
		pub mod pow;
	}
	pub mod unary;
}

pub mod linalg;
pub mod reduction;
pub mod shape;
pub mod model;
pub mod export; // Automatic ONNX export
pub mod stats;
pub mod conv;
pub mod batchnorm;
pub mod dropout;

// Re-export all operations at top level for convenient access
pub use elementwise::binary::add::add;
pub use elementwise::binary::mul::mul;
pub use elementwise::binary::div::div;
pub use elementwise::binary::sub::sub;
pub use elementwise::binary::pow::pow;

pub use elementwise::unary::{
    sqrt, sin, cos, tan, abs, exp, log, asin, acos, atan, relu, leaky_relu,
    sigmoid, tanh, softplus, neg
};

pub use reduction::{sum, max, min, mean, variance, argmax};
pub use linalg::{matmul, dot};
pub use shape::{reshape, transpose, concat, broadcast_to, flatten};
pub use stats::{norm, softmax, cross_entropy};

// Model operations for ONNX compatibility
pub use model::{
    save_onnx, load_onnx, save_checkpoint, load_checkpoint,
    create_mlp, create_linear_node, create_relu_node, create_softmax_node,
    create_matmul_node, create_add_node, array_to_onnx_tensor, infer
};

// Compatibility alias - todas las operaciones están disponibles directamente en ops::*
// Para compatibilidad con código legacy que usa ops::fast::*, re-exportamos todo aquí también
pub mod fast {
    //! Alias de compatibilidad para código existente que usa ops::fast::*
    //! 
    //! Todas estas funciones son idénticas a las exportadas en ops::* directamente.
    //! El namespace "fast" ya no es necesario ya que todas las operaciones usan dispatch.
    pub use super::*;
}


