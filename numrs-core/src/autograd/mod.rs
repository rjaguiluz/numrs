//! Automatic differentiation system for NumRs
//!
//! Este módulo implementa:
//! 1. Compute graph con referencias a operaciones
//! 2. Backward pass automático
//! 3. Gradient accumulation
//!
//! Arquitectura:
//! ```text
//! Tensor (Array + autograd metadata)
//!   ↓
//! ComputeNode (operación + inputs)
//!   ↓
//! Backward functions (chain rule)
//! ```

use crate::array::Array;
use anyhow::Result;
use std::cell::RefCell;
use std::rc::Rc;

pub mod backward;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod tensor;
pub mod train;

pub use nn::{
    BatchNorm1d, Conv1d, Dropout, Flatten, Linear, Module, ReLU, Sequential, Sigmoid, Softmax,
};
pub use optim::{
    AdaBound, AdaDelta, AdaGrad, Adam, AdamW, CosineAnnealingLR, ExponentialLR, LinearWarmup,
    Lookahead, NAdam, Optimizer, RAdam, RMSprop, ReduceLROnPlateau, Rprop, Scheduler, StepLR, LAMB,
    LBFGS, SGD,
};
pub use tensor::Tensor;
pub use train::{
    CrossEntropyLoss, Dataset, LossFunction, MSELoss, Metrics, Trainer, TrainerBuilder,
};

/// Identificador único para cada nodo en el compute graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// Tipo de operación en el compute graph
#[derive(Debug, Clone)]
pub enum OpKind {
    // Operaciones básicas
    Add,
    Mul,
    Sub,
    Div,
    Neg, // -x
    Abs, // |x|

    // Operaciones de matriz
    MatMul,
    Transpose,

    // Operaciones elementwise
    Exp,
    Log,
    Pow(f32),
    Sqrt,

    // Activaciones
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,

    // Trig
    Sin,
    Cos,
    Tan,

    // Loss functions
    MSE,
    CrossEntropy,

    // Reductions
    Sum {
        axis: Option<usize>,
    },
    Mean {
        axis: Option<usize>,
    },
    Max {
        axis: Option<usize>,
    },

    // Neural Network
    Conv1D {
        stride: usize,
        padding: usize,
    },
    Flatten {
        start_dim: usize,
        end_dim: usize,
    },
    Reshape {
        shape: Vec<usize>,
    },
    BatchNorm {
        training: bool,
        momentum: f32,
        eps: f32,
    },
    Dropout {
        p: f32,
        training: bool,
    },

    // Placeholder para leaf nodes
    Leaf,
}

/// Función de backward para una operación específica
///
/// Argumentos:
/// - `grad_output`: Gradiente que llega desde arriba (dL/dout)
/// - `inputs`: Tensors de entrada de la operación forward
/// - `output`: Tensor de salida de la operación forward
///
/// Retorna:
/// - Vec de gradientes para cada input (dL/dinput_i)
pub type BackwardFn = Box<dyn Fn(&Array, &[Tensor], &Tensor) -> Result<Vec<Array>>>;

/// Nodo en el compute graph
#[derive(Clone)]
pub struct ComputeNode {
    pub id: NodeId,
    pub op: OpKind,
    pub inputs: Vec<Tensor>,
    pub backward_fn: Option<Rc<BackwardFn>>,
}

impl ComputeNode {
    pub fn new(op: OpKind, inputs: Vec<Tensor>, backward_fn: Option<BackwardFn>) -> Self {
        Self {
            id: NodeId::new(),
            op,
            inputs,
            backward_fn: backward_fn.map(Rc::new),
        }
    }

    pub fn leaf() -> Self {
        Self {
            id: NodeId::new(),
            op: OpKind::Leaf,
            inputs: vec![],
            backward_fn: None,
        }
    }
}

impl std::fmt::Debug for ComputeNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeNode")
            .field("id", &self.id)
            .field("op", &self.op)
            .field("inputs", &self.inputs.len())
            .field("has_backward", &self.backward_fn.is_some())
            .finish()
    }
}

/// Contexto de autograd - guarda el estado del compute graph
#[derive(Debug, Clone)]
pub struct AutogradContext {
    pub enabled: bool,
}

impl AutogradContext {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn no_grad() -> Self {
        Self { enabled: false }
    }
}

thread_local! {
    static AUTOGRAD_ENABLED: RefCell<bool> = RefCell::new(true);
}

/// Verifica si autograd está habilitado en el thread actual
pub fn is_grad_enabled() -> bool {
    AUTOGRAD_ENABLED.with(|enabled| *enabled.borrow())
}

/// Establece el estado de autograd
pub fn set_grad_enabled(enabled: bool) {
    AUTOGRAD_ENABLED.with(|e| *e.borrow_mut() = enabled);
}

/// Context manager para deshabilitar temporalmente autograd
pub struct NoGrad {
    prev_state: bool,
}

impl NoGrad {
    pub fn new() -> Self {
        let prev_state = is_grad_enabled();
        set_grad_enabled(false);
        Self { prev_state }
    }
}

impl Drop for NoGrad {
    fn drop(&mut self) {
        set_grad_enabled(self.prev_state);
    }
}

/// Macro para ejecutar código sin gradientes
///
/// # Ejemplo
/// ```
/// use numrs::no_grad;
/// # struct MockModel;
/// # impl MockModel { fn forward(&self, _i: &i32) {} }
/// # let model = MockModel;
/// # let input = 0;
/// no_grad! {
///     let pred = model.forward(&input);  // No construye compute graph
/// }
/// ```
#[macro_export]
macro_rules! no_grad {
    ($($body:tt)*) => {{
        let _guard = $crate::autograd::NoGrad::new();
        $($body)*
    }};
}
