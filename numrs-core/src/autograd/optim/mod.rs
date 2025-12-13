//! Modular optimizer implementations

pub mod base;
pub mod sgd;
pub mod adam;
pub mod adamw;
pub mod rmsprop;
pub mod adagrad;
pub mod nadam;
pub mod radam;
pub mod adadelta;
pub mod lamb;
pub mod lookahead;
pub mod adabound;
pub mod lbfgs;
pub mod rprop;
pub mod schedulers;

// Re-export main types
pub use base::Optimizer;
pub use sgd::SGD;
pub use adam::Adam;
pub use adamw::AdamW;
pub use rmsprop::RMSprop;
pub use adagrad::AdaGrad;
pub use nadam::NAdam;
pub use radam::RAdam;
pub use adadelta::AdaDelta;
pub use lamb::LAMB;
pub use lookahead::Lookahead;
pub use adabound::AdaBound;
pub use lbfgs::LBFGS;
pub use rprop::Rprop;
pub use schedulers::{
    Scheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearWarmup,
};
