use crate::ir::hlo::HloOp;
use crate::llo::{LLOProgram, LloOp};
use anyhow::{Result, bail};

/// Lower an HLO graph to a simple LLO program. This demonstrates the concept
/// and provides a base for backend-aware lowering strategies.
pub fn lower(graph: crate::ir::IRGraph) -> Result<LLOProgram> {
    // For prototype: linear translate nodes into LLO with simple mapping
    let mut progr = LLOProgram::new();

    // Lowering must be hardware-agnostic: emit abstract/default strategies only.
    // All runtime hardware detection and kernel resolution happens at startup.

    for node in graph.nodes.into_iter() {
        match node.op {
            HloOp::Add => {
                // Emit an abstract default strategy; runtime resolver will pick the concrete kernel.
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Add, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Mul => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Mul, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Sub => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Sub, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Div => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Div, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Pow => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Pow, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Sqrt => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Sqrt, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Abs => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Abs, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Exp => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Exp, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Sin => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Sin, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Cos => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Cos, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Tan => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Tan, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Asin => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Asin, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Acos => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Acos, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Atan => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Atan, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Relu => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::Relu, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::LeakyRelu => {
                let strat = crate::llo::ElementwiseStrategy::Default;
                progr.add_op(LloOp::Elementwise { kind: crate::llo::ElementwiseKind::LeakyRelu, inputs: node.inputs.clone(), output_shape: node.shape.clone(), strategy: strat });
            }
            HloOp::Sum { axis } => {
                progr.add_op(LloOp::Reduction { axis, inputs: node.inputs.clone(), output_shape: node.shape.clone() });
            }
            HloOp::Const => {
                // constants are treated as inputs; no LLO emitted for constant/inputs
            }
            HloOp::MatMul => {
                // assume inputs contain a and b ids
                if node.inputs.len() < 2 { bail!("MatMul HLO node requires two inputs"); }
                progr.add_op(LloOp::MatMul { a: node.inputs[0], b: node.inputs[1], output_shape: node.shape.clone() });
            }
            // exhaustively handled above; future cases should be handled explicitly
        }
    }

    Ok(progr)
}
