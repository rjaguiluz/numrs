use numrs::array::Array;
use numrs::ir::{HloNode, HloOp, IRGraph};

fn make_binary_graph(op: HloOp, left_shape: Vec<usize>, right_shape: Vec<usize>) -> IRGraph {
    let mut g = IRGraph::new();
    g.nodes.push(HloNode::new(0, HloOp::Const, vec![], left_shape));
    g.nodes.push(HloNode::new(1, HloOp::Const, vec![], right_shape));
    g.nodes.push(HloNode::new(2, op, vec![0,1], vec![0]));
    g
}

fn backend_name(be: &numrs::backend::SelectedBackend) -> &'static str {
    match be {
        numrs::backend::SelectedBackend::CPU(_) => "cpu",
        numrs::backend::SelectedBackend::WebGPU(_) => "webgpu",
        numrs::backend::SelectedBackend::CUDA(_) => "cuda",
        numrs::backend::SelectedBackend::Metal(_) => "metal",
        numrs::backend::SelectedBackend::Blas(_) => "blas",
    }
}

