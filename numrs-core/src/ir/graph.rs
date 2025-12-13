use crate::ir::hlo::{HloNode, HloOp};

/// Very small IR graph container. Holds nodes and entry points.
#[derive(Clone, Debug)]
pub struct IRGraph {
    pub nodes: Vec<HloNode>,
}

impl IRGraph {
    pub fn new() -> Self { Self { nodes: vec![] } }

    /// Append a node and return its id
    pub fn push(&mut self, node: HloNode) -> usize {
        let id = node.id;
        self.nodes.push(node);
        id
    }

    /// Simple convenience: create a 1-node graph for an op between two shapes
    pub fn binary_op(op: HloOp, left_shape: Vec<usize>, right_shape: Vec<usize>) -> Self {
        // naive broadcast: pick the bigger shape
        let shape = if left_shape.len() >= right_shape.len() { left_shape.clone() } else { right_shape.clone() };
        let mut g = IRGraph::new();
        // constants as id 0,1 then op id 2
        g.nodes.push(HloNode::new(0, op.clone(), vec![], left_shape));
        g.nodes.push(HloNode::new(1, op.clone(), vec![], right_shape));
        g.nodes.push(HloNode::new(2, op, vec![0,1], shape));
        g
    }
}
