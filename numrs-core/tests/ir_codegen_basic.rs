use numrs::ir::{hlo::HloNode, hlo::HloOp, lowering, IRGraph};

#[test]
fn test_lower_basic_ops() {
    // Construct a dummy graph
    let mut graph = IRGraph::new();

    // Add nodes manually if possible or via helper
    // IRGraph fields are likely pub
    graph
        .nodes
        .push(HloNode::new(0, HloOp::Const, vec![], vec![2, 2]));

    graph
        .nodes
        .push(HloNode::new(1, HloOp::Add, vec![0, 0], vec![2, 2]));

    graph
        .nodes
        .push(HloNode::new(2, HloOp::Relu, vec![1], vec![2, 2]));

    graph
        .nodes
        .push(HloNode::new(3, HloOp::MatMul, vec![2, 0], vec![2, 2]));

    let llo_prog = lowering::lower(graph).expect("Lowering failed");

    // Verify LLO program content
    // Const is skipped
    // Add -> 1 op
    // Relu -> 1 op
    // MatMul -> 1 op
    // Total 3 ops
    assert_eq!(llo_prog.ops.len(), 3);
}

#[test]
fn test_lower_all_ops_smoke() {
    // Iterate through many ops to ensure coverage of the match arm
    let ops = vec![
        HloOp::Sub,
        HloOp::Mul,
        HloOp::Div,
        HloOp::Pow,
        HloOp::Sqrt,
        HloOp::Abs,
        HloOp::Exp,
        HloOp::Sin,
        HloOp::Cos,
        HloOp::Tan,
        HloOp::Asin,
        HloOp::Acos,
        HloOp::Atan,
        HloOp::LeakyRelu,
    ];

    for op in ops {
        let mut graph = IRGraph::new();
        graph
            .nodes
            .push(HloNode::new(0, HloOp::Const, vec![], vec![1]));
        graph.nodes.push(HloNode::new(1, op, vec![0, 0], vec![1]));

        // Should succeed
        let _ = lowering::lower(graph).unwrap();
    }
}
