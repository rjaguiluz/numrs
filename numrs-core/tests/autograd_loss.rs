use numrs::array::Array;
use numrs::autograd::{CrossEntropyLoss, LossFunction, MSELoss, Tensor};

#[test]
fn test_mse_loss() {
    // Pred: [1, 2, 3]
    // Target: [1, 2, 5]
    // Diff: [0, 0, -2]
    // Sq: [0, 0, 4]
    // Mean: 4/3 = 1.333

    let pred = Tensor::new(Array::new(vec![3], vec![1.0, 2.0, 3.0]), true);
    let target = Tensor::new(Array::new(vec![3], vec![1.0, 2.0, 5.0]), false);

    let mse = MSELoss; // Unit struct, no new()
    let loss = mse.compute(&pred, &target).unwrap(); // Use strict API from LossFunction trait if needed, or forward if implemented
                                                     // Wait, LossFunction trait has `compute`. Does it have `forward`?
                                                     // src/autograd/train.rs defines `trait LossFunction { fn compute(...) }`.
                                                     // It does NOT inherit from Module.
                                                     // So distinct from Module. The test called `forward` earlier?
                                                     // Let me double check if `Module` is implemented for `MSELoss`.
                                                     // `src/autograd/train.rs` does NOT show `impl Module for MSELoss`.
                                                     // So I MUST use `.compute()`.

    let val = loss.item();
    assert!((val - 1.3333333).abs() < 1e-5);

    loss.backward().unwrap();

    // Grad = 2*(pred - target) / N
    // [0, 0, 2*(-2)/3] = [0, 0, -1.333]
    let p_grad = pred.grad.unwrap();
    let g_data = p_grad.borrow().to_f32().data.clone();

    assert_eq!(g_data[0], 0.0);
    assert!((g_data[2] + 1.333333).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_basic() {
    // Batch=1, Classes=2
    // Logits: [2.0, -1.0]
    // Target Index: 0
    // Softmax: exp(2)/(exp(2)+exp(-1)) = 7.389 / (7.389 + 0.367) = 7.389/7.756 = 0.952
    // Loss: -log(0.952) = 0.048

    let logits = Tensor::new(Array::new(vec![1, 2], vec![2.0, -1.0]), false);
    let targets = Tensor::new(Array::new(vec![1, 2], vec![1.0, 0.0]), false); // One-hot for index 0

    let ce = CrossEntropyLoss;
    let loss = ce.compute(&logits, &targets).unwrap();

    assert!(loss.item() < 0.1);
}
