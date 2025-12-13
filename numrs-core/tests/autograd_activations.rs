use numrs::array::Array;
use numrs::autograd::{Module, ReLU, Sigmoid, Softmax, Tensor};

#[test]
fn test_relu_backward() {
    // f(x) = relu(x)
    // x = [-1, 0, 1]
    let x = Tensor::new(Array::new(vec![3], vec![-1.0, 0.0, 1.0]), true);
    let relu = ReLU;
    let y = relu.forward(&x).unwrap();

    // y = [0, 0, 1]
    // dy/dx = [0, 0, 1] (at 0 it's technically 0 in this impl)

    // Gradient is ones by default in backward() for non-scalar if logic allows or just use implicit
    // Actually backward() sets 1.0s.

    y.backward().unwrap();

    let x_grad = x.grad.unwrap();
    let data = x_grad.borrow().to_f32().data.clone();

    assert_eq!(data[0], 0.0);
    assert_eq!(data[2], 1.0);
}

#[test]
fn test_sigmoid_values() {
    let x = Tensor::new(Array::new(vec![3], vec![0.0, 2.0, -2.0]), false);
    let sig = Sigmoid;
    let y = sig.forward(&x).unwrap();

    let data = y.data.to_f32().data;
    assert_eq!(data[0], 0.5); // sig(0) = 0.5
    assert!((data[1] - 0.880797).abs() < 1e-5);
    assert!((data[2] - 0.119202).abs() < 1e-5);
}

#[test]
#[ignore] // Core softmax doesn't support axis yet
fn test_softmax_dim() {
    // Softmax along dim 1
    let x = Tensor::new(Array::new(vec![1, 2], vec![0.0, 0.0]), false);
    let sm = Softmax;
    let y = sm.forward(&x).unwrap();

    // exp(0)/2 = 1/2 = 0.5
    let data = y.data.to_f32().data;
    assert_eq!(data[0], 0.5);
    assert_eq!(data[1], 0.5);
}
