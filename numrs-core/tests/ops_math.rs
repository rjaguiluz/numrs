use numrs::array::Array;
use numrs::autograd::Tensor;

#[test]
fn test_sin_backward() {
    // y = sin(x)
    // x = 0 -> y = 0, dy/dx = cos(0) = 1
    // x = pi/2 -> y = 1, dy/dx = cos(pi/2) = 0
    let x = Tensor::new(Array::new(vec![2], vec![0.0, 1.570796]), true);
    let y = x.sin().unwrap();

    // let grad = Tensor::new(Array::ones(vec![2]), false); // Unused
    y.backward().unwrap();

    let g = x.grad.unwrap();
    let d = g.borrow().to_f32().data.clone();

    assert!((d[0] - 1.0).abs() < 1e-4);
    assert!(d[1].abs() < 1e-4);
}

#[test]
fn test_exp_log() {
    // y = log(exp(x)) = x
    let x = Tensor::new(Array::new(vec![1], vec![2.0]), true);
    let y = x.exp().unwrap().log().unwrap();

    assert!((y.item() - 2.0).abs() < 1e-5);

    y.backward().unwrap();
    // dy/dx should be 1.0 (chain rule: 1/exp(x) * exp(x) = 1)
    let g = x.grad.unwrap();
    assert!((g.borrow().to_f32().data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_pow_sqrt() {
    // x = 4
    // y = sqrt(x) = 2
    // z = y^3 = 8
    let x = Tensor::new(Array::new(vec![1], vec![4.0]), true);
    let y = x.sqrt().unwrap();
    let z = y.pow(3.0).unwrap();

    assert!((z.item() - 8.0).abs() < 1e-5);

    z.backward().unwrap();
    // z = (x^0.5)^3 = x^1.5
    // dz/dx = 1.5 * x^0.5 = 1.5 * 2 = 3.0

    let g = x.grad.unwrap();
    assert!((g.borrow().to_f32().data[0] - 3.0).abs() < 1e-5);
}
