use numrs::array::Array;
use numrs::autograd::optim::{Adam, Optimizer};
use numrs::autograd::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn test_adam_step_one() {
    // p = [1.0]
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));

    // grad = [1.0]
    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    // Adam: lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8
    let mut optim = Adam::new(vec![param.clone()], 0.1, 0.9, 0.999, 1e-8, 0.0);
    optim.step().unwrap();

    // t = 1
    // m = 0.9*0 + 0.1*1 = 0.1
    // v = 0.999*0 + 0.001*1^2 = 0.001
    // m_hat = 0.1 / (1 - 0.9) = 1.0
    // v_hat = 0.001 / (1 - 0.999) = 1.0
    // p = 1.0 - 0.1 * 1.0 / (sqrt(1.0) + 1e-8) ~= 1.0 - 0.1 = 0.9

    let p = param.borrow().data.data[0];
    assert!((p - 0.9).abs() < 1e-5);
}

#[test]
fn test_adam_step_two() {
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad.clone())));

    let mut optim = Adam::new(vec![param.clone()], 0.1, 0.9, 0.999, 1e-8, 0.0);
    optim.step().unwrap(); // p -> 0.9 (validated above)
    optim.step().unwrap(); // Same grad

    // t=2
    // m = 0.9*0.1 + 0.1*1 = 0.09 + 0.1 = 0.19
    // v = 0.999*0.001 + 0.001*1 = 0.000999 + 0.001 = 0.001999
    // m_hat = 0.19 / (1-0.81) = 0.19 / 0.19 = 1.0
    // v_hat = 0.001999 / (1 - 0.998001) = 0.001999 / 0.001999 = 1.0
    // p = 0.9 - 0.1 * 1.0 / 1.0 = 0.8

    let p = param.borrow().data.data[0];
    assert!((p - 0.8).abs() < 1e-4);
}

#[test]
fn test_adam_default_constructor() {
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    // Just verify it compiles and runs. Mut not needed for zero_grad/learning_rate access.
    let optim = Adam::default(vec![param], 0.001);
    optim.zero_grad();
    assert_eq!(optim.learning_rate(), 0.001);
}
