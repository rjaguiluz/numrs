use numrs::array::Array;
use numrs::autograd::optim::{AdaDelta, AdaGrad, AdamW, Optimizer, RMSprop};
use numrs::autograd::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn test_adamw_weight_decay() {
    let data = Array::new(vec![1], vec![10.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![0.0]); // No gradient
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    // AdamW: weight_decay=0.1
    // AdamW applies decay to param *before* regular update or as decoupled decay
    // depending on impl. Usually p -= lr * (wd * p) separate from grad op
    let mut optim = AdamW::new(vec![param.clone()], 0.1, 0.9, 0.999, 1e-8, 0.1);
    optim.step().unwrap();

    // p should decrease purely due to decay
    let p = param.borrow().data.data[0];
    assert!(p < 10.0);
}

#[test]
fn test_rmsprop_momentum() {
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad.clone())));

    // RMSprop with momentum = 0.9
    let mut optim = RMSprop::new(vec![param.clone()], 0.01, 0.99, 1e-8, 0.0, 0.9);
    optim.step().unwrap();

    let p1 = param.borrow().data.data[0];

    // Second step
    optim.step().unwrap();
    let p2 = param.borrow().data.data[0];

    // With momentum, updates should accumulate/accelerate or at least be different
    assert!(p2 < p1);
}

#[test]
fn test_adagrad_weight_decay() {
    let data = Array::new(vec![1], vec![10.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    let mut optim = AdaGrad::new(vec![param.clone()], 0.01, 1e-10, 0.1);
    optim.step().unwrap();

    // Check it runs
    assert!(param.borrow().data.data[0] < 10.0);
}

#[test]
fn test_adamw_basic() {
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    // AdamW: lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1
    let mut optim = AdamW::new(vec![param.clone()], 0.1, 0.9, 0.999, 1e-8, 0.1);
    optim.step().unwrap();

    // Verify it changed
    let p = param.borrow().data.data[0];
    assert!(p < 1.0);
}

#[test]
fn test_rmsprop_basic() {
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    // RMSprop: lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0
    // Removed 'centered' arg
    let mut optim = RMSprop::new(vec![param.clone()], 0.01, 0.99, 1e-8, 0.0, 0.0);
    optim.step().unwrap();

    let p = param.borrow().data.data[0];
    assert!(p < 1.0);
}

#[test]
fn test_adagrad_basic() {
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    // AdaGrad: lr=0.01, eps=1e-10, weight_decay=0.0
    // Removed unused args
    let mut optim = AdaGrad::new(vec![param.clone()], 0.01, 1e-10, 0.0);
    optim.step().unwrap();

    let p = param.borrow().data.data[0];
    assert!(p < 1.0);
}

#[test]
fn test_adadelta_basic() {
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    // AdaDelta: rho=0.9, eps=1e-6, weight_decay=0.0
    let mut optim = AdaDelta::new(vec![param.clone()], 0.9, 1e-6, 0.0);
    optim.step().unwrap();

    let p = param.borrow().data.data[0];
    // AdaDelta should adaptively change param
    assert!(p != 1.0);
}
