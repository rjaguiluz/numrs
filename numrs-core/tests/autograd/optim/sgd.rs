use numrs::array::Array;
use numrs::autograd::optim::{Optimizer, SGD};
use numrs::autograd::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn test_sgd_step_basic() {
    // p = [1.0, 2.0]
    let data = Array::new(vec![2], vec![1.0, 2.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));

    // grad = [0.1, 0.2] manually set
    let grad = Array::new(vec![2], vec![0.1, 0.2]);
    // Fix: Wrap grad in Rc<RefCell<>>
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    // SGD: lr=0.1, momentum=0
    let mut optim = SGD::new(vec![param.clone()], 0.1, 0.0, 0.0);
    optim.step().unwrap();

    // Expected:
    // p[0] = 1.0 - 0.1 * 0.1 = 0.99
    // p[1] = 2.0 - 0.1 * 0.2 = 1.98
    let p_data = param.borrow().data.clone();
    assert!((p_data.data[0] - 0.99).abs() < 1e-5);
    assert!((p_data.data[1] - 1.98).abs() < 1e-5);
}

#[test]
fn test_sgd_momentum() {
    // p = [1.0], grad = [1.0]
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));

    let grad = Array::new(vec![1], vec![1.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad.clone())));

    // SGD: lr=0.1, momentum=0.9
    let mut optim = SGD::new(vec![param.clone()], 0.1, 0.9, 0.0);

    // Step 1:
    // v = 0.9*0 + 1.0 = 1.0
    // p = 1.0 - 0.1*1.0 = 0.9
    optim.step().unwrap();
    let p1 = param.borrow().data.data[0];
    assert!((p1 - 0.9).abs() < 1e-5);

    // Step 2:
    // grad still 1.0 (need to re-wrap because we might have optimized/consumed logic?
    // actually param.grad is persistent unless zeroed, but let's be safe if impl changed)
    // Actually, SGD implementation reads grad via get_gradient helper which clones data.
    // So existing grad matches.

    // v = 0.9*1.0 + 1.0 = 1.9
    // p = 0.9 - 0.1*1.9 = 0.9 - 0.19 = 0.71
    optim.step().unwrap();
    let p2 = param.borrow().data.data[0];
    assert!((p2 - 0.71).abs() < 1e-5);
}

#[test]
fn test_sgd_weight_decay() {
    // p = [10.0]
    let data = Array::new(vec![1], vec![10.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));

    // grad = [0.0]
    let grad = Array::new(vec![1], vec![0.0]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));

    // decay = 0.1
    // effective grad = grad + wd * p = 0 + 0.1 * 10 = 1.0
    // p = p - lr * eff_grad = 10 - 0.1 * 1.0 = 9.9
    let mut optim = SGD::new(vec![param.clone()], 0.1, 0.0, 0.1);
    optim.step().unwrap();

    let p = param.borrow().data.data[0];
    assert!((p - 9.9).abs() < 1e-5);
}
