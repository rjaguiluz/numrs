use numrs::array::Array;
use numrs::autograd::optim::{AdaBound, NAdam, Optimizer, RAdam, Rprop, LAMB};
use numrs::autograd::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

fn create_param(val: f32, grad_val: f32) -> Rc<RefCell<Tensor>> {
    let data = Array::new(vec![1], vec![val]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    let grad = Array::new(vec![1], vec![grad_val]);
    param.borrow_mut().grad = Some(Rc::new(RefCell::new(grad)));
    param
}

#[test]
fn test_nadam_basic() {
    let param = create_param(1.0, 1.0);
    // NAdam: lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
    // Removed extra momentum_decay arg
    let mut optim = NAdam::new(vec![param.clone()], 0.002, 0.9, 0.999, 1e-8, 0.0);
    optim.step().unwrap();
    assert!(param.borrow().data.data[0] < 1.0);
}

#[test]
fn test_radam_basic() {
    let param = create_param(1.0, 1.0);
    // RAdam: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
    let mut optim = RAdam::new(vec![param.clone()], 0.001, 0.9, 0.999, 1e-8, 0.0);
    optim.step().unwrap();
    assert!(param.borrow().data.data[0] < 1.0);
}

#[test]
fn test_lamb_basic() {
    let param = create_param(1.0, 1.0);
    // LAMB: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
    let mut optim = LAMB::new(vec![param.clone()], 0.001, 0.9, 0.999, 1e-8, 0.0);
    optim.step().unwrap();
    assert!(param.borrow().data.data[0] < 1.0);
}

#[test]
fn test_rprop_basic() {
    let param = create_param(1.0, 1.0);
    // Rprop: lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0)
    let mut optim = Rprop::new(vec![param.clone()], 0.01, 0.5, 1.2, 1e-6, 50.0);
    optim.step().unwrap();
    // gradient > 0 -> step is -lr if grad > 0 (simplified logic check)
    assert!(param.borrow().data.data[0] < 1.0);
}

#[test]
fn test_adabound_basic() {
    let param = create_param(1.0, 1.0);
    // AdaBound: lr=0.001, final_lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, gamma=0.001
    // Added gamma arg
    let mut optim = AdaBound::new(
        vec![param.clone()],
        0.001,
        0.1,
        0.9,
        0.999,
        1e-8,
        0.0,
        0.001,
    );
    optim.step().unwrap();
    assert!(param.borrow().data.data[0] < 1.0);
}

#[test]
fn test_rprop_sign_flip() {
    let param = create_param(1.0, 0.0);
    // Rprop with large step sizes
    let mut optim = Rprop::new(vec![param.clone()], 0.01, 0.5, 1.2, 1e-6, 50.0);

    // Step 1: grad > 0
    param.borrow_mut().grad.as_ref().unwrap().borrow_mut().data[0] = 1.0;
    optim.step().unwrap();

    // Step 2: grad < 0 (sign flip)
    param.borrow_mut().grad.as_ref().unwrap().borrow_mut().data[0] = -1.0;
    optim.step().unwrap();

    // Rprop logic should handle sign flip by reducing step size (multiplying by 0.5)
    // We just verify it runs and parameter updates.
    let p = param.borrow().data.data[0];
    assert!(p != 1.0);
}

#[test]
fn test_adabound_clipping() {
    // Large step that might trigger bounds
    let param = create_param(0.0, 100.0);
    // AdaBound with gamma=0.1 for faster transition
    let mut optim = AdaBound::new(vec![param.clone()], 0.01, 0.1, 0.9, 0.999, 1e-8, 0.0, 0.1);

    // Run multiple steps to trigger bound logic
    for _ in 0..5 {
        optim.step().unwrap();
    }

    let p = param.borrow().data.data[0];
    assert!(p < 0.0);
}
