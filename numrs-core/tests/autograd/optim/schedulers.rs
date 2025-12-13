use numrs::array::Array;
use numrs::autograd::optim::{ExponentialLR, Optimizer, Scheduler, StepLR, SGD};
use numrs::autograd::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

fn create_optimizer() -> SGD {
    let data = Array::new(vec![1], vec![1.0]);
    let param = Rc::new(RefCell::new(Tensor::new(data, true)));
    // lr = 1.0 for easy calculation
    SGD::new(vec![param], 1.0, 0.0, 0.0)
}

#[test]
fn test_step_lr() {
    let mut optim = create_optimizer();
    // Step size 1, gamma 0.1
    let mut scheduler = StepLR::new(1, 0.1);

    // Initial LR
    assert_eq!(optim.learning_rate(), 1.0);

    // Step 1 -> LR * 0.1 = 0.1
    scheduler.step(&mut optim);
    assert!((optim.learning_rate() - 0.1).abs() < 1e-5);

    // Step 2 -> LR * 0.1 = 0.01
    scheduler.step(&mut optim);
    assert!((optim.learning_rate() - 0.01).abs() < 1e-5);
}

#[test]
fn test_exponential_lr() {
    let mut optim = create_optimizer();
    // gamma 0.9
    let mut scheduler = ExponentialLR::new(0.9);

    scheduler.step(&mut optim);
    // 1.0 * 0.9 = 0.9
    assert!((optim.learning_rate() - 0.9).abs() < 1e-5);

    scheduler.step(&mut optim);
    // 0.9 * 0.9 = 0.81
    assert!((optim.learning_rate() - 0.81).abs() < 1e-5);
}
