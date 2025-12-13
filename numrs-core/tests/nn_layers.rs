use numrs::array::Array;
use numrs::autograd::nn::{BatchNorm1d, Conv1d, Dropout, Linear, Module};
use numrs::autograd::Tensor;
use numrs::no_grad;

#[test]
fn test_linear_forward() {
    // Input: (Batch=2, In=3)
    let input = Tensor::new(
        Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        true,
    );

    let linear = Linear::new(3, 2).unwrap(); // Out=2

    // Manually set weights for predictable output
    // W: (Out=2, In=3)
    // B: (Out=2)
    no_grad! {
        // Use getters for weight/bias access
        *linear.weight().borrow_mut() = Tensor::new(Array::new(vec![2, 3], vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6
        ]), true);
        *linear.bias().borrow_mut() = Tensor::new(Array::new(vec![2], vec![0.1, 0.2]), true);
    }

    let output = linear.forward(&input).unwrap();

    // Expected:
    // Row 1:
    //   0.1*1 + 0.2*2 + 0.3*3 + 0.1 = 0.1 + 0.4 + 0.9 + 0.1 = 1.5
    //   0.4*1 + 0.5*2 + 0.6*3 + 0.2 = 0.4 + 1.0 + 1.8 + 0.2 = 3.4
    // Row 2:
    //   0.1*4 + 0.2*5 + 0.3*6 + 0.1 = 0.4 + 1.0 + 1.8 + 0.1 = 3.3
    //   0.4*4 + 0.5*5 + 0.6*6 + 0.2 = 1.6 + 2.5 + 3.6 + 0.2 = 7.9

    assert_eq!(output.shape(), vec![2, 2]);
    let data = output.data.to_f32().data;

    assert!((data[0] - 1.5).abs() < 1e-5);
    assert!((data[1] - 3.4).abs() < 1e-5);
    assert!((data[2] - 3.3).abs() < 1e-5);
    assert!((data[3] - 7.9).abs() < 1e-5);
}

#[test]
fn test_conv1d_forward_shapes() {
    // Input: (Batch=1, Channels=3, Len=10)
    let input = Tensor::new(Array::zeros(vec![1, 3, 10]), false);

    // Conv: In=3, Out=6, Kernel=3, Stride=1, Pad=1
    let conv = Conv1d::new(3, 6, 3, 1, 1).unwrap();

    let output = conv.forward(&input).unwrap();

    // Output length: (10 + 2*1 - 3)/1 + 1 = 10
    // Shape: (1, 6, 10)
    assert_eq!(output.shape(), vec![1, 6, 10]);
}

#[test]
#[ignore] // WebGPU BatchNorm Training not yet implemented
fn test_batchnorm_running_stats() {
    let mut bn = BatchNorm1d::new(4).unwrap();

    // Before valid pass, running stats are default
    {
        // Use getter instead of direct field access
        let mean = bn.running_mean();
        let mean_borrow = mean.borrow();
        assert_eq!(mean_borrow.data.to_f32().data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    // Input: (Batch=2, Channels=4, Len=1) for simplicity
    let input = Tensor::new(
        Array::new(
            vec![2, 4, 1],
            vec![
                1.0, 2.0, 3.0, 4.0, // Sample 1
                1.0, 2.0, 3.0, 4.0, // Sample 2 (Same, so var=0)
            ],
        ),
        false,
    );

    // Training mode update stats
    bn.train();
    let _ = bn.forward(&input).unwrap();

    // Check if running stats updated (momentum default 0.1)
    // mean target = [1, 2, 3, 4]
    // new_running = 0.9 * 0 + 0.1 * target
    {
        let mean = bn.running_mean();
        let mean_borrow = mean.borrow();
        let data = mean_borrow.data.to_f32().data.clone();
        assert!((data[0] - 0.1).abs() < 1e-5);
        assert!((data[1] - 0.2).abs() < 1e-5);
    }
}

#[test]
fn test_dropout_training() {
    let mut drop = Dropout::new(0.5);
    let input = Tensor::new(Array::ones(vec![10, 10]), false);

    drop.train();
    let out_train = drop.forward(&input).unwrap();

    // In training, should have zeros
    let data = out_train.data.to_f32().data;
    let zeros = data.iter().filter(|&&x| x == 0.0).count();
    assert!(zeros > 0, "Dropout should zero out elements in training");

    drop.eval();
    let out_eval = drop.forward(&input).unwrap();
    let data_eval = out_eval.data.to_f32().data;
    // In eval, should be identity
    assert!(data_eval.iter().all(|&x| x == 1.0));
}
