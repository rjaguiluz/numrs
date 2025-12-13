use anyhow::Result;
use numrs::autograd::Tensor;
use numrs::Array;

#[test]
fn test_sum_backward() -> Result<()> {
    // y = sum(x)
    // dy/dx_i = 1

    let x = Tensor::new(Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]), true);
    let sum = x.sum()?; // Reduce all, keep_dims=None

    assert_eq!(sum.values(), &[10.0]);

    sum.backward()?;

    let grad = x.gradient().unwrap();
    assert_eq!(grad.data, vec![1.0, 1.0, 1.0, 1.0]);

    Ok(())
}

#[test]
fn test_mean_backward() -> Result<()> {
    // y = mean(x)
    // dy/dx_i = 1/N

    let x = Tensor::new(Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]), true);
    let mean = x.mean()?;

    assert_eq!(mean.values()[0], 2.5);

    mean.backward()?;

    let grad = x.gradient().unwrap();
    // 1/4 = 0.25
    assert_eq!(grad.data, vec![0.25, 0.25, 0.25, 0.25]);

    Ok(())
}
