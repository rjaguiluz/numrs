use anyhow::Result;
use numrs::autograd::Tensor;
use numrs::Array;

#[test]
fn test_creates_tensor() {
    let data = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let tensor = Tensor::new(data, true);
    assert_eq!(tensor.shape(), &[2, 2]);
    assert!(tensor.requires_grad);
    assert!(tensor.grad.is_some());
}

#[test]
fn test_backward_add() -> Result<()> {
    // z = x + y
    // dz/dx = 1
    // dz/dy = 1

    let x_data = Array::new(vec![2], vec![1.0, 2.0]);
    let y_data = Array::new(vec![2], vec![3.0, 4.0]);

    let x = Tensor::new(x_data, true);
    let y = Tensor::new(y_data, true);

    let z = x.add(&y)?;

    // Check forward pass
    assert_eq!(z.values(), &[4.0, 6.0]);

    // Backward
    z.backward()?;

    // Check gradients
    let x_grad = x.gradient().unwrap();
    let y_grad = y.gradient().unwrap();

    assert_eq!(x_grad.data, vec![1.0, 1.0]);
    assert_eq!(y_grad.data, vec![1.0, 1.0]);

    Ok(())
}

#[test]
fn test_backward_mul() -> Result<()> {
    // z = x * y
    // dz/dx = y
    // dz/dy = x

    let x = Tensor::new(Array::new(vec![2], vec![2.0, 3.0]), true);
    let y = Tensor::new(Array::new(vec![2], vec![4.0, 5.0]), true);

    let z = x.mul(&y)?;

    // Forward: [8.0, 15.0]
    assert_eq!(z.values(), &[8.0, 15.0]);

    z.backward()?;

    let x_grad = x.gradient().unwrap();
    let y_grad = y.gradient().unwrap();

    // grad_x should be y: [4.0, 5.0]
    assert_eq!(x_grad.data, vec![4.0, 5.0]);

    // grad_y should be x: [2.0, 3.0]
    assert_eq!(y_grad.data, vec![2.0, 3.0]);

    Ok(())
}

#[test]
fn test_complex_graph() -> Result<()> {
    // z = x * y + x
    // dz/dx = y + 1
    // dz/dy = x

    let x = Tensor::new(Array::new(vec![1], vec![2.0]), true);
    let y = Tensor::new(Array::new(vec![1], vec![3.0]), true);

    let prod = x.mul(&y)?;
    let z = prod.add(&x)?;

    // Forward: 2*3 + 2 = 8
    assert_eq!(z.values()[0], 8.0);

    z.backward()?;

    let x_grad = x.gradient().unwrap();
    let y_grad = y.gradient().unwrap();

    // grad_x = y + 1 = 3 + 1 = 4
    assert_eq!(x_grad.data[0], 4.0);

    // grad_y = x = 2
    assert_eq!(y_grad.data[0], 2.0);

    Ok(())
}

#[test]
fn test_no_grad_tensor() -> Result<()> {
    // x requires grad, y does not
    // z = x + y
    // dz/dx = 1, dz/dy = None

    let x = Tensor::new(Array::new(vec![1], vec![1.0]), true);
    let y = Tensor::new(Array::new(vec![1], vec![1.0]), false);

    let z = x.add(&y)?;

    z.backward()?;

    assert!(x.gradient().is_some());
    assert!(y.gradient().is_none());

    Ok(())
}
