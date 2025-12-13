use anyhow::Result;
use numrs::autograd::Tensor;
use numrs::Array;

#[test]
fn test_matmul_backward() -> Result<()> {
    // A: 2x2, B: 2x2
    // C = A @ B

    let a_data = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    // B = I
    let b_data = Array::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);

    let a = Tensor::new(a_data, true);
    let b = Tensor::new(b_data, true);

    let c = a.matmul(&b)?;

    // C should be equal to A
    assert_eq!(c.values(), &[1.0, 2.0, 3.0, 4.0]);

    // Backward
    c.backward()?;

    // d(A@B)/dA = grad_output @ B^T
    // grad_output = 1 (initialized)
    // grad_output is actually initialized to all 1s (matrix of 1s)
    // So grad_A = Ones @ B^T = Ones @ I = Ones

    let a_grad = a.gradient().unwrap();
    // Expect 1.0 everywhere?
    // Let's verify manually:
    // L = sum(C_ij)  (implicitly, backward initializes grad_output with 1s)
    // C_00 = A_00*B_00 + A_01*B_10 = A_00*1 + A_01*0 = A_00
    // dL/dA_00 = dC_00/dA_00 * 1 = 1

    assert_eq!(a_grad.data, vec![1.0, 1.0, 1.0, 1.0]);

    // grad_B = A^T @ Ones
    // A^T = [[1, 3], [2, 4]]
    // A^T @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]

    let b_grad = b.gradient().unwrap();
    assert_eq!(b_grad.data, vec![4.0, 4.0, 6.0, 6.0]);

    Ok(())
}
