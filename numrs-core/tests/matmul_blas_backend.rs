use numrs::array::Array;

#[test]
fn blas_backend_matmul_small() {
    // Test usando el API moderno (dispatch system)
    let a = Array::new(vec![2,3], vec![1.0,2.0,3.0, 4.0,5.0,6.0]);
    let b = Array::new(vec![3,2], vec![7.0,8.0, 9.0,10.0, 11.0,12.0]);

    let out = numrs::ops::matmul(&a, &b).expect("matmul should succeed");

    // expected: [[58,64],[139,154]] flattened
    assert_eq!(out.shape(), vec![2usize, 2usize]);
    let expected = vec![58.0f32, 64.0, 139.0, 154.0];
    assert_eq!(out.to_f32().data, expected);
}
