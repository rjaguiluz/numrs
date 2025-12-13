use numrs::array::Array;
use numrs::ops;

#[test]
fn test_variance_simple() {
    let a = Array::new(vec![8], vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    let var = ops::variance(&a, None).unwrap();
    // Mean = 5.0
    // Diff^2: 9+1+1+1+0+0+4+16 = 32
    // Var = 32/8 = 4.0
    assert_eq!(var.data[0], 4.0);
}

#[test]
fn test_variance_axis() {
    let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // Axis 0 -> variances of cols [1,4], [2,5], [3,6]
    // Axis 0 means: (1+4)/2=2.5. (1-2.5)^2=2.25. (4-2.5)^2=2.25. Sum=4.5. Var=4.5/2=2.25. All same.
    let var0 = ops::variance(&a, Some(0)).unwrap();
    assert_eq!(var0.shape, vec![3]);
    for v in var0.data {
        assert_eq!(v, 2.25);
    }
}

#[test]
fn test_argmax_simple() {
    let a = Array::new(vec![5], vec![1.0, 5.0, 2.0, 9.0, 3.0]);
    let idx = ops::argmax(&a, None).unwrap();
    assert_eq!(idx.data[0], 3.0); // Index 3 is 9.0
}

#[test]
fn test_argmax_axis() {
    let a = Array::new(vec![2, 3], vec![1.0, 5.0, 2.0, 9.0, 3.0, 4.0]);

    // Axis 0: max of cols. [1,9]->9(idx 1), [5,3]->5(idx 0), [2,4]->4(idx 1)
    let idx0 = ops::argmax(&a, Some(0)).unwrap();
    assert_eq!(idx0.shape, vec![3]);
    assert_eq!(idx0.data, vec![1.0, 0.0, 1.0]);

    // Axis 1: max of rows. [1,5,2]->5(idx 1), [9,3,4]->9(idx 0)
    let idx1 = ops::argmax(&a, Some(1)).unwrap();
    assert_eq!(idx1.shape, vec![2]);
    assert_eq!(idx1.data, vec![1.0, 0.0]);
}

#[test]
fn test_norm_l2() {
    let a = Array::new(vec![3], vec![3.0, 4.0, 0.0]);
    let n = ops::norm(&a).unwrap();
    assert_eq!(n.data[0], 5.0);
}

#[test]
fn test_complex_reduction_chain() {
    // (a - mean(a)) / std(a)
    let a = Array::new(vec![4], vec![2.0, 4.0, 4.0, 4.0]); // Mean=3.5.
                                                           // Just ensure it runs without error
    let mean = ops::mean(&a, None).unwrap();
    let var = ops::variance(&a, None).unwrap();
    let std = ops::sqrt(&var).unwrap();

    let cent = ops::sub(&a, &mean).unwrap(); // Broadcasting scalar
    let z = ops::div(&cent, &std).unwrap();

    // Check shape
    assert_eq!(z.shape, vec![4]);
}
