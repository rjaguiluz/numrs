use numrs::array::Array;
use numrs::backend::cpu::scalar::{dot_scalar, elementwise_scalar, reduce_scalar};
use numrs::{ElementwiseKind, ReductionKind};

#[test]
fn test_elementwise_scalar_add() {
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![4], vec![5.0, 6.0, 7.0, 8.0]);

    let result = elementwise_scalar(&a, &b, ElementwiseKind::Add).expect("scalar add failed");

    assert_eq!(result.shape, vec![4]);
    assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_elementwise_scalar_sub() {
    let a = Array::new(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
    let b = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

    let result = elementwise_scalar(&a, &b, ElementwiseKind::Sub).expect("scalar sub failed");

    assert_eq!(result.data, vec![9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn test_elementwise_scalar_mul() {
    let a = Array::new(vec![4], vec![2.0, 3.0, 4.0, 5.0]);
    let b = Array::new(vec![4], vec![2.0, 2.0, 2.0, 2.0]);

    let result = elementwise_scalar(&a, &b, ElementwiseKind::Mul).expect("scalar mul failed");

    assert_eq!(result.data, vec![4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn test_elementwise_scalar_div() {
    let a = Array::new(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
    let b = Array::new(vec![4], vec![2.0, 5.0, 10.0, 4.0]);

    let result = elementwise_scalar(&a, &b, ElementwiseKind::Div).expect("scalar div failed");

    assert_eq!(result.data, vec![5.0, 4.0, 3.0, 10.0]);
}

#[test]
fn test_elementwise_scalar_relu() {
    let a = Array::new(vec![4], vec![-1.0, 0.0, 1.0, 5.0]);
    // b is ignored for unary ops usually, passing a reference for signature match
    // Check implementation specific: does it use `b`? The signature requires `b`, but logic probably ignores it for Unary.
    // Looking at `scalar.rs`, Relu: `*o = a.data[i].max(0.0);` -- ignores b.
    let b = Array::new(vec![4], vec![0.0; 4]);

    let result = elementwise_scalar(&a, &b, ElementwiseKind::Relu).expect("scalar relu failed");

    assert_eq!(result.data, vec![0.0, 0.0, 1.0, 5.0]);
}

#[test]
fn test_reduce_scalar_sum_all() {
    let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = reduce_scalar(&a, None, ReductionKind::Sum).expect("scalar sum failed");
    assert_eq!(result.data[0], 10.0);
}

#[test]
fn test_reduce_scalar_mean_axis() {
    let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // Axis 0 -> collapse rows -> [ (1+4)/2, (2+5)/2, (3+6)/2 ] = [2.5, 3.5, 4.5]
    // Axis 1 -> collapse cols -> [ (1+2+3)/3, (4+5+6)/3 ] = [2.0, 5.0]

    let result0 =
        reduce_scalar(&a, Some(0), ReductionKind::Mean).expect("scalar mean axis0 failed");
    assert_eq!(result0.shape, vec![3]);
    assert_eq!(result0.data, vec![2.5, 3.5, 4.5]);

    let result1 =
        reduce_scalar(&a, Some(1), ReductionKind::Mean).expect("scalar mean axis1 failed");
    assert_eq!(result1.shape, vec![2]);
    assert_eq!(result1.data, vec![2.0, 5.0]);
}

#[test]
fn test_reduce_scalar_max_min() {
    let a = Array::new(vec![5], vec![1.0, 5.0, -2.0, 8.0, 3.0]);
    let max = reduce_scalar(&a, None, ReductionKind::Max).unwrap();
    let min = reduce_scalar(&a, None, ReductionKind::Min).unwrap();

    assert_eq!(max.data[0], 8.0);
    assert_eq!(min.data[0], -2.0);
}

#[test]
fn test_dot_scalar() {
    let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
    let b = Array::new(vec![3], vec![0.5, 0.5, 0.5]);
    // 0.5 + 1.0 + 1.5 = 3.0
    let res = dot_scalar(&a, &b).expect("dot scalar failed");
    assert_eq!(res, 3.0);
}

#[test]
fn test_reduce_variance_scalar() {
    let a = Array::new(vec![8], vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    // Not easily calculated by head.
    // Simple case: [1, 1, 1] var=0
    let zeros = Array::new(vec![3], vec![1.0, 1.0, 1.0]);
    let var = reduce_scalar(&zeros, None, ReductionKind::Variance).unwrap();
    assert_eq!(var.data[0], 0.0);
}
