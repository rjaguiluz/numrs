use numrs::array::Array;
use numrs::backend::cpu::simd::{
    dot_simd, elementwise_simd, elementwise_simd_supported, matmul_simd, reduce_simd,
};
use numrs::{ElementwiseKind, ReductionKind};

#[test]
fn test_simd_feature_detection() {
    let supported = elementwise_simd_supported();
    println!("SIMD Supported: {}", supported);
    // Cannot assert true/false as it depends on host, but function should run.
}

#[test]
fn test_simd_elementwise_add() {
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![4], vec![5.0, 6.0, 7.0, 8.0]);

    let result = elementwise_simd(&a, &b, ElementwiseKind::Add).expect("simd add failed");

    assert_eq!(result.shape, vec![4]);
    assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_simd_elementwise_mul_large() {
    // Large enough to trigger SIMD loops (e.g. > 8 elements for AVX2)
    let size = 100;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|_| 2.0).collect();

    let a = Array::new(vec![size], data_a.clone());
    let b = Array::new(vec![size], data_b);

    let result = elementwise_simd(&a, &b, ElementwiseKind::Mul).expect("simd mul failed");

    for i in 0..size {
        assert_eq!(result.data[i], data_a[i] * 2.0);
    }
}

#[test]
fn test_simd_reduce_sum() {
    let size = 100;
    let data: Vec<f32> = vec![1.0; size]; // Sum should be 100.0
    let a = Array::new(vec![size], data);

    let result = reduce_simd(&a, None, ReductionKind::Sum).expect("simd reduce sum failed");

    assert_eq!(result.data[0], 100.0);
}

#[test]
fn test_simd_reduce_max() {
    let a = Array::new(vec![5], vec![1.0, 5.0, 3.0, 9.0, 2.0]);
    let result = reduce_simd(&a, None, ReductionKind::Max).expect("simd max failed");
    assert_eq!(result.data[0], 9.0);
}

#[test]
fn test_simd_reduce_min() {
    let a = Array::new(vec![5], vec![1.0, 5.0, -3.0, 9.0, 2.0]);
    let result = reduce_simd(&a, None, ReductionKind::Min).expect("simd min failed");
    assert_eq!(result.data[0], -3.0);
}

#[test]
fn test_simd_dot_product() {
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![4], vec![0.5, 0.5, 0.5, 0.5]);
    // 0.5 + 1.0 + 1.5 + 2.0 = 5.0

    let result = dot_simd(&a, &b).expect("simd dot failed");
    assert!((result - 5.0).abs() < 1e-6);
}

#[test]
fn test_simd_matmul() {
    // 2x3 * 3x2 -> 2x2
    let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Array::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    // Row 0, Col 0: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // Row 0, Col 1: 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // Row 1, Col 0: 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // Row 1, Col 1: 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    let result = matmul_simd(&a, &b); // Note: matmul_simd returns Array, not Result<Array> in current signature

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_simd_matmul_large() {
    // Check correct propagation of large parallel/blocked logic
    let m = 64;
    let k = 64;
    let n = 64;

    let a = Array::new(vec![m, k], vec![1.0; m * k]);
    let b = Array::new(vec![k, n], vec![1.0; k * n]);

    let result = matmul_simd(&a, &b);

    assert_eq!(result.shape, vec![m, n]);
    // Each element = sum(1.0 * 1.0 for K times) = K
    assert!((result.data[0] - 64.0).abs() < 1e-4);
    assert!((result.data[m * n - 1] - 64.0).abs() < 1e-4);
}
