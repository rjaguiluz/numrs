use numrs::array::Array;
use numrs::backend::cpu::parallel::matmul_with_kernel;
use numrs::backend::cpu::simd::matmul_simd;

// Simple scalar kernel for verification
fn simple_matmul_kernel(a: &Array, b: &Array) -> Array {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    let mut result = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a.data[i * k + l] * b.data[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    Array::new(vec![m, n], result)
}

#[test]
fn test_parallel_matmul_small() {
    // Should fallback to direct execution
    let a = Array::new(vec![10, 10], vec![1.0; 100]);
    let b = Array::new(vec![10, 10], vec![1.0; 100]);

    let res = matmul_with_kernel(&a, &b, simple_matmul_kernel).unwrap();
    assert_eq!(res.shape, vec![10, 10]);
    assert_eq!(res.data[0], 10.0);
}

#[test]
fn test_parallel_matmul_large_threaded() {
    // Should trigger parallel path (m >= 500)
    let m = 600;
    let k = 10;
    let n = 10;

    let a = Array::new(vec![m, k], vec![1.0; m * k]);
    let b = Array::new(vec![k, n], vec![1.0; k * n]);

    // We use simple kernel to verify the chunking logic is correct
    // even if simple kernel is slow, 600x10x10 is small enough to run fast
    let res = matmul_with_kernel(&a, &b, simple_matmul_kernel).unwrap();

    assert_eq!(res.shape, vec![m, n]);
    // Each element = sum(1*1 for K=10) = 10
    assert!((res.data[0] - 10.0).abs() < 1e-5);
    assert!((res.data[m * n - 1] - 10.0).abs() < 1e-5);
}

#[test]
fn test_parallel_matmul_integration() {
    // Test with actual SIMD kernel
    let m = 600;
    let k = 64;
    let n = 64;

    let a = Array::new(vec![m, k], vec![1.0; m * k]);
    let b = Array::new(vec![k, n], vec![0.5; k * n]);

    // Use matmul_simd as kernel, but wrapped because matmul_simd signature might not match exactly if not closure?
    // matmul_simd signature: fn(&Array, &Array) -> Array. Matches fn ptr.

    let res = matmul_with_kernel(&a, &b, matmul_simd).unwrap();

    let expected = k as f32 * 1.0 * 0.5; // 64 * 0.5 = 32.0

    assert_eq!(res.shape, vec![m, n]);
    assert!((res.data[0] - expected).abs() < 1e-4);
}
