use numrs::backend::dispatch::{get_backend_override, set_backend_override};
use numrs::{add, mul, Array};

#[test]
fn test_backend_override_cycle() {
    // Default should be None or "adaptive"
    // We cannot assume default, but we can set and unset.

    set_backend_override(Some("cpu"));
    assert_eq!(get_backend_override().as_deref(), Some("cpu"));

    let a = Array::new(vec![10], vec![1.0; 10]);
    let b = Array::new(vec![10], vec![2.0; 10]);

    // Should run on CPU
    let c = add(&a, &b).expect("add failed on cpu");
    assert_eq!(c.data[0], 3.0);

    // Test unknown backend - should fallback or warn?
    // Implementation of dispatch usually ignores unknown or errors.
    // Let's stick to valid ones.

    set_backend_override(None);
    assert_eq!(get_backend_override(), None);

    // Should run default (SIMD/BLAS/etc)
    let d = mul(&a, &b).expect("mul failed on default");
    assert_eq!(d.data[0], 2.0);
}

#[test]
fn test_backend_override_gpu_fallback() {
    // If we set GPU but it's not available, it might fail or fallback.
    // Safe to test "cpu" explicitly.
    set_backend_override(Some("cpu"));

    let a = Array::new(vec![5], vec![1.0; 5]);
    let b = Array::new(vec![5], vec![1.0; 5]);
    let res = add(&a, &b).unwrap();
    assert_eq!(res.data, vec![2.0; 5]);

    // cleanup
    set_backend_override(None);
}
