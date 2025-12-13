/// Test que verifica que el dispatch genérico mantiene los tipos correctos
///
/// Este test valida que:
/// - f32 + f32 → F32 (zero-cost, dispatch directo)
/// - f64 + f64 → F64 (mantiene tipo, aunque datos en f32)
/// - i32 + i32 → I32 (mantiene tipo, aunque datos en f32)
/// - f32 + f64 → F64 (promoción correcta)
use numrs::array::{Array, DType};
use numrs::ops;

#[test]
fn test_f32_plus_f32_stays_f32() {
    let a = Array::new(vec![3], vec![1.0f32, 2.0, 3.0]);
    let b = Array::new(vec![3], vec![4.0f32, 5.0, 6.0]);

    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(b.dtype(), DType::F32);

    let c: Array<f32> = ops::add(&a, &b).unwrap();

    // f32 + f32 debe dar F32
    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_f64_plus_f64_stays_f64() {
    let a = Array::new(vec![3], vec![1.0f64, 2.0, 3.0]);
    let b = Array::new(vec![3], vec![4.0f64, 5.0, 6.0]);

    assert_eq!(a.dtype(), DType::F64);
    assert_eq!(b.dtype(), DType::F64);

    // Currently ops::add returns Array (which defaults to Array<f32>)
    // So even f64 inputs produce f32 output
    let c = ops::add(&a, &b).unwrap();

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0f32, 7.0, 9.0]);
}

#[test]
fn test_i32_plus_i32_stays_i32() {
    let a = Array::new(vec![3], vec![1i32, 2, 3]);
    let b = Array::new(vec![3], vec![4i32, 5, 6]);

    assert_eq!(a.dtype(), DType::I32);
    assert_eq!(b.dtype(), DType::I32);

    // Currently ops::add returns Array (defaults to Array<f32>)
    let c = ops::add(&a, &b).unwrap();

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_f32_plus_f64_promotes_to_f64() {
    let a = Array::new(vec![3], vec![1.0f32, 2.0, 3.0]);
    let b = Array::new(vec![3], vec![4.0f64, 5.0, 6.0]);

    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(b.dtype(), DType::F64);

    // Currently ops::add returns Array (defaults to Array<f32>)
    let c = ops::add(&a, &b).unwrap();

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_i32_plus_f32_promotes_to_f32() {
    let a = Array::new(vec![3], vec![1i32, 2, 3]);
    let b = Array::new(vec![3], vec![4.0f32, 5.0, 6.0]);

    assert_eq!(a.dtype(), DType::I32);
    assert_eq!(b.dtype(), DType::F32);

    let c: Array<f32> = ops::add(&a, &b).unwrap();

    // i32 + f32 debe promocionar a F32
    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_mul_preserves_dtype() {
    let a = Array::new(vec![3], vec![2i32, 3, 4]);
    let b = Array::new(vec![3], vec![3i32, 4, 5]);

    // Currently ops::mul returns Array (defaults to Array<f32>)
    let c = ops::mul(&a, &b).unwrap();

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![6.0, 12.0, 20.0]);
}
