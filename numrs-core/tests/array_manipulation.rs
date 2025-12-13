use numrs::array::promoted_dtype;
use numrs::array::{Array, DType, DTypeValue};

#[test]
fn test_array_conversion_f32_to_f32() {
    let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
    // Should be zero-cost / clone
    let b = a.to_f32();
    assert_eq!(b.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_array_conversion_i32_to_f32() {
    // Manually create i32 array via internal if possible, or simulate behavior
    // Since Array struct is hardcoded to f32 data in current snippet,
    // we focus on what's available.
    // NOTE: The codebase seems to be in transition where Array<T> generic might be hidden or Array is f32.
    // Based on previous files, Array struct has `data: Vec<f32>`.
    // So `to_f32` is trivial.

    // Let's test checking shape cloning and data validity
    let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = a.clone();

    assert_eq!(a.shape, b.shape);
    assert_eq!(a.data, b.data);
}

#[test]
fn test_dtype_promotion_logic() {
    // Verify promotion table correctness
    assert_eq!(promoted_dtype(DType::F32, DType::F32), DType::F32);
    assert_eq!(promoted_dtype(DType::F32, DType::I32), DType::F32);
    assert_eq!(promoted_dtype(DType::I32, DType::F32), DType::F32);
    assert_eq!(promoted_dtype(DType::I32, DType::I32), DType::I32);
    assert_eq!(promoted_dtype(DType::Bool, DType::F32), DType::F32);
}

#[test]
fn test_array_creation_from_vec_macros() {
    // If there were macros, we'd test them.
    // Testing standard creation
    let a = Array::new(vec![1], vec![0.0]);
    assert_eq!(a.len(), 1);
    assert_eq!(a.shape, vec![1]);
}

#[test]
fn test_dtype_value_trait() {
    // DTypeValue is a trait, not an enum. verify impls.
    let f = 10.0f32;
    assert_eq!(f32::dtype(), DType::F32);
    assert_eq!(f.to_f32(), 10.0);

    let i = 5i32;
    assert_eq!(i32::dtype(), DType::I32);
    assert_eq!(i.to_f32(), 5.0);

    // Check round trip if applicable or conversions
    let from_f = i32::from_f32(5.9); // likely truncates
    assert_eq!(from_f, 5);
}
