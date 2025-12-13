//! Tests for implicit type conversion in Array::from_vec()

use numrs::array::{Array, DType};

#[test]
fn test_from_vec_i32_to_f32() {
    // Create f32 array from i32 data - automatic conversion
    let arr = Array::<f32>::from_vec(vec![3], vec![1i32, 2, 3]);

    assert_eq!(arr.dtype, DType::F32);
    assert_eq!(arr.shape, vec![3]);
    assert_eq!(arr.data, vec![1.0f32, 2.0, 3.0]);
}

#[test]
fn test_from_vec_f32_to_i32() {
    // Create i32 array from f32 data - automatic conversion (truncation)
    let arr = Array::<i32>::from_vec(vec![4], vec![1.5f32, 2.7, 3.2, 4.9]);

    assert_eq!(arr.dtype, DType::I32);
    assert_eq!(arr.shape, vec![4]);
    assert_eq!(arr.data, vec![1, 2, 3, 4]); // Truncated
}

#[test]
fn test_from_vec_i32_to_f64() {
    // Create f64 array from i32 data
    let arr = Array::<f64>::from_vec(vec![2, 2], vec![1i32, 2, 3, 4]);

    assert_eq!(arr.dtype, DType::F64);
    assert_eq!(arr.shape, vec![2, 2]);
    assert_eq!(arr.data, vec![1.0f64, 2.0, 3.0, 4.0]);
}

#[test]
fn test_from_vec_u8_to_f32() {
    // Create f32 array from u8 data (common for image processing)
    let arr = Array::<f32>::from_vec(vec![2, 2], vec![10u8, 20, 30, 40]);

    assert_eq!(arr.dtype, DType::F32);
    assert_eq!(arr.shape, vec![2, 2]);
    assert_eq!(arr.data, vec![10.0f32, 20.0, 30.0, 40.0]);
}

#[test]
fn test_from_vec_f64_to_f32() {
    // Create f32 array from f64 data (precision loss)
    let arr = Array::<f32>::from_vec(vec![3], vec![1.5f64, 2.5, 3.5]);

    assert_eq!(arr.dtype, DType::F32);
    assert_eq!(arr.data, vec![1.5f32, 2.5, 3.5]);
}

#[test]
fn test_from_vec_bool_to_i32() {
    // Create i32 array from bool data
    let arr = Array::<i32>::from_vec(vec![4], vec![true, false, true, true]);

    assert_eq!(arr.dtype, DType::I32);
    assert_eq!(arr.data, vec![1, 0, 1, 1]);
}

#[test]
fn test_from_vec_bool_to_f32() {
    // Create f32 array from bool data
    let arr = Array::<f32>::from_vec(vec![3], vec![true, false, true]);

    assert_eq!(arr.dtype, DType::F32);
    assert_eq!(arr.data, vec![1.0f32, 0.0, 1.0]);
}

#[test]
fn test_from_vec_same_type_f32() {
    // When types match, should be zero-cost (no conversion)
    let arr = Array::<f32>::from_vec(vec![3], vec![1.0f32, 2.0, 3.0]);

    assert_eq!(arr.dtype, DType::F32);
    assert_eq!(arr.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_from_vec_same_type_i32() {
    // When types match, should be zero-cost (no conversion)
    let arr = Array::<i32>::from_vec(vec![3], vec![1i32, 2, 3]);

    assert_eq!(arr.dtype, DType::I32);
    assert_eq!(arr.data, vec![1, 2, 3]);
}

#[test]
#[should_panic(expected = "shape length mismatch")]
fn test_from_vec_shape_mismatch() {
    // Should panic if shape doesn't match data length
    let _arr = Array::<f32>::from_vec(vec![2], vec![1i32, 2, 3]); // 2 != 3
}

#[test]
fn test_from_vec_multidimensional() {
    // Test with multi-dimensional shapes
    let arr = Array::<f32>::from_vec(vec![2, 3], vec![1i32, 2, 3, 4, 5, 6]);

    assert_eq!(arr.dtype, DType::F32);
    assert_eq!(arr.shape, vec![2, 3]);
    assert_eq!(arr.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_from_vec_i8_to_f32() {
    // Test with i8 (signed 8-bit)
    let arr = Array::<f32>::from_vec(vec![4], vec![-10i8, 0, 10, 20]);

    assert_eq!(arr.dtype, DType::F32);
    assert_eq!(arr.data, vec![-10.0f32, 0.0, 10.0, 20.0]);
}

#[test]
fn test_from_vec_f32_to_u8() {
    // Create u8 array from f32 (common for image output)
    let arr = Array::<u8>::from_vec(vec![4], vec![10.5f32, 20.7, 30.2, 255.9]);

    assert_eq!(arr.dtype, DType::U8);
    assert_eq!(arr.data, vec![10u8, 20, 30, 255]);
}

#[test]
fn test_new_still_works() {
    // Verify that Array::new() still works normally (no breaking changes)
    let arr = Array::<f32>::new(vec![3], vec![1.0f32, 2.0, 3.0]);

    assert_eq!(arr.dtype, DType::F32);
    assert_eq!(arr.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_from_vec_with_ops() {
    use numrs::ops;

    // Create arrays with different input types using from_vec
    let a = Array::<f32>::from_vec(vec![3], vec![1i32, 2, 3]);
    let b = Array::<f32>::from_vec(vec![3], vec![4u8, 5, 6]);

    // Operations should work normally
    // Result is Array (not DynArray)
    let c = ops::add(&a, &b).expect("add should work");

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0f32, 7.0, 9.0]);
}
