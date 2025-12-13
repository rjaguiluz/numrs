//! Tests para Array<T> genérico
//!
//! Estos tests verifican que Array funciona con diferentes tipos:
//! - f32, f64, i32, i8, u8, bool

use numrs::{Array, DType, DTypeValue};

#[test]
fn test_array_f32() {
    let a = Array::new(vec![3], vec![1.0_f32, 2.0, 3.0]);
    assert_eq!(a.dtype, DType::F32);
    assert_eq!(a.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_array_f64() {
    let a = Array::new(vec![3], vec![1.0_f64, 2.0, 3.0]);
    assert_eq!(a.dtype, DType::F64);
    assert_eq!(a.data, vec![1.0_f64, 2.0, 3.0]);
}

#[test]
fn test_array_i32() {
    let a = Array::new(vec![3], vec![1_i32, 2, 3]);
    assert_eq!(a.dtype, DType::I32);
    assert_eq!(a.data, vec![1, 2, 3]);
}

#[test]
fn test_array_i8() {
    let a = Array::new(vec![4], vec![1_i8, -2, 3, -4]);
    assert_eq!(a.dtype, DType::I8);
    assert_eq!(a.data, vec![1, -2, 3, -4]);
}

#[test]
fn test_array_u8() {
    let a = Array::new(vec![4], vec![1_u8, 2, 3, 255]);
    assert_eq!(a.dtype, DType::U8);
    assert_eq!(a.data, vec![1, 2, 3, 255]);
}

#[test]
fn test_array_bool() {
    let a = Array::new(vec![4], vec![true, false, true, false]);
    assert_eq!(a.dtype, DType::Bool);
    assert_eq!(a.data, vec![true, false, true, false]);
}

#[test]
fn test_zeros_different_types() {
    let f32_zeros = Array::<f32>::zeros(vec![3]);
    assert_eq!(f32_zeros.dtype, DType::F32);
    assert_eq!(f32_zeros.data, vec![0.0; 3]);

    let f64_zeros = Array::<f64>::zeros(vec![3]);
    assert_eq!(f64_zeros.dtype, DType::F64);
    assert_eq!(f64_zeros.data, vec![0.0_f64; 3]);

    let i32_zeros = Array::<i32>::zeros(vec![3]);
    assert_eq!(i32_zeros.dtype, DType::I32);
    assert_eq!(i32_zeros.data, vec![0_i32; 3]);

    let bool_zeros = Array::<bool>::zeros(vec![3]);
    assert_eq!(bool_zeros.dtype, DType::Bool);
    assert_eq!(bool_zeros.data, vec![false; 3]);
}

#[test]
fn test_ones_different_types() {
    let f32_ones = Array::<f32>::ones(vec![3]);
    assert_eq!(f32_ones.dtype, DType::F32);
    assert_eq!(f32_ones.data, vec![1.0; 3]);

    let f64_ones = Array::<f64>::ones(vec![3]);
    assert_eq!(f64_ones.dtype, DType::F64);
    assert_eq!(f64_ones.data, vec![1.0_f64; 3]);

    let i32_ones = Array::<i32>::ones(vec![3]);
    assert_eq!(i32_ones.dtype, DType::I32);
    assert_eq!(i32_ones.data, vec![1_i32; 3]);
}

#[test]
fn test_dtype_value_conversions() {
    // Test to_f32
    assert_eq!(5_i32.to_f32(), 5.0);
    assert_eq!(3.14_f64.to_f32(), 3.14_f32);
    assert_eq!(true.to_f32(), 1.0);
    assert_eq!(false.to_f32(), 0.0);

    // Test from_f32
    assert_eq!(i32::from_f32(5.7), 5);
    assert_eq!(bool::from_f32(0.0), false);
    assert_eq!(bool::from_f32(1.0), true);
    assert_eq!(u8::from_f32(255.0), 255);
}

#[test]
fn test_dtype_trait_methods() {
    assert_eq!(f32::dtype(), DType::F32);
    assert_eq!(f64::dtype(), DType::F64);
    assert_eq!(i32::dtype(), DType::I32);
    assert_eq!(i8::dtype(), DType::I8);
    assert_eq!(u8::dtype(), DType::U8);
    assert_eq!(bool::dtype(), DType::Bool);
}
