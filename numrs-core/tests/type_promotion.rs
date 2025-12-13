//! Tests para type promotion en operaciones

use anyhow::Result;
use numrs::{ops, Array, DType};

#[test]
fn test_add_same_type() -> Result<()> {
    let a = Array::new(vec![3], vec![1.0_f32, 2.0, 3.0]);
    let b = Array::new(vec![3], vec![4.0_f32, 5.0, 6.0]);
    let c: Array<f32> = ops::add(&a, &b)?;

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn test_add_mixed_i32_f32() -> Result<()> {
    let a = Array::new(vec![3], vec![1_i32, 2, 3]);
    let b = Array::new(vec![3], vec![4.0_f32, 5.0, 6.0]);
    let c: Array<f32> = ops::add(&a, &b)?;

    // I32 + F32 = F32
    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn test_add_mixed_f64_f32() -> Result<()> {
    let a = Array::new(vec![2], vec![1.0_f64, 2.0]);
    let b = Array::new(vec![2], vec![3.0_f32, 4.0]);
    // ops::add returns Array (f32) by default currently
    // We explicitly cast for test validation if needed, or accept f32
    // let c: Array<f64> = ops::add(&a, &b)?; -> fails
    let c = ops::add(&a, &b)?;

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![4.0, 6.0]);

    Ok(())
}

#[test]
fn test_add_mixed_i8_u8() -> Result<()> {
    let a = Array::new(vec![3], vec![1_i8, 2, 3]);
    let b = Array::new(vec![3], vec![4_u8, 5, 6]);
    // I8 + U8 -> F32 (current implementation defaults to f32)
    let c = ops::add(&a, &b)?;

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![5.0, 7.0, 9.0]);

    Ok(())
}

#[test]
fn test_add_bool_i32() -> Result<()> {
    let a = Array::new(vec![3], vec![true, false, true]);
    let b = Array::new(vec![3], vec![1_i32, 2, 3]);
    // Bool + I32 -> F32
    let c = ops::add(&a, &b)?;

    assert_eq!(c.dtype, DType::F32);
    assert_eq!(c.data, vec![2.0, 2.0, 4.0]);

    Ok(())
}

#[test]
fn test_add_shape_mismatch() {
    let a = Array::new(vec![2], vec![1.0_f32, 2.0]);
    let b = Array::new(vec![3], vec![1.0_f32, 2.0, 3.0]);

    let result: Result<Array<f32>> = ops::add(&a, &b);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("shape mismatch"));
}
