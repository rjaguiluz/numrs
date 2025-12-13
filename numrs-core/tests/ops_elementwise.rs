use anyhow::Result;
use numrs::ops::{add, div, exp, mul, sin, sub};
use numrs::Array;

#[test]
fn test_basic_arithmetic() -> Result<()> {
    let a = Array::new(vec![2], vec![10.0, 20.0]);
    let b = Array::new(vec![2], vec![2.0, 5.0]);

    let s = add(&a, &b)?;
    assert_eq!(s.data, vec![12.0, 25.0]);

    let d = sub(&a, &b)?;
    assert_eq!(d.data, vec![8.0, 15.0]);

    let m = mul(&a, &b)?;
    assert_eq!(m.data, vec![20.0, 100.0]);

    let q = div(&a, &b)?;
    assert_eq!(q.data, vec![5.0, 4.0]);

    Ok(())
}

#[test]
fn test_unary_ops() -> Result<()> {
    let a = Array::new(vec![2], vec![0.0, std::f32::consts::PI / 2.0]);

    let s = sin(&a)?;
    assert!((s.data[0] - 0.0).abs() < 1e-5);
    assert!((s.data[1] - 1.0).abs() < 1e-5);

    let e = Array::new(vec![2], vec![0.0, 1.0]);
    let exp_res = exp(&e)?;
    assert!((exp_res.data[0] - 1.0).abs() < 1e-5);
    assert!((exp_res.data[1] - std::f32::consts::E).abs() < 1e-5);

    Ok(())
}
