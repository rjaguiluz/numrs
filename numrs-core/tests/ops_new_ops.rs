use numrs::Array;
use numrs::ops::{pow, abs, exp};

fn approx_eq_vec(a: &[f32], b: &[f32], tol: f32) -> bool {
    if a.len() != b.len() { return false; }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > tol { return false; }
    }
    true
}

#[test]
fn test_abs_pow_exp_basic() {
    // abs
    let a = Array::new(vec![3], vec![-1.0, 0.0, 2.0]);
    let out_abs = abs(&a).expect("abs failed");
    assert_eq!(out_abs.to_f32().data, vec![1.0, 0.0, 2.0]);

    // pow
    let a1 = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
    let b1 = Array::new(vec![3], vec![2.0, 3.0, 0.5]);
    let out_pow = pow(&a1, &b1).expect("pow failed");
    let expected_pow = vec![1.0_f32, 8.0_f32, 3.0_f32.powf(0.5)];
    assert!(approx_eq_vec(&out_pow.to_f32().data, &expected_pow, 1e-6));

    // exp
    let e_in = Array::new(vec![2], vec![0.0, 1.0]);
    let out_exp = exp(&e_in).expect("exp failed");
    let expected_exp = vec![1.0_f32, std::f32::consts::E];
    assert!(approx_eq_vec(&out_exp.to_f32().data, &expected_exp, 1e-6));
}
