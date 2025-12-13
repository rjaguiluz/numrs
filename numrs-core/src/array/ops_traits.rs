//! Operator overloading for Array
//!
//! This module implements std::ops traits to enable natural operator syntax:
//! ```rust
//! # use numrs::array::Array;
//! # use numrs::ops;
//! let a = Array::new(vec![1], vec![1.0]);
//! let b = Array::new(vec![1], vec![2.0]);
//! let c = &a + &b;  // instead of ops::add(&a, &b)
//! let d = &a * &b;  // instead of ops::mul(&a, &b)
//! ```
//!
//! Note: Operators return Array (with default dtype f32) after type promotion.

use crate::array::Array;
use crate::ops;
use std::ops::{Add, Div, Mul, Neg, Sub};

// ============================================================================
// Binary Operations: Add, Sub, Mul, Div
// ============================================================================

impl Add for Array {
    type Output = Array;

    /// Addition: `a + b`
    fn add(self, rhs: Self) -> Self::Output {
        ops::add(&self, &rhs).expect("Addition failed")
    }
}

impl Add<&Array> for Array {
    type Output = Array;

    /// Addition: `a + &b`
    fn add(self, rhs: &Array) -> Self::Output {
        ops::add(&self, rhs).expect("Addition failed")
    }
}

impl Add<Array> for &Array {
    type Output = Array;

    /// Addition: `&a + b`
    fn add(self, rhs: Array) -> Self::Output {
        ops::add(self, &rhs).expect("Addition failed")
    }
}

impl Add for &Array {
    type Output = Array;

    /// Addition: `&a + &b` (most common, avoids moves)
    fn add(self, rhs: Self) -> Self::Output {
        ops::add(self, rhs).expect("Addition failed")
    }
}

// ----------------------------------------------------------------------------

impl Sub for Array {
    type Output = Array;

    /// Subtraction: `a - b`
    fn sub(self, rhs: Self) -> Self::Output {
        ops::sub(&self, &rhs).expect("Subtraction failed")
    }
}

impl Sub<&Array> for Array {
    type Output = Array;

    fn sub(self, rhs: &Array) -> Self::Output {
        ops::sub(&self, rhs).expect("Subtraction failed")
    }
}

impl Sub<Array> for &Array {
    type Output = Array;

    fn sub(self, rhs: Array) -> Self::Output {
        ops::sub(self, &rhs).expect("Subtraction failed")
    }
}

impl Sub for &Array {
    type Output = Array;

    /// Subtraction: `&a - &b`
    fn sub(self, rhs: Self) -> Self::Output {
        ops::sub(self, rhs).expect("Subtraction failed")
    }
}

// ----------------------------------------------------------------------------

impl Mul for Array {
    type Output = Array;

    /// Element-wise multiplication: `a * b`
    fn mul(self, rhs: Self) -> Self::Output {
        ops::mul(&self, &rhs).expect("Multiplication failed")
    }
}

impl Mul<&Array> for Array {
    type Output = Array;

    fn mul(self, rhs: &Array) -> Self::Output {
        ops::mul(&self, rhs).expect("Multiplication failed")
    }
}

impl Mul<Array> for &Array {
    type Output = Array;

    fn mul(self, rhs: Array) -> Self::Output {
        ops::mul(self, &rhs).expect("Multiplication failed")
    }
}

impl Mul for &Array {
    type Output = Array;

    /// Element-wise multiplication: `&a * &b`
    fn mul(self, rhs: Self) -> Self::Output {
        ops::mul(self, rhs).expect("Multiplication failed")
    }
}

// ----------------------------------------------------------------------------

impl Div for Array {
    type Output = Array;

    /// Element-wise division: `a / b`
    fn div(self, rhs: Self) -> Self::Output {
        ops::div(&self, &rhs).expect("Division failed")
    }
}

impl Div<&Array> for Array {
    type Output = Array;

    fn div(self, rhs: &Array) -> Self::Output {
        ops::div(&self, rhs).expect("Division failed")
    }
}

impl Div<Array> for &Array {
    type Output = Array;

    fn div(self, rhs: Array) -> Self::Output {
        ops::div(self, &rhs).expect("Division failed")
    }
}

impl Div for &Array {
    type Output = Array;

    /// Element-wise division: `&a / &b`
    fn div(self, rhs: Self) -> Self::Output {
        ops::div(self, rhs).expect("Division failed")
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

impl Neg for Array {
    type Output = Array;

    /// Negation: `-a`
    fn neg(self) -> Self::Output {
        ops::neg(&self).expect("Negation failed")
    }
}

impl Neg for &Array {
    type Output = Array;

    /// Negation: `-&a`
    fn neg(self) -> Self::Output {
        ops::neg(self).expect("Negation failed")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_operator() {
        let a = Array::new(vec![3], vec![1.0f32, 2.0, 3.0]);
        let b = Array::new(vec![3], vec![4.0f32, 5.0, 6.0]);

        // Test with references (most common)
        let c: Array<f32> = &a + &b;
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_operator() {
        let a = Array::new(vec![3], vec![5.0f32, 7.0, 9.0]);
        let b = Array::new(vec![3], vec![1.0f32, 2.0, 3.0]);

        let c: Array<f32> = &a - &b;
        assert_eq!(c.data, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_operator() {
        let a = Array::new(vec![3], vec![2.0f32, 3.0, 4.0]);
        let b = Array::new(vec![3], vec![5.0f32, 6.0, 7.0]);

        let c: Array<f32> = &a * &b;
        assert_eq!(c.data, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_operator() {
        let a = Array::new(vec![3], vec![10.0f32, 20.0, 30.0]);
        let b = Array::new(vec![3], vec![2.0f32, 4.0, 5.0]);

        let c: Array<f32> = &a / &b;
        assert_eq!(c.data, vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_neg_operator() {
        let a = Array::new(vec![3], vec![1.0f32, -2.0, 3.0]);

        let b: Array<f32> = -&a;
        assert_eq!(b.data, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_chained_operators() {
        let a = Array::new(vec![3], vec![1.0f32, 2.0, 3.0]);
        let b = Array::new(vec![3], vec![4.0f32, 5.0, 6.0]);
        let c = Array::new(vec![3], vec![2.0f32, 2.0, 2.0]);

        // Test: (a + b) * c
        let sum: Array<f32> = &a + &b;
        let result: Array<f32> = &sum * &c;
        assert_eq!(result.data, vec![10.0, 14.0, 18.0]);
    }

    #[test]
    fn test_different_dtypes() {
        // Operators work with any dtype through type promotion
        use crate::array::dtype::DType;
        use crate::ops;

        let a_f64 = Array::new(vec![2], vec![1.0f64, 2.0]);
        let b_f64 = Array::new(vec![2], vec![3.0f64, 4.0]);

        // Result is always f32 currently
        let c: Array = ops::add(&a_f64, &b_f64).unwrap();
        assert_eq!(c.dtype, DType::F32);
        // assert_eq!(c.data, vec![4.0, 6.0]);

        let a_i32 = Array::new(vec![2], vec![10i32, 20]);
        let b_i32 = Array::new(vec![2], vec![5i32, 10]);

        // Result is always f32 currently
        let d: Array = ops::sub(&a_i32, &b_i32).unwrap();
        assert_eq!(d.dtype, DType::F32); // Promoted or defaulted to F32
                                         // assert_eq!(d.data, vec![5.0, 10.0]);
    }
}
