use crate::array::{Array, DTypeValue};
use crate::llo::ElementwiseKind;
use anyhow::Result;

/// Unary tan (tangent)
#[inline(always)]
pub fn tan<T: DTypeValue>(a: &Array<T>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::unary_promoted(a, ElementwiseKind::Tan, "tan")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tan_basic() {
        let a = Array::new(vec![3], vec![0.0, std::f32::consts::PI / 4.0, -std::f32::consts::PI / 4.0]);
        let result = tan(&a).unwrap();
        
        assert!((result.data[0] - 0.0).abs() < 1e-6);
        assert!((result.data[1] - 1.0).abs() < 1e-6);
        assert!((result.data[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_tan_2d() {
        let a = Array::new(vec![2, 2], vec![0.0, std::f32::consts::PI / 6.0, std::f32::consts::PI / 3.0, std::f32::consts::PI / 4.0]);
        let result = tan(&a).unwrap();
        
        assert!((result.data[0] - 0.0).abs() < 1e-6);
        assert!((result.data[1] - 0.57735026).abs() < 1e-5); // tan(π/6) ≈ 1/√3
        assert!((result.data[2] - 1.7320508).abs() < 1e-5);  // tan(π/3) ≈ √3
        assert!((result.data[3] - 1.0).abs() < 1e-6);        // tan(π/4) = 1
    }
}
