use crate::array::{Array, DTypeValue};
use crate::llo::reduction::ReductionKind;
use anyhow::Result;

/// Reduce-mean over whole array (axis=None) or specific axis
#[inline(always)]
pub fn mean<T: DTypeValue>(a: &Array<T>, axis: Option<usize>) -> Result<Array> {
    let dyn_result = crate::ops::promotion_wrappers::reduction_promoted(a, axis, ReductionKind::Mean, "mean")?;
    dyn_result.into_typed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_all() {
        let a = Array::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = mean(&a, None).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert!((result.data[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_negative() {
        let a = Array::new(vec![6], vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]);
        let result = mean(&a, None).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert!(result.data[0].abs() < 1e-5); // Mean should be 0
    }

    #[test]
    fn test_mean_single() {
        let a = Array::new(vec![1], vec![42.0]);
        let result = mean(&a, None).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert!((result.data[0] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_axis0_2d() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        // mean(axis=0) = [2.5, 3.5, 4.5]
        let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = mean(&a, Some(0)).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert!((result.data[0] - 2.5).abs() < 1e-5);
        assert!((result.data[1] - 3.5).abs() < 1e-5);
        assert!((result.data[2] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_mean_axis1_2d() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        // mean(axis=1) = [2.0, 5.0]
        let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = mean(&a, Some(1)).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert!((result.data[0] - 2.0).abs() < 1e-5);
        assert!((result.data[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_axis0_3d() {
        // shape [2, 2, 2]
        // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        // mean(axis=0) reduces first dim -> [2, 2]
        // result: [[3, 4], [5, 6]]
        let a = Array::new(
            vec![2, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = mean(&a, Some(0)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert!((result.data[0] - 3.0).abs() < 1e-5);
        assert!((result.data[1] - 4.0).abs() < 1e-5);
        assert!((result.data[2] - 5.0).abs() < 1e-5);
        assert!((result.data[3] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_axis1_3d() {
        // shape [2, 2, 2]
        // mean(axis=1) reduces middle dim -> [2, 2]
        let a = Array::new(
            vec![2, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = mean(&a, Some(1)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert!((result.data[0] - 2.0).abs() < 1e-5);
        assert!((result.data[1] - 3.0).abs() < 1e-5);
        assert!((result.data[2] - 6.0).abs() < 1e-5);
        assert!((result.data[3] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_axis2_3d() {
        // shape [2, 2, 2]
        // mean(axis=2) reduces last dim -> [2, 2]
        let a = Array::new(
            vec![2, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let result = mean(&a, Some(2)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert!((result.data[0] - 1.5).abs() < 1e-5);
        assert!((result.data[1] - 3.5).abs() < 1e-5);
        assert!((result.data[2] - 5.5).abs() < 1e-5);
        assert!((result.data[3] - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_mean_axis_out_of_bounds() {
        let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = mean(&a, Some(2));
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_1d_axis0() {
        // Reducing 1D array along axis 0 should give scalar
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let result = mean(&a, Some(0)).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert!((result.data[0] - 2.5).abs() < 1e-5);
    }
}

