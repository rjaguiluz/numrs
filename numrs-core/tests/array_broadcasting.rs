use anyhow::Result;
use numrs::ops::add;
use numrs::Array;

#[test]
fn test_broadcast_scalar_to_matrix() -> Result<()> {
    // [1] + [2, 2]
    let scalar = Array::new(vec![1], vec![10.0]);
    let matrix = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let result = add(&scalar, &matrix)?;

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![11.0, 12.0, 13.0, 14.0]);

    Ok(())
}

#[test]
fn test_broadcast_row_to_matrix() -> Result<()> {
    // [1, 2] + [2, 2]
    let row = Array::new(vec![1, 2], vec![10.0, 20.0]);
    let matrix = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    // row broadcasts to [[10, 20], [10, 20]]
    let result = add(&row, &matrix)?;

    // Result: [[11, 22], [13, 24]]
    assert_eq!(result.data, vec![11.0, 22.0, 13.0, 24.0]);

    Ok(())
}

#[test]
fn test_broadcast_col_to_matrix() -> Result<()> {
    // [2, 1] + [2, 2]
    let col = Array::new(vec![2, 1], vec![10.0, 20.0]);
    let matrix = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    // col broadcasts to [[10, 10], [20, 20]]
    let result = add(&col, &matrix)?;

    // Result: [[11, 12], [23, 24]]
    assert_eq!(result.data, vec![11.0, 12.0, 23.0, 24.0]);

    Ok(())
}

#[test]
fn test_broadcast_both_dims() -> Result<()> {
    // [2, 1] + [1, 2] -> [2, 2]
    let col = Array::new(vec![2, 1], vec![10.0, 20.0]);
    let row = Array::new(vec![1, 2], vec![1.0, 2.0]);

    // col: [[10, 10], [20, 20]]
    // row: [[1, 2], [1, 2]]
    // sum: [[11, 12], [21, 22]]

    let result = add(&col, &row)?;

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![11.0, 12.0, 21.0, 22.0]);

    Ok(())
}
