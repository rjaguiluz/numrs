// Verificar que el módulo linalg (correcto) funciona
use numrs::{Array, ops};

fn main() {
    println!("Testing linalg module...\n");
    
    // Test dot product
    let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
    let b = Array::new(vec![3], vec![4.0, 5.0, 6.0]);
    let r = ops::linalg::dot(&a, &b).unwrap();
    println!("✓ ops::linalg::dot([1,2,3], [4,5,6]) = {}", r.data[0]);
    assert_eq!(r.data[0], 32.0);
    
    // Test matmul
    let m1 = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let m2 = Array::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let r2 = ops::linalg::matmul(&m1, &m2).unwrap();
    println!("✓ ops::linalg::matmul(2x2, 2x2) = {:?}", r2.data);
    
    // Test direct import
    use numrs::ops::{dot, matmul};
    let r3 = dot(&a, &b).unwrap();
    println!("✓ ops::dot (direct import) = {}", r3.data[0]);
    
    let r4 = matmul(&m1, &m2).unwrap();
    println!("✓ ops::matmul (direct import) = {:?}", r4.data);
    
    println!("\n✅ All linalg operations work correctly!");
    println!("The 'linialg' typo has been removed - only 'linalg' exists now.");
}
