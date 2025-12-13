//! Test directo de matmul

use numrs::{Array, Tensor};
use anyhow::Result;

fn main() -> Result<()> {
    println!("Test de matmul directo:\n");
    
    // A: [1, 2]
    let a = Tensor::new(Array::new(vec![1, 2], vec![0.5, 0.3]), false);
    
    // B: [2, 8]
    let b = Tensor::new(Array::new(vec![2, 8], vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,  // fila 1
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  // fila 2
    ]), false);
    
    println!("A shape: {:?}", a.shape());
    println!("B shape: {:?}", b.shape());
    println!("\nExecutando: A @ B...");
    
    let c = a.matmul(&b)?;
    
    println!("\nC shape: {:?}", c.shape());
    println!("C values: {:?}", c.values());
    
    println!("\n✓ Matmul funcionó!");
    
    Ok(())
}
