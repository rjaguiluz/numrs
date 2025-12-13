//! Test simple de Linear

use numrs::{Array, Tensor, Module, Linear};
use anyhow::Result;

fn main() -> Result<()> {
    println!("Test de Linear:\n");
    
    // Linear(2, 8): 2 inputs → 8 outputs
    let linear = Linear::new(2, 8)?;
    
    // Debug: Ver parámetros
    let params = linear.parameters();
    println!("Weight shape: {:?}", params[0].borrow().shape());
    println!("Bias shape: {:?}", params[1].borrow().shape());
    
    // Input: batch_size=1, in_features=2
    let input = Tensor::new(Array::new(vec![1, 2], vec![0.5, 0.3]), false);
    
    println!("\nInput shape: {:?}", input.shape());
    println!("Expected output shape: [1, 8]\n");
    
    // Test transpose manualmente
    let w = params[0].borrow().clone();
    println!("Weight original shape: {:?}", w.shape());
    let w_t = w.transpose()?;
    println!("Weight transposed shape: {:?}", w_t.shape());
    
    // Forward
    println!("\nExecuting forward...");
    let output = linear.forward(&input)?;
    
    println!("\nOutput shape: {:?}", output.shape());
    println!("Output values (first 5): {:?}", &output.values()[..5.min(output.values().len())]);
    
    println!("\n✓ Test passed!");
    
    Ok(())
}
