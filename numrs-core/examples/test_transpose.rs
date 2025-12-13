//! Test simple de transpose

use numrs::{Array, Tensor};
use anyhow::Result;

fn main() -> Result<()> {
    println!("Test de transpose:\n");
    
    // Crear matriz [2, 3]
    let w = Tensor::new(
        Array::new(vec![2, 3], vec![
            1.0, 2.0, 3.0,  // fila 1
            4.0, 5.0, 6.0,  // fila 2
        ]),
        false
    );
    
    println!("Weight original: shape = {:?}", w.shape());
    println!("Values: {:?}", w.values());
    
    // Transpose
    let w_t = w.transpose()?;
    
    println!("\nWeight transposed: shape = {:?}", w_t.shape());
    println!("Values: {:?}", w_t.values());
    
    // Debería ser [3, 2]:
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]
    
    Ok(())
}
