//! Ejemplo de type promotion en operaciones
//!
//! Demuestra cómo NumRs maneja automáticamente operaciones entre arrays
//! de diferentes tipos, promoviendo al tipo común más apropiado.

use numrs::{Array, DType, ops};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== NumRs - Type Promotion Demo ===\n");
    
    // 1. Operaciones entre el mismo tipo (no hay promoción)
    println!("1. Same type operations (no promotion):");
    let a_f32 = Array::new(vec![3], vec![1.0_f32, 2.0, 3.0]);
    let b_f32 = Array::new(vec![3], vec![4.0_f32, 5.0, 6.0]);
    let c = ops::add(&a_f32, &b_f32)?;
    println!("   f32 + f32 = {:?} (dtype: {})", c.data, c.dtype);
    
    // 2. Int + Float = Float
    println!("\n2. Integer + Float promotion:");
    let a_i32 = Array::new(vec![3], vec![1_i32, 2, 3]);
    let b_f32 = Array::new(vec![3], vec![0.5_f32, 1.5, 2.5]);
    let c = ops::add(&a_i32, &b_f32)?;
    println!("   i32 + f32 = {:?} (dtype: {})", c.data, c.dtype);
    
    // 3. F64 gana sobre F32
    println!("\n3. F64 wins over F32:");
    let a_f64 = Array::new(vec![2], vec![1.0_f64, 2.0]);
    let b_f32 = Array::new(vec![2], vec![3.0_f32, 4.0]);
    let c = ops::mul(&a_f64, &b_f32)?;
    println!("   f64 * f32 = {:?} (dtype: {})", c.data, c.dtype);
    
    // 4. Bool + I32 = I32
    println!("\n4. Bool + Integer:");
    let a_bool = Array::new(vec![4], vec![true, false, true, false]);
    let b_i32 = Array::new(vec![4], vec![10_i32, 20, 30, 40]);
    let c = ops::add(&a_bool, &b_i32)?;
    println!("   bool + i32 = {:?} (dtype: {})", c.data, c.dtype);
    println!("   (true=1.0, false=0.0)");
    
    // 5. I8 + U8 = I32 (para evitar overflow)
    println!("\n5. Mixed sign integers (I8 + U8 = I32 for safety):");
    let a_i8 = Array::new(vec![3], vec![-5_i8, 0, 5]);
    let b_u8 = Array::new(vec![3], vec![10_u8, 20, 30]);
    let c = ops::add(&a_i8, &b_u8)?;
    println!("   i8 + u8 = {:?} (dtype: {})", c.data, c.dtype);
    
    // 6. Diferentes operaciones
    println!("\n6. Different operations with promotion:");
    let a = Array::new(vec![3], vec![10_i32, 20, 30]);
    let b = Array::new(vec![3], vec![2.0_f32, 2.0, 2.0]);
    
    let sum = ops::add(&a, &b)?;
    let product = ops::mul(&a, &b)?;
    let division = ops::div(&a, &b)?;
    let difference = ops::sub(&a, &b)?;
    
    println!("   i32 + f32 = {:?}", sum.data);
    println!("   i32 * f32 = {:?}", product.data);
    println!("   i32 / f32 = {:?}", division.data);
    println!("   i32 - f32 = {:?}", difference.data);
    println!("   All results have dtype: {}", sum.dtype);
    
    // 7. Promotion rules summary
    println!("\n7. Promotion hierarchy:");
    println!("   bool < u8 < i8 < i32 < f16 < bf16 < f32 < f64");
    println!("   Float always wins over integer");
    println!("   Wider types win over narrower types");
    
    Ok(())
}
