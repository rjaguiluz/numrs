//! Ejemplo básico de dtypes (Fase 1)
//!
//! Demuestra:
//! - Array tiene campo dtype
//! - Backend capabilities pueden ser consultadas
//! - Operaciones funcionan igual que antes

use numrs::{Array, DType, ops};
use numrs::backend::{BLAS_CAPABILITIES, SIMD_CAPABILITIES, SCALAR_CAPABILITIES};

fn main() {
    println!("=== NumRs - DType System (Fase 1) ===\n");
    
    // 1. Arrays tienen dtype
    println!("1. Creating arrays with dtype:");
    let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
    println!("   a = {:?}", a.data);
    println!("   dtype: {}", a.dtype);
    println!("   size: {} bytes per element\n", a.dtype.size_bytes());
    
    // 2. Zeros con diferentes tipos
    println!("2. Creating zeros with different types:");
    let z_f32 = Array::<f32>::zeros(vec![4]);
    println!("   zeros<f32>([4]) -> dtype: {}", z_f32.dtype);
    println!("   data: {:?}\n", z_f32.data);
    
    let z_f64 = Array::<f64>::zeros(vec![4]);
    println!("   zeros<f64>([4]) -> dtype: {}", z_f64.dtype);
    println!("   data: {:?}\n", z_f64.data);
    
    let z_i32 = Array::<i32>::zeros(vec![4]);
    println!("   zeros<i32>([4]) -> dtype: {}", z_i32.dtype);
    println!("   data: {:?}\n", z_i32.data);
    
    // 3. Backend capabilities
    println!("3. Backend capabilities:");
    println!("   BLAS supports:");
    println!("     - F32: {}", BLAS_CAPABILITIES.supports(DType::F32));
    println!("     - F64: {}", BLAS_CAPABILITIES.supports(DType::F64));
    println!("     - I32: {}", BLAS_CAPABILITIES.supports(DType::I32));
    println!("     Types: {}\n", BLAS_CAPABILITIES.supported_types_str());
    
    println!("   SIMD supports:");
    println!("     - F32: {}", SIMD_CAPABILITIES.supports(DType::F32));
    println!("     - I32: {}", SIMD_CAPABILITIES.supports(DType::I32));
    println!("     - U8:  {}", SIMD_CAPABILITIES.supports(DType::U8));
    println!("     Types: {}\n", SIMD_CAPABILITIES.supported_types_str());
    
    println!("   Scalar supports ALL types:");
    println!("     Types: {}\n", SCALAR_CAPABILITIES.supported_types_str());
    
    // 4. Operations work as before
    println!("4. Operations (same API as before):");
    let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
    let b = Array::new(vec![3], vec![2.0, 2.0, 2.0]);
    
    let c = ops::add(&a, &b).expect("add failed");
    println!("   {} + {} = {}", 
        format_array(&a.data), 
        format_array(&b.data), 
        format_array(&c.data)
    );
    println!("   result dtype: {}", c.dtype);
    
    let d = ops::mul(&a, &b).expect("mul failed");
    println!("   {} * {} = {}", 
        format_array(&a.data), 
        format_array(&b.data), 
        format_array(&d.data)
    );
    
    let s = ops::sum(&a, None).expect("sum failed");
    println!("   sum({}) = {}", 
        format_array(&a.data), 
        format_array(&s.data)
    );
    
    println!("\n5. DType properties:");
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64, DType::U8, DType::I8, DType::I32, DType::Bool] {
        println!("   {:5} - {} bytes, float: {:5}, int: {:5}, bool: {:5}",
            dtype,
            dtype.size_bytes(),
            dtype.is_float(),
            dtype.is_int(),
            dtype.is_bool()
        );
    }
    
    println!("\n✅ Fase 1 completada:");
    println!("   - DType enum definido (f16, bf16, f32, f64, u8, i8, i32, bool)");
    println!("   - Array tiene campo dtype (default F32)");
    println!("   - Backend capabilities declaradas");
    println!("   - API existente sigue funcionando sin cambios");
    println!("\n📝 Siguiente: Fase 2 - ArrayData enum (múltiples tipos reales)");
}

fn format_array(data: &[f32]) -> String {
    format!("[{}]", data.iter()
        .map(|x| format!("{:.1}", x))
        .collect::<Vec<_>>()
        .join(", "))
}
