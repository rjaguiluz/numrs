//! Tests básicos para el sistema de dtypes (Fase 1)
//!
//! Estos tests verifican:
//! - Array tiene campo dtype
//! - dtype se infiere del tipo genérico
//! - zeros() funciona con diferentes tipos
//! - Backend capabilities están definidas correctamente

use numrs::{Array, DType};
use numrs::backend::{
    BLAS_CAPABILITIES, SIMD_CAPABILITIES, SCALAR_CAPABILITIES,
    WEBGPU_CAPABILITIES,
};

#[test]
fn test_array_has_dtype() {
    let a = Array::new(vec![3], vec![1.0_f32, 2.0, 3.0]);
    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(a.shape, vec![3]);
    assert_eq!(a.data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_zeros_defaults_to_f32() {
    let a = Array::<f32>::zeros(vec![2, 3]);
    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(a.shape, vec![2, 3]);
    assert_eq!(a.data.len(), 6);
    assert_eq!(a.data, vec![0.0; 6]);
}

#[test]
fn test_zeros_f64() {
    let a = Array::<f64>::zeros(vec![4]);
    assert_eq!(a.dtype(), DType::F64);
    assert_eq!(a.data, vec![0.0_f64; 4]);
}

#[test]
fn test_zeros_i32() {
    let a = Array::<i32>::zeros(vec![3]);
    assert_eq!(a.dtype(), DType::I32);
    assert_eq!(a.data, vec![0_i32; 3]);
}

#[test]
fn test_ones_has_dtype() {
    let a = Array::<f32>::ones(vec![3]);
    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(a.data, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_dtype_properties() {
    // Tamaños
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F64.size_bytes(), 8);
    assert_eq!(DType::U8.size_bytes(), 1);
    assert_eq!(DType::I8.size_bytes(), 1);
    assert_eq!(DType::I32.size_bytes(), 4);
    assert_eq!(DType::Bool.size_bytes(), 1);

    // Float types
    assert!(DType::F16.is_float());
    assert!(DType::BF16.is_float());
    assert!(DType::F32.is_float());
    assert!(DType::F64.is_float());
    assert!(!DType::I32.is_float());

    // Integer types
    assert!(DType::I32.is_int());
    assert!(DType::I8.is_int());
    assert!(DType::U8.is_int());
    assert!(!DType::F32.is_int());

    // Bool
    assert!(DType::Bool.is_bool());
    assert!(!DType::F32.is_bool());
}

#[test]
fn test_dtype_display() {
    assert_eq!(format!("{}", DType::F16), "f16");
    assert_eq!(format!("{}", DType::BF16), "bf16");
    assert_eq!(format!("{}", DType::F32), "f32");
    assert_eq!(format!("{}", DType::F64), "f64");
    assert_eq!(format!("{}", DType::U8), "u8");
    assert_eq!(format!("{}", DType::I8), "i8");
    assert_eq!(format!("{}", DType::I32), "i32");
    assert_eq!(format!("{}", DType::Bool), "bool");
}

// ============================================================================
// Backend Capabilities Tests
// ============================================================================

#[test]
fn test_blas_capabilities() {
    assert_eq!(BLAS_CAPABILITIES.name, "blas");
    assert!(BLAS_CAPABILITIES.supports(DType::F32));
    assert!(BLAS_CAPABILITIES.supports(DType::F64));
    assert!(!BLAS_CAPABILITIES.supports(DType::I32));
    assert!(!BLAS_CAPABILITIES.supports(DType::U8));
    assert!(!BLAS_CAPABILITIES.supports(DType::Bool));
}

#[test]
fn test_simd_capabilities() {
    assert_eq!(SIMD_CAPABILITIES.name, "cpu-simd");
    assert!(SIMD_CAPABILITIES.supports(DType::F16));
    assert!(SIMD_CAPABILITIES.supports(DType::F32));
    assert!(SIMD_CAPABILITIES.supports(DType::F64));
    assert!(SIMD_CAPABILITIES.supports(DType::I32));
    assert!(!SIMD_CAPABILITIES.supports(DType::BF16));  // BF16 necesita AVX512_BF16
    assert!(!SIMD_CAPABILITIES.supports(DType::U8));
    assert!(!SIMD_CAPABILITIES.supports(DType::Bool));
}

#[test]
fn test_scalar_capabilities() {
    assert_eq!(SCALAR_CAPABILITIES.name, "cpu-scalar");
    // Scalar debe soportar TODOS los tipos
    assert!(SCALAR_CAPABILITIES.supports(DType::F16));
    assert!(SCALAR_CAPABILITIES.supports(DType::BF16));
    assert!(SCALAR_CAPABILITIES.supports(DType::F32));
    assert!(SCALAR_CAPABILITIES.supports(DType::F64));
    assert!(SCALAR_CAPABILITIES.supports(DType::U8));
    assert!(SCALAR_CAPABILITIES.supports(DType::I8));
    assert!(SCALAR_CAPABILITIES.supports(DType::I32));
    assert!(SCALAR_CAPABILITIES.supports(DType::Bool));
}

#[test]
fn test_webgpu_capabilities() {
    assert_eq!(WEBGPU_CAPABILITIES.name, "webgpu");
    assert!(WEBGPU_CAPABILITIES.supports(DType::F16));
    assert!(WEBGPU_CAPABILITIES.supports(DType::F32));
    assert!(!WEBGPU_CAPABILITIES.supports(DType::F64));
    assert!(!WEBGPU_CAPABILITIES.supports(DType::I32));
}

#[test]
fn test_capabilities_supported_types_str() {
    let blas_str = BLAS_CAPABILITIES.supported_types_str();
    assert!(blas_str.contains("f32"));
    assert!(blas_str.contains("f64"));
    
    let scalar_str = SCALAR_CAPABILITIES.supported_types_str();
    assert!(scalar_str.contains("f32"));
    assert!(scalar_str.contains("bool"));
}

// ============================================================================
// Integration: Operations still work with dtype field
// ============================================================================

#[test]
fn test_ops_still_work_with_dtype() {
    use numrs::ops;
    
    let a = Array::new(vec![3], vec![1.0f32, 2.0, 3.0]);
    let b = Array::new(vec![3], vec![2.0f32, 2.0, 2.0]);
    
    // Verificar que tienen dtype
    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(b.dtype(), DType::F32);
    
    // Operaciones deben seguir funcionando
    let c = ops::add(&a, &b).expect("add should work");
    assert_eq!(c.dtype(), DType::F32);
    assert_eq!(c.to_f32().data, vec![3.0, 4.0, 5.0]);
    
    let d = ops::mul(&a, &b).expect("mul should work");
    assert_eq!(d.dtype(), DType::F32);
    assert_eq!(d.to_f32().data, vec![2.0, 4.0, 6.0]);
    
    let s = ops::sum(&a, None).expect("sum should work");
    assert_eq!(s.dtype(), DType::F32);
    assert_eq!(s.to_f32().data, vec![6.0]);
}

#[test]
fn test_matmul_with_dtype() {
    use numrs::ops;
    
    let a = Array::new(vec![2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0]);
    
    assert_eq!(a.dtype(), DType::F32);
    assert_eq!(b.dtype(), DType::F32);
    
    let c = ops::matmul(&a, &b).expect("matmul should work");
    assert_eq!(c.dtype(), DType::F32);
    assert_eq!(c.to_f32().data, vec![1.0, 2.0, 3.0, 4.0]);
}
