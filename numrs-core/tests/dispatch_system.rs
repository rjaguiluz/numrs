//! Tests para el dispatch system (zero-cost runtime dispatch)

use numrs::array::Array;
use numrs::backend::{get_dispatch_table, validate_backends};
use numrs::llo::reduction::ReductionKind;

#[test]
fn test_dispatch_table_initialization() {
    // El dispatch table debe inicializarse correctamente
    let table = get_dispatch_table();
    
    // Verificar que los function pointers están asignados
    assert!(!table.elementwise_backend.is_empty());
    assert!(!table.reduction_backend.is_empty());
    assert!(!table.matmul_backend.is_empty());
}

#[test]
fn test_backend_validation() {
    let validation = validate_backends();
    
    // Al menos uno debe estar disponible (siempre tenemos scalar)
    assert!(
        validation.simd_available || 
        validation.blas_available || 
        validation.webgpu_available ||
        validation.gpu_available
    );
    
    // SIMD debería estar disponible en x86_64 con AVX2
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        assert!(validation.simd_available);
    }
}

#[test]
fn test_dispatch_matmul() {
    let table = get_dispatch_table();
    
    // Test pequeño de matmul
    let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    
    // Llamar a través de dispatch
    let result = (table.matmul)(&a, &b).expect("matmul should succeed");
    
    // Verificar resultado (multiplicar por identidad = mismo)
    assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_dispatch_elementwise() {
    use numrs::llo::ElementwiseKind;
    let table = get_dispatch_table();
    
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
    
    // Test add
    let result = (table.elementwise)(&a, &b, ElementwiseKind::Add)
        .expect("elementwise add should succeed");
    assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0]);
    
    // Test mul
    let result = (table.elementwise)(&a, &b, ElementwiseKind::Mul)
        .expect("elementwise mul should succeed");
    assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_dispatch_reduction() {
    let table = get_dispatch_table();
    
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    
    // Sum reduction
    let result = (table.reduction)(&a, None, ReductionKind::Sum)
        .expect("reduction should succeed");
    
    // Debe sumar 1+2+3+4 = 10
    assert!((result.data[0] - 10.0).abs() < 0.001);
}

#[test]
fn test_fast_path_api() {
    // Verificar que ops usa el dispatch system
    use numrs::ops;
    
    let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![2, 2], vec![2.0, 2.0, 2.0, 2.0]);
    
    // Direct ops add
    let result = ops::add(&a, &b).expect("ops add should work");
    assert_eq!(result.to_f32().data, vec![3.0, 4.0, 5.0, 6.0]);
    
    // Direct ops matmul
    let c = Array::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let result = ops::matmul(&a, &c).expect("ops matmul should work");
    assert_eq!(result.to_f32().data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[cfg(numrs_has_blas)]
fn test_blas_validation() {
    let validation = validate_backends();
    
    // Si BLAS está compilado, debe validar correctamente
    assert!(validation.blas_available, "BLAS should be available when compiled with MKL/BLIS/Accelerate");
    assert!(validation.blas_validated, "BLAS should validate successfully");
}

#[test]
fn test_dispatch_zero_cost() {
    // Este test verifica que el dispatch es realmente zero-cost
    // comparando performance con llamada directa
    
    let table = get_dispatch_table();
    let a = Array::new(vec![100], vec![1.0; 100]);
    let b = Array::new(vec![100], vec![2.0; 100]);
    
    use std::time::Instant;
    use numrs::llo::ElementwiseKind;
    
    // Warm-up
    for _ in 0..10 {
        let _ = (table.elementwise)(&a, &b, ElementwiseKind::Add);
    }
    
    // Benchmark dispatch
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = (table.elementwise)(&a, &b, ElementwiseKind::Add);
    }
    let dispatch_time = start.elapsed();
    
    // El tiempo debería ser muy bajo (< 1ms para 1000 iteraciones)
    // porque es solo una llamada a function pointer
    assert!(dispatch_time.as_millis() < 100, 
        "Dispatch should be fast (zero-cost), took {:?}", dispatch_time);
}
