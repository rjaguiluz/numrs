//! Tests modernos del dispatch system (zero-cost runtime selection)
//! 
//! Este archivo reemplaza los tests legacy de kernel selection con
//! tests basados en el nuevo dispatch system.

use numrs::array::Array;
use numrs::backend::{get_dispatch_table, validate_backends};
use numrs::ops::fast;

#[test]
fn dispatch_system_initialization() {
    // Verificar que el dispatch system se inicializa correctamente
    let table = get_dispatch_table();
    let validation = validate_backends();
    
    println!("📊 Dispatch Table:");
    println!("  Elementwise: {}", table.elementwise_backend);
    println!("  Reduction: {}", table.reduction_backend);
    println!("  MatMul: {}", table.matmul_backend);
    
    // Al menos un backend debe estar disponible
    assert!(
        validation.simd_available || 
        validation.blas_available || 
        validation.webgpu_available ||
        validation.gpu_available,
        "Al menos un backend debe estar disponible"
    );
    
    // Los backend strings no deben estar vacíos
    assert!(!table.elementwise_backend.is_empty());
    assert!(!table.reduction_backend.is_empty());
    assert!(!table.matmul_backend.is_empty());
}

#[test]
fn fast_path_operations() {
    // Test end-to-end del fast-path (usa dispatch automáticamente)
    
    // Test 1: Elementwise add
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
    
    let result = fast::add(&a, &b).expect("add failed");
    assert_eq!(result.to_f32().data, vec![2.0, 3.0, 4.0, 5.0]);
    
    // Test 2: Reduction sum
    let xs = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let sum_result = fast::sum(&xs, None).expect("sum failed");
    assert_eq!(sum_result.to_f32().data[0], 10.0);
    
    // Test 3: MatMul pequeño
    let m_a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let m_b = Array::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]); // Identity
    
    let mm_result = fast::matmul(&m_a, &m_b).expect("matmul failed");
    assert_eq!(mm_result.to_f32().data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn dispatch_is_consistent() {
    // Verificar que el dispatch siempre selecciona el mismo backend
    // (no hay variabilidad runtime innecesaria)
    
    let table1 = get_dispatch_table();
    let table2 = get_dispatch_table();
    
    assert_eq!(table1.elementwise_backend, table2.elementwise_backend);
    assert_eq!(table1.reduction_backend, table2.reduction_backend);
    assert_eq!(table1.matmul_backend, table2.matmul_backend);
}

#[test]
fn hot_path_no_overhead() {
    // Test que el hot-path es realmente zero-cost
    // (llamadas repetidas deben ser igual de rápidas)
    
    let a = Array::new(vec![100], vec![1.0; 100]);
    let b = Array::new(vec![100], vec![2.0; 100]);
    
    use std::time::Instant;
    
    // Warm-up
    for _ in 0..10 {
        let _ = fast::add(&a, &b);
    }
    
    // Medir muchas operaciones
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = fast::add(&a, &b).unwrap();
    }
    let elapsed = start.elapsed();
    
    // Debería ser muy rápido (< 10ms para 1000 ops en arrays pequeños)
    println!("1000 operaciones tomaron: {:?}", elapsed);
    assert!(elapsed.as_millis() < 100, "Dispatch tiene demasiado overhead");
}

#[test]
fn backend_validation_functional() {
    // Verificar que la validación es funcional (no solo cfg checks)
    let validation = validate_backends();
    
    // Si BLAS está disponible, debe estar validado funcionalmente
    if validation.blas_available {
        assert!(
            validation.blas_validated,
            "BLAS disponible pero validación funcional falló"
        );
    }
    
    // Si SIMD está disponible, debe estar validado
    if validation.simd_available {
        assert!(
            validation.simd_validated,
            "SIMD disponible pero validación funcional falló"
        );
    }
}

/// Test legacy mantenido por compatibilidad (deprecated)
#[test]
#[allow(deprecated)]
#[cfg(feature = "__test_legacy_selection")]
fn legacy_selection_compatibility() {
    use numrs::backend::selection::{detect_capabilities, KernelSelectionContext};
    use numrs::backend::kernels::*;
    use numrs::method_select;
    
    let caps = detect_capabilities();
    let mut ctx_sum: KernelSelectionContext<Vec<f32>, Vec<f32>> = KernelSelectionContext::new(&caps);
    let xs = vec![1.0f32, 2.0, 3.0, 4.0];

    method_select! {
        CONTEXT ctx_sum;
        WHEN (xs.len() <= 4) THEN (sum_scalar),
        WHEN (ctx_sum.caps.has_avx2) THEN (sum_simd),
        WHEN (ctx_sum.caps.has_blas) THEN (sum_blas),
        ELSE (sum_scalar_fallback)
    }

    let out = (ctx_sum.selected)(&xs);
    assert_eq!(out[0], 10.0f32);
}
