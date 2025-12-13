//! Test de preferencia de backend via dispatch system
//!
//! Este test verifica que el dispatch system respeta las preferencias
//! de BLAS cuando están configuradas.

use numrs::backend::{get_dispatch_table, validate_backends};

#[test]
fn runtime_prefers_blas_when_available() {
    // Con el nuevo dispatch system, la selección de backend se hace
    // automáticamente en base a validación funcional, no variables de entorno.

    let validation = validate_backends();
    let table = get_dispatch_table();

    // Si BLAS está disponible y validado, matmul debe usarlo
    if validation.blas_available && validation.blas_validated {
        assert!(
            table.matmul_backend.contains("blas") || table.matmul_backend.contains("adaptive"),
            "MatMul debe usar BLAS o 'adaptive' cuando está disponible y validado. Got: {}",
            table.matmul_backend
        );
        println!("✅ BLAS disponible y siendo usado por matmul");
    } else {
        // Si BLAS no está disponible, debe usar fallback (SIMD o scalar)
        assert!(
            table.matmul_backend.contains("simd") || table.matmul_backend.contains("scalar"),
            "MatMul debe usar SIMD o scalar cuando BLAS no está disponible. Got: {}",
            table.matmul_backend
        );
        println!(
            "ℹ️ BLAS no disponible, usando fallback: {}",
            table.matmul_backend
        );
    }
}
