// Test para verificar que BLAS está correctamente linkeado de forma estática
// Actualizado para usar el dispatch system (zero-cost runtime dispatch)

use numrs::backend::{get_dispatch_table, validate_backends};

#[test]
fn verify_blas_availability() {
    let validation = validate_backends();

    println!("🔍 Backends validados:");
    println!(
        "  - SIMD: available={} validated={}",
        validation.simd_available, validation.simd_validated
    );
    println!(
        "  - BLAS: available={} validated={}",
        validation.blas_available, validation.blas_validated
    );
    println!(
        "  - WebGPU: available={} validated={}",
        validation.webgpu_available, validation.webgpu_validated
    );
    println!(
        "  - GPU: available={} validated={}",
        validation.gpu_available, validation.gpu_validated
    );

    // Verificar que en plataformas con BLAS, está disponible y validado
    #[cfg(target_os = "macos")]
    {
        assert!(
            validation.blas_available,
            "macOS debe tener Accelerate framework disponible"
        );
        assert!(
            validation.blas_validated,
            "Accelerate debe validar correctamente"
        );
        println!("✅ macOS: Accelerate framework detectado y validado");
    }

    #[cfg(all(feature = "mkl", any(target_arch = "x86_64", target_arch = "x86")))]
    {
        assert!(
            validation.blas_available,
            "MKL debe estar disponible cuando feature 'mkl' está habilitado"
        );
        assert!(validation.blas_validated, "MKL debe validar correctamente");
        println!("✅ Intel MKL (static) detectado y validado");
    }

    #[cfg(feature = "blis")]
    {
        assert!(
            validation.blas_available,
            "BLIS debe estar disponible cuando feature 'blis' está habilitado"
        );
        assert!(validation.blas_validated, "BLIS debe validar correctamente");
        println!("✅ BLIS (static) detectado y validado");
    }
}

#[test]
fn verify_kernel_selection() {
    let table = get_dispatch_table();

    println!("📦 Dispatch table (kernels seleccionados):");
    println!("  - Elementwise: {}", table.elementwise_backend);
    println!("  - Reduction: {}", table.reduction_backend);
    println!("  - MatMul: {}", table.matmul_backend);

    // En plataformas con BLAS, matmul debe usar BLAS
    #[cfg(any(target_os = "macos", feature = "mkl", feature = "blis"))]
    {
        // Al menos matmul debe usar BLAS (es la operación más crítica)
        assert!(
            table.matmul_backend.contains("blas") || table.matmul_backend.contains("adaptive"),
            "MatMul debe usar BLAS o 'adaptive' en plataformas con backend BLAS disponible. Got: {}",
            table.matmul_backend
        );
        println!("✅ MatMul está usando backend BLAS via dispatch system");
    }
}

#[test]
fn test_matmul_with_static_blas() {
    // Test funcional: verificar que matmul funciona correctamente con dispatch system
    use numrs::array::Array;
    use numrs::ops::fast;

    // Crear matrices pequeñas para test rápido
    let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Array::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    // Ejecutar via fast-path (usa dispatch system automáticamente)
    let result = fast::matmul(&a, &b);

    match result {
        Ok(output) => {
            println!("✅ MatMul ejecutado exitosamente via dispatch system");
            println!("   Shape: {:?}", output.shape());
            println!("   Data: {:?}", output.to_f32().data);

            // Verificar resultado esperado:
            // [1*7+2*9+3*11, 1*8+2*10+3*12]   = [58, 64]
            // [4*7+5*9+6*11, 4*8+5*10+6*12]   = [139, 154]
            assert_eq!(output.shape(), vec![2, 2]);
            assert!((output.to_f32().data[0] - 58.0).abs() < 0.001);
            assert!((output.to_f32().data[1] - 64.0).abs() < 0.001);
            assert!((output.to_f32().data[2] - 139.0).abs() < 0.001);
            assert!((output.to_f32().data[3] - 154.0).abs() < 0.001);

            println!("✅ Resultado correcto verificado");

            // Mostrar qué backend se usó
            let table = get_dispatch_table();
            println!("   Backend usado: {}", table.matmul_backend);
        }
        Err(e) => {
            panic!("MatMul via dispatch system falló: {}", e);
        }
    }
}

#[test]
fn verify_no_dynamic_blas_dependency() {
    // Este test verifica que no hay referencias a librerías dinámicas de BLAS
    // Solo se ejecuta en CI o cuando se solicita explícitamente

    if std::env::var("NUMRS_VERIFY_STATIC").is_ok() {
        println!("🔍 Verificando que no hay dependencias dinámicas de BLAS...");

        #[cfg(target_os = "linux")]
        {
            // En Linux, podemos ejecutar ldd para verificar
            let output = std::process::Command::new("ldd")
                .arg(std::env::current_exe().unwrap())
                .output();

            if let Ok(output) = output {
                let stdout = String::from_utf8_lossy(&output.stdout);
                println!("Dependencias dinámicas:\n{}", stdout);

                // No debe haber referencias a libopenblas, libmkl_rt, libblis
                assert!(
                    !stdout.contains("libopenblas")
                        && !stdout.contains("libmkl_rt")
                        && !stdout.contains("libblis"),
                    "Encontradas dependencias dinámicas de BLAS no deseadas"
                );

                println!("✅ No hay dependencias dinámicas de BLAS");
            }
        }

        #[cfg(target_os = "windows")]
        {
            // En Windows, verificar con dumpbin (si está disponible)
            let output = std::process::Command::new("dumpbin")
                .arg("/DEPENDENTS")
                .arg(std::env::current_exe().unwrap())
                .output();

            if let Ok(output) = output {
                let stdout = String::from_utf8_lossy(&output.stdout);
                println!("Dependencias DLL:\n{}", stdout);

                // No debe haber referencias a mkl_rt.dll, openblas.dll
                assert!(
                    !stdout.to_lowercase().contains("mkl_rt.dll")
                        && !stdout.to_lowercase().contains("openblas.dll"),
                    "Encontradas dependencias dinámicas de BLAS no deseadas"
                );

                println!("✅ No hay dependencias dinámicas de BLAS");
            }
        }

        #[cfg(target_os = "macos")]
        {
            // En macOS, Accelerate es un framework del sistema, es normal que aparezca
            println!("ℹ️ macOS usa Accelerate framework (parte del sistema)");
        }
    } else {
        println!("ℹ️ Test de verificación de static linking omitido");
        println!("   Ejecutar con NUMRS_VERIFY_STATIC=1 para verificar");
    }
}
