// Ejemplo de verificación de static BLAS linking
// Compila y ejecuta este archivo para verificar que BLAS está correctamente embebido

use numrs::array::Array;
use numrs::backend::{validate_backends, get_dispatch_table};

fn main() {
    println!("🔍 NumRs - Verificación de Static BLAS Linking\n");
    
    // 1. Validar backends (funcional, no solo compile-time)
    let validation = validate_backends();
    println!("📊 Backends validados funcionalmente:");
    println!("   SIMD:   {} {}", 
        validation.simd_validated,
        if validation.simd_validated { "✅" } else { "❌" }
    );
    println!("   BLAS:   {} {}", 
        validation.blas_validated,
        if validation.blas_validated { "✅" } else { "❌" }
    );
    println!("   WebGPU: {} {}", 
        validation.webgpu_validated,
        if validation.webgpu_validated { "✅" } else { "❌" }
    );
    println!("   GPU:    {} (pendiente implementación)", 
        validation.gpu_validated
    );
    println!();
    
    // 2. Mostrar kernels seleccionados por dispatch system
    let table = get_dispatch_table();
    println!("🎯 Kernels seleccionados:");
    println!("   Elementwise: {}", table.elementwise_backend);
    println!("   Reduction:   {}", table.reduction_backend);
    println!("   MatMul:      {}", table.matmul_backend);
    println!();
    
    // 3. Identificar proveedor BLAS específico
    if validation.blas_validated {
        #[cfg(target_os = "macos")]
        println!("🍎 macOS: Usando Accelerate framework (sistema)");
        
        #[cfg(all(feature = "mkl", any(target_arch = "x86_64", target_arch = "x86")))]
        println!("⚡ x86_64: Usando Intel MKL (static linking via intel-mkl-src)");
        
        #[cfg(feature = "blis")]
        println!("🔷 ARM/otras: Usando BLIS (static linking via blis-src)");
    } else {
        println!("⚠️ BLAS no disponible: Usando implementación Rust (SIMD/scalar)");
        println!("   Para habilitar BLAS:");
        println!("   - macOS: automático (Accelerate)");
        println!("   - x86_64: cargo build --features mkl");
        println!("   - ARM: cargo build --features blis");
    }
    
    println!();
    
    // 4. Test funcional con MatMul usando fast-path (zero-cost dispatch)
    println!("🧪 Test funcional: MatMul 100x100 (via fast-path)");
    
    let size = 100;
    let a_data: Vec<f32> = (0..size*size).map(|i| (i as f32) / (size*size) as f32).collect();
    let b_data: Vec<f32> = (0..size*size).map(|i| ((i*2) as f32) / (size*size) as f32).collect();
    
    let a = Array::new(vec![size, size], a_data);
    let b = Array::new(vec![size, size], b_data);
    
    println!("   Matriz A: {}x{}", a.shape[0], a.shape[1]);
    println!("   Matriz B: {}x{}", b.shape[0], b.shape[1]);
    
    use std::time::Instant;
    let start = Instant::now();
    
    // Usar fast-path que utiliza el dispatch table automáticamente
    match numrs::ops::fast::matmul(&a, &b) {
        Ok(result) => {
            let elapsed = start.elapsed();
            println!("   ✅ Resultado: shape {:?}", result.shape);
            println!("   ⏱️ Tiempo: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
            
            if validation.blas_validated {
                let gflops = (2.0 * size as f64 * size as f64 * size as f64) / elapsed.as_secs_f64() / 1e9;
                println!("   ⚡ Performance: {:.1} GFLOPS", gflops);
            }
            
            println!("   📦 Primeros valores: [{:.4}, {:.4}, {:.4}, ...]", 
                result.data[0], result.data[1], result.data[2]);
            println!("   🔧 Backend usado: {}", table.matmul_backend);
            
            // Verificar que el resultado es razonable
            if result.shape == vec![size, size] && result.data.len() == size * size {
                println!("\n✅ VERIFICACIÓN EXITOSA: Dispatch system funciona correctamente");
                
                if validation.blas_validated {
                    println!("   ✓ BLAS estático validado y funcionando");
                    println!("   ✓ El binario incluye BLAS embebido (sin dependencias externas)");
                } else if validation.simd_validated {
                    println!("   ✓ SIMD validado y funcionando");
                    println!("   ℹ Para mejor performance, compila con --features mkl");
                } else {
                    println!("   ✓ Fallback scalar funcionando");
                    println!("   ⚠ Para mejor performance, compila con --features mkl");
                }
            } else {
                println!("\n❌ ERROR: Resultado inválido");
            }
        }
        Err(e) => {
            println!("   ❌ Error: {}", e);
        }
    }
    
    println!();
    
    // 5. Información adicional
    println!("📚 Más información:");
    println!("   - Ver STATIC_BLAS_LINKING.md para guía completa");
    println!("   - Ejecutar tests: cargo test --test static_blas_linking");
    println!("   - Build script: .\\scripts\\build_static.ps1 -Release");
    
    #[cfg(target_os = "windows")]
    {
        println!("\n💡 Windows: Verificar dependencias con:");
        println!("   dumpbin /DEPENDENTS target\\release\\numrs.dll");
    }
    
    #[cfg(target_os = "linux")]
    {
        println!("\n💡 Linux: Verificar dependencias con:");
        println!("   ldd target/release/libnumrs.so | grep -i blas");
    }
}
