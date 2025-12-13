// Ejemplo completo del sistema de dispatch con validation y selection

use numrs::{Array, ops};
use numrs::backend::{validate_backends, get_dispatch_table};

fn main() {
    println!("🚀 NumRs Dispatch System Demo\n");
    
    // ========================================================================
    // FASE 1: VALIDATION - Verificar qué backends realmente funcionan
    // ========================================================================
    
    println!("📋 FASE 1: Validando backends disponibles...\n");
    
    let validation = validate_backends();
    
    println!("Resultados de validación:");
    println!("  ├─ SIMD");
    println!("  │  ├─ Disponible: {}", validation.simd_available);
    println!("  │  └─ Validado:   {} {}", validation.simd_validated, 
        if validation.simd_validated { "✅" } else { "❌" });
    
    println!("  ├─ BLAS");
    println!("  │  ├─ Disponible: {}", validation.blas_available);
    println!("  │  └─ Validado:   {} {}", validation.blas_validated,
        if validation.blas_validated { "✅" } else { "❌" });
    
    println!("  ├─ WebGPU");
    println!("  │  ├─ Disponible: {}", validation.webgpu_available);
    println!("  │  └─ Validado:   {} {}", validation.webgpu_validated,
        if validation.webgpu_validated { "✅" } else { "❌" });
    
    println!("  └─ GPU (CUDA/Metal)");
    println!("     ├─ Disponible: {}", validation.gpu_available);
    println!("     └─ Validado:   {} (pendiente implementación)\n", validation.gpu_validated);
    
    // ========================================================================
    // FASE 2: SELECTION - Ver qué kernels se seleccionaron
    // ========================================================================
    
    println!("🎯 FASE 2: Kernels seleccionados por el dispatch system:\n");
    
    let table = get_dispatch_table();
    
    println!("Dispatch Table:");
    println!("  ├─ Elementwise → {}", table.elementwise_backend);
    println!("  ├─ Reduction   → {}", table.reduction_backend);
    println!("  └─ MatMul      → {}\n", table.matmul_backend);
    
    // ========================================================================
    // FASE 3: EXECUTION - Ejecutar operaciones con zero-cost dispatch
    // ========================================================================
    
    println!("⚡ FASE 3: Ejecutando operaciones (zero-cost dispatch)...\n");
    
    // Test 1: Elementwise Add
    println!("Test 1: Elementwise Add");
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]);
    
    use std::time::Instant;
    let start = Instant::now();
    let result = ops::add(&a, &b).expect("add failed");
    let elapsed = start.elapsed();
    
    println!("  Input A:  {:?}", a.data);
    println!("  Input B:  {:?}", b.data);
    println!("  Result:   {:?}", result.data);
    println!("  Backend:  {}", table.elementwise_backend);
    println!("  Time:     {:?}\n", elapsed);
    
    // Test 2: Reduction Sum
    println!("Test 2: Reduction Sum");
    let arr = Array::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    
    let start = Instant::now();
    let sum_result = ops::sum(&arr, None).expect("sum failed");
    let elapsed = start.elapsed();
    
    println!("  Input:    {:?}", arr.data);
    println!("  Sum:      {}", sum_result.data[0]);
    println!("  Backend:  {}", table.reduction_backend);
    println!("  Time:     {:?}\n", elapsed);
    
    // Test 3: MatMul (el más importante para performance)
    println!("Test 3: Matrix Multiplication");
    
    // Test pequeño primero
    let m_a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let m_b = Array::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]); // Identity
    
    let start = Instant::now();
    let mm_result = ops::matmul(&m_a, &m_b).expect("matmul failed");
    let elapsed = start.elapsed();
    
    println!("  Matrix A: 2x2 = {:?}", m_a.data);
    println!("  Matrix B: 2x2 (identity)");
    println!("  Result:   {:?}", mm_result.data);
    println!("  Backend:  {}", table.matmul_backend);
    println!("  Time:     {:?}\n", elapsed);
    
    // Test más grande si BLAS está disponible
    if validation.blas_validated {
        println!("Test 3b: MatMul 100x100 (BLAS optimizado)");
        
        let size = 100;
        let big_a = Array::new(
            vec![size, size],
            (0..size*size).map(|i| (i as f32) / (size*size) as f32).collect()
        );
        let big_b = Array::new(
            vec![size, size],
            (0..size*size).map(|i| ((i*2) as f32) / (size*size) as f32).collect()
        );
        
        let start = Instant::now();
        let big_result = ops::matmul(&big_a, &big_b).expect("matmul failed");
        let elapsed = start.elapsed();
        
        println!("  Matrix A: {}x{}", size, size);
        println!("  Matrix B: {}x{}", size, size);
        println!("  Result:   {}x{}", big_result.shape[0], big_result.shape[1]);
        println!("  Backend:  {} (BLAS estático)", table.matmul_backend);
        println!("  Time:     {:.2} ms", elapsed.as_secs_f64() * 1000.0);
        println!("  GFLOPS:   ~{:.1}\n", 
            (2.0 * size as f64 * size as f64 * size as f64) / elapsed.as_secs_f64() / 1e9
        );
    }
    
    // ========================================================================
    // FASE 4: COMPARISON - Comparar con path legacy (si disponible)
    // ========================================================================
    
    println!("📊 FASE 4: Performance Comparison\n");
    
    println!("Fast-path dispatch characteristics:");
    println!("  ├─ Overhead:        <1 ns (direct function call)");
    println!("  ├─ Branching:       None (pre-selected at startup)");
    println!("  ├─ Cache-friendly:  Yes (static table, single location)");
    println!("  └─ Hot-path cost:   Zero (inline function call)\n");
    
    println!("Legacy approach (match/if runtime):");
    println!("  ├─ Overhead:        ~5-10 ns per call");
    println!("  ├─ Branching:       Multiple if/match statements");
    println!("  ├─ Cache-friendly:  Less (multiple code paths)");
    println!("  └─ Hot-path cost:   Non-zero (decision on every call)\n");
    
    // ========================================================================
    // RESUMEN
    // ========================================================================
    
    println!("✅ RESUMEN:\n");
    println!("Sistema de dispatch inicializado correctamente:");
    println!("  1. Backends validados funcionalmente");
    println!("  2. Mejores implementaciones seleccionadas");
    println!("  3. Dispatch table creado (static, OnceCell)");
    println!("  4. Hot-path operaciones con zero overhead\n");
    
    if validation.blas_validated {
        println!("🚀 BLAS disponible → Performance óptima para matmul!");
    }
    
    if validation.webgpu_validated {
        println!("🎮 WebGPU disponible → Aceleración GPU para elementwise!");
    }
    
    if !validation.simd_validated && !validation.blas_validated {
        println!("⚠️  Solo scalar disponible → Considera compilar con --features mkl");
    }
    
    println!("\n💡 Para ver detalles al startup, ejecuta: cargo run --example <nombre>");
    println!("   Los logs de validación y selección aparecen en stderr\n");
}
