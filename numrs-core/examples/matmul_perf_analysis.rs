// Análisis detallado de performance de matmul con MKL
// Mide overhead, tamaños variables, y compara con expectativas

use numrs::Array;
use numrs::ops::fast;
use std::time::Instant;

fn benchmark_matmul(size: usize, iterations: usize) -> (f64, f64) {
    // Crear matrices
    let a_data: Vec<f32> = (0..size*size).map(|i| (i as f32) / (size*size) as f32).collect();
    let b_data: Vec<f32> = (0..size*size).map(|i| ((i*2) as f32) / (size*size) as f32).collect();
    
    let a = Array::new(vec![size, size], a_data);
    let b = Array::new(vec![size, size], b_data);
    
    // Warmup (importante para MKL)
    for _ in 0..3 {
        let _ = fast::matmul(&a, &b).unwrap();
    }
    
    // Benchmark real
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fast::matmul(&a, &b).unwrap();
    }
    let elapsed = start.elapsed();
    
    let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    
    // GFLOPS = (2*M*N*K) / time_in_seconds / 1e9
    let flops = 2.0 * size as f64 * size as f64 * size as f64;
    let gflops = flops / (avg_time_ms / 1000.0) / 1e9;
    
    (avg_time_ms, gflops)
}

fn main() {
    println!("🔬 NumRs MatMul Performance Analysis\n");
    
    // Backend info
    let table = numrs::backend::get_dispatch_table();
    println!("Backend: {}\n", table.matmul_backend);
    
    #[cfg(feature = "mkl")]
    println!("✅ MKL feature enabled (static linking)");
    
    #[cfg(not(feature = "mkl"))]
    println!("⚠️  MKL feature NOT enabled (using fallback)");
    
    println!();
    
    // Detectar threading
    #[cfg(feature = "mkl")]
    {
        println!("MKL variant: mkl-static-lp64-seq (single-threaded)");
        println!("⚠️  Para mejor performance, considera mkl-static-lp64-iomp");
    }
    
    println!("\n{}\n", "=".repeat(70));
    println!("{:<12} {:<12} {:<12} {:<12} {:<12}", 
        "Size", "Iterations", "Avg Time", "GFLOPS", "Expected");
    println!("{}", "-".repeat(70));
    
    // Benchmark diferentes tamaños
    let test_cases = vec![
        (10, 1000, 1.0),      // Muy pequeño - overhead domina
        (50, 500, 3.0),       // Pequeño - warming up
        (100, 200, 10.0),     // Referencia actual
        (200, 50, 20.0),      // Cache L1 fit
        (500, 10, 40.0),      // Cache L2 fit
        (1000, 5, 80.0),      // Cache L3 fit
        (2000, 2, 100.0),     // Peak performance zone
    ];
    
    for (size, iters, expected_gflops) in test_cases {
        let (time_ms, gflops) = benchmark_matmul(size, iters);
        
        let efficiency = (gflops / expected_gflops * 100.0).min(100.0);
        let status = if efficiency > 80.0 {
            "✅"
        } else if efficiency > 50.0 {
            "⚠️ "
        } else {
            "❌"
        };
        
        println!("{:<12} {:<12} {:>9.3} ms {:<12.1} {:<12.1} {}",
            format!("{}x{}", size, size),
            iters,
            time_ms,
            gflops,
            expected_gflops,
            status
        );
    }
    
    println!("\n{}\n", "=".repeat(70));
    
    // Overhead analysis con matrices pequeñas
    println!("📊 Overhead Analysis (10x10 matrix):\n");
    
    let small_size = 10;
    let a = Array::new(vec![small_size, small_size], vec![1.0; small_size * small_size]);
    let b = Array::new(vec![small_size, small_size], vec![1.0; small_size * small_size]);
    
    // Tiempo total
    let start = Instant::now();
    let n_ops = 10000;
    for _ in 0..n_ops {
        let _ = fast::matmul(&a, &b).unwrap();
    }
    let total_time = start.elapsed();
    
    let time_per_op_us = total_time.as_micros() as f64 / n_ops as f64;
    
    println!("  Time per operation: {:.2} µs", time_per_op_us);
    println!("  Throughput: {:.0} ops/sec", 1_000_000.0 / time_per_op_us);
    
    // Flops para 10x10
    let flops_10x10 = 2.0 * 10.0 * 10.0 * 10.0; // 2000 FLOPS
    let compute_time_ideal = flops_10x10 / (100.0 * 1e9); // Asumiendo 100 GFLOPS peak
    let overhead_us = time_per_op_us - (compute_time_ideal * 1e6);
    
    println!("  Estimated overhead: {:.2} µs", overhead_us);
    println!("  Overhead ratio: {:.1}%", (overhead_us / time_per_op_us) * 100.0);
    
    println!("\n{}\n", "=".repeat(70));
    
    // Diagnóstico
    println!("💡 Diagnóstico:\n");
    
    let (_, gflops_100) = benchmark_matmul(100, 200);
    let (_, gflops_1000) = benchmark_matmul(1000, 5);
    
    if gflops_100 < 15.0 {
        println!("❌ Performance baja en matrices pequeñas");
        println!("   Causas posibles:");
        println!("   - Overhead de wrapper Rust → CBLAS");
        println!("   - Arrays no contiguos (requires copies)");
        println!("   - MKL sequential (no threading)");
    }
    
    if gflops_1000 < 50.0 {
        println!("❌ Performance baja en matrices grandes");
        println!("   Causas posibles:");
        println!("   - MKL sequential (necesitas parallel variant)");
        println!("   - TurboBoost deshabilitado");
        println!("   - Thermal throttling");
    }
    
    if gflops_1000 / gflops_100 < 3.0 {
        println!("⚠️  Scaling pobre con tamaño");
        println!("   El ratio debería ser ~5-10x entre 100x100 y 1000x1000");
    }
    
    println!("\n📚 Recomendaciones:\n");
    println!("1. Para mejor performance, usa MKL parallel:");
    println!("   Cargo.toml: features = [\"mkl-static-lp64-iomp\"]");
    println!("\n2. Verificar que arrays sean contiguos:");
    println!("   assert!(array.data.as_ptr() == &array.data[0] as *const _);");
    println!("\n3. Para hot-paths, considera batching:");
    println!("   En lugar de muchas ops pequeñas, agrupa en una grande");
    println!("\n4. Matrices grandes (>1000x1000) aprovechan mejor el hardware");
}
