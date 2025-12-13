// Test de sistema adaptativo de dispatch
// 
// Ejecutar con:
// cargo run --release --manifest-path=numrs-core/Cargo.toml --example test_adaptive
//
// Con microbenchmarks:
// $env:NUMRS_ENABLE_PROBING="1" ; cargo run --release --manifest-path=numrs-core/Cargo.toml --example test_adaptive

use numrs::{Array, ops};

fn main() {
    println!("=== Test Sistema Adaptativo de Dispatch ===\n");
    
    // Trigger startup y mostrar tabla de dispatch
    numrs::print_startup_log();
    
    println!("\n=== Probando matmul con diferentes tamaños ===\n");
    
    // Pequeño (32x32 = 1,024 elementos)
    {
        println!("Test 1: Matriz pequeña 32x32");
        let a = Array::new(vec![32, 32], vec![1.0f32; 32*32]);
        let b = Array::new(vec![32, 32], vec![1.0f32; 32*32]);
        
        let start = std::time::Instant::now();
        let c = ops::matmul(&a, &b).expect("matmul failed");
        let elapsed = start.elapsed();
        
        assert_eq!(c.shape, vec![32, 32]);
        assert!((c.data[0] - 32.0).abs() < 0.01, "Expected 32.0, got {}", c.data[0]);
        println!("  ✓ Resultado correcto: {}x{}", c.shape[0], c.shape[1]);
        println!("  ⏱  Tiempo: {:.3}ms\n", elapsed.as_secs_f64() * 1000.0);
    }
    
    // Mediano (128x128 = 16,384 elementos)
    {
        println!("Test 2: Matriz mediana 128x128");
        let a = Array::new(vec![128, 128], vec![1.0f32; 128*128]);
        let b = Array::new(vec![128, 128], vec![1.0f32; 128*128]);
        
        let start = std::time::Instant::now();
        let c = ops::matmul(&a, &b).expect("matmul failed");
        let elapsed = start.elapsed();
        
        assert_eq!(c.shape, vec![128, 128]);
        assert!((c.data[0] - 128.0).abs() < 0.01, "Expected 128.0, got {}", c.data[0]);
        println!("  ✓ Resultado correcto: {}x{}", c.shape[0], c.shape[1]);
        println!("  ⏱  Tiempo: {:.3}ms\n", elapsed.as_secs_f64() * 1000.0);
    }
    
    // Grande (512x512 = 262,144 elementos)
    {
        println!("Test 3: Matriz grande 512x512");
        let a = Array::new(vec![512, 512], vec![1.0f32; 512*512]);
        let b = Array::new(vec![512, 512], vec![1.0f32; 512*512]);
        
        let start = std::time::Instant::now();
        let c = ops::matmul(&a, &b).expect("matmul failed");
        let elapsed = start.elapsed();
        
        assert_eq!(c.shape, vec![512, 512]);
        assert!((c.data[0] - 512.0).abs() < 0.01, "Expected 512.0, got {}", c.data[0]);
        println!("  ✓ Resultado correcto: {}x{}", c.shape[0], c.shape[1]);
        println!("  ⏱  Tiempo: {:.3}ms\n", elapsed.as_secs_f64() * 1000.0);
    }
    
    // Muy grande (1024x1024 = 1,048,576 elementos)
    {
        println!("Test 4: Matriz muy grande 1024x1024");
        let a = Array::new(vec![1024, 1024], vec![1.0f32; 1024*1024]);
        let b = Array::new(vec![1024, 1024], vec![1.0f32; 1024*1024]);
        
        let start = std::time::Instant::now();
        let c = ops::matmul(&a, &b).expect("matmul failed");
        let elapsed = start.elapsed();
        
        assert_eq!(c.shape, vec![1024, 1024]);
        assert!((c.data[0] - 1024.0).abs() < 0.01, "Expected 1024.0, got {}", c.data[0]);
        println!("  ✓ Resultado correcto: {}x{}", c.shape[0], c.shape[1]);
        println!("  ⏱  Tiempo: {:.3}ms\n", elapsed.as_secs_f64() * 1000.0);
    }
    
    println!("=== Probando elementwise (add) ===\n");
    
    {
        let a = Array::new(vec![1000], vec![2.0f32; 1000]);
        let b = Array::new(vec![1000], vec![3.0f32; 1000]);
        
        let c = ops::add(&a, &b).expect("add failed");
        assert_eq!(c.data[0], 5.0);
        println!("  ✓ add correcto\n");
    }
    
    println!("=== Probando reduction (sum) ===\n");
    
    {
        let a = Array::new(vec![100], vec![1.0f32; 100]);
        let s = ops::sum(&a, None).expect("sum failed");
        assert!((s.data[0] - 100.0).abs() < 0.01);
        println!("  ✓ sum correcto\n");
    }
    
    println!("=== ✅ Todos los tests pasaron ===");
}
