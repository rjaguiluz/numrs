//! Linear algebra benchmarks

use numrs::{ops, Array};
use crate::types::{BenchResult, BENCHMARK_ITERATIONS};
use crate::benchmarks::benchmark_fn;

pub fn benchmark_linalg(results: &mut Vec<BenchResult>, backend: &str) {
    println!("  📊 Linear Algebra...");
    
    // Dot product F32
    for &size in &[1_000, 10_000, 100_000] {
        let data_a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let a = Array::new(vec![size], data_a);
        let b = Array::new(vec![size], data_b);
        
        let test_result = ops::dot(&a, &b);
        if test_result.is_ok() {
            let result = test_result.unwrap();
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            
            if result_dtype_str.contains("f32") {
                let (mean, std_dev) = benchmark_fn("dot", || {
                    let _ = ops::dot(&a, &b).unwrap();
                }, BENCHMARK_ITERATIONS);
                
                results.push(BenchResult {
                    operation: "dot".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}", size),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: (2.0 * size as f64) / mean,
                    dtype1: "f32".to_string(),
                    dtype2: Some("f32".to_string()),
                    dtype_res: "f32".to_string(),
                });
            }
        }
    }
    
    // Dot product F64
    for &size in &[1_000, 10_000, 100_000] {
        let data_a: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
        let data_b: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
        let a = Array::<f64>::new(vec![size], data_a);
        let b = Array::<f64>::new(vec![size], data_b);
        
        let test_result = ops::dot(&a, &b);
        if test_result.is_ok() {
            let result = test_result.unwrap();
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            
            if result_dtype_str.contains("f64") {
                let (mean, std_dev) = benchmark_fn("dot", || {
                    let _ = ops::dot(&a, &b).unwrap();
                }, BENCHMARK_ITERATIONS);
                
                results.push(BenchResult {
                    operation: "dot".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}", size),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: (2.0 * size as f64) / mean,
                    dtype1: "f64".to_string(),
                    dtype2: Some("f64".to_string()),
                    dtype_res: "f64".to_string(),
                });
            }
        }
    }
    
    // Dot product with i32
    let size = 10_000;
    let data_a: Vec<i32> = (0..size).map(|i| (i % 100) as i32).collect();
    let data_b: Vec<i32> = (0..size).map(|i| ((i + 1) % 100) as i32).collect();
    let a = Array::<i32>::new(vec![size], data_a);
    let b = Array::<i32>::new(vec![size], data_b);
    
    let test_result = ops::dot(&a, &b);
    if test_result.is_ok() {
        let result = test_result.unwrap();
        let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
        
        if result_dtype_str.contains("i32") {
            let (mean, std_dev) = benchmark_fn("dot", || {
                let _ = ops::dot(&a, &b).unwrap();
            }, BENCHMARK_ITERATIONS);
            
            results.push(BenchResult {
                operation: "dot".to_string(),
                backend: backend.to_string(),
                size: format!("{}", size),
                mean_us: mean,
                std_dev_us: std_dev,
                throughput_mops: (2.0 * size as f64) / mean,
                dtype1: "i32".to_string(),
                dtype2: Some("i32".to_string()),
                dtype_res: "i32".to_string(),
            });
        }
    }

    // Matrix multiplication F32
    for &dim in &[128, 256, 512, 1024, 2048] {
        let data_a: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.01).collect();
        let data_b: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.01).collect();
        let arr_a = Array::new(vec![dim * dim], data_a);
        let a = ops::reshape(&arr_a, &[dim as isize, dim as isize]).unwrap();
        let arr_b = Array::new(vec![dim * dim], data_b);
        let b = ops::reshape(&arr_b, &[dim as isize, dim as isize]).unwrap();
        let total_ops = (2 * dim * dim * dim) as f64;
        
        // Ajustar número de iteraciones según el tamaño
        let iterations = if dim >= 1024 {
            (BENCHMARK_ITERATIONS / 100).max(1)  // Matrices muy grandes: al menos 1 iteración
        } else if dim >= 512 {
            (BENCHMARK_ITERATIONS / 50).max(1)
        } else {
            BENCHMARK_ITERATIONS / 10
        };
        
        let test_result = ops::matmul(&a, &b);
        if test_result.is_ok() {
            let result = test_result.unwrap();
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            
            if result_dtype_str.contains("f32") {
                let (mean, std_dev) = benchmark_fn("matmul", || {
                    let _ = ops::matmul(&a, &b).unwrap();
                }, iterations);
                
                results.push(BenchResult {
                    operation: "matmul".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}x{}", dim, dim),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: total_ops / mean,
                    dtype1: "f32".to_string(),
                    dtype2: Some("f32".to_string()),
                    dtype_res: "f32".to_string(),
                });
            }
        }
    }
    
    // Matrix multiplication F64
    for &dim in &[128, 256, 512, 1024, 2048] {
        let data_a: Vec<f64> = (0..dim * dim).map(|i| (i as f64) * 0.01).collect();
        let data_b: Vec<f64> = (0..dim * dim).map(|i| (i as f64) * 0.01).collect();
        let arr_a = Array::<f64>::new(vec![dim * dim], data_a);
        let a = ops::reshape(&arr_a, &[dim as isize, dim as isize]).unwrap();
        let arr_b = Array::<f64>::new(vec![dim * dim], data_b);
        let b = ops::reshape(&arr_b, &[dim as isize, dim as isize]).unwrap();
        let total_ops = (2 * dim * dim * dim) as f64;
        
        // Ajustar número de iteraciones según el tamaño
        let iterations = if dim >= 1024 {
            (BENCHMARK_ITERATIONS / 100).max(1)
        } else if dim >= 512 {
            (BENCHMARK_ITERATIONS / 50).max(1)
        } else {
            BENCHMARK_ITERATIONS / 10
        };
        
        let test_result = ops::matmul(&a, &b);
        if test_result.is_ok() {
            let result = test_result.unwrap();
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            
            if result_dtype_str.contains("f64") {
                let (mean, std_dev) = benchmark_fn("matmul", || {
                    let _ = ops::matmul(&a, &b).unwrap();
                }, iterations);
                
                results.push(BenchResult {
                    operation: "matmul".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}x{}", dim, dim),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: total_ops / mean,
                    dtype1: "f64".to_string(),
                    dtype2: Some("f64".to_string()),
                    dtype_res: "f64".to_string(),
                });
            }
        }
    }
    
    // Matrix multiplication I32
    let dim = 128;
    let data_a: Vec<i32> = (0..dim * dim).map(|i| (i % 100) as i32).collect();
    let data_b: Vec<i32> = (0..dim * dim).map(|i| (i % 100) as i32).collect();
    let arr_a = Array::<i32>::new(vec![dim * dim], data_a);
    let a = ops::reshape(&arr_a, &[dim as isize, dim as isize]).unwrap();
    let arr_b = Array::<i32>::new(vec![dim * dim], data_b);
    let b = ops::reshape(&arr_b, &[dim as isize, dim as isize]).unwrap();
    let total_ops = (2 * dim * dim * dim) as f64;
    
    if let Ok(_) = ops::matmul(&a, &b) {
        let (mean, std_dev) = benchmark_fn("matmul_i32", || {
            let _ = ops::matmul(&a, &b).unwrap();
        }, BENCHMARK_ITERATIONS / 10);
        
        results.push(BenchResult {
            operation: "matmul_i32".to_string(),
            backend: backend.to_string(),
            size: format!("{}x{}", dim, dim),
            mean_us: mean,
            std_dev_us: std_dev,
            throughput_mops: total_ops / mean,
            dtype1: "i32".to_string(),
            dtype2: Some("i32".to_string()),
            dtype_res: "i32".to_string(),
        });
    }
}
