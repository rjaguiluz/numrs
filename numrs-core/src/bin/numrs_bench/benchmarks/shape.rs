//! Shape operations benchmarks

use numrs::{ops, Array};
use crate::types::{BenchResult, BENCHMARK_ITERATIONS};
use crate::benchmarks::benchmark_fn;

pub fn benchmark_shape_ops(results: &mut Vec<BenchResult>, backend: &str) {
    println!("  📊 Shape Operations...");
    
    // Test F32
    for &dim in &[500, 1000] {
        let data: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.01).collect();
        let arr_1d = Array::new(vec![dim * dim], data);
        let a = ops::reshape(&arr_1d, &[dim as isize, dim as isize]).unwrap();
        let total_elements = (dim * dim) as f64;
        
        let test_result = ops::transpose(&a, None);
        if test_result.is_ok() {
            let result = test_result.unwrap();
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            
            if result_dtype_str.contains("f32") {
                let (mean, std_dev) = benchmark_fn("transpose", || {
                    let _ = ops::transpose(&a, None).unwrap();
                }, BENCHMARK_ITERATIONS / 4);
                
                results.push(BenchResult {
                    operation: "transpose".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}x{}", dim, dim),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: total_elements / mean,
                    dtype1: "f32".to_string(),
                    dtype2: None,
                    dtype_res: "f32".to_string(),
                });
            }
        }
    }
    
    // Test F64
    for &dim in &[500, 1000] {
        let data: Vec<f64> = (0..dim * dim).map(|i| (i as f64) * 0.01).collect();
        let arr_1d = Array::<f64>::new(vec![dim * dim], data);
        let a = ops::reshape(&arr_1d, &[dim as isize, dim as isize]).unwrap();
        let total_elements = (dim * dim) as f64;
        
        let test_result = ops::transpose(&a, None);
        if test_result.is_ok() {
            let result = test_result.unwrap();
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            
            if result_dtype_str.contains("f64") {
                let (mean, std_dev) = benchmark_fn("transpose", || {
                    let _ = ops::transpose(&a, None).unwrap();
                }, BENCHMARK_ITERATIONS / 4);
                
                results.push(BenchResult {
                    operation: "transpose".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}x{}", dim, dim),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: total_elements / mean,
                    dtype1: "f64".to_string(),
                    dtype2: None,
                    dtype_res: "f64".to_string(),
                });
            }
        }
    }
    
    // Test I32
    let dim = 500;
    let data: Vec<i32> = (0..dim * dim).map(|i| i as i32).collect();
    let arr_1d = Array::<i32>::new(vec![dim * dim], data);
    let a = ops::reshape(&arr_1d, &[dim as isize, dim as isize]).unwrap();
    let total_elements = (dim * dim) as f64;
    
    if let Ok(_) = ops::transpose(&a, None) {
        let (mean, std_dev) = benchmark_fn("transpose_i32", || {
            let _ = ops::transpose(&a, None).unwrap();
        }, BENCHMARK_ITERATIONS / 4);
        
        results.push(BenchResult {
            operation: "transpose_i32".to_string(),
            backend: backend.to_string(),
            size: format!("{}x{}", dim, dim),
            mean_us: mean,
            std_dev_us: std_dev,
            throughput_mops: total_elements / mean,
            dtype1: "i32".to_string(),
            dtype2: None,
            dtype_res: "i32".to_string(),
        });
    }
    
    // Test U8
    let data: Vec<u8> = (0..dim * dim).map(|i| (i % 256) as u8).collect();
    let arr_1d = Array::<u8>::new(vec![dim * dim], data);
    let a = ops::reshape(&arr_1d, &[dim as isize, dim as isize]).unwrap();
    let total_elements = (dim * dim) as f64;
    
    if let Ok(_) = ops::transpose(&a, None) {
        let (mean, std_dev) = benchmark_fn("transpose_u8", || {
            let _ = ops::transpose(&a, None).unwrap();
        }, BENCHMARK_ITERATIONS / 4);
        
        results.push(BenchResult {
            operation: "transpose_u8".to_string(),
            backend: backend.to_string(),
            size: format!("{}x{}", dim, dim),
            mean_us: mean,
            std_dev_us: std_dev,
            throughput_mops: total_elements / mean,
            dtype1: "u8".to_string(),
            dtype2: None,
            dtype_res: "u8".to_string(),
        });
    }
}
