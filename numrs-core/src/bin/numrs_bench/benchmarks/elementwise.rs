//! Elementwise operations benchmarks

use numrs::{ops, Array};
use crate::types::{BenchResult, BENCHMARK_ITERATIONS};
use crate::benchmarks::benchmark_fn;

pub fn benchmark_elementwise_binary(results: &mut Vec<BenchResult>, backend: &str) {
    println!("  📊 Elementwise Binary...");
    
    // Test F32
    for &size in &[10_000, 100_000] {
        for (op_name, op) in &[
            ("add", ops::add as fn(&Array, &Array) -> Result<Array, _>),
            ("mul", ops::mul as fn(&Array, &Array) -> Result<Array, _>),
            ("sub", ops::sub as fn(&Array, &Array) -> Result<Array, _>),
            ("div", ops::div as fn(&Array, &Array) -> Result<Array, _>),
        ] {
            let data_a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let data_b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let a = Array::new(vec![size], data_a);
            let b = Array::new(vec![size], data_b);
            
            let test_result = op(&a, &b);
            if test_result.is_ok() {
                let result = test_result.unwrap();
                let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
                
                if result_dtype_str.contains("f32") {
                    let (mean, std_dev) = benchmark_fn(op_name, || {
                        let _ = op(&a, &b).unwrap();
                    }, BENCHMARK_ITERATIONS);
                    
                    results.push(BenchResult {
                        operation: op_name.to_string(),
                        backend: backend.to_string(),
                        size: format!("{}", size),
                        mean_us: mean,
                        std_dev_us: std_dev,
                        throughput_mops: (size as f64) / mean,
                        dtype1: "f32".to_string(),
                        dtype2: Some("f32".to_string()),
                        dtype_res: "f32".to_string(),
                    });
                }
            }
        }
    }
    
    // Test F64
    for &size in &[10_000, 100_000] {
        let data_a: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
        let data_b: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
        let a = Array::<f64>::new(vec![size], data_a);
        let b = Array::<f64>::new(vec![size], data_b);
        
        // Test add
        if let Ok(result) = ops::add(&a, &b) {
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            if result_dtype_str.contains("f64") {
                let (mean, std_dev) = benchmark_fn("add", || {
                    let _ = ops::add(&a, &b).unwrap();
                }, BENCHMARK_ITERATIONS);
                results.push(BenchResult {
                    operation: "add".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}", size),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: (size as f64) / mean,
                    dtype1: "f64".to_string(),
                    dtype2: Some("f64".to_string()),
                    dtype_res: "f64".to_string(),
                });
            }
        }
        
        // Test mul
        if let Ok(result) = ops::mul(&a, &b) {
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            if result_dtype_str.contains("f64") {
                let (mean, std_dev) = benchmark_fn("mul", || {
                    let _ = ops::mul(&a, &b).unwrap();
                }, BENCHMARK_ITERATIONS);
                results.push(BenchResult {
                    operation: "mul".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}", size),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: (size as f64) / mean,
                    dtype1: "f64".to_string(),
                    dtype2: Some("f64".to_string()),
                    dtype_res: "f64".to_string(),
                });
            }
        }
        
        // Test sub
        if let Ok(result) = ops::sub(&a, &b) {
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            if result_dtype_str.contains("f64") {
                let (mean, std_dev) = benchmark_fn("sub", || {
                    let _ = ops::sub(&a, &b).unwrap();
                }, BENCHMARK_ITERATIONS);
                results.push(BenchResult {
                    operation: "sub".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}", size),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: (size as f64) / mean,
                    dtype1: "f64".to_string(),
                    dtype2: Some("f64".to_string()),
                    dtype_res: "f64".to_string(),
                });
            }
        }
        
        // Test div
        if let Ok(result) = ops::div(&a, &b) {
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            if result_dtype_str.contains("f64") {
                let (mean, std_dev) = benchmark_fn("div", || {
                    let _ = ops::div(&a, &b).unwrap();
                }, BENCHMARK_ITERATIONS);
                results.push(BenchResult {
                    operation: "div".to_string(),
                    backend: backend.to_string(),
                    size: format!("{}", size),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: (size as f64) / mean,
                    dtype1: "f64".to_string(),
                    dtype2: Some("f64".to_string()),
                    dtype_res: "f64".to_string(),
                });
            }
        }
    }
    
    // Test type promotion: i32 + f32 → f32 (smaller size for cross-type ops)
    println!("  📊 Type Promotion Tests...");
    let size = 10_000;
    
    // i32 + f32 → f32
    let data_i32: Vec<i32> = (0..size).map(|i| i as i32).collect();
    let data_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
    let a_i32 = Array::<i32>::new(vec![size], data_i32);
    let b_f32 = Array::<f32>::new(vec![size], data_f32);
    
    if let Ok(result) = ops::add(&a_i32, &b_f32) {
        let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
        if result_dtype_str.contains("f32") {
            let (mean, std_dev) = benchmark_fn("add", || {
                let _ = ops::add(&a_i32, &b_f32).unwrap();
            }, BENCHMARK_ITERATIONS);
            
            results.push(BenchResult {
                operation: "add".to_string(),
                backend: backend.to_string(),
                size: format!("{}", size),
                mean_us: mean,
                std_dev_us: std_dev,
                throughput_mops: (size as f64) / mean,
                dtype1: "i32".to_string(),
                dtype2: Some("f32".to_string()),
                dtype_res: "f32".to_string(),
            });
        }
    }
    
    // f64 + i32 → f64
    let data_f64: Vec<f64> = (0..size).map(|i| (i as f64) * 0.5).collect();
    let a_f64 = Array::<f64>::new(vec![size], data_f64);
    
    if let Ok(result) = ops::mul(&a_f64, &a_i32) {
        let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
        if result_dtype_str.contains("f64") {
            let (mean, std_dev) = benchmark_fn("mul", || {
                let _ = ops::mul(&a_f64, &a_i32).unwrap();
            }, BENCHMARK_ITERATIONS);
            
            results.push(BenchResult {
                operation: "mul".to_string(),
                backend: backend.to_string(),
                size: format!("{}", size),
                mean_us: mean,
                std_dev_us: std_dev,
                throughput_mops: (size as f64) / mean,
                dtype1: "f64".to_string(),
                dtype2: Some("i32".to_string()),
                dtype_res: "f64".to_string(),
            });
        }
    }
    
    // Test pure i32 operations
    let data_i32_b: Vec<i32> = (0..size).map(|i| (i as i32) + 1).collect();
    let b_i32 = Array::<i32>::new(vec![size], data_i32_b);
    
    if let Ok(result) = ops::add(&a_i32, &b_i32) {
        let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
        if result_dtype_str.contains("i32") {
            let (mean, std_dev) = benchmark_fn("add", || {
                let _ = ops::add(&a_i32, &b_i32).unwrap();
            }, BENCHMARK_ITERATIONS);
            
            results.push(BenchResult {
                operation: "add".to_string(),
                backend: backend.to_string(),
                size: format!("{}", size),
                mean_us: mean,
                std_dev_us: std_dev,
                throughput_mops: (size as f64) / mean,
                dtype1: "i32".to_string(),
                dtype2: Some("i32".to_string()),
                dtype_res: "i32".to_string(),
            });
        }
    }
}

pub fn benchmark_elementwise_unary(results: &mut Vec<BenchResult>, backend: &str) {
    println!("  📊 Elementwise Unary...");
    
    // Test F32
    for &size in &[10_000, 100_000] {
        for (op_name, op) in &[
            ("exp", ops::exp as fn(&Array) -> Result<Array, _>),
            ("cos", ops::cos as fn(&Array) -> Result<Array, _>),
            ("sin", ops::sin as fn(&Array) -> Result<Array, _>),
            ("sqrt", ops::sqrt as fn(&Array) -> Result<Array, _>),
        ] {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let a = Array::new(vec![size], data);
            
            let test_result = op(&a);
            if test_result.is_ok() {
                let result = test_result.unwrap();
                let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
                
                if result_dtype_str.contains("f32") {
                    let (mean, std_dev) = benchmark_fn(op_name, || {
                        let _ = op(&a).unwrap();
                    }, BENCHMARK_ITERATIONS);
                    
                    results.push(BenchResult {
                        operation: op_name.to_string(),
                        backend: backend.to_string(),
                        size: format!("{}", size),
                        mean_us: mean,
                        std_dev_us: std_dev,
                        throughput_mops: (size as f64) / mean,
                        dtype1: "f32".to_string(),
                        dtype2: None,
                        dtype_res: "f32".to_string(),
                    });
                }
            }
        }
    }
    
    // Test F64 unary operations
    for &size in &[10_000, 100_000] {
        let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
        let a = Array::<f64>::new(vec![size], data);
        
        for (op_name, op) in &[
            ("exp", ops::exp as fn(&Array<f64>) -> Result<Array, _>),
            ("cos", ops::cos as fn(&Array<f64>) -> Result<Array, _>),
            ("sin", ops::sin as fn(&Array<f64>) -> Result<Array, _>),
            ("sqrt", ops::sqrt as fn(&Array<f64>) -> Result<Array, _>),
        ] {
            let test_result = op(&a);
            if test_result.is_ok() {
                let result = test_result.unwrap();
                let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
                
                if result_dtype_str.contains("f64") {
                    let (mean, std_dev) = benchmark_fn(op_name, || {
                        let _ = op(&a).unwrap();
                    }, BENCHMARK_ITERATIONS);
                    
                    results.push(BenchResult {
                        operation: op_name.to_string(),
                        backend: backend.to_string(),
                        size: format!("{}", size),
                        mean_us: mean,
                        std_dev_us: std_dev,
                        throughput_mops: (size as f64) / mean,
                        dtype1: "f64".to_string(),
                        dtype2: None,
                        dtype_res: "f64".to_string(),
                    });
                }
            }
        }
    }
    
    // Test with i32 unary operations
    let size = 10_000;
    let data_i32: Vec<i32> = (1..=size).map(|i| i as i32).collect();
    let a_i32 = Array::<i32>::new(vec![size], data_i32);
    
    if let Ok(_) = ops::sqrt(&a_i32) {
        let (mean, std_dev) = benchmark_fn("sqrt_i32", || {
            let _ = ops::sqrt(&a_i32).unwrap();
        }, BENCHMARK_ITERATIONS);
        
        results.push(BenchResult {
            operation: "sqrt_i32".to_string(),
            backend: backend.to_string(),
            size: format!("{}", size),
            mean_us: mean,
            std_dev_us: std_dev,
            throughput_mops: (size as f64) / mean,
            dtype1: "i32".to_string(),
            dtype2: None,
            dtype_res: "i32".to_string(),
        });
    }
    
    // Test with U8 unary operations
    let data_u8: Vec<u8> = (1..=size).map(|i| (i % 256) as u8).collect();
    let a_u8 = Array::<u8>::new(vec![size], data_u8);
    
    if let Ok(_) = ops::sqrt(&a_u8) {
        let (mean, std_dev) = benchmark_fn("sqrt_u8", || {
            let _ = ops::sqrt(&a_u8).unwrap();
        }, BENCHMARK_ITERATIONS);
        
        results.push(BenchResult {
            operation: "sqrt_u8".to_string(),
            backend: backend.to_string(),
            size: format!("{}", size),
            mean_us: mean,
            std_dev_us: std_dev,
            throughput_mops: (size as f64) / mean,
            dtype1: "u8".to_string(),
            dtype2: None,
            dtype_res: "u8".to_string(),
        });
    }
    
    // Test with I8 unary operations
    let data_i8: Vec<i8> = (1..=size).map(|i| (i % 128) as i8).collect();
    let a_i8 = Array::<i8>::new(vec![size], data_i8);
    
    if let Ok(_) = ops::exp(&a_i8) {
        let (mean, std_dev) = benchmark_fn("exp_i8", || {
            let _ = ops::exp(&a_i8).unwrap();
        }, BENCHMARK_ITERATIONS);
        
        results.push(BenchResult {
            operation: "exp_i8".to_string(),
            backend: backend.to_string(),
            size: format!("{}", size),
            mean_us: mean,
            std_dev_us: std_dev,
            throughput_mops: (size as f64) / mean,
            dtype1: "i8".to_string(),
            dtype2: None,
            dtype_res: "i8".to_string(),
        });
    }
    
    // Additional cross-dtype tests: u8 + f32 -> f32
    let data_u8_b: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    let data_f32_b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
    let a_u8_b = Array::<u8>::new(vec![size], data_u8_b);
    let b_f32_b = Array::<f32>::new(vec![size], data_f32_b);
    
    if let Ok(_) = ops::add(&a_u8_b, &b_f32_b) {
        let (mean, std_dev) = benchmark_fn("add_u8_f32", || {
            let _ = ops::add(&a_u8_b, &b_f32_b).unwrap();
        }, BENCHMARK_ITERATIONS);
        
        results.push(BenchResult {
            operation: "add_u8_f32".to_string(),
            backend: backend.to_string(),
            size: format!("{}", size),
            mean_us: mean,
            std_dev_us: std_dev,
            throughput_mops: (size as f64) / mean,
            dtype1: "u8".to_string(),
            dtype2: Some("f32".to_string()),
            dtype_res: "f32".to_string(),
        });
    }
    
    // Cross-dtype test: i8 + i32 -> i32
    let data_i8_c: Vec<i8> = (0..size).map(|i| (i % 128) as i8).collect();
    let data_i32_c: Vec<i32> = (0..size).map(|i| i as i32).collect();
    let a_i8_c = Array::<i8>::new(vec![size], data_i8_c);
    let b_i32_c = Array::<i32>::new(vec![size], data_i32_c);
    
    if let Ok(_) = ops::mul(&a_i8_c, &b_i32_c) {
        let (mean, std_dev) = benchmark_fn("mul_i8_i32", || {
            let _ = ops::mul(&a_i8_c, &b_i32_c).unwrap();
        }, BENCHMARK_ITERATIONS);
        
        results.push(BenchResult {
            operation: "mul_i8_i32".to_string(),
            backend: backend.to_string(),
            size: format!("{}", size),
            mean_us: mean,
            std_dev_us: std_dev,
            throughput_mops: (size as f64) / mean,
            dtype1: "i8".to_string(),
            dtype2: Some("i32".to_string()),
            dtype_res: "i32".to_string(),
        });
    }
}

