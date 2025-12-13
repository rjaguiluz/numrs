//! Reduction operations benchmarks

use numrs::{Array, ops};
use crate::types::{BenchResult, BENCHMARK_ITERATIONS};
use crate::benchmarks::benchmark_fn;

pub fn benchmark_reductions_global(results: &mut Vec<BenchResult>, backend: &str) {
    println!("  📊 Reductions (Global)...");
    
    // Test F32
    for &size in &[10_000, 100_000, 1_000_000] {
        for (op_name, op) in &[
            ("sum", ops::sum as fn(&Array, Option<usize>) -> Result<Array, _>),
            ("mean", ops::mean as fn(&Array, Option<usize>) -> Result<Array, _>),
            ("variance", ops::variance as fn(&Array, Option<usize>) -> Result<Array, _>),
            ("max", ops::max as fn(&Array, Option<usize>) -> Result<Array, _>),
        ] {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let a = Array::new(vec![size], data);
            
            let test_result = op(&a, None);
            if test_result.is_ok() {
                let result = test_result.unwrap();
                let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
                
                if result_dtype_str.contains("f32") {
                    let (mean, std_dev) = benchmark_fn(op_name, || {
                        let _ = op(&a, None).unwrap();
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
    
    // Test F64
    for &size in &[10_000, 100_000, 1_000_000] {
        let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01).collect();
        let a = Array::<f64>::new(vec![size], data);
        
        for (op_name, op) in &[
            ("sum", ops::sum as fn(&Array<f64>, Option<usize>) -> Result<Array, _>),
            ("mean", ops::mean as fn(&Array<f64>, Option<usize>) -> Result<Array, _>),
            ("variance", ops::variance as fn(&Array<f64>, Option<usize>) -> Result<Array, _>),
            ("max", ops::max as fn(&Array<f64>, Option<usize>) -> Result<Array, _>),
        ] {
            let test_result = op(&a, None);
            if test_result.is_ok() {
                let result = test_result.unwrap();
                let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
                
                if result_dtype_str.contains("f64") {
                    let (mean, std_dev) = benchmark_fn(op_name, || {
                        let _ = op(&a, None).unwrap();
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
    
    // Test with i32
    let size = 100_000;
    let data_i32: Vec<i32> = (0..size).map(|i| (i % 100) as i32).collect();
    let a_i32 = Array::<i32>::new(vec![size], data_i32);
    
    for (op_name, op) in &[
        ("sum", ops::sum as fn(&Array<i32>, Option<usize>) -> Result<Array, _>),
        ("mean", ops::mean as fn(&Array<i32>, Option<usize>) -> Result<Array, _>),
        ("max", ops::max as fn(&Array<i32>, Option<usize>) -> Result<Array, _>),
    ] {
        let test_result = op(&a_i32, None);
        if test_result.is_ok() {
            let result = test_result.unwrap();
            let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
            
            if result_dtype_str.contains("i32") {
                let (mean, std_dev) = benchmark_fn(op_name, || {
                    let _ = op(&a_i32, None).unwrap();
                }, BENCHMARK_ITERATIONS);
                
                results.push(BenchResult {
                    operation: op_name.to_string(),
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
        }
    }
}

pub fn benchmark_reductions_axis(results: &mut Vec<BenchResult>, backend: &str) {
    println!("  📊 Reductions (Axis)...");
    
    // Test F32
    for &dim in &[500, 1000] {
        for (op_name, axis, op) in &[
            ("sum_axis0", 0, ops::sum as fn(&Array, Option<usize>) -> Result<Array, _>),
            ("mean_axis1", 1, ops::mean as fn(&Array, Option<usize>) -> Result<Array, _>),
        ] {
            let data: Vec<f32> = (0..dim * dim).map(|i| (i as f32) * 0.01).collect();
            let arr_1d = Array::new(vec![dim * dim], data);
            let a = ops::reshape(&arr_1d, &[dim as isize, dim as isize]).unwrap();
            let total_ops = (dim * dim) as f64;
            
            let test_result = op(&a, Some(*axis));
            if test_result.is_ok() {
                let result = test_result.unwrap();
                let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
                
                if result_dtype_str.contains("f32") {
                    let (mean, std_dev) = benchmark_fn(op_name, || {
                        let _ = op(&a, Some(*axis)).unwrap();
                    }, BENCHMARK_ITERATIONS / 4);
                    
                    results.push(BenchResult {
                        operation: op_name.to_string(),
                        backend: backend.to_string(),
                        size: format!("{}x{}", dim, dim),
                        mean_us: mean,
                        std_dev_us: std_dev,
                        throughput_mops: total_ops / mean,
                        dtype1: "f32".to_string(),
                        dtype2: None,
                        dtype_res: "f32".to_string(),
                    });
                }
            }
        }
    }
    
    // Test F64
    for &dim in &[500, 1000] {
        let data: Vec<f64> = (0..dim * dim).map(|i| (i as f64) * 0.01).collect();
        let arr_1d = Array::<f64>::new(vec![dim * dim], data);
        let a = ops::reshape(&arr_1d, &[dim as isize, dim as isize]).unwrap();
        let total_ops = (dim * dim) as f64;
        
        for (op_name, axis, op) in &[
            ("sum_axis0", 0, ops::sum as fn(&Array<f64>, Option<usize>) -> Result<Array, _>),
            ("mean_axis1", 1, ops::mean as fn(&Array<f64>, Option<usize>) -> Result<Array, _>),
        ] {
            let test_result = op(&a, Some(*axis));
            if test_result.is_ok() {
                let result = test_result.unwrap();
                let result_dtype_str = format!("{:?}", result.dtype()).to_lowercase();
                
                if result_dtype_str.contains("f64") {
                    let (mean, std_dev) = benchmark_fn(op_name, || {
                        let _ = op(&a, Some(*axis)).unwrap();
                    }, BENCHMARK_ITERATIONS / 4);
                    
                    results.push(BenchResult {
                        operation: op_name.to_string(),
                        backend: backend.to_string(),
                        size: format!("{}x{}", dim, dim),
                        mean_us: mean,
                        std_dev_us: std_dev,
                        throughput_mops: total_ops / mean,
                        dtype1: "f64".to_string(),
                        dtype2: None,
                        dtype_res: "f64".to_string(),
                    });
                }
            }
        }
    }
    
    // Test I32
    for &dim in &[500] {
        let data: Vec<i32> = (0..dim * dim).map(|i| i as i32).collect();
        let arr_1d = Array::<i32>::new(vec![dim * dim], data);
        let a = ops::reshape(&arr_1d, &[dim as isize, dim as isize]).unwrap();
        let total_ops = (dim * dim) as f64;
        
        for (op_name, axis, op) in &[
            ("sum_axis0_i32", 0, ops::sum as fn(&Array<i32>, Option<usize>) -> Result<Array, _>),
            ("max_axis1_i32", 1, ops::max as fn(&Array<i32>, Option<usize>) -> Result<Array, _>),
        ] {
            if let Ok(_) = op(&a, Some(*axis)) {
                let (mean, std_dev) = benchmark_fn(op_name, || {
                    let _ = op(&a, Some(*axis)).unwrap();
                }, BENCHMARK_ITERATIONS / 4);
                
                results.push(BenchResult {
                    operation: op_name.to_string(),
                    backend: backend.to_string(),
                    size: format!("{}x{}", dim, dim),
                    mean_us: mean,
                    std_dev_us: std_dev,
                    throughput_mops: total_ops / mean,
                    dtype1: "i32".to_string(),
                    dtype2: None,
                    dtype_res: "i32".to_string(),
                });
            }
        }
    }
    
    // Test U8
    let dim = 500;
    let data: Vec<u8> = (0..dim * dim).map(|i| (i % 256) as u8).collect();
    let arr_1d = Array::<u8>::new(vec![dim * dim], data);
    let a = ops::reshape(&arr_1d, &[dim as isize, dim as isize]).unwrap();
    let total_ops = (dim * dim) as f64;
    
    for (op_name, axis, op) in &[
        ("sum_axis0_u8", 0, ops::sum as fn(&Array<u8>, Option<usize>) -> Result<Array, _>),
        ("mean_axis1_u8", 1, ops::mean as fn(&Array<u8>, Option<usize>) -> Result<Array, _>),
    ] {
        if let Ok(_) = op(&a, Some(*axis)) {
            let (mean, std_dev) = benchmark_fn(op_name, || {
                let _ = op(&a, Some(*axis)).unwrap();
            }, BENCHMARK_ITERATIONS / 4);
            
            results.push(BenchResult {
                operation: op_name.to_string(),
                backend: backend.to_string(),
                size: format!("{}x{}", dim, dim),
                mean_us: mean,
                std_dev_us: std_dev,
                throughput_mops: total_ops / mean,
                dtype1: "u8".to_string(),
                dtype2: None,
                dtype_res: "u8".to_string(),
            });
        }
    }
}

