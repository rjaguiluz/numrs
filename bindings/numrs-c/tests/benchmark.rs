// Note: The crate name with hyphens becomes underscores in Rust
extern crate numrs_c;

use numrs_c::*;
use std::time::Instant;

fn benchmark_op<F>(
    name: &str,
    mut op_fn: F,
    size: usize,
    warmup: usize,
    iterations: usize,
) where
    F: FnMut(&[f32], &[f32], &mut [f32]),
{
    let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();
    let mut result = vec![0.0f32; size];

    // Warmup
    for _ in 0..warmup {
        op_fn(&a, &b, &mut result);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        op_fn(&a, &b, &mut result);
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    let throughput = (size as f64 / avg_us) * 1e6 / 1e6; // Mops/s

    println!(
        "{:<10} | size: {:>8} | avg: {:>8.2} μs | throughput: {:>8.2} Mops/s",
        name, size, avg_us, throughput
    );
}

#[test]
fn test_c_api_correctness() {
    println!("\n=== C API Correctness Test ===");
    
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let mut result = vec![0.0f32; 4];
    
    unsafe {
        let status = numrs_add(
            a.as_ptr() as *const std::ffi::c_void,
            b.as_ptr() as *const std::ffi::c_void,
            result.as_mut_ptr() as *mut std::ffi::c_void,
            4,
            NumRsDType::Float32,
        );
        
        assert_eq!(status, NumRsStatus::Ok);
    }
    
    println!("a = {:?}", a);
    println!("b = {:?}", b);
    println!("result = {:?}", result);
    
    let expected = vec![6.0, 8.0, 10.0, 12.0];
    assert_eq!(result, expected);
    println!("✓ Add test passed");
}

#[test]
fn test_c_api_benchmark() {
    println!("\n=== NumRs C API Benchmark ===\n");
    
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];
    let warmup = 10;
    let iterations = 100;

    println!("ADD operations:");
    for &size in &sizes {
        benchmark_op(
            "add",
            |a, b, out| unsafe {
                numrs_add(
                    a.as_ptr() as *const std::ffi::c_void,
                    b.as_ptr() as *const std::ffi::c_void,
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    size,
                    NumRsDType::Float32,
                );
            },
            size,
            warmup,
            iterations,
        );
    }
    
    println!("\nMUL operations:");
    for &size in &sizes {
        benchmark_op(
            "mul",
            |a, b, out| unsafe {
                numrs_mul(
                    a.as_ptr() as *const std::ffi::c_void,
                    b.as_ptr() as *const std::ffi::c_void,
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    size,
                    NumRsDType::Float32,
                );
            },
            size,
            warmup,
            iterations,
        );
    }
}

#[test]
fn test_c_api_matmul_benchmark() {
    println!("\n=== NumRs C API MatMul Benchmark ===\n");
    
    let matrix_sizes = vec![64, 128, 256, 512, 1024, 2048];
    let warmup = 3;
    let iterations = 10;

    println!("MatMul operations (square matrices M×M @ M×M):");
    for &m in &matrix_sizes {
        benchmark_matmul("matmul", m, warmup, iterations);
    }
}

fn benchmark_matmul(name: &str, m: usize, warmup: usize, iterations: usize) {
    // Create square matrices M×M
    let a: Vec<f32> = (0..(m * m)).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..(m * m)).map(|i| (i as f32) * 0.02).collect();
    let mut result = vec![0.0f32; m * m];

    // Warmup
    for _ in 0..warmup {
        unsafe {
            numrs_matmul_f32(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                m,
                m,
                m,
            );
        }
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        unsafe {
            numrs_matmul_f32(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                m,
                m,
                m,
            );
        }
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_micros() as f64 / iterations as f64 / 1000.0;
    
    // Calculate GFLOPS: 2*m^3 operations per matmul
    let flops = 2.0 * (m as f64).powi(3);
    let gflops = (flops / (avg_ms * 1e6)) * 1e3;

    println!(
        "{:<10} | size: {:>4}×{:<4} | avg: {:>10.2} ms | GFLOPS: {:>8.2}",
        name, m, m, avg_ms, gflops
    );
}
