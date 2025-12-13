use std::time::Instant;

// Importamos las funciones C directamente
extern "C" {
    fn numrs_add(
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        len: usize,
        dtype: u32,
    ) -> u32;
    
    fn numrs_mul(
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        len: usize,
        dtype: u32,
    ) -> u32;
}

const DTYPE_F32: u32 = 0;
const DTYPE_F64: u32 = 1;
const DTYPE_I32: u32 = 2;

fn benchmark_op(
    name: &str,
    op_fn: unsafe extern "C" fn(*const std::ffi::c_void, *const std::ffi::c_void, *mut std::ffi::c_void, usize, u32) -> u32,
    size: usize,
    dtype: u32,
    warmup: usize,
    iterations: usize,
) {
    let (a_data, b_data, mut result): (Vec<f32>, Vec<f32>, Vec<f32>) = match dtype {
        DTYPE_F32 => {
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.2).collect();
            let result = vec![0.0f32; size];
            (a, b, result)
        }
        _ => panic!("Only F32 supported in this benchmark"),
    };

    // Warmup
    for _ in 0..warmup {
        unsafe {
            op_fn(
                a_data.as_ptr() as *const std::ffi::c_void,
                b_data.as_ptr() as *const std::ffi::c_void,
                result.as_mut_ptr() as *mut std::ffi::c_void,
                size,
                dtype,
            );
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            op_fn(
                a_data.as_ptr() as *const std::ffi::c_void,
                b_data.as_ptr() as *const std::ffi::c_void,
                result.as_mut_ptr() as *mut std::ffi::c_void,
                size,
                dtype,
            );
        }
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    let throughput = (size as f64 / avg_us) * 1e6 / 1e6; // Mops/s

    println!(
        "{:<10} | size: {:>8} | avg: {:>8.2} μs | throughput: {:>8.2} Mops/s",
        name, size, avg_us, throughput
    );
}

fn main() {
    println!("=== NumRs C API Benchmark ===\n");
    println!("Testing add and mul operations with f32 dtype\n");
    
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];
    let warmup = 10;
    let iterations = 100;

    for &size in &sizes {
        benchmark_op("add", numrs_add, size, DTYPE_F32, warmup, iterations);
    }
    
    println!();
    
    for &size in &sizes {
        benchmark_op("mul", numrs_mul, size, DTYPE_F32, warmup, iterations);
    }

    // Correctness test
    println!("\n=== Correctness Test ===");
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let mut result = vec![0.0f32; 4];
    
    unsafe {
        numrs_add(
            a.as_ptr() as *const std::ffi::c_void,
            b.as_ptr() as *const std::ffi::c_void,
            result.as_mut_ptr() as *mut std::ffi::c_void,
            4,
            DTYPE_F32,
        );
    }
    
    println!("a = {:?}", a);
    println!("b = {:?}", b);
    println!("result = {:?}", result);
    println!("expected = [6.0, 8.0, 10.0, 12.0]");
    
    let expected = vec![6.0, 8.0, 10.0, 12.0];
    let correct = result.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
    println!("Correctness: {}", if correct { "✓ PASSED" } else { "✗ FAILED" });
}
