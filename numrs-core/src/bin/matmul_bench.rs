//! Small runner for Criterion benchmarks so they can be executed on stable
//! using `cargo run --release --bin matmul_bench`.

use std::time::{Duration, Instant};
use numrs::array::Array;

fn make_matrix(n: usize) -> Array {
    let mut data = Vec::with_capacity(n * n);
    for i in 0..(n * n) {
        data.push((i % 100) as f32 + 1.0);
    }
    Array::new(vec![n, n], data)
}

fn time_matmul(n: usize, backend: &str, iterations: usize) -> Duration {
    std::env::set_var("NUMRS_BACKEND", backend);
    let a = make_matrix(n);
    let b = make_matrix(n);

    // Warmup single run
    let _ = numrs::ops::matmul(&a, &b).expect("warmup matmul failed");

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = numrs::ops::matmul(&a, &b).expect("matmul failed");
    }
    start.elapsed()
}

fn main() {
    // sizes and iterations tuned for development machines — increase for thorough profiling
    let sizes = [64usize, 128usize, 256usize];
    let iterations = 10usize;

    println!("Running simple matmul microbenchmarks ({} iterations each)\n", iterations);

    for &backend in ["blas", "cpu"].iter() {
        println!("Backend: {}", backend);
        for &n in sizes.iter() {
            let d = time_matmul(n, backend, iterations);
            let avg = d / iterations as u32;
            println!("  size={:4} total={:?} avg per op={:?}", n, d, avg);
        }
        println!();
    }
}
