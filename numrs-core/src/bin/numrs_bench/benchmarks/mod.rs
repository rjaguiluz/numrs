//! Benchmark modules

pub mod elementwise;
pub mod reductions;
pub mod linalg;
pub mod shape;

use std::time::Instant;
use crate::types::WARMUP_ITERATIONS;

/// Benchmark a function with warmup and iterations
pub fn benchmark_fn<F>(_name: &str, mut f: F, iterations: usize) -> (f64, f64)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        f();
    }

    // Measure
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1_000_000.0); // microseconds
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    let std_dev = variance.sqrt();

    (mean, std_dev)
}
