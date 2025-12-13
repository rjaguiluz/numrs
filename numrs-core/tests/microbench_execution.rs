use numrs::backend::dispatch::validate_backends;
use numrs::backend::microbench::{benchmark_matmul, AdaptiveLookupTable, BenchConfig};

#[test]
fn test_microbench_execution() {
    // Manually run benchmark suite
    let validation = validate_backends();

    // Config that enables benchmarking
    let config = BenchConfig {
        enabled: true,
        iterations: 1,     // fast
        max_time_ms: 1000, // plenty of time
    };

    let table = benchmark_matmul(&validation, &config);

    // Check table has ranges
    assert!(!table.ranges.is_empty());

    // Check lookup
    let _kernel = table.lookup(100);
    let backend = table.backend_name(100);
    println!("Selected backend for size 100: {}", backend);
}

#[test]
fn test_microbench_execution_disabled() {
    let validation = validate_backends();
    let config = BenchConfig {
        enabled: false,
        iterations: 1,
        max_time_ms: 50,
    };

    let table = benchmark_matmul(&validation, &config);
    assert!(!table.ranges.is_empty());

    // Should fallback to static heuristics (likely "cpu-simd" or "cpu-scalar")
    let backend = table.backend_name(10);
    println!("Static backend for size 10: {}", backend);
}
