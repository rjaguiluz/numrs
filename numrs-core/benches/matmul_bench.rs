use criterion::{BenchmarkId, Criterion, Throughput};

use numrs::array::Array;

fn make_matrix(n: usize) -> Array {
    // Fill with deterministic numbers so runs are comparable
    let mut data = Vec::with_capacity(n * n);
    for i in 0..(n * n) {
        data.push((i % 100) as f32 + 1.0);
    }
    Array::new(vec![n, n], data)
}

fn bench_matmul_backend(c: &mut Criterion) {
    // Benchmark with the BLAS backend (blocked GEMM implemented in Rust prototype)
    std::env::set_var("NUMRS_BACKEND", "blas");

    let mut group = c.benchmark_group("matmul_blas");
    // sizes that run reasonably quickly but show growth
    for &n in &[64usize, 128usize, 256usize] {
        let a = make_matrix(n);
        let b = make_matrix(n);

        let id = BenchmarkId::from_parameter(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(id, &n, |bencher, &_n| {
            bencher.iter(|| {
                // call the library API and ensure it runs; unwrap to fail loudly
                let out = numrs::ops::matmul(&a, &b).expect("matmul should succeed");
                // touch result to avoid dead-code elimination
                assert_eq!(out.shape(), &[n, n]);
            })
        });
    }
    group.finish();
}

fn bench_matmul_cpu(c: &mut Criterion) {
    // Benchmark with explicit CPU backend
    std::env::set_var("NUMRS_BACKEND", "cpu");

    let mut group = c.benchmark_group("matmul_cpu");
    for &n in &[64usize, 128usize, 256usize] {
        let a = make_matrix(n);
        let b = make_matrix(n);

        let id = BenchmarkId::from_parameter(n);
        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(id, &n, |bencher, &_n| {
            bencher.iter(|| {
                let out = numrs::ops::matmul(&a, &b).expect("matmul should succeed");
                assert_eq!(out.shape(), &[n, n]);
            })
        });
    }
    group.finish();
}

criterion::criterion_group!(matmul_benches, bench_matmul_backend, bench_matmul_cpu);
criterion::criterion_main!(matmul_benches);
