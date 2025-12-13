//! Benchmark: Type promotion overhead
//!
//! Mide el overhead de la lógica de type promotion:
//! - Caso óptimo: f32 + f32 (fast path)
//! - Caso medio: mismo tipo no-f32 (conversión pero sin promoción)
//! - Caso lento: tipos diferentes (promoción completa)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use numrs::{Array, ops};

fn bench_add_same_f32(c: &mut Criterion) {
    let a = Array::new(vec![1000], vec![1.0_f32; 1000]);
    let b = Array::new(vec![1000], vec![2.0_f32; 1000]);
    
    c.bench_function("add_f32_f32_fast_path", |bench| {
        bench.iter(|| {
            ops::add(black_box(&a), black_box(&b)).unwrap()
        });
    });
}

fn bench_add_same_f64(c: &mut Criterion) {
    let a = Array::new(vec![1000], vec![1.0_f64; 1000]);
    let b = Array::new(vec![1000], vec![2.0_f64; 1000]);
    
    c.bench_function("add_f64_f64_same_type", |bench| {
        bench.iter(|| {
            ops::add(black_box(&a), black_box(&b)).unwrap()
        });
    });
}

fn bench_add_same_i32(c: &mut Criterion) {
    let a = Array::new(vec![1000], vec![1_i32; 1000]);
    let b = Array::new(vec![1000], vec![2_i32; 1000]);
    
    c.bench_function("add_i32_i32_same_type", |bench| {
        bench.iter(|| {
            ops::add(black_box(&a), black_box(&b)).unwrap()
        });
    });
}

fn bench_add_mixed_i32_f32(c: &mut Criterion) {
    let a = Array::new(vec![1000], vec![1_i32; 1000]);
    let b = Array::new(vec![1000], vec![2.0_f32; 1000]);
    
    c.bench_function("add_i32_f32_promotion", |bench| {
        bench.iter(|| {
            ops::add(black_box(&a), black_box(&b)).unwrap()
        });
    });
}

fn bench_add_mixed_f64_f32(c: &mut Criterion) {
    let a = Array::new(vec![1000], vec![1.0_f64; 1000]);
    let b = Array::new(vec![1000], vec![2.0_f32; 1000]);
    
    c.bench_function("add_f64_f32_promotion", |bench| {
        bench.iter(|| {
            ops::add(black_box(&a), black_box(&b)).unwrap()
        });
    });
}

fn bench_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f32_different_sizes");
    
    for size in [100, 1000, 10000].iter() {
        let a = Array::new(vec![*size], vec![1.0_f32; *size]);
        let b = Array::new(vec![*size], vec![2.0_f32; *size]);
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                ops::add(black_box(&a), black_box(&b)).unwrap()
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_add_same_f32,
    bench_add_same_f64,
    bench_add_same_i32,
    bench_add_mixed_i32_f32,
    bench_add_mixed_f64_f32,
    bench_different_sizes
);

criterion_main!(benches);
