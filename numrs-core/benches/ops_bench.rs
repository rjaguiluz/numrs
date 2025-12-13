use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs::array::Array;
use numrs::ops;

// ============================================================================
// Helper functions to create test data
// ============================================================================

fn make_vector(n: usize) -> Array {
    let data: Vec<f32> = (0..n).map(|i| (i % 100) as f32 + 1.0).collect();
    Array::new(vec![n], data)
}

fn make_matrix(rows: usize, cols: usize) -> Array {
    let data: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 + 1.0).collect();
    Array::new(vec![rows, cols], data)
}

fn make_tensor_3d(d1: usize, d2: usize, d3: usize) -> Array {
    let data: Vec<f32> = (0..d1 * d2 * d3).map(|i| (i % 100) as f32 + 1.0).collect();
    Array::new(vec![d1, d2, d3], data)
}

// ============================================================================
// ELEMENTWISE BENCHMARKS
// ============================================================================

fn bench_elementwise_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_binary");
    
    for &size in &[1000, 10_000, 100_000] {
        let a = make_vector(size);
        let b = make_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Add
        group.bench_with_input(BenchmarkId::new("add", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::add(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
        
        // Multiply
        group.bench_with_input(BenchmarkId::new("mul", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::mul(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
        
        // Sub
        group.bench_with_input(BenchmarkId::new("sub", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::sub(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
        
        // Div
        group.bench_with_input(BenchmarkId::new("div", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::div(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_elementwise_unary(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise_unary");
    
    for &size in &[1000, 10_000, 100_000] {
        let a = make_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Trigonometric
        group.bench_with_input(BenchmarkId::new("sin", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::sin(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("cos", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::cos(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("tan", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::tan(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        // Exponential
        group.bench_with_input(BenchmarkId::new("exp", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::exp(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("log", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::log(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("sqrt", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::sqrt(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        // Activation functions
        group.bench_with_input(BenchmarkId::new("relu", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::relu(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("sigmoid", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::sigmoid(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("tanh", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::tanh(black_box(&a)).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// REDUCTION BENCHMARKS
// ============================================================================

fn bench_reductions_global(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_global");
    
    for &size in &[1000, 10_000, 100_000, 1_000_000] {
        let a = make_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Sum
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::sum(black_box(&a), None).unwrap();
                black_box(result);
            });
        });
        
        // Mean
        group.bench_with_input(BenchmarkId::new("mean", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::mean(black_box(&a), None).unwrap();
                black_box(result);
            });
        });
        
        // Variance (optimized with Welford)
        group.bench_with_input(BenchmarkId::new("variance", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::variance(black_box(&a), None).unwrap();
                black_box(result);
            });
        });
        
        // Max
        group.bench_with_input(BenchmarkId::new("max", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::max(black_box(&a), None).unwrap();
                black_box(result);
            });
        });
        
        // Min
        group.bench_with_input(BenchmarkId::new("min", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::min(black_box(&a), None).unwrap();
                black_box(result);
            });
        });
        
        // ArgMax
        group.bench_with_input(BenchmarkId::new("argmax", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::argmax(black_box(&a), None).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_reductions_axis(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_axis");
    
    for &size in &[100, 500, 1000] {
        let a = make_matrix(size, size);
        let total_elements = size * size;
        
        group.throughput(Throughput::Elements(total_elements as u64));
        
        // Sum along axis 0
        group.bench_with_input(BenchmarkId::new("sum_axis0", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::sum(black_box(&a), Some(0)).unwrap();
                black_box(result);
            });
        });
        
        // Mean along axis 1
        group.bench_with_input(BenchmarkId::new("mean_axis1", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::mean(black_box(&a), Some(1)).unwrap();
                black_box(result);
            });
        });
        
        // Variance along axis 0
        group.bench_with_input(BenchmarkId::new("variance_axis0", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::variance(black_box(&a), Some(0)).unwrap();
                black_box(result);
            });
        });
        
        // ArgMax along axis 1
        group.bench_with_input(BenchmarkId::new("argmax_axis1", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::argmax(black_box(&a), Some(1)).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// LINALG BENCHMARKS
// ============================================================================

fn bench_linalg_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_dot");
    
    for &size in &[100, 1000, 10_000, 100_000, 1_000_000] {
        let a = make_vector(size);
        let b = make_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::dot(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_linalg_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_matmul");
    
    for &size in &[64, 128, 256, 512] {
        let a = make_matrix(size, size);
        let b = make_matrix(size, size);
        
        // FLOPS = 2*n^3 for square matrix multiplication
        group.throughput(Throughput::Elements((2 * size * size * size) as u64));
        
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::matmul(black_box(&a), black_box(&b)).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// SHAPE BENCHMARKS
// ============================================================================

fn bench_shape_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape_ops");
    
    for &size in &[100, 500, 1000] {
        let a = make_matrix(size, size);
        let total_elements = size * size;
        
        group.throughput(Throughput::Elements(total_elements as u64));
        
        // Transpose
        group.bench_with_input(BenchmarkId::new("transpose", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::transpose(black_box(&a), None).unwrap();
                black_box(result);
            });
        });
        
        // Reshape
        group.bench_with_input(BenchmarkId::new("reshape", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::reshape(black_box(&a), &[1, total_elements as isize]).unwrap();
                black_box(result);
            });
        });
        
        // Concat (concatenate two matrices along axis 0)
        let b = make_matrix(size, size);
        group.bench_with_input(BenchmarkId::new("concat", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::concat(&[black_box(&a), black_box(&b)], 0).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// STATS BENCHMARKS
// ============================================================================

fn bench_stats_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_ops");
    
    for &size in &[100, 1000, 10_000] {
        let a = make_vector(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Norm (L2 norm)
        group.bench_with_input(BenchmarkId::new("norm", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::norm(black_box(&a)).unwrap();
                black_box(result);
            });
        });
        
        // Softmax
        group.bench_with_input(BenchmarkId::new("softmax", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::softmax(black_box(&a), None).unwrap();
                black_box(result);
            });
        });
        
        // Cross entropy (needs positive values)
        let predictions = make_vector(size);
        let targets = make_vector(size);
        group.bench_with_input(BenchmarkId::new("cross_entropy", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = ops::cross_entropy(black_box(&predictions), black_box(&targets)).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// COMBINED ML WORKFLOW BENCHMARKS
// ============================================================================

fn bench_ml_workflow(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_workflow");
    
    // Batch sizes: small (32), medium (128), large (512)
    for &batch_size in &[32, 128, 512] {
        let features = 784; // MNIST-like
        let hidden = 256;
        let classes = 10;
        
        // Simulate a forward pass: matmul -> relu -> matmul -> softmax
        let input = make_matrix(batch_size, features);
        let w1 = make_matrix(features, hidden);
        let w2 = make_matrix(hidden, classes);
        
        group.throughput(Throughput::Elements((batch_size * features) as u64));
        
        group.bench_with_input(BenchmarkId::new("forward_pass", batch_size), &batch_size, |bencher, _| {
            bencher.iter(|| {
                // Layer 1: matmul + relu
                let h1 = ops::matmul(black_box(&input), black_box(&w1)).unwrap();
                let h1_act = ops::relu(&h1.to_f32()).unwrap();
                
                // Layer 2: matmul + softmax
                let logits = ops::matmul(&h1_act.to_f32(), black_box(&w2)).unwrap();
                let output = ops::softmax(&logits.to_f32(), Some(1)).unwrap();
                
                black_box(output);
            });
        });
    }
    
    group.finish();
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    elementwise_benches,
    bench_elementwise_binary,
    bench_elementwise_unary
);

criterion_group!(
    reduction_benches,
    bench_reductions_global,
    bench_reductions_axis
);

criterion_group!(
    linalg_benches,
    bench_linalg_dot,
    bench_linalg_matmul
);

criterion_group!(
    shape_benches,
    bench_shape_ops
);

criterion_group!(
    stats_benches,
    bench_stats_ops
);

criterion_group!(
    ml_benches,
    bench_ml_workflow
);

criterion_main!(
    elementwise_benches,
    reduction_benches,
    linalg_benches,
    shape_benches,
    stats_benches,
    ml_benches
);
