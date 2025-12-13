use std::time::Instant;
use std::env;

fn make_matrix(n: usize) -> numrs::array::Array {
    let mut data = Vec::with_capacity(n * n);
    for i in 0..(n * n) {
        data.push((i % 100) as f32 + 1.0);
    }
    numrs::array::Array::new(vec![n, n], data)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: sgemm_cpu_micro <dim> <reps>");
        std::process::exit(1);
    }
    let dim: usize = args[1].parse().expect("dim must be int");
    let reps: usize = args[2].parse().expect("reps must be int");

    println!("sgemm_cpu_micro: dim={} reps={}", dim, reps);

    let a = make_matrix(dim);
    let b = make_matrix(dim);

    // warmup
    let _ = numrs::backend::cpu::CpuBackend::matmul_fallback(&a, &b);

    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let out = numrs::backend::cpu::CpuBackend::matmul_fallback(&a, &b);
        let dur = start.elapsed();
        assert_eq!(out.shape(), &[dim, dim]);
        let secs = dur.as_secs_f64();
        times.push(secs);
        println!("rep: {:.6} s", secs);
    }

    let sum: f64 = times.iter().sum();
    let mean = sum / (times.len() as f64);
    println!("mean: {:.6} s", mean);
}
