#!/usr/bin/env cargo
//! NumRs Benchmark Runner
//! 
//! Ejecuta benchmarks de todas las operaciones y genera BENCHMARK.md
//! Benchmarks are run per backend (Scalar, SIMD, BLAS, etc.)
//! Uso: cargo run --bin numrs-bench --release



mod types;
mod hardware;
mod report;
mod benchmarks;

use types::BenchResult;
use hardware::{detect_hardware, generate_benchmark_filename};
use report::generate_markdown_report;
use benchmarks::{
    elementwise::{benchmark_elementwise_binary, benchmark_elementwise_unary},
    reductions::{benchmark_reductions_global, benchmark_reductions_axis},
    linalg::benchmark_linalg,
    shape::benchmark_shape_ops,
};

fn main() {
    println!("🚀 NumRs Benchmark Runner");
    println!("=========================\n");
    
    // Detect hardware
    let hw_info = detect_hardware();
    println!("Hardware detected:");
    println!("  CPU: {}", hw_info.cpu);
    println!("  GPU: {}", hw_info.gpu);
    println!("  RAM: {} GB", hw_info.ram_gb);
    println!("  OS: {}\n", hw_info.os);
    
    // Detect available backends
    let backends = detect_backends();
    println!("Detected backends: {}", backends.join(", "));
    println!();

    let mut results = Vec::new();

    // Run benchmarks for each backend
    for backend in &backends {
        println!("🔧 Testing with backend: {}", backend);
        set_backend_env(backend);
        
        run_all_benchmarks(&mut results, backend);
        println!();
    }

    // Generate markdown report
    let filename = generate_benchmark_filename(&hw_info);
    println!("📝 Generating {}...", filename);
    generate_markdown_report(&results, &hw_info, &filename).unwrap();
    println!("✅ {} generated successfully!", filename);
}

fn detect_backends() -> Vec<String> {
    let mut backends = vec!["Scalar".to_string()];
    
    // Check for SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            backends.push("SIMD".to_string());
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            backends.push("SIMD".to_string());
        }
    }
    
    // Check for BLAS support (MKL on Windows/Linux, Accelerate on macOS)
    #[cfg(any(feature = "blas-backend", target_os = "macos"))]
    {
        backends.push("BLAS".to_string());
    }
    
    // Check for Metal support (macOS only)
    #[cfg(target_os = "macos")]
    {
        backends.push("Metal".to_string());
    }
    
    // Check for WebGPU support (always available on this platform)
    #[cfg(not(target_arch = "wasm32"))]
    {
        backends.push("WebGPU".to_string());
    }
    
    backends
}

fn set_backend_env(backend: &str) {
    use numrs::set_backend_override;
    
    let backend_str = match backend {
        "Scalar" => Some("scalar"),
        "SIMD" => Some("simd"),
        "BLAS" => Some("blas"),
        "Metal" => Some("metal"),
        "WebGPU" => Some("webgpu"),
        _ => None,
    };
    
    set_backend_override(backend_str);
    eprintln!("[BENCH_MAIN] Set backend override to: {:?}", backend_str);
}

fn run_all_benchmarks(results: &mut Vec<BenchResult>, backend: &str) {
    // Define which operations each backend supports natively (no fallback)
    let supported_ops = get_supported_operations(backend);
    
    if supported_ops.contains(&"elementwise_binary") {
        benchmark_elementwise_binary(results, backend);
    }
    
    if supported_ops.contains(&"elementwise_unary") {
        benchmark_elementwise_unary(results, backend);
    }
    
    if supported_ops.contains(&"reduction_global") {
        benchmark_reductions_global(results, backend);
    }
    
    if supported_ops.contains(&"reduction_axis") {
        benchmark_reductions_axis(results, backend);
    }
    
    if supported_ops.contains(&"linalg") {
        benchmark_linalg(results, backend);
    }
    
    if supported_ops.contains(&"shape") {
        benchmark_shape_ops(results, backend);
    }
}

fn get_supported_operations(backend: &str) -> Vec<&'static str> {
    match backend {
        "Scalar" => {
            // Scalar supports everything (it's the fallback)
            vec![
                "elementwise_binary",
                "elementwise_unary",
                "reduction_global",
                "reduction_axis",
                "linalg",
                "shape",
            ]
        },
        "SIMD" => {
            // SIMD only supports elementwise and some reductions
            // Axis reductions and shape ops fall back to scalar
            vec![
                "elementwise_binary",
                "elementwise_unary",
                "reduction_global",
                "linalg", // dot product has SIMD
            ]
        },
        "BLAS" => {
            // BLAS only supports linear algebra and some reductions
            vec![
                "reduction_global", // sum uses BLAS
                "linalg", // dot and matmul use BLAS
            ]
        },
        "Metal" => {
            // Metal supports elementwise and linalg (similar to WebGPU)
            vec![
                "elementwise_binary",
                "elementwise_unary",
                "linalg", // matmul uses Metal
            ]
        },
        "WebGPU" => {
            // WebGPU supports elementwise and matmul
            vec![
                "elementwise_binary",
                "elementwise_unary",
                "linalg", // matmul uses WebGPU
            ]
        },
        _ => vec![],
    }
}
