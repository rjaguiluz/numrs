//! Type definitions for benchmarking

pub struct BenchResult {
    pub operation: String,
    pub backend: String,
    pub size: String,
    pub mean_us: f64,
    pub std_dev_us: f64,
    pub throughput_mops: f64,
    pub dtype1: String,
    pub dtype2: Option<String>,
    pub dtype_res: String,
}

pub struct HardwareInfo {
    pub cpu: String,
    pub gpu: String,
    pub ram_gb: u64,
    pub os: String,
}

// Configuración de benchmarks
pub const WARMUP_ITERATIONS: usize = 10;
pub const BENCHMARK_ITERATIONS: usize = 50;
