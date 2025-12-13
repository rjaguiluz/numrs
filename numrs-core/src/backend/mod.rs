//! Backend selection and execution helpers
//!
//! ## Modern API
//! - `dispatch::get_dispatch_table()` - Zero-cost function pointer dispatch
//! - `dispatch::validate_backends()` - Runtime backend validation
//! - `ops::fast::*` - Fast-path operations using dispatch system

pub mod cpu;
// WebGPU module (handles both native and WASM internally)
#[cfg(all(feature = "blas-backend", not(feature = "disabled-blas")))]
pub mod blas;
pub mod capabilities;
pub mod cuda;
pub mod dispatch;
pub mod heuristics;
pub mod metal;
pub mod microbench;
pub mod webgpu; // Sistema de microbenchmarking para kernel selection

/// Selected backend abstraction
#[derive(Debug, Clone)]
pub enum SelectedBackend {
    CPU(cpu::CpuBackend),
    WebGPU(webgpu::WebGpuBackend),
    CUDA(cuda::CudaBackend),
    Metal(metal::MetalBackend),
    #[cfg(all(feature = "blas-backend", not(feature = "disabled-blas")))]
    Blas(blas::BlasBackend),
}

impl SelectedBackend {
    /// Detect a backend automatically. On this prototype we use an env var
    /// `NUMRS_BACKEND` (webgpu, cuda, metal, cpu, blas) or choose best-effort.
    pub fn auto_detect() -> Self {
        use std::env;
        match env::var("NUMRS_BACKEND").ok().as_deref() {
            Some("webgpu") => SelectedBackend::WebGPU(webgpu::WebGpuBackend::new()),
            Some("cuda") => SelectedBackend::CUDA(cuda::CudaBackend::new()),
            Some("metal") => SelectedBackend::Metal(metal::MetalBackend::new()),
            #[cfg(all(feature = "blas-backend", not(feature = "disabled-blas")))]
            Some("blas") => SelectedBackend::Blas(blas::BlasBackend::new()),
            Some("auto") | None => {
                // Use dispatch system to determine best backend
                let validation = dispatch::validate_backends();
                #[cfg(all(feature = "blas-backend", not(feature = "disabled-blas")))]
                if validation.blas_validated {
                    return SelectedBackend::Blas(blas::BlasBackend::new());
                }
                if validation.webgpu_validated {
                    return SelectedBackend::WebGPU(webgpu::WebGpuBackend::new());
                }
                SelectedBackend::CPU(cpu::CpuBackend::new())
            }
            _ => {
                // default fallback to CPU
                SelectedBackend::CPU(cpu::CpuBackend::new())
            }
        }
    }

    // execute_llo() removed - use ops::fast::* functions with dispatch system instead
}

// ============================================================================
// Public API: Modern Dispatch System
// ============================================================================

pub use dispatch::{
    get_dispatch_table, init_dispatch_table, validate_backends, BackendValidation, DispatchTable,
    RuntimeCapabilities,
};

pub use capabilities::{
    BackendCapabilities, BLAS_CAPABILITIES, CUDA_CAPABILITIES, METAL_CAPABILITIES,
    SCALAR_CAPABILITIES, SIMD_CAPABILITIES, WEBGPU_CAPABILITIES,
};
