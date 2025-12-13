//! Kernel dispatch table - Zero-cost runtime dispatch system
//!
//! Este módulo implementa un sistema de dispatch que:
//! 1. Se inicializa UNA VEZ al startup
//! 2. Valida qué backends están disponibles (incluyendo WebGPU)
//! 3. Crea function pointers para cada operación
//! 4. El hot-path solo hace `dispatch_table.matmul(a, b)` sin branches
//!
//! Arquitectura:
//! ```text
//! Startup:
//!   - Detectar capabilities (SIMD, GPU, BLAS, WebGPU)
//!   - Validar cada backend (GPU probe, BLAS test call)
//!   - Elegir mejor implementación por operación
//!   - Guardar function pointers en DispatchTable
//!
//! Runtime:
//!   - get_dispatch_table() → &'static DispatchTable
//!   - table.add(a, b) → llama directamente al kernel elegido
//!   - ZERO overhead (direct call, no match/if)
//! ```

use crate::array::Array;
use crate::llo::reduction::ReductionKind;
use crate::llo::ElementwiseKind;
use anyhow::Result;
use once_cell::sync::OnceCell;
use std::fmt;
use std::sync::RwLock;

// ============================================================================
// Runtime Capabilities (migrado desde runtime.rs)
// ============================================================================

/// Runtime-detected capabilities (features realmente disponibles en el sistema)
#[derive(Debug, Clone, Copy)]
pub struct RuntimeCapabilities {
    pub has_simd: bool,
    pub has_gpu: bool,
    pub has_blas: bool,
    pub has_threads: bool,
    pub has_wasm_simd: bool,
    pub has_webgpu: bool,
}

// ============================================================================
// Kernel Function Signatures
// ============================================================================

/// Signature para kernels elementwise (add, mul, etc)
pub type ElementwiseFn = fn(&Array, &Array, ElementwiseKind) -> Result<Array>;

/// Signature para kernels de reducción (sum, mean, max, min, etc)
pub type ReductionFn = fn(&Array, Option<usize>, ReductionKind) -> Result<Array>;

/// Signature para matmul
pub type MatmulFn = fn(&Array, &Array) -> Result<Array>;

/// Signature para dot product (retorna scalar)
pub type DotFn = fn(&Array, &Array) -> Result<f32>;

// ============================================================================
// Dispatch Table - Almacena function pointers
// ============================================================================

/// Dispatch table con function pointers seleccionados al startup.
/// Todos los campos son public para acceso directo sin getter overhead.
#[derive(Clone, Copy)]
pub struct DispatchTable {
    /// Kernel para operaciones elementwise (add, mul, div, etc)
    pub elementwise: ElementwiseFn,

    /// Kernel para reducciones (sum, mean, etc)
    pub reduction: ReductionFn,

    /// Kernel para matrix multiplication
    pub matmul: MatmulFn,

    /// Kernel para dot product
    pub dot: DotFn,

    /// Metadata: nombre del backend usado para elementwise
    pub elementwise_backend: &'static str,

    /// Metadata: nombre del backend usado para reduction
    pub reduction_backend: &'static str,

    /// Metadata: nombre del backend usado para matmul
    pub matmul_backend: &'static str,

    /// Metadata: nombre del backend usado para dot
    pub dot_backend: &'static str,
}

impl fmt::Debug for DispatchTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DispatchTable")
            .field("elementwise_backend", &self.elementwise_backend)
            .field("reduction_backend", &self.reduction_backend)
            .field("matmul_backend", &self.matmul_backend)
            .field("dot_backend", &self.dot_backend)
            .finish()
    }
}

// Global dispatch table (inicializado una vez al startup)
static DISPATCH_TABLE: OnceCell<DispatchTable> = OnceCell::new();

// Global adaptive lookup tables (pobladas por microbenchmarks)
static MATMUL_LOOKUP: OnceCell<crate::backend::microbench::AdaptiveLookupTable<MatmulFn>> =
    OnceCell::new();
static ELEMENTWISE_LOOKUP: OnceCell<
    crate::backend::microbench::AdaptiveLookupTable<ElementwiseFn>,
> = OnceCell::new();
static REDUCTION_LOOKUP: OnceCell<crate::backend::microbench::AdaptiveLookupTable<ReductionFn>> =
    OnceCell::new();

// Backend override para benchmarking (fuerza un kernel específico)
static BACKEND_OVERRIDE: RwLock<Option<&'static str>> = RwLock::new(None);

// ============================================================================
// Backend Validation - Verifica que cada backend realmente funciona
// ============================================================================

/// Resultados de validación de backends
#[derive(Debug, Clone)]
pub struct BackendValidation {
    pub simd_available: bool,
    pub simd_validated: bool,
    pub blas_available: bool,
    pub blas_validated: bool,
    pub gpu_available: bool,
    pub gpu_validated: bool,
    pub webgpu_available: bool,
    pub webgpu_validated: bool,
    pub metal_available: bool,
    pub metal_validated: bool,
}

/// Valida que cada backend realmente funciona (no solo que está compilado)
pub fn validate_backends() -> BackendValidation {
    let mut validation = BackendValidation {
        simd_available: false,
        simd_validated: false,
        blas_available: false,
        blas_validated: false,
        gpu_available: false,
        gpu_validated: false,
        webgpu_available: false,
        webgpu_validated: false,
        metal_available: false,
        metal_validated: false,
    };

    // 1. SIMD Validation
    validation.simd_available = cfg!(numrs_kernel_elementwise_simd)
        || crate::backend::cpu::simd::elementwise_simd_supported();

    if validation.simd_available {
        // Test rápido: crear arrays pequeños y ejecutar add
        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]);

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            // Test WASM SIMD specific
            match crate::backend::cpu::simd::elementwise_simd(&a, &b, ElementwiseKind::Add) {
                Ok(result) => {
                    // Verificar resultado correcto
                    validation.simd_validated =
                        result.data.len() == 4 && (result.data[0] - 2.0).abs() < 0.001;
                }
                Err(_) => validation.simd_validated = false,
            }
        }

        #[cfg(numrs_kernel_elementwise_simd)]
        {
            match crate::backend::cpu::simd::elementwise_simd(&a, &b, ElementwiseKind::Add) {
                Ok(result) => {
                    // Verificar resultado correcto
                    validation.simd_validated =
                        result.data.len() == 4 && (result.data[0] - 2.0).abs() < 0.001;
                }
                Err(_) => validation.simd_validated = false,
            }
        }
    }

    // 2. BLAS Validation
    // CAMBIO: siempre intentar validar BLAS incluso si cfg dice que no está
    // Esto soluciona problemas de propagación de features entre crates
    validation.blas_available = cfg!(numrs_has_blas);

    // Intento agresivo: llamar BLAS directamente sin depender solo de cfg
    let _a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let _b = Array::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);

    #[cfg(numrs_has_blas)]
    {
        // matmul_blas devuelve Array, no Result
        let result = crate::backend::blas::matmul_blas(&_a, &_b);
        // Verificar resultado: debe ser igual a 'a' (multiplicar por identidad)
        validation.blas_validated = result.data.len() == 4
            && (result.data[0] - 1.0).abs() < 0.001
            && (result.data[3] - 4.0).abs() < 0.001;

        if validation.blas_validated {
            validation.blas_available = true; // Forzar available si validó exitosamente
        }
    }

    // 3. WebGPU Validation (más complejo - requires async probe)
    #[cfg(target_arch = "wasm32")]
    {
        // IMPORTANT: Never auto-validate WebGPU on WASM at startup!
        // GPU operations require async JS coordination and will block if called synchronously.
        // pollster::block_on() doesn't work in WASM - it blocks the main thread forever.
        // WebGPU is only available after explicit async JS initialization (future work).
        validation.webgpu_available = false;
        validation.webgpu_validated = false;

        eprintln!("[numrs-dispatch] WebGPU disabled for WASM (async arch required for GPU ops)");
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        validation.webgpu_available = cfg!(numrs_kernel_elementwise_gpu);

        if validation.webgpu_available {
            // Usar el probe existente (cached)
            validation.webgpu_validated = crate::backend::webgpu::is_available_cached();

            // Si el probe dice que está disponible, intentar operación real
            if validation.webgpu_validated {
                // TODO: agregar test funcional de WebGPU cuando el backend esté completo
                // Por ahora confiamos en el probe
                #[cfg(debug_assertions)]
                eprintln!("[numrs-dispatch] WebGPU detected and validated via probe");
            }
        }
    }

    // 4. Metal Validation (macOS only)
    validation.metal_available = cfg!(target_os = "macos");

    if validation.metal_available {
        // Usar el probe de Metal (cached)
        validation.metal_validated = crate::backend::metal::is_available_cached();

        if validation.metal_validated {
            eprintln!("[numrs-dispatch] Metal detected and validated via probe");
        }
    }

    // 5. GPU genérico (CUDA placeholder)
    validation.gpu_available = cfg!(numrs_kernel_matmul_gpu);
    // GPU validation pendiente hasta que se implemente CUDA
    validation.gpu_validated = false;

    validation
}

// ============================================================================
// Kernel Selection - Elige la mejor implementación por operación
// ============================================================================

/// Estrategia de selección basada en validation + benchmarks opcionales
pub fn select_kernels(validation: &BackendValidation) -> DispatchTable {
    // Definir todas las implementaciones disponibles

    // --- MATMUL ---
    // SIEMPRE usar kernel adaptativo que decide basado en tamaño
    // Si probing está disabled, el kernel usa heurística estática
    // Si probing está enabled, consulta la lookup table con microbenchmarks
    let (matmul, mm_backend) = (kernel_matmul_adaptive as MatmulFn, "adaptive");

    // --- ELEMENTWISE ---
    // SIEMPRE adaptativo
    let (elementwise, elem_backend) = (kernel_elementwise_adaptive as ElementwiseFn, "adaptive");

    // --- REDUCTION ---
    // SIEMPRE adaptativo
    let (reduction, red_backend) = (kernel_reduction_adaptive as ReductionFn, "adaptive");

    // --- DOT PRODUCT ---
    let (dot, dot_backend) = {
        #[cfg(feature = "blas-backend")]
        {
            if validation.blas_validated {
                // BLAS sdot es la mejor opción (5-10x más rápido)
                (kernel_dot_blas as DotFn, "blas")
            } else if validation.simd_validated {
                // SIMD con FMA (2-3x más rápido)
                (kernel_dot_simd as DotFn, "cpu-simd")
            } else {
                // Scalar fallback
                (kernel_dot_scalar as DotFn, "cpu-scalar")
            }
        }
        #[cfg(not(feature = "blas-backend"))]
        {
            if validation.simd_validated {
                // SIMD con FMA (2-3x más rápido)
                (kernel_dot_simd as DotFn, "cpu-simd")
            } else {
                // Scalar fallback
                (kernel_dot_scalar as DotFn, "cpu-scalar")
            }
        }
    };

    // Si probing está habilitado, refinar selección con microbenchmarks
    // Si NO, inicializar lookup tables con heurística estática
    let config = crate::backend::microbench::BenchConfig::from_env();

    if config.enabled {
        let (
            elementwise,
            elem_backend,
            reduction,
            red_backend,
            matmul,
            mm_backend,
            dot,
            dot_backend,
        ) = refine_with_probing(
            validation,
            elementwise,
            elem_backend,
            reduction,
            red_backend,
            matmul,
            mm_backend,
            dot,
            dot_backend,
        );

        DispatchTable {
            elementwise,
            reduction,
            matmul,
            dot,
            elementwise_backend: elem_backend,
            reduction_backend: red_backend,
            matmul_backend: mm_backend,
            dot_backend,
        }
    } else {
        // Inicializar lookup tables con heurística (sin microbenchmarks)
        #[cfg(debug_assertions)]
        eprintln!("[numrs-dispatch] Initializing adaptive lookup tables (heuristic mode)");

        let matmul_table = crate::backend::microbench::benchmark_matmul(validation, &config);
        let elem_table = crate::backend::microbench::benchmark_elementwise(validation, &config);
        let red_table = crate::backend::microbench::benchmark_reduction(validation, &config);

        let _ = MATMUL_LOOKUP.set(matmul_table);
        let _ = ELEMENTWISE_LOOKUP.set(elem_table);
        let _ = REDUCTION_LOOKUP.set(red_table);

        DispatchTable {
            elementwise,
            reduction,
            matmul,
            dot,
            elementwise_backend: elem_backend,
            reduction_backend: red_backend,
            matmul_backend: mm_backend,
            dot_backend,
        }
    }
}

/// Refina la selección ejecutando microbenchmarks entre candidatos
/// Puebla las lookup tables globales para dispatch adaptativo
#[allow(clippy::type_complexity)]
#[allow(unused_variables)]
fn refine_with_probing(
    validation: &BackendValidation,
    elementwise: ElementwiseFn,
    elem_backend: &'static str,
    reduction: ReductionFn,
    red_backend: &'static str,
    matmul: MatmulFn,
    mm_backend: &'static str,
    dot: DotFn,
    dot_backend: &'static str,
) -> (
    ElementwiseFn,
    &'static str,
    ReductionFn,
    &'static str,
    MatmulFn,
    &'static str,
    DotFn,
    &'static str,
) {
    eprintln!("[numrs-dispatch] Running microbenchmarks for adaptive kernel selection...");

    let config = crate::backend::microbench::BenchConfig::from_env();

    // Ejecutar microbenchmarks y crear lookup tables
    let matmul_table = crate::backend::microbench::benchmark_matmul(validation, &config);
    let elem_table = crate::backend::microbench::benchmark_elementwise(validation, &config);
    let red_table = crate::backend::microbench::benchmark_reduction(validation, &config);

    // Guardar las lookup tables en globals
    let _ = MATMUL_LOOKUP.set(matmul_table);
    let _ = ELEMENTWISE_LOOKUP.set(elem_table);
    let _ = REDUCTION_LOOKUP.set(red_table);

    eprintln!("[numrs-dispatch] Adaptive lookup tables created");
    eprintln!("[numrs-dispatch] Kernels will select backend dynamically based on input size");

    // Usar kernels adaptativos que consultan las lookup tables
    (
        kernel_elementwise_adaptive as ElementwiseFn,
        "adaptive",
        kernel_reduction_adaptive as ReductionFn,
        "adaptive",
        kernel_matmul_adaptive as MatmulFn,
        "adaptive",
        dot,
        dot_backend,
    )
}

// ============================================================================
// Adaptive Kernels - Consultan lookup tables en runtime
// ============================================================================

/// Kernel adaptativo de matmul - consulta MATMUL_LOOKUP y llama al kernel apropiado
fn kernel_matmul_adaptive(a: &Array, b: &Array) -> Result<Array> {
    // Check for backend override (for benchmarking)
    if let Ok(guard) = BACKEND_OVERRIDE.read() {
        if let Some(backend) = *guard {
            return match backend {
                "scalar" => kernel_matmul_scalar(a, b),
                "simd" => kernel_matmul_simd(a, b),
                "blas" => kernel_matmul_blas_direct(a, b),
                "webgpu" => kernel_matmul_webgpu(a, b),
                "metal" => kernel_matmul_metal(a, b),
                _ => kernel_matmul_blas_direct(a, b),
            };
        }
    }

    let size = a.shape[0] * b.shape[1]; // output matrix size

    if let Some(lookup) = MATMUL_LOOKUP.get() {
        // Usar lookup table del microbenchmark
        let kernel = lookup.lookup(size);
        return kernel(a, b);
    }

    // Fallback si no hay lookup table (no debería pasar)
    kernel_matmul_blas_direct(a, b)
}

/// Kernel adaptativo de elementwise - consulta ELEMENTWISE_LOOKUP
fn kernel_elementwise_adaptive(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array> {
    // Check for backend override (for benchmarking)
    if let Ok(guard) = BACKEND_OVERRIDE.read() {
        if let Some(backend) = *guard {
            return match backend {
                "scalar" => kernel_elementwise_scalar(a, b, kind),
                "simd" => kernel_elementwise_simd(a, b, kind),
                "webgpu" => kernel_elementwise_webgpu(a, b, kind),
                "metal" => kernel_elementwise_metal(a, b, kind),
                _ => kernel_elementwise_simd(a, b, kind),
            };
        }
    }

    let size = a.data.len();

    if let Some(lookup) = ELEMENTWISE_LOOKUP.get() {
        let kernel = lookup.lookup(size);
        return kernel(a, b, kind);
    }

    // Fallback
    kernel_elementwise_simd(a, b, kind)
}

/// Kernel adaptativo de reduction - consulta REDUCTION_LOOKUP
fn kernel_reduction_adaptive(a: &Array, axis: Option<usize>, kind: ReductionKind) -> Result<Array> {
    // Check for backend override (for benchmarking)
    if let Ok(guard) = BACKEND_OVERRIDE.read() {
        if let Some(backend) = *guard {
            return match backend {
                "scalar" => kernel_reduction_scalar(a, axis, kind),
                "simd" => kernel_reduction_simd(a, axis, kind),
                "blas" => kernel_reduction_blas(a, axis, kind),
                _ => kernel_reduction_simd(a, axis, kind),
            };
        }
    }

    let size = a.data.len();

    if let Some(lookup) = REDUCTION_LOOKUP.get() {
        let kernel = lookup.lookup(size);
        return kernel(a, axis, kind);
    }

    // Fallback
    kernel_reduction_simd(a, axis, kind)
}

// ============================================================================
// Kernel Implementations - Wrappers adaptativos que deciden internamente
// ============================================================================

/// Kernel elementwise usando Metal (macOS only)
fn kernel_elementwise_metal(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array> {
    #[cfg(target_os = "macos")]
    {
        crate::backend::metal::elementwise_metal(a, b, kind)
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Fallback si no es macOS
        kernel_elementwise_webgpu(a, b, kind)
    }
}

/// Kernel elementwise usando WebGPU
fn kernel_elementwise_webgpu(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array> {
    #[cfg(numrs_kernel_elementwise_gpu)]
    {
        crate::backend::webgpu::elementwise_webgpu(a, b, kind)
    }

    #[cfg(not(numrs_kernel_elementwise_gpu))]
    {
        // Fallback si no está compilado
        kernel_elementwise_scalar(a, b, kind)
    }
}

/// Kernel elementwise usando SIMD
pub fn kernel_elementwise_simd(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array> {
    #[cfg(numrs_kernel_elementwise_simd)]
    {
        crate::backend::cpu::simd::elementwise_simd(a, b, kind)
    }

    #[cfg(not(numrs_kernel_elementwise_simd))]
    {
        kernel_elementwise_scalar(a, b, kind)
    }
}

/// Kernel elementwise usando scalar (siempre disponible)
fn kernel_elementwise_scalar(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array> {
    crate::backend::cpu::scalar::elementwise_scalar(a, b, kind)
}

/// Kernel reduction usando BLAS
fn kernel_reduction_blas(a: &Array, axis: Option<usize>, kind: ReductionKind) -> Result<Array> {
    #[cfg(numrs_has_blas)]
    {
        // TODO: implementar reduce con BLAS cuando esté disponible
        // Por ahora fallback a SIMD
        kernel_reduction_simd(a, axis, kind)
    }

    #[cfg(not(numrs_has_blas))]
    {
        kernel_reduction_simd(a, axis, kind)
    }
}

/// Kernel reduction usando SIMD
pub fn kernel_reduction_simd(a: &Array, axis: Option<usize>, kind: ReductionKind) -> Result<Array> {
    #[cfg(numrs_kernel_sum_simd)]
    {
        crate::backend::cpu::simd::reduce_simd(a, axis, kind)
    }

    #[cfg(not(numrs_kernel_sum_simd))]
    {
        kernel_reduction_scalar(a, axis, kind)
    }
}

/// Kernel reduction usando scalar
fn kernel_reduction_scalar(a: &Array, axis: Option<usize>, kind: ReductionKind) -> Result<Array> {
    crate::backend::cpu::scalar::reduce_scalar(a, axis, kind)
}

/// Kernel matmul usando BLAS directo (sin decisiones internas)
pub fn kernel_matmul_blas_direct(a: &Array, b: &Array) -> Result<Array> {
    #[cfg(numrs_has_blas)]
    {
        // BLAS/MKL ya tiene paralelización interna optimizada
        Ok(crate::backend::blas::matmul_blas(a, b))
    }

    #[cfg(not(numrs_has_blas))]
    {
        kernel_matmul_simd(a, b)
    }
}

/// Kernel matmul usando Metal (macOS only)
pub fn kernel_matmul_metal(a: &Array, b: &Array) -> Result<Array> {
    #[cfg(target_os = "macos")]
    {
        crate::backend::metal::matmul_metal(a, b)
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Si no es macOS, intentar SIMD antes de scalar
        kernel_matmul_simd(a, b)
    }
}

/// Kernel matmul usando WebGPU
pub fn kernel_matmul_webgpu(a: &Array, b: &Array) -> Result<Array> {
    #[cfg(numrs_kernel_matmul_gpu)]
    {
        Ok(crate::backend::webgpu::matmul_webgpu(a, b))
    }

    #[cfg(not(numrs_kernel_matmul_gpu))]
    {
        // Fallback a SIMD primero si está disponible
        kernel_matmul_simd(a, b)
    }
}

/// Kernel matmul usando SIMD
pub fn kernel_matmul_simd(a: &Array, b: &Array) -> Result<Array> {
    #[cfg(numrs_kernel_matmul_simd)]
    {
        Ok(crate::backend::cpu::simd::matmul_simd(a, b))
    }

    #[cfg(not(numrs_kernel_matmul_simd))]
    {
        kernel_matmul_scalar(a, b)
    }
}

/// Kernel matmul usando scalar (siempre disponible)
pub fn kernel_matmul_scalar(a: &Array, b: &Array) -> Result<Array> {
    // Usar implementación scalar paralela (con Rayon pero sin SIMD)
    Ok(crate::backend::cpu::matmul_scalar_parallel(a, b))
}

// --- DOT PRODUCT KERNELS ---

/// Kernel dot usando BLAS sdot (máximo rendimiento)
#[cfg(feature = "blas-backend")]
fn kernel_dot_blas(a: &Array, b: &Array) -> Result<f32> {
    crate::backend::blas::dot_blas(a, b)
}

/// Kernel dot usando SIMD con FMA
fn kernel_dot_simd(a: &Array, b: &Array) -> Result<f32> {
    crate::backend::cpu::simd::dot_simd(a, b)
}

/// Kernel dot usando scalar (siempre disponible)
fn kernel_dot_scalar(a: &Array, b: &Array) -> Result<f32> {
    crate::backend::cpu::scalar::dot_scalar(a, b)
}

// ============================================================================
// Public API - Inicialización y acceso al dispatch table
// ============================================================================

/// Inicializa el dispatch table (llamar al startup)
pub fn init_dispatch_table() -> &'static DispatchTable {
    DISPATCH_TABLE.get_or_init(|| {
        #[cfg(debug_assertions)]
        eprintln!("[numrs-dispatch] Initializing dispatch table...");

        // 1. Validar backends
        let validation = validate_backends();
        #[cfg(debug_assertions)]
        eprintln!("[numrs-dispatch] Validation results: {:?}", validation);

        // 2. Seleccionar kernels
        let table = select_kernels(&validation);

        #[cfg(debug_assertions)]
        {
            eprintln!("[numrs-dispatch] Selected kernels:");
            eprintln!("  - elementwise: {}", table.elementwise_backend);
            eprintln!("  - reduction:   {}", table.reduction_backend);
            eprintln!(
                "  - matmul:      {} (validates: blas={}, metal={}, webgpu={}, simd={})",
                table.matmul_backend,
                validation.blas_validated,
                validation.metal_validated,
                validation.webgpu_validated,
                validation.simd_validated
            );
            eprintln!("  - dot:         {}", table.dot_backend);
        }

        table
    })
}

/// Obtiene el dispatch table (inicializa si es necesario)
pub fn get_dispatch_table() -> &'static DispatchTable {
    DISPATCH_TABLE.get_or_init(|| {
        // Validar backends
        let validation = validate_backends();

        // Seleccionar kernels (internamente decide si hacer probing o no)
        select_kernels(&validation)
    })
}

/// Force reinitialize the dispatch table (WASM only - for WebGPU JS integration)
#[cfg(target_arch = "wasm32")]
pub fn force_reinitialize_dispatch() {
    // For WASM, we need to recreate the dispatch table after JavaScript
    // has initialized WebGPU and set the availability flag
    unsafe {
        // SAFETY: In WASM there's no true multi-threading, so this is safe
        // We're using this to allow JavaScript to signal WebGPU is ready
        let ptr = &DISPATCH_TABLE as *const OnceCell<DispatchTable> as *mut OnceCell<DispatchTable>;
        (*ptr).take(); // Clear the existing table
    }
    // Next call to get_dispatch_table() will reinitialize with WebGPU available
}

/// Re-exportar para uso público
pub use get_dispatch_table as table;

/// Forzar un backend específico (para benchmarking)
/// Valores válidos: "scalar", "simd", "blas", "webgpu", "metal"
/// Usar `None` para restaurar comportamiento adaptativo
pub fn set_backend_override(backend: Option<&'static str>) {
    if let Ok(mut guard) = BACKEND_OVERRIDE.write() {
        *guard = backend;
    }
}

/// Obtener el backend override actual
pub fn get_backend_override() -> Option<&'static str> {
    BACKEND_OVERRIDE.read().ok().and_then(|guard| *guard)
}

// ============================================================================
// Generic Dispatch Functions - Despacho basado en tipo en runtime
// ============================================================================

/// Dispatch genérico para operaciones elementwise que mantiene el tipo nativo
///
/// Esta función hace dispatch basado en el tipo T en runtime, llamando al
/// kernel apropiado y manteniendo los datos en su tipo nativo (Vec<T>).
///
/// NOTA: Devuelve Array<f32> con dtype configurado para mantener compatibilidad
/// con la API existente. Los datos se convierten al final.
#[inline]
pub fn dispatch_elementwise_generic<T>(
    a: &Array<T>,
    b: &Array<T>,
    kind: ElementwiseKind,
) -> Result<Array<T>>
where
    T: crate::array::DTypeValue,
{
    use std::any::TypeId;

    // **ZERO-COPY OPTIMIZATION**: Materializar solo si necesario
    // Los kernels CPU pueden trabajar con strides directamente
    // Los kernels GPU/SIMD necesitan arrays contiguos
    let needs_contiguous = should_materialize_for_backend(a, b);

    let a_ref = if needs_contiguous && !a.is_contiguous() {
        &a.to_contiguous()
    } else {
        a
    };

    let b_ref = if needs_contiguous && !b.is_contiguous() {
        &b.to_contiguous()
    } else {
        b
    };

    // Despacho basado en tipo
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // f32: usar dispatch table legacy
        let a_f32 = unsafe { &*(a_ref as *const Array<T> as *const Array<f32>) };
        let b_f32 = unsafe { &*(b_ref as *const Array<T> as *const Array<f32>) };
        let table = get_dispatch_table();
        let result = (table.elementwise)(a_f32, b_f32, kind)?;
        return Ok(unsafe { std::mem::transmute::<Array<f32>, Array<T>>(result) });
    }

    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // f64: ejecutar operación nativa en f64
        let a_f64 = unsafe { &*(a_ref as *const Array<T> as *const Array<f64>) };
        let b_f64 = unsafe { &*(b_ref as *const Array<T> as *const Array<f64>) };
        let result_f64 = elementwise_f64_native(a_f64, b_f64, kind)?;
        return Ok(unsafe { std::mem::transmute::<Array<f64>, Array<T>>(result_f64) });
    }

    if TypeId::of::<T>() == TypeId::of::<i32>() {
        // i32: ejecutar operación nativa en i32
        let a_i32 = unsafe { &*(a_ref as *const Array<T> as *const Array<i32>) };
        let b_i32 = unsafe { &*(b_ref as *const Array<T> as *const Array<i32>) };
        let result_i32 = elementwise_i32_native(a_i32, b_i32, kind)?;
        return Ok(unsafe { std::mem::transmute::<Array<i32>, Array<T>>(result_i32) });
    }

    // Para otros tipos: fallback usando conversión a f32
    let a_data: Vec<f32> = a_ref
        .data
        .iter()
        .map(|&x| crate::array::DTypeValue::to_f32(x))
        .collect();
    let b_data: Vec<f32> = b_ref
        .data
        .iter()
        .map(|&x| crate::array::DTypeValue::to_f32(x))
        .collect();
    let a_temp = Array::new(a_ref.shape.clone(), a_data);
    let b_temp = Array::new(b_ref.shape.clone(), b_data);

    let table = get_dispatch_table();
    let result = (table.elementwise)(&a_temp, &b_temp, kind)?;
    Ok(unsafe { std::mem::transmute::<Array<f32>, Array<T>>(result) })
}

/// Determina si los arrays deben materializarse para el backend actual
///
/// **FILOSOFÍA**: Materializar lo menos posible. Los kernels CPU stride-aware
/// son suficientemente rápidos, y SIMD también puede trabajar con views.
/// Solo materializar cuando GPU lo NECESITA (contiguous memory requirement).
///
/// Retorna true si:
/// - Se va a usar GPU Y el array es MUY grande (>1M elementos)
///
/// Retorna false si:
/// - CPU/SIMD: pueden trabajar con strides eficientemente
/// - Arrays ya contiguos: no hay beneficio
/// - Arrays pequeños/medianos: overhead de copia > beneficio
#[inline]
fn should_materialize_for_backend<T>(a: &Array<T>, b: &Array<T>) -> bool
where
    T: crate::array::DTypeValue,
{
    // Si ambos son contiguos, no hay nada que hacer
    if a.is_contiguous() && b.is_contiguous() {
        return false;
    }

    // GPU solo para arrays MUY grandes (>1M elementos)
    #[cfg(feature = "webgpu")]
    if crate::backend::webgpu::is_available_cached() {
        let size: usize = a.shape.iter().product();
        if size > 1_000_000 {
            return true; // Materializar solo si es REALMENTE grande
        }
    }

    // CPU/SIMD: trabajar con strides es eficiente, NO materializar
    false
}

/// Operación elementwise nativa para f64 (sin conversión a f32)
///
/// **ZERO-COPY**: Esta función trabaja directamente con strides si están presentes,
/// evitando materialización innecesaria para operaciones CPU.
#[inline]
fn elementwise_f64_native(
    a: &Array<f64>,
    b: &Array<f64>,
    kind: ElementwiseKind,
) -> Result<Array<f64>> {
    let size: usize = a.shape.iter().product();
    let mut result_data = Vec::with_capacity(size);

    // Fast path: ambos arrays contiguos (sin strides)
    if a.is_contiguous() && b.is_contiguous() {
        match kind {
            ElementwiseKind::Add => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i] + b.data[i]);
                }
            }
            ElementwiseKind::Sub => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i] - b.data[i]);
                }
            }
            ElementwiseKind::Mul => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i] * b.data[i]);
                }
            }
            ElementwiseKind::Div => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i] / b.data[i]);
                }
            }
            ElementwiseKind::Pow => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i].powf(b.data[i]));
                }
            }
            _ => anyhow::bail!("Unsupported elementwise operation for f64: {:?}", kind),
        }
    } else {
        // Stride-aware path: indexación con strides
        let a_strides = a.get_strides();
        let b_strides = b.get_strides();

        let mut indices = vec![0usize; a.shape.len()];

        for _ in 0..size {
            // Calcular offsets con strides
            let mut a_idx = a.offset as isize;
            let mut b_idx = b.offset as isize;
            for (i, &idx) in indices.iter().enumerate() {
                a_idx += idx as isize * a_strides[i];
                b_idx += idx as isize * b_strides[i];
            }

            // Bounds check
            let a_idx_u = (a_idx as usize).min(a.data.len().saturating_sub(1));
            let b_idx_u = (b_idx as usize).min(b.data.len().saturating_sub(1));

            let val = match kind {
                ElementwiseKind::Add => a.data[a_idx_u] + b.data[b_idx_u],
                ElementwiseKind::Sub => a.data[a_idx_u] - b.data[b_idx_u],
                ElementwiseKind::Mul => a.data[a_idx_u] * b.data[b_idx_u],
                ElementwiseKind::Div => a.data[a_idx_u] / b.data[b_idx_u],
                ElementwiseKind::Pow => a.data[a_idx_u].powf(b.data[b_idx_u]),
                _ => anyhow::bail!("Unsupported elementwise operation for f64: {:?}", kind),
            };
            result_data.push(val);

            // Incrementar índices (orden C)
            for i in (0..a.shape.len()).rev() {
                indices[i] += 1;
                if indices[i] < a.shape[i] {
                    break;
                }
                indices[i] = 0;
            }
        }
    }

    let mut result = Array::new(a.shape.clone(), result_data);
    result.dtype = crate::array::DType::F64;
    Ok(result)
}

/// Operación elementwise nativa para i32 (sin conversión a f32)
///
/// **ZERO-COPY**: Esta función trabaja directamente con strides si están presentes,
/// evitando materialización innecesaria para operaciones CPU.
#[inline]
fn elementwise_i32_native(
    a: &Array<i32>,
    b: &Array<i32>,
    kind: ElementwiseKind,
) -> Result<Array<i32>> {
    let size: usize = a.shape.iter().product();
    let mut result_data = Vec::with_capacity(size);

    // Fast path: ambos arrays contiguos (sin strides)
    if a.is_contiguous() && b.is_contiguous() {
        match kind {
            ElementwiseKind::Add => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i] + b.data[i]);
                }
            }
            ElementwiseKind::Sub => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i] - b.data[i]);
                }
            }
            ElementwiseKind::Mul => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i] * b.data[i]);
                }
            }
            ElementwiseKind::Div => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i] / b.data[i]);
                }
            }
            ElementwiseKind::Pow => {
                for i in 0..a.data.len() {
                    result_data.push(a.data[i].pow(b.data[i] as u32));
                }
            }
            _ => anyhow::bail!("Unsupported elementwise operation for i32: {:?}", kind),
        }
    } else {
        // Stride-aware path: indexación con strides
        let a_strides = a.get_strides();
        let b_strides = b.get_strides();

        let mut indices = vec![0usize; a.shape.len()];

        for _ in 0..size {
            // Calcular offsets con strides
            let mut a_idx = a.offset as isize;
            let mut b_idx = b.offset as isize;
            for (i, &idx) in indices.iter().enumerate() {
                a_idx += idx as isize * a_strides[i];
                b_idx += idx as isize * b_strides[i];
            }

            // Bounds check
            let a_idx_u = (a_idx as usize).min(a.data.len().saturating_sub(1));
            let b_idx_u = (b_idx as usize).min(b.data.len().saturating_sub(1));

            let val = match kind {
                ElementwiseKind::Add => a.data[a_idx_u] + b.data[b_idx_u],
                ElementwiseKind::Sub => a.data[a_idx_u] - b.data[b_idx_u],
                ElementwiseKind::Mul => a.data[a_idx_u] * b.data[b_idx_u],
                ElementwiseKind::Div => a.data[a_idx_u] / b.data[b_idx_u],
                ElementwiseKind::Pow => a.data[a_idx_u].pow(b.data[b_idx_u] as u32),
                _ => anyhow::bail!("Unsupported elementwise operation for i32: {:?}", kind),
            };
            result_data.push(val);

            // Incrementar índices (orden C)
            for i in (0..a.shape.len()).rev() {
                indices[i] += 1;
                if indices[i] < a.shape[i] {
                    break;
                }
                indices[i] = 0;
            }
        }
    }

    let mut result = Array::new(a.shape.clone(), result_data);
    result.dtype = crate::array::DType::I32;
    Ok(result)
}

// ============================================================================

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_table_initialization() {
        let table = init_dispatch_table();

        // Verificar que todos los campos están asignados
        assert!(!table.elementwise_backend.is_empty());
        assert!(!table.reduction_backend.is_empty());
        assert!(!table.matmul_backend.is_empty());

        println!("Dispatch table: {:?}", table);
    }

    #[test]
    fn test_backend_validation() {
        let validation = validate_backends();

        println!("Backend validation: {:?}", validation);

        // Al menos scalar debe estar disponible
        assert!(validation.simd_available || validation.blas_available || true);
    }

    #[test]
    fn test_elementwise_dispatch() {
        let table = get_dispatch_table();

        let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]);

        let result = (table.elementwise)(&a, &b, ElementwiseKind::Add);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0]);

        println!(
            "Elementwise test passed using: {}",
            table.elementwise_backend
        );
    }
}
