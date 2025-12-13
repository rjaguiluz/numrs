//! Microbenchmarking system for kernel selection
//!
//! Este módulo ejecuta microbenchmarks al startup para determinar
//! qué backend usar para cada operación basado en el tamaño de los datos.
//!
//! Estrategia:
//! 1. Correr benchmarks rápidos (1-5ms cada uno) con diferentes tamaños
//! 2. Crear un DICCIONARIO de rangos → function pointers
//! 3. En runtime, hacer lookup O(log n) en el diccionario
//!
//! Ejemplo de diccionario para matmul:
//! - [0, 4096): kernel_simd
//! - [4096, 262144): kernel_blas  
//! - [262144, ∞): kernel_gpu

use crate::array::Array;
use crate::backend::dispatch::{BackendValidation, ElementwiseFn, MatmulFn, ReductionFn};
use std::time::Instant;

/// Rango de tamaños con su función asociada
#[derive(Debug, Clone, Copy)]
pub struct SizeRange<F: Copy> {
    /// Tamaño mínimo (inclusive)
    pub min_size: usize,
    /// Tamaño máximo (exclusive), None = infinito
    pub max_size: Option<usize>,
    /// Function pointer para este rango
    pub kernel: F,
    /// Nombre del backend para logging
    pub backend_name: &'static str,
}

/// Lookup table: lista ordenada de rangos
/// En runtime, se busca linealmente (O(n) pero n es pequeño, ~3-5 rangos)
#[derive(Debug, Clone)]
pub struct AdaptiveLookupTable<F: Copy> {
    pub ranges: Vec<SizeRange<F>>,
}

impl<F: Copy> AdaptiveLookupTable<F> {
    /// Buscar el kernel apropiado para un tamaño dado
    #[inline]
    pub fn lookup(&self, size: usize) -> F {
        for range in &self.ranges {
            if size >= range.min_size {
                if let Some(max) = range.max_size {
                    if size < max {
                        return range.kernel;
                    }
                } else {
                    // Sin límite superior, este es el último rango
                    return range.kernel;
                }
            }
        }

        // Fallback: primer rango (no debería llegar aquí si la tabla está bien construida)
        self.ranges[0].kernel
    }

    /// Obtener nombre del backend para un tamaño
    pub fn backend_name(&self, size: usize) -> &'static str {
        for range in &self.ranges {
            if size >= range.min_size {
                if let Some(max) = range.max_size {
                    if size < max {
                        return range.backend_name;
                    }
                } else {
                    return range.backend_name;
                }
            }
        }
        self.ranges[0].backend_name
    }
}

/// Configuración de microbenchmarking
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Habilitar microbenchmarks (default: false por performance)
    pub enabled: bool,

    /// Número de iteraciones por benchmark (default: 3)
    pub iterations: usize,

    /// Timeout máximo en ms para todos los benchmarks (default: 50ms)
    pub max_time_ms: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            iterations: 3,
            max_time_ms: 50,
        }
    }
}

impl BenchConfig {
    /// Crear configuración desde variables de entorno
    pub fn from_env() -> Self {
        let enabled = std::env::var("NUMRS_ENABLE_PROBING")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false);

        let iterations = std::env::var("NUMRS_BENCH_ITERATIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);

        let max_time_ms = std::env::var("NUMRS_BENCH_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);

        Self {
            enabled,
            iterations,
            max_time_ms,
        }
    }
}

/// Medir tiempo de ejecución de una función (promedio de N iteraciones)
#[inline]
fn bench_fn<F>(f: F, iterations: usize) -> f64
where
    F: Fn() -> (),
{
    let mut total = 0.0;

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        total += start.elapsed().as_secs_f64();
    }

    total / iterations as f64
}

/// Ejecutar microbenchmarks para matmul y construir lookup table
pub fn benchmark_matmul(
    validation: &BackendValidation,
    config: &BenchConfig,
) -> AdaptiveLookupTable<MatmulFn> {
    if !config.enabled {
        // Sin benchmarking, usar heurística estática
        #[cfg(debug_assertions)]
        eprintln!("[numrs-bench] Matmul: using static heuristic (probing disabled)");

        let mut ranges = Vec::new();

        // Rango pequeño: SIMD
        if validation.simd_validated {
            ranges.push(SizeRange {
                min_size: 0,
                max_size: Some(4_096),
                kernel: crate::backend::dispatch::kernel_matmul_simd as MatmulFn,
                backend_name: "cpu-simd",
            });
        }

        // Rango mediano: BLAS
        if validation.blas_validated {
            ranges.push(SizeRange {
                min_size: 4_096,
                max_size: Some(262_144),
                kernel: crate::backend::dispatch::kernel_matmul_blas_direct as MatmulFn,
                backend_name: "blas",
            });
        }

        // Rango grande: GPU (si disponible) o BLAS
        if validation.metal_validated {
            ranges.push(SizeRange {
                min_size: 262_144,
                max_size: None,
                kernel: crate::backend::dispatch::kernel_matmul_metal as MatmulFn,
                backend_name: "metal",
            });
        } else if validation.webgpu_validated {
            ranges.push(SizeRange {
                min_size: 262_144,
                max_size: None,
                kernel: crate::backend::dispatch::kernel_matmul_webgpu as MatmulFn,
                backend_name: "webgpu",
            });
        } else if validation.blas_validated {
            // Si no hay GPU, BLAS maneja todo desde 4096 en adelante
            ranges.push(SizeRange {
                min_size: 262_144,
                max_size: None,
                kernel: crate::backend::dispatch::kernel_matmul_blas_direct as MatmulFn,
                backend_name: "blas",
            });
        } else if validation.simd_validated {
            // Si no hay ni GPU ni BLAS, SIMD maneja rangos grandes
            ranges.push(SizeRange {
                min_size: 262_144,
                max_size: None,
                kernel: crate::backend::dispatch::kernel_matmul_simd as MatmulFn,
                backend_name: "cpu-simd",
            });
        }

        // Fallback final: scalar (solo si no se creó ningún rango)
        if ranges.is_empty() {
            ranges.push(SizeRange {
                min_size: 0,
                max_size: None,
                kernel: crate::backend::dispatch::kernel_matmul_scalar as MatmulFn,
                backend_name: "cpu-scalar",
            });
        }

        return AdaptiveLookupTable { ranges };
    }

    // Microbenchmarking REAL habilitado
    eprintln!("[numrs-bench] Running matmul microbenchmarks...");

    // Test con diferentes tamaños y medir cada backend
    let test_sizes = vec![32, 64, 128, 256, 512];
    let mut results: Vec<(usize, &'static str, MatmulFn, f64)> = Vec::new();

    let start_total = Instant::now();

    for &size in &test_sizes {
        if start_total.elapsed().as_millis() > config.max_time_ms as u128 {
            eprintln!("[numrs-bench] Timeout reached");
            break;
        }

        let output_size = size * size;
        let a = Array::new(vec![size, size], vec![1.0f32; size * size]);
        let b = Array::new(vec![size, size], vec![1.0f32; size * size]);

        let mut times = Vec::new();

        // Benchmark SIMD
        if validation.simd_validated {
            let time = bench_fn(
                || {
                    let _ = crate::backend::dispatch::kernel_matmul_simd(&a, &b);
                },
                config.iterations.min(2),
            );
            times.push((
                "cpu-simd",
                crate::backend::dispatch::kernel_matmul_simd as MatmulFn,
                time,
            ));
        }

        // Benchmark BLAS
        if validation.blas_validated {
            let time = bench_fn(
                || {
                    let _ = crate::backend::dispatch::kernel_matmul_blas_direct(&a, &b);
                },
                config.iterations.min(2),
            );
            times.push((
                "blas",
                crate::backend::dispatch::kernel_matmul_blas_direct as MatmulFn,
                time,
            ));
        }

        // Benchmark GPU (solo tamaños >= 128 para evitar overhead)
        if size >= 128 {
            if validation.metal_validated {
                if let Ok(_) = crate::backend::dispatch::kernel_matmul_metal(&a, &b) {
                    let time = bench_fn(
                        || {
                            let _ = crate::backend::dispatch::kernel_matmul_metal(&a, &b);
                        },
                        1,
                    );
                    times.push((
                        "metal",
                        crate::backend::dispatch::kernel_matmul_metal as MatmulFn,
                        time,
                    ));
                }
            }

            if validation.webgpu_validated {
                if let Ok(_) = crate::backend::dispatch::kernel_matmul_webgpu(&a, &b) {
                    let time = bench_fn(
                        || {
                            let _ = crate::backend::dispatch::kernel_matmul_webgpu(&a, &b);
                        },
                        1,
                    );
                    times.push((
                        "webgpu",
                        crate::backend::dispatch::kernel_matmul_webgpu as MatmulFn,
                        time,
                    ));
                }
            }
        }

        // Encontrar ganador para este tamaño
        if let Some((name, kernel, time)) =
            times.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        {
            results.push((output_size, *name, *kernel, *time));
            eprintln!(
                "[numrs-bench]   {}x{} (size={}): {} ({:.3}ms)",
                size,
                size,
                output_size,
                name,
                time * 1000.0
            );
        }
    }

    // Construir lookup table basada en resultados
    build_lookup_table_from_results(results, validation)
}

/// Construir lookup table desde resultados de benchmark
fn build_lookup_table_from_results(
    results: Vec<(usize, &'static str, MatmulFn, f64)>,
    _validation: &BackendValidation,
) -> AdaptiveLookupTable<MatmulFn> {
    let mut ranges = Vec::new();

    if results.is_empty() {
        // Fallback a heurística estática - crear config deshabilitado
        let disabled_config = BenchConfig {
            enabled: false,
            iterations: 3,
            max_time_ms: 50,
        };
        return benchmark_matmul(_validation, &disabled_config);
    }

    // Agrupar resultados consecutivos con el mismo backend
    let mut current_backend = results[0].1;
    let mut current_kernel = results[0].2;
    let mut range_start = 0;

    for (i, (size, backend, kernel, _)) in results.iter().enumerate() {
        if *backend != current_backend || i == results.len() - 1 {
            // Cambio de backend o último elemento
            let range_end = if i == results.len() - 1 {
                None
            } else {
                Some(*size)
            };

            ranges.push(SizeRange {
                min_size: range_start,
                max_size: range_end,
                kernel: current_kernel,
                backend_name: current_backend,
            });

            if i < results.len() - 1 {
                current_backend = *backend;
                current_kernel = *kernel;
                range_start = *size;
            }
        }
    }

    // Asegurar que hay un rango que cubre hasta infinito
    if let Some(last_range) = ranges.last_mut() {
        if last_range.max_size.is_some() {
            last_range.max_size = None;
        }
    }

    eprintln!(
        "[numrs-bench] Matmul lookup table created with {} ranges",
        ranges.len()
    );
    for range in &ranges {
        let max_str = range
            .max_size
            .map(|m| m.to_string())
            .unwrap_or_else(|| "∞".to_string());
        eprintln!(
            "[numrs-bench]   [{}, {}): {}",
            range.min_size, max_str, range.backend_name
        );
    }

    AdaptiveLookupTable { ranges }
}

/// Stub para elementwise (implementar después)
pub fn benchmark_elementwise(
    _validation: &BackendValidation,
    _config: &BenchConfig,
) -> AdaptiveLookupTable<ElementwiseFn> {
    use crate::backend::dispatch::kernel_elementwise_simd;

    #[cfg(debug_assertions)]
    eprintln!("[numrs-bench] Elementwise: using static fallback (not implemented yet)");

    AdaptiveLookupTable {
        ranges: vec![SizeRange {
            min_size: 0,
            max_size: None,
            kernel: kernel_elementwise_simd as ElementwiseFn,
            backend_name: "cpu-simd",
        }],
    }
}

/// Stub para reduction (implementar después)
pub fn benchmark_reduction(
    _validation: &BackendValidation,
    _config: &BenchConfig,
) -> AdaptiveLookupTable<ReductionFn> {
    use crate::backend::dispatch::kernel_reduction_simd;

    #[cfg(debug_assertions)]
    eprintln!("[numrs-bench] Reduction: using static fallback (not implemented yet)");

    AdaptiveLookupTable {
        ranges: vec![SizeRange {
            min_size: 0,
            max_size: None,
            kernel: kernel_reduction_simd as ReductionFn,
            backend_name: "cpu-simd",
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn test_lookup_table() {
        // Crear tabla de ejemplo
        fn dummy_kernel(_a: &Array, _b: &Array) -> Result<Array> {
            Ok(Array::new(vec![1], vec![0.0]))
        }

        let table = AdaptiveLookupTable {
            ranges: vec![
                SizeRange {
                    min_size: 0,
                    max_size: Some(100),
                    kernel: dummy_kernel as MatmulFn,
                    backend_name: "small",
                },
                SizeRange {
                    min_size: 100,
                    max_size: Some(1000),
                    kernel: dummy_kernel as MatmulFn,
                    backend_name: "medium",
                },
                SizeRange {
                    min_size: 1000,
                    max_size: None,
                    kernel: dummy_kernel as MatmulFn,
                    backend_name: "large",
                },
            ],
        };

        assert_eq!(table.backend_name(50), "small");
        assert_eq!(table.backend_name(500), "medium");
        assert_eq!(table.backend_name(5000), "large");
    }
}
