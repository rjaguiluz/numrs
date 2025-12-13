//! Multi-threaded matmul usando Rayon
//!
//! Strategy: Dividir la matriz en bloques y procesar en paralelo con Rayon.
//! IMPORTANTE: Esta función NO decide el backend, solo paraleliza.
//! El kernel específico se pasa como parámetro.

use crate::array::Array;
use anyhow::{bail, Result};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Matmul usando un kernel específico (BLAS, SIMD, o Scalar)
/// Esta función NO decide qué backend usar - solo ejecuta el kernel dado
// Native implementation using Rayon
#[cfg(not(target_arch = "wasm32"))]
pub fn matmul_with_kernel(
    a: &Array,
    b: &Array,
    kernel: fn(&Array, &Array) -> Array,
) -> Result<Array> {
    // Validación
    if a.shape.len() != 2 || b.shape.len() != 2 {
        bail!("matmul requires 2-D arrays");
    }

    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    if k != b.shape[0] {
        bail!("inner dimension mismatch: {} != {}", k, b.shape[0]);
    }

    // Para matrices pequeñas, ejecutar directamente sin overhead de threading
    if m < 500 {
        return Ok(kernel(a, b));
    }

    // Determinar tamaño de bloques adaptativo
    let num_threads = rayon::current_num_threads();

    // Para matrices muy grandes (>2048), limitar el tamaño de bloque para evitar overhead de memoria
    // Para matrices medianas, usar bloques más grandes para mejor aprovechamiento de cache
    let block_size = if m >= 2048 {
        // Matrices muy grandes: bloques más pequeños, más bloques
        256.max((m + num_threads * 4 - 1) / (num_threads * 4))
    } else {
        // Matrices medianas: bloques balanceados
        64.max((m + num_threads - 1) / num_threads)
    };

    // Pre-alocar el resultado completo para evitar concatenaciones costosas
    let mut result = vec![0.0f32; m * n];

    // Procesar bloques en paralelo escribiendo directamente al resultado
    result
        .par_chunks_mut(block_size * n)
        .enumerate()
        .for_each(|(block_idx, out_block)| {
            let start = block_idx * block_size;
            let end = (start + block_size).min(m);
            let block_rows = end - start;

            // Extraer sub-matriz de A (sin copiar toda la fila, solo lo necesario)
            let a_block_data: Vec<f32> = (start..end)
                .flat_map(|i| &a.data[i * k..(i + 1) * k])
                .copied()
                .collect();

            let a_block = Array::new(vec![block_rows, k], a_block_data);

            // Ejecutar matmul del bloque usando el kernel dado
            let result_block = kernel(&a_block, b);

            // Copiar resultado directamente al slice de salida
            out_block[..result_block.data.len()].copy_from_slice(&result_block.data);
        });

    Ok(Array::new(vec![m, n], result))
}

// WASM implementation (Serial) - Rayon panics on WASM so we force serial execution
#[cfg(target_arch = "wasm32")]
pub fn matmul_with_kernel(
    a: &Array,
    b: &Array,
    kernel: fn(&Array, &Array) -> Array,
) -> Result<Array> {
    // En WASM simplemente llamamos al kernel directamente (ya sea BLAS simulado o SIMD)
    // No vale la pena el overhead de blocking/splitting si es single-thread
    Ok(kernel(a, b))
}

/// Función legacy para compatibilidad - usa BLAS si está disponible
#[cfg(numrs_has_blas)]
pub fn matmul_parallel(a: &Array, b: &Array) -> Result<Array> {
    matmul_with_kernel(a, b, crate::backend::blas::matmul_blas)
}

/// Función legacy para compatibilidad - usa SIMD si BLAS no está disponible
#[cfg(not(numrs_has_blas))]
pub fn matmul_parallel(a: &Array, b: &Array) -> Result<Array> {
    matmul_with_kernel(a, b, super::matmul_simd_direct)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_matmul() {
        let a = Array::new(vec![1000, 1000], vec![1.0; 1000 * 1000]);
        let b = Array::new(vec![1000, 1000], vec![1.0; 1000 * 1000]);

        let result = matmul_parallel(&a, &b).unwrap();

        assert_eq!(result.shape, vec![1000, 1000]);
        // Cada elemento debería ser suma de 1000 unos = 1000.0
        assert!((result.data[0] - 1000.0).abs() < 0.1);
    }
}
