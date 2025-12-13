// ============================================================================
// BLAS Backend - Static Linking Implementation
// ============================================================================
//
// Este backend usa STATIC LINKING de BLAS para distribución autocontenida.
// El binario compilado incluye la implementación BLAS completa embebida.
//
// Estrategia por plataforma:
// - macOS: Accelerate framework (sistema, siempre disponible)
// - x86_64: Intel MKL via intel-mkl-src (feature "mkl")
// - ARM/otros: BLIS via blis-src (feature "blis")
// - Fallback: GEMM implementado en Rust puro
//
// Ver STATIC_BLAS_LINKING.md para guía completa de uso.
// ============================================================================

// --- STATIC BLAS LINKING ---
// Los siguientes imports garantizan que las implementaciones BLAS se linkeen
// estáticamente en el binario final.

/// macOS: Accelerate framework (parte del sistema operativo)
#[cfg(numrs_has_accelerate)]
use accelerate_src as _;

/// x86_64: Intel MKL estatico (via intel-mkl-src crate)
/// Linkeo automático basado en target_arch = "x86_64"
/// Tamaño: ~80-120 MB embebido en el binario
#[cfg(numrs_has_mkl)]
extern crate intel_mkl_src as _;

/// ARM/otras arquitecturas: BLIS estático (via blis-src crate)
/// Linkeo automático basado en target_arch != "x86_64"
/// Tamaño: ~15-30 MB embebido en el binario
#[cfg(numrs_has_blis)]
extern crate blis_src as _;

#[derive(Debug, Clone)]
pub struct BlasBackend {}

impl BlasBackend {
    pub fn new() -> Self { Self {} }
    
    // execute() method removed - use ops::fast::* functions with dispatch system instead
}

/// Dot product using BLAS sdot (single precision dot product)
#[cfg(numrs_has_blas)]
pub fn dot_blas(a: &crate::array::Array, b: &crate::array::Array) -> anyhow::Result<f32> {
    use anyhow::bail;
    
    if a.shape.len() != 1 || b.shape.len() != 1 {
        bail!("dot_blas: both inputs must be 1-D arrays");
    }
    if a.shape[0] != b.shape[0] {
        bail!("dot_blas: arrays must have same length");
    }

    let n = a.shape[0] as i32;
    
    // SAFETY: cblas::sdot is safe when:
    // - n is correct length
    // - slices are valid and have correct length
    // - incx and incy are 1 (contiguous)
    unsafe {
        let result = cblas::sdot(
            n,
            &a.data,  // Takes &[f32] slice
            1,        // incx: stride for a
            &b.data,  // Takes &[f32] slice
            1,        // incy: stride for b
        );
        Ok(result)
    }
}

/// Fallback when BLAS not available
#[cfg(not(numrs_has_blas))]
pub fn dot_blas(a: &crate::array::Array, b: &crate::array::Array) -> anyhow::Result<f32> {
    crate::backend::cpu::scalar::dot_scalar(a, b)
}

/// Fast inline matmul for tiny matrices (avoids MKL overhead)
/// Uses AVX2 SIMD when available for better performance
/// 
/// # Safety
/// This function is specialized for f32. The caller must ensure that
/// the input slices contain f32 data. This is enforced by matmul_blas
/// which only works with Array<f32>.
#[inline]
fn matmul_tiny_inline(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    // For very small matrices, use simple triple-loop
    // SIMD overhead dominates for matrices < 16 elements
    if m * n < 16 {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        return;
    }
    
    #[cfg(target_arch = "x86_64")]
    {
        // Use AVX2 for medium-tiny matrices (16-128 elements)
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { matmul_tiny_avx2(a, b, out, m, k, n) };
            return;
        }
    }
    
    // Fallback: scalar with loop unrolling
    matmul_tiny_scalar(a, b, out, m, k, n);
}

/// Scalar tiny matmul with aggressive loop unrolling
#[inline]
fn matmul_tiny_scalar(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            
            // Unroll by 4 for better ILP
            let mut kk = 0;
            while kk + 4 <= k {
                sum += a[i * k + kk] * b[kk * n + j];
                sum += a[i * k + kk + 1] * b[(kk + 1) * n + j];
                sum += a[i * k + kk + 2] * b[(kk + 2) * n + j];
                sum += a[i * k + kk + 3] * b[(kk + 3) * n + j];
                kk += 4;
            }
            
            // Handle remainder
            while kk < k {
                sum += a[i * k + kk] * b[kk * n + j];
                kk += 1;
            }
            
            out[i * n + j] = sum;
        }
    }
}

/// AVX2 optimized tiny matmul
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_tiny_avx2(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    
    // Process output in blocks
    for i in 0..m {
        let mut j = 0;
        
        // Process 8 columns at a time with AVX2
        while j + 8 <= n {
            let mut sum = _mm256_setzero_ps();
            
            for kk in 0..k {
                let a_val = _mm256_set1_ps(a[i * k + kk]);
                let b_ptr = b.as_ptr().add(kk * n + j);
                let b_vals = _mm256_loadu_ps(b_ptr);
                sum = _mm256_fmadd_ps(a_val, b_vals, sum);
            }
            
            let out_ptr = out.as_mut_ptr().add(i * n + j);
            _mm256_storeu_ps(out_ptr, sum);
            j += 8;
        }
        
        // Process 4 columns at a time with SSE
        if j + 4 <= n {
            let mut sum = _mm_setzero_ps();
            
            for kk in 0..k {
                let a_val = _mm_set1_ps(a[i * k + kk]);
                let b_ptr = b.as_ptr().add(kk * n + j);
                let b_vals = _mm_loadu_ps(b_ptr);
                sum = _mm_fmadd_ps(a_val, b_vals, sum);
            }
            
            let out_ptr = out.as_mut_ptr().add(i * n + j);
            _mm_storeu_ps(out_ptr, sum);
            j += 4;
        }
        
        // Handle remaining columns scalar
        while j < n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = sum;
            j += 1;
        }
    }
}

/// Matrix multiplication using BLAS (SGEMM for single precision)
/// Optimized adaptive strategy:
/// - Tiny matrices (output < 128): Inline optimized kernel (avoids MKL overhead)
/// - Small matrices (< 512): Direct MKL single-threaded
/// - Large matrices (>= 512): Rayon + MKL for best performance
/// 
/// # Important
/// This function only supports f32 (single precision). The input Array must
/// contain f32 data. This is enforced by the ops::matmul wrapper which uses
/// binary_promoted_with to convert inputs to f32 before calling this function.
/// 
/// For f64 support, DGEMM would need to be used instead of SGEMM.
#[cfg(numrs_has_blas)]
pub fn matmul_blas(a: &crate::array::Array, b: &crate::array::Array) -> crate::array::Array {
    use cblas::{Layout, Transpose};
    
    // Assert that we're working with f32 data
    debug_assert_eq!(a.dtype, crate::array::DType::F32, 
        "matmul_blas only supports f32, got {:?}", a.dtype);
    debug_assert_eq!(b.dtype, crate::array::DType::F32, 
        "matmul_blas only supports f32, got {:?}", b.dtype);
    
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    
    // Tiny matrices (output < 128 elements): Use inline optimized kernel
    // MKL overhead dominates for very small matrices
    let output_size = m * n;
    if output_size < 128 {
        let mut out = vec![0.0f32; output_size];
        matmul_tiny_inline(&a.data, &b.data, &mut out, m, k, n);
        return crate::array::Array::new(vec![m, n], out);
    }
    
    // Small matrices (<512): Use direct MKL for best performance
    // Rayon overhead dominates for small matrices
    if m < 512 {
        let mut out = vec![0.0f32; m * n];
        
        unsafe {
            cblas::sgemm(
                Layout::RowMajor,
                Transpose::None,
                Transpose::None,
                m as i32,
                n as i32,
                k as i32,
                1.0,          // alpha
                &a.data,
                k as i32,     // lda
                &b.data,
                n as i32,     // ldb
                0.0,          // beta
                &mut out,
                n as i32,     // ldc
            );
        }
        
        return crate::array::Array::new(vec![m, n], out);
    }
    
    // For matrices >= 512, use Rayon parallelization with MKL
    // This consistently outperforms MKL's internal threading
    if m >= 512 {
        use rayon::prelude::*;
                
        // Optimized block sizes for different matrix ranges
        // Larger matrices need bigger blocks to reduce Rayon overhead
        // while maintaining good load balance across cores
        let block_size = if m >= 3072 {
            1024  // Very large: 1024-row blocks
        } else if m >= 2048 {
            896   // Large: 896-row blocks for 2048 (2048/896 ≈ 2.3 blocks)
        } else if m >= 1536 {
            768   // Medium-large: 768-row blocks
        } else if m >= 1024 {
            512   // Medium: 512-row blocks
        } else {
            256   // Small-medium: 256-row blocks
        };
        
        let mut out = vec![0.0f32; m * n];
        
        // Process blocks in parallel WITHOUT copying A - use direct slices
        out.par_chunks_mut(block_size * n)
            .enumerate()
            .for_each(|(block_idx, out_block)| {
                let start_row = block_idx * block_size;
                let end_row = (start_row + block_size).min(m);
                let block_rows = end_row - start_row;
                
                let (m_i32, n_i32, k_i32) = (block_rows as i32, n as i32, k as i32);
                let (lda, ldb, ldc) = (k as i32, n as i32, n as i32);
                
                // Direct slice of A without copying - zero-copy optimization
                let a_block_start = start_row * k;
                let a_block_end = end_row * k;
                let a_block_slice = &a.data[a_block_start..a_block_end];
                
                unsafe {
                    cblas::sgemm(
                        Layout::RowMajor,
                        Transpose::None,
                        Transpose::None,
                        m_i32, n_i32, k_i32,
                        1.0f32,
                        a_block_slice, lda,
                        &b.data, ldb,
                        0.0f32,
                        out_block, ldc,
                    );
                }
            });
        
        return crate::array::Array::new(vec![m, n], out);
    }
    
    // For small matrices (< 512), use MKL single-threaded
    // Avoids Rayon overhead for small problem sizes
    let mut out = vec![0.0f32; m * n];
    let (m_i32, n_i32, k_i32) = (m as i32, n as i32, k as i32);
    let (lda, ldb, ldc) = (k as i32, n as i32, n as i32);
    
    unsafe {
        cblas::sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m_i32, n_i32, k_i32,
            1.0f32,
            &a.data, lda,
            &b.data, ldb,
            0.0f32,
            &mut out, ldc,
        );
    }
    
    crate::array::Array::new(vec![m, n], out)
}

/// BLAS con paralelización Rayon para matrices grandes
#[cfg(numrs_has_blas)]
pub fn matmul_blas_parallel(a: &crate::array::Array, b: &crate::array::Array) -> crate::array::Array {
    use cblas::{Layout, Transpose};
    use rayon::prelude::*;
    
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    
    // Dividir en bloques de ~256 filas para balance entre overhead y paralelización
    let block_size = 256;
    let mut out = vec![0.0f32; m * n];
    
    out.par_chunks_mut(block_size * n)
        .enumerate()
        .for_each(|(block_idx, out_block)| {
            let start = block_idx * block_size;
            let end = (start + block_size).min(m);
            let block_rows = end - start;
            
            // Extraer bloque de A
            let a_block_data: Vec<f32> = (start..end)
                .flat_map(|i| &a.data[i*k..(i+1)*k])
                .copied()
                .collect();
            
            let (m_i32, n_i32, k_i32) = (block_rows as i32, n as i32, k as i32);
            let (lda, ldb, ldc) = (k as i32, n as i32, n as i32);
            
            unsafe {
                cblas::sgemm(
                    Layout::RowMajor,
                    Transpose::None,
                    Transpose::None,
                    m_i32, n_i32, k_i32,
                    1.0f32,
                    &a_block_data, lda,
                    &b.data, ldb,
                    0.0f32,
                    out_block, ldc,
                );
            }
        });
    
    crate::array::Array::new(vec![m, n], out)
}

/// Fallback when BLAS not available - delegates to SIMD implementation
#[cfg(not(numrs_has_blas))]
pub fn matmul_blas(a: &crate::array::Array, b: &crate::array::Array) -> crate::array::Array {
    crate::backend::cpu::matmul_simd_direct(a, b)
}


