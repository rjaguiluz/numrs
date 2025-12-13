pub mod batchnorm;
pub mod conv;
pub mod dropout;
pub mod parallel;
pub mod random;
pub mod scalar;
pub mod simd;
pub mod simd_conv;

// Pub use elementwise removed (doesn't exist)
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::array::Array;

// Local wrappers used by method selection. Each wrapper accepts a tuple
// `(&Array, &Array)` and returns an `Array`, matching the `fn(&T)->R` signature
// used by KernelSelectionContext.
#[cfg(not(target_arch = "wasm32"))]
fn matmul_scalar_impl(inputs: &(&Array, &Array)) -> Array {
    let a_arr = inputs.0;
    let b_arr = inputs.1;
    let m = a_arr.shape[0];
    let k1 = a_arr.shape[1];
    let n = b_arr.shape[1];

    // Optimized block sizes for better cache utilization
    // BM/BN sized for L2 cache (~256KB), BK for L1 cache
    let bm = 128usize; // Larger blocks for better parallelization
    let bn = 128usize;
    let bk = 256usize; // Larger K blocks to amortize memory access

    let mut out = vec![0.0f32; m * n];

    // Partition output into row-blocks and parallelize
    let rows_per_block = bm;
    out.par_chunks_mut(n * rows_per_block)
        .enumerate()
        .for_each(|(block_idx, out_block)| {
            let i0 = block_idx * rows_per_block;
            let i_max = (i0 + rows_per_block).min(m);
            let rows_in_block = i_max - i0;

            // Reorder loops: k -> i -> j for better cache locality
            for k0 in (0..k1).step_by(bk) {
                let k_max = (k0 + bk).min(k1);

                for j0 in (0..n).step_by(bn) {
                    let j_max = (j0 + bn).min(n);

                    for ii in 0..rows_in_block {
                        let i = i0 + ii;
                        let a_row_off = i * k1;
                        let out_row_off = ii * n;

                        // Process 4 columns at a time (manual loop unrolling)
                        let mut j = j0;
                        while j + 4 <= j_max {
                            let mut sum0 = out_block[out_row_off + j];
                            let mut sum1 = out_block[out_row_off + j + 1];
                            let mut sum2 = out_block[out_row_off + j + 2];
                            let mut sum3 = out_block[out_row_off + j + 3];

                            for kk in k0..k_max {
                                let a_val = a_arr.data[a_row_off + kk];
                                let b_row_off = kk * n;
                                sum0 += a_val * b_arr.data[b_row_off + j];
                                sum1 += a_val * b_arr.data[b_row_off + j + 1];
                                sum2 += a_val * b_arr.data[b_row_off + j + 2];
                                sum3 += a_val * b_arr.data[b_row_off + j + 3];
                            }

                            out_block[out_row_off + j] = sum0;
                            out_block[out_row_off + j + 1] = sum1;
                            out_block[out_row_off + j + 2] = sum2;
                            out_block[out_row_off + j + 3] = sum3;
                            j += 4;
                        }

                        // Handle remaining columns
                        while j < j_max {
                            let mut sum = out_block[out_row_off + j];
                            for kk in k0..k_max {
                                sum += a_arr.data[a_row_off + kk] * b_arr.data[kk * n + j];
                            }
                            out_block[out_row_off + j] = sum;
                            j += 1;
                        }
                    }
                }
            }
        });

    crate::array::Array::new(vec![m, n], out)
}

#[cfg(target_arch = "wasm32")]
fn matmul_scalar_impl(inputs: &(&Array, &Array)) -> Array {
    let a_arr = inputs.0;
    let b_arr = inputs.1;
    let m = a_arr.shape[0];
    let k1 = a_arr.shape[1];
    let n = b_arr.shape[1];

    let bm = 128usize;
    let bn = 128usize;
    let bk = 256usize;

    let mut out = vec![0.0f32; m * n];

    // Serial execution for WASM
    let rows_per_block = bm;

    // Using standard iter_mut().enumerate() for serial processing
    out.chunks_mut(n * rows_per_block)
        .enumerate()
        .for_each(|(block_idx, out_block)| {
            let i0 = block_idx * rows_per_block;
            let i_max = (i0 + rows_per_block).min(m);
            let rows_in_block = i_max - i0;

            for k0 in (0..k1).step_by(bk) {
                let k_max = (k0 + bk).min(k1);

                for j0 in (0..n).step_by(bn) {
                    let j_max = (j0 + bn).min(n);

                    for ii in 0..rows_in_block {
                        let i = i0 + ii;
                        let a_row_off = i * k1;
                        let out_row_off = ii * n;

                        let mut j = j0;
                        while j + 4 <= j_max {
                            let mut sum0 = out_block[out_row_off + j];
                            let mut sum1 = out_block[out_row_off + j + 1];
                            let mut sum2 = out_block[out_row_off + j + 2];
                            let mut sum3 = out_block[out_row_off + j + 3];

                            for kk in k0..k_max {
                                let a_val = a_arr.data[a_row_off + kk];
                                let b_row_off = kk * n;
                                sum0 += a_val * b_arr.data[b_row_off + j];
                                sum1 += a_val * b_arr.data[b_row_off + j + 1];
                                sum2 += a_val * b_arr.data[b_row_off + j + 2];
                                sum3 += a_val * b_arr.data[b_row_off + j + 3];
                            }

                            out_block[out_row_off + j] = sum0;
                            out_block[out_row_off + j + 1] = sum1;
                            out_block[out_row_off + j + 2] = sum2;
                            out_block[out_row_off + j + 3] = sum3;
                            j += 4;
                        }

                        while j < j_max {
                            let mut sum = out_block[out_row_off + j];
                            for kk in k0..k_max {
                                sum += a_arr.data[a_row_off + kk] * b_arr.data[kk * n + j];
                            }
                            out_block[out_row_off + j] = sum;
                            j += 1;
                        }
                    }
                }
            }
        });

    crate::array::Array::new(vec![m, n], out)
}

// ============================================================================
// Public kernel wrappers for dispatch system
// ============================================================================

/// Matmul scalar con paralelización Rayon (para benchmarking)
/// Usa bloques optimizados pero sin instrucciones SIMD
pub fn matmul_scalar_parallel(a: &Array, b: &Array) -> Array {
    eprintln!(
        "[SCALAR_IMPL] matmul_scalar_parallel called for {}x{}",
        a.shape[0], a.shape[1]
    );
    matmul_scalar_impl(&(a, b))
}

/// SIMD-accelerated matmul (uses AVX2+FMA when available, falls back to scalar)
pub fn matmul_simd_direct(a: &Array, b: &Array) -> Array {
    simd::matmul_simd(a, b)
}

/// Scalar fallback matmul (always available)
pub fn matmul_scalar_direct(a: &Array, b: &Array) -> Array {
    matmul_scalar_impl(&(a, b))
}

/// CPU backend orchestrates scalar and SIMD strategies.
#[derive(Debug, Clone)]
pub struct CpuBackend {
    // future: CPU threads, affinity, simd levels
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {}
    }

    /// Expose a fallback matmul entry point for microbench/testing that
    /// directly invokes the scalar/parallel implementation without using BLAS.
    pub fn matmul_fallback(
        a: &crate::array::Array,
        b: &crate::array::Array,
    ) -> crate::array::Array {
        matmul_scalar_impl(&(a, b))
    }

    // execute() method removed - use ops::fast::* functions with dispatch system instead
}
