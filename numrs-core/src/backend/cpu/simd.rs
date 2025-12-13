/// Placeholder for SIMD CPU execution strategy. In a full implementation this
/// would contain vectorized loops and checks for AVX/NEON availability.

pub fn elementwise_simd_supported() -> bool {
    // Try to detect SIMD availability at runtime on x86/x86_64 targets.
    // We conservatively require at least SSE2 or AVX2 to consider SIMD available.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // is_x86_feature_detected! is a macro that expands to a runtime helper
        // to check for available CPU features on the host.
        if std::is_x86_feature_detected!("avx2") {
            return true;
        }
        if std::is_x86_feature_detected!("sse2") {
            return true;
        }
        false
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        // On other architectures we conservatively return false for now.
        false
    }
}

use crate::array::Array;
use crate::llo::ElementwiseKind;
use crate::llo::reduction::ReductionKind;
use anyhow::{Result, anyhow};

// We use the portable `core::simd::Simd` type so this code works in a
// cross-platform manner — if a host doesn't provide hardware SIMD, the
// compiler/runtime will still produce correct scalar fallbacks.
// We intentionally avoid the unstable `portable_simd` API to keep the
// prototype building on stable Rust. Instead we use arch intrinsics for
// x86/x86_64 (AVX2/SSE) with a scalar fallback.

/// Prototype SIMD path that currently delegates to scalar implementation.
pub fn elementwise_simd(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array> {
    // Prototype SIMD implementation mirroring the scalar logic but operating
    // on small fixed-size vector chunks. This keeps correctness identical to
    // the scalar version while allowing the compiler to generate vectorized
    // code paths on supported targets.

    if a.shape != b.shape { return Err(anyhow!("shape mismatch in simd elementwise")); }

    let mut out = Array::<f32>::zeros(a.shape.clone());
    let n = a.len();

    let i = 0usize;

    // x86/x86_64 specialised fast paths (AVX2 -> 8 floats, SSE -> 4 floats)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if std::is_x86_feature_detected!("avx2") {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

                while i + 8 <= n {
                let pa = _mm256_loadu_ps(a.data.as_ptr().add(i));
                    let pb = _mm256_loadu_ps(b.data.as_ptr().add(i));
                    let pr = match kind {
                        ElementwiseKind::Add => _mm256_add_ps(pa, pb),
                        ElementwiseKind::Mul => _mm256_mul_ps(pa, pb),
                        ElementwiseKind::Sub => _mm256_sub_ps(pa, pb),
                        ElementwiseKind::Div => _mm256_div_ps(pa, pb),
                        ElementwiseKind::Sqrt => _mm256_sqrt_ps(pa),
                        // fall back to scalar for certain ops inside vectorized loop
                        ElementwiseKind::Sin | ElementwiseKind::Cos | ElementwiseKind::Tan | ElementwiseKind::Abs | ElementwiseKind::Neg | ElementwiseKind::Exp | ElementwiseKind::Log | ElementwiseKind::Pow | ElementwiseKind::Asin | ElementwiseKind::Acos | ElementwiseKind::Atan | ElementwiseKind::Relu | ElementwiseKind::LeakyRelu | ElementwiseKind::Sigmoid | ElementwiseKind::Tanh | ElementwiseKind::Softplus => {
                            // convert to scalars for this chunk; load both a and b lanes
                            let mut tmp_a = [0.0f32; 8];
                            let mut tmp_b = [0.0f32; 8];
                            _mm256_storeu_ps(tmp_a.as_mut_ptr(), pa);
                            _mm256_storeu_ps(tmp_b.as_mut_ptr(), pb);
                            for j in 0..8 {
                                tmp_a[j] = match kind {
                                    ElementwiseKind::Sin => tmp_a[j].sin(),
                                    ElementwiseKind::Cos => tmp_a[j].cos(),
                                    ElementwiseKind::Tan => tmp_a[j].tan(),
                                    ElementwiseKind::Abs => tmp_a[j].abs(),
                                    ElementwiseKind::Neg => -tmp_a[j],
                                    ElementwiseKind::Exp => tmp_a[j].exp(),
                                    ElementwiseKind::Log => tmp_a[j].ln(),
                                    ElementwiseKind::Pow => tmp_a[j].powf(tmp_b[j]),
                                    ElementwiseKind::Asin => tmp_a[j].asin(),
                                    ElementwiseKind::Acos => tmp_a[j].acos(),
                                    ElementwiseKind::Atan => tmp_a[j].atan(),
                                    ElementwiseKind::Relu => tmp_a[j].max(0.0),
                                    ElementwiseKind::LeakyRelu => if tmp_a[j] > 0.0 { tmp_a[j] } else { 0.01 * tmp_a[j] },
                                    ElementwiseKind::Sigmoid => 1.0 / (1.0 + (-tmp_a[j]).exp()),
                                    ElementwiseKind::Tanh => tmp_a[j].tanh(),
                                    ElementwiseKind::Softplus => (1.0 + tmp_a[j].exp()).ln(),
                                    _ => tmp_a[j]
                                };
                            }
                            let pr = _mm256_loadu_ps(tmp_a.as_ptr());
                            pr
                        }
                    };
                _mm256_storeu_ps(out.data.as_mut_ptr().add(i), pr);
                i += 8;
            }
        } else if std::is_x86_feature_detected!("sse2") {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

                while i + 4 <= n {
                let pa = _mm_loadu_ps(a.data.as_ptr().add(i));
                    let pb = _mm_loadu_ps(b.data.as_ptr().add(i));
                    let pr = match kind {
                        ElementwiseKind::Add => _mm_add_ps(pa, pb),
                        ElementwiseKind::Mul => _mm_mul_ps(pa, pb),
                        ElementwiseKind::Sub => _mm_sub_ps(pa, pb),
                        ElementwiseKind::Div => _mm_div_ps(pa, pb),
                        ElementwiseKind::Sqrt => _mm_sqrt_ps(pa),
                        ElementwiseKind::Sin | ElementwiseKind::Cos | ElementwiseKind::Tan | ElementwiseKind::Abs | ElementwiseKind::Neg | ElementwiseKind::Exp | ElementwiseKind::Log | ElementwiseKind::Pow | ElementwiseKind::Asin | ElementwiseKind::Acos | ElementwiseKind::Atan | ElementwiseKind::Relu | ElementwiseKind::LeakyRelu | ElementwiseKind::Sigmoid | ElementwiseKind::Tanh | ElementwiseKind::Softplus => {
                            let mut tmp_a = [0.0f32; 4];
                            let mut tmp_b = [0.0f32; 4];
                            _mm_storeu_ps(tmp_a.as_mut_ptr(), pa);
                            _mm_storeu_ps(tmp_b.as_mut_ptr(), pb);
                            for j in 0..4 {
                                tmp_a[j] = match kind {
                                    ElementwiseKind::Sin => tmp_a[j].sin(),
                                    ElementwiseKind::Cos => tmp_a[j].cos(),
                                    ElementwiseKind::Tan => tmp_a[j].tan(),
                                    ElementwiseKind::Abs => tmp_a[j].abs(),
                                    ElementwiseKind::Neg => -tmp_a[j],
                                    ElementwiseKind::Exp => tmp_a[j].exp(),
                                    ElementwiseKind::Log => tmp_a[j].ln(),
                                    ElementwiseKind::Pow => tmp_a[j].powf(tmp_b[j]),
                                    ElementwiseKind::Asin => tmp_a[j].asin(),
                                    ElementwiseKind::Acos => tmp_a[j].acos(),
                                    ElementwiseKind::Atan => tmp_a[j].atan(),
                                    ElementwiseKind::Relu => tmp_a[j].max(0.0),
                                    ElementwiseKind::LeakyRelu => if tmp_a[j] > 0.0 { tmp_a[j] } else { 0.01 * tmp_a[j] },
                                    ElementwiseKind::Sigmoid => 1.0 / (1.0 + (-tmp_a[j]).exp()),
                                    ElementwiseKind::Tanh => tmp_a[j].tanh(),
                                    ElementwiseKind::Softplus => (1.0 + tmp_a[j].exp()).ln(),
                                    _ => tmp_a[j]
                                };
                            }
                            let pr = _mm_loadu_ps(tmp_a.as_ptr());
                            pr
                        }
                    };
                _mm_storeu_ps(out.data.as_mut_ptr().add(i), pr);
                i += 4;
            }
        }
    }

    // Remaining / fallback scalar
    for j in i..n {
        out.data[j] = match kind {
            ElementwiseKind::Add => a.data[j] + b.data[j],
            ElementwiseKind::Mul => a.data[j] * b.data[j],
            ElementwiseKind::Sub => a.data[j] - b.data[j],
            ElementwiseKind::Div => a.data[j] / b.data[j],
            ElementwiseKind::Sqrt => a.data[j].sqrt(),
            ElementwiseKind::Abs => a.data[j].abs(),
            ElementwiseKind::Neg => -a.data[j],
            ElementwiseKind::Exp => a.data[j].exp(),
            ElementwiseKind::Log => a.data[j].ln(),
            ElementwiseKind::Tan => a.data[j].tan(),
            ElementwiseKind::Pow => a.data[j].powf(b.data[j]),
            ElementwiseKind::Sin => a.data[j].sin(),
            ElementwiseKind::Cos => a.data[j].cos(),
            ElementwiseKind::Asin => a.data[j].asin(),
            ElementwiseKind::Acos => a.data[j].acos(),
            ElementwiseKind::Atan => a.data[j].atan(),
            ElementwiseKind::Relu => a.data[j].max(0.0),
            ElementwiseKind::LeakyRelu => if a.data[j] > 0.0 { a.data[j] } else { 0.01 * a.data[j] },
            ElementwiseKind::Sigmoid => 1.0 / (1.0 + (-a.data[j]).exp()),
            ElementwiseKind::Tanh => a.data[j].tanh(),
            ElementwiseKind::Softplus => (1.0 + a.data[j].exp()).ln(),
        };
    }

    Ok(out)
}

/// SIMD-accelerated reduction (sum, max, min, mean). For the full-sum case (axis None) we
/// implement an AVX2 vectorized loop that accumulates into an __m256
/// register and then horizontally reduces it. For other architectures / when
/// AVX2 absent we fall back to scalar.
pub fn reduce_simd(a: &Array, axis: Option<usize>, kind: ReductionKind) -> Result<Array> {
    if axis.is_none() {
        let n = a.len();

        match kind {
            ReductionKind::Sum => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                unsafe {
                    if std::is_x86_feature_detected!("avx2") {
                        #[cfg(target_arch = "x86_64")]
                        use std::arch::x86_64::*;
                        #[cfg(target_arch = "x86")]
                        use std::arch::x86::*;

                        let mut i = 0usize;
                        let mut acc = _mm256_setzero_ps();

                        while i + 8 <= n {
                            let p = _mm256_loadu_ps(a.data.as_ptr().add(i));
                            acc = _mm256_add_ps(acc, p);
                            i += 8;
                        }

                        // Horizontal sum of acc
                        let mut s = [0.0f32; 8];
                        _mm256_storeu_ps(s.as_mut_ptr(), acc);
                        let mut sum = s.iter().copied().sum::<f32>();

                        // tail
                        while i < n {
                            sum += a.data[i];
                            i += 1;
                        }

                        return Ok(Array::new(vec![1], vec![sum]));
                    }
                }

                // Non x86/AVX2 path or if not detected: fallback to scalar
                let sum: f32 = a.data.iter().copied().sum();
                Ok(Array::new(vec![1], vec![sum]))
            }
            ReductionKind::Max => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                unsafe {
                    if std::is_x86_feature_detected!("avx2") {
                        #[cfg(target_arch = "x86_64")]
                        use std::arch::x86_64::*;
                        #[cfg(target_arch = "x86")]
                        use std::arch::x86::*;

                        let mut i = 0usize;
                        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);

                        while i + 8 <= n {
                            let p = _mm256_loadu_ps(a.data.as_ptr().add(i));
                            acc = _mm256_max_ps(acc, p);
                            i += 8;
                        }

                        // Horizontal max of acc
                        let mut s = [0.0f32; 8];
                        _mm256_storeu_ps(s.as_mut_ptr(), acc);
                        let mut max_val = s[0];
                        for &v in &s[1..] {
                            if v > max_val {
                                max_val = v;
                            }
                        }

                        // tail
                        while i < n {
                            if a.data[i] > max_val {
                                max_val = a.data[i];
                            }
                            i += 1;
                        }

                        return Ok(Array::new(vec![1], vec![max_val]));
                    }
                }

                // Fallback to scalar
                let max_val = a.data.iter().copied().fold(f32::NEG_INFINITY, |acc, x| acc.max(x));
                Ok(Array::new(vec![1], vec![max_val]))
            }
            ReductionKind::Min => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                unsafe {
                    if std::is_x86_feature_detected!("avx2") {
                        #[cfg(target_arch = "x86_64")]
                        use std::arch::x86_64::*;
                        #[cfg(target_arch = "x86")]
                        use std::arch::x86::*;

                        let mut i = 0usize;
                        let mut acc = _mm256_set1_ps(f32::INFINITY);

                        while i + 8 <= n {
                            let p = _mm256_loadu_ps(a.data.as_ptr().add(i));
                            acc = _mm256_min_ps(acc, p);
                            i += 8;
                        }

                        // Horizontal min of acc
                        let mut s = [0.0f32; 8];
                        _mm256_storeu_ps(s.as_mut_ptr(), acc);
                        let mut min_val = s[0];
                        for &v in &s[1..] {
                            if v < min_val {
                                min_val = v;
                            }
                        }

                        // tail
                        while i < n {
                            if a.data[i] < min_val {
                                min_val = a.data[i];
                            }
                            i += 1;
                        }

                        return Ok(Array::new(vec![1], vec![min_val]));
                    }
                }

                // Fallback to scalar
                let min_val = a.data.iter().copied().fold(f32::INFINITY, |acc, x| acc.min(x));
                Ok(Array::new(vec![1], vec![min_val]))
            }
            ReductionKind::Mean => {
                // Reuse sum and divide
                let sum_result = reduce_simd(a, axis, ReductionKind::Sum)?;
                let mean = sum_result.data[0] / n as f32;
                Ok(Array::new(vec![1], vec![mean]))
            }
            ReductionKind::ArgMax | ReductionKind::Variance => {
                // Fallback to scalar - complex algorithms
                crate::backend::cpu::scalar::reduce_scalar(a, None, kind)
            }
        }
    } else {
        // Axis-based reduction
        let axis = axis.unwrap();
        
        // OPTIMIZED PATH: Reducing over last axis with SIMD
        if axis == a.shape.len() - 1 {
            return reduce_last_axis_simd(a, axis, kind);
        }
        
        // For other axes, fallback to scalar implementation
        // TODO: optimize specific cases (e.g., reducing axis 0 with proper striding)
        crate::backend::cpu::scalar::reduce_scalar(a, Some(axis), kind)
    }
}

/// Optimized SIMD reduction over the last axis
/// This is the most cache-friendly case as data is contiguous
fn reduce_last_axis_simd(a: &Array, axis: usize, kind: ReductionKind) -> Result<Array> {

    
    // Compute output shape
    let mut out_shape: Vec<usize> = a.shape.iter().enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .collect();
    
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    
    let out_size: usize = out_shape.iter().product();
    let axis_size = a.shape[axis];
    let _out_data = vec![0.0; out_size];
    
    match kind {
        ReductionKind::Sum | ReductionKind::Mean => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("avx2") {
                    out_data.par_iter_mut().enumerate().for_each(|(row_idx, out_val)| {
                        let start = row_idx * axis_size;
                        let end = start + axis_size;
                        
                        unsafe {
                            #[cfg(target_arch = "x86_64")]
                            use std::arch::x86_64::*;
                            #[cfg(target_arch = "x86")]
                            use std::arch::x86::*;
                            
                            let mut acc = _mm256_setzero_ps();
                            let mut i = start;
                            
                            // Process 8 elements at a time with SIMD
                            while i + 8 <= end {
                                let p = _mm256_loadu_ps(a.data.as_ptr().add(i));
                                acc = _mm256_add_ps(acc, p);
                                i += 8;
                            }
                            
                            // Horizontal sum
                            let mut s = [0.0f32; 8];
                            _mm256_storeu_ps(s.as_mut_ptr(), acc);
                            let mut sum: f32 = s.iter().sum();
                            
                            // Handle remaining elements
                            while i < end {
                                sum += a.data[i];
                                i += 1;
                            }
                            
                            *out_val = sum;
                        }
                    });
                    
                    if kind == ReductionKind::Mean {
                        out_data.par_iter_mut().for_each(|x| *x /= axis_size as f32);
                    }
                    
                    return Ok(Array::new(out_shape, out_data));
                }
            }
            
            // Fallback to scalar
            return crate::backend::cpu::scalar::reduce_last_axis_optimized(
                a, axis_size, out_size, out_shape, kind
            );
        }
        ReductionKind::Max => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("avx2") {
                    out_data.par_iter_mut().enumerate().for_each(|(row_idx, out_val)| {
                        let start = row_idx * axis_size;
                        let end = start + axis_size;
                        
                        unsafe {
                            #[cfg(target_arch = "x86_64")]
                            use std::arch::x86_64::*;
                            #[cfg(target_arch = "x86")]
                            use std::arch::x86::*;
                            
                            let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
                            let mut i = start;
                            
                            while i + 8 <= end {
                                let p = _mm256_loadu_ps(a.data.as_ptr().add(i));
                                acc = _mm256_max_ps(acc, p);
                                i += 8;
                            }
                            
                            // Horizontal max
                            let mut s = [0.0f32; 8];
                            _mm256_storeu_ps(s.as_mut_ptr(), acc);
                            let mut max_val = s[0];
                            for &v in &s[1..] {
                                if v > max_val {
                                    max_val = v;
                                }
                            }
                            
                            // Handle remaining elements
                            while i < end {
                                if a.data[i] > max_val {
                                    max_val = a.data[i];
                                }
                                i += 1;
                            }
                            
                            *out_val = max_val;
                        }
                    });
                    
                    return Ok(Array::new(out_shape, out_data));
                }
            }
            
            // Fallback
            return crate::backend::cpu::scalar::reduce_last_axis_optimized(
                a, axis_size, out_size, out_shape, kind
            );
        }
        _ => {
            // For other operations, use scalar optimized version
            return crate::backend::cpu::scalar::reduce_last_axis_optimized(
                a, axis_size, out_size, out_shape, kind
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::scalar as scalar;

    fn make_arrays(len: usize) -> (Array, Array) {
        let a = (0..len).map(|i| i as f32 * 0.5 + 0.1).collect::<Vec<_>>();
        let b = (0..len).map(|i| (i as f32).sin()).collect::<Vec<_>>();
        (Array::new(vec![len], a), Array::new(vec![len], b))
    }

    #[test]
    fn simd_add_matches_scalar() {
        for len in &[1usize, 3, 7, 8, 15, 16, 33, 64] {
            let (a, b) = make_arrays(*len);
            let out_simd = elementwise_simd(&a, &b, ElementwiseKind::Add).unwrap();
            let out_scalar = scalar::elementwise_scalar(&a, &b, ElementwiseKind::Add).unwrap();
            assert_eq!(out_simd.data, out_scalar.data);
        }
    }

    #[test]
    fn simd_mul_matches_scalar() {
        for len in &[1usize, 3, 7, 8, 15, 16, 33, 64] {
            let (a, b) = make_arrays(*len);
            let out_simd = elementwise_simd(&a, &b, ElementwiseKind::Mul).unwrap();
            let out_scalar = scalar::elementwise_scalar(&a, &b, ElementwiseKind::Mul).unwrap();
            assert_eq!(out_simd.data, out_scalar.data);
        }
    }
}

/// Dot product SIMD implementation with FMA (fused multiply-add)
pub fn dot_simd(a: &Array, b: &Array) -> Result<f32> {
    if a.shape.len() != 1 || b.shape.len() != 1 {
        return Err(anyhow!("dot_simd: both inputs must be 1-D arrays"));
    }
    if a.shape[0] != b.shape[0] {
        return Err(anyhow!("dot_simd: arrays must have same length"));
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx2") {
            // SAFETY: We checked for AVX2+FMA support
            unsafe { return dot_simd_avx2_fma(a, b); }
        }
    }

    // Fallback to scalar if SIMD not available
    crate::backend::cpu::scalar::dot_scalar(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_simd_avx2_fma(a: &Array, b: &Array) -> Result<f32> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.data.len();
    let mut sum = _mm256_setzero_ps();
    
    // Process 8 floats at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a.data.as_ptr().add(offset));
        let vb = _mm256_loadu_ps(b.data.as_ptr().add(offset));
        // FMA: sum = sum + (va * vb)
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of 8 lanes
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Handle remaining elements
    for i in (chunks * 8)..n {
        total += a.data[i] * b.data[i];
    }

    Ok(total)
}

/// SIMD-accelerated matrix multiplication
/// Uses blocked tiled algorithm with SIMD vectorization for inner loops
pub fn matmul_simd(a: &Array, b: &Array) -> Array {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        panic!("matmul_simd: both inputs must be 2-D arrays");
    }
    
    let _m = a.shape[0];
    let k = a.shape[1];
    let _n = b.shape[1];
    
    if k != b.shape[0] {
        panic!("matmul_simd: inner dimension mismatch: {} != {}", k, b.shape[0]);
    }
    
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // Siempre usar Rayon + SIMD para consistencia
            // El adaptive dispatch ya decidió si este kernel es apropiado
            return matmul_simd_parallel(a, b, m, k, n);
        }
    }
    
    // Fallback to scalar if SIMD not available
    super::matmul_scalar_direct(a, b)
}

/// SIMD matmul con paralelización Rayon para matrices grandes
/// Optimizado con zero-copy y bloques adaptativos
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn matmul_simd_parallel(a: &Array, b: &Array, m: usize, k: usize, n: usize) -> Array {
    use rayon::prelude::*;
    
    // Adaptive block size based on matrix dimensions
    // Larger blocks for better cache utilization on large matrices
    let block_size = if m >= 2048 { 256 } else { 128 };
    let mut result = vec![0.0f32; m * n];
    
    result.par_chunks_mut(block_size * n)
        .enumerate()
        .for_each(|(block_idx, out_block)| {
            let start = block_idx * block_size;
            let end = (start + block_size).min(m);
            let block_rows = end - start;
            
            // Zero-copy: use direct slice view of A (no allocation)
            let a_block_start = start * k;
            let a_block_end = end * k;
            let a_block_slice = &a.data[a_block_start..a_block_end];
            
            // Create temporary Array view without copying
            let a_block = Array::new(vec![block_rows, k], a_block_slice.to_vec());
            
            // Procesar bloque con SIMD
            // SAFETY: We already checked for AVX2+FMA in parent function
            unsafe {
                let block_result = matmul_simd_avx2_fma_blocked(
                    &a_block, 
                    b, 
                    block_rows, 
                    k, 
                    n, 
                    vec![0.0f32; block_rows * n]
                );
                out_block.copy_from_slice(&block_result.data);
            }
        });
    
    Array::new(vec![m, n], result)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_simd_avx2_fma_blocked(
    a: &Array,
    b: &Array,
    m: usize,
    k: usize,
    n: usize,
    mut result: Vec<f32>
) -> Array {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    
    // Optimized block sizes: larger blocks for better arithmetic intensity
    const BLOCK_M: usize = 96;  // Process 96 rows per block
    const BLOCK_N: usize = 256; // Process 256 cols per block (32 AVX2 registers)
    const BLOCK_K: usize = 512; // Larger K block for better data reuse
    
    // Blocked matrix multiplication with advanced SIMD
    for i0 in (0..m).step_by(BLOCK_M) {
        let i_end = (i0 + BLOCK_M).min(m);
        
        for j0 in (0..n).step_by(BLOCK_N) {
            let j_end = (j0 + BLOCK_N).min(n);
            
            for k0 in (0..k).step_by(BLOCK_K) {
                let k_end = (k0 + BLOCK_K).min(k);
                
                // Process 2 rows at a time for better register utilization
                let mut i = i0;
                while i + 2 <= i_end {
                    let a_row0_offset = i * k;
                    let a_row1_offset = (i + 1) * k;
                    let result_row0_offset = i * n;
                    let result_row1_offset = (i + 1) * n;
                    
                    // Process 16 columns at a time (2 AVX2 registers)
                    let mut j = j0;
                    while j + 16 <= j_end {
                        let mut sum0_0 = _mm256_loadu_ps(result.as_ptr().add(result_row0_offset + j));
                        let mut sum0_1 = _mm256_loadu_ps(result.as_ptr().add(result_row0_offset + j + 8));
                        let mut sum1_0 = _mm256_loadu_ps(result.as_ptr().add(result_row1_offset + j));
                        let mut sum1_1 = _mm256_loadu_ps(result.as_ptr().add(result_row1_offset + j + 8));
                        
                        // Inner loop: accumulate over k-block with FMA
                        for kk in k0..k_end {
                            let a_val0 = _mm256_set1_ps(a.data[a_row0_offset + kk]);
                            let a_val1 = _mm256_set1_ps(a.data[a_row1_offset + kk]);
                            let b_row_offset = kk * n;
                            let b_vals0 = _mm256_loadu_ps(b.data.as_ptr().add(b_row_offset + j));
                            let b_vals1 = _mm256_loadu_ps(b.data.as_ptr().add(b_row_offset + j + 8));
                            
                            // FMA: sum = sum + (a_val * b_vals)
                            // Process both rows and both column sets simultaneously
                            sum0_0 = _mm256_fmadd_ps(a_val0, b_vals0, sum0_0);
                            sum0_1 = _mm256_fmadd_ps(a_val0, b_vals1, sum0_1);
                            sum1_0 = _mm256_fmadd_ps(a_val1, b_vals0, sum1_0);
                            sum1_1 = _mm256_fmadd_ps(a_val1, b_vals1, sum1_1);
                        }
                        
                        _mm256_storeu_ps(result.as_mut_ptr().add(result_row0_offset + j), sum0_0);
                        _mm256_storeu_ps(result.as_mut_ptr().add(result_row0_offset + j + 8), sum0_1);
                        _mm256_storeu_ps(result.as_mut_ptr().add(result_row1_offset + j), sum1_0);
                        _mm256_storeu_ps(result.as_mut_ptr().add(result_row1_offset + j + 8), sum1_1);
                        j += 16;
                    }
                    
                    // Process remaining columns in chunks of 8
                    while j + 8 <= j_end {
                        let mut sum0 = _mm256_loadu_ps(result.as_ptr().add(result_row0_offset + j));
                        let mut sum1 = _mm256_loadu_ps(result.as_ptr().add(result_row1_offset + j));
                        
                        for kk in k0..k_end {
                            let a_val0 = _mm256_set1_ps(a.data[a_row0_offset + kk]);
                            let a_val1 = _mm256_set1_ps(a.data[a_row1_offset + kk]);
                            let b_vals = _mm256_loadu_ps(b.data.as_ptr().add(kk * n + j));
                            
                            sum0 = _mm256_fmadd_ps(a_val0, b_vals, sum0);
                            sum1 = _mm256_fmadd_ps(a_val1, b_vals, sum1);
                        }
                        
                        _mm256_storeu_ps(result.as_mut_ptr().add(result_row0_offset + j), sum0);
                        _mm256_storeu_ps(result.as_mut_ptr().add(result_row1_offset + j), sum1);
                        j += 8;
                    }
                    
                    // Handle remaining columns with scalar code
                    for j in j..j_end {
                        let mut sum0 = result[result_row0_offset + j];
                        let mut sum1 = result[result_row1_offset + j];
                        for kk in k0..k_end {
                            let b_val = b.data[kk * n + j];
                            sum0 += a.data[a_row0_offset + kk] * b_val;
                            sum1 += a.data[a_row1_offset + kk] * b_val;
                        }
                        result[result_row0_offset + j] = sum0;
                        result[result_row1_offset + j] = sum1;
                    }
                    
                    i += 2;
                }
                
                // Handle remaining single row if m is odd
                if i < i_end {
                    let a_row_offset = i * k;
                    let result_row_offset = i * n;
                    
                    let mut j = j0;
                    while j + 8 <= j_end {
                        let mut sum = _mm256_loadu_ps(result.as_ptr().add(result_row_offset + j));
                        
                        for kk in k0..k_end {
                            let a_val = _mm256_set1_ps(a.data[a_row_offset + kk]);
                            let b_vals = _mm256_loadu_ps(b.data.as_ptr().add(kk * n + j));
                            sum = _mm256_fmadd_ps(a_val, b_vals, sum);
                        }
                        
                        _mm256_storeu_ps(result.as_mut_ptr().add(result_row_offset + j), sum);
                        j += 8;
                    }
                    
                    for j in j..j_end {
                        let mut sum = result[result_row_offset + j];
                        for kk in k0..k_end {
                            sum += a.data[a_row_offset + kk] * b.data[kk * n + j];
                        }
                        result[result_row_offset + j] = sum;
                    }
                }
            }
        }
    }
    
    Array::new(vec![m, n], result)
}

/// SIMD implementation of Conv1D (Stub)
/// Re-export Conv1D SIMD implementation
pub use super::simd_conv::conv1d_simd;
