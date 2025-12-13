//! Backend heuristics and compile-time constants used for strategy selection
use crate::llo::ElementwiseStrategy;

/// Minimum number of elements where vectorized/SIMD path is expected to be beneficial
pub const ELEMENTWISE_SIMD_MIN_ELEMENTS: usize = 64;

/// Minimum number of elements where a GPU kernel may become more optimal
pub const ELEMENTWISE_GPU_MIN_ELEMENTS: usize = 1024;

/// Minimum workload for which a parallel reduction makes sense
pub const REDUCTION_PARALLEL_MIN: usize = 256;

/// Choose an elementwise strategy given the total element count and availability flags.
pub fn choose_elementwise_strategy(
    total_elems: usize,
    simd_available: bool,
    gpu_available: bool,
) -> ElementwiseStrategy {
    // Prefer compile-time selected kernels when available (emitted by build.rs).
    // This makes the decision deterministic at build time when those cfgs exist.
    // If no compile-time flag is present, fall back to runtime heuristics.
    if cfg!(numrs_kernel_elementwise_gpu) {
        return ElementwiseStrategy::GpuKernel;
    }

    if cfg!(numrs_kernel_elementwise_simd) {
        return ElementwiseStrategy::Vectorized;
    }

    // Fallback to runtime heuristics when compile-time kernels aren't specified.
    if gpu_available && total_elems >= ELEMENTWISE_GPU_MIN_ELEMENTS {
        ElementwiseStrategy::GpuKernel
    } else if simd_available && total_elems >= ELEMENTWISE_SIMD_MIN_ELEMENTS {
        ElementwiseStrategy::Vectorized
    } else {
        ElementwiseStrategy::Scalar
    }
}

/// Choose reduction strategy (keeps prototype simple)
pub fn choose_reduction_strategy(total_elems: usize) -> crate::llo::ReductionStrategy {
    if total_elems >= REDUCTION_PARALLEL_MIN {
        crate::llo::ReductionStrategy::Parallel
    } else {
        crate::llo::ReductionStrategy::ScalarLoop
    }
}
