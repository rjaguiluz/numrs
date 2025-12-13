//! Test that prints which kernel implementations are available and which will
//! be selected at compile-time for `sum` (reduction) and `mul` (elementwise).
//!
//! This inspects the kernel-level cfg flags emitted by `build.rs` so you can
//! confirm when building from JS which method was chosen.

#[test]
fn print_selected_methods() {
    // Describe priority mapping for SUM (reduction)
    println!("sum: WHEN (numrs_kernel_sum_blas) THEN sum_blas, WHEN (numrs_kernel_sum_simd) THEN sum_simd, WHEN (numrs_kernel_sum_gpu) THEN sum_gpu, ELSE sum_scalar");

    let sum_selected = if cfg!(numrs_kernel_sum_blas) { "sum_blas" }
        else if cfg!(numrs_kernel_sum_simd) { "sum_simd" }
        else if cfg!(numrs_kernel_sum_gpu) { "sum_gpu" }
        else { "sum_scalar" };

    println!("selected for sum: {}", sum_selected);

    // Describe priority mapping for MUL (elementwise)
    println!("mul: WHEN (numrs_kernel_elementwise_blas) THEN elementwise_blas, WHEN (numrs_kernel_elementwise_simd) THEN elementwise_simd, WHEN (numrs_kernel_elementwise_gpu) THEN elementwise_gpu, ELSE elementwise_scalar");

    let mul_selected = if cfg!(numrs_kernel_elementwise_blas) { "elementwise_blas" }
        else if cfg!(numrs_kernel_elementwise_simd) { "elementwise_simd" }
        else if cfg!(numrs_kernel_elementwise_gpu) { "elementwise_gpu" }
        else { "elementwise_scalar" };

    println!("selected for mul: {}", mul_selected);

    // Helpful hint for CI / JS build debugging
    println!("To view this output when running tests: cargo test print_selected_methods -- --nocapture");
}
