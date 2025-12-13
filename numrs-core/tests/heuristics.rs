use numrs::backend::heuristics::*;
use numrs::llo::ElementwiseStrategy;

#[test]
fn heuristic_elementwise_choice_simd() {
    // Test de selección de estrategia elementwise con diferentes tamaños

    // Workloads pequeños deben usar Scalar (a menos que compile-time SIMD esté forzado)
    let small_result = choose_elementwise_strategy(10, true, false);

    if cfg!(numrs_kernel_elementwise_gpu) {
        // Compile-time GPU forces GpuKernel even for small workloads
        assert_eq!(
            small_result,
            ElementwiseStrategy::GpuKernel,
            "Compile-time GPU fuerza GpuKernel"
        );
    } else if cfg!(numrs_kernel_elementwise_simd) {
        // Compile-time SIMD kernel presente → elección compile-time fuerza Vectorized
        assert_eq!(
            small_result,
            ElementwiseStrategy::Vectorized,
            "Con compile-time SIMD, incluso workloads pequeños usan Vectorized"
        );
    } else {
        // Sin compile-time kernels, workload pequeño usa Scalar
        assert_eq!(
            small_result,
            ElementwiseStrategy::Scalar,
            "Sin compile-time kernels, workloads pequeños usan Scalar"
        );
    }

    // Workloads medianos deben usar SIMD cuando disponible (supera threshold)
    let medium_result = choose_elementwise_strategy(ELEMENTWISE_SIMD_MIN_ELEMENTS, true, false);

    if cfg!(numrs_kernel_elementwise_gpu) {
        assert_eq!(medium_result, ElementwiseStrategy::GpuKernel);
    } else if cfg!(numrs_kernel_elementwise_simd) {
        // Compile-time SIMD siempre gana
        assert_eq!(
            medium_result,
            ElementwiseStrategy::Vectorized,
            "Compile-time SIMD selecciona Vectorized"
        );
    } else {
        // Runtime: debe preferir Vectorized cuando simd_available=true y size >= threshold
        assert_eq!(
            medium_result,
            ElementwiseStrategy::Vectorized,
            "Runtime con SIMD disponible y size >= threshold debe usar Vectorized"
        );
    }

    // Workloads muy grandes pueden sugerir GPU si disponible
    let large_result = choose_elementwise_strategy(ELEMENTWISE_GPU_MIN_ELEMENTS, true, true);

    // La prioridad depende de cfgs compile-time
    if cfg!(numrs_kernel_elementwise_gpu) {
        // Si GPU está compilado, tiene mayor prioridad
        assert_eq!(
            large_result,
            ElementwiseStrategy::GpuKernel,
            "Compile-time GPU kernel tiene mayor prioridad"
        );
    } else if cfg!(numrs_kernel_elementwise_simd) {
        // Si SIMD está compilado pero GPU no, usa SIMD
        assert_eq!(
            large_result,
            ElementwiseStrategy::Vectorized,
            "Sin GPU compile-time, usa SIMD compile-time"
        );
    } else {
        // Runtime: GPU disponible y size grande → GpuKernel
        assert_eq!(
            large_result,
            ElementwiseStrategy::GpuKernel,
            "Runtime con GPU disponible y size grande debe usar GPU"
        );
    }
}

#[test]
fn heuristic_elementwise_no_simd() {
    // Test cuando SIMD no está disponible (runtime)
    let result = choose_elementwise_strategy(100, false, false);

    if cfg!(numrs_kernel_elementwise_simd) || cfg!(numrs_kernel_elementwise_gpu) {
        // Compile-time kernel forzado, ignora runtime availability
        // (el test pasa porque la decisión es compile-time)
        return;
    }

    // Sin compile-time kernels y sin runtime SIMD/GPU → Scalar
    assert_eq!(
        result,
        ElementwiseStrategy::Scalar,
        "Sin SIMD/GPU disponible debe usar Scalar"
    );
}

#[test]
fn heuristic_reduction_strategy() {
    // Test de selección de estrategia de reducción

    // Workload pequeño → ScalarLoop
    let small = choose_reduction_strategy(100);
    assert_eq!(
        small,
        numrs::llo::ReductionStrategy::ScalarLoop,
        "Reducción pequeña usa ScalarLoop"
    );

    // Workload grande → Parallel
    let large = choose_reduction_strategy(REDUCTION_PARALLEL_MIN);
    assert_eq!(
        large,
        numrs::llo::ReductionStrategy::Parallel,
        "Reducción grande usa Parallel cuando supera threshold"
    );
}
