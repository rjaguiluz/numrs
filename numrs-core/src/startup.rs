/// Print a concise startup banner / log with available public APIs and detected backend
pub fn print_startup_log() {
    // Initialize dispatch table (valida backends y selecciona mejores implementaciones)
    let dispatch_table = crate::backend::get_dispatch_table();
    let validation = crate::backend::validate_backends();

    // Use compile-time package name/version for clarity
    let name = env!("CARGO_PKG_NAME");
    let version = env!("CARGO_PKG_VERSION");

    // What public functions and macro surfaces we advertise
    let funcs = [
        "add",
        "mul",
        "sum",
        "Array::new",
        "add_ct! / mul_ct! / sum_ct!",
    ];

    // Available backends in this prototype
    let backends = ["cpu", "webgpu", "cuda", "metal", "blas"];

    // Determine selected backend from dispatch validation
    let selected = if validation.blas_validated {
        "blas"
    } else if validation.webgpu_validated {
        "webgpu"
    } else if validation.gpu_validated {
        "cuda/metal"
    } else if validation.simd_validated {
        "cpu-simd"
    } else {
        "cpu-scalar"
    };

    // Emit a compact log banner — use stderr so it is visible when loaded by other runtimes
    eprintln!(
        "[{} {}] available APIs: {}",
        name,
        version,
        funcs.join(", ")
    );
    eprintln!(
        "[{} {}] supported backends: {}",
        name,
        version,
        backends.join(", ")
    );
    eprintln!("[{} {}] selected backend: {}", name, version, selected);

    // Show per-API preferred backend mapping from dispatch table
    eprintln!(
        "[{} {}] op -> backend mapping: add -> {} ; mul -> {} ; sum -> {} ; matmul -> {}",
        name,
        version,
        dispatch_table.elementwise_backend,
        dispatch_table.elementwise_backend, // mul uses same as add
        dispatch_table.reduction_backend,
        dispatch_table.matmul_backend
    );

    // Print runtime-detected capabilities
    let caps = crate::backend::RuntimeCapabilities {
        has_simd: validation.simd_validated,
        has_gpu: validation.gpu_validated || validation.webgpu_validated,
        has_blas: validation.blas_validated,
        has_threads: num_cpus::get() > 1,
        has_wasm_simd: false,
        has_webgpu: validation.webgpu_validated,
    };
    eprintln!("[{} {}] runtime caps: has_simd={} has_gpu={} has_blas={} has_threads={} has_wasm_simd={} has_webgpu={}",
        name, version, caps.has_simd, caps.has_gpu, caps.has_blas, caps.has_threads, caps.has_wasm_simd, caps.has_webgpu);

    // Final dispatch table info
    eprintln!("[{} {}] === DISPATCH TABLE (ZERO-COST) ===", name, version);
    eprintln!(
        "[{} {}] elementwise: {} | reduction: {} | matmul: {} | dot: {}",
        name,
        version,
        dispatch_table.elementwise_backend,
        dispatch_table.reduction_backend,
        dispatch_table.matmul_backend,
        dispatch_table.dot_backend
    );
    eprintln!(
        "[{} {}] validation: blas={} metal={} webgpu={} simd={}",
        name,
        version,
        validation.blas_validated,
        validation.metal_validated,
        validation.webgpu_validated,
        validation.simd_validated
    );
}
