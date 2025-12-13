fn main() {
    // Detect target OS from env set by Cargo
    let target = std::env::var("TARGET").unwrap_or_default();

    // Announce custom cfg names for `check-cfg` so rustc won't warn about
    // unexpected cfg identifiers when we use them in source files.
    // note: OpenBLAS automatic detection removed; prefer BLIS/MKL/Accelerate
    println!("cargo:rustc-check-cfg=cfg(numrs_has_mkl)");
    println!("cargo:rustc-check-cfg=cfg(numrs_has_blis)");
    println!("cargo:rustc-check-cfg=cfg(numrs_has_blas)");
    println!("cargo:rustc-check-cfg=cfg(numrs_has_accelerate)");
    println!("cargo:rustc-check-cfg=cfg(numrs_has_avx2)");
    println!("cargo:rustc-check-cfg=cfg(numrs_has_fma)");
    println!("cargo:rustc-check-cfg=cfg(numrs_prefer_blas_for_matmul)");
    // Also declare kernel-level cfg names for compile-time selection logic
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_matmul_blas)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_matmul_simd)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_matmul_gpu)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_matmul_scalar)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_sum_blas)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_sum_simd)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_sum_gpu)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_sum_scalar)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_elementwise_blas)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_elementwise_simd)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_elementwise_gpu)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_elementwise_scalar)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_conv_gpu)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_conv_simd)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_batchnorm_gpu)");
    println!("cargo:rustc-check-cfg=cfg(numrs_kernel_dropout_gpu)");

    // Allow an opt-in override to compile all kernel implementations.
    // When `NUMRS_COMPILE_ALL_KERNELS=1` is set in the environment, we
    // emit all `numrs_kernel_*` cfgs so that every implementation is
    // compiled into the crate and runtime can probe them at startup.
    let compile_all = std::env::var("NUMRS_COMPILE_ALL_KERNELS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    if compile_all {
        println!("cargo:warning=NumRs: NUMRS_COMPILE_ALL_KERNELS=1 -> compiling all kernel variants for runtime probing");
        println!("cargo:rustc-cfg=numrs_kernel_matmul_blas");
        println!("cargo:rustc-cfg=numrs_kernel_matmul_simd");
        println!("cargo:rustc-cfg=numrs_kernel_matmul_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_matmul_scalar");
        println!("cargo:rustc-cfg=numrs_kernel_sum_blas");
        println!("cargo:rustc-cfg=numrs_kernel_sum_simd");
        println!("cargo:rustc-cfg=numrs_kernel_sum_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_sum_scalar");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_blas");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_simd");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_scalar");
        println!("cargo:rustc-cfg=numrs_kernel_conv_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_conv_simd");
        println!("cargo:rustc-cfg=numrs_kernel_batchnorm_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_dropout_gpu");
    }

    // Default: no BLAS providers detected
    let mut has_accelerate = false;
    let mut has_mkl = false;
    let mut has_blis = false;

    // STATIC LINKING STRATEGY:
    // 1. macOS → Accelerate (system framework, always available)
    // 2. x86_64 (not wasm32) → MKL via intel-mkl-src (feature "mkl")
    // 3. ARM/other (not wasm32) → BLIS via blis-src (feature "blis")
    // 4. wasm32 → No BLAS (pure Rust)
    // 5. disabled-blas feature → Force no BLAS

    let is_wasm = target.contains("wasm32");
    let blas_disabled = std::env::var("CARGO_FEATURE_DISABLED_BLAS").is_ok();

    if blas_disabled {
        println!("cargo:warning=NumRs: disabled-blas feature enabled -> no BLAS (pure Rust)");
    } else if target.contains("apple-darwin") || target.contains("macos") {
        // On macOS prefer Accelerate framework (statically available)
        has_accelerate = true;
        println!("cargo:warning=NumRs: macOS detected -> using Accelerate framework (static)");
    } else if !is_wasm && (target.contains("x86_64") || target.contains("x86")) {
        // x86_64: MKL is automatically linked via target-specific dependency
        // No feature flag needed - it's always present on x86_64
        has_mkl = true;
        println!("cargo:warning=NumRs: x86_64 detected -> using intel-mkl-src (static, automatic)");
    } else if !is_wasm {
        // ARM, RISC-V, etc: BLIS is automatically linked via target-specific dependency
        has_blis = true;
        println!("cargo:warning=NumRs: Non-x86 architecture detected -> using blis-src (static, automatic)");
    } else {
        println!("cargo:warning=NumRs: wasm32 detected -> no BLAS (pure Rust)");
    }

    // Remove dynamic linking detection logic (Windows vcpkg, pkg-config)
    // All BLAS linking is now static via -src crates

    // Allow opt-in override for MKL if user wants to force it
    let prefer_mkl_env = std::env::var("NUMRS_USE_MKL")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    let prefer_blis_env = std::env::var("NUMRS_PREFER_BLIS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);

    if prefer_mkl_env && !has_mkl {
        println!("cargo:warning=NumRs: NUMRS_USE_MKL=1 but mkl feature not enabled, ignoring");
    }
    if prefer_blis_env && !has_blis {
        println!("cargo:warning=NumRs: NUMRS_PREFER_BLIS=1 but blis feature not enabled, ignoring");
    }

    if has_accelerate {
        println!("cargo:rustc-cfg=numrs_has_accelerate");
        println!("cargo:rustc-env=NUMRS_HAS_ACCELERATE=1");
        // Mark BLAS-backed kernel availability
        println!("cargo:rustc-cfg=numrs_kernel_matmul_blas");
        println!("cargo:rustc-cfg=numrs_kernel_sum_blas");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_blas");
    }
    if has_mkl {
        println!("cargo:rustc-cfg=numrs_has_mkl");
        println!("cargo:rustc-env=NUMRS_HAS_MKL=1");
        println!("cargo:rustc-cfg=numrs_kernel_matmul_blas");
        println!("cargo:rustc-cfg=numrs_kernel_sum_blas");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_blas");
        // intel-mkl-src handles linking automatically when feature is enabled
        // No manual link directives needed here
    }
    if has_blis {
        println!("cargo:rustc-cfg=numrs_has_blis");
        println!("cargo:rustc-env=NUMRS_HAS_BLIS=1");
        println!("cargo:rustc-cfg=numrs_kernel_matmul_blas");
        println!("cargo:rustc-cfg=numrs_kernel_sum_blas");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_blas");
        // blis-src handles linking automatically when feature is enabled
    }
    // Note: removed OpenBLAS automatic detection and runtime DLL deployment
    // All BLAS providers are now statically linked via -src crates

    // If any provider present, expose a generic BLAS cfg used by runtime
    // Only consider BLIS, MKL or Accelerate as BLAS providers. OpenBLAS is
    // intentionally excluded from automatic detection to prefer BLIS.
    // Also respect disabled-blas feature which forces BLAS off
    let has_blas = !blas_disabled && (has_accelerate || has_mkl || has_blis);
    if has_blas {
        println!("cargo:rustc-cfg=numrs_has_blas");
    }

    // Attempt to detect host CPU features when building for the host (no cross-compilation).
    // If cross-compiling, allow env variables to override detection: NUMRS_ASSUME_AVX2, NUMRS_ASSUME_FMA
    let host = std::env::var("HOST").unwrap_or_default();
    let target = std::env::var("TARGET").unwrap_or_default();

    let mut has_avx2 = false;
    let mut has_fma = false;
    let mut has_wasm_simd = false;

    let disable_wasm_simd = std::env::var("NUMRS_DISABLE_WASM_SIMD")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);

    if is_wasm && !disable_wasm_simd {
        // For WASM, enable SIMD128 by default
        // SIMD128 is widely supported in modern browsers
        has_wasm_simd = true;
        println!("cargo:rustc-cfg=numrs_has_wasm_simd");
        println!("cargo:rustc-cfg=numrs_kernel_matmul_simd");
        println!("cargo:rustc-cfg=numrs_kernel_sum_simd");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_simd");
        println!("cargo:warning=NumRs: WASM SIMD128 enabled");
    } else if host == target && (target.contains("x86_64") || target.contains("x86")) {
        // We can check the host CPU features
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            has_avx2 = std::is_x86_feature_detected!("avx2");
            has_fma = std::is_x86_feature_detected!("fma");

            // Expose detection at compile-time so selection macros can use cfg
            if has_avx2 {
                println!("cargo:rustc-cfg=numrs_has_avx2");
            }
            if has_avx2 {
                // SIMD kernels available
                println!("cargo:rustc-cfg=numrs_kernel_matmul_simd");
                println!("cargo:rustc-cfg=numrs_kernel_sum_simd");
                println!("cargo:rustc-cfg=numrs_kernel_elementwise_simd");
                println!("cargo:rustc-cfg=numrs_kernel_conv_simd");
            }
            if has_fma {
                println!("cargo:rustc-cfg=numrs_has_fma");
            }
        }
    } else {
        // Cross-compiling — use offers from environment or conservative defaults
        if let Ok(v) = std::env::var("NUMRS_ASSUME_AVX2") {
            has_avx2 = matches!(v.as_str(), "1" | "true" | "TRUE");
        }
        if let Ok(v) = std::env::var("NUMRS_ASSUME_FMA") {
            has_fma = matches!(v.as_str(), "1" | "true" | "TRUE");
        }
    }

    // GPU flag from features or env var
    // Check for webgpu or webgl features
    let has_webgpu = cfg!(feature = "webgpu") || std::env::var("CARGO_FEATURE_WEBGPU").is_ok();
    let has_webgl = cfg!(feature = "webgl") || std::env::var("CARGO_FEATURE_WEBGL").is_ok();
    let has_gpu_env = std::env::var("NUMRS_HAS_GPU")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);
    let has_gpu = has_webgpu || has_webgl || has_gpu_env;

    if has_gpu {
        println!("cargo:rustc-cfg=numrs_kernel_matmul_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_sum_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_elementwise_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_conv_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_batchnorm_gpu");
        println!("cargo:rustc-cfg=numrs_kernel_dropout_gpu");
    }

    // alignment heuristic
    let alignment_32 = target.contains("64") && !is_wasm;

    // Decide default methods (best effort) and print them for the user in Spanish
    let sum_method = if has_blas {
        "sum_blas"
    } else if has_avx2 || has_wasm_simd {
        "sum_simd"
    } else {
        "sum_scalar"
    };

    let elementwise_method = if has_gpu {
        "elementwise_gpu"
    } else if has_avx2 || has_wasm_simd {
        "elementwise_simd"
    } else {
        "elementwise_scalar"
    };

    // Selection priority: on Apple prefer the Accelerate framework first.
    // Otherwise prefer MKL when available, then BLIS. GPU/SIMD/Scalar follow.
    let matmul_method = if has_accelerate {
        "matmul_accel"
    } else if has_mkl {
        "matmul_mkl"
    } else if has_blis {
        "matmul_blis"
    } else if has_gpu {
        "matmul_gpu"
    } else if has_avx2 || has_wasm_simd {
        "matmul_simd"
    } else {
        "matmul_scalar"
    };

    // Expose scalar kernels (always available) so compile-time selection can fall back
    println!("cargo:rustc-cfg=numrs_kernel_matmul_scalar");
    println!("cargo:rustc-cfg=numrs_kernel_sum_scalar");
    println!("cargo:rustc-cfg=numrs_kernel_elementwise_scalar");

    // Emit a summary to the build output so users see which methods are preferred.
    println!("cargo:warning=NumRs: detectadas capacidades -> avx2={}, fma={}, mkl={}, blis={}, accelerate={}, blas_any={}, gpu={}, wasm={}, wasm_simd={}, align32={}", has_avx2, has_fma, has_mkl, has_blis, has_accelerate, has_blas, has_gpu, is_wasm, has_wasm_simd, alignment_32);
    println!("cargo:warning=NumRs: selección por defecto de métodos (prioridad):");
    println!("cargo:warning=  SUMA:   {}", sum_method);
    println!("cargo:warning=  ELEM:   {}", elementwise_method);
    println!("cargo:warning=  MATMUL: {}", matmul_method);

    // Allow opt-in preference for BLAS matmul at build time
    if std::env::var("NUMRS_PREFER_BLAS_FOR_MATMUL")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
    {
        println!("cargo:rustc-cfg=numrs_prefer_blas_for_matmul");
        println!("cargo:warning=NumRs: opt-in build: prefer BLAS for MatMul enabled (NUMRS_PREFER_BLAS_FOR_MATMUL)");
    }
}
