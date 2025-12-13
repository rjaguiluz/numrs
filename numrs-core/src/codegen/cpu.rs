/// CPU codegen - for prototype we produce simple textual "IR" or Rust kernel string
pub fn scalar_kernel_template() -> &'static str {
    "// Rust scalar loop kernel: apply elementwise(op, a, b) -> out\nfor i in 0..N { out[i] = op(a[i], b[i]); }"
}

pub fn simd_kernel_template() -> &'static str {
    "// SIMD kernel template (prototype) - replace with real intrinsics per arch"
}
