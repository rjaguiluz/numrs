# specialized webgpu backend

This directory contains the experimental GPU backend using `wgpu`.

## Contents

- **`mod.rs`**: The main driver for WebGPU execution. It handles device selection, buffer management (host to device copy), pipeline creation, and shader execution.
- **`codegen.rs`**: Helper logic to generate WGSL (WebGPU Shading Language) shader code dynamically at runtime. This allows us to fuse operations or specialize kernels for specific array shapes/strides.

## Design Rationale

- **Cross-Platform GPU**: We chose `wgpu` because it abstracts Vulkan, DirectX, Metal, and OpenGL, allowing NumRs to run on almost any GPU.
- **Asynchronous Execution**: GPU operations are inherently async. This backend is designing to manage command encoders and queues, submitting work and waiting for results only when explicitly requested (synchronization points).
- **Future-Proofing**: As WebGPU becomes standard in browsers, this backend will enable NumRs to run high-performance ML in the browser via WASM without code changes.

## Interaction

1.  **Selection**: The `dispatch.rs` system will select this backend if the `webgpu` feature is enabled AND a compatible GPU adapter is found at runtime.
2.  **Buffers**: Arrays are uploaded to GPU buffers (`wgpu::Buffer`).
3.  **Shaders**: Compute shaders (written in WGSL) are dispatched to process the data in parallel on thousands of GPU threads.
