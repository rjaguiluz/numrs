use crate::array::Array;
use anyhow::{Result, anyhow};
use once_cell::sync::OnceCell;

#[cfg(target_os = "macos")]
use metal::*;
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;
#[cfg(target_os = "macos")]
use std::sync::Mutex;
#[cfg(target_os = "macos")]
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MetalBackend {}

impl MetalBackend {
    pub fn new() -> Self { Self {} }

    /// Check if Metal is available (macOS only)
    pub fn is_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            Device::system_default().is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

// ============================================================================
// Optimized Metal Context with Caching
// ============================================================================

#[cfg(target_os = "macos")]
struct MetalContext {
    device: Device,
    queue: CommandQueue,
    max_threads_per_threadgroup: u64,
    // Cached pipelines
    _elementwise_vec4_pipeline: ComputePipelineState,
    elementwise_scalar_pipeline: ComputePipelineState,
    matmul_pipeline_cache: Mutex<HashMap<(u32, u32), ComputePipelineState>>,
    reduction_pipeline: ComputePipelineState,
    // Buffer pool
    buffer_pool: Mutex<BufferPool>,
}

#[cfg(target_os = "macos")]
struct BufferPool {
    free_buffers: HashMap<usize, Vec<Buffer>>,
    max_cached_size: usize,
}

#[cfg(target_os = "macos")]
impl BufferPool {
    fn new() -> Self {
        Self {
            free_buffers: HashMap::new(),
            max_cached_size: 100 * 1024 * 1024, // 100 MB max per size bucket
        }
    }

    fn get_or_create(&mut self, device: &Device, size: usize, mode: MTLResourceOptions) -> Buffer {
        // Round up to nearest power of 2 for better reuse
        let bucket_size = size.next_power_of_two();
        
        if bucket_size <= self.max_cached_size {
            if let Some(buffers) = self.free_buffers.get_mut(&bucket_size) {
                if let Some(buffer) = buffers.pop() {
                    return buffer;
                }
            }
        }
        
        device.new_buffer(bucket_size as u64, mode)
    }

    fn return_buffer(&mut self, buffer: Buffer, size: usize) {
        let bucket_size = size.next_power_of_two();
        
        if bucket_size <= self.max_cached_size {
            self.free_buffers.entry(bucket_size)
                .or_insert_with(Vec::new)
                .push(buffer);
        }
    }
}

#[cfg(target_os = "macos")]
static METAL_DEVICE: OnceCell<Result<MetalContext, anyhow::Error>> = OnceCell::new();

#[cfg(target_os = "macos")]
fn get_metal_device() -> Result<&'static MetalContext> {
    METAL_DEVICE.get_or_init(|| {
        autoreleasepool(|| {
            let device = Device::system_default()
                .ok_or_else(|| anyhow!("No Metal device available"))?;
            let queue = device.new_command_queue();

            // Get device capabilities
            let max_threads_per_threadgroup = device.max_threads_per_threadgroup().width;

            // Compile all shaders ONCE
            let elementwise_vec4_pipeline = compile_elementwise_vec4_pipeline(&device)?;
            let elementwise_scalar_pipeline = compile_elementwise_scalar_pipeline(&device)?;
            let reduction_pipeline = compile_reduction_pipeline(&device)?;

            Ok(MetalContext {
                device,
                queue,
                max_threads_per_threadgroup,
                _elementwise_vec4_pipeline: elementwise_vec4_pipeline,
                elementwise_scalar_pipeline,
                matmul_pipeline_cache: Mutex::new(HashMap::new()),
                reduction_pipeline,
                buffer_pool: Mutex::new(BufferPool::new()),
            })
        })
    });

    match METAL_DEVICE.get().unwrap() {
        Ok(ctx) => Ok(ctx),
        Err(e) => Err(anyhow!("Metal init failed: {:?}", e)),
    }
}

// ============================================================================
// Shader Compilation (ONCE per shader type)
// ============================================================================

#[cfg(target_os = "macos")]
fn compile_elementwise_vec4_pipeline(device: &Device) -> Result<ComputePipelineState> {
    let shader_src = r#"
#include <metal_stdlib>
using namespace metal;

constant uint OP_ADD = 0; constant uint OP_MUL = 1; constant uint OP_SUB = 2; constant uint OP_DIV = 3;
constant uint OP_SQRT = 4; constant uint OP_SIN = 5; constant uint OP_COS = 6; constant uint OP_POW = 7;
constant uint OP_ABS = 8; constant uint OP_EXP = 9; constant uint OP_LOG = 10; constant uint OP_TAN = 11;
constant uint OP_ASIN = 12; constant uint OP_ACOS = 13; constant uint OP_ATAN = 14; constant uint OP_RELU = 15;
constant uint OP_LEAKY_RELU = 16; constant uint OP_SIGMOID = 17; constant uint OP_TANH = 18; constant uint OP_SOFTPLUS = 19;

struct Params { uint size; uint op_kind; };

kernel void elementwise_vec4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.size / 4) return;
    float4 a_val = a[idx];
    float4 b_val = b[idx];
    float4 result;
    
    switch(params.op_kind) {
        case OP_ADD: result = a_val + b_val; break;
        case OP_MUL: result = a_val * b_val; break;
        case OP_SUB: result = a_val - b_val; break;
        case OP_DIV: result = a_val / b_val; break;
        case OP_SQRT: result = fast::sqrt(a_val); break;
        case OP_SIN: result = fast::sin(a_val); break;
        case OP_COS: result = fast::cos(a_val); break;
        case OP_POW: result = fast::pow(a_val, b_val); break;
        case OP_ABS: result = fast::fabs(a_val); break;
        case OP_EXP: result = fast::exp(a_val); break;
        case OP_LOG: result = fast::log(a_val); break;
        case OP_TAN: result = fast::tan(a_val); break;
        case OP_ASIN: result = asin(a_val); break;
        case OP_ACOS: result = acos(a_val); break;
        case OP_ATAN: result = atan(a_val); break;
        case OP_RELU: result = fast::max(a_val, float4(0.0f)); break;
        case OP_LEAKY_RELU: result = select(float4(0.01f) * a_val, a_val, a_val > float4(0.0f)); break;
        case OP_SIGMOID: result = 1.0f / (1.0f + fast::exp(-a_val)); break;
        case OP_TANH: result = fast::tanh(a_val); break;
        case OP_SOFTPLUS: result = fast::log(1.0f + fast::exp(a_val)); break;
        default: result = a_val; break;
    }
    out[idx] = result;
}
"#;

    let library = device.new_library_with_source(shader_src, &CompileOptions::new())
        .map_err(|e| anyhow!("Failed to compile vec4 shader: {}", e))?;
    let kernel = library.get_function("elementwise_vec4", None)
        .map_err(|e| anyhow!("Failed to get vec4 kernel: {}", e))?;
    device.new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| anyhow!("Failed to create vec4 pipeline: {}", e))
}

#[cfg(target_os = "macos")]
fn compile_elementwise_scalar_pipeline(device: &Device) -> Result<ComputePipelineState> {
    let shader_src = r#"
#include <metal_stdlib>
using namespace metal;

constant uint OP_ADD = 0; constant uint OP_MUL = 1; constant uint OP_SUB = 2; constant uint OP_DIV = 3;
constant uint OP_SQRT = 4; constant uint OP_SIN = 5; constant uint OP_COS = 6; constant uint OP_POW = 7;
constant uint OP_ABS = 8; constant uint OP_EXP = 9; constant uint OP_LOG = 10; constant uint OP_TAN = 11;
constant uint OP_ASIN = 12; constant uint OP_ACOS = 13; constant uint OP_ATAN = 14; constant uint OP_RELU = 15;
constant uint OP_LEAKY_RELU = 16; constant uint OP_SIGMOID = 17; constant uint OP_TANH = 18; constant uint OP_SOFTPLUS = 19;

struct Params { uint size; uint op_kind; };

kernel void elementwise_scalar(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.size) return;
    float a_val = a[idx];
    float b_val = b[idx];
    float result;
    
    switch(params.op_kind) {
        case OP_ADD: result = a_val + b_val; break;
        case OP_MUL: result = a_val * b_val; break;
        case OP_SUB: result = a_val - b_val; break;
        case OP_DIV: result = a_val / b_val; break;
        case OP_SQRT: result = fast::sqrt(a_val); break;
        case OP_SIN: result = fast::sin(a_val); break;
        case OP_COS: result = fast::cos(a_val); break;
        case OP_POW: result = fast::pow(a_val, b_val); break;
        case OP_ABS: result = fast::fabs(a_val); break;
        case OP_EXP: result = fast::exp(a_val); break;
        case OP_LOG: result = fast::log(a_val); break;
        case OP_TAN: result = fast::tan(a_val); break;
        case OP_ASIN: result = asin(a_val); break;
        case OP_ACOS: result = acos(a_val); break;
        case OP_ATAN: result = atan(a_val); break;
        case OP_RELU: result = fast::max(a_val, 0.0f); break;
        case OP_LEAKY_RELU: result = (a_val > 0.0f) ? a_val : 0.01f * a_val; break;
        case OP_SIGMOID: result = 1.0f / (1.0f + fast::exp(-a_val)); break;
        case OP_TANH: result = fast::tanh(a_val); break;
        case OP_SOFTPLUS: result = fast::log(1.0f + fast::exp(a_val)); break;
        default: result = a_val; break;
    }
    out[idx] = result;
}
"#;

    let library = device.new_library_with_source(shader_src, &CompileOptions::new())
        .map_err(|e| anyhow!("Failed to compile scalar shader: {}", e))?;
    let kernel = library.get_function("elementwise_scalar", None)
        .map_err(|e| anyhow!("Failed to get scalar kernel: {}", e))?;
    device.new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| anyhow!("Failed to create scalar pipeline: {}", e))
}

#[cfg(target_os = "macos")]
fn compile_reduction_pipeline(device: &Device) -> Result<ComputePipelineState> {
    // Optimized reduction using manual threadgroup reduction
    let shader_src = r#"
#include <metal_stdlib>
using namespace metal;

constant uint WG_SIZE = 256;

struct Params {
    uint size;
};

kernel void reduction_sum(
    device const float* data [[buffer(0)]],
    device float* partials [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float shared[WG_SIZE];
    
    // Load data
    float value = 0.0f;
    if (gid < params.size) {
        value = data[gid];
    }
    shared[lid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction in threadgroup memory
    for (uint s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread writes result
    if (lid == 0) {
        partials[group_id] = shared[0];
    }
}
"#;

    let library = device.new_library_with_source(shader_src, &CompileOptions::new())
        .map_err(|e| anyhow!("Failed to compile reduction shader: {}", e))?;
    
    let kernel = library.get_function("reduction_sum", None)
        .map_err(|e| anyhow!("Failed to get reduction kernel: {}", e))?;
    
    device.new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| anyhow!("Failed to create reduction pipeline: {}", e))
}

#[cfg(target_os = "macos")]
fn get_or_compile_matmul_pipeline(device: &Device, tile_size: u32) -> Result<ComputePipelineState> {
    let shader_src = format!(r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = {tile};

struct Params {{
    uint m;
    uint n;
    uint k;
}};

kernel void matmul_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {{
    // Simple safe implementation: each thread computes one output element
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= params.m || col >= params.n) {{
        return;
    }}
    
    float sum = 0.0f;
    for (uint k = 0; k < params.k; k++) {{
        sum = fast::fma(a[row * params.k + k], b[k * params.n + col], sum);
    }}
    
    out[row * params.n + col] = sum;
}}
"#, tile = tile_size);

    let library = device.new_library_with_source(&shader_src, &CompileOptions::new())
        .map_err(|e| anyhow!("Failed to compile matmul shader: {}", e))?;
    
    let kernel = library.get_function("matmul_tiled", None)
        .map_err(|e| anyhow!("Failed to get matmul kernel: {}", e))?;
    
    device.new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| anyhow!("Failed to create matmul pipeline: {}", e))
}

/// Cached probe helper
pub fn is_available_cached() -> bool {
    static PROBE: OnceCell<bool> = OnceCell::new();
    *PROBE.get_or_init(|| MetalBackend::is_available())
}

// ============================================================================
// Public API for Dispatch System
// ============================================================================

/// Elementwise operations on Metal (public API for dispatch)
pub fn elementwise_metal(a: &Array, b: &Array, kind: crate::llo::ElementwiseKind) -> Result<Array> {
    #[cfg(target_os = "macos")]
    {
        run_elementwise_metal_optimized(a, b, kind)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (a, b, kind);
        Err(anyhow!("Metal backend only available on macOS"))
    }
}

/// Matrix multiplication on Metal (public API for dispatch)
pub fn matmul_metal(a: &Array, b: &Array) -> Result<Array> {
    #[cfg(target_os = "macos")]
    {
        run_matmul_metal_optimized(a, b)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (a, b);
        Err(anyhow!("Metal backend only available on macOS"))
    }
}

/// Reduction operations on Metal (public API for dispatch)
pub fn reduction_metal(a: &Array, axis: Option<usize>) -> Result<Array> {
    #[cfg(target_os = "macos")]
    {
        run_reduction_metal_optimized(a, axis)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (a, axis);
        Err(anyhow!("Metal backend only available on macOS"))
    }
}

// ============================================================================
// Optimized Implementation (macOS only)
// ============================================================================

#[cfg(target_os = "macos")]
fn kind_to_u32(kind: crate::llo::ElementwiseKind) -> u32 {
    use crate::llo::ElementwiseKind::*;
    match kind {
        Add => 0, Mul => 1, Sub => 2, Div => 3,
        Sqrt => 4, Sin => 5, Cos => 6, Pow => 7,
        Abs => 8, Exp => 9, Log => 10, Tan => 11,
        Asin => 12, Acos => 13, Atan => 14,
        Relu => 15, LeakyRelu => 16, Sigmoid => 17,
        Tanh => 18, Softplus => 19, Neg => 20,
    }
}

#[cfg(target_os = "macos")]
fn run_elementwise_metal_optimized(a: &Array, b: &Array, kind: crate::llo::ElementwiseKind) -> Result<Array> {
    let ctx = get_metal_device()?;
    let len = a.len();

    let command_buffer = ctx.queue.new_command_buffer();

    let a_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(a.data.as_ptr() as *const u8, a.data.len() * std::mem::size_of::<f32>())
    };
    let b_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(b.data.as_ptr() as *const u8, b.data.len() * std::mem::size_of::<f32>())
    };

    // Use buffer pool with Shared mode (simple and safe)
    let mut pool = ctx.buffer_pool.lock().unwrap();
    
    let a_buf = {
        let buf = pool.get_or_create(&ctx.device, a_bytes.len(), MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(
                a_bytes.as_ptr(),
                buf.contents() as *mut u8,
                a_bytes.len()
            );
        }
        buf
    };

    let b_buf = {
        let buf = pool.get_or_create(&ctx.device, b_bytes.len(), MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(
                b_bytes.as_ptr(),
                buf.contents() as *mut u8,
                b_bytes.len()
            );
        }
        buf
    };

    let out_buf = pool.get_or_create(
        &ctx.device,
        len * std::mem::size_of::<f32>(),
        MTLResourceOptions::StorageModeShared
    );

    drop(pool); // Release lock

    // Compute encoder - use ONLY scalar pipeline to avoid alignment issues
    let encoder = command_buffer.new_compute_command_encoder();
    let op_kind = kind_to_u32(kind);

    let params = [len as u32, op_kind];
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * std::mem::size_of::<u32>())
    };
    let params_buf = ctx.device.new_buffer_with_data(
        params_bytes.as_ptr() as *const _,
        params_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    encoder.set_compute_pipeline_state(&ctx.elementwise_scalar_pipeline);
    encoder.set_buffer(0, Some(&a_buf), 0);
    encoder.set_buffer(1, Some(&b_buf), 0);
    encoder.set_buffer(2, Some(&out_buf), 0);
    encoder.set_buffer(3, Some(&params_buf), 0);

    let thread_count = MTLSize::new(len as u64, 1, 1);
    let thread_group_size = MTLSize::new(ctx.max_threads_per_threadgroup.min(256), 1, 1);
    encoder.dispatch_threads(thread_count, thread_group_size);

    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back results
    let out_ptr = out_buf.contents() as *const f32;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, len) };
    let result = out_slice.to_vec();

    // Return buffers to pool
    let mut pool = ctx.buffer_pool.lock().unwrap();
    pool.return_buffer(a_buf, a_bytes.len());
    pool.return_buffer(b_buf, b_bytes.len());
    pool.return_buffer(out_buf, len * std::mem::size_of::<f32>());

    Ok(Array::new(a.shape.clone(), result))
}

#[cfg(target_os = "macos")]
fn run_matmul_metal_optimized(a: &Array, b: &Array) -> Result<Array> {
    let ctx = get_metal_device()?;
    
    let m = a.shape[0] as u32;
    let k = a.shape[1] as u32;
    let n = b.shape[1] as u32;
    let len = (m * n) as usize;

    // Get pipeline (tile_size is unused in current simple implementation)
    let pipeline = {
        let mut cache = ctx.matmul_pipeline_cache.lock().unwrap();
        if let Some(p) = cache.get(&(16, 0)) {
            p.clone()
        } else {
            let p = get_or_compile_matmul_pipeline(&ctx.device, 16)?;
            cache.insert((16, 0), p.clone());
            p
        }
    };

    let a_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(a.data.as_ptr() as *const u8, a.data.len() * std::mem::size_of::<f32>())
    };
    let b_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(b.data.as_ptr() as *const u8, b.data.len() * std::mem::size_of::<f32>())
    };

    // Use buffer pool
    let mut pool = ctx.buffer_pool.lock().unwrap();
    
    let a_buf = {
        let buf = pool.get_or_create(&ctx.device, a_bytes.len(), MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(
                a_bytes.as_ptr(),
                buf.contents() as *mut u8,
                a_bytes.len()
            );
        }
        buf
    };

    let b_buf = {
        let buf = pool.get_or_create(&ctx.device, b_bytes.len(), MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(
                b_bytes.as_ptr(),
                buf.contents() as *mut u8,
                b_bytes.len()
            );
        }
        buf
    };

    let out_buf = pool.get_or_create(
        &ctx.device,
        len * std::mem::size_of::<f32>(),
        MTLResourceOptions::StorageModeShared,
    );

    drop(pool); // Release lock

    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let params = [m, n, k];
    let params_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * std::mem::size_of::<u32>())
    };
    let params_buf = ctx.device.new_buffer_with_data(
        params_bytes.as_ptr() as *const _,
        params_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&a_buf), 0);
    encoder.set_buffer(1, Some(&b_buf), 0);
    encoder.set_buffer(2, Some(&out_buf), 0);
    encoder.set_buffer(3, Some(&params_buf), 0);

    // Simple 2D dispatch: one thread per output element
    let thread_group_size = MTLSize::new(16, 16, 1);
    let grid_size = MTLSize::new(n as u64, m as u64, 1);
    encoder.dispatch_threads(grid_size, thread_group_size);
    
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back results
    let out_ptr = out_buf.contents() as *const f32;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, len) };
    let result = out_slice.to_vec();

    // Return buffers to pool
    let mut pool = ctx.buffer_pool.lock().unwrap();
    pool.return_buffer(a_buf, a_bytes.len());
    pool.return_buffer(b_buf, b_bytes.len());
    pool.return_buffer(out_buf, len * std::mem::size_of::<f32>());

    Ok(Array::new(vec![m as usize, n as usize], result))
}

#[cfg(target_os = "macos")]
fn run_reduction_metal_optimized(a: &Array, axis: Option<usize>) -> Result<Array> {
    let ctx = get_metal_device()?;

    if axis.is_some() {
        return Err(anyhow!("axis-based reduction not implemented in Metal prototype"));
    }

    let size = a.len() as u32;
    if size == 0 {
        return Ok(Array::new(vec![1], vec![0.0]));
    }

    const WG_SIZE: u32 = 256;

    let data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(a.data.as_ptr() as *const u8, a.data.len() * std::mem::size_of::<f32>())
    };

    // Use buffer pool - ALL Shared mode for simplicity
    let mut pool = ctx.buffer_pool.lock().unwrap();
    
    // Input buffer with data
    let in_buf = {
        let buf = pool.get_or_create(&ctx.device, data_bytes.len(), MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data_bytes.as_ptr(),
                buf.contents() as *mut u8,
                data_bytes.len()
            );
        }
        buf
    };

    // Calculate max buffer size needed
    let mut max_groups = size;
    let mut temp_size = max_groups;
    while temp_size > 1 {
        temp_size = (temp_size + WG_SIZE - 1) / WG_SIZE;
        max_groups = max_groups.max(temp_size);
    }

    // Preallocate temp buffers (Shared mode)
    let temp_buf1 = pool.get_or_create(&ctx.device, max_groups as usize * std::mem::size_of::<f32>(), MTLResourceOptions::StorageModeShared);
    let temp_buf2 = pool.get_or_create(&ctx.device, max_groups as usize * std::mem::size_of::<f32>(), MTLResourceOptions::StorageModeShared);

    drop(pool); // Release lock

    // Single command buffer for all reduction stages
    let command_buffer = ctx.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    
    let mut current_size = size;
    let mut iteration = 0;

    loop {
        let groups = ((current_size + WG_SIZE - 1) / WG_SIZE) as u32;
        let is_final = groups == 1;

        let params = [current_size];
        let params_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * std::mem::size_of::<u32>())
        };
        let params_buf = ctx.device.new_buffer_with_data(
            params_bytes.as_ptr() as *const _,
            params_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(&ctx.reduction_pipeline);
        
        // Determine input/output buffers
        let (input_buf, output_buf) = if iteration == 0 {
            (&in_buf, &temp_buf1)
        } else if iteration % 2 == 1 {
            (&temp_buf1, &temp_buf2)
        } else {
            (&temp_buf2, &temp_buf1)
        };
        
        encoder.set_buffer(0, Some(input_buf), 0);
        encoder.set_buffer(1, Some(output_buf), 0);
        encoder.set_buffer(2, Some(&params_buf), 0);

        let thread_count = MTLSize::new((groups * WG_SIZE) as u64, 1, 1);
        let thread_group_size = MTLSize::new(WG_SIZE as u64, 1, 1);
        encoder.dispatch_threads(thread_count, thread_group_size);

        if is_final {
            encoder.end_encoding();
            break;
        }

        iteration += 1;
        current_size = groups;
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read final result directly from output buffer (it's Shared mode)
    let final_buf = if iteration == 0 {
        &temp_buf1
    } else if iteration % 2 == 1 {
        &temp_buf2
    } else {
        &temp_buf1
    };
    let out_ptr = final_buf.contents() as *const f32;
    let final_value = unsafe { *out_ptr };

    // Return buffers to pool
    let mut pool = ctx.buffer_pool.lock().unwrap();
    pool.return_buffer(in_buf, data_bytes.len());
    pool.return_buffer(temp_buf1, max_groups as usize * std::mem::size_of::<f32>());
    pool.return_buffer(temp_buf2, max_groups as usize * std::mem::size_of::<f32>());

    Ok(Array::new(vec![1], vec![final_value]))
}
