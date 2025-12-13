//! WebGPU backend
//!
//! Cross-platform GPU acceleration using wgpu.
//! Works on both native (Vulkan/Metal/DX12) and WASM (WebGPU/WebGL).

pub mod batchnorm;
pub mod codegen;
pub mod conv;
pub mod dropout;

use crate::array::Array;
use anyhow::{anyhow, Result};
use std::borrow::Cow;

// Para native: usamos OnceCell + Mutex para cache thread-safe
#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

// Para WASM: usamos thread_local! que no requiere Sync
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;

// Global flag for WASM WebGPU availability (set from JS binding)
#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_arch = "wasm32")]
static WEBGPU_AVAILABLE_FROM_JS: AtomicBool = AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
pub fn set_webgpu_available_wasm(available: bool) {
    WEBGPU_AVAILABLE_FROM_JS.store(available, Ordering::SeqCst);
    eprintln!("[numrs-webgpu] WebGPU available flag set to: {}", available);
}

#[cfg(target_arch = "wasm32")]
pub fn get_webgpu_available_wasm() -> bool {
    WEBGPU_AVAILABLE_FROM_JS.load(Ordering::SeqCst)
}

#[derive(Debug, Clone)]
pub struct WebGpuBackend {}

impl WebGpuBackend {
    pub fn new() -> Self {
        Self {}
    }

    /// Quick probe helper: is there a usable WebGPU adapter available?
    pub fn is_available() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            get_webgpu_available_wasm()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }));

            adapter.is_some()
        }
    }
}

// Cached GPU context to avoid re-creating adapter/device/pipelines every call.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_bgl: wgpu::BindGroupLayout,
}

use std::sync::Arc;

struct DeviceQueue {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[cfg(target_arch = "wasm32")]
pub async fn init_webgpu_wasm() -> Result<()> {
    // 1. Check if already initialized to likely avoid re-initialization overhead
    // (Though wgpu might handle this, it's safer to check our cache)
    let already_init = GPU_DEVICE.with(|cell| cell.borrow().is_some());
    if already_init {
        return Ok(());
    }

    // 2. Request adapter and device asynchronously (safe in browser main thread)
    // No debug logs needed anymore
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow!("no WebGPU adapter available"))?;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .map_err(|e| anyhow!("request device failed: {:?}", e))?;

    // 3. Store in thread-local cache
    GPU_DEVICE.with(|cell| {
        *cell.borrow_mut() = Some(Ok(DeviceQueue {
            device: Arc::new(device),
            queue: Arc::new(queue),
        }));
    });

    set_webgpu_available_wasm(true);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn get_gpu_device() -> Result<DeviceQueue> {
    GPU_DEVICE.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(Ok(dq)) => {
                // Return checked clone (Arc clone is cheap)
                Ok(DeviceQueue {
                    device: dq.device.clone(),
                    queue: dq.queue.clone(),
                })
            }
            Some(Err(e)) => Err(anyhow!("WebGPU init failed previously: {:?}", e)),
            None => Err(anyhow!(
                "WebGPU/WebGL not initialized. Ensure init_webgpu() is called."
            )),
        }
    })
}

// ============================================================================
// GPU Device Cache - Native vs WASM
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
static GPU_DEVICE: OnceCell<Result<DeviceQueue, anyhow::Error>> = OnceCell::new();

#[cfg(target_arch = "wasm32")]
thread_local! {
    static GPU_DEVICE: RefCell<Option<Result<DeviceQueue, anyhow::Error>>> = RefCell::new(None);
}

#[cfg(not(target_arch = "wasm32"))]
fn get_gpu_device() -> Result<&'static DeviceQueue> {
    GPU_DEVICE.get_or_init(|| {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("no WebGPU adapter available"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(), // Minimal features
                required_limits: wgpu::Limits::default(),   // Default limits
                memory_hints: Default::default(),
            },
            None,
        ))
        .map_err(|e| anyhow!("request device failed: {:?}", e))?;

        Ok(DeviceQueue {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    });

    let init_ref = GPU_DEVICE.get().expect("gpu device was just initialized");
    match init_ref {
        Ok(dq) => Ok(dq),
        Err(e) => Err(anyhow!("gpu init failed: {:?}", e)),
    }
}

// Helper macro: ejecuta código con referencias a device/queue sin importar el target
// En native devuelve &'static, en WASM devuelve owned pero podemos tomar prestado
#[cfg(not(target_arch = "wasm32"))]
macro_rules! with_gpu_device {
    ($dq:ident, $code:expr) => {{
        let $dq = get_gpu_device()?;
        $code
    }};
}

#[cfg(target_arch = "wasm32")]
macro_rules! with_gpu_device {
    ($dq:ident, $code:expr) => {{
        let $dq = get_gpu_device()?;
        $code
    }};
}

// Cached reduction pipeline (pipeline + bind group layout) - solo en native
#[cfg(not(target_arch = "wasm32"))]
static REDUCTION_PIPELINE: OnceCell<
    Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), anyhow::Error>,
> = OnceCell::new();

// get_gpu_context solo se usa en fast path (deshabilitado en WASM)
#[cfg(not(target_arch = "wasm32"))]
fn get_gpu_context(shader_src: &str) -> Result<&'static GpuContext> {
    static CTX: OnceCell<Result<GpuContext, anyhow::Error>> = OnceCell::new();

    CTX.get_or_init(|| -> Result<GpuContext, anyhow::Error> {
        // Create instance, adapter, device and queue
        eprintln!("NumRs-Core: Starting WebGPU Init...");

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        eprintln!("NumRs-Core: Instance created. Requesting adapter...");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
        }));

        if adapter.is_none() {
            eprintln!("NumRs-Core: Adapter request returned None!");
            return Err(anyhow!("no WebGPU adapter available"));
        }
        let adapter = adapter.unwrap();

        eprintln!("NumRs-Core: Adapter found. Requesting device...");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        ))
        .map_err(|e| anyhow!("request device failed: {:?}", e))?;

        // create shader module and pipeline for matmul
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_src.to_string())),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_matmul"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_matmul"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline_matmul"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        Ok(GpuContext {
            device,
            queue,
            matmul_pipeline: compute_pipeline,
            matmul_bgl: bgl,
        })
    });

    let init_ref = CTX.get().expect("OnceCell was just initialized");
    match init_ref {
        Ok(ctx) => Ok(ctx),
        Err(e) => Err(anyhow!("gpu init failed: {:?}", e)),
    }
}

// Cached buffers to avoid re-allocating buffers across repeated matmul calls.
#[cfg(not(target_arch = "wasm32"))]
struct CachedBuffers {
    m: u32,
    n: u32,
    k: u32,
    _len: usize,
    a_buf: wgpu::Buffer,
    b_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
}

#[cfg(not(target_arch = "wasm32"))]
static BUFFERS: OnceCell<Mutex<Option<CachedBuffers>>> = OnceCell::new();

/// Cached probe helper that performs a single adapter check and caches the result.
#[cfg(not(target_arch = "wasm32"))]
pub fn is_available_cached() -> bool {
    static PROBE: OnceCell<bool> = OnceCell::new();
    *PROBE.get_or_init(|| WebGpuBackend::is_available())
}

// En WASM: implementación simplificada sin cache
#[cfg(target_arch = "wasm32")]
pub fn is_available_cached() -> bool {
    // Check the global flag set by JavaScript
    get_webgpu_available_wasm()
}

fn run_elementwise_gpu(a: &Array, b: &Array, kind: crate::llo::ElementwiseKind) -> Result<Array> {
    use wgpu::util::DeviceExt;

    let len = a.len();

    // Generate WGSL depending on the kind
    let op = match kind {
        crate::llo::ElementwiseKind::Add => "a[idx] + b[idx]",
        crate::llo::ElementwiseKind::Mul => "a[idx] * b[idx]",
        crate::llo::ElementwiseKind::Sub => "a[idx] - b[idx]",
        crate::llo::ElementwiseKind::Div => "a[idx] / b[idx]",
        crate::llo::ElementwiseKind::Sqrt => "sqrt(a[idx])",
        crate::llo::ElementwiseKind::Sin => "sin(a[idx])",
        crate::llo::ElementwiseKind::Cos => "cos(a[idx])",
        crate::llo::ElementwiseKind::Pow => "pow(a[idx], b[idx])",
        crate::llo::ElementwiseKind::Abs => "abs(a[idx])",
        crate::llo::ElementwiseKind::Neg => "-a[idx]",
        crate::llo::ElementwiseKind::Exp => "exp(a[idx])",
        crate::llo::ElementwiseKind::Log => "log(a[idx])",
        crate::llo::ElementwiseKind::Tan => "tan(a[idx])",
        crate::llo::ElementwiseKind::Asin => "asin(a[idx])",
        crate::llo::ElementwiseKind::Acos => "acos(a[idx])",
        crate::llo::ElementwiseKind::Atan => "atan(a[idx])",
        crate::llo::ElementwiseKind::Relu => "max(a[idx], 0.0)",
        crate::llo::ElementwiseKind::LeakyRelu => "select(0.01 * a[idx], a[idx], a[idx] > 0.0)",
        crate::llo::ElementwiseKind::Sigmoid => "1.0 / (1.0 + exp(-a[idx]))",
        crate::llo::ElementwiseKind::Tanh => "tanh(a[idx])",
        crate::llo::ElementwiseKind::Softplus => "log(1.0 + exp(a[idx]))",
    };

    let shader = format!(
        r#"
    struct Params {{ size: u32, }};
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx: u32 = gid.x;
    if (idx >= params.size) {{ return; }}
    out[idx] = {op};
}}
"#
    );

    // Create instance, adapter, device and queue
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow!("no WebGPU adapter available"))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .map_err(|e| anyhow!("request device failed: {:?}", e))?;

    // Create buffers
    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a_buf"),
        contents: bytemuck::cast_slice(&a.data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b_buf"),
        contents: bytemuck::cast_slice(&b.data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out_buf"),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = [len as u32];
    let params_bytes = bytemuck::cast_slice(&params);
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: params_bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // staging buffer for readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (len * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("elementwise_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader)),
    });

    // Bind group layout and pipeline
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ce") });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cp"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let workgroups = ((len as u32) + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    // copy to staging
    encoder.copy_buffer_to_buffer(
        &out_buf,
        0,
        &staging,
        0,
        (len * std::mem::size_of::<f32>()) as u64,
    );

    queue.submit(Some(encoder.finish()));

    // map staging and read back
    let buffer_slice = staging.slice(..);
    // map_async requires a callback for non-async contexts — we'll use a
    // synchronous channel to wait for completion.
    use std::sync::mpsc::channel;
    let (tx, rx) = channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    let ok = rx
        .recv()
        .map_err(|_| anyhow!("map callback channel error"))?;
    ok.map_err(|e| anyhow!("map async failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let mut out_vec = Vec::with_capacity(len);
    // convert bytes -> f32
    for chunk in data.chunks_exact(4) {
        let b = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out_vec.push(f32::from_bits(u32::from_le_bytes(b)));
    }

    drop(data);
    staging.unmap();

    Ok(Array::new(a.shape.clone(), out_vec))
}

// Fast path: original optimized tiled matmul that uses monolithic buffers and a
// WGSL kernel with workgroup/shared memory. This is high-performance but requires
// that the device supports buffers large enough for the full A/B/output tiles.
#[cfg(not(target_arch = "wasm32"))]
fn run_matmul_gpu_fast(a: &Array, b: &Array) -> Result<Array> {
    use wgpu::util::DeviceExt;
    let m = a.shape[0] as u32;
    let k = a.shape[1] as u32;
    let n = b.shape[1] as u32;
    let len = (m as usize) * (n as usize);

    // Optimized tiled WGSL matmul using workgroup memory.
    // TILE = 32x32 (increased from 16x16 for better occupancy)
    // Each thread computes a 4x4 sub-block using vec4 operations for maximum throughput
    // Total: 8x8 threads = 64 threads per workgroup processing 32x32 outputs
    let shader = format!(
        r#"
struct Params {{ m: u32, n: u32, k: u32, }};
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
var<workgroup> tileA: array<f32, 1024>;  // 32x32 tile
var<workgroup> tileB: array<f32, 1024>;  // 32x32 tile

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) _gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let row_base: u32 = wid.y * TILE;
    let col_base: u32 = wid.x * TILE;

    // Each thread computes a 4x4 block starting at these local indices
    let local_r0: u32 = lid.y * 4u;
    let local_c0: u32 = lid.x * 4u;

    // Accumulator registers for 4x4 output block (16 values)
    var sum00: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sum10: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sum20: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sum30: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    var k0: u32 = 0u;
    loop {{
        if (k0 >= params.k) {{ break; }}

        // Cooperative loading: each thread loads 4x4 elements into shared memory
        // This ensures coalesced memory access and full utilization
        for (var dy: u32 = 0u; dy < 4u; dy = dy + 1u) {{
            for (var dx: u32 = 0u; dx < 4u; dx = dx + 1u) {{
                let lr = local_r0 + dy;
                let lc = local_c0 + dx;
                
                // Load A tile
                let a_row = row_base + lr;
                let a_col = k0 + lc;
                if (a_row < params.m && a_col < params.k) {{
                    tileA[lr * TILE + lc] = a[a_row * params.k + a_col];
                }} else {{
                    tileA[lr * TILE + lc] = 0.0;
                }}

                // Load B tile
                let b_row = k0 + lr;
                let b_col = col_base + lc;
                if (b_row < params.k && b_col < params.n) {{
                    tileB[lr * TILE + lc] = b[b_row * params.n + b_col];
                }} else {{
                    tileB[lr * TILE + lc] = 0.0;
                }}
            }}
        }}

        workgroupBarrier();

        // Inner loop: matrix multiply within tile using vec4 operations
        // Process 4 k-elements at a time for better arithmetic throughput
        var t: u32 = 0u;
        loop {{
            if (t + 4u > TILE) {{ break; }}

            // Load A vectors for 4 output rows
            let a0 = vec4<f32>(
                tileA[local_r0 * TILE + t],
                tileA[local_r0 * TILE + t + 1u],
                tileA[local_r0 * TILE + t + 2u],
                tileA[local_r0 * TILE + t + 3u]
            );
            let a1 = vec4<f32>(
                tileA[(local_r0 + 1u) * TILE + t],
                tileA[(local_r0 + 1u) * TILE + t + 1u],
                tileA[(local_r0 + 1u) * TILE + t + 2u],
                tileA[(local_r0 + 1u) * TILE + t + 3u]
            );
            let a2 = vec4<f32>(
                tileA[(local_r0 + 2u) * TILE + t],
                tileA[(local_r0 + 2u) * TILE + t + 1u],
                tileA[(local_r0 + 2u) * TILE + t + 2u],
                tileA[(local_r0 + 2u) * TILE + t + 3u]
            );
            let a3 = vec4<f32>(
                tileA[(local_r0 + 3u) * TILE + t],
                tileA[(local_r0 + 3u) * TILE + t + 1u],
                tileA[(local_r0 + 3u) * TILE + t + 2u],
                tileA[(local_r0 + 3u) * TILE + t + 3u]
            );

            // Load B vectors for 4 output columns (transposed access pattern)
            let b0 = vec4<f32>(
                tileB[t * TILE + local_c0],
                tileB[(t + 1u) * TILE + local_c0],
                tileB[(t + 2u) * TILE + local_c0],
                tileB[(t + 3u) * TILE + local_c0]
            );
            let b1 = vec4<f32>(
                tileB[t * TILE + local_c0 + 1u],
                tileB[(t + 1u) * TILE + local_c0 + 1u],
                tileB[(t + 2u) * TILE + local_c0 + 1u],
                tileB[(t + 3u) * TILE + local_c0 + 1u]
            );
            let b2 = vec4<f32>(
                tileB[t * TILE + local_c0 + 2u],
                tileB[(t + 1u) * TILE + local_c0 + 2u],
                tileB[(t + 2u) * TILE + local_c0 + 2u],
                tileB[(t + 3u) * TILE + local_c0 + 2u]
            );
            let b3 = vec4<f32>(
                tileB[t * TILE + local_c0 + 3u],
                tileB[(t + 1u) * TILE + local_c0 + 3u],
                tileB[(t + 2u) * TILE + local_c0 + 3u],
                tileB[(t + 3u) * TILE + local_c0 + 3u]
            );

            // Compute 4x4 block using dot products (16 FMA operations)
            sum00 = sum00 + vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
            sum10 = sum10 + vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
            sum20 = sum20 + vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
            sum30 = sum30 + vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));

            t = t + 4u;
        }}

        workgroupBarrier();
        k0 = k0 + TILE;
    }}

    // Write 4x4 block results back to output (with bounds checking)
    for (var row_off: u32 = 0u; row_off < 4u; row_off = row_off + 1u) {{
        let global_row = row_base + local_r0 + row_off;
        if (global_row >= params.m) {{ continue; }}
        
        let result_vec = select(
            sum00,
            select(sum10, select(sum20, sum30, row_off == 3u), row_off == 2u),
            row_off == 1u
        );
        
        for (var col_off: u32 = 0u; col_off < 4u; col_off = col_off + 1u) {{
            let global_col = col_base + local_c0 + col_off;
            if (global_col < params.n) {{
                out[global_row * params.n + global_col] = result_vec[col_off];
            }}
        }}
    }}
}}
"#
    );

    // Use cached GPU context (device/queue/pipeline/layout) to avoid setup overhead
    let ctx = get_gpu_context(&shader)?;

    let device = &ctx.device;
    let queue = &ctx.queue;

    // Prepare or reuse GPU buffers for this shape
    let buf_mutex = BUFFERS.get_or_init(|| Mutex::new(None));
    let mut guard = buf_mutex.lock().unwrap();

    if guard
        .as_ref()
        .map(|c| c.m != m || c.n != n || c.k != k)
        .unwrap_or(true)
    {
        // allocate fresh buffers sized for this matmul
        let a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("a_buf"),
            size: ((m as usize * k as usize) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("b_buf"),
            size: ((k as usize * n as usize) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out_buf"),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [m, n, k];
        let params_bytes = bytemuck::cast_slice(&params);
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        *guard = Some(CachedBuffers {
            m,
            n,
            k,
            _len: len,
            a_buf,
            b_buf,
            out_buf,
            params_buf,
            staging,
        });
    }

    let cached = guard.as_ref().unwrap();

    // upload input matrices using queue.write_buffer (avoids buffer re-allocation)
    // a.data and b.data are f32 slices
    let a_bytes = bytemuck::cast_slice(&a.data);
    let b_bytes = bytemuck::cast_slice(&b.data);
    queue.write_buffer(&cached.a_buf, 0, a_bytes);
    queue.write_buffer(&cached.b_buf, 0, b_bytes);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg_matmul"),
        layout: &ctx.matmul_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cached.a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cached.b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: cached.out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cached.params_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ce_matmul"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cp_matmul"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.matmul_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        // Updated for 32x32 tiles (increased from 16x16)
        let wg_x = ((n + 31) / 32) as u32;
        let wg_y = ((m + 31) / 32) as u32;
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    encoder.copy_buffer_to_buffer(
        &cached.out_buf,
        0,
        &cached.staging,
        0,
        (len * std::mem::size_of::<f32>()) as u64,
    );

    queue.submit(Some(encoder.finish()));

    // read back
    let buffer_slice = cached.staging.slice(..);
    use std::sync::mpsc::channel;
    let (tx, rx) = channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    let ok = rx
        .recv()
        .map_err(|_| anyhow!("map callback channel error"))?;
    ok.map_err(|e| anyhow!("map async failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let mut out_vec = Vec::with_capacity(len);
    for chunk in data.chunks_exact(4) {
        let b = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out_vec.push(f32::from_bits(u32::from_le_bytes(b)));
    }
    drop(data);
    cached.staging.unmap();

    Ok(Array::new(vec![m as usize, n as usize], out_vec))
}

// ============================================================================
// Public API for Dispatch System
// ============================================================================

/// Elementwise operations on WebGPU (public API for dispatch)
pub fn elementwise_webgpu(
    a: &Array,
    b: &Array,
    kind: crate::llo::ElementwiseKind,
) -> Result<Array> {
    run_elementwise_gpu(a, b, kind)
}

/// Matrix multiplication on WebGPU (public API for dispatch)
pub fn matmul_webgpu(a: &Array, b: &Array) -> Array {
    run_matmul_gpu(a, b).expect("WebGPU matmul failed")
}

/// Reduction operations on WebGPU (public API for dispatch)
pub fn reduction_webgpu(a: &Array, axis: Option<usize>) -> Result<Array> {
    run_reduction_gpu(a, axis)
}

/// Broadcast operation on WebGPU (public API for dispatch)
pub fn broadcast_to_webgpu(a: &Array, target_shape: &[usize]) -> Result<Array> {
    run_broadcast_gpu(a, target_shape)
}

// ============================================================================
// Internal Implementation
// ============================================================================

// Streaming path: for very large matrices that would require buffers exceeding the
// device limits, process tiles on the host and stream smaller A/B tiles to GPU.
fn run_matmul_gpu_streaming(a: &Array, b: &Array) -> Result<Array> {
    use std::cmp::min;
    use wgpu::util::DeviceExt;

    let m = a.shape[0] as usize;
    let k = a.shape[1] as usize;
    let n = b.shape[1] as usize;

    // simple per-tile kernel: computes a tm x tn partial for given tk
    let tile_shader = r#"
struct Params { tm: u32, tn: u32, tk: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.tm || col >= params.tn) { return; }
    var sum: f32 = 0.0;
    var kk: u32 = 0u;
    loop {
        if (kk >= params.tk) { break; }
        sum = sum + a[row * params.tk + kk] * b[kk * params.tn + col];
        kk = kk + 1u;
    }
    let idx = row * params.tn + col;
    out[idx] = out[idx] + sum;
}
"#;

    with_gpu_device!(dq, {
        let device = &dq.device;
        let queue = &dq.queue;

        let max_buf_bytes = device.limits().max_storage_buffer_binding_size as usize;
        let max_elems = max_buf_bytes / std::mem::size_of::<f32>();

        let prefer_tile = 1024usize;
        let tile_k = std::cmp::min(
            k,
            std::cmp::min(
                prefer_tile,
                std::cmp::max(1, max_elems / std::cmp::max(1, std::cmp::max(m, n))),
            ),
        );
        let tile_m = std::cmp::min(
            m,
            std::cmp::max(
                64,
                std::cmp::min(
                    prefer_tile,
                    std::cmp::max(1, max_elems / std::cmp::max(1, k)),
                ),
            ),
        );
        let tile_n = tile_m;

        // create module/pipeline once
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_matmul"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(tile_shader.to_string())),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_tile_matmul"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl_tile_matmul"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline_tile_matmul"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let mut result = vec![0.0f32; m * n];

        for i in (0..m).step_by(tile_m) {
            let tm = min(tile_m, m - i);
            for j in (0..n).step_by(tile_n) {
                let tn = min(tile_n, n - j);

                let out_size = (tm * tn) * std::mem::size_of::<f32>();
                let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("out_tile"),
                    size: out_size as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let zeros = vec![0u8; out_size];
                queue.write_buffer(&out_buf, 0, &zeros);

                let mut p = 0usize;
                while p < k {
                    let tk = min(tile_k, k - p);

                    let mut a_tile = Vec::with_capacity(tm * tk);
                    for ii in 0..tm {
                        let row = i + ii;
                        let src_off = row * k + p;
                        a_tile.extend_from_slice(&a.data[src_off..src_off + tk]);
                    }

                    let mut b_tile = Vec::with_capacity(tk * tn);
                    for kk in 0..tk {
                        let src_off = (p + kk) * n + j;
                        b_tile.extend_from_slice(&b.data[src_off..src_off + tn]);
                    }

                    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("a_tile"),
                        contents: bytemuck::cast_slice(&a_tile),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("b_tile"),
                        contents: bytemuck::cast_slice(&b_tile),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                    let params = [tm as u32, tn as u32, tk as u32];
                    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("params_tile"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("bg_tile"),
                        layout: &bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: a_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: b_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: out_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: params_buf.as_entire_binding(),
                            },
                        ],
                    });

                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("ce_tile"),
                        });
                    {
                        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("cp_tile"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(&compute_pipeline);
                        cpass.set_bind_group(0, &bind_group, &[]);
                        let wg_x = ((tn as u32) + 15) / 16;
                        let wg_y = ((tm as u32) + 15) / 16;
                        cpass.dispatch_workgroups(wg_x, wg_y, 1);
                    }

                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::Maintain::Wait);

                    p += tk;
                }

                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging_out_tile"),
                    size: out_size as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ce_copy_out"),
                });
                encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size as u64);
                queue.submit(Some(encoder.finish()));

                let buffer_slice = staging.slice(..);
                use std::sync::mpsc::channel;
                let (tx, rx) = channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                device.poll(wgpu::Maintain::Wait);
                let ok = rx
                    .recv()
                    .map_err(|_| anyhow!("map callback channel error"))?;
                ok.map_err(|e| anyhow!("map async failed: {:?}", e))?;
                let data = buffer_slice.get_mapped_range();

                let mut idx = 0usize;
                for rr in 0..tm {
                    let dest_off = (i + rr) * n + j;
                    let row_bytes = &data[idx * 4..(idx + tn) * 4];
                    for cc in 0..tn {
                        let b0 = row_bytes[cc * 4..cc * 4 + 4].try_into().unwrap();
                        result[dest_off + cc] = f32::from_bits(u32::from_le_bytes(b0));
                    }
                    idx += tn;
                }
                drop(data);
                staging.unmap();
            }
        }

        Ok(Array::new(vec![m, n], result))
    })
}

// Dispatcher: choose fast path when buffers fit, otherwise fall back to streaming.
fn run_matmul_gpu(a: &Array, b: &Array) -> Result<Array> {
    // En WASM: siempre usar streaming (evita complicaciones con static caches)
    #[cfg(target_arch = "wasm32")]
    {
        return run_matmul_gpu_streaming(a, b);
    }

    // En native: usar fast path si los buffers caben
    #[cfg(not(target_arch = "wasm32"))]
    {
        with_gpu_device!(dq, {
            let device = &dq.device;

            let m = a.shape[0] as usize;
            let k = a.shape[1] as usize;
            let n = b.shape[1] as usize;

            let bytes_a = m * k * std::mem::size_of::<f32>();
            let bytes_b = k * n * std::mem::size_of::<f32>();
            let bytes_out = m * n * std::mem::size_of::<f32>();

            let max = device.limits().max_storage_buffer_binding_size as usize;

            if bytes_a <= max && bytes_b <= max && bytes_out <= max {
                run_matmul_gpu_fast(a, b)
            } else {
                run_matmul_gpu_streaming(a, b)
            }
        })
    }
}

fn run_reduction_gpu(a: &Array, axis: Option<usize>) -> Result<Array> {
    use wgpu::util::DeviceExt;

    if axis.is_some() {
        return Err(anyhow!(
            "axis-based reduction not implemented in GPU prototype"
        ));
    }

    let size = a.len() as u32;
    if size == 0 {
        return Ok(Array::new(vec![1], vec![0.0]));
    }

    // WGSL: each workgroup reduces up to WORKGROUP_SIZE elements into one partial
    const WG_SIZE: u32 = 256u32;

    // simple WGSL kernel: each invocation loads one element (if in range), does local reduction into workgroup memory, thread 0 writes partial
    let shader = format!(
        r#"
struct Params {{ size: u32, }};
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> sdata: array<f32, {wg}>;

@compute @workgroup_size({wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let local = lid.x;
    let group = wid.x;
    let idx = group * {wg}u + local;
    var v: f32 = 0.0;
    if (idx < params.size) {{ v = data[idx]; }}
    sdata[local] = v;
    workgroupBarrier();

    var stride: u32 = {wg}u / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (local < stride) {{
            sdata[local] = sdata[local] + sdata[local + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (local == 0u) {{
        partials[group] = sdata[0];
    }}
}}
"#,
        wg = WG_SIZE
    );

    // get cached device/queue
    with_gpu_device!(dq, {
        let device = &dq.device;
        let queue = &dq.queue;

        // En native: cachear pipeline para evitar reconstrucción
        #[cfg(not(target_arch = "wasm32"))]
        let pipe_res = REDUCTION_PIPELINE.get_or_init(
            || -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), anyhow::Error> {
                let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("reduction_shader"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader.clone())),
                });

                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("bgl_reduction"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("pl_reduction"),
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    });

                let compute_pipeline =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("pipeline_reduction"),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: Some("main"),
                        cache: None,
                        compilation_options: Default::default(),
                    });

                Ok((compute_pipeline, bgl))
            },
        );

        #[cfg(not(target_arch = "wasm32"))]
        let (compute_pipeline, bgl) = pipe_res
            .as_ref()
            .map_err(|e| anyhow!("reduction pipeline init failed: {:?}", e))?;

        // En WASM: crear fresh cada vez (sin cache estático)
        #[cfg(target_arch = "wasm32")]
        let (compute_pipeline, bgl) = {
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reduction_shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader.clone())),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bgl_reduction"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pl_reduction"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let compute_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("pipeline_reduction"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

            (compute_pipeline, bgl)
        };

        // iterative GPU reduction: feed input buffer, run reduction producing partials, then repeat on partials until single value
        let mut current_size = size as u32;
        let mut in_buf = {
            let data_bytes = bytemuck::cast_slice(&a.data);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("reduce_data"),
                size: data_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, data_bytes);
            buf
        };

        // temporary staging used only for final scalar readback
        let final_value: f32;

        loop {
            let groups = ((current_size + WG_SIZE - 1) / WG_SIZE) as u32;

            let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("partials"),
                size: (groups as usize * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let params = [current_size];
            let params_bytes = bytemuck::cast_slice(&params);
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params_reduction"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_reduction"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                ],
            });

            // dispatch
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ce_reduction_iter"),
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cp_reduction_iter"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(groups, 1, 1);
            }

            // if this is final (groups == 1) copy to staging and read back
            if groups == 1 {
                let staging = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("staging_final"),
                    size: std::mem::size_of::<f32>() as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(
                    &out_buf,
                    0,
                    &staging,
                    0,
                    std::mem::size_of::<f32>() as u64,
                );
                queue.submit(Some(encoder.finish()));

                // read back
                let buffer_slice = staging.slice(..);
                use std::sync::mpsc::channel;
                let (tx, rx) = channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx.send(r);
                });
                device.poll(wgpu::Maintain::Wait);
                let ok = rx
                    .recv()
                    .map_err(|_| anyhow!("map callback channel error"))?;
                ok.map_err(|e| anyhow!("map async failed: {:?}", e))?;

                let data = buffer_slice.get_mapped_range();
                let b = [data[0], data[1], data[2], data[3]];
                final_value = f32::from_bits(u32::from_le_bytes(b));
                drop(data);
                staging.unmap();
                break;
            } else {
                // prepare for next iteration: set in_buf = out_buf and continue
                queue.submit(Some(encoder.finish()));
                in_buf = out_buf;
                current_size = groups;
                // loop
            }
        }

        let total = final_value;
        Ok(Array::new(vec![1], vec![total]))
    })
}

/// GPU implementation of broadcast_to using WebGPU compute shader
fn run_broadcast_gpu(a: &Array, target_shape: &[usize]) -> Result<Array> {
    use wgpu::util::DeviceExt;

    let src_ndim = a.shape.len() as u32;
    let target_ndim = target_shape.len() as u32;
    let target_size: usize = target_shape.iter().product();

    if target_size == 0 {
        return Ok(Array::new(target_shape.to_vec(), vec![]));
    }

    // Preparar shapes y strides para el shader
    let mut src_shape_padded = vec![1u32; 4];
    let mut target_shape_padded = vec![1u32; 4];
    let mut src_strides = vec![0u32; 4];

    // Copiar shapes (alineados a la derecha)
    for i in 0..src_ndim.min(4) as usize {
        src_shape_padded[4 - src_ndim as usize + i] = a.shape[i] as u32;
    }
    for i in 0..target_ndim.min(4) as usize {
        target_shape_padded[4 - target_ndim as usize + i] = target_shape[i] as u32;
    }

    // Calcular strides para source
    let mut stride = 1u32;
    for i in (0..src_ndim as usize).rev() {
        let idx = 4 - src_ndim as usize + i;
        src_strides[idx] = stride;
        stride *= a.shape[i] as u32;
    }

    // Shader de broadcast
    let shader_code = format!(
        r#"
struct Params {{
    src_shape: vec4<u32>,
    target_shape: vec4<u32>,
    src_strides: vec4<u32>,
    src_ndim: u32,
    target_ndim: u32,
}};

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= {target_size}u) {{ return; }}
    
    // Convertir índice plano a multi-índice en target
    var target_idx: vec4<u32> = vec4<u32>(0u);
    var remaining = idx;
    
    for (var i = 0u; i < 4u; i++) {{
        let dim_idx = 3u - i;
        if (dim_idx >= (4u - params.target_ndim)) {{
            target_idx[dim_idx] = remaining % params.target_shape[dim_idx];
            remaining = remaining / params.target_shape[dim_idx];
        }}
    }}
    
    // Mapear a índice en source (con broadcasting)
    var src_flat_idx = 0u;
    let src_start_dim = 4u - params.src_ndim;
    let target_start_dim = 4u - params.target_ndim;
    
    for (var i = 0u; i < params.src_ndim; i++) {{
        let src_dim_idx = src_start_dim + i;
        let target_dim_idx = target_start_dim + i;
        let src_dim = params.src_shape[src_dim_idx];
        
        // Si la dimensión es 1, usar índice 0 (broadcasting)
        var idx_val: u32;
        if (src_dim == 1u) {{
            idx_val = 0u;
        }} else {{
            idx_val = target_idx[target_dim_idx];
        }}
        
        src_flat_idx += idx_val * params.src_strides[src_dim_idx];
    }}
    
    dst[idx] = src[src_flat_idx];
}}
"#,
        target_size = target_size
    );

    // Crear instancia, adapter y device
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow!("no WebGPU adapter available"))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("broadcast device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .map_err(|e| anyhow!("request device failed: {:?}", e))?;

    // Crear buffers
    let src_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("broadcast src"),
        contents: bytemuck::cast_slice(&a.data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("broadcast dst"),
        size: (target_size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Crear uniform buffer con parámetros
    let params_data = [
        src_shape_padded[0],
        src_shape_padded[1],
        src_shape_padded[2],
        src_shape_padded[3],
        target_shape_padded[0],
        target_shape_padded[1],
        target_shape_padded[2],
        target_shape_padded[3],
        src_strides[0],
        src_strides[1],
        src_strides[2],
        src_strides[3],
        src_ndim,
        target_ndim,
        0,
        0, // padding
    ];

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("broadcast params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Compilar shader
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("broadcast shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_code)),
    });

    // Crear pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("broadcast bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("broadcast pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("broadcast pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("broadcast bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Ejecutar compute pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("broadcast encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("broadcast pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (target_size + 255) / 256;
        cpass.dispatch_workgroups(workgroups as u32, 1, 1);
    }

    // Copiar resultado a staging buffer
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("broadcast staging"),
        size: (target_size * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&dst_buffer, 0, &staging, 0, (target_size * 4) as u64);
    queue.submit(Some(encoder.finish()));

    // Leer resultado
    let buffer_slice = staging.slice(..);
    use std::sync::mpsc::channel;
    let (tx, rx) = channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    let ok = rx.recv().map_err(|_| anyhow!("map callback error"))?;
    ok.map_err(|e| anyhow!("map async failed: {:?}", e))?;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    Ok(Array::new(target_shape.to_vec(), result))
}
