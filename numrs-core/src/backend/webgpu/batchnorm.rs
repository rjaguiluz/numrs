
use crate::array::Array;
use anyhow::{Result, anyhow};
use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::backend::webgpu::get_gpu_device;

pub fn batch_norm_1d_training_webgpu(
    _input: &Array,
    _running_mean: &mut Array,
    _running_var: &mut Array,
    _weight: &Array,
    _bias: &Array,
    _momentum: f32,
    _eps: f32,
) -> Result<Array> {
    // Training requires reduction (mean/var calc) which is complex in WGSL.
    // Fallback to CPU/SIMD for training.
    Err(anyhow!("WebGPU BatchNorm Training not yet implemented"))
}

pub fn batch_norm_1d_inference_webgpu(
    input: &Array,
    running_mean: &Array,
    running_var: &Array,
    weight: &Array,
    bias: &Array,
    eps: f32,
) -> Result<Array> {
    // Shapes
    // Input: [Batch, Channels, Length] or [Batch, Channels]
    // We treat everything as [Batch, Channels, Spatial]
    // If dims=2 [N, C], Spatial=1.
    // If dims=3 [N, C, L], Spatial=L.
    
    let _batch_size = input.shape[0];
    let channels = input.shape[1];
    let spatial = if input.shape.len() > 2 { input.shape[2] } else { 1 };
    
    // Output same shape
    let output_shape = input.shape.clone();
    let num_elements = input.len();

    let shader_src = format!(r#"
    struct Uniforms {{
        spatial_size: u32,
        channels: u32,
        eps: f32,
    }};
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> mean: array<f32>;
    @group(0) @binding(2) var<storage, read> var_: array<f32>;
    @group(0) @binding(3) var<storage, read> weight: array<f32>;
    @group(0) @binding(4) var<storage, read> bias: array<f32>;
    @group(0) @binding(5) var<storage, read_write> output: array<f32>;
    @group(0) @binding(6) var<uniform> uniforms: Uniforms;
    
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
        let idx = global_id.x;
        let total_elements = u32(arrayLength(&input));
        
        if (idx >= total_elements) {{
            return;
        }}
        
        // Map 1D index to (b, c, s)
        // input is [Batch, Channel, Spatial] - Row Major
        // idx = b * (C*S) + c * S + s
        
        // spatial index s = idx % Spatial
        // c = (idx / Spatial) % Channels
        // b = idx / (Spatial * Channels)
        
        let s = idx % uniforms.spatial_size;
        let tmp = idx / uniforms.spatial_size;
        let c = tmp % uniforms.channels;
        
        // Read params for channel c
        let m = mean[c];
        let v = var_[c];
        let w = weight[c];
        let b = bias[c];
        let x = input[idx];
        
        // y = (x - mean) / sqrt(var + eps) * w + b
        let norm = (x - m) / sqrt(v + uniforms.eps);
        output[idx] = norm * w + b;
    }}
    "#);
    
    // Uniforms
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Uniforms {
        spatial_size: u32,
        channels: u32,
        eps: f32,
    }
    
    let uniforms_data = Uniforms {
        spatial_size: spatial as u32,
        channels: channels as u32,
        eps,
    };
    
    #[cfg(target_arch = "wasm32")]
    let dq_owned = get_gpu_device()?;
    #[cfg(target_arch = "wasm32")]
    let dq = &dq_owned;
    
    #[cfg(not(target_arch = "wasm32"))]
    let dq = get_gpu_device()?;

    let device = &dq.device;
    let queue = &dq.queue;
    
    // Buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BN Input"),
        contents: bytemuck::cast_slice(&input.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let mean_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BN Mean"),
        contents: bytemuck::cast_slice(&running_mean.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let var_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BN Var"),
        contents: bytemuck::cast_slice(&running_var.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BN Weight"),
        contents: bytemuck::cast_slice(&weight.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BN Bias"),
        contents: bytemuck::cast_slice(&bias.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let output_size = (num_elements * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BN Output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BN Uniforms"),
        contents: bytemuck::bytes_of(&uniforms_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    // Pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("BN Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_src)),
    });
    
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("BN Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BN Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: mean_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: var_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: weight_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: bias_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: uniforms_buffer.as_entire_binding() },
        ],
    });
    
    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("BN Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("BN Pass"), timestamp_writes: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        
        let workgroup_size = 64;
        let groups = (num_elements as u32 + workgroup_size - 1) / workgroup_size;
        cpass.dispatch_workgroups(groups, 1, 1);
    }
    
    // Readback
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));
    
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    
    device.poll(wgpu::Maintain::Wait);
    
    if let Ok(Ok(())) = pollster::block_on(receiver) {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        
        Ok(Array::new(output_shape, result_vec))
    } else {
        Err(anyhow!("Failed to read GPU buffer"))
    }
}
