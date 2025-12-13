
use crate::array::Array;
use anyhow::{Result, anyhow};
use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::backend::webgpu::get_gpu_device;

pub fn dropout_webgpu(
    input: &Array,
    p: f32,
    training: bool
) -> Result<Array> {
    if !training {
        // Identity for inference
        return Ok(input.clone());
    }
    
    // Validate p
    if p < 0.0 || p >= 1.0 {
        return Err(anyhow!("Dropout p must be in [0, 1)"));
    }
    
    let num_elements = input.len();
    let output_shape = input.shape.clone();
    
    // Scale factor
    let scale = 1.0 / (1.0 - p);
    
    // Seed generation
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u32;

    let shader_src = format!(r#"
    struct Uniforms {{
        num_elements: u32,
        prob: f32,
        scale: f32,
        seed: u32,
    }};
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> uniforms: Uniforms;
    
    // Simple PCG Hash
    fn hash(index: u32) -> u32 {{
        var len = index;
        var x = uniforms.seed + len;
        x = x * 747796405u + 2891336453u;
        var word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
        return (word >> 22u) ^ word;
    }}
    
    fn rand(index: u32) -> f32 {{
        return f32(hash(index)) / 4294967295.0;
    }}
    
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
        let idx = global_id.x;
        if (idx >= uniforms.num_elements) {{
            return;
        }}
        
        // Random check
        let r = rand(idx);
        if (r < uniforms.prob) {{
            // Dropped
            output[idx] = 0.0;
        }} else {{
            // Kept and scaled
            output[idx] = input[idx] * uniforms.scale;
        }}
    }}
    "#);
    
    // Uniforms
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Uniforms {
        num_elements: u32,
        prob: f32,
        scale: f32,
        seed: u32,
    }
    
    let uniforms_data = Uniforms {
        num_elements: num_elements as u32,
        prob: p,
        scale,
        seed,
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
        label: Some("Dropout Input"),
        contents: bytemuck::cast_slice(&input.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let output_size = (num_elements * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dropout Output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dropout Uniforms"),
        contents: bytemuck::bytes_of(&uniforms_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    // Pipeline
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Dropout Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_src)),
    });
    
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Dropout Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Dropout Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: uniforms_buffer.as_entire_binding() },
        ],
    });
    
    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Dropout Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Dropout Pass"), timestamp_writes: None });
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
