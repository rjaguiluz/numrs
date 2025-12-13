
use crate::array::Array;
use anyhow::{Result, anyhow};
use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::backend::webgpu::get_gpu_device;

/// WebGPU Conv1D Implementation
pub fn conv1d_webgpu(
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    stride: usize,
    padding: usize,
) -> Result<Array> {
    // 1. Validate shapes
    let batch_size = input.shape[0];
    let in_channels = input.shape[1];
    let in_length = input.shape[2];
    
    let out_channels = weight.shape[0];
    let kernel_size = weight.shape[2];
    
    // Output length calculation
    let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    let output_shape = vec![batch_size, out_channels, out_length];
    
    // Shader Source
    let shader_src = format!(r#"
    struct Uniforms {{
        batch_size: u32,
        in_channels: u32,
        in_length: u32,
        out_channels: u32,
        kernel_size: u32,
        out_length: u32,
        stride: u32,
        padding: u32,
    }};
    
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weight: array<f32>;
    @group(0) @binding(2) var<storage, read> bias: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> uniforms: Uniforms;
    
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
        // x: out_index (time step)
        // y: out_channel
        // z: batch_index
        
        let out_t = global_id.x;
        let out_c = global_id.y;
        let b = global_id.z;
        
        if (out_t >= uniforms.out_length || out_c >= uniforms.out_channels || b >= uniforms.batch_size) {{
            return;
        }}
        
        // Init accumulator with bias
        var sum = bias[out_c];
        
        // Loop over InChannels and Kernel
        for (var ic = 0u; ic < uniforms.in_channels; ic = ic + 1u) {{
            for (var k = 0u; k < uniforms.kernel_size; k = k + 1u) {{
                // Determine input time index
                // in_t = out_t * stride + k - padding
                // Signed arithmetic needed for padding check
                let in_t_signed = i32(out_t * uniforms.stride + k) - i32(uniforms.padding);
                
                if (in_t_signed >= 0 && in_t_signed < i32(uniforms.in_length)) {{
                    let in_t = u32(in_t_signed);
                    
                    // Input index: b * (C_in * LEN_in) + ic * LEN_in + t
                    let in_idx = b * (uniforms.in_channels * uniforms.in_length) + ic * uniforms.in_length + in_t;
                    
                    // Weight index: out_c * (C_in * K) + ic * K + k
                    let w_idx = out_c * (uniforms.in_channels * uniforms.kernel_size) + ic * uniforms.kernel_size + k;
                    
                    sum = sum + input[in_idx] * weight[w_idx];
                }}
            }}
        }}
        
        // Output index: b * (C_out * LEN_out) + out_c * LEN_out + out_t
        let out_idx = b * (uniforms.out_channels * uniforms.out_length) + out_c * uniforms.out_length + out_t;
        output[out_idx] = sum;
    }}
    "#);

    // 2. Uniforms Data
    let uniforms_data = [
        batch_size as u32,
        in_channels as u32,
        in_length as u32,
        out_channels as u32,
        kernel_size as u32,
        out_length as u32,
        stride as u32,
        padding as u32,
    ];

    #[cfg(target_arch = "wasm32")]
    let dq_owned = get_gpu_device()?;
    #[cfg(target_arch = "wasm32")]
    let dq = &dq_owned;
    
    #[cfg(not(target_arch = "wasm32"))]
    let dq = get_gpu_device()?;

    let device = &dq.device;
    let queue = &dq.queue;

    // 3. Create Buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Conv1D Input"),
        contents: bytemuck::cast_slice(&input.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Conv1D Weight"),
        contents: bytemuck::cast_slice(&weight.data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    // Handle optional bias (create zero buffer if None)
    let bias_content = if let Some(b) = bias {
        Cow::Borrowed(&b.data)
    } else {
        Cow::Owned(vec![0.0f32; out_channels])
    };
    
    let bias_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Conv1D Bias"),
        contents: bytemuck::cast_slice(&bias_content),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let output_size = (batch_size * out_channels * out_length * std::mem::size_of::<f32>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Conv1D Output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Conv1D Uniforms"),
        contents: bytemuck::cast_slice(&uniforms_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // 4. Pipeline Setup
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Conv1D Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_src)),
    });
    
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Conv1D Pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Conv1D Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: weight_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bias_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: uniforms_buffer.as_entire_binding() },
        ],
    });

    // 5. Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Conv1D Encoder") });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Conv1D Pass"), timestamp_writes: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        
        let workgroup_size_x = 64;
        let group_x = (out_length as u32 + workgroup_size_x - 1) / workgroup_size_x;
        cpass.dispatch_workgroups(group_x, out_channels as u32, batch_size as u32);
    }
    
    // 6. Readback
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
