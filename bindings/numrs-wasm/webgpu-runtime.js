// ============================================================================
// WebGPU Host Runtime
// ============================================================================
//
// Este módulo ejecuta operaciones GPU desde descriptores enviados por WASM.
// WASM no tiene acceso directo a WebGPU, así que:
// 1. WASM genera un descriptor de operación
// 2. JS recibe el descriptor
// 3. JS ejecuta la operación en WebGPU
// 4. JS escribe el resultado de vuelta en memoria WASM

class WebGPURuntime {
    constructor() {
        this.device = null;
        this.queue = null;
        this.initialized = false;
        this.pipelineCache = new Map();
        this.shaderCache = new Map();
    }

    async init() {
        if (this.initialized) return true;

        try {
            // Check WebGPU support
            if (!navigator.gpu) {
                console.warn('WebGPU not supported');
                return false;
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.warn('No WebGPU adapter found');
                return false;
            }

            this.device = await adapter.requestDevice();
            this.queue = this.device.queue;
            this.initialized = true;

            console.log('✓ WebGPU initialized');
            return true;
        } catch (error) {
            console.error('WebGPU initialization failed:', error);
            return false;
        }
    }

    // ========================================================================
    // Shader Generators
    // ========================================================================

    getAddShader() {
        return `
            @group(0) @binding(0) var<storage, read> inputA: array<f32>;
            @group(0) @binding(1) var<storage, read> inputB: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx < arrayLength(&output)) {
                    output[idx] = inputA[idx] + inputB[idx];
                }
            }
        `;
    }

    getMulShader() {
        return `
            @group(0) @binding(0) var<storage, read> inputA: array<f32>;
            @group(0) @binding(1) var<storage, read> inputB: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx < arrayLength(&output)) {
                    output[idx] = inputA[idx] * inputB[idx];
                }
            }
        `;
    }

    getSinShader() {
        return `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx < arrayLength(&output)) {
                    output[idx] = sin(input[idx]);
                }
            }
        `;
    }

    getExpShader() {
        return `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256, 1, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx < arrayLength(&output)) {
                    output[idx] = exp(input[idx]);
                }
            }
        `;
    }

    getMatmulShader() {
        return `
            @group(0) @binding(0) var<storage, read> matrixA: array<f32>;
            @group(0) @binding(1) var<storage, read> matrixB: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            @group(0) @binding(3) var<storage, read> dims: array<u32>; // [M, K, N]

            @compute @workgroup_size(16, 16, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let M = dims[0];
                let K = dims[1];
                let N = dims[2];
                
                let row = global_id.y;
                let col = global_id.x;
                
                if (row >= M || col >= N) {
                    return;
                }
                
                var sum = 0.0;
                for (var i = 0u; i < K; i = i + 1u) {
                    sum += matrixA[row * K + i] * matrixB[i * N + col];
                }
                
                result[row * N + col] = sum;
            }
        `;
    }

    getSumShader() {
        return `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            var<workgroup> shared: array<f32, 256>;

            @compute @workgroup_size(256, 1, 1)
            fn main(
                @builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) workgroup_id: vec3<u32>
            ) {
                let tid = local_id.x;
                let gid = workgroup_id.x * 256u + tid;
                
                // Load data
                var val = 0.0;
                if (gid < arrayLength(&input)) {
                    val = input[gid];
                }
                shared[tid] = val;
                workgroupBarrier();
                
                // Reduction
                for (var s = 128u; s > 0u; s = s >> 1u) {
                    if (tid < s) {
                        shared[tid] += shared[tid + s];
                    }
                    workgroupBarrier();
                }
                
                // Write result
                if (tid == 0u) {
                    output[workgroup_id.x] = shared[0];
                }
            }
        `;
    }

    getShaderForOp(opType) {
        const cached = this.shaderCache.get(opType);
        if (cached) return cached;

        let shaderCode;
        switch (opType) {
            case 'Add': shaderCode = this.getAddShader(); break;
            case 'Mul': shaderCode = this.getMulShader(); break;
            case 'Sin': shaderCode = this.getSinShader(); break;
            case 'Exp': shaderCode = this.getExpShader(); break;
            case 'Matmul': shaderCode = this.getMatmulShader(); break;
            case 'Sum': shaderCode = this.getSumShader(); break;
            default:
                throw new Error(`Unknown op type: ${opType}`);
        }

        const shaderModule = this.device.createShaderModule({
            code: shaderCode
        });
        
        this.shaderCache.set(opType, shaderModule);
        return shaderModule;
    }

    // ========================================================================
    // Execute GPU Operation
    // ========================================================================

    async executeOp(descriptor, ...dataArrays) {
        if (!this.initialized) {
            await this.init();
        }

        if (!this.initialized) {
            throw new Error('WebGPU not available');
        }

        const { op_type, input_shapes, output_shape, workgroup_size, num_workgroups } = descriptor;

        // Create buffers
        const inputBuffers = dataArrays.map((data, i) => {
            const buffer = this.device.createBuffer({
                size: data.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(buffer.getMappedRange()).set(data);
            buffer.unmap();
            return buffer;
        });

        const outputSize = output_shape.reduce((a, b) => a * b, 1) * 4;
        const outputBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const stagingBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Get shader
        const shaderModule = this.getShaderForOp(op_type);

        // Create bind group layout
        const entries = [
            ...inputBuffers.map((_, i) => ({
                binding: i,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' }
            })),
            {
                binding: inputBuffers.length,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' }
            }
        ];

        const bindGroupLayout = this.device.createBindGroupLayout({ entries });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        // Create pipeline
        const pipeline = this.device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });

        // Create bind group
        const bindGroupEntries = [
            ...inputBuffers.map((buffer, i) => ({
                binding: i,
                resource: { buffer }
            })),
            {
                binding: inputBuffers.length,
                resource: { buffer: outputBuffer }
            }
        ];

        const bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: bindGroupEntries
        });

        // Execute
        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(...num_workgroups);
        passEncoder.end();

        commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, outputSize);

        this.queue.submit([commandEncoder.finish()]);

        // Read results
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(stagingBuffer.getMappedRange()).slice();
        stagingBuffer.unmap();

        // Cleanup
        inputBuffers.forEach(b => b.destroy());
        outputBuffer.destroy();
        stagingBuffer.destroy();

        return result;
    }
}

// Export singleton
const webgpuRuntime = new WebGPURuntime();

module.exports = { webgpuRuntime, WebGPURuntime };
