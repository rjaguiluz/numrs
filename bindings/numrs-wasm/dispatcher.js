// ============================================================================
// Smart Dispatcher: CPU (WASM) vs GPU (WebGPU)
// ============================================================================
//
// Decide automáticamente el mejor backend basado en:
// - Tamaño del tensor
// - Overhead de transferencia GPU
// - Disponibilidad de WebGPU

const { webgpuRuntime } = require('./webgpu-runtime');

class SmartDispatcher {
    constructor(wasmModule) {
        this.wasm = wasmModule;
        this.gpuAvailable = false;
        this.gpuThreshold = 10_000; // Elementos
        this.stats = {
            cpuOps: 0,
            gpuOps: 0,
            fallbacks: 0
        };
    }

    async init() {
        this.gpuAvailable = await webgpuRuntime.init();
        console.log(`GPU available: ${this.gpuAvailable}`);
        return this;
    }

    // ========================================================================
    // Heurísticas de selección de backend
    // ========================================================================

    shouldUseGPU(numElements, opType) {
        // Re-habilitar cuando tengamos WebGPU real funcionando
        // Por ahora, mantener en false para testing
        return false;
        
        /*
        if (!this.gpuAvailable) return false;

        // Operaciones que se benefician más de GPU
        const gpuFriendlyOps = ['matmul', 'sin', 'exp', 'log'];
        const isFriendly = gpuFriendlyOps.includes(opType);

        // Usar GPU solo si:
        // 1. Hay suficientes elementos
        // 2. La operación se beneficia de paralelización
        if (numElements < this.gpuThreshold) return false;
        if (!isFriendly && numElements < this.gpuThreshold * 2) return false;

        return true;
        */
    }

    estimateTransferTime(numElements) {
        // ~0.001ms por 1000 elementos (estimación)
        return (numElements / 1000) * 0.001;
    }

    // ========================================================================
    // Dispatch Functions
    // ========================================================================

    async add(a, shapeA, b, shapeB) {
        const numElements = shapeA.reduce((acc, x) => acc * x, 1);
        
        if (this.shouldUseGPU(numElements, 'add')) {
            try {
                this.stats.gpuOps++;
                // Obtener descriptor desde WASM
                const descriptor = this.wasm.add_f32_gpu_descriptor(shapeA);
                return await this.gpuRuntime.executeOp(descriptor, a, b);
            } catch (error) {
                console.warn('GPU add failed, falling back to CPU:', error.message);
                this.stats.fallbacks++;
            }
        }

        // CPU path
        this.stats.cpuOps++;
        return this.wasm.add_f32(a, shapeA, b, shapeB);
    }

    async mul(a, shapeA, b, shapeB) {
        const numElements = shapeA.reduce((acc, x) => acc * x, 1);
        
        if (this.shouldUseGPU(numElements, 'mul')) {
            try {
                this.stats.gpuOps++;
                const descriptor = this.wasm.mul_f32_gpu_descriptor(shapeA);
                return await this.gpuRuntime.executeOp(descriptor, a, b);
            } catch (error) {
                console.warn('GPU mul failed, falling back to CPU:', error.message);
                this.stats.fallbacks++;
            }
        }

        this.stats.cpuOps++;
        return this.wasm.mul_f32(a, shapeA, b, shapeB);
    }

    async sin(a, shape) {
        const numElements = shape.reduce((acc, x) => acc * x, 1);
        
        if (this.shouldUseGPU(numElements, 'sin')) {
            try {
                this.stats.gpuOps++;
                const descriptor = this.wasm.sin_f32_gpu_descriptor(shape);
                return await this.gpuRuntime.executeOp(descriptor, a);
            } catch (error) {
                console.warn('GPU sin failed, falling back to CPU:', error.message);
                this.stats.fallbacks++;
            }
        }

        this.stats.cpuOps++;
        return this.wasm.sin_f32(a, shape);
    }

    async matmul(a, shapeA, b, shapeB) {
        const [m, k] = shapeA;
        const [_k, n] = shapeB;
        const numElements = m * n;
        
        // Matmul se beneficia mucho de GPU incluso con arrays más pequeños
        if (this.gpuAvailable && numElements >= 1000) {
            try {
                this.stats.gpuOps++;
                const descriptor = this.wasm.matmul_f32_gpu_descriptor(shapeA, shapeB);
                return await this.gpuRuntime.executeOp(descriptor, a, b);
            } catch (error) {
                console.warn('GPU matmul failed, falling back to CPU:', error.message);
                this.stats.fallbacks++;
            }
        }

        this.stats.cpuOps++;
        return this.wasm.matmul_f32(a, shapeA, b, shapeB);
    }

    async sum(a, shape) {
        const numElements = shape.reduce((acc, x) => acc * x, 1);
        
        if (this.shouldUseGPU(numElements, 'sum')) {
            try {
                this.stats.gpuOps++;
                const descriptor = this.wasm.sum_f32_gpu_descriptor(shape);
                return await this.gpuRuntime.executeOp(descriptor, a);
            } catch (error) {
                console.warn('GPU sum failed, falling back to CPU:', error.message);
                this.stats.fallbacks++;
            }
        }

        this.stats.cpuOps++;
        return this.wasm.sum_f32(a, shape);
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    getStats() {
        return {
            ...this.stats,
            total: this.stats.cpuOps + this.stats.gpuOps,
            gpuUsagePercent: ((this.stats.gpuOps / (this.stats.cpuOps + this.stats.gpuOps)) * 100).toFixed(1)
        };
    }

    resetStats() {
        this.stats = { cpuOps: 0, gpuOps: 0, fallbacks: 0 };
    }
}

module.exports = { SmartDispatcher };
