# NumRs Metal Backend

Backend de alto rendimiento para operaciones numéricas usando Metal, la API de gráficos de Apple.

## Características

- **Plataforma**: Exclusivo para macOS/iOS
- **Operaciones soportadas**:
  - Elementwise: add, mul, sub, div, sqrt, sin, cos, exp, log, relu, sigmoid, etc.
  - Linear Algebra: matmul (tiled con threadgroup memory), dot product
  - Reductions: sum (iterativo con shared memory)

## Arquitectura

El backend de Metal está basado en la misma arquitectura que WebGPU, pero optimizado para las características específicas de Metal:

### 1. Device Management
```rust
static METAL_DEVICE: OnceCell<Result<MetalContext, anyhow::Error>> = OnceCell::new();

fn get_metal_device() -> Result<&'static MetalContext> {
    METAL_DEVICE.get_or_init(|| {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        Ok(MetalContext { device, queue })
    })
}
```

### 2. Shader Compilation
Los shaders se compilan dinámicamente usando MSL (Metal Shading Language):

```metal
#include <metal_stdlib>
using namespace metal;

kernel void elementwise(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= size) return;
    out[idx] = a[idx] + b[idx];
}
```

### 3. Optimized Matmul
Matrix multiplication usa tiling con threadgroup memory (similar a shared memory en CUDA):

- **Tile size**: 16x16
- **Threads per threadgroup**: 8x8 (cada thread computa un bloque 2x2)
- **Vectorización**: Usa `float4` y operaciones `dot()` para mejor throughput
- **Memoria compartida**: `threadgroup float tileA[256]` y `tileB[256]`

```metal
threadgroup float tileA[TILE * TILE];
threadgroup float tileB[TILE * TILE];

// Each thread computes a 2x2 block
for (uint t = 0; t < TILE; t += 4) {
    float4 a00 = ...;
    float4 b0 = ...;
    sum00 += dot(a00, b0);
    // ...
}
```

### 4. Reduction Strategy
Usa reducción iterativa con shared memory:

1. Primera pasada: reduce workgroups de 256 elementos
2. Pasadas siguientes: reduce los parciales hasta obtener un único valor
3. Usa `threadgroup_barrier(mem_flags::mem_threadgroup)` para sincronización

## API Pública

```rust
// Elementwise operations
pub fn elementwise_metal(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array>

// Matrix multiplication
pub fn matmul_metal(a: &Array, b: &Array) -> Result<Array>

// Reduction
pub fn reduction_metal(a: &Array, axis: Option<usize>) -> Result<Array>

// Availability check
pub fn is_available_cached() -> bool
```

## Integración con Dispatch System

El backend de Metal se integra automáticamente en el sistema de dispatch:

```rust
// En dispatch.rs
let (elementwise, elem_backend) = if validation.metal_validated {
    (kernel_elementwise_metal as ElementwiseFn, "metal")
} else if validation.webgpu_validated {
    (kernel_elementwise_webgpu as ElementwiseFn, "webgpu")
} else {
    // fallback...
}
```

**Prioridad de backends (macOS)**:
1. **BLAS (Accelerate)**: Para matmul y dot product (mejor rendimiento)
2. **Metal**: Para elementwise y cuando BLAS no está disponible
3. **SIMD (NEON)**: Fallback en Apple Silicon
4. **Scalar**: Último recurso

## Performance Expectations

En Apple Silicon (M1/M2/M3):

- **Elementwise**: 2-5x más rápido que SIMD NEON
- **Matmul (pequeño)**: Similar a BLAS Accelerate
- **Matmul (grande)**: Puede ser más lento que BLAS por overhead de GPU
- **Latencia**: ~50-100μs overhead por kernel launch

## Benchmarking

```bash
# En macOS
cargo run --bin numrs-bench --release

# Output esperado:
# Detected backends: Scalar, SIMD, BLAS, Metal
# Metal matmul (256x256): ~100 Gops/s
# Metal elementwise add: ~500 Mops/s
```

## Limitaciones Actuales

1. **Axis reductions**: No implementadas (solo reduction global)
2. **Streaming**: Para matrices muy grandes, podría requerir tiling en host
3. **macOS only**: No compila en Windows/Linux (dependencias condicionales)
4. **Float32 only**: No soporta float64 o int32 todavía

## Desarrollo Futuro

- [ ] Implementar axis reductions con Metal
- [ ] Optimizar para matrices grandes (>2048x2048)
- [ ] Agregar soporte para float16 (half precision)
- [ ] Batch operations (múltiples matrices en una sola llamada)
- [ ] Integration con Metal Performance Shaders (MPS) para BLAS alternativo
- [ ] iOS support

## Referencias

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [metal-rs crate](https://docs.rs/metal/)
