# NumRs Examples

Este directorio contiene ejemplos prácticos del sistema de dispatch y validación de backends.

## dispatch_demo.rs

**Propósito**: Demostración completa del sistema de dispatch con validación y selección de kernels.

**Qué muestra**:

1. **FASE 1: Validación de backends**
   - Verifica SIMD, BLAS, WebGPU, GPU
   - Prueba funcional de cada backend (no solo compile-time)
   - Output: `✅` o `❌` por cada backend

2. **FASE 2: Selección de kernels**
   - Muestra qué implementación se eligió para cada tipo de operación
   - Prioridades: BLAS > WebGPU > SIMD > Scalar
   - Output: `elementwise: cpu-simd | reduction: cpu-simd | matmul: blas`

3. **FASE 3: Ejecución con zero-cost dispatch**
   - Ejecuta operaciones reales (add, sum, matmul)
   - Mide tiempos de ejecución
   - Para matmul, benchmarkea 100x100 si BLAS disponible

4. **FASE 4: Comparación de performance**
   - Explica overhead del fast-path (<1ns) vs legacy (5-10ns)
   - Muestra por qué el dispatch table es zero-cost

**Cómo ejecutar**:

```bash
# Con backend CPU+SIMD (default)
cargo run --example dispatch_demo

# Con backend BLAS (MKL) estático
cargo run --example dispatch_demo --features mkl

# Con backend BLAS (BLIS) estático
cargo run --example dispatch_demo --features blis

# Release mode para performance real
cargo run --release --example dispatch_demo --features mkl
```

**Output esperado**:

```
🚀 NumRs Dispatch System Demo

📋 FASE 1: Validando backends disponibles...

Resultados de validación:
  ├─ SIMD
  │  ├─ Disponible: true
  │  └─ Validado:   true ✅
  ├─ BLAS
  │  ├─ Disponible: true
  │  └─ Validado:   true ✅
  ├─ WebGPU
  │  ├─ Disponible: false
  │  └─ Validado:   false ❌
  └─ GPU (CUDA/Metal)
     ├─ Disponible: false
     └─ Validado:   false (pendiente implementación)

🎯 FASE 2: Kernels seleccionados por el dispatch system:

Dispatch Table:
  ├─ Elementwise → cpu-simd
  ├─ Reduction   → cpu-simd
  └─ MatMul      → blas

⚡ FASE 3: Ejecutando operaciones (zero-cost dispatch)...

Test 1: Elementwise Add
  Input A:  [1.0, 2.0, 3.0, 4.0]
  Input B:  [1.0, 1.0, 1.0, 1.0]
  Result:   [2.0, 3.0, 4.0, 5.0]
  Backend:  cpu-simd
  Time:     152ns

Test 3: Matrix Multiplication
  Matrix A: 2x2 = [1.0, 2.0, 3.0, 4.0]
  Matrix B: 2x2 (identity)
  Result:   [1.0, 2.0, 3.0, 4.0]
  Backend:  blas
  Time:     3.2µs

Test 3b: MatMul 100x100 (BLAS optimizado)
  Matrix A: 100x100
  Matrix B: 100x100
  Result:   100x100
  Backend:  blas (BLAS estático)
  Time:     0.48 ms
  GFLOPS:   ~4.1

✅ RESUMEN:

Sistema de dispatch inicializado correctamente:
  1. Backends validados funcionalmente
  2. Mejores implementaciones seleccionadas
  3. Dispatch table creado (static, OnceCell)
  4. Hot-path operaciones con zero overhead

🚀 BLAS disponible → Performance óptima para matmul!
```

## verify_static_blas.rs

**Propósito**: Verifica que BLAS esté correctamente linkeado de forma estática.

**Qué muestra**:
- Compile-time info (qué features están activos)
- Runtime test (ejecuta sgemm)
- Verifica que no necesite librerías externas

**Cómo ejecutar**:

```bash
cargo run --example verify_static_blas --features mkl
```

## Notas importantes

1. **Validación vs Disponibilidad**:
   - `available = true` → compilado con el feature
   - `validated = true` → probado funcionalmente y confirmado que funciona

2. **Prioridades de selección**:
   - Elementwise: WebGPU > SIMD > Scalar
   - Reduction: BLAS > SIMD > Scalar
   - MatMul: BLAS > WebGPU > SIMD > Scalar

3. **Performance esperado (release mode)**:
   - Elementwise SIMD: ~100-200ns para vectores pequeños
   - MatMul BLAS 100x100: ~0.5ms (~4 GFLOPS)
   - MatMul BLAS 1000x1000: ~50ms (~40 GFLOPS)
   - MatMul BLAS 2048x2048: ~1.5s (~11 GFLOPS)

4. **Troubleshooting**:
   - Si BLAS no valida: verifica que feature esté activo (`cargo build --features mkl`)
   - Si SIMD no valida: tu CPU puede no soportar las instrucciones
   - Si solo Scalar: considera usar `--features mkl` para mejor performance
