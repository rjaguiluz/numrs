# Tutorial 5: Inferencia con Modelos ONNX

## Introducción
Una vez exportado un modelo (Tutorial 04), puedes cargarlo en una aplicación diferente.

### 🆚 ONNX Runtime (Python) vs NumRs Inference
| Concepto   | ONNX Runtime                                | NumRs (Native)                         |
| ---------- | ------------------------------------------- | -------------------------------------- |
| **Cargar** | `sess = ort.InferenceSession("model.onnx")` | `let model = load_onnx("model.onnx")?` |
| **Inputs** | Diccionario `{"in": np_array}`              | `HashMap<String, Array>`               |
| **Run**    | `sess.run(None, inputs)`                    | `ops::model::infer(&model, inputs)?`   |

## Flujo de Trabajo

### 1. Cargar el Modelo
La función `load_onnx` lee el archivo y reconstruye la estructura del grafo en memoria.

```rust
use numrs::ops::model::{load_onnx, infer};

let model_path = "forecast_cnn.onnx.json";
let model = load_onnx(model_path)?;

println!("Modelo cargado. Operaciones: {}", model.graph.nodes.len());
```

### 2. Identificar Inputs y Outputs
Es útil inspeccionar los nombres de los nodos de entrada y salida, ya que ONNX los requiere para asignar los tensores correctamente.

```rust
let input_name = &model.graph.inputs[0].name; // e.g., "tensor_0"
let output_name = &model.graph.outputs[0];    // e.g., "tensor_15"

println!("Input: {}, Output: {}", input_name, output_name);
```

### 3. Preparar los Datos
Los datos de entrada deben ser un `Array` de NumRs. 

**Nota sobre Shapes**: Si tu modelo incluía operaciones de `Reshape` que dependían de dimensiones fijas, asegúrate de proveer el input con la forma exacta que el modelo espera, o pre-procesarlo si el `Reshape` original no fue capturado en el grafo (como vimos en el ejemplo de Series de Tiempo).

```rust
use numrs::Array;
use std::collections::HashMap;

// Crear batch de inferencia (1 ejemplo, len 128)
// Importante: Si el modelo espera [1, 1, 128] (NCW), asegúrate de dar esa shape.
let data = Array::new(vec![1, 1, 128], vec![0.5; 128]);

// Mapa de Inputs
let mut inputs = HashMap::new();
inputs.insert(input_name.clone(), data);
```

### 4. Ejecutar Inferencia
La función `infer` es stateless (sin estado). Toma el modelo y los inputs, y retorna un mapa de outputs.

```rust
let results = infer(&model, inputs)?;

// Extraer el resultado deseado
let prediction_tensor = results.get(output_name)
    .expect("Output tenso not found");

println!("Predicción: {:?}", prediction_tensor);
```

## Optimizaciones de Producción
- **Reutilización**: Carga el modelo (`load_onnx`) una sola vez al inicio de tu aplicación (es un recurso pesado). Ejecuta `infer` múltiples veces (es ligero).
- **Batching**: Para mayor throughput, agrupa múltiples ejemplos en un solo `Array` de entrada (e.g., `[32, 1, 128]`).

## Conclusión
Con este flujo, has completado el ciclo de vida de MLOps en Rust:
1.  **Datos** (`Dataset`)
2.  **Entrenamiento** (`Trainer`)
3.  **Exportación** (`export_to_onnx`)
4.  **Despliegue** (`infer`)
