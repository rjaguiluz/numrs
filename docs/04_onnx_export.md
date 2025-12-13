# Tutorial 4: Exportación de Modelos a ONNX

## ¿Qué es ONNX?
**ONNX (Open Neural Network Exchange)** es un formato estándar abierto para representar modelos de Machine Learning. NumRs soporta la exportación nativa a ONNX, lo que significa que puedes entrenar tu modelo en Rust y desplegarlo en cualquier entorno (Python, C++, Web, Edge Devices) que soporte ONNX Runtime.

## Cómo funciona en NumRs
La exportación en NumRs funciona mediante **Tracing** (Rastreo), similar a `torch.jit.trace`.

### 🆚 PyTorch vs NumRs: Exportación
| Concepto    | PyTorch                                  | NumRs                                                                                      |
| ----------- | ---------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Función** | `torch.onnx.export(model, args, path)`   | `numrs::ops::export::export_to_onnx(&tensor, path)`                                        |
| **Input**   | Tupla de argumentos `(x, y)`             | Tensor "final" del grafo computacional                                                     |
| **Alcance** | Exporta la ejecución completa del modelo | Exporta **el grafo que generó el tensor**, ya sea un modelo entero o una operación simple. |

> [!NOTE]
> **Exportación de Tensores Puros**: No necesitas un `Module` para exportar. Si calculas `let z = x.add(&y)?`, puedes llamar a `export_to_onnx(&z, ...)` y obtendrás un archivo ONNX válido que contiene solo esa suma. Esto es útil para utilidades o pre-procesamiento.

## Ejemplo de Exportación

Tomemos el modelo `ForecastCNN` de los tutoriales anteriores. Una vez entrenado, el proceso de exportación es el siguiente:

### 1. Preparar un Input Dummy
El input debe tener la misma forma (shape) que los datos reales, pero no necesita tener valores significativos.

**Importante**: Los inputs del grafo deben tener `requires_grad = false`. Si tienen `true`, el exportador podría confundirlos con pesos entrenables.

```rust
use numrs::autograd::Tensor;
use numrs::Array;

// Input dummy: [Batch=1, SeqLen=128]
// Nota: Backends como WebGPU requieren shapes consistentes.
let dummy_data = Array::zeros(vec![1, 128]); 
let input_tensor = Tensor::new(dummy_data, false); // requires_grad = false
```

### 2. Ejecutar un Forward Pass
El modelo debe estar en modo evaluación (`eval()`) si usas BatchNorm o Dropout, para asegurar un comportamiento determinista.

```rust
// Asegurar modo evaluación
model.eval();

// Ejecutar forward para trazar el grafo
let output = model.forward(&input_tensor)?;
```

### 3. Llamar a `export_to_onnx`
La función toma el tensor de salida final y recorre el grafo hacia atrás hasta los inputs.

```rust
use numrs::ops::export::export_to_onnx;

let path = "forecast_cnn.onnx.json"; // NumRs usa un formato JSON-friendly intermedio actualmente
export_to_onnx(&output, path)?;

println!("Modelo exportado a {}", path);
```

## Consideraciones Avanzadas

### Batch Normalization & Running Stats
NumRs maneja automáticamente los estados internos (`running_mean`, `running_var`) de BatchNorm. Estos se exportan como inputs adicionales al nodo ONNX pero marcados como pesos fijos (Initializers), por lo que el usuario final no necesita proveerlos manualmente.

### Dynamic Shapes
Actualmente, NumRs exporta grafos con formas estáticas basadas en el `dummy_input`. Si necesitas formas dinámicas, asegúrate de que el runtime de inferencia soporte redimensionamiento, o exporta versiones para cada tamaño esperado.

## Siguiente Paso
Una vez tengas tu archivo `.onnx.json`, puedes cargarlo para inferencia usando NumRs (Tutorial 05) o cualquier otro runtime compatible.
