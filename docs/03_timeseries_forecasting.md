# Tutorial 3: Series de Tiempo con NumRs

## Introducción
Las series de tiempo pueden ser analizadas eficazmente usando **Convoluciones 1D (CNNs)**.

### 🆚 PyTorch vs NumRs: CNN Layers
| Layer         | PyTorch                              | NumRs                                            |
| ------------- | ------------------------------------ | ------------------------------------------------ |
| **Conv1D**    | `nn.Conv1d(in, out, kernel, stride)` | `Conv1d::new(in, out, kernel, stride, padding)?` |
| **BatchNorm** | `nn.BatchNorm1d(features)`           | `BatchNorm1d::new(features)?`                    |
| **Dropout**   | `nn.Dropout(p)`                      | `Dropout::new(p)`                                |

## 1. Arquitectura del Modelo (`ForecastCNN`)

Usamos una arquitectura apilada de convoluciones 1D, BatchNorm y Dropout para regularización.

```rust
struct ForecastCNN {
    conv1: Conv1d,      // Extrae features locales
    bn1: BatchNorm1d,   // Estabiliza entrenamiento
    dropout1: Dropout,  // Evita overfitting
    head: Sequential,   // Regresión final
}

impl ForecastCNN {
    pub fn new(seq_len: usize) -> Result<Self> {
        let conv1 = Conv1d::new(1, 16, 3, 1, 0)?; // Input 1 canal -> 16 filtros
        // ... (definición completa)
        
        let head = Sequential::new(vec![
            Box::new(Flatten::new(1, 2)), // Aplanar [Batch, Ch, Len] -> [Batch, Feat]
            Box::new(Linear::new(flat_features, 64)?),
            Box::new(ReLU),
            Box::new(Linear::new(64, 1)?), // Predicción escalar
        ]);
        
        Ok(Self { ... })
    }
}
```

## 2. Preparación de Datos

Las series de tiempo requieren normalización y ventaneo (windowing).

- **Windowing**: Convertir una secuencia larga en pares `(Input[t-N..t], Target[t+1])`.
- **Normalización**: Es crucial escalar los datos (e.g., restar media, dividir por desviación estándar) para que el modelo converja.

```rust
// Ejemplo simplificado
fn create_windows(data: Vec<f32>, seq_len: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    // Implementación de ventana deslizante...
}
```

## 3. Entrenamiento y Validación

El entrenamiento usa el mismo `Trainer` visto en el tutorial anterior.

```rust
let mut trainer = TrainerBuilder::new(model)
    .learning_rate(0.001)
    .build_adam(Box::new(MSELoss));

// Entrenar con validación automática
trainer.fit(&train_dataset, Some(&test_dataset), 50, true)?;
```

## 4. Caso de Uso Real: Detección de Anomalías (IoT)
Además de presagiar el futuro, este modelo puede detectar fallos en sensores. Si el modelo (entrenado en comportamiento normal) predice un valor muy diferente al real, es probable que sea una anomalía.

```rust
// Inferencia en tiempo real (Pseudo-código)
let history = sensor.read_last_128_values();
let prediction = model.predict(history);
let real_value = sensor.read_current();

// Threshold dinámico (e.g., 3 desviaciones estándar)
let threshold = 0.5; 

if (prediction - real_value).abs() > threshold {
    alert_system.trigger("Anomalía detectada en Sensor #5");
    println!("Esperado: {}, Real: {}", prediction, real_value);
}
```

## 5. Exportación y Despliegue
Para llevar este modelo a producción, `NumRs` utiliza el estándar ONNX.

- Consulta el **[Tutorial 04: Exportación a ONNX](./04_onnx_export.md)** para ver cómo guardar tu modelo entrenado.
- Consulta el **[Tutorial 05: Inferencia ONNX](./05_onnx_inference.md)** para aprender a cargar el modelo y hacer predicciones de alto rendimiento.

## Conclusión
Has aprendido a crear arrays, construir modelos de Deep Learning y aplicarlos a problemas reales de series de tiempo con `NumRs`. ¡Explora los `examples/` del repositorio para más código funcional!
