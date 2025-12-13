# Tutorial 6: Hands On - Proyecto End-to-End

## Introducción
En este tutorial final, conectaremos todo lo aprendido para construir una solución completa de Machine Learning.
Simularemos un sistema de **Detección de Fraude** (Clasificación Binaria).

Este tutorial profundiza en el **por qué** de cada decisión matemática y arquitectónica.

---

## 1. Generación de Datos
El primer paso es entender nuestros datos.

```rust
use numrs::Array;
use rand::prelude::*;

fn get_fraud_data(n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rng = thread_rng();
    let mut x = Vec::new();
    let mut y = Vec::new();

    for _ in 0..n {
        // Feature 1: Monto normalizado (0.0 = $0, 1.0 = $10,000)
        let amount = rng.gen_range(0.0..1.0);
        // Feature 2: Hora del día normalizada (0.0 = 00:00, 1.0 = 23:59)
        let hour = rng.gen_range(0.0..1.0);
        // Feature 3: Score de Riesgo externo (0.0 = Seguro, 1.0 = Peligroso)
        let risk = rng.gen_range(0.0..1.0);

        // LÓGICA DE NEGOCIO (Ground Truth):
        // 1. Si el riesgo es extremo (>0.9) -> FRAUDE SEGURO
        // 2. Si el riesgo es alto (>0.8) Y el monto es alto (>0.5) -> FRAUDE
        let is_fraud = (risk > 0.8 && amount > 0.5) || (risk > 0.9);
        
        let label = if is_fraud { 1.0 } else { 0.0 };

        x.push(vec![amount, hour, risk]);
        y.push(label);
    }
    (x, y)
}
```

### 🧠 ¿Por qué normalizar?
Las redes neuronales funcionan mejor cuando todas las entradas están en rangos similares (como 0 a 1).
- Si usáramos "Monto" en dólares (e.g., 5000.0) y "Riesgo" (0.9), el gradiente del "Monto" dominaría totalmente la actualización de pesos, haciendo que el modelo ignore el "Riesgo".

---

## 2. Loss Personalizada (Custom BCELoss)
Para clasificación binaria, queremos minimizar la **Binary Cross Entropy**.
La fórmula matemática es:
$$ Loss = - \frac{1}{N} \sum ( y \cdot \log(p) + (1-y) \cdot \log(1-p) ) $$

Implementémosla paso a paso usando Tensores:

```rust
use numrs::autograd::{Tensor, LossFunction};
use numrs::ops::{log, mean, add, mul, sub, neg};

struct BCELoss;

impl LossFunction for BCELoss {
    fn forward(&self, preds: &Tensor, targets: &Tensor) -> anyhow::Result<Tensor> {
        // preds: Probabilidades [0..1]
        // targets: Realidad [0 o 1]
        
        // 1. Estabilidad Numérica (Epsilon)
        // EXPLICACIÓN: log(0) es -infinito. Si el modelo predice 0.0 exacto, el programa explota.
        // Sumamos un valor minúsculo (1e-7) para evitarlo.
        let epsilon = Tensor::new(Array::new(vec![1], vec![1e-7]), false);
        let preds_safe = preds.add(&epsilon)?;

        // 2. Primer Término: Cuando es Fraude (y=1)
        // Queremos maximizar log(p). Si p=1, log(1)=0 (Loss baja). Si p=0.1, log(0.1)=-2.3 (Loss alta).
        let log_p = log(&preds_safe)?;
        let term1 = targets.mul(&log_p)?;

        // 3. Segundo Término: Cuando es Legítimo (y=0)
        // Queremos maximizar log(1-p). Si p=0, log(1)=0.
        let one = Tensor::new(Array::ones(targets.shape().to_vec()), false);
        let one_minus_y = one.sub(targets)?;
        let one_minus_p = one.sub(&preds_safe)?; 
        let term2 = one_minus_y.mul(&log(&one_minus_p)?)?;

        // 4. Promedio Negativo
        // Sumamos ambos términos y negamos porque el optimizador MINIMIZA la loss,
        // pero nosotros queremos MAXIMIZAR la probabilidad (Maximum Likelihood).
        let total = term1.add(&term2)?;
        let mean_val = mean(&total, None)?;
        neg(&mean_val)
    }
}
```

---

## 3. Entrenamiento del Modelo

Usamos un MLP (Multi-Layer Perceptron).
- **ReLU** en capas ocultas: Introduce no-linealidad para aprender patrones complejos (como "Riesgo alto Y Monto alto").
- **Sigmoid** en salida: Fuerza al valor a estar entre 0 y 1 (Probabilidad).

```rust
use numrs::autograd::nn::{Sequential, Linear, ReLU, Sigmoid};
use numrs::TrainerBuilder;

// Input 3 -> L1(16) -> L2(8) -> Out(1)
let model = Sequential::new(vec![
    Box::new(Linear::new(3, 16)?),
    Box::new(ReLU),
    Box::new(Linear::new(16, 8)?),
    Box::new(ReLU),
    Box::new(Linear::new(8, 1)?),
    Box::new(Sigmoid), // Convierte logits a probabilidad [0, 1]
]);

// Loss Function: Nuestra BCELoss personalizada
let mut trainer = TrainerBuilder::new(model)
    .learning_rate(0.01)
    .build_adamw(Box::new(BCELoss));

trainer.fit(&train_dataset, Some(&test_dataset), 20, true)?;
```

---

## 4. Exportación y Producción (Final)

Una vez entrenado, no queremos depender del código de entrenamiento. Exportamos el "cerebro" congelado.

### A. Exportar a ONNX
Generamos un trazo (trace) pasando un dato ficticio. NumRs graba el grafo.

```rust
// Dato dummy [1, 3]
let dummy_input = Tensor::new(Array::zeros(vec![1, 3]), false);
let output = trainer.model().forward(&dummy_input)?;

// Guardar
numrs::ops::export::export_to_onnx(&output, "fraud_detector.onnx.json")?;
println!("Modelo exportado exitosamente.");
```

### B. Inferencia en Producción
En tu servidor web o dispositivo IoT, cargas el modelo y ejecutas.

```rust
use numrs::ops::model::{load_onnx, infer};

// 1. Cargar modelo (hacer esto AL INICIO de la app, es pesado)
let onnx_model = load_onnx("fraud_model.onnx.json")?;

// 2. Función de API
fn predict_fraud(amount: f32, hour: f32, risk: f32) -> bool {
    // Convertir input real a Array [B=1, Feat=3]
    let input_data = Array::new(vec![1, 3], vec![amount, hour, risk]);
    
    // Mapear input (nombre "tensor_0" viene del export, se puede inspeccionar)
    let mut inputs = std::collections::HashMap::new();
    inputs.insert("tensor_0".to_string(), input_data);

    // Ejecutar (Thread-safe y ligero)
    let results = infer(&onnx_model, inputs).unwrap();
    
    // Obtener probabilidad
    let prob = results.get("output").unwrap().data[0];
    
    // Decisión de negocio
    println!("Probabilidad de Fraude: {:.2}%", prob * 100.0);
    prob > 0.5 
}
```

---

## 5. Código Completo

Este tutorial completo está implementado como un ejemplo ejecutable en NumRs.
Puedes ver cómo se integran la generación de datos, la loss personalizada, el entrenamiento y la exportación ONNX en un solo archivo.

**Código Fuente:** `examples/fraud_detection.rs`

Para ejecutarlo:

```bash
cargo run --release --example fraud_detection
```

¡Felicidades! Has construido un sistema de detección de fraude end-to-end con NumRs. 🎉
