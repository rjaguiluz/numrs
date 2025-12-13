# Tutorial 2: NumRs Tensores y Deep Learning

## Introducción
Mientras que `Array` es el motor de cómputo, `Tensor` es el alma del Deep Learning en `NumRs`. Un `Tensor` envuelve un `Array` y añade la capacidad de **Diferenciación Automática (Autograd)**.

### 🆚 PyTorch vs NumRs: Deep Learning
| Concepto     | PyTorch (Python)                            | NumRs (Rust)                     |
| ------------ | ------------------------------------------- | -------------------------------- |
| **Autograd** | `x = torch.tensor(..., requires_grad=True)` | `let x = Tensor::new(..., true)` |
| **Linear**   | `nn.Linear(10, 32)`                         | `Linear::new(10, 32)?`           |
| **Forward**  | `model(x)`                                  | `model.forward(&x)?`             |
| **Backward** | `loss.backward()`                           | `loss.backward()?`               |

## 1. El Objeto Tensor

```rust
use numrs::autograd::Tensor;
use numrs::Array;

// Crear un tensor que requiere gradientes (Weight/Bias)
let data = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
let t = Tensor::new(data, true); // requires_grad = true

println!("Requiere gradiente: {}", t.requires_grad);
```

## 2. Autograd: El Grafo Computacional

Cuando operas con tensores que requieren gradiente, `NumRs` construye dinámicamente un grafo computacional.

```rust
// y = x^2 + 2
let x = Tensor::new(Array::new(vec![1], vec![3.0]), true);
let dos = Tensor::new(Array::new(vec![1], vec![2.0]), false);

let x_sq = x.mul(&x)?; // x^2
let y = x_sq.add(&dos)?; // x^2 + 2

// Backpropagation
y.backward()?;

// Gradiente: dy/dx = 2x = 2(3) = 6
println!("Gradiente de x: {:?}", x.gradient().unwrap());
```

## 3. Construyendo Redes Neuronales (Module API)

`NumRs` ofrece una API estilo PyTorch (`numrs::autograd::nn`).

```rust
use numrs::autograd::nn::{Linear, ReLU, Sequential, Module};

// Definir un MLP simple
let model = Sequential::new(vec![
    Box::new(Linear::new(10, 32)?), // Input 10 -> Hidden 32
    Box::new(ReLU),
    Box::new(Linear::new(32, 1)?),  // Hidden 32 -> Output 1
]);

// Forward pass
let input = Tensor::new(Array::zeros(vec![1, 10]), false);
let output = model.forward(&input)?;
```

## 4. Capas Avanzadas
NumRs soporta capas modernas para estabilizar y regularizar el entrenamiento.

```rust
use numrs::autograd::nn::{Conv1d, BatchNorm1d, Dropout};

// Convolución 1D: ideal para series de tiempo
let conv = Conv1d::new(1, 16, 3, 1, 0)?; // In: 1, Out: 16, Kernel: 3

// Batch Normalization: Mantiene la media=0 y var=1
let bn = BatchNorm1d::new(16)?;

// Dropout: Apaga neuronas aleatoriamente (p=0.5)
let dropout = Dropout::new(0.5);

// Uso en modo entrenamiento vs evaluación
// dropout.train(); // p = 0.5
// dropout.eval();  // p = 0.0 (Identity)
```

## 5. Entrenamiento (Trainer API)

Para evitar escribir bucles de entrenamiento manuales, `NumRs` provee `TrainerBuilder`.

```rust
use numrs::{TrainerBuilder, Dataset};
use numrs::autograd::MSELoss;

// Datos Dummy
let dataset = Dataset::new(inputs, targets, 32);

// Configurar Trainer
let mut trainer = TrainerBuilder::new(model)
    .learning_rate(0.001)
    .build_adam(Box::new(MSELoss)); // Optimizador Adam + MSE Loss

// Entrenar
trainer.fit(&dataset, None, 10, true)?; // 10 épocas
```

## 6. Caso de Uso Real: Regresión de Precios (Tabular Data)
Supongamos que queremos predecir el precio de casas basándonos en 10 características (m2, habitaciones, año, etc.).

```rust
// Modelo: MLP de 2 capas ocultas
let model = Sequential::new(vec![
    // Input Layer: 10 features -> 64 neuronas
    Box::new(Linear::new(10, 64)?), 
    Box::new(ReLU),
    
    // Hidden Layer: 64 -> 32 neuronas
    Box::new(Linear::new(64, 32)?),
    Box::new(ReLU),
    Box::new(Dropout::new(0.2)), // Regularización
    
    // Output Layer: 32 -> 1 (Precio estimado)
    Box::new(Linear::new(32, 1)?),
]);

// Loss: Mean Squared Error (típico para regresión)
trainer.build_adam(Box::new(MSELoss));
```

## Resumen
Con `Tensor` y `Autograd`, puedes construir cualquier modelo diferenciable. En el siguiente tutorial, aplicaremos esto a **Series de Tiempo**.
