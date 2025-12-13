# Referencia: Operaciones de NumRs (v0.1.0)

## Sobre NumRs
**NumRs** es un framework de Deep Learning y Computación Científica escrito 100% en Rust. 

**¿Por qué existe?**
Nació de la necesidad de tener un ecosistema de ML en Rust que fuera:
1.  **Nativo**: Sin bindings pesados a C++ (como Torch/TensorFlow).
2.  **Modular**: Arquitectura de backends intercambiables (CPU-SIMD, Metal, WebGPU, WASM).
3.  **Transparente**: Código legible y hackeable, ideal para educación e investigación.

### ✅ Aspectos Destacados
- **Diseño Extensible**: Añadir nuevas operaciones u optimizadores es trivial gracias al sistema de Traits (`Module`, `Optimizer`, `Backend`).
- **ONNX First**: Exportación nativa a ONNX para interoperabilidad total.
- **Seguridad de Memoria**: Aprovecha el borrow checker de Rust para prevenir errores comunes de concurrencia y memoria.

### 🚧 Áreas de Mejora
- **Madurez**: El ecosistema es joven comparado con Python/PyTorch.
- **Broadcasting**: Aún no tiene paridad total con NumPy en reglas de difusión automática complejas.

---

Este documento lista todas las operaciones disponibles en el módulo `numrs::ops` y su propósito.

## 1. Operaciones Element-wise (Elemento a Elemento)
Operaciones que se aplican independientemente a cada elemento del array. Soportan broadcasting limitado.

### Binarias
| Operación          | Función            | Descripción |
| ------------------ | ------------------ | ----------- |
| **Suma**           | `ops::add(&a, &b)` | `a + b`     |
| **Resta**          | `ops::sub(&a, &b)` | `a - b`     |
| **Multiplicación** | `ops::mul(&a, &b)` | `a * b`     |
| **División**       | `ops::div(&a, &b)` | `a / b`     |
| **Potencia**       | `ops::pow(&a, &b)` | `a ^ b`     |

### Unarias (Matemáticas)
| Operación           | Función                | Descripción                        |
| ------------------- | ---------------------- | ---------------------------------- |
| **Absoluto**        | `ops::abs(&a)`         | `\|x\|`                            |
| **Negativo**        | `ops::neg(&a)`         | `-x`                               |
| **Exponencial**     | `ops::exp(&a)`         | `e^x`                              |
| **Logaritmo**       | `ops::log(&a)`         | `ln(x)`                            |
| **Raíz Cuadrada**   | `ops::sqrt(&a)`        | `√x`                               |
| **Trigonométricas** | `sin`, `cos`, `tan`    | Funciones trigonométricas estándar |
| **Trig. Inversas**  | `asin`, `acos`, `atan` | Arcoseno, Arcocoseno, Arcotangente |

### Activaciones (Neural Networks)
| Operación      | Función                      | Descripción               |
| -------------- | ---------------------------- | ------------------------- |
| **ReLU**       | `ops::relu(&a)`              | `max(0, x)`               |
| **Sigmoid**    | `ops::sigmoid(&a)`           | `1 / (1 + e^-x)`          |
| **Tanh**       | `ops::tanh(&a)`              | Tangente hiperbólica      |
| **Softplus**   | `ops::softplus(&a)`          | `ln(1 + e^x)`             |
| **Leaky ReLU** | `ops::leaky_relu(&a, alpha)` | `x if x > 0 else alpha*x` |

---

## 2. Reducción (Agregación)
Operaciones que reducen una o más dimensiones del array.

| Operación    | Función                   | Descripción                      |
| ------------ | ------------------------- | -------------------------------- |
| **Suma**     | `ops::sum(&a, axis)`      | Suma elementos (total o por eje) |
| **Promedio** | `ops::mean(&a, axis)`     | Promedio aritmético              |
| **Varianza** | `ops::variance(&a, axis)` | Varianza muestral                |
| **Mínimo**   | `ops::min(&a, axis)`      | Valor mínimo                     |
| **Máximo**   | `ops::max(&a, axis)`      | Valor máximo                     |
| **ArgMax**   | `ops::argmax(&a, axis)`   | Índices del valor máximo         |

---

## 3. Álgebra Lineal (`ops::linalg`)
Operaciones matriciales y vectoriales.

| Operación  | Función               | Descripción                                |
| ---------- | --------------------- | ------------------------------------------ |
| **MatMul** | `ops::matmul(&a, &b)` | Multiplicación de matrices (2D o 3D batch) |
| **Dot**    | `ops::dot(&a, &b)`    | Producto punto de vectores                 |

---

## 4. Manipulación de Formas (`ops::shape`)
Reorganización de dimensiones sin cambiar los datos subyacentes (en la mayoría de los casos).

| Operación     | Función                        | Descripción                                            |
| ------------- | ------------------------------ | ------------------------------------------------------ |
| **Reshape**   | `ops::reshape(&a, shape)`      | Cambia las dimensiones del array                       |
| **Transpose** | `ops::transpose(&a, axis)`     | Permuta dimensiones (invertir o específico)            |
| **Flatten**   | `ops::flatten(&a, start, end)` | Aplana un rango de dimensiones                         |
| **Concat**    | `ops::concat(&[arrays], axis)` | Concatena múltiples arrays a lo largo de un eje        |
| **Broadcast** | `ops::broadcast_to(&a, shape)` | Expande dimensiones unitarias para coincidir con shape |

---

## 5. Estadística y Probabilidad (`ops::stats`)

| Operación         | Función                                | Descripción                                     |
| ----------------- | -------------------------------------- | ----------------------------------------------- |
| **Softmax**       | `ops::softmax(&a, axis)`               | Normaliza vector a distribución de probabilidad |
| **Cross Entropy** | `ops::cross_entropy(&preds, &targets)` | Loss function para clasificación                |
| **Norm**          | `ops::norm(&a, p)`                     | Norma vectorial (L1, L2, Lp)                    |

---

## 6. Deep Learning Layers (Root Ops)
Operaciones complejas con estado o kernels específicos.

| Operación     | Función                         | Descripción                                       |
| ------------- | ------------------------------- | ------------------------------------------------- |
| **Conv1D**    | `ops::conv1d(...)`              | Convolución 1D (Señales, Texto, Series de Tiempo) |
| **BatchNorm** | `ops::batch_norm(...)`          | Normalización por lotes (Train/Eval modes)        |
| **Dropout**   | `ops::dropout(&a, p, training)` | Aleatoriamente pone ceros con probabilidad `p`    |

---

## 7. Exportación (`ops::export`)
**ONNX Export**: `numrs::ops::export::export_to_onnx(&tensor, path)` permite guardar cualquier grafo computacional ejecutado en `NumRs` a formato estándar ONNX.
