# Tutorial 1: NumRs Arrays para Ciencia de Datos

## Introducción
`NumRs` es una biblioteca de computación numérica de alto rendimiento para Rust, inspirada en NumPy. Su núcleo es el struct `Array`, que permite operaciones n-dimensionales eficientes en CPU (vía SIMD/Apple Accelerate) y GPU (vía WebGPU/Metal).

## Introducción
`NumRs` es una biblioteca de computación numérica de alto rendimiento para Rust, inspirada en NumPy. Su núcleo es el struct `Array`, que permite operaciones n-dimensionales eficientes.

### 🆚 PyTorch vs NumRs: Arrays
| Concepto        | PyTorch (Python)             | NumRs (Rust)                                |
| --------------- | ---------------------------- | ------------------------------------------- |
| **Crear Array** | `x = torch.tensor([1., 2.])` | `let x = Array::new(vec![2], vec![1., 2.])` |
| **Shape**       | `x.shape`                    | `x.shape` (Vec<usize>)                      |
| **Zeros**       | `torch.zeros(2, 3)`          | `Array::zeros(vec![2, 3])`                  |
| **Suma**        | `x + y`                      | `ops::add(&x, &y)`                          |

## 1. Creación de Arrays

La creación de arrays es directa. Utilizamos `Array::new` pasando la forma (shape) y los datos linealizados.

```rust
use numrs::Array;

fn main() -> anyhow::Result<()> {
    // Crear un array 2x3
    let a = Array::new(
        vec![2, 3],           // Shape: 2 filas, 3 columnas
        vec![1.0, 2.0, 3.0,   // Datos fila 1
             4.0, 5.0, 6.0]   // Datos fila 2
    );
    
    // Helpers comunes
    let zeros = Array::zeros(vec![10, 10]); // Matriz 10x10 de ceros
    let ones = Array::ones(vec![5]);        // Vector de 5 unos
    
    println!("Array A:\n{:?}", a);
    Ok(())
}
```

## 2. Operaciones Matemáticas Básicas

`NumRs` soporta aritmética elemento a elemento (element-wise) y difusión (broadcasting) limitada.

```rust
use numrs::ops::{add, sub, mul, div};

// Suma elemento a elemento
let b = Array::ones(vec![2, 3]);
let suma = add(&a, &b)?; // a + 1

// Multiplicación escalar (broadcasting simple)
// Nota: Actualmente requiere crear un array escalar o del mismo tamaño
let factor = Array::new(vec![1], vec![2.0]);
// let doble = mul(&a, &factor)?; // (Futura implementación de broadcasting completo)
```

## 3. Álgebra Lineal

La operación más importante para ciencia de datos y ML es la multiplicación de matrices.

```rust
use numrs::ops::matmul;

// A: [2, 3]
// B: [3, 2]
let b = Array::new(vec![3, 2], vec![
    1.0, 4.0,
    2.0, 5.0,
    3.0, 6.0
]);

// C = A @ B -> [2, 2]
let c = matmul(&a, &b)?;
```

## 4. Manipulación de Formas (Reshape & Transpose)

Cambiar la forma de los datos sin copiar memoria (Zero-Copy views en el futuro, actualmente operaciones eficientes).

```rust
use numrs::ops::{reshape, transpose};

// Aplanar a vector [6]
let flat = reshape(&a, &[6])?;

// Transponer [2, 3] -> [3, 2]
let transpuesta = transpose(&a, None)?;
```

## 5. Reducciones (Aggregation)

Operaciones que reducen dimensiones, fundamentales para estadística.

```rust
use numrs::ops::{sum, mean};

let total = sum(&a, None)?; // Suma todo -> escalar
let promedio = mean(&a, None)?; // Promedio todo
```

## 6. Estadística y Probabilidad
Operaciones comunes en ML.

```rust
use numrs::ops::{softmax, norm};

// Softmax (probabilidad)
// input: [2.0, 1.0, 0.1] -> output: [0.65, 0.24, 0.11]
let probs = softmax(&a, 1)?; 

// Norma L2 (Euclidiana)
let l2 = norm(&a, 2.0)?;
```

## 7. Caso de Uso Real: Pre-procesamiento de Imágenes
En Computer Vision, las imágenes se cargan como arrays [Height, Width, Channels] (HWC) pero los modelos suelen esperar [Access, Channels, Height, Width] (NCHW) normalizado.

```rust
// Simulación: Imagen 224x224 RGB (uint8 scale 0-255)
let raw_pixels = vec![255.0; 224 * 224 * 3]; 
let image = Array::new(vec![224, 224, 3], raw_pixels);

// 1. Normalizar a [0, 1]
let factor = Array::new(vec![1], vec![255.0]);
// let normalized = div(&image, &factor)?; // (Broadcasting futuro)

// 2. Transponer HWC -> CHW
// Permutamos ejes: (0, 1, 2) -> (2, 0, 1)
let chw = numrs::ops::transpose(&image, Some(vec![2, 0, 1]))?;

// 3. Añadir dimensión de Batch -> NCHW
let batch_img = numrs::ops::reshape(&chw, &[1, 3, 224, 224])?;

println!("Input listo para CNN: {:?}", batch_img.shape()); // [1, 3, 224, 224]
```

## Próximos Pasos
Ahora que dominas los `Arrays`, pasa al siguiente tutorial para aprender sobre `Tensors` y **Deep Learning**.
