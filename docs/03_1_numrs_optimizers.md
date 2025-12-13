# Tutorial 3.1: Optimizadores en NumRs

## Introducción
El optimizador es el algoritmo que actualiza los pesos del modelo basándose en los gradientes calculados. NumRs posee una colección impresionantemente rica de optimizadores, cubriendo desde los clásicos hasta el estado del arte.

Para usar uno, simplemente configúralo en el `TrainerBuilder`:
```rust
let trainer = TrainerBuilder::new(model)
    .build_adam(loss); // O build_sgd, build_rmsprop, etc.
```

O instálalo manualmente si necesitas configuración fina:
```rust
use numrs::autograd::optim::{AdamW, Optimizer};

let mut optim = AdamW::new(model.parameters(), 0.001)
    .weight_decay(0.01)
    .build();
```

## Lista de Optimizadores

### 1. Estándar y Clásicos
Son los caballos de batalla del Deep Learning.

| Optimizador  | Uso Recomendado         | Características                                                                                    |
| ------------ | ----------------------- | -------------------------------------------------------------------------------------------------- |
| **SGD**      | **Fine-tuning**, Vision | Stochastic Gradient Descent. Simple, robusto, pero puede ser lento en converger. Soporta Momentum. |
| **RMSProp**  | **RNNs**, RL            | Adaptive learning rate. Bueno para series de tiempo y objetivos no estacionarios.                  |
| **Adagrad**  | **Sparse Data** (NLP)   | Adapta el learning rate por parámetro basándose en la frecuencia de actualización.                 |
| **Adadelta** | **No requiere LR**      | Extensión de Adagrad que reduce la agresividad de la caída del learning rate.                      |

### 2. Familia Adam (Adaptativos Modernos)
La opción por defecto para la mayoría de tareas.

| Optimizador | Uso Recomendado         | Características                                                                                           |
| ----------- | ----------------------- | --------------------------------------------------------------------------------------------------------- |
| **Adam**    | **General Purpose**     | Combina Momentum y RMSProp. Converge rápido. El estándar de facto.                                        |
| **AdamW**   | **Transformers**, LLMs  | Adam con *Weight Decay desacoplado*. Crucial para generalización en modelos grandes.                      |
| **NAdam**   | General                 | Adam con Nesterov momentum. A veces converge ligeramente mejor que Adam.                                  |
| **RAdam**   | Entrenamiento inestable | Rectified Adam. Estabiliza la varianza del learning rate al inicio del entrenamiento (warmup automático). |

### 3. Especializados y Avanzados

| Optimizador   | Uso Recomendado             | Características                                                                                                                                                  |
| ------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LAMB**      | **Large Batch Training**    | Layer-wise Adaptive Moments for BERT. Permite entrenar con batches gigantes sin perder precisión.                                                                |
| **LBFGS**     | **Small Datasets**, Physics | Algoritmo Quasi-Newton. Converge en *muchos menos pasos* pero requiere mucha memoria. Ideal para problemas “suaves” o simulaciones físicas.                      |
| **Rprop**     | **Full-Batch**              | Resilient Backpropagation. Solo usa el signo del gradiente. Muy rápido pero **no funciona con mini-batches**.                                                    |
| **AdaBound**  | Convergencia final          | Empieza como Adam (rápido) y termina como SGD (estable).                                                                                                         |
| **Lookahead** | Wrapper                     | "Optimiza el optimizador". Mantiene pesos 'rápidos' y 'lentos', mejorando la estabilidad. Se puede envolver alrededor de cualquier otro (e.g., Lookahead(Adam)). |

## ¿Cuál elegir?
- **Regla de oro**: Empieza con `Adam` (lr=1e-3).
- **Si usas Transformers/Wav2Vec**: Usa `AdamW`.
- **Si entrenas con batches enormes (> 1024)**: Usa `LAMB`.
- **Si tu dataset es pequeño (< 5000) o es un problema de física**: Prueba `LBFGS`.
- **Si buscas el último 0.1% de precisión**: Prueba `SGD` con Momentum al final del entrenamiento.
