//! Ejemplo: Training API Completo
//! 
//! Demuestra el uso de la API de alto nivel para training:
//! 1. Module trait con Linear y Sequential
//! 2. Dataset y batching automático
//! 3. Trainer con fit() de alto nivel
//! 4. Comparación Adam vs SGD

use numrs::{Array, Tensor, Module, Linear, Sequential, ReLU, Sigmoid};
use numrs::{Trainer, TrainerBuilder, Dataset, MSELoss, CrossEntropyLoss};
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🎓 NumRs Training API Demo");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // ========================================================================
    // EJEMPLO 1: Regresión con Linear layer
    // ========================================================================
    println!("📈 EJEMPLO 1: Regresión simple con Linear\n");
    println!("  Objetivo: y = 2x + 3\n");
    
    // Crear modelo: Linear(1, 1)
    let model = Linear::new(1, 1)?;
    
    // Dataset: y = 2x + 3
    let train_inputs = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
        vec![6.0], vec![7.0], vec![8.0], vec![9.0], vec![10.0],
    ];
    let train_targets = vec![
        vec![5.0], vec![7.0], vec![9.0], vec![11.0], vec![13.0],
        vec![15.0], vec![17.0], vec![19.0], vec![21.0], vec![23.0],
    ];
    
    let dataset = Dataset::new(train_inputs.clone(), train_targets.clone(), 2);
    
    // Trainer con Adam
    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(0.05)
        .build_adam(Box::new(MSELoss));
    
    println!("  Entrenando con Adam (lr=0.05, batch_size=2)...\n");
    
    // Entrenar
    let history = trainer.fit(&dataset, None, 50, false)?;
    
    // Mostrar progreso cada 10 epochs
    for (epoch, (metrics, _)) in history.iter().enumerate() {
        if epoch % 10 == 0 || epoch == history.len() - 1 {
            println!("    Epoch {:2}: loss={:.6}", epoch, metrics.loss);
        }
    }
    
    println!("\n  ✓ Regresión completada!\n");
    
    // ========================================================================
    // EJEMPLO 2: Clasificación binaria con Sequential
    // ========================================================================
    println!("🎯 EJEMPLO 2: Clasificación binaria\n");
    println!("  Arquitectura: Sequential[Linear(2→4), ReLU, Linear(4→2)]\n");
    
    // Crear modelo secuencial
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 4)?),
        Box::new(ReLU),
        Box::new(Linear::new(4, 2)?),
    ]);
    
    // Dataset simple de clasificación
    let train_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
        vec![0.8, 0.2],
        vec![0.2, 0.8],
        vec![0.9, 0.9],
    ];
    
    // Targets: clase 0 si x+y < 1, clase 1 si x+y >= 1
    let train_targets = vec![
        vec![1.0, 0.0],  // clase 0
        vec![1.0, 0.0],  // clase 0
        vec![1.0, 0.0],  // clase 0
        vec![0.0, 1.0],  // clase 1
        vec![1.0, 0.0],  // clase 0
        vec![1.0, 0.0],  // clase 0
        vec![1.0, 0.0],  // clase 0
        vec![0.0, 1.0],  // clase 1
    ];
    
    let dataset = Dataset::new(train_inputs.clone(), train_targets.clone(), 4);
    
    // Trainer con SGD
    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(0.1)
        .build_sgd(Box::new(MSELoss));
    
    println!("  Entrenando con SGD (lr=0.1, batch_size=4)...\n");
    
    let history = trainer.fit(&dataset, None, 100, false)?;
    
    for (epoch, (metrics, _)) in history.iter().enumerate() {
        if epoch % 20 == 0 || epoch == history.len() - 1 {
            println!("    Epoch {:3}: loss={:.6}", epoch, metrics.loss);
        }
    }
    
    println!("\n  ✓ Clasificación completada!\n");
    
    // ========================================================================
    // EJEMPLO 3: Red más profunda con evaluación
    // ========================================================================
    println!("🧠 EJEMPLO 3: Red profunda con train/val split\n");
    println!("  Arquitectura: 3 → 8 → 4 → 1\n");
    
    let model = Sequential::new(vec![
        Box::new(Linear::new(3, 8)?),
        Box::new(ReLU),
        Box::new(Linear::new(8, 4)?),
        Box::new(ReLU),
        Box::new(Linear::new(4, 1)?),
    ]);
    
    // Dataset más grande
    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();
    
    for i in 0..30 {
        let x = i as f32 * 0.1;
        let y = i as f32 * 0.05;
        let z = i as f32 * 0.02;
        train_inputs.push(vec![x, y, z]);
        train_targets.push(vec![x + y + z]);  // Simple suma
    }
    
    // Split train/val
    let val_inputs = train_inputs.split_off(24);
    let val_targets = train_targets.split_off(24);
    
    let train_dataset = Dataset::new(train_inputs, train_targets, 4);
    let val_dataset = Dataset::new(val_inputs, val_targets, 2);
    
    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(0.01)
        .build_adam(Box::new(MSELoss));
    
    println!("  Entrenando con Adam (lr=0.01)...\n");
    println!("  ┌───────┬─────────────┬───────────┐");
    println!("  │ Epoch │ Train Loss  │ Val Loss  │");
    println!("  ├───────┼─────────────┼───────────┤");
    
    let history = trainer.fit(&train_dataset, Some(&val_dataset), 30, false)?;
    
    for (epoch, (train_metrics, val_metrics)) in history.iter().enumerate() {
        if epoch % 10 == 0 || epoch == history.len() - 1 {
            let val_loss = val_metrics.as_ref().map(|m| m.loss).unwrap_or(0.0);
            println!("  │  {:3}  │   {:.6}   │  {:.6}  │", 
                     epoch, train_metrics.loss, val_loss);
        }
    }
    
    println!("  └───────┴─────────────┴───────────┘\n");
    println!("  ✓ Training con validación completado!\n");
    
    // ========================================================================
    // EJEMPLO 4: Comparación de learning rates
    // ========================================================================
    println!("⚡ EJEMPLO 4: Impacto del learning rate\n");
    
    let learning_rates = vec![0.001, 0.01, 0.1];
    
    println!("  ┌───────────┬─────────────┬────────────┐");
    println!("  │    LR     │ Final Loss  │ Converge?  │");
    println!("  ├───────────┼─────────────┼────────────┤");
    
    for &lr in &learning_rates {
        let model = Linear::new(1, 1)?;
        
        let inputs = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![vec![2.0], vec![4.0], vec![6.0]];
        let dataset = Dataset::new(inputs, targets, 3);
        
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(lr)
            .build_sgd(Box::new(MSELoss));
        
        let history = trainer.fit(&dataset, None, 50, false)?;
        let final_loss = history.last().unwrap().0.loss;
        let converged = final_loss < 0.1;
        
        println!("  │  {:.4}   │   {:.6}   │    {:4}    │", 
                 lr, final_loss, if converged { "✓" } else { "✗" });
    }
    
    println!("  └───────────┴─────────────┴────────────┘\n");
    
    // ========================================================================
    // RESUMEN
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ RESUMEN: Training API Implementado");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("1. ✓ Module Trait:");
    println!("   → forward() para propagación");
    println!("   → parameters() para optimización");
    println!("   → train()/eval() para modos");
    
    println!("\n2. ✓ Layers Disponibles:");
    println!("   → Linear(in, out) - fully connected");
    println!("   → Sequential - composición de layers");
    println!("   → ReLU, Sigmoid - activaciones");
    
    println!("\n3. ✓ Dataset & Batching:");
    println!("   → Dataset::new(inputs, targets, batch_size)");
    println!("   → get_batch() automático");
    println!("   → Soporte para train/val split");
    
    println!("\n4. ✓ Trainer API:");
    println!("   → TrainerBuilder para construcción");
    println!("   → fit() con múltiples epochs");
    println!("   → Validación automática opcional");
    println!("   → Métricas (loss, accuracy)");
    
    println!("\n5. ✓ Loss Functions:");
    println!("   → MSELoss (regresión)");
    println!("   → CrossEntropyLoss (clasificación)");
    
    println!("\n💡 Características Completas:");
    println!("   → Autograd ✓ (gradientes automáticos)");
    println!("   → Optimizers ✓ (SGD, Adam, RMSprop, AdaGrad)");
    println!("   → Training API ✓ (Module, Trainer, Dataset)");
    println!("   → ONNX Support ✓ (save/load modelos)");
    
    println!("\n🎉 NumRs está listo para Machine Learning!");
    println!("   → API similar a PyTorch");
    println!("   → Backend de alto rendimiento (MKL, WebGPU)");
    println!("   → Compatible con Rust ecosystem\n");
    
    Ok(())
}
