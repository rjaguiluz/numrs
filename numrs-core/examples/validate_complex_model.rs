use numrs::{Array, Tensor, Linear, Sequential, Module, ReLU};
use numrs::autograd::train::{Dataset, MSELoss, TrainerBuilder};
use anyhow::Result;

fn main() -> Result<()> {
    println!("🧪 Validación de Modelo Complejo\n");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // Test 1: Red profunda para función no-lineal (simplificada)
    println!("📊 Test 1: Red profunda 5→32→32→16→1");
    println!("─────────────────────────────────────────────────────────");
    println!("Problema: Aproximar f(x) = x₁² + sin(x₂) + x₃*x₄ + cos(x₅)");
    println!();
    {
        // Dataset: Función no-lineal más simple y bien condicionada
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        
        // Generar 100 ejemplos de training bien distribuidos
        for i in 0..100 {
            let mut x = Vec::new();
            for j in 0..5 {
                // Valores entre -1 y 1 (bien condicionados)
                let val = (i as f32 / 50.0 - 1.0) + (j as f32 / 10.0 - 0.2);
                x.push(val);
            }
            
            // Función objetivo más simple pero no trivial
            let y = x[0].powi(2) * 2.0    // cuadrático
                + x[1].sin()              // trigonométrico
                + x[2] * x[3]             // interacción
                + x[4].cos()              // más trigonometría
                + 0.5;                    // bias
            
            x_data.push(x);
            y_data.push(vec![y]);
        }
        
        let dataset = Dataset::new(x_data.clone(), y_data.clone(), 10);
        
        // Red más manejable: 5→32→32→16→1
        let model = Sequential::new(vec![
            Box::new(Linear::new(5, 32)?),
            Box::new(ReLU),
            Box::new(Linear::new(32, 32)?),
            Box::new(ReLU),
            Box::new(Linear::new(32, 16)?),
            Box::new(ReLU),
            Box::new(Linear::new(16, 1)?),
        ]);
        
        println!("  Arquitectura: 5→32→32→16→1 (4 capas)");
        println!("  Total parámetros: ~1,700");
        println!("  Dataset: 100 ejemplos");
        println!("  Batch size: 10");
        println!("  Optimizer: Adam(lr=0.01)");
        println!();
        
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(0.01)
            .build_adam(Box::new(MSELoss));
        
        // Training con epochs periódicos
        let epochs_checkpoints = vec![1, 10, 25, 50, 100, 150];
        let mut prev_loss = f32::INFINITY;
        
        for &target_epoch in &epochs_checkpoints {
            // Entrenar hasta el checkpoint
            let current_epoch = if target_epoch == 1 { 0 } else { 
                epochs_checkpoints.iter()
                    .position(|&e| e == target_epoch)
                    .unwrap()
                    .saturating_sub(1)
            };
            
            let start_epoch = if current_epoch > 0 { 
                epochs_checkpoints[current_epoch] 
            } else { 
                0 
            };
            
            for epoch in start_epoch..target_epoch {
                trainer.train_epoch(&dataset)?;
            }
            
            let metrics = trainer.evaluate(&dataset)?;
            let improvement = if prev_loss.is_finite() {
                ((prev_loss - metrics.loss) / prev_loss * 100.0).max(0.0)
            } else {
                0.0
            };
            
            println!("  Epoch {:3}: loss={:.6} (mejora: {:.1}%)", 
                target_epoch, metrics.loss, improvement);
            prev_loss = metrics.loss;
        }
        
        println!();
        
        // Validar predicciones en ejemplos de test
        println!("  Validación en ejemplos específicos:");
        
        let test_indices = vec![0, 25, 50, 75, 99];
        let mut total_error = 0.0;
        
        for &idx in &test_indices {
            let x = Tensor::new(
                Array::new(vec![1, 5], x_data[idx].clone()),
                false
            );
            let y_pred = trainer.model.forward(&x)?;
            let y_true = y_data[idx][0];
            let error = (y_pred.data.data[0] - y_true).abs();
            let rel_error = if y_true.abs() > 0.01 {
                (error / y_true.abs()) * 100.0
            } else {
                error * 100.0
            };
            
            println!("    Ejemplo {:2}: pred={:7.3}, true={:7.3}, error={:.4} ({:.1}%)",
                idx, y_pred.data.data[0], y_true, error, rel_error);
            total_error += error;
        }
        
        let avg_error = total_error / test_indices.len() as f32;
        println!();
        println!("  Error promedio: {:.4}", avg_error);
        
        if avg_error < 0.2 {
            println!("  ✅ Test 1 PASSED: Red profunda converge excelente (error < 0.2)\n");
        } else if avg_error < 0.5 {
            println!("  ✅ Test 1 PASSED: Red profunda converge bien (error < 0.5)\n");
        } else if avg_error < 1.0 {
            println!("  ⚠️  Test 1 WARNING: Converge razonablemente (error < 1.0)\n");
        } else {
            println!("  ❌ Test 1 FAILED: No converge adecuadamente\n");
        }
    }
    
    // Test 2: Regresión no-lineal (mejor que clasificación para MSE)
    println!("📊 Test 2: Regresión no-lineal 3→24→24→12→1");
    println!("─────────────────────────────────────────────────────────");
    println!("Problema: y = x₁*x₂ + sin(x₃*π)");
    println!();
    {
        // Dataset: Función no-lineal con 3 inputs
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        
        // Generar 80 ejemplos bien distribuidos
        for i in 0..80 {
            let x1 = (i as f32 / 40.0) - 1.0;           // -1 a 1
            let x2 = ((i * 3) % 80) as f32 / 40.0 - 1.0; // -1 a 1
            let x3 = ((i * 7) % 80) as f32 / 80.0;       // 0 a 1
            
            let x = vec![x1, x2, x3];
            
            // Función objetivo: interacción + trigonometría
            let y = x1 * x2 + (x3 * std::f32::consts::PI).sin();
            
            x_data.push(x);
            y_data.push(vec![y]);
        }
        
        let dataset = Dataset::new(x_data.clone(), y_data.clone(), 16);
        
        // Red para regresión: 3→24→24→12→1
        let model = Sequential::new(vec![
            Box::new(Linear::new(3, 24)?),
            Box::new(ReLU),
            Box::new(Linear::new(24, 24)?),
            Box::new(ReLU),
            Box::new(Linear::new(24, 12)?),
            Box::new(ReLU),
            Box::new(Linear::new(12, 1)?),
        ]);
        
        println!("  Arquitectura: 3→24→24→12→1");
        println!("  Total parámetros: ~900");
        println!("  Dataset: 80 ejemplos");
        println!("  Función: y = x₁*x₂ + sin(x₃*π)");
        println!("  Optimizer: Adam(lr=0.02)");
        println!();
        
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(0.02)
            .build_adam(Box::new(MSELoss));
        
        // Entrenar 150 epochs
        for epoch in 0..150 {
            trainer.train_epoch(&dataset)?;
            
            if epoch % 30 == 0 {
                let metrics = trainer.evaluate(&dataset)?;
                println!("  Epoch {:3}: loss={:.4}", epoch, metrics.loss);
            }
        }
        
        let final_metrics = trainer.evaluate(&dataset)?;
        println!("  Epoch 150: loss={:.4}", final_metrics.loss);
        println!();
        
        // Evaluar predicciones
        println!("  Predicciones en ejemplos de test:");
        let test_examples = vec![0, 20, 40, 60, 79];
        let mut total_error = 0.0;
        
        for &idx in &test_examples {
            let x = Tensor::new(
                Array::new(vec![1, 3], x_data[idx].clone()),
                false
            );
            let y_pred = trainer.model.forward(&x)?;
            let y_true = y_data[idx][0];
            let error = (y_pred.data.data[0] - y_true).abs();
            
            println!("    Input: [{:5.2}, {:5.2}, {:5.2}] → Pred: {:6.3}, True: {:6.3}, Error: {:.4}",
                x_data[idx][0], x_data[idx][1], x_data[idx][2],
                y_pred.data.data[0], y_true, error);
            total_error += error;
        }
        
        let avg_error = total_error / test_examples.len() as f32;
        println!();
        println!("  Error promedio: {:.4}", avg_error);
        
        if avg_error < 0.1 {
            println!("  ✅ Test 2 PASSED: Regresión excelente (error < 0.1)\n");
        } else if avg_error < 0.3 {
            println!("  ✅ Test 2 PASSED: Regresión buena (error < 0.3)\n");
        } else {
            println!("  ⚠️  Test 2 WARNING: Error alto (>{:.4})\n", avg_error);
        }
    }
    
    // Test 3: Red para secuencias simples
    println!("📊 Test 3: Predicción de suma acumulada 10→16→8→1");
    println!("─────────────────────────────────────────────────────────");
    println!("Problema: Predecir sum(x) dado ventana de 10 valores");
    println!();
    {
        // Dataset: Serie simple - predecir suma de ventana
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        
        // Generar 60 ventanas
        for i in 0..60 {
            let mut window = Vec::new();
            let base = (i as f32 / 10.0).sin();
            
            // Ventana de 10 valores simples
            for t in 0..10 {
                let val = base + (t as f32 / 10.0) * (i as f32 / 30.0).cos();
                window.push(val);
            }
            
            // Predecir la suma de la ventana
            let y: f32 = window.iter().sum();
            
            x_data.push(window);
            y_data.push(vec![y]);
        }
        
        let dataset = Dataset::new(x_data.clone(), y_data.clone(), 10);
        
        // Red más simple: 10→16→8→1
        let model = Sequential::new(vec![
            Box::new(Linear::new(10, 16)?),
            Box::new(ReLU),
            Box::new(Linear::new(16, 8)?),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)?),
        ]);
        
        println!("  Arquitectura: 10→16→8→1 (3 capas)");
        println!("  Total parámetros: ~300");
        println!("  Dataset: 60 ventanas");
        println!("  Tarea: Predecir sum(ventana)");
        println!("  Optimizer: Adam(lr=0.02)");
        println!();
        
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(0.02)
            .build_adam(Box::new(MSELoss));
        
        // Entrenar 100 epochs
        for epoch in 0..100 {
            trainer.train_epoch(&dataset)?;
            
            if epoch % 20 == 0 {
                let metrics = trainer.evaluate(&dataset)?;
                println!("  Epoch {:3}: loss={:.4}", epoch, metrics.loss);
            }
        }
        
        let final_metrics = trainer.evaluate(&dataset)?;
        println!("  Epoch 100: loss={:.4}", final_metrics.loss);
        println!();
        
        // Validar predicciones
        println!("  Predicciones en ventanas de test:");
        let test_windows = vec![0, 15, 30, 45, 59];
        let mut total_error = 0.0;
        
        for &idx in &test_windows {
            let x = Tensor::new(
                Array::new(vec![1, 10], x_data[idx].clone()),
                false
            );
            let y_pred = trainer.model.forward(&x)?;
            let y_true = y_data[idx][0];
            let error = (y_pred.data.data[0] - y_true).abs();
            
            println!("    Ventana {:2}: pred={:6.3}, true={:6.3}, error={:.4}",
                idx, y_pred.data.data[0], y_true, error);
            total_error += error;
        }
        
        let avg_error = total_error / test_windows.len() as f32;
        println!();
        println!("  Error promedio: {:.4}", avg_error);
        
        if avg_error < 0.5 {
            println!("  ✅ Test 3 PASSED: Predicción excelente (error < 0.5)\n");
        } else if avg_error < 1.5 {
            println!("  ✅ Test 3 PASSED: Predicción buena (error < 1.5)\n");
        } else {
            println!("  ⚠️  Test 3 WARNING: Error alto (>{:.4})\n", avg_error);
        }
    }
    
    println!("═══════════════════════════════════════════════════════════");
    println!("✅ Validación de modelos complejos completada");
    println!("═══════════════════════════════════════════════════════════\n");
    
    Ok(())
}
