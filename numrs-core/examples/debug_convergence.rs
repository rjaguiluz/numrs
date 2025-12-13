//! Test de diagnóstico para convergencia de training
//! 
//! Este test verifica:
//! 1. Que los gradientes se calculen correctamente
//! 2. Que el optimizer actualice los pesos
//! 3. Que la loss disminuya consistentemente

use numrs::{Array, Tensor, Linear, Sequential, ReLU, Module};
use numrs::{TrainerBuilder, Dataset, MSELoss};
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔍 Diagnóstico: Training Convergence");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // ========================================================================
    // TEST 1: Función simple y = 2x + 3
    // ========================================================================
    println!("📊 TEST 1: Regresión lineal simple (y = 2x + 3)\n");
    
    // Dataset simple: y = 2x + 3
    let x_data = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
        vec![6.0], vec![7.0], vec![8.0], vec![9.0], vec![10.0],
    ];
    
    let y_data: Vec<Vec<f32>> = x_data.iter()
        .map(|x| vec![2.0 * x[0] + 3.0])
        .collect();
    
    println!("  Dataset: 10 puntos de y = 2x + 3");
    println!("  X: [1, 2, 3, ..., 10]");
    println!("  Y: [5, 7, 9, ..., 23]");
    println!();
    
    // Modelo simple: Linear(1 -> 1)
    let model = Sequential::new(vec![
        Box::new(Linear::new(1, 1)?),
    ]);
    
    let dataset = Dataset::new(x_data.clone(), y_data.clone(), 10);
    
    println!("  Modelo: Linear(1 → 1)");
    println!("  Optimizer: Adam (lr=0.05)");
    println!("  Epochs: 500");
    println!();
    
    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(0.05)
        .build_adam(Box::new(MSELoss));
    
    // Get initial weights
    let initial_params = trainer.model.parameters();
    let initial_weight = initial_params[0].borrow().data.data[0];
    let initial_bias = initial_params[1].borrow().data.data[0];
    
    println!("  Pesos iniciales:");
    println!("    Weight: {:.4}", initial_weight);
    println!("    Bias:   {:.4}", initial_bias);
    println!();
    
    println!("  Entrenando...");
    let mut prev_loss = f32::MAX;
    let mut loss_increased_count = 0;
    
    for epoch in 0..500 {
        let metrics = trainer.train_epoch(&dataset)?;
        
        if epoch % 50 == 0 || epoch == 499 {
            println!("    Epoch {:3}: loss = {:.6}", epoch + 1, metrics.loss);
        }
        
        // Verificar que la loss disminuye
        if metrics.loss > prev_loss {
            loss_increased_count += 1;
        }
        prev_loss = metrics.loss;
    }
    
    let final_params = trainer.model.parameters();
    let final_weight = final_params[0].borrow().data.data[0];
    let final_bias = final_params[1].borrow().data.data[0];
    
    println!();
    println!("  Pesos finales:");
    println!("    Weight: {:.4} (esperado: 2.0)", final_weight);
    println!("    Bias:   {:.4} (esperado: 3.0)", final_bias);
    println!();
    
    // Validación
    let weight_error = (final_weight - 2.0).abs();
    let bias_error = (final_bias - 3.0).abs();
    
    println!("  📊 Resultados:");
    println!("    Loss final: {:.6}", prev_loss);
    println!("    Loss aumentó {} veces", loss_increased_count);
    println!("    Error en weight: {:.4}", weight_error);
    println!("    Error en bias: {:.4}", bias_error);
    println!();
    
    if prev_loss < 0.1 && weight_error < 0.5 && bias_error < 0.5 {
        println!("  ✅ TEST 1 PASADO: El modelo converge correctamente");
    } else {
        println!("  ❌ TEST 1 FALLIDO: El modelo NO converge");
        println!("     → La loss debería ser < 0.1");
        println!("     → Los pesos deberían estar cerca de [2.0, 3.0]");
    }
    
    println!();
    
    // ========================================================================
    // TEST 2: XOR problem (no lineal)
    // ========================================================================
    println!("📊 TEST 2: Problema XOR (no lineal)\n");
    
    let xor_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    
    let xor_targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];
    
    println!("  Dataset: XOR");
    println!("    [0, 0] → 0");
    println!("    [0, 1] → 1");
    println!("    [1, 0] → 1");
    println!("    [1, 1] → 0");
    println!();
    
    let xor_model = Sequential::new(vec![
        Box::new(Linear::new(2, 8)?),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1)?),
    ]);
    
    println!("  Modelo: 2 → 8 → 1 (con ReLU)");
    println!("  Optimizer: Adam (lr=0.05)");
    println!("  Epochs: 500");
    println!();
    
    let xor_dataset = Dataset::new(xor_inputs.clone(), xor_targets.clone(), 4);
    
    let mut xor_trainer = TrainerBuilder::new(xor_model)
        .learning_rate(0.05)
        .build_adam(Box::new(MSELoss));
    
    println!("  Entrenando...");
    let mut xor_prev_loss = f32::MAX;
    let mut xor_loss_decreased = false;
    
    for epoch in 0..500 {
        let metrics = xor_trainer.train_epoch(&xor_dataset)?;
        
        if epoch % 50 == 0 || epoch == 499 {
            println!("    Epoch {:3}: loss = {:.6}", epoch + 1, metrics.loss);
        }
        
        if metrics.loss < xor_prev_loss * 0.95 {
            xor_loss_decreased = true;
        }
        xor_prev_loss = metrics.loss;
    }
    
    println!();
    println!("  📊 Resultados:");
    println!("    Loss final: {:.6}", xor_prev_loss);
    println!();
    
    if xor_prev_loss < 0.1 && xor_loss_decreased {
        println!("  ✅ TEST 2 PASADO: XOR aprende correctamente");
    } else {
        println!("  ❌ TEST 2 FALLIDO: XOR NO converge");
        println!("     → La loss debería ser < 0.1");
        println!("     → La loss debería disminuir significativamente");
    }
    
    println!();
    
    // ========================================================================
    // TEST 3: Verificar gradientes manualmente
    // ========================================================================
    println!("📊 TEST 3: Verificación manual de gradientes\n");
    
    // Crear un tensor simple
    let x = Tensor::new(Array::new(vec![1, 1], vec![2.0]), true);
    let w = Tensor::new(Array::new(vec![1, 1], vec![3.0]), true);
    
    println!("  x = 2.0 (requires_grad=true)");
    println!("  w = 3.0 (requires_grad=true)");
    println!();
    
    // y = w * x
    let y = w.mul(&x)?;
    
    println!("  y = w * x = 6.0");
    println!("  Esperado: dy/dw = x = 2.0");
    println!("             dy/dx = w = 3.0");
    println!();
    
    // Backward
    y.backward()?;
    
    let x_grad = x.gradient().unwrap();
    let w_grad = w.gradient().unwrap();
    
    println!("  Gradientes calculados:");
    println!("    dy/dx = {:.4} (esperado: 3.0)", x_grad.data[0]);
    println!("    dy/dw = {:.4} (esperado: 2.0)", w_grad.data[0]);
    println!();
    
    let x_grad_correct = (x_grad.data[0] - 3.0).abs() < 0.01;
    let w_grad_correct = (w_grad.data[0] - 2.0).abs() < 0.01;
    
    if x_grad_correct && w_grad_correct {
        println!("  ✅ TEST 3 PASADO: Gradientes correctos");
    } else {
        println!("  ❌ TEST 3 FALLIDO: Gradientes incorrectos");
    }
    
    println!();
    
    // ========================================================================
    // RESUMEN
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════");
    println!("  📋 RESUMEN DEL DIAGNÓSTICO");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("  Si todos los tests pasan:");
    println!("    → El problema está en la escala del dataset");
    println!("    → Probar con normalización de features");
    println!("    → Ajustar learning rate");
    println!();
    println!("  Si TEST 1 falla:");
    println!("    → Problema en optimizer o gradientes básicos");
    println!();
    println!("  Si TEST 2 falla pero TEST 1 pasa:");
    println!("    → Problema en ReLU o composición de layers");
    println!();
    println!("  Si TEST 3 falla:");
    println!("    → Problema fundamental en backpropagation");
    println!();
    
    Ok(())
}
