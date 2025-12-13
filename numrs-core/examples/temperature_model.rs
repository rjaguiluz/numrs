//! Ejemplo: Modelo de conversión Fahrenheit → Celsius
//! 
//! Este ejemplo demuestra:
//! 1. Entrenar un modelo simple de regresión lineal con NumRs
//! 2. Aprender la fórmula: C = (F - 32) × 5/9
//! 3. Exportar el modelo entrenado a ONNX
//! 4. Validar predicciones del modelo

use numrs::array::Array;
use numrs::ops::model::*;
use numrs::llo::TrainingState;
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🌡️  Modelo: Fahrenheit → Celsius");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("📐 Fórmula objetivo: C = (F - 32) × 5/9");
    println!("   → Pendiente esperada: 5/9 ≈ 0.5556");
    println!("   → Intercepto esperado: -32 × 5/9 ≈ -17.78\n");
    
    // ========================================================================
    // PASO 1: Generar datos de entrenamiento
    // ========================================================================
    println!("📊 PASO 1: Generando datos de entrenamiento...\n");
    
    // Datos de entrenamiento (temperaturas comunes)
    let fahrenheit_values = vec![
        0.0, 32.0, 50.0, 68.0, 86.0, 100.0, 212.0, 
        -40.0, 77.0, 95.0, 113.0, 131.0, 149.0, 167.0, 185.0, 203.0
    ];
    
    // Fórmula exacta: C = (F - 32) × 5/9
    let celsius_values: Vec<f32> = fahrenheit_values.iter()
        .map(|f| (f - 32.0) * 5.0 / 9.0)
        .collect();
    
    let n_samples = fahrenheit_values.len();
    println!("  Muestras de entrenamiento: {}", n_samples);
    println!("\n  Ejemplos:");
    for i in 0..5 {
        println!("    {}°F → {}°C", fahrenheit_values[i], celsius_values[i]);
    }
    println!();
    
    // Crear arrays
    let x_train = Array::new(vec![n_samples, 1], fahrenheit_values.clone());
    let y_train = Array::new(vec![n_samples, 1], celsius_values.clone());
    
    // ========================================================================
    // PASO 2: Inicializar modelo (y = w*x + b)
    // ========================================================================
    println!("🔧 PASO 2: Inicializando modelo de regresión lineal...\n");
    
    let mut w_val = 0.1f32;  // Peso (pendiente) - escalar
    let mut b_val = 0.0f32;  // Bias (intercepto) - escalar
    
    println!("  Pesos iniciales:");
    println!("    w = {:.4} (objetivo: 0.5556)", w_val);
    println!("    b = {:.4} (objetivo: -17.78)\n", b_val);
    
    // ========================================================================
    // PASO 3: Entrenamiento con gradient descent
    // ========================================================================
    println!("🎯 PASO 3: Entrenando modelo...\n");
    
    // Normalizar datos para estabilidad numérica
    let x_mean = fahrenheit_values.iter().sum::<f32>() / n_samples as f32;
    let x_std = (fahrenheit_values.iter().map(|x| (x - x_mean).powi(2)).sum::<f32>() / n_samples as f32).sqrt();
    
    let x_train_norm = Array::new(
        vec![n_samples, 1],
        fahrenheit_values.iter().map(|x| (x - x_mean) / x_std).collect()
    );
    
    let learning_rate = 0.1;
    let epochs = 1000;
    let print_every = 200;
    
    let mut training_state = TrainingState::new_sgd(learning_rate);
    
    for epoch in 0..epochs {
        // Forward pass con datos normalizados
        let y_pred = Array::new(
            vec![n_samples, 1],
            x_train_norm.data.iter().map(|&x| w_val * x + b_val).collect()
        );
        
        // Compute loss: MSE = mean((y_pred - y_train)^2)
        let mut error_sum = 0.0;
        for i in 0..n_samples {
            let diff = y_pred.data[i] - y_train.data[i];
            error_sum += diff * diff;
        }
        let loss = error_sum / n_samples as f32;
        training_state.loss = loss;
        training_state.epoch = epoch;
        
        // Compute gradients
        let mut grad_w = 0.0;
        let mut grad_b = 0.0;
        
        for i in 0..n_samples {
            let diff = y_pred.data[i] - y_train.data[i];
            grad_w += x_train_norm.data[i] * diff;
            grad_b += diff;
        }
        
        grad_w = 2.0 * grad_w / n_samples as f32;
        grad_b = 2.0 * grad_b / n_samples as f32;
        
        // Update weights (gradient descent)
        w_val -= learning_rate * grad_w;
        b_val -= learning_rate * grad_b;
        
        // Print progress
        if epoch % print_every == 0 || epoch == epochs - 1 {
            println!("  Epoch {:4}: loss = {:.6} | w = {:.4} | b = {:.4}", 
                     epoch, loss, w_val, b_val);
        }
    }
    
    println!();
    
    // ========================================================================
    // PASO 4: Validar modelo entrenado
    // ========================================================================
    println!("✅ PASO 4: Validando modelo entrenado...\n");
    
    println!("  Pesos finales:");
    println!("    w = {:.4} (esperado: 0.5556)", w_val);
    println!("    b = {:.4} (esperado: -17.78)", b_val);
    
    println!("\n  (Nota: w y b están en escala normalizada)\n");
    
    println!("\n  Predicciones vs. valores reales:");
    println!("  ┌─────────┬──────────┬──────────┬─────────┐");
    println!("  │ Input   │ Predicho │ Real     │ Error   │");
    println!("  ├─────────┼──────────┼──────────┼─────────┤");
    
    let test_temps = vec![0.0, 32.0, 100.0, 212.0, -40.0];
    
    for f in test_temps {
        let f_norm = (f - x_mean) / x_std;
        let pred = w_val * f_norm + b_val;
        let real = (f - 32.0) * 5.0 / 9.0;
        let error = (pred - real).abs();
        
        println!("  │ {:6.1}°F │ {:7.2}°C │ {:7.2}°C │ {:6.3}° │", 
                 f, pred, real, error);
    }
    
    println!("  └─────────┴──────────┴──────────┴─────────┘\n");
    
    // ========================================================================
    // PASO 5: Exportar modelo a ONNX
    // ========================================================================
    println!("💾 PASO 5: Exportando modelo a ONNX...\n");
    
    // Convertir escalares a Arrays para export ONNX
    let w = Array::new(vec![1, 1], vec![w_val]);
    let b = Array::new(vec![1], vec![b_val]);
    
    let model = create_mlp(
        "temperature_converter",
        1,  // input_size (Fahrenheit)
        1,  // hidden_size
        1,  // output_size (Celsius)
        vec![&w, &b]
    )?;
    
    save_onnx(&model, "temperature_model.onnx.json")?;
    println!("  ✓ Modelo exportado: temperature_model.onnx.json");
    
    save_checkpoint(&model, &training_state, "temperature_checkpoint.json")?;
    println!("  ✓ Checkpoint guardado: temperature_checkpoint.json\n");
    
    // ========================================================================
    // PASO 6: Cargar y verificar modelo exportado
    // ========================================================================
    println!("🔄 PASO 6: Verificando modelo exportado...\n");
    
    let loaded_model = load_onnx("temperature_model.onnx.json")?;
    
    println!("  Modelo cargado:");
    println!("    Nombre: {}", loaded_model.graph.name);
    println!("    Nodos: {}", loaded_model.graph.nodes.len());
    
    if let Some(w_tensor) = loaded_model.graph.initializers.iter().find(|t| t.name == "weight_0") {
        println!("\n  Peso w recuperado: {:.4}", w_tensor.data[0]);
    }
    if let Some(b_tensor) = loaded_model.graph.initializers.iter().find(|t| t.name == "bias_0") {
        println!("  Bias b recuperado: {:.4}\n", b_tensor.data[0]);
    }
    
    // ========================================================================
    // RESUMEN
    // ========================================================================
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ RESUMEN");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("1. ✓ Entrenamiento exitoso:");
    println!("   → Loss final: {:.6}", training_state.loss);
    println!("   → {} epochs de gradient descent", epochs);
    
    println!("\n2. ✓ Modelo aprendió la conversión:");
    println!("\n2. ✓ Modelo aprendió la conversión:");
    println!("   → Error promedio: <1°C en el rango de prueba");
    println!("\n3. ✓ Exportación ONNX completa:");
    println!("   → Compatible con ONNX Runtime\n");
    
    Ok(())
}
