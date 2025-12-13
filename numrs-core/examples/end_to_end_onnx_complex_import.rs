//! Ejemplo End-to-End: Importando y Ejecutando Modelo ONNX Exportado
//! 
//! Este ejemplo demuestra el flujo inverso: Consumir un modelo ONNX generado automáticamente.
//! 
//! Pasos:
//! 1. Cargar `industrial_model_auto.onnx.json` del disco.
//! 2. Inspeccionar la metadata del modelo (inputs/outputs).
//! 3. Generar datos de prueba (mismo generador que en training).
//! 4. Ejecutar inferencia usando el motor ONNX de NumRs.
//! 5. Validar predicciones.

use numrs::Array;
use numrs::ops::model::{load_onnx, infer};
use anyhow::{Result, anyhow};
use std::collections::HashMap;

/// Genera datos sintéticos (Copia idéntica del proceso de enternamiento)
fn generate_synthetic_validation_data(num_samples: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut data = Vec::with_capacity(num_samples);
    let mut classes = Vec::with_capacity(num_samples);
    
    // Seed diferente para probar generalización
    let mut seed: u64 = 99999;
    let mut rng = |min: f32, max: f32| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (seed >> 33) as f32 / 2147483648.0; 
        min + val * (max - min)
    };

    for i in 0..num_samples {
        let label = i % 3;
        let mut sensors = vec![0.0; 10];
        
        match label {
            0 => { // Normal
                for s in sensors.iter_mut() { *s = rng(0.0, 0.4); }
            },
            1 => { // Warning
                for (idx, s) in sensors.iter_mut().enumerate() {
                    if idx % 2 == 0 { *s = rng(0.5, 0.8); } 
                    else { *s = rng(0.2, 0.5); }
                }
            },
            2 => { // Critical
                for (idx, s) in sensors.iter_mut().enumerate() {
                    if idx >= 8 { *s = rng(0.8, 1.0); }
                    else { *s = rng(0.4, 0.7); }
                }
            },
            _ => unreachable!()
        }
        
        data.push(sensors);
        classes.push(label);
    }
    
    (data, classes)
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🏭  NumRs: Inferencia con Modelo Industrial ONNX");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // 1. Cargar Modelo
    let model_path = "industrial_model_auto.onnx.json";
    println!("📂 Cargando modelo desde '{}'...", model_path);
    
    if !std::path::Path::new(model_path).exists() {
        return Err(anyhow!("El archivo de modelo no existe. Ejecuta primero 'cargo run --example end_to_end_onnx_complex'"));
    }
    
    let model = load_onnx(model_path)?;
    println!("   ✅ Modelo cargado correctamente.");
    println!("      Nombre: {}", model.metadata.producer_name);
    println!("      Graph Inputs:  {:?}", model.graph.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
    println!("      Graph Outputs: {:?}", model.graph.outputs);
    println!("      Initializers:  {} (Pesos/Bias)", model.graph.initializers.len());
    println!("      Nodos (Ops):   {}\n", model.graph.nodes.len());
    
    // Validar estructura básica
    if model.graph.inputs.is_empty() {
        return Err(anyhow!("El modelo no tiene inputs definidos"));
    }
    let input_name = &model.graph.inputs[0].name;
    let output_name = &model.graph.outputs[0]; // Assuming single output
    
    // 2. Generar Datos de Prueba
    println!("📊 Generando datos de validación...");
    let (test_data, test_labels) = generate_synthetic_validation_data(5);
    
    println!("   Validando 5 muestras...\n");
    
    // 3. Inferencia Loop
    println!("   ┌──────────────────────────────────┬────────────┬────────────┬────────┐");
    println!("   │ Input (Resumen Sensores)         │  Esperado  │ Predicción │ Estado │");
    println!("   ├──────────────────────────────────┼────────────┼────────────┼────────┤");
    
    let mut correct = 0;
    
    for (i, (sensors, label_idx)) in test_data.iter().zip(test_labels.iter()).enumerate() {
        // Preparar Input Array [1, 10]
        let input_array = Array::new(vec![1, 10], sensors.clone());
        
        // Crear mapa de inputs
        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), input_array);
        
        // Ejecutar Inferencia
        let outputs = infer(&model, inputs)?;
        
        // Obtener resultado
        let result = outputs.get(output_name)
            .ok_or_else(|| anyhow!("Output '{}' no encontrado en resultados", output_name))?;
            
        // Procesar Logits -> Clase
        // result es [1, 3] (logits o probabilidades dependiendo si softmax está incluido en el export)
        // El export automático incluía Softmax? 
        // Revisando export.rs: Softmax NO estaba en el Sequential del ejemplo original (solo Linear+ReLU+Linear).
        // La Loss (CrossEntropy) aplicaba softmax internamente.
        // PERO Trainer.evaluate usaba argmax directament de logits.
        // IMPORTANTE: Si el modelo exportado es SOLO lo que estaba en Sequential, son LOGITS puros.
        // Argmax funciona igual sobre logits que sobre probs.
        
        let logits = &result.data; // Vec<f32>
        
        let pred_idx = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
            
        // Validar
        let is_ok = pred_idx == *label_idx;
        if is_ok { correct += 1; }
        
        let status_icon = if is_ok { "✅" } else { "❌" };
        let class_names = ["Normal", "Warning", "Critical"];
        
        let avg_sensor: f32 = sensors.iter().sum::<f32>() / 10.0;
        let s0 = sensors[0];
        let s9 = sensors[9];
        
        println!("   │ Avg={:.2}, S0={:.2}, S9={:.2}      │ {:<10} │ {:<10} │   {}   │", 
            avg_sensor, s0, s9, 
            class_names[*label_idx], 
            class_names[pred_idx], 
            status_icon
        );
    }
    
    println!("   └──────────────────────────────────┴────────────┴────────────┴────────┘");
    println!("\n   Resultado: {}/{} Correctos", correct, test_data.len());
    
    if correct >= 4 {
        println!("   🚀 ÉXITO: El modelo importado funciona correctamente.");
    } else {
        println!("   ⚠️  ADVERTENCIA: Baja precisión. Verificar consistencia del modelo.");
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    Ok(())
}
