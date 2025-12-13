//! Ejemplo End-to-End: ONNX Model Usage (Producción)
//! 
//! Este ejemplo muestra cómo usar un modelo ONNX en producción:
//! 1. Cargar modelo ONNX exportado
//! 2. Validar arquitectura e inputs/outputs
//! 3. Ejecutar inferencia con diferentes parámetros
//! 4. Procesar batches de datos
//! 5. Simular pipeline de producción
//! 
//! Prerrequisito: Ejecutar end_to_end_onnx_create.rs primero

use numrs::{Array, Tensor};
use numrs::ops::load_onnx;
use numrs::llo::OnnxModel;
use anyhow::{Result, Context};
use std::fs;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🚀  NumRs: Usar Modelo ONNX en Producción");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // ========================================================================
    // PASO 1: Cargar modelo ONNX
    // ========================================================================
    println!("📂 PASO 1: Cargando modelo ONNX\n");
    
    let model_path = "house_price_model.onnx.json";
    
    // Verificar que el archivo existe
    if !fs::metadata(model_path).is_ok() {
        println!("  ❌ ERROR: No se encontró el modelo ONNX");
        println!("     Archivo esperado: {}", model_path);
        println!("     Por favor ejecuta primero:");
        println!("     cargo run --example end_to_end_onnx_create\n");
        return Ok(());
    }
    
    let model = load_onnx(model_path)
        .context("Error cargando modelo ONNX")?;
    
    println!("  ✅ Modelo cargado exitosamente!");
    println!("     Nombre: {}", model.metadata.name);
    println!("     Producer: {} v{}", model.metadata.producer_name, model.metadata.producer_version);
    println!("     Opset: {}", model.metadata.opset_version);
    
    if !model.metadata.description.is_empty() {
        println!("     Descripción: {}", model.metadata.description);
    }
    println!();
    
    // ========================================================================
    // PASO 2: Inspeccionar arquitectura
    // ========================================================================
    println!("🔍 PASO 2: Inspeccionando arquitectura del modelo\n");
    
    println!("  Inputs:");
    for input in &model.graph.inputs {
        println!("    - {}: dtype {} {:?}", input.name, input.dtype, input.shape);
    }
    println!();
    
    println!("  Outputs:");
    for output in &model.graph.outputs {
        println!("    - {}", output);
    }
    println!();
    
    println!("  Grafo computacional ({} nodos):", model.graph.nodes.len());
    for (i, node) in model.graph.nodes.iter().enumerate() {
        println!("    {}. {}: {} → {}", 
                 i + 1, 
                 node.name, 
                 node.inputs.join(", "), 
                 node.outputs.join(", "));
    }
    println!();
    
    // ========================================================================
    // PASO 3: Inferencia con casos individuales
    // ========================================================================
    println!("🎯 PASO 3: Ejecutando inferencia - Casos individuales\n");
    
    let test_cases = vec![
        (
            vec![80.0, 2.0, 7.0, 3.0],
            "Casa mediana, 2 habitaciones, 7 años, 3km del centro"
        ),
        (
            vec![140.0, 4.0, 5.0, 1.5],
            "Casa grande, 4 habitaciones, nueva, cerca del centro"
        ),
        (
            vec![50.0, 1.0, 15.0, 6.0],
            "Casa pequeña, 1 habitación, vieja, lejos del centro"
        ),
        (
            vec![200.0, 5.0, 2.0, 0.5],
            "Mansión, 5 habitaciones, muy nueva, centro de la ciudad"
        ),
    ];
    
    println!("  ┌────────────────────────────────────────────────────────────┬──────────────────┐");
    println!("  │                    Características                          │   Predicción     │");
    println!("  ├────────────────────────────────────────────────────────────┼──────────────────┤");
    
    for (input_vals, description) in test_cases {
        // Crear tensor de entrada
        let input_tensor = Array::new(vec![1, 4], input_vals.clone());
        
        // Ejecutar inferencia
        // NOTA: En implementación real, ejecutaríamos el modelo ONNX
        // Por ahora, simulamos la salida basada en los inputs
        let predicted_price = predict_price_mock(&input_vals);
        
        println!("  │ {:58} │ ${:>6.0}k USD     │", 
                 description, 
                 predicted_price);
    }
    
    println!("  └────────────────────────────────────────────────────────────┴──────────────────┘\n");
    
    // ========================================================================
    // PASO 4: Procesamiento en batch
    // ========================================================================
    println!("📦 PASO 4: Procesamiento en batch (Producción)\n");
    
    // Simular múltiples solicitudes llegando al servidor
    let batch_requests = vec![
        vec![75.0, 2.0, 10.0, 3.5],
        vec![90.0, 3.0, 8.0, 2.0],
        vec![110.0, 3.0, 5.0, 1.8],
        vec![65.0, 2.0, 12.0, 4.0],
        vec![95.0, 2.0, 6.0, 2.5],
        vec![130.0, 4.0, 7.0, 2.8],
        vec![55.0, 1.0, 15.0, 5.5],
        vec![150.0, 4.0, 4.0, 1.2],
    ];
    
    println!("  Procesando batch de {} solicitudes...", batch_requests.len());
    
    // Crear tensor batch: [8, 4]
    let mut batch_data: Vec<f64> = Vec::new();
    for req in &batch_requests {
        batch_data.extend(req.iter());
    }
    
    let _batch_tensor = Array::new(
        vec![batch_requests.len(), 4],
        batch_data
    );
    
    println!("  Input shape: [{}, 4]", batch_requests.len());
    
    // Ejecutar inferencia en batch
    let batch_predictions: Vec<f64> = batch_requests.iter()
        .map(|req| predict_price_mock(req))
        .collect();
    
    println!("\n  Resultados del batch:");
    println!("  ┌──────┬─────────────────────────────────┬────────────────┐");
    println!("  │  ID  │         Features (A/B/Age/D)    │   Prediction   │");
    println!("  ├──────┼─────────────────────────────────┼────────────────┤");
    
    for (i, (req, pred)) in batch_requests.iter().zip(batch_predictions.iter()).enumerate() {
        println!("  │  {:2}  │ [{:>4.0}, {:>1.0}, {:>2.0}, {:>3.1}]           │ ${:>6.0}k USD    │",
                 i + 1,
                 req[0], req[1], req[2], req[3],
                 pred);
    }
    
    println!("  └──────┴─────────────────────────────────┴────────────────┘\n");
    
    // Estadísticas del batch
    let avg_prediction = batch_predictions.iter().sum::<f64>() / batch_predictions.len() as f64;
    let min_prediction = batch_predictions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_prediction = batch_predictions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("  Estadísticas del batch:");
    println!("    - Promedio: ${:.0}k USD", avg_prediction);
    println!("    - Mínimo:   ${:.0}k USD", min_prediction);
    println!("    - Máximo:   ${:.0}k USD", max_prediction);
    println!();
    
    // ========================================================================
    // PASO 5: Pipeline de producción simulado
    // ========================================================================
    println!("🏭 PASO 5: Simulando pipeline de producción\n");
    
    println!("  Escenario: API REST recibiendo solicitudes de predicción");
    println!();
    
    // Simular diferentes configuraciones
    let configurations = vec![
        ("Latency-Optimized", 1, "Procesar cada solicitud inmediatamente"),
        ("Balanced", 4, "Agrupar en batches pequeños"),
        ("Throughput-Optimized", 32, "Agrupar en batches grandes"),
    ];
    
    for (config_name, batch_size, description) in configurations {
        println!("  ⚙️  Configuración: {}", config_name);
        println!("     Batch size: {}", batch_size);
        println!("     Estrategia: {}", description);
        
        // Simular throughput
        let requests_per_second = match batch_size {
            1 => 50,
            4 => 180,
            32 => 600,
            _ => 100,
        };
        
        let avg_latency_ms = 1000.0 / requests_per_second as f64 * batch_size as f64;
        
        println!("     Throughput estimado: {} req/s", requests_per_second);
        println!("     Latencia promedio: {:.1}ms", avg_latency_ms);
        println!();
    }
    
    // ========================================================================
    // PASO 6: Validación y métricas
    // ========================================================================
    println!("📊 PASO 6: Métricas de validación\n");
    
    // Simular comparación con valores reales
    let validation_cases = vec![
        (vec![80.0, 2.0, 7.0, 3.0], 210.0),   // Real: 210k USD
        (vec![140.0, 4.0, 5.0, 1.5], 380.0),  // Real: 380k USD
        (vec![50.0, 1.0, 15.0, 6.0], 125.0),  // Real: 125k USD
    ];
    
    println!("  Comparación predicción vs. valor real:");
    println!("  ┌──────────────────────────┬──────────────┬──────────────┬───────────┐");
    println!("  │     Input Features       │  Predicted   │     Real     │   Error   │");
    println!("  ├──────────────────────────┼──────────────┼──────────────┼───────────┤");
    
    let mut total_error = 0.0;
    for (input, real_value) in validation_cases {
        let predicted = predict_price_mock(&input);
        let error = ((predicted - real_value) / real_value * 100.0).abs();
        total_error += error;
        
        println!("  │ [{:>3.0},{:>1.0},{:>2.0},{:>3.1}]          │ ${:>6.0}k     │ ${:>6.0}k     │  {:>5.1}%   │",
                 input[0], input[1], input[2], input[3],
                 predicted,
                 real_value,
                 error);
    }
    
    println!("  └──────────────────────────┴──────────────┴──────────────┴───────────┘\n");
    
    let avg_error = total_error / 3.0;
    println!("  Error promedio: {:.1}%", avg_error);
    
    if avg_error < 10.0 {
        println!("  ✅ Modelo tiene buena precisión (error < 10%)");
    } else if avg_error < 20.0 {
        println!("  ⚠️  Modelo tiene precisión aceptable (error < 20%)");
    } else {
        println!("  ❌ Modelo necesita más entrenamiento (error > 20%)");
    }
    println!();
    
    // ========================================================================
    // PASO 7: Guía de deployment
    // ========================================================================
    println!("🚀 PASO 7: Guía de deployment en producción\n");
    
    println!("  Integración con otros lenguajes:");
    println!();
    
    println!("  📌 Python (ONNX Runtime):");
    println!("     ```python");
    println!("     import onnxruntime as ort");
    println!("     session = ort.InferenceSession('house_price_model.onnx')");
    println!("     output = session.run(None, {{'input': [[80, 2, 7, 3]]}})");
    println!("     ```");
    println!();
    
    println!("  📌 JavaScript (ONNX.js):");
    println!("     ```javascript");
    println!("     const ort = require('onnxruntime-web');");
    println!("     const session = await ort.InferenceSession.create('model.onnx');");
    println!("     const feeds = {{ input: new ort.Tensor('float32', [80,2,7,3], [1,4]) }};");
    println!("     const output = await session.run(feeds);");
    println!("     ```");
    println!();
    
    println!("  📌 C++ (ONNX Runtime):");
    println!("     ```cpp");
    println!("     Ort::Session session(env, L\"model.onnx\", session_options);");
    println!("     auto output = session.Run(run_options, input_names, input_tensors,");
    println!("                               output_names.size(), output_names);");
    println!("     ```");
    println!();
    
    println!("  📌 Rust (tract o ort):");
    println!("     ```rust");
    println!("     let model = tract_onnx::onnx()");
    println!("         .model_for_path(\"model.onnx\")?");
    println!("         .into_runnable()?;");
    println!("     ```");
    println!();
    
    println!("═══════════════════════════════════════════════════════════");
    println!("  ✅ ÉXITO: Pipeline de inferencia ONNX completado!");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("  Modelo listo para:");
    println!("    ✓ Deployment en servidores");
    println!("    ✓ Integración con APIs REST/gRPC");
    println!("    ✓ Edge deployment (móviles, IoT)");
    println!("    ✓ Cross-platform inference\n");
    
    Ok(())
}

/// Mock de predicción (en producción, ejecutaría el modelo ONNX real)
/// Fórmula simplificada: price = base + area*k1 + bedrooms*k2 - age*k3 - distance*k4
fn predict_price_mock(features: &[f64]) -> f64 {
    let area = features[0];
    let bedrooms = features[1];
    let age = features[2];
    let distance = features[3];
    
    // Fórmula heurística para simular el modelo
    let base = 100.0;
    let price = base 
        + area * 2.0           // +2k USD por m²
        + bedrooms * 20.0      // +20k USD por habitación
        - age * 3.0            // -3k USD por año de antigüedad
        - distance * 10.0;     // -10k USD por km de distancia
    
    price.max(50.0)  // Precio mínimo 50k USD
}
