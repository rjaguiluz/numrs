use numrs::{Array, Tensor};
use numrs::ops::model::{load_onnx, infer};
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use rand::prelude::*;
use rand::rngs::StdRng;

// ========================================================================
// REUSE: Data Generation Logic (Must match timeseries_cnn.rs EXACTLY)
// ========================================================================
fn generate_synthetic_timeseries(num_samples: usize, seq_len: usize, offset: usize) -> (Vec<Array>, Vec<Array>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    // Fixed seed not strictly needed here as we use offset, but good for reproducibility
    let mut rng = StdRng::seed_from_u64(42); 
    
    // Config matches training:
    // Period = 2PI/0.1 = ~62 steps
    // Amplitude = 10.0
    // Noise ~ [-2.5, 2.5]
    
    let total_points = num_samples + seq_len;
    let mut raw_data = Vec::with_capacity(total_points);
    
    for k in 0..total_points {
        let t = (offset + k) as f32;
        let mut val = (t * 0.1).sin() * 10.0;
        
        // Consistent noise generation (simple pseudo-random to match trailing logic if needed, 
        // but here we just need similar distribution)
        let noise: f32 = rng.gen_range(-2.5..2.5);
        val += noise;
        
        raw_data.push(val);
    }
    
    // Normalization Stats (Must match training!)
    let center = 0.0;
    let scale = 12.5;
    
    for i in 0..num_samples {
        let start_idx = i;
        
        let mut window_data = Vec::with_capacity(seq_len);
        for j in 0..seq_len {
            let norm_val = (raw_data[start_idx + j] - center) / scale;
            window_data.push(norm_val);
        }
        
        let target_val = (raw_data[start_idx + seq_len] - center) / scale;
        
        inputs.push(Array::new(vec![seq_len], window_data));
        targets.push(Array::new(vec![1], vec![target_val]));
    }
    
    (inputs, targets)
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔮  Time-Series Forecasting Inference (ONNX)");
    println!("═══════════════════════════════════════════════════════════\n");

    let model_path = "timeseries_cnn.onnx.json";
    
    if !std::path::Path::new(model_path).exists() {
        return Err(anyhow!("Model file '{}' not found. Run 'cargo run --release --example timeseries_cnn' first.", model_path));
    }
    
    println!("📂 Loading model from '{}'...", model_path);
    let model = load_onnx(model_path)?;
    println!("   ✅ Model loaded. Ops: {}", model.graph.nodes.len());
    
    // Identify input/output names
    if model.graph.inputs.is_empty() {
        return Err(anyhow!("Model has no inputs"));
    }
    let input_name = &model.graph.inputs[0].name; // e.g., "tensor_0"
    let output_name = &model.graph.outputs[0];    // e.g., "tensor_xx"
    
    println!("   Input Node:  {}", input_name);
    println!("   Output Node: {}\n", output_name);

    // Generate Test Data (Offset 1500.0 to be far from training range 0..500)
    let seq_len = 128; // Must match training
    println!("📊 Generating validation data (offset=1500)...");
    let (test_x, test_y) = generate_synthetic_timeseries(100, seq_len, 1500);
    
    // Select 25 random samples
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..test_x.len()).collect();
    indices.shuffle(&mut rng);
    let selected_indices = &indices[0..25];
    
    println!("   Running inference on 25 samples...\n");
    
    println!("   ┌────────┬────────────┬────────────┬────────────┬────────┐");
    println!("   │ Sample │   Target   │ Prediction │   Delta    │ Status │");
    println!("   ├────────┼────────────┼────────────┼────────────┼────────┤");
    
    let mut total_error = 0.0;
    let mut pass_count = 0;
    
    for &idx in selected_indices {
        let input_sample = &test_x[idx]; // [50]
        let target_val = test_y[idx].data[0];
        
        // Model expects [Batch, SeqLen] -> We provide [1, 50]
        // But wait, the model reshape logic inside `timeseries_cnn.rs` handles [Batch, SeqLen].
        // The ONNX Export traced the graph.
        // In `timeseries_cnn.rs`:
        //   let reshaped_input = input.reshape(vec![batch_size, 1, self.seq_len])?;
        //   let x = self.head.forward(&x)?;
        //
        // Model expects [Batch, 1, SeqLen] because Reshape was outside of traceable graph
        // So we provide pre-reshaped input [1, 1, 50].
        
        let input_array = Array::new(vec![1, 1, seq_len], input_sample.data.clone());
        
        let mut inputs = HashMap::new();
        inputs.insert(input_name.clone(), input_array);
        
        let results = infer(&model, inputs)?;
        let prediction_tensor = results.get(output_name)
            .ok_or_else(|| anyhow!("Output not found"))?;
            
        let prediction = prediction_tensor.data[0];
        let delta = (prediction - target_val).abs();
        
        total_error += delta;
        
        // Threshold: 0.15 (Acceptable for this noisy sine wave)
        let is_ok = delta < 0.15;
        if is_ok { pass_count += 1; }
        
        let status = if is_ok { "✅" } else { "⚠️ " };
        
        println!("   │ {:<6} │ {:>10.4} │ {:>10.4} │ {:>10.4} │   {}   │", 
            idx, target_val, prediction, delta, status);
    }
    
    println!("   └────────┴────────────┴────────────┴────────────┴────────┘");
    
    let avg_error = total_error / 25.0;
    println!("\n   Average Error: {:.4}", avg_error);
    println!("   Pass Rate: {}/25\n", pass_count);
    
    if avg_error < 0.1 {
        println!("   🚀 SUCCESS: Model generalizes well.");
    } else {
        println!("   ⚠️  WARNING: High error rate. Model might need more training or tuning.");
    }
    
    Ok(())
}
