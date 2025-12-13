//! Time-Series Forecasting with ONNX (Regression)
//! 
//! This example shows how to use NumRs for a regression task:
//! 1. Generate synthetic sine wave data.
//! 2. Train an MLP to predict x[t] given x[t-window..t-1].
//! 3. Use MSELoss (Mean Squared Error).
//! 4. Export the trained model to ONNX.
//! 5. Perform inference to forecast future values.

use numrs::{Linear, Sequential, ReLU};
use numrs::{TrainerBuilder, Dataset, MSELoss, Module};
use numrs::{Array, Tensor};
use numrs::ops::model::{load_onnx, infer};
use numrs::ops::export::export_to_onnx;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::f32::consts::PI;

/// Generate synthetic time series data: Sin(x) + Noise
fn generate_sine_wave(steps: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(steps);
    for i in 0..steps {
        let x = (i as f32) * 0.1;
        let noise = (x.sin() * i as f32).cos() * 0.05; // Deterministic "noise" for reproducibility
        let y = x.sin() + noise;
        data.push(y);
    }
    data
}

/// Create sliding windows (Features, Targets)
/// window_size: Number of past steps to use as input
fn create_windows(data: &[f32], window_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut features = Vec::new();
    let mut targets = Vec::new(); // Regression target is vector of size 1
    
    for i in 0..(data.len() - window_size) {
        let window = data[i..(i + window_size)].to_vec();
        let target = vec![data[i + window_size]]; // Predict next step
        
        features.push(window);
        targets.push(target);
    }
    
    (features, targets)
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  📈  NumRs: Time-Series Forecasting (Regression -> ONNX)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ========================================================================
    // 1. Data Preparation
    // ========================================================================
    println!("📊 Step 1: Generating Data...");
    let total_steps = 500;
    let window_size = 10;
    
    let time_series = generate_sine_wave(total_steps);
    let (x_all, y_all) = create_windows(&time_series, window_size);
    
    // Split Train/Test (80/20)
    let split_idx = (x_all.len() as f32 * 0.8) as usize;
    
    let train_x = x_all[0..split_idx].to_vec();
    let train_y = y_all[0..split_idx].to_vec();
    
    let test_x = x_all[split_idx..].to_vec();
    let test_y = y_all[split_idx..].to_vec();
    
    println!("   Total Windows: {}", x_all.len());
    println!("   Train Size:    {}", train_x.len());
    println!("   Test Size:     {}", test_x.len());
    println!("   Input Window:  {} steps", window_size);
    
    let train_dataset = Dataset::new(train_x, train_y, 16); // Batch 16
    
    // ========================================================================
    // 2. Model Architecture
    // ========================================================================
    println!("\n🧠 Step 2: Defining Regression Model...");
    // Input: 10 -> Hidden: 32 -> Output: 1
    // No activation on output (Linear regression)
    let model = Sequential::new(vec![
        Box::new(Linear::new(window_size, 32)?),
        Box::new(ReLU),
        Box::new(Linear::new(32, 1)?),
    ]);
    
    println!("   Layer 1: Linear(10 -> 32) + ReLU");
    println!("   Layer 2: Linear(32 -> 1)");
    println!("   Loss:    MSELoss");

    // ========================================================================
    // 3. Training
    // ========================================================================
    println!("\n🎯 Step 3: Training...");
    
    // Regression often needs smaller LR or tuning compared to classification with LogSoftmax
    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(0.001)
        .build_sgd(Box::new(MSELoss));
        
    let history = trainer.fit(&train_dataset, None, 50, true)?; // 50 epochs
    let final_loss = history.last().unwrap().0.loss;
    println!("   ✓ Training Complete. Final MSE Loss: {:.6}", final_loss);

    // ========================================================================
    // 4. Export to ONNX
    // ========================================================================
    println!("\n💾 Step 4: Auto-Exporting to ONNX...");
    
    // Create dummy input for tracing [1, 10]
    let dummy_data = vec![0.0; window_size];
    let dummy_input = Tensor::new(Array::new(vec![1, window_size], dummy_data), false);
    
    // Run forward to get graph
    let output = trainer.model().forward(&dummy_input)?;
    
    let model_path = "forecast_model.onnx.json";
    export_to_onnx(&output, model_path)?;
    println!("   ✅ Exported to '{}'", model_path);

    // ========================================================================
    // 5. ONNX Inference & Forecast
    // ========================================================================
    println!("\n🔮 Step 5: Forecasting via ONNX Inference...");
    
    // Reload model
    let onnx_model = load_onnx(model_path)?;
    let input_node_name = &onnx_model.graph.inputs[0].name;
    let output_node_name = &onnx_model.graph.outputs[0];
    
    println!("   Model Loaded. Running forecast on Test Set...");
    println!("   Comparison (First 10 test points):");
    println!("   ┌───────────┬───────────┬──────────────┐");
    println!("   │  Actual   │ Predicted │  Diff (Abs)  │");
    println!("   ├───────────┼───────────┼──────────────┤");
    
    let mut total_error = 0.0;
    let limit = 10;
    
    for (i, (input_win, target_val)) in test_x.iter().zip(test_y.iter()).enumerate() {
        // Prepare input tensor [1, 10]
        let input_arr = Array::new(vec![1, window_size], input_win.clone());
        let mut inputs = HashMap::new();
        inputs.insert(input_node_name.clone(), input_arr);
        
        // Infer
        let outputs_map = infer(&onnx_model, inputs)?;
        let pred_arr = outputs_map.get(output_node_name).ok_or(anyhow!("Missing output"))?;
        
        // Get scalar prediction
        let pred_val = pred_arr.data[0];
        let actual_val = target_val[0];
        let diff = (pred_val - actual_val).abs();
        
        total_error += diff;
        
        if i < limit {
            let bar_len = (diff * 100.0) as usize; 
            let bar = "█".repeat(bar_len.min(10));
            // Colored output roughly via ASCII
            let icon = if diff < 0.1 { "✅" } else { "⚠️" };
            
            println!("   │ {:>9.4} │ {:>9.4} │ {:>12.4} {} │", 
                actual_val, pred_val, diff, icon
            );
        }
    }
    
    println!("   └───────────┴───────────┴──────────────┘");
    
    let avg_mae = total_error / test_x.len() as f32;
    println!("\n   Test Set MAE (Mean Absolute Error): {:.6}", avg_mae);
    
    if avg_mae < 0.2 {
        println!("   🚀 SUCCESS: Forecast is reasonably accurate!");
    } else {
        println!("   ⚠️  WARNING: Forecast error is high. Tune hyperparameters.");
    }

    Ok(())
}
