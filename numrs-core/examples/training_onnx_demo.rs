//! Simple training loop example with ONNX model export
//! 
//! This demonstrates how to:
//! 1. Initialize a simple model
//! 2. Run a training loop (simplified)
//! 3. Save checkpoints during training
//! 4. Export final model to ONNX

use numrs::array::Array;
use numrs::ops::model::*;
use numrs::llo::TrainingState;
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  NumRs Training + ONNX Export Example");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // Model hyperparameters
    let input_size = 10;
    let hidden_size = 20;
    let output_size = 2;
    let learning_rate = 0.01;
    let epochs = 5;
    
    println!("📋 Model Configuration:");
    println!("  Input size:     {}", input_size);
    println!("  Hidden size:    {}", hidden_size);
    println!("  Output size:    {}", output_size);
    println!("  Learning rate:  {}", learning_rate);
    println!("  Epochs:         {}\n", epochs);
    
    // Initialize weights (in practice, use proper initialization like Xavier/He)
    let mut w1 = Array::new(
        vec![input_size, hidden_size],
        (0..input_size * hidden_size)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
            .collect()
    );
    let b1 = Array::new(vec![hidden_size], vec![0.0; hidden_size]);
    let mut w2 = Array::new(
        vec![hidden_size, output_size],
        (0..hidden_size * output_size)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.1)
            .collect()
    );
    let b2 = Array::new(vec![output_size], vec![0.0; output_size]);
    
    println!("✓ Weights initialized");
    
    // Create training state
    let mut training_state = TrainingState::new_adam(learning_rate);
    
    // Dummy training data (flattened - no batch dimension for simplicity)
    let _x_train = Array::new(
        vec![input_size],
        vec![1.0; input_size]
    );
    let _y_train = Array::new(
        vec![output_size],
        vec![1.0, 0.0]
    );
    
    println!("✓ Training data prepared\n");
    
    // Training loop (simplified - no actual gradient computation)
    println!("🎯 Starting training...\n");
    
    for epoch in 0..epochs {
        training_state.epoch = epoch;
        
        // Simplified forward pass (without actual matrix operations for demo)
        // In practice, you would do: x @ w1 + b1, relu, @ w2 + b2, softmax
        
        // Simulate loss computation
        let loss = 1.0 / (epoch + 1) as f32; // Decreasing loss
        
        training_state.loss = loss;
        training_state.iteration = epoch * 100; // Simulate batches
        
        println!("  Epoch {}/{}: loss = {:.6}", 
                 epoch + 1, epochs, loss);
        
        // Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0 {
            let checkpoint_path = format!("checkpoint_epoch_{}.json", epoch + 1);
            let model = create_mlp(
                "training_model",
                input_size,
                hidden_size,
                output_size,
                vec![&w1, &b1, &w2, &b2]
            )?;
            save_checkpoint(&model, &training_state, &checkpoint_path)?;
            println!("    → Checkpoint saved: {}", checkpoint_path);
        }
        
        // Simulate weight update (in practice, use actual gradients)
        for val in w1.data.iter_mut() {
            *val -= learning_rate * 0.001; // Dummy gradient
        }
        for val in w2.data.iter_mut() {
            *val -= learning_rate * 0.001;
        }
    }
    
    println!("\n✓ Training completed!\n");
    
    // Export final model
    println!("💾 Exporting final model to ONNX...\n");
    
    let final_model = create_mlp(
        "trained_classifier",
        input_size,
        hidden_size,
        output_size,
        vec![&w1, &b1, &w2, &b2]
    )?;
    
    save_onnx(&final_model, "trained_model.onnx.json")?;
    
    println!("  ✓ Model exported: trained_model.onnx.json");
    
    // Save final training state
    save_checkpoint(&final_model, &training_state, "final_checkpoint.json")?;
    println!("  ✓ Final checkpoint: final_checkpoint.json\n");
    
    // Model summary
    println!("📊 Model Summary:");
    println!("  Name:        {}", final_model.metadata.name);
    println!("  Version:     {}", final_model.metadata.version);
    println!("  Opset:       {}", final_model.metadata.opset_version);
    println!("  Nodes:       {}", final_model.graph.nodes.len());
    println!("  Parameters:  {} tensors", final_model.graph.initializers.len());
    
    let total_params: usize = final_model.graph.initializers.iter()
        .map(|t| t.data.len() / 4) // Assuming float32
        .sum();
    println!("  Total size:  {} parameters\n", total_params);
    
    println!("🔍 Graph Structure:");
    for (i, node) in final_model.graph.nodes.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, node.name, node.op_type);
        println!("     Inputs:  {}", node.inputs.join(", "));
        println!("     Outputs: {}", node.outputs.join(", "));
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Training example completed! ✓");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("📝 Files created:");
    println!("  - checkpoint_epoch_2.json");
    println!("  - checkpoint_epoch_4.json");
    println!("  - trained_model.onnx.json");
    println!("  - final_checkpoint.json\n");
    
    println!("💡 Next steps:");
    println!("  1. Load the model with: load_onnx(\"trained_model.onnx.json\")");
    println!("  2. Resume training from checkpoint");
    println!("  3. Convert to protobuf for ONNX Runtime");
    println!("  4. Deploy for inference\n");
    
    // Cleanup
    println!("🧹 Cleaning up example files...");
    std::fs::remove_file("checkpoint_epoch_2.json").ok();
    std::fs::remove_file("checkpoint_epoch_4.json").ok();
    std::fs::remove_file("trained_model.onnx.json").ok();
    std::fs::remove_file("final_checkpoint.json").ok();
    println!("  ✓ Cleanup complete\n");
    
    Ok(())
}
