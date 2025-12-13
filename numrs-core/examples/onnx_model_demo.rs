//! Example: ONNX Model Creation and Export
//! 
//! This example demonstrates how to:
//! 1. Create a simple neural network model
//! 2. Save it in ONNX-compatible format
//! 3. Load the model back
//! 4. Save/load training checkpoints

use numrs::array::Array;
use numrs::ops::model::*;
use numrs::llo::{OnnxModel, TrainingState, OnnxNode, OnnxTensor, OnnxAttribute};
use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  NumRs ONNX Model Example");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // ========================================================================
    // EXAMPLE 1: Create a simple MLP (Multi-Layer Perceptron)
    // ========================================================================
    println!("📦 Example 1: Creating a simple MLP model\n");
    
    let input_size = 784;   // MNIST-like input (28x28)
    let hidden_size = 128;  // Hidden layer
    let output_size = 10;   // 10 classes
    
    // Initialize random weights (in practice, these would be trained)
    let w1 = Array::new(vec![input_size, hidden_size], 
                        vec![0.01; input_size * hidden_size]);
    let b1 = Array::new(vec![hidden_size], vec![0.0; hidden_size]);
    let w2 = Array::new(vec![hidden_size, output_size], 
                        vec![0.01; hidden_size * output_size]);
    let b2 = Array::new(vec![output_size], vec![0.0; output_size]);
    
    println!("  Input size:  {}", input_size);
    println!("  Hidden size: {}", hidden_size);
    println!("  Output size: {}", output_size);
    println!("  Weights: w1({:?}), b1({:?}), w2({:?}), b2({:?})\n", 
             w1.shape(), b1.shape(), w2.shape(), b2.shape());
    
    // Create the MLP model
    let model = create_mlp(
        "mnist_classifier",
        input_size,
        hidden_size,
        output_size,
        vec![&w1, &b1, &w2, &b2]
    )?;
    
    println!("  ✓ Model created: {}", model.metadata.name);
    println!("    Version: {}", model.metadata.version);
    println!("    Opset: {}", model.metadata.opset_version);
    println!("    Nodes: {}", model.graph.nodes.len());
    println!("    Inputs: {}", model.graph.inputs.len());
    println!("    Outputs: {}", model.graph.outputs.len());
    println!("    Initializers: {}\n", model.graph.initializers.len());
    
    // ========================================================================
    // EXAMPLE 2: Save model to ONNX format
    // ========================================================================
    println!("💾 Example 2: Saving model to ONNX format\n");
    
    let model_path = "mnist_model.onnx.json";
    save_onnx(&model, model_path)?;
    
    println!("  ✓ Model saved to: {}", model_path);
    
    // Check file size
    let metadata = std::fs::metadata(model_path)?;
    println!("  File size: {} bytes\n", metadata.len());
    
    // ========================================================================
    // EXAMPLE 3: Load model from file
    // ========================================================================
    println!("📂 Example 3: Loading model from file\n");
    
    let loaded_model = load_onnx(model_path)?;
    
    println!("  ✓ Model loaded: {}", loaded_model.metadata.name);
    println!("    Version: {}", loaded_model.metadata.version);
    println!("    Nodes: {}", loaded_model.graph.nodes.len());
    println!("    Graph structure:");
    
    for (i, node) in loaded_model.graph.nodes.iter().enumerate() {
        println!("      {}. {} ({}) {} -> {}", 
                 i + 1,
                 node.name,
                 node.op_type,
                 node.inputs.join(", "),
                 node.outputs.join(", "));
    }
    println!();
    
    // ========================================================================
    // EXAMPLE 4: Training checkpoint
    // ========================================================================
    println!("🎯 Example 4: Training checkpoint\n");
    
    // Create a training state
    let mut training_state = TrainingState::new_adam(0.001);
    training_state.epoch = 5;
    training_state.iteration = 1000;
    training_state.loss = 0.234;
    
    println!("  Training state:");
    println!("    Epoch: {}", training_state.epoch);
    println!("    Iteration: {}", training_state.iteration);
    println!("    Loss: {:.4}", training_state.loss);
    println!("    Optimizer: {:?}\n", training_state.optimizer);
    
    // Save checkpoint
    let checkpoint_path = "checkpoint_epoch_5.json";
    save_checkpoint(&model, &training_state, checkpoint_path)?;
    
    println!("  ✓ Checkpoint saved to: {}", checkpoint_path);
    
    // Load checkpoint
    let (loaded_model_chk, loaded_state) = load_checkpoint(checkpoint_path)?;
    
    println!("  ✓ Checkpoint loaded");
    println!("    Model: {}", loaded_model_chk.metadata.name);
    println!("    Epoch: {}", loaded_state.epoch);
    println!("    Loss: {:.4}\n", loaded_state.loss);
    
    // ========================================================================
    // EXAMPLE 5: Custom model construction
    // ========================================================================
    println!("🔧 Example 5: Custom model construction\n");
    
    let mut custom_model = OnnxModel::new("linear_regression");
    
    // Define a simple linear regression: y = w*x + b
    custom_model.add_input(OnnxTensor {
        name: "x".to_string(),
        dtype: 1, // FLOAT
        shape: vec![1, 10],
        data: Vec::new(),
    });
    
    custom_model.add_initializer(OnnxTensor {
        name: "w".to_string(),
        dtype: 1,
        shape: vec![10, 1],
        data: vec![0u8; 40], // 10 floats * 4 bytes
    });
    
    custom_model.add_initializer(OnnxTensor {
        name: "b".to_string(),
        dtype: 1,
        shape: vec![1],
        data: vec![0u8; 4], // 1 float * 4 bytes
    });
    
    // Add MatMul node: x @ w
    custom_model.add_node(create_matmul_node("matmul", "x", "w", "xw"));
    
    // Add bias: xw + b
    custom_model.add_node(create_add_node("add_bias", "xw", "b", "y"));
    
    custom_model.set_outputs(vec!["y".to_string()]);
    
    println!("  ✓ Custom model created: {}", custom_model.metadata.name);
    println!("    Inputs: {:?}", custom_model.graph.inputs.iter()
                                             .map(|t| &t.name).collect::<Vec<_>>());
    println!("    Outputs: {:?}", custom_model.graph.outputs);
    println!("    Operations:");
    for node in &custom_model.graph.nodes {
        println!("      - {} ({})", node.name, node.op_type);
    }
    println!();
    
    // Save custom model
    let custom_path = "linear_regression.onnx.json";
    save_onnx(&custom_model, custom_path)?;
    println!("  ✓ Custom model saved to: {}\n", custom_path);
    
    // ========================================================================
    // EXAMPLE 6: Model inspection
    // ========================================================================
    println!("🔍 Example 6: Model inspection\n");
    
    println!("  Model metadata:");
    println!("    Name: {}", model.metadata.name);
    println!("    Version: {}", model.metadata.version);
    println!("    Author: {}", model.metadata.author);
    println!("    Domain: {}", model.metadata.domain);
    println!("    Producer: {} v{}", 
             model.metadata.producer_name,
             model.metadata.producer_version);
    println!("    Description: {}", model.metadata.description);
    
    println!("\n  Graph topology:");
    println!("    Input → {} → {} → {} → {} → Output",
             model.graph.nodes[0].name,
             model.graph.nodes[1].name,
             model.graph.nodes[2].name,
             model.graph.nodes[3].name);
    
    println!("\n  Parameters:");
    for init in &model.graph.initializers {
        let size = init.data.len() / 4; // Assuming float32
        println!("    {}: {:?} ({} elements)", init.name, init.shape, size);
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  All examples completed successfully! ✓");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("📝 Files created:");
    println!("  - {}", model_path);
    println!("  - {}", checkpoint_path);
    println!("  - {}", custom_path);
    println!("\nThese files are in ONNX-compatible JSON format.");
    println!("For production, convert to protobuf format using ONNX libraries.\n");
    
    // Clean up example files
    println!("🧹 Cleaning up example files...");
    std::fs::remove_file(model_path).ok();
    std::fs::remove_file(checkpoint_path).ok();
    std::fs::remove_file(custom_path).ok();
    println!("  ✓ Cleanup complete\n");
    
    Ok(())
}
