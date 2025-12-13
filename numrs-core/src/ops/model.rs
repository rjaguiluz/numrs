//! Model operations for saving, loading and inference
//!
//! This module provides high-level operations for:
//! - Saving models in ONNX format
//! - Loading models from ONNX format
//! - Running inference on loaded models
//! - Training support (forward/backward passes)

use crate::array::{Array, DTypeValue};
pub use crate::llo::{OnnxAttribute, OnnxModel, OnnxNode, OnnxTensor, TrainingState};
use crate::ops::{add, div, matmul, mul, relu, sigmoid, softmax, sub, transpose};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Serialize model to JSON string
pub fn serialize_onnx(model: &OnnxModel) -> Result<String> {
    Ok(serde_json::to_string_pretty(model)?)
}

/// Save model to ONNX format
///
/// # Arguments
/// * `model` - The ONNX model to save
/// * `path` - File path to save the model
///
/// # Example
/// ```no_run
/// use numrs::ops::model::{save_onnx, OnnxModel};
///
/// let model = OnnxModel::new("my_model");
/// save_onnx(&model, "model.onnx")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn save_onnx(model: &OnnxModel, path: &str) -> Result<()> {
    // Serialize to JSON (or could use protobuf for full ONNX compatibility)
    let json = serialize_onnx(model)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Deserialize model from JSON string
pub fn deserialize_onnx(json: &str) -> Result<OnnxModel> {
    let model = serde_json::from_str(json)?;
    Ok(model)
}

/// Load model from ONNX format
///
/// # Arguments
/// * `path` - File path to load the model from
///
/// # Returns
/// The loaded ONNX model
///
/// # Example
/// ```no_run
/// use numrs::ops::model::load_onnx;
///
/// let model = load_onnx("model.onnx")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_onnx(path: &str) -> Result<OnnxModel> {
    let json = std::fs::read_to_string(path)?;
    deserialize_onnx(&json)
}

/// Create a linear layer node
///
/// # Arguments
/// * `name` - Node name
/// * `input` - Input tensor name
/// * `weights` - Weights tensor name
/// * `bias` - Bias tensor name (optional)
/// * `output` - Output tensor name
pub fn create_linear_node(
    name: &str,
    input: &str,
    weights: &str,
    bias: Option<&str>,
    output: &str,
) -> OnnxNode {
    let mut inputs = vec![input.to_string(), weights.to_string()];
    if let Some(b) = bias {
        inputs.push(b.to_string());
    }

    OnnxNode {
        name: name.to_string(),
        op_type: "Gemm".to_string(), // ONNX Gemm = General Matrix Multiply
        inputs,
        outputs: vec![output.to_string()],
        attributes: {
            let mut attrs = HashMap::new();
            attrs.insert("alpha".to_string(), OnnxAttribute::Float(1.0));
            attrs.insert("beta".to_string(), OnnxAttribute::Float(1.0));
            attrs.insert("transA".to_string(), OnnxAttribute::Int(0));
            attrs.insert("transB".to_string(), OnnxAttribute::Int(1)); // Transpose weights
            attrs
        },
    }
}

/// Create a ReLU activation node
pub fn create_relu_node(name: &str, input: &str, output: &str) -> OnnxNode {
    OnnxNode {
        name: name.to_string(),
        op_type: "Relu".to_string(),
        inputs: vec![input.to_string()],
        outputs: vec![output.to_string()],
        attributes: HashMap::new(),
    }
}

/// Create a Softmax activation node
pub fn create_softmax_node(name: &str, input: &str, output: &str, axis: i64) -> OnnxNode {
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), OnnxAttribute::Int(axis));

    OnnxNode {
        name: name.to_string(),
        op_type: "Softmax".to_string(),
        inputs: vec![input.to_string()],
        outputs: vec![output.to_string()],
        attributes: attrs,
    }
}

/// Create a MatMul node
pub fn create_matmul_node(name: &str, input_a: &str, input_b: &str, output: &str) -> OnnxNode {
    OnnxNode {
        name: name.to_string(),
        op_type: "MatMul".to_string(),
        inputs: vec![input_a.to_string(), input_b.to_string()],
        outputs: vec![output.to_string()],
        attributes: HashMap::new(),
    }
}

/// Create an Add node
pub fn create_add_node(name: &str, input_a: &str, input_b: &str, output: &str) -> OnnxNode {
    OnnxNode {
        name: name.to_string(),
        op_type: "Add".to_string(),
        inputs: vec![input_a.to_string(), input_b.to_string()],
        outputs: vec![output.to_string()],
        attributes: HashMap::new(),
    }
}

/// Create a tensor from an Array
pub fn array_to_onnx_tensor<T: DTypeValue>(name: &str, array: &Array<T>) -> Result<OnnxTensor> {
    // Convert array data to bytes
    let data = &array.data;
    let bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
        .to_vec()
    };

    // Determine ONNX dtype (1=FLOAT, 6=INT32, 7=INT64, 11=DOUBLE)
    let dtype = match std::any::type_name::<T>() {
        "f32" => 1,  // FLOAT
        "f64" => 11, // DOUBLE
        "i32" => 6,  // INT32
        "i64" => 7,  // INT64
        _ => return Err(anyhow!("Unsupported data type for ONNX")),
    };

    Ok(OnnxTensor {
        name: name.to_string(),
        dtype,
        shape: array.shape().to_vec(),
        data: bytes,
    })
}

/// Create a simple feedforward neural network model
///
/// # Arguments
/// * `name` - Model name
/// * `input_size` - Input feature size
/// * `hidden_size` - Hidden layer size
/// * `output_size` - Output size
/// * `weights` - Layer weights [w1, b1, w2, b2]
///
/// # Example
/// ```no_run
/// use numrs::ops::model::create_mlp;
/// use numrs::Array;
///
/// let w1 = Array::new(vec![784, 128], vec![0.0; 784 * 128]);
/// let b1 = Array::new(vec![128], vec![0.0; 128]);
/// let w2 = Array::new(vec![128, 10], vec![0.0; 128 * 10]);
/// let b2 = Array::new(vec![10], vec![0.0; 10]);
///
/// let model = create_mlp("mnist_classifier", 784, 128, 10,
///                        vec![&w1, &b1, &w2, &b2])?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn create_mlp(
    name: &str,
    input_size: usize,
    _hidden_size: usize,
    _output_size: usize,
    weights: Vec<&Array>,
) -> Result<OnnxModel> {
    if weights.len() != 4 {
        return Err(anyhow!("Expected 4 weight arrays [w1, b1, w2, b2]"));
    }

    let mut model = OnnxModel::new(name);

    // Add input tensor
    let input = OnnxTensor {
        name: "input".to_string(),
        dtype: 1,                   // FLOAT
        shape: vec![1, input_size], // Batch size 1
        data: Vec::new(),
    };
    model.add_input(input);

    // Add weights as initializers
    model.add_initializer(array_to_onnx_tensor("w1", weights[0])?);
    model.add_initializer(array_to_onnx_tensor("b1", weights[1])?);
    model.add_initializer(array_to_onnx_tensor("w2", weights[2])?);
    model.add_initializer(array_to_onnx_tensor("b2", weights[3])?);

    // Layer 1: Linear + ReLU
    model.add_node(create_linear_node(
        "fc1",
        "input",
        "w1",
        Some("b1"),
        "hidden",
    ));
    model.add_node(create_relu_node("relu1", "hidden", "hidden_act"));

    // Layer 2: Linear + Softmax
    model.add_node(create_linear_node(
        "fc2",
        "hidden_act",
        "w2",
        Some("b2"),
        "logits",
    ));
    model.add_node(create_softmax_node("softmax", "logits", "output", 1));

    // Set output
    model.set_outputs(vec!["output".to_string()]);

    Ok(model)
}

/// Run inference on a model
///
/// # Arguments
/// * `model` - The ONNX model
/// * `inputs` - Input tensors as a map of name -> Array
///
/// # Returns
/// Output tensors as a map of name -> Array
///
/// # Note
/// This is a simplified inference engine. For production use, consider using
/// a full ONNX runtime like onnxruntime-rs.
/// Run inference on a model
///
/// # Arguments
/// * `model` - The ONNX model
/// * `inputs` - Input tensors as a map of name -> Array
///
/// # Returns
/// Output tensors as a map of name -> Array
pub fn infer(model: &OnnxModel, inputs: HashMap<String, Array>) -> Result<HashMap<String, Array>> {
    // 1. Initialize value store with inputs and initializers
    let mut values: HashMap<String, Array> = inputs;

    // Load initializers (weights) into values
    for init in &model.graph.initializers {
        // Convert OnnxTensor bytes back to Array
        // Assuming f32 for now as per export limitation, but handling INT64 shape tensors via cast
        if init.dtype == 1 {
            // FLOAT (f32)
            let data_f32: Vec<f32> = init
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let array = Array::new(init.shape.clone(), data_f32);
            values.insert(init.name.clone(), array);
        } else if init.dtype == 7 {
            // INT64 (cast to f32 for compatibility with Array<f32> map)
            let data_f32: Vec<f32> = init
                .data
                .chunks_exact(8)
                .map(|b| {
                    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                })
                .collect();
            let array = Array::new(init.shape.clone(), data_f32);
            values.insert(init.name.clone(), array);
        } else {
            return Err(anyhow!("Unsupported initializer dtype: {}", init.dtype));
        }
    }

    // 2. Execute nodes sequentially (assuming topological sort in export)
    for node in &model.graph.nodes {
        // Prepare inputs
        let mut node_inputs = Vec::new();
        for input_name in &node.inputs {
            let val = values.get(input_name).ok_or_else(|| {
                anyhow!("Missing input '{}' for node '{}'", input_name, node.name)
            })?;
            node_inputs.push(val);
        }

        let output_name = &node.outputs[0]; // Assuming single output for now

        let output_val = match node.op_type.as_str() {
            "Add" => add(&node_inputs[0], &node_inputs[1])?,
            "Sub" => sub(&node_inputs[0], &node_inputs[1])?,
            "Mul" => mul(&node_inputs[0], &node_inputs[1])?,
            "Div" => div(&node_inputs[0], &node_inputs[1])?,
            "MatMul" => matmul(&node_inputs[0], &node_inputs[1])?,
            "Relu" => relu(&node_inputs[0])?,
            "Sigmoid" => sigmoid(&node_inputs[0])?,
            "Softmax" => {
                // Check axis attribute
                let axis = node
                    .attributes
                    .get("axis")
                    .and_then(|a| match a {
                        OnnxAttribute::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(1); // Default axis 1
                softmax(&node_inputs[0], Some(axis))?
            }
            "Gemm" => {
                // Y = alpha * A' * B' + beta * C
                let a = node_inputs[0];
                let b = node_inputs[1];
                let c = node_inputs.get(2); // Optional bias

                let alpha = node
                    .attributes
                    .get("alpha")
                    .and_then(|v| match v {
                        OnnxAttribute::Float(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(1.0);
                let beta = node
                    .attributes
                    .get("beta")
                    .and_then(|v| match v {
                        OnnxAttribute::Float(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(1.0);

                let trans_a = node
                    .attributes
                    .get("transA")
                    .and_then(|v| match v {
                        OnnxAttribute::Int(i) => Some(*i == 1),
                        _ => None,
                    })
                    .unwrap_or(false);
                let trans_b = node
                    .attributes
                    .get("transB")
                    .and_then(|v| match v {
                        OnnxAttribute::Int(i) => Some(*i == 1),
                        _ => None,
                    })
                    .unwrap_or(false);

                // Transpose if needed
                let a_processed = if trans_a {
                    transpose(a, None)?
                } else {
                    a.clone()
                };
                let b_processed = if trans_b {
                    transpose(b, None)?
                } else {
                    b.clone()
                };

                // MatMul
                let mut mul_res = matmul(&a_processed, &b_processed)?;

                // Alpha scaling
                if alpha != 1.0 {
                    let scalar = Array::new(vec![1], vec![alpha]);
                    mul_res = mul(&mul_res, &scalar)?;
                }

                // Add bias (Beta * C)
                if let Some(bias) = c {
                    if beta != 1.0 {
                        let scalar = Array::new(vec![1], vec![beta]);
                        let bias_scaled = mul(bias, &scalar)?;
                        add(&mul_res, &bias_scaled)?
                    } else {
                        add(&mul_res, bias)?
                    }
                } else {
                    mul_res
                }
            }
            "Transpose" => transpose(node_inputs[0], None)?,
            "Conv" => {
                // Inputs: X, W, [B]
                let x = node_inputs[0];
                let w = node_inputs[1];
                let b = node_inputs.get(2).map(|&v| v);

                // Attributes
                let padding = node
                    .attributes
                    .get("pads")
                    .and_then(|a| match a {
                        OnnxAttribute::Ints(v) => Some(v[0] as usize),
                        _ => None,
                    })
                    .unwrap_or(0);

                let stride = node
                    .attributes
                    .get("strides")
                    .and_then(|a| match a {
                        OnnxAttribute::Ints(v) => Some(v[0] as usize),
                        _ => None,
                    })
                    .unwrap_or(1);

                crate::ops::conv::conv1d(x, w, b, stride, padding)?
            }
            "Reshape" => {
                // Inputs: Data, Shape (Tensor)
                let data = node_inputs[0];
                let shape_tensor = node_inputs[1];

                // Extract shape from tensor data (assuming int64 or int32)
                let shape_vec: Vec<isize> = shape_tensor.data.iter().map(|&v| v as isize).collect();

                crate::ops::shape::reshape(data, &shape_vec)?
            }
            "Flatten" => {
                // Inputs: Data
                // Attribute: axis (default 1)
                let axis = node
                    .attributes
                    .get("axis")
                    .and_then(|a| match a {
                        OnnxAttribute::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(1);

                // Flatten from axis to end
                // Note: numrs ops::flatten might differ, using reshape approach for safety if flattened dims are contiguous
                // But simpler: just use ops::flatten (assuming it exists matching Tensor::flatten)
                // Or implementing via reshape: [batch, -1] usually

                // Let's use ops::flatten if available, checking imports
                // For now, manual reshape for safety: flatten [d0...axis-1, axis...end]
                // Actually my ops export usually maps Flatten -> Reshape in PyTorch legacy, but here explicit Flatten op.
                // ops::flatten(a, start_dim, end_dim)
                crate::ops::shape::flatten(node_inputs[0], axis, node_inputs[0].shape().len() - 1)?
            }
            "BatchNormalization" => {
                // Inputs: X, scale, B, mean, var
                let x = node_inputs[0];
                let scale = node_inputs[1];
                let b = node_inputs[2];
                let mean = node_inputs[3];
                let var = node_inputs[4];

                let epsilon = node
                    .attributes
                    .get("epsilon")
                    .and_then(|a| match a {
                        OnnxAttribute::Float(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(1e-5);

                let momentum = node
                    .attributes
                    .get("momentum")
                    .and_then(|a| match a {
                        OnnxAttribute::Float(f) => Some(*f),
                        _ => None,
                    })
                    .unwrap_or(0.9);

                // Clone running stats because batch_norm expects &mut even if training=false
                let mut mean_clone = mean.clone();
                let mut var_clone = var.clone();

                crate::ops::batchnorm::batch_norm(
                    x,
                    &mut mean_clone,
                    &mut var_clone,
                    scale,
                    b,
                    false,
                    momentum,
                    epsilon,
                )?
            }
            "Dropout" => {
                // Identity for inference
                node_inputs[0].clone()
            }
            _ => return Err(anyhow!("Unsupported op type: {}", node.op_type)),
        };

        values.insert(output_name.clone(), output_val);
    }

    // 3. Collect outputs
    let mut results = HashMap::new();
    for out_name in &model.graph.outputs {
        if let Some(val) = values.get(out_name) {
            results.insert(out_name.clone(), val.clone());
        } else {
            return Err(anyhow!(
                "Model output '{}' not found after execution",
                out_name
            ));
        }
    }

    Ok(results)
}

/// Save training checkpoint
///
/// # Arguments
/// * `model` - The model to save
/// * `training_state` - Current training state (epoch, loss, optimizer state)
/// * `path` - Path to save checkpoint
pub fn save_checkpoint(
    model: &OnnxModel,
    training_state: &TrainingState,
    path: &str,
) -> Result<()> {
    #[derive(serde::Serialize)]
    struct Checkpoint<'a> {
        model: &'a OnnxModel,
        training_state: &'a TrainingState,
    }

    let checkpoint = Checkpoint {
        model,
        training_state,
    };
    let json = serde_json::to_string_pretty(&checkpoint)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load training checkpoint
///
/// # Arguments
/// * `path` - Path to load checkpoint from
///
/// # Returns
/// Tuple of (model, training_state)
pub fn load_checkpoint(path: &str) -> Result<(OnnxModel, TrainingState)> {
    #[derive(serde::Deserialize)]
    struct Checkpoint {
        model: OnnxModel,
        training_state: TrainingState,
    }

    let json = std::fs::read_to_string(path)?;
    let checkpoint: Checkpoint = serde_json::from_str(&json)?;
    Ok((checkpoint.model, checkpoint.training_state))
}
