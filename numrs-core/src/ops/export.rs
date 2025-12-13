//! ONNX Export Module
//!
//! Provides functionality to automatically export Autograd graphs to ONNX.

use crate::autograd::{OpKind, Tensor};
use crate::llo::{array_to_onnx_tensor, OnnxAttribute, OnnxModel, OnnxNode, OnnxTensor};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};

/// Exporter state context
struct GraphExporter {
    model: OnnxModel,
    _visited: HashSet<usize>,             // Visited NodeIds
    node_outputs: HashMap<usize, String>, // NodeId -> Output Name
    tensor_names: HashMap<usize, String>, // Tensor ID (pointer) -> Name
    name_counter: usize,
}

impl GraphExporter {
    fn new(name: &str) -> Self {
        GraphExporter {
            model: OnnxModel::new(name),
            _visited: HashSet::new(),
            node_outputs: HashMap::new(),
            tensor_names: HashMap::new(),
            name_counter: 0,
        }
    }

    fn get_tensor_name(&mut self, tensor: &Tensor) -> String {
        // Use tensor's data pointer as unique ID roughly
        let id = tensor.data.data.as_ptr() as usize;

        if let Some(name) = self.tensor_names.get(&id) {
            return name.clone();
        }

        let name = format!("tensor_{}", self.name_counter);
        self.name_counter += 1;
        self.tensor_names.insert(id, name.clone());
        name
    }

    fn traverse(&mut self, tensor: &Tensor) -> Result<String> {
        let output_name = self.get_tensor_name(tensor);

        // If leaf (no compute node), it's either Input or Weight
        if tensor.compute_node.is_none() {
            if self
                .model
                .graph
                .inputs
                .iter()
                .any(|i| i.name == output_name)
                || self
                    .model
                    .graph
                    .initializers
                    .iter()
                    .any(|i| i.name == output_name)
            {
                return Ok(output_name);
            }

            if tensor.requires_grad {
                // Weight (Parameter)
                let onnx_tensor = array_to_onnx_tensor(&output_name, &tensor.data)?;
                self.model.add_initializer(onnx_tensor);
            } else {
                // Input
                let input = OnnxTensor {
                    name: output_name.clone(),
                    dtype: 1, // FLOAT
                    shape: tensor.shape().to_vec(),
                    data: vec![], // Inputs don't have data
                };
                self.model.add_input(input);
            }
            return Ok(output_name);
        }

        // If compute node, traverse recurrently
        let node = tensor.compute_node.as_ref().unwrap();
        // (We assume NodeId is unique per operation instance)
        // Accessing private field 'id' might be tricky if not pub, checking mod.rs...
        // Assuming we can check visitation by output name existence for now
        // A better way is needed if multiple outputs per node, but here 1 tensor = 1 node mostly.

        // Check if we already visited this operation by checking if its output is defined?
        // Actually, Autograd structure in NumRs is: Tensor HAS A Node.
        // So visiting the Tensor visits the Node.
        // We need to avoid re-emitting the same node if multiple tensors point to it (not common here)
        // OR if the graph splits and joins.

        // To handle diamond dependencies, we should check availability of inputs.
        // But here we are traversing BACKWARDS from output.
        // Better approach:
        // 1. Traverse recursively to inputs.
        // 2. Once inputs are ready (names returned), emit THIS node.

        // Check cycle/visited
        // Using output_name as unique key for "tensor computed status"
        if self.node_outputs.values().any(|n| n == &output_name) {
            return Ok(output_name);
        }

        let mut input_names = Vec::new();
        for input in &node.inputs {
            input_names.push(self.traverse(input)?);
        }

        // Emit ONNX Node
        let node_name = format!("node_{}_{}", self.name_counter, output_name);

        let onnx_node = match &node.op {
            OpKind::Add => OnnxNode {
                op_type: "Add".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::Mul => OnnxNode {
                op_type: "Mul".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::MatMul => OnnxNode {
                op_type: "MatMul".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::ReLU => OnnxNode {
                op_type: "Relu".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::Sigmoid => OnnxNode {
                op_type: "Sigmoid".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::Softmax => {
                let mut attrs = HashMap::new();
                attrs.insert("axis".to_string(), OnnxAttribute::Int(1));
                OnnxNode {
                    op_type: "Softmax".to_string(),
                    name: node_name,
                    inputs: input_names,
                    outputs: vec![output_name.clone()],
                    attributes: attrs,
                }
            }
            OpKind::Log => OnnxNode {
                op_type: "Log".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::Exp => OnnxNode {
                op_type: "Exp".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::Sub => OnnxNode {
                op_type: "Sub".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::Div => OnnxNode {
                op_type: "Div".to_string(),
                name: node_name,
                inputs: input_names,
                outputs: vec![output_name.clone()],
                attributes: HashMap::new(),
            },
            OpKind::Transpose => {
                let mut attrs = HashMap::new();
                // Default transpose in autograd is likely 2D swap (perm=[1,0])
                attrs.insert("perm".to_string(), OnnxAttribute::Ints(vec![1, 0]));
                OnnxNode {
                    op_type: "Transpose".to_string(),
                    name: node_name,
                    inputs: input_names,
                    outputs: vec![output_name.clone()],
                    attributes: attrs,
                }
            }
            OpKind::Conv1D { stride, padding } => {
                let mut attrs = HashMap::new();
                attrs.insert(
                    "strides".to_string(),
                    OnnxAttribute::Ints(vec![*stride as i64]),
                );
                attrs.insert(
                    "pads".to_string(),
                    OnnxAttribute::Ints(vec![*padding as i64, *padding as i64]),
                ); // Start, End

                // Get kernel_shape from weight (input[1])
                if node.inputs.len() >= 2 {
                    let weight = &node.inputs[1];
                    let k = weight.shape()[2];
                    attrs.insert(
                        "kernel_shape".to_string(),
                        OnnxAttribute::Ints(vec![k as i64]),
                    );
                }

                OnnxNode {
                    op_type: "Conv".to_string(),
                    name: node_name,
                    inputs: input_names,
                    outputs: vec![output_name.clone()],
                    attributes: attrs,
                }
            }
            OpKind::Flatten { start_dim, end_dim } => {
                // If standard Flatten(1, -1):
                if *start_dim == 1 && (*end_dim == usize::MAX || *end_dim == 2) {
                    let mut attrs = HashMap::new();
                    attrs.insert("axis".to_string(), OnnxAttribute::Int(1));
                    OnnxNode {
                        op_type: "Flatten".to_string(),
                        name: node_name,
                        inputs: input_names,
                        outputs: vec![output_name.clone()],
                        attributes: attrs,
                    }
                } else {
                    let mut attrs = HashMap::new();
                    attrs.insert("axis".to_string(), OnnxAttribute::Int(*start_dim as i64));
                    OnnxNode {
                        op_type: "Flatten".to_string(),
                        name: node_name,
                        inputs: input_names,
                        outputs: vec![output_name.clone()],
                        attributes: attrs,
                    }
                }
            }
            OpKind::Reshape { shape } => {
                // Reshape consumes: Data, Shape (as a Tensor!)

                // 1. Create Shape Initializer
                let shape_name = format!("{}_shape_const", output_name);

                // Manually construct OnnxTensor
                let shape_tensor = OnnxTensor {
                    name: shape_name.clone(),
                    dtype: 7, // INT64
                    shape: vec![shape.len()],
                    data: unsafe {
                        let i64s: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
                        let ptr = i64s.as_ptr() as *const u8;
                        let len = i64s.len() * 8;
                        std::slice::from_raw_parts(ptr, len).to_vec()
                    },
                };
                self.model.add_initializer(shape_tensor);

                // 2. Add input dependency
                let mut node_inputs = input_names.clone();
                node_inputs.push(shape_name);

                OnnxNode {
                    op_type: "Reshape".to_string(),
                    name: node_name,
                    inputs: node_inputs,
                    outputs: vec![output_name.clone()],
                    attributes: HashMap::new(),
                }
            }
            OpKind::BatchNorm {
                training: _,
                momentum,
                eps,
            } => {
                let mut attrs = HashMap::new();
                attrs.insert("epsilon".to_string(), OnnxAttribute::Float(*eps));
                attrs.insert("momentum".to_string(), OnnxAttribute::Float(*momentum));

                OnnxNode {
                    op_type: "BatchNormalization".to_string(),
                    name: node_name,
                    inputs: input_names,
                    outputs: vec![output_name.clone()],
                    attributes: attrs,
                }
            }
            OpKind::Dropout { p, training: _ } => {
                let mut attrs = HashMap::new();
                attrs.insert("ratio".to_string(), OnnxAttribute::Float(*p));
                OnnxNode {
                    op_type: "Dropout".to_string(),
                    name: node_name,
                    inputs: input_names,
                    outputs: vec![output_name.clone()],
                    attributes: attrs,
                }
            }
            // Handle specific complex ops or fallbacks
            _ => return Err(anyhow!("Unsupported op for export: {:?}", node.op)),
        };

        self.model.add_node(onnx_node);
        self.node_outputs
            .insert(self.name_counter, output_name.clone()); // Dummy ID usage

        Ok(output_name)
    }
}

/// Export a tensor's computational graph to ONNX JSON string
///
/// Use this when file system access is not available (e.g. WASM).
pub fn export_to_json(output: &Tensor) -> Result<String> {
    let mut exporter = GraphExporter::new("exported_model");

    // Traverse graph to populate model
    let output_name = exporter.traverse(output)?;

    // Set output
    exporter.model.set_outputs(vec![output_name]);

    // Serialize
    crate::ops::model::serialize_onnx(&exporter.model)
}

/// Export a tensor's computational graph to ONNX
///
/// Automatically traverses the graph backwards from `output`, identifying
/// parameters (requires_grad=true) and inputs (requires_grad=false).
///
/// # Arguments
/// * `output` - The output tensor of the model (must be a computed tensor)
/// * `path` - Path to save the .onnx (json) file
pub fn export_to_onnx(output: &Tensor, path: &str) -> Result<()> {
    let json = export_to_json(output)?;
    std::fs::write(path, json)?;
    Ok(())
}
