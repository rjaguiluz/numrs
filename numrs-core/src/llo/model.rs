//! Model operations for ONNX compatibility
//! 
//! This module defines operations for training, saving, and loading models
//! compatible with ONNX format.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Model save/load operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelKind {
    /// Save model to ONNX format
    SaveONNX,
    /// Load model from ONNX format
    LoadONNX,
    /// Export model graph
    ExportGraph,
    /// Import model graph
    ImportGraph,
}

/// Training-related operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingKind {
    /// Forward pass
    Forward,
    /// Backward pass (gradient computation)
    Backward,
    /// Update weights (gradient descent step)
    UpdateWeights,
    /// Compute loss
    Loss,
}

/// Loss function types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LossKind {
    /// Mean Squared Error
    MSE,
    /// Cross Entropy
    CrossEntropy,
    /// Binary Cross Entropy
    BinaryCrossEntropy,
    /// L1 Loss (Mean Absolute Error)
    L1,
    /// Huber Loss
    Huber,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizerKind {
    /// Stochastic Gradient Descent
    SGD { learning_rate: f32, momentum: f32 },
    /// Adam optimizer
    Adam { learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32 },
    /// RMSprop optimizer
    RMSprop { learning_rate: f32, decay: f32, epsilon: f32 },
    /// AdaGrad optimizer
    AdaGrad { learning_rate: f32, epsilon: f32 },
}

/// ONNX Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: i64,
    /// Model author
    pub author: String,
    /// Model description
    pub description: String,
    /// Model domain (reverse DNS)
    pub domain: String,
    /// ONNX opset version
    pub opset_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "numrs_model".to_string(),
            version: 1,
            author: "NumRs".to_string(),
            description: "Model created with NumRs".to_string(),
            domain: "ai.numrs".to_string(),
            opset_version: 18, // ONNX 1.13+ uses opset 18
            producer_name: "NumRs".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            metadata: HashMap::new(),
        }
    }
}

/// ONNX Node representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxNode {
    /// Node name
    pub name: String,
    /// Operator type (Add, MatMul, Relu, etc.)
    pub op_type: String,
    /// Input names
    pub inputs: Vec<String>,
    /// Output names
    pub outputs: Vec<String>,
    /// Operator attributes
    pub attributes: HashMap<String, OnnxAttribute>,
}

/// ONNX Attribute value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnnxAttribute {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(Vec<usize>, Vec<f32>), // shape, data
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}

/// ONNX Tensor representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensor {
    /// Tensor name
    pub name: String,
    /// Data type (1=FLOAT, 6=INT32, 7=INT64, 11=DOUBLE)
    pub dtype: i32,
    /// Shape
    pub shape: Vec<usize>,
    /// Raw data bytes
    pub data: Vec<u8>,
}

/// ONNX Graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxGraph {
    /// Graph name
    pub name: String,
    /// List of nodes
    pub nodes: Vec<OnnxNode>,
    /// Input tensors
    pub inputs: Vec<OnnxTensor>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Initializers (constant tensors like weights)
    pub initializers: Vec<OnnxTensor>,
}

/// ONNX Model representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModel {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model graph
    pub graph: OnnxGraph,
}

impl OnnxModel {
    /// Create a new ONNX model
    pub fn new(name: &str) -> Self {
        let mut metadata = ModelMetadata::default();
        metadata.name = name.to_string();
        
        Self {
            metadata,
            graph: OnnxGraph {
                name: name.to_string(),
                nodes: Vec::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                initializers: Vec::new(),
            },
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, node: OnnxNode) {
        self.graph.nodes.push(node);
    }
    
    /// Add an input tensor
    pub fn add_input(&mut self, tensor: OnnxTensor) {
        self.graph.inputs.push(tensor);
    }
    
    /// Add an initializer (weights/constants)
    pub fn add_initializer(&mut self, tensor: OnnxTensor) {
        self.graph.initializers.push(tensor);
    }
    
    /// Set output names
    pub fn set_outputs(&mut self, outputs: Vec<String>) {
        self.graph.outputs = outputs;
    }
}

/// Training state for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Current iteration
    pub iteration: usize,
    /// Current loss value
    pub loss: f32,
    /// Optimizer state
    pub optimizer: OptimizerKind,
    /// Parameter gradients
    pub gradients: HashMap<String, Vec<f32>>,
    /// Optimizer momentum/state variables
    pub optimizer_state: HashMap<String, Vec<f32>>,
}

impl TrainingState {
    /// Create new training state with SGD optimizer
    pub fn new_sgd(learning_rate: f32) -> Self {
        Self {
            epoch: 0,
            iteration: 0,
            loss: 0.0,
            optimizer: OptimizerKind::SGD { 
                learning_rate, 
                momentum: 0.9 
            },
            gradients: HashMap::new(),
            optimizer_state: HashMap::new(),
        }
    }
    
    /// Create new training state with Adam optimizer
    pub fn new_adam(learning_rate: f32) -> Self {
        Self {
            epoch: 0,
            iteration: 0,
            loss: 0.0,
            optimizer: OptimizerKind::Adam {
                learning_rate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            gradients: HashMap::new(),
            optimizer_state: HashMap::new(),
        }
    }
}
