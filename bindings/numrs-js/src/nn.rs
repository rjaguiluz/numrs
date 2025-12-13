use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::Tensor;
use numrs::{
    Linear as CoreLinear, 
    Sequential as CoreSequential, 
    Module, 
    ReLU as CoreReLU, 
    Sigmoid as CoreSigmoid,
    autograd::nn::Softmax as CoreSoftmax,
    Conv1d as CoreConv1d,
    BatchNorm1d as CoreBatchNorm1d,
    Flatten as CoreFlatten,
    Dropout as CoreDropout,
};
use std::rc::Rc;
use std::cell::RefCell;

#[napi]
pub struct Linear {
    inner: CoreLinear,
}

#[napi]
impl Linear {
    #[napi(constructor)]
    pub fn new(in_features: u32, out_features: u32) -> Result<Self> {
        let linear = CoreLinear::new(in_features as usize, out_features as usize)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Linear { inner: linear })
    }

    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn parameters(&self) -> Vec<Tensor> {
        self.inner.parameters()
            .into_iter()
            .map(|p| Tensor { inner: p })
            .collect()
    }
}

#[napi]
pub struct ReLU {
    inner: CoreReLU,
}

#[napi]
impl ReLU {
    #[napi(constructor)]
    pub fn new() -> Self {
        ReLU { inner: CoreReLU }
    }

    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}

#[napi]
pub struct Sigmoid {
    inner: CoreSigmoid,
}

#[napi]
impl Sigmoid {
    #[napi(constructor)]
    pub fn new() -> Self {
        Sigmoid { inner: CoreSigmoid }
    }

    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}

#[napi]
pub struct Softmax {
    inner: CoreSoftmax,
}

#[napi]
impl Softmax {
    #[napi(constructor)]
    pub fn new() -> Self {
        Softmax { inner: CoreSoftmax }
    }

    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}

// --- New Layers ---

#[napi]
pub struct Conv1d {
    inner: CoreConv1d,
}

#[napi]
impl Conv1d {
    #[napi(constructor)]
    pub fn new(in_channels: u32, out_channels: u32, kernel_size: u32, stride: Option<u32>, padding: Option<u32>) -> Result<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        
        let layer = CoreConv1d::new(
            in_channels as usize, 
            out_channels as usize, 
            kernel_size as usize, 
            stride as usize, 
            padding as usize
        ).map_err(|e| Error::from_reason(e.to_string()))?;
            
        Ok(Conv1d { inner: layer })
    }
    
    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}

#[napi]
pub struct Flatten {
    inner: CoreFlatten,
}

#[napi]
impl Flatten {
    #[napi(constructor)]
    pub fn new(start_dim: u32, end_dim: u32) -> Self {
        Flatten { inner: CoreFlatten::new(start_dim as usize, end_dim as usize) }
    }
    
    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}

#[napi]
pub struct Dropout {
    inner: CoreDropout,
}

#[napi]
impl Dropout {
    #[napi(constructor)]
    pub fn new(p: f64) -> Self {
        Dropout { inner: CoreDropout::new(p as f32) }
    }
    
    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
}


#[napi]
pub struct BatchNorm1d {
    inner: CoreBatchNorm1d,
}

#[napi]
impl BatchNorm1d {
    #[napi(constructor)]
    pub fn new(num_features: u32) -> Result<Self> {
        let layer = CoreBatchNorm1d::new(num_features as usize)
             .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(BatchNorm1d { inner: layer })
    }
    
    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
    
    #[napi]
    pub fn train(&mut self) {
        self.inner.train();
    }
    
    #[napi]
    pub fn eval(&mut self) {
        self.inner.eval();
    }
}

// Sequential needs a bit more work because it holds Box<dyn Module>
// For simplicity in binding, we can allow constructing from a list of known layers (simulated)
// Or just bind it if possible. napi doesn't support generic traits well.
// We might need to implement add() for specific types or a generic wrapper?
// NAPI doesn't allow "Box<dyn Module>" easily across FFI without wrapping.
// Strategy: Construct empty, then add layers? But add takes inner trait object.
// We can make Sequential hold specific variants in an enum or just simplified binding for now?
// Actually, `numrs-core` Sequential::new takes `Vec<Box<dyn Module>>`.
// We can't easily expose `Box<dyn Module>` to JS.
// Workaround: exposing a JS-side Sequential that manages a list, but that defeats "minimize JS".
// Better: Helper function that takes specific known binding types and converts them to Box<dyn Module>.
// But NAPI constructor can't take mixed types easily.
// Let's implement Sequential::new taking a specialized list? Or just skip Sequential binding for now and let user chain them in JS?
// User asked to use what core provides.
// Let's implement a limited Sequential that can take a predefined list of layers, OR rely on JS composition.
// Actually, Python `nn.Sequential` behaves like a container.
// Let's skip Sequential binding in `nn.rs` for a moment and just let user use `Linear`, `ReLU` etc. in loop or array.
// But `fraud_detection` uses `Sequential`.
// Let's implement `Sequential` in `src/nn.rs` but make `add_linear`, `add_relu` helper methods?
// Or: constructor takes `Vec<JsValue>` and we sniff types?
// NAPI allows `JsObject`. We can check instance type.
// But we need to extract the `inner` from binding wrappers. This requires `napi::unwrap`.

// Let's implement Sequential with specific `add` methods for now: `add_linear`, `add_relu`, `add_sigmoid`.
// And a `forward`.

#[napi]
pub struct Sequential {
    inner: CoreSequential,
}

#[napi]
impl Sequential {
    #[napi(constructor)]
    pub fn new() -> Self {
        Sequential { 
            inner: CoreSequential::new(vec![]),
        }
    }
    
    // Helper to clone the wrapper's inner model reference
    pub(crate) fn clone_inner(&self) -> CoreSequential {
        self.inner.clone()
    }

    #[napi]
    pub fn add_linear(&mut self, layer: &Linear) {
        self.inner.add(Box::new(layer.inner.clone()));
    }
    
    #[napi]
    pub fn add_relu(&mut self, _layer: &ReLU) {
        self.inner.add(Box::new(CoreReLU));
    }

    #[napi]
    pub fn add_sigmoid(&mut self, _layer: &Sigmoid) {
        self.inner.add(Box::new(CoreSigmoid));
    }

    #[napi]
    pub fn add_softmax(&mut self, _layer: &Softmax) {
        self.inner.add(Box::new(CoreSoftmax));
    }

    #[napi]
    pub fn add_conv1d(&mut self, layer: &Conv1d) {
        self.inner.add(Box::new(layer.inner.clone()));
    }
    
    #[napi]
    pub fn add_batch_norm1d(&mut self, layer: &BatchNorm1d) {
        self.inner.add(Box::new(layer.inner.clone()));
    }
    
    #[napi]
    pub fn add_flatten(&mut self, layer: &Flatten) {
        self.inner.add(Box::new(layer.inner.clone()));
    }
    
    #[napi]
    pub fn add_dropout(&mut self, layer: &Dropout) {
         self.inner.add(Box::new(layer.inner.clone()));
    }

    #[napi]
    pub fn add_unsqueeze(&mut self, dim: u32) {
        self.inner.add(Box::new(UnsqueezeModule { dim: dim as usize }));
    }

    #[napi]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let res = self.inner.forward(&input.inner.borrow())
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
    
    #[napi]
    pub fn parameters(&self) -> Vec<Tensor> {
        self.inner.parameters()
            .into_iter()
            .map(|p| Tensor { inner: p })
            .collect()
    }

    #[napi]
    pub fn save_onnx(&self, input: &Tensor, path: String) -> Result<()> {
        // Run forward pass to build graph
        let output = self.inner.forward(&input.inner.borrow())
             .map_err(|e| Error::from_reason(e.to_string()))?;
             
        numrs::ops::export::export_to_onnx(&output, &path)
            .map_err(|e| Error::from_reason(e.to_string()))?;
            
        Ok(())
    }
}

// Internal helper module for reshaping (Unsqueeze) to support CNN inputs
// Not exposed to JS as a class, but added via Sequential.addUnsqueeze(dim)
#[derive(Clone)]
struct UnsqueezeModule {
    dim: usize,
}

impl Module for UnsqueezeModule {
    fn forward(&self, input: &numrs::Tensor) -> std::result::Result<numrs::Tensor, anyhow::Error> {
        let mut new_shape = input.shape().to_vec();
        // Check bounds
        if self.dim > new_shape.len() {
             return Err(anyhow::anyhow!("Unsqueeze dimension out of bounds"));
        }
        new_shape.insert(self.dim, 1);
        input.reshape(new_shape)
    }
    
    fn parameters(&self) -> Vec<Rc<RefCell<numrs::Tensor>>> {
        vec![]
    }
    
    fn train(&mut self) {}
    fn eval(&mut self) {}
    
    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}
