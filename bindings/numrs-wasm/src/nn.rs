use crate::tensor::Tensor;
use numrs::{
    autograd::nn::Softmax as CoreSoftmax, BatchNorm1d as CoreBatchNorm1d, Conv1d as CoreConv1d,
    Dropout as CoreDropout, Flatten as CoreFlatten, Linear as CoreLinear, Module, ReLU as CoreReLU,
    Sequential as CoreSequential, Sigmoid as CoreSigmoid,
};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Linear {
    pub(crate) inner: CoreLinear,
}

#[wasm_bindgen]
impl Linear {
    #[wasm_bindgen(constructor)]
    pub fn new(in_features: usize, out_features: usize) -> Result<Linear, JsValue> {
        let linear = CoreLinear::new(in_features, out_features)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Linear { inner: linear })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| Tensor { inner: p })
            .collect()
    }
}

#[wasm_bindgen]
pub struct ReLU {
    pub(crate) inner: CoreReLU,
}

#[wasm_bindgen]
impl ReLU {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ReLU {
        ReLU { inner: CoreReLU }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }
}

#[wasm_bindgen]
pub struct Sigmoid {
    pub(crate) inner: CoreSigmoid,
}

#[wasm_bindgen]
impl Sigmoid {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Sigmoid {
        Sigmoid { inner: CoreSigmoid }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }
}

#[wasm_bindgen]
pub struct Softmax {
    pub(crate) inner: CoreSoftmax,
}

#[wasm_bindgen]
impl Softmax {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Softmax {
        Softmax { inner: CoreSoftmax }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }
}

#[wasm_bindgen]
pub struct Conv1d {
    pub(crate) inner: CoreConv1d,
}

#[wasm_bindgen]
impl Conv1d {
    #[wasm_bindgen(constructor)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
    ) -> Result<Conv1d, JsValue> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);

        let layer = CoreConv1d::new(in_channels, out_channels, kernel_size, stride, padding)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Conv1d { inner: layer })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }
}

#[wasm_bindgen]
pub struct Flatten {
    pub(crate) inner: CoreFlatten,
}

#[wasm_bindgen]
impl Flatten {
    #[wasm_bindgen(constructor)]
    pub fn new(start_dim: usize, end_dim: usize) -> Flatten {
        Flatten {
            inner: CoreFlatten::new(start_dim, end_dim),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }
}

#[wasm_bindgen]
pub struct Dropout {
    pub(crate) inner: CoreDropout,
}

#[wasm_bindgen]
impl Dropout {
    #[wasm_bindgen(constructor)]
    pub fn new(p: f32) -> Dropout {
        Dropout {
            inner: CoreDropout::new(p),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }
}

#[wasm_bindgen]
pub struct BatchNorm1d {
    pub(crate) inner: CoreBatchNorm1d,
}

#[wasm_bindgen]
impl BatchNorm1d {
    #[wasm_bindgen(constructor)]
    pub fn new(num_features: usize) -> Result<BatchNorm1d, JsValue> {
        let layer =
            CoreBatchNorm1d::new(num_features).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(BatchNorm1d { inner: layer })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn train(&mut self) {
        self.inner.train();
    }

    pub fn eval(&mut self) {
        self.inner.eval();
    }
}

#[wasm_bindgen]
pub struct Sequential {
    pub(crate) inner: CoreSequential,
}

#[wasm_bindgen]
impl Sequential {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Sequential {
        Sequential {
            inner: CoreSequential::new(vec![]),
        }
    }

    pub fn add_linear(&mut self, layer: &Linear) {
        self.inner.add(Box::new(layer.inner.clone()));
    }

    pub fn add_relu(&mut self, _layer: &ReLU) {
        self.inner.add(Box::new(CoreReLU));
    }

    pub fn add_sigmoid(&mut self, _layer: &Sigmoid) {
        self.inner.add(Box::new(CoreSigmoid));
    }

    pub fn add_softmax(&mut self, _layer: &Softmax) {
        self.inner.add(Box::new(CoreSoftmax));
    }

    pub fn add_conv1d(&mut self, layer: &Conv1d) {
        self.inner.add(Box::new(layer.inner.clone()));
    }

    pub fn add_batch_norm1d(&mut self, layer: &BatchNorm1d) {
        self.inner.add(Box::new(layer.inner.clone()));
    }

    pub fn add_flatten(&mut self, layer: &Flatten) {
        self.inner.add(Box::new(layer.inner.clone()));
    }

    pub fn add_dropout(&mut self, layer: &Dropout) {
        self.inner.add(Box::new(layer.inner.clone()));
    }

    pub fn add_unsqueeze(&mut self, dim: usize) {
        self.inner.add(Box::new(UnsqueezeModule { dim }));
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.inner
            .parameters()
            .into_iter()
            .map(|p| Tensor { inner: p })
            .collect()
    }
}

// Internal helper for Unsqueeze
#[derive(Clone)]
struct UnsqueezeModule {
    dim: usize,
}

impl Module for UnsqueezeModule {
    fn forward(&self, input: &numrs::Tensor) -> std::result::Result<numrs::Tensor, anyhow::Error> {
        let mut new_shape = input.shape().to_vec();
        // Check bounds logic or just let reshape fail if invalid
        if self.dim > new_shape.len() {
            // Convert string error to anyhow::Error manually if needed, or better:
            // numrs-core probably uses anyhow::Error for Module::forward
            return Err(anyhow::Error::msg("Unsqueeze dimension out of bounds"));
        }
        new_shape.insert(self.dim, 1);
        input.reshape(new_shape)
    }

    // Logic for parameters in core Module trait returns Vec<Rc<RefCell<Tensor>>>
    fn parameters(&self) -> Vec<Rc<RefCell<numrs::Tensor>>> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }
}
