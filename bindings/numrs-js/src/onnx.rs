use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::tensor::Tensor;
use crate::NumRsArray;
use numrs::ops::export::export_to_onnx;
use numrs::ops::model::{load_onnx, infer, OnnxModel as CoreOnnxModel};
use std::collections::HashMap;

#[napi]
pub fn save_onnx(tensor: &Tensor, path: String) -> Result<()> {
    export_to_onnx(&tensor.inner.borrow(), &path)
        .map_err(|e| Error::from_reason(e.to_string()))
}

#[napi]
pub struct OnnxModel {
    pub(crate) inner: CoreOnnxModel,
}

#[napi]
impl OnnxModel {
    /// Load model from path
    #[napi(factory)]
    pub fn load(path: String) -> Result<Self> {
        let model = load_onnx(&path).map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(OnnxModel { inner: model })
    }

    /// Run inference
    /// For simplicity in JS, currently supports single input (name="input") -> single output
    /// Returns NumRsArray (raw data), user can wrap in Tensor if needed.
    #[napi]
    pub fn infer(&self, input: &NumRsArray, input_name: Option<String>) -> Result<NumRsArray> {
        let name = input_name.unwrap_or_else(|| "input".to_string());
        
        let mut inputs = HashMap::new();
        inputs.insert(name, input.inner.clone());
        
        let outputs = infer(&self.inner, inputs)
            .map_err(|e| Error::from_reason(e.to_string()))?;
            
        // Return first output found
        if let Some(out_arr) = outputs.values().next() {
            Ok(NumRsArray { inner: out_arr.clone() })
        } else {
            Err(Error::from_reason("Model produced no outputs"))
        }
    }
}
