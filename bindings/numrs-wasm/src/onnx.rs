use crate::nn::Sequential;
use crate::tensor::Tensor;
use numrs::llo::OnnxModel;
use numrs::ops::export::export_to_json as core_export_to_json;
use numrs::ops::model::deserialize_onnx as core_deserialize_onnx;
use numrs::Module;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct OnnxModelWrapper {
    pub(crate) inner: OnnxModel,
}

#[wasm_bindgen]
impl OnnxModelWrapper {
    // Static method (no self) automatically becomes static on class
    pub fn load_from_json(json: &str) -> Result<OnnxModelWrapper, JsValue> {
        let model = core_deserialize_onnx(json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(OnnxModelWrapper { inner: model })
    }

    // Static method
    pub fn export_model_to_json(model: &Sequential, input: &Tensor) -> Result<String, JsValue> {
        let output = model
            .inner
            .forward(&input.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        core_export_to_json(&output).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    //     for (key, val) in results {
    //         // val is numrs::Array<T>. We need to ensure it's f32 for WASM Array.
    //         // core_infer usually works on f32 for now. Assume f32.
    //         // If it's not f32, we might panic or need to cast.
    //         // Let's assume f32.

    //         // To create WASM Array we use its constructor new(data, shape)
    //         // But data is Vec<f32>.
    //         let data = val.data; // This works if T is f32.
    //                              // If val is NOT f32 compilation will fail here if generic.
    //                              // Assuming strict f32 in context or using `into_typed` if `Array` is dynamic.
    //                              // Wait, core Array is generic struct Array<T>.
    //                              // `core_infer` return `HashMap<String, Array>`.
    //                              // Wait, standard `Array` struct requires T.
    //                              // Ah, `use numrs::Array` imports struct Array. It doesn't bind T.
    //                              // `core_infer` signature: `HashMap<String, Array>`. This means `HashMap<String, Array<f32>>` default?
    //                              // Actually checking `model.rs` signature: `pub fn infer(...) -> Result<HashMap<String, Array>>`.
    //                              // This implies `Array` is used as a type. Maybe `Array` is an alias or struct without params in some contexts?
    //                              // In core/lib.rs: `pub use array::{Array, ...}`. Array definition `pub struct Array<T: DTypeValue>`.
    //                              // So `Array` requires generic args.
    //                              // If `model.rs` uses `HashMap<String, Array>`, it is an error unless `Array` default is set or it refers to `DynArray`?
    //                              // Wait, `model.rs` line 254: `HashMap<String, Array>`.
    //                              // Maybe `type Array = Array<f32>` in `model.rs`?
    //                              // checking imports in `model.rs`: `use crate::array::{Array, DTypeValue}`.
    //                              // This is confusing unless `Array` has default `T=f32`? Rust 1.x?

    //         // Re-check `model.rs` content I viewed.
    //         // Line 9: `use crate::array::{Array, DTypeValue};`
    //         // Line 254: `-> Result<HashMap<String, Array>>`
    //         // This suggests Array defaults to something or I missed a type alias.
    //         // But regardless, assuming f32 is safe for WASM context usually.

    //         let wasm_array = crate::Array::new(data, val.shape);
    //         output_map.set(&JsValue::from_str(&key), &JsValue::from(wasm_array));
    //     }

    //     Ok(output_map)
    // }
}
