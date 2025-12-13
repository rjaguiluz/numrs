use wasm_bindgen::prelude::*;
// use crate::tensor::Tensor; // Unused
use std::sync::atomic::{AtomicBool, Ordering};

// Default allocator is used
// #[global_allocator]
// static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// mod gpu_ops;
// mod memory; // Already empty of functions
pub mod nn;
pub mod onnx;
pub mod tensor;
pub mod train;
pub use onnx::OnnxModelWrapper;
pub use train::Trainer;

// pub use gpu_ops::*;
// pub use memory::*;

// Global flag to track WebGPU availability (set from JS)
static WEBGPU_AVAILABLE: AtomicBool = AtomicBool::new(false);

// Inicialización con panic hook para debugging
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

// ========================================
// WebGPU Initialization from JavaScript
// ========================================

/// Initialize WebGPU backend asynchronously.
/// Must be called and awaited from JavaScript before using any WebGPU features.
#[wasm_bindgen]
pub async fn init_webgpu() -> Result<(), JsValue> {
    numrs::init_webgpu_wasm()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Set WebGPU availability flag (called from JS after successful init)
#[wasm_bindgen]
pub fn set_webgpu_available(available: bool) {
    numrs::set_webgpu_available_wasm(available);
}

/// Force re-initialization of the backend dispatch table.
/// Useful after async WebGPU init to switch from CPU to GPU.
#[wasm_bindgen]
pub fn init_backend() {
    numrs::backend::dispatch::force_reinitialize_dispatch();
}

// ========================================
// Array Wrapper for WASM
// ========================================

#[wasm_bindgen]
pub struct NumRsArray {
    inner: numrs::Array<f32>,
}

#[wasm_bindgen]
impl NumRsArray {
    #[wasm_bindgen(constructor)]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> NumRsArray {
        NumRsArray {
            inner: numrs::Array::new(shape, data),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.inner.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ndim(&self) -> usize {
        self.inner.shape.len()
    }

    #[wasm_bindgen]
    pub fn size(&self) -> usize {
        self.inner.data.len()
    }

    #[wasm_bindgen]
    pub fn zeros(shape: Vec<usize>) -> NumRsArray {
        NumRsArray {
            inner: numrs::Array::<f32>::zeros(shape),
        }
    }

    #[wasm_bindgen]
    pub fn ones(shape: Vec<usize>) -> NumRsArray {
        NumRsArray {
            inner: numrs::Array::<f32>::ones(shape),
        }
    }

    #[wasm_bindgen]
    pub fn to_string(&self) -> String {
        format!("{:?}", self.inner)
    }

    #[wasm_bindgen]
    pub fn reshape(&self, shape: Vec<usize>) -> Result<NumRsArray, JsValue> {
        let shape_isize: Vec<isize> = shape.iter().map(|&x| x as isize).collect();
        let res = numrs::ops::shape::reshape(&self.inner, &shape_isize)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(NumRsArray { inner: res })
    }

    #[wasm_bindgen]
    pub fn flatten(&self) -> Result<NumRsArray, JsValue> {
        let ndim = self.inner.shape.len();
        let end = if ndim > 0 { ndim - 1 } else { 0 };
        let res = numrs::ops::shape::flatten::flatten(&self.inner, 0, end)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(NumRsArray { inner: res })
    }

    #[wasm_bindgen]
    pub fn transpose(&self, perm: Option<Vec<usize>>) -> Result<NumRsArray, JsValue> {
        let res = numrs::ops::shape::transpose::transpose(&self.inner, perm.as_deref())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(NumRsArray { inner: res })
    }

    #[wasm_bindgen]
    pub fn broadcast_to(&self, shape: Vec<usize>) -> Result<NumRsArray, JsValue> {
        let res = numrs::ops::shape::broadcast_to::broadcast_to(&self.inner, &shape)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(NumRsArray { inner: res })
    }
}

// ========================================
// Array Operations - Simplified API
// NumRs auto-detects best backend (CPU SIMD, WebGPU, etc.)
// ========================================

// Binary Operations

#[wasm_bindgen]
pub fn add(a: &NumRsArray, b: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result =
        numrs::ops::add(&a.inner, &b.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn sub(a: &NumRsArray, b: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result =
        numrs::ops::sub(&a.inner, &b.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn mul(a: &NumRsArray, b: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result =
        numrs::ops::mul(&a.inner, &b.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn div(a: &NumRsArray, b: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result =
        numrs::ops::div(&a.inner, &b.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn matmul(a: &NumRsArray, b: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result =
        numrs::ops::matmul(&a.inner, &b.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(NumRsArray { inner: result })
}

/*
#[wasm_bindgen]
pub fn pow(a: &NumRsArray, b: &NumRsArray) -> Result<NumRsArray, JsValue> {
    // Stub implementation to test signature vs body
    // let result = numrs::ops::elementwise::binary::pow::pow(&a.inner, &b.inner)
    //     .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: a.inner.clone() })
}
*/

// Unary Operations

#[wasm_bindgen]
pub fn sin(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result = numrs::ops::sin(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn cos(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result = numrs::ops::cos(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn exp(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result = numrs::ops::exp(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn sqrt(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result = numrs::ops::sqrt(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn neg(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result = numrs::ops::neg(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn abs(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result = numrs::ops::abs(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn log(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result = numrs::ops::log(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

// Reduction Operations
// (removed redundant start)
#[wasm_bindgen]

pub fn sum(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result =
        numrs::ops::sum(&arr.inner, None).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn mean(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result =
        numrs::ops::mean(&arr.inner, None).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn max(arr: &NumRsArray, axis: Option<i32>) -> Result<NumRsArray, JsValue> {
    let ax = axis.map(|d| d as usize);
    let result = numrs::ops::reduction::max::max(&arr.inner, ax)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn min(arr: &NumRsArray, axis: Option<i32>) -> Result<NumRsArray, JsValue> {
    let ax = axis.map(|d| d as usize);
    let result = numrs::ops::reduction::min::min(&arr.inner, ax)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn argmax(arr: &NumRsArray, axis: Option<i32>) -> Result<NumRsArray, JsValue> {
    let ax = axis.map(|d| d as usize);
    let result = numrs::ops::reduction::argmax::argmax(&arr.inner, ax)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn variance(arr: &NumRsArray, axis: Option<i32>) -> Result<NumRsArray, JsValue> {
    let ax = axis.map(|d| d as usize);
    let result = numrs::ops::reduction::variance::variance(&arr.inner, ax)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

#[wasm_bindgen]
pub fn norm(arr: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result =
        numrs::ops::stats::norm::norm(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

// Linear Algebra

#[wasm_bindgen]
pub fn dot(a: &NumRsArray, b: &NumRsArray) -> Result<NumRsArray, JsValue> {
    let result = numrs::ops::linalg::dot::dot(&a.inner, &b.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(NumRsArray { inner: result })
}

// Shape Operations

#[wasm_bindgen]
pub fn concat(inputs: Vec<NumRsArray>, axis: usize) -> Result<NumRsArray, JsValue> {
    // Collect references to inner arrays
    // inputs is Vec<NumRsArray>, we need Vec<&numrs::Array>
    let inputs_inner: Vec<numrs::Array> = inputs.iter().map(|a| a.inner.clone()).collect();
    let input_refs: Vec<&numrs::Array> = inputs_inner.iter().collect();

    let res_arr = numrs::ops::shape::concat::concat(&input_refs, axis)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(NumRsArray { inner: res_arr })
}

#[wasm_bindgen]
pub fn startup_log() {
    numrs::print_startup_log();
}

#[wasm_bindgen]
pub fn backend_info() -> String {
    let table = numrs::backend::get_dispatch_table();
    let validation = numrs::backend::validate_backends();

    let selected = if validation.webgpu_validated {
        "webgpu"
    } else if validation.simd_validated {
        "cpu-simd"
    } else {
        "cpu-scalar"
    };

    format!(
        r#"{{"selected": "{}", "add": "{}", "mul": "{}", "sum": "{}", "matmul": "{}"}}"#,
        selected,
        table.elementwise_backend,
        table.elementwise_backend,
        table.reduction_backend,
        table.matmul_backend
    )
}
