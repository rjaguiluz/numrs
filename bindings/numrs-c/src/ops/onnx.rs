use crate::tensor::NumRsTensor;
use crate::array::NumRsArray;
use numrs::ops::export::{export_to_onnx};
use numrs::ops::model::{load_onnx, save_onnx, infer, OnnxModel};
use numrs::{Tensor, Array};
use std::ffi::{CStr, c_char, CString};
use std::collections::HashMap;

pub struct NumRsOnnxModel {
    pub(crate) inner: OnnxModel,
}


#[no_mangle]
pub unsafe extern "C" fn numrs_onnx_export(output: *mut NumRsTensor, path: *const c_char) -> i32 {
    if output.is_null() || path.is_null() {
        return -1;
    }
    
    let c_path = CStr::from_ptr(path);
    let path_str = match c_path.to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    
    let tensor = &(*output).inner;
    
    // Pass direct reference (tensor is &Tensor)
    match export_to_onnx(tensor, path_str) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("ONNX Export Error: {}", e);
            -3
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn numrs_onnx_load(path: *const c_char) -> *mut NumRsOnnxModel {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    
    let c_path = CStr::from_ptr(path);
    let path_str = match c_path.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    
    match load_onnx(path_str) {
        Ok(model) => Box::into_raw(Box::new(NumRsOnnxModel { inner: model })),
        Err(e) => {
            eprintln!("ONNX Load Error: {}", e);
            std::ptr::null_mut()
        }
    }
}


#[no_mangle]
pub unsafe extern "C" fn numrs_onnx_free(ptr: *mut NumRsOnnxModel) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr);
    }
}

// Simple Inference: Single Input (name="input"), Single Output
// Returns Output Tensor (wrapped in NumRsTensor for now, but usually inference returns Array? 
// Core infer returns HashMap<String, Array>. We wrap Array in Tensor usually for API consistency).
#[no_mangle]
pub unsafe extern "C" fn numrs_onnx_infer_simple(
    model: *mut NumRsOnnxModel, 
    input_data: *mut NumRsArray, 
    input_name: *const c_char
) -> *mut NumRsArray {
    if model.is_null() || input_data.is_null() {
        return std::ptr::null_mut();
    }
    
    let in_name_cstr = if !input_name.is_null() {
        CStr::from_ptr(input_name)
    } else {
        // Default name?
        CStr::from_bytes_with_nul(b"input\0").unwrap()
    };
    
    let in_name = match in_name_cstr.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => "input".to_string(),
    };

    let input_arr = (*input_data).inner.clone();
    
    let mut inputs = HashMap::new();
    inputs.insert(in_name, input_arr);
    
    let m = &(*model).inner;
    
    match infer(m, inputs) {
        Ok(outputs) => {
            // Get first output
            if let Some(first_out) = outputs.values().next() {
                Box::into_raw(Box::new(NumRsArray { inner: first_out.clone() }))
            } else {
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            eprintln!("ONNX Inference Error: {}", e);
            std::ptr::null_mut()
        }
    }
}
