//! GPU Operation Descriptors
//!
//! Este módulo define los descriptores que WASM envía a JavaScript
//! para ejecutar operaciones en WebGPU.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Tipos de operaciones GPU soportadas
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(u8)]
pub enum GpuOpType {
    Add = 0,
    Mul = 1,
    Sub = 2,
    Div = 3,
    Sin = 4,
    Cos = 5,
    Exp = 6,
    Log = 7,
    Sqrt = 8,
    Matmul = 9,
    Sum = 10,
    Mean = 11,
}

/// Descriptor de operación GPU
/// Este struct se serializa a JSON y se envía al host JS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOpDescriptor {
    pub op_type: GpuOpType,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub workgroup_size: [u32; 3],
    pub num_workgroups: [u32; 3],
}

impl GpuOpDescriptor {
    /// Crear descriptor para operación binaria elementwise
    pub fn elementwise_binary(op_type: GpuOpType, shape: &[usize]) -> Self {
        let total_elements = shape.iter().product::<usize>() as u32;
        let workgroup_size = 256;
        let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;

        GpuOpDescriptor {
            op_type,
            input_shapes: vec![shape.to_vec(), shape.to_vec()],
            output_shape: shape.to_vec(),
            workgroup_size: [workgroup_size, 1, 1],
            num_workgroups: [num_workgroups, 1, 1],
        }
    }

    /// Crear descriptor para operación unaria elementwise
    pub fn elementwise_unary(op_type: GpuOpType, shape: &[usize]) -> Self {
        let total_elements = shape.iter().product::<usize>() as u32;
        let workgroup_size = 256;
        let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;

        GpuOpDescriptor {
            op_type,
            input_shapes: vec![shape.to_vec()],
            output_shape: shape.to_vec(),
            workgroup_size: [workgroup_size, 1, 1],
            num_workgroups: [num_workgroups, 1, 1],
        }
    }

    /// Crear descriptor para matmul
    pub fn matmul(m: usize, k: usize, n: usize) -> Self {
        // Tile size 16x16
        let tile_size = 16u32;
        let wg_x = ((n as u32) + tile_size - 1) / tile_size;
        let wg_y = ((m as u32) + tile_size - 1) / tile_size;

        GpuOpDescriptor {
            op_type: GpuOpType::Matmul,
            input_shapes: vec![vec![m, k], vec![k, n]],
            output_shape: vec![m, n],
            workgroup_size: [tile_size, tile_size, 1],
            num_workgroups: [wg_x, wg_y, 1],
        }
    }

    /// Crear descriptor para reducción
    pub fn reduction(op_type: GpuOpType, shape: &[usize]) -> Self {
        let _total_elements = shape.iter().product::<usize>() as u32;
        let workgroup_size = 256;

        GpuOpDescriptor {
            op_type,
            input_shapes: vec![shape.to_vec()],
            output_shape: vec![1],
            workgroup_size: [workgroup_size, 1, 1],
            num_workgroups: [1, 1, 1],
        }
    }
}

/// Convertir descriptor a JsValue para pasar a JavaScript
#[wasm_bindgen]
pub fn create_gpu_descriptor_add(shape: Vec<usize>) -> JsValue {
    let descriptor = GpuOpDescriptor::elementwise_binary(GpuOpType::Add, &shape);
    serde_wasm_bindgen::to_value(&descriptor).unwrap()
}

#[wasm_bindgen]
pub fn create_gpu_descriptor_mul(shape: Vec<usize>) -> JsValue {
    let descriptor = GpuOpDescriptor::elementwise_binary(GpuOpType::Mul, &shape);
    serde_wasm_bindgen::to_value(&descriptor).unwrap()
}

#[wasm_bindgen]
pub fn create_gpu_descriptor_sin(shape: Vec<usize>) -> JsValue {
    let descriptor = GpuOpDescriptor::elementwise_unary(GpuOpType::Sin, &shape);
    serde_wasm_bindgen::to_value(&descriptor).unwrap()
}

#[wasm_bindgen]
pub fn create_gpu_descriptor_matmul(m: usize, k: usize, n: usize) -> JsValue {
    let descriptor = GpuOpDescriptor::matmul(m, k, n);
    serde_wasm_bindgen::to_value(&descriptor).unwrap()
}

#[wasm_bindgen]
pub fn create_gpu_descriptor_sum(shape: Vec<usize>) -> JsValue {
    let descriptor = GpuOpDescriptor::reduction(GpuOpType::Sum, &shape);
    serde_wasm_bindgen::to_value(&descriptor).unwrap()
}
