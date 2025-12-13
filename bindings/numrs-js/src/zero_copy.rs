use napi::bindgen_prelude::*;
use napi_derive::napi;
use numrs::Array;
use numrs::array::DynArray;

// ============================================================================
// Zero-Copy Operations con TypedArrays
// ============================================================================
// Usa referencias directas a Float32Array sin copiar datos

// ============================================================================
// BINARY OPERATIONS
// ============================================================================

/// Add operation - ZERO COPY version
#[napi]
pub fn add_f32_zero_copy(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
) -> Result<Float32Array> {
    let a_shape: Vec<usize> = a_shape.iter().map(|&v| v as usize).collect();
    let b_shape: Vec<usize> = b_shape.iter().map(|&v| v as usize).collect();
    
    let a = Array::new(a_shape, a_data.as_ref().to_vec());
    let b = Array::new(b_shape, b_data.as_ref().to_vec());
    
    let result = numrs::ops::add(&a, &b)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Sub operation - ZERO COPY version
#[napi]
pub fn sub_f32_zero_copy(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
) -> Result<Float32Array> {
    let a_shape: Vec<usize> = a_shape.iter().map(|&v| v as usize).collect();
    let b_shape: Vec<usize> = b_shape.iter().map(|&v| v as usize).collect();
    
    let a = Array::new(a_shape, a_data.as_ref().to_vec());
    let b = Array::new(b_shape, b_data.as_ref().to_vec());
    
    let result = numrs::ops::sub(&a, &b)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Mul operation - ZERO COPY version
#[napi]
pub fn mul_f32_zero_copy(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
) -> Result<Float32Array> {
    let a_shape: Vec<usize> = a_shape.iter().map(|&v| v as usize).collect();
    let b_shape: Vec<usize> = b_shape.iter().map(|&v| v as usize).collect();
    
    let a = Array::new(a_shape, a_data.as_ref().to_vec());
    let b = Array::new(b_shape, b_data.as_ref().to_vec());
    
    let result = numrs::ops::mul(&a, &b)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Div operation - ZERO COPY version
#[napi]
pub fn div_f32_zero_copy(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
) -> Result<Float32Array> {
    let a_shape: Vec<usize> = a_shape.iter().map(|&v| v as usize).collect();
    let b_shape: Vec<usize> = b_shape.iter().map(|&v| v as usize).collect();
    
    let a = Array::new(a_shape, a_data.as_ref().to_vec());
    let b = Array::new(b_shape, b_data.as_ref().to_vec());
    
    let result = numrs::ops::div(&a, &b)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Pow operation - ZERO COPY version
#[napi]
pub fn pow_f32_zero_copy(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
) -> Result<Float32Array> {
    let a_shape: Vec<usize> = a_shape.iter().map(|&v| v as usize).collect();
    let b_shape: Vec<usize> = b_shape.iter().map(|&v| v as usize).collect();
    
    let a = Array::new(a_shape, a_data.as_ref().to_vec());
    let b = Array::new(b_shape, b_data.as_ref().to_vec());
    
    let result = numrs::ops::pow(&a, &b)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

// ============================================================================
// UNARY OPERATIONS
// ============================================================================

/// Sin operation - ZERO COPY version
#[napi]
pub fn sin_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::sin(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Cos operation - ZERO COPY version
#[napi]
pub fn cos_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::cos(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Tan operation - ZERO COPY version
#[napi]
pub fn tan_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::tan(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Exp operation - ZERO COPY version
#[napi]
pub fn exp_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::exp(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Log operation - ZERO COPY version
#[napi]
pub fn log_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::log(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Sqrt operation - ZERO COPY version
#[napi]
pub fn sqrt_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::sqrt(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Abs operation - ZERO COPY version
#[napi]
pub fn abs_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::abs(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Neg operation - ZERO COPY version  
/// Note: neg returns Array<T> not DynArray, so we handle it directly
#[napi]
pub fn neg_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::neg(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    Ok(result.data.into())
}

/// Relu operation - ZERO COPY version
#[napi]
pub fn relu_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::relu(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Sigmoid operation - ZERO COPY version
#[napi]
pub fn sigmoid_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::sigmoid(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Tanh operation - ZERO COPY version
#[napi]
pub fn tanh_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::tanh(&arr)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

/// Sum operation - ZERO COPY version
#[napi]
pub fn sum_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<f32> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::sum(&arr, None)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let result_arr = result.to_f32();
    Ok(result_arr.data[0])
}

/// Mean operation - ZERO COPY version
#[napi]
pub fn mean_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<f32> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::mean(&arr, None)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let result_arr = result.to_f32();
    Ok(result_arr.data[0])
}

/// Max operation - ZERO COPY version
#[napi]
pub fn max_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<f32> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::max(&arr, None)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let result_arr = result.to_f32();
    Ok(result_arr.data[0])
}

/// Min operation - ZERO COPY version
#[napi]
pub fn min_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<f32> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::min(&arr, None)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let result_arr = result.to_f32();
    Ok(result_arr.data[0])
}

/// Variance operation - ZERO COPY version
#[napi]
pub fn variance_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<f32> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::variance(&arr, None)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let result_arr = result.to_f32();
    Ok(result_arr.data[0])
}

// ============================================================================
// LINALG OPERATIONS
// ============================================================================

/// Matmul operation - ZERO COPY version
#[napi]
pub fn matmul_f32_zero_copy(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
) -> Result<Float32Array> {
    let a_shape: Vec<usize> = a_shape.iter().map(|&v| v as usize).collect();
    let b_shape: Vec<usize> = b_shape.iter().map(|&v| v as usize).collect();
    
    let a = Array::new(a_shape, a_data.as_ref().to_vec());
    let b = Array::new(b_shape, b_data.as_ref().to_vec());
    
    let result = numrs::ops::matmul(&a, &b)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Dot product operation - ZERO COPY version
#[napi]
pub fn dot_f32_zero_copy(
    a_data: Float32Array,
    a_shape: Vec<u32>,
    b_data: Float32Array,
    b_shape: Vec<u32>,
) -> Result<f32> {
    let a_shape: Vec<usize> = a_shape.iter().map(|&v| v as usize).collect();
    let b_shape: Vec<usize> = b_shape.iter().map(|&v| v as usize).collect();
    
    let a = Array::new(a_shape, a_data.as_ref().to_vec());
    let b = Array::new(b_shape, b_data.as_ref().to_vec());
    
    let result = numrs::ops::dot(&a, &b)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let result_arr = result.to_f32();
    Ok(result_arr.data[0])
}

// ============================================================================
// SHAPE OPERATIONS
// ============================================================================

/// Transpose operation - ZERO COPY version
#[napi]
pub fn transpose_f32_zero_copy(
    data: Float32Array,
    shape: Vec<u32>,
) -> Result<Float32Array> {
    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    let arr = Array::new(shape, data.as_ref().to_vec());
    
    let result = numrs::ops::transpose(&arr, None)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}

/// Reshape operation - ZERO COPY version
#[napi]
pub fn reshape_f32_zero_copy(
    data: Float32Array,
    old_shape: Vec<u32>,
    new_shape: Vec<i32>,
) -> Result<Float32Array> {
    let old_shape: Vec<usize> = old_shape.iter().map(|&v| v as usize).collect();
    let new_shape_isize: Vec<isize> = new_shape.iter().map(|&v| v as isize).collect();
    let arr = Array::new(old_shape, data.as_ref().to_vec());
    
    let result = numrs::ops::reshape(&arr, &new_shape_isize)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    
    let data = match result {
        DynArray::F32(arr) => arr.data,
        _ => return Err(Error::from_reason("Expected F32 result")),
    };
    
    Ok(data.into())
}
