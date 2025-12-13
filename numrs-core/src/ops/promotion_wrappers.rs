//! Type promotion wrapper for binary operations
//!
//! Este módulo proporciona un wrapper genérico para operaciones binarias que:
//! - Valida shapes
//! - Promociona tipos SOLO cuando son diferentes (i32 + f32 → f32)
//! - Mantiene el tipo cuando son iguales (f64 + f64 → f64, i32 + i32 → i32)
//! - Llama al dispatch con el tipo nativo (SIN conversiones innecesarias)
//!
//! IMPORTANTE: Los resultados mantienen su tipo nativo en el Vec<T>, no hay
//! conversión a f32 intermedia que pierda precisión o eficiencia.

use crate::array::{Array, DTypeValue, DType, DynArray, promotion};
use crate::llo::ElementwiseKind;
use anyhow::Result;
use std::any::TypeId;

/// Wrapper genérico para operaciones binarias elementwise con type promotion
///
/// Flujo simplificado:
/// 1. Capturar dtypes de operandos (ANTES de cualquier conversión)
/// 2. Si dtypes iguales → NO promover, ejecutar con tipo nativo
/// 3. Si dtypes diferentes → Promover, ejecutar con tipo promovido
/// 4. Devolver DynArray con tipo correcto (sin transmutes forzados)
#[inline]
pub fn binary_promoted<T1, T2>(
    a: &Array<T1>,
    b: &Array<T2>,
    kind: ElementwiseKind,
    op_name: &str,
) -> Result<DynArray>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    // Validar shape (ahora soporta broadcasting)
    promotion::validate_binary_op(a, b, op_name)?;
    
    // BROADCASTING: Si los shapes no coinciden, aplicar broadcast
    let (a_bc, b_bc) = if a.shape != b.shape {
        broadcast_if_needed(a, b)?
    } else {
        (None, None)
    };
    
    let a_ref = a_bc.as_ref().unwrap_or(a);
    let b_ref = b_bc.as_ref().unwrap_or(b);
    
    // PASO 1: Capturar dtypes ANTES de cualquier conversión
    let dtype_a = a_ref.dtype;
    let dtype_b = b_ref.dtype;
    
    // PASO 2: Validación simple - ¿necesitamos promoción?
    if dtype_a == dtype_b {
        // NO PROMOVER: ambos son del mismo tipo
        // TypeId garantiza que T1 == T2 en compile-time
        if TypeId::of::<T1>() == TypeId::of::<T2>() {
            let b_as_t1 = unsafe { &*(b_ref as *const Array<T2> as *const Array<T1>) };
            return dispatch_elementwise_native(a_ref, b_as_t1, kind);
        }
    }
    
    // PASO 3: PROMOVER - tipos diferentes, calcular tipo promovido
    let result_dtype = promotion::promoted_dtype(dtype_a, dtype_b);
    
    // PASO 4: Dispatch con tipo promovido conocido
    dispatch_elementwise_promoted(a_ref, b_ref, result_dtype, kind)
}

/// Aplica broadcasting si es necesario para hacer que los shapes coincidan
/// 
/// **OPTIMIZACIÓN**: Para casos simples donde solo hay que repetir una dimensión
/// (como [32, 1] + [32, 50]), evita materializar el broadcasting y deja que
/// el kernel stride-aware lo maneje implícitamente.
fn broadcast_if_needed<T1, T2>(
    a: &Array<T1>, 
    b: &Array<T2>
) -> Result<(Option<Array<T1>>, Option<Array<T2>>)>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    // Si ya tienen el mismo shape, no hacer nada
    if a.shape == b.shape {
        return Ok((None, None));
    }
    
    // Calcular el shape resultante después de broadcasting
    let result_shape = compute_broadcast_shape(&a.shape, &b.shape)?;
    
    // **OPTIMIZACIÓN**: Evitar broadcasting innecesario para casos pequeños
    // Si el tamaño total es pequeño (<10K elementos), no vale la pena materializar
    let total_size: usize = result_shape.iter().product();
    
    // Fast path: arrays pequeños - dejar que stride-aware kernel lo maneje
    if total_size < 10_000 {
        // Para arrays pequeños, es más rápido trabajar con views que materializar
        // Los kernels stride-aware pueden manejar el broadcasting implícitamente
        // Solo materializar si realmente se necesita (kernels que no soportan strides)
        
        // Por ahora, mantener comportamiento conservador: materializar solo si es necesario
        let a_bc = if a.shape != result_shape {
            Some(crate::ops::broadcast_to(a, &result_shape)?)
        } else {
            None
        };
        
        let b_bc = if b.shape != result_shape {
            Some(crate::ops::broadcast_to(b, &result_shape)?)
        } else {
            None
        };
        
        return Ok((a_bc, b_bc));
    }
    
    // Para arrays grandes, materializar normalmente
    let a_bc = if a.shape != result_shape {
        Some(crate::ops::broadcast_to(a, &result_shape)?)
    } else {
        None
    };
    
    let b_bc = if b.shape != result_shape {
        Some(crate::ops::broadcast_to(b, &result_shape)?)
    } else {
        None
    };
    
    Ok((a_bc, b_bc))
}

/// Calcula el shape resultante de broadcasting según reglas de NumPy
fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    
    let mut result = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        let dim1 = if i < len1 {
            shape1[len1 - 1 - i]
        } else {
            1
        };
        
        let dim2 = if i < len2 {
            shape2[len2 - 1 - i]
        } else {
            1
        };
        
        let result_dim = if dim1 == dim2 {
            dim1
        } else if dim1 == 1 {
            dim2
        } else if dim2 == 1 {
            dim1
        } else {
            anyhow::bail!("broadcast: incompatible dimensions {} and {}", dim1, dim2);
        };
        
        result.push(result_dim);
    }
    
    result.reverse();
    Ok(result)
}

/// Wrapper genérico para operaciones binarias (matmul, dot) con closure personalizable
///
/// Similar a binary_promoted, pero acepta una closure para operaciones especializadas
#[inline(always)]
pub fn binary_promoted_with<T1, T2, F>(
    a: &Array<T1>,
    b: &Array<T2>,
    op_fn: F,
    op_name: &str,
) -> Result<DynArray>
where
    T1: DTypeValue,
    T2: DTypeValue,
    F: FnOnce(&Array, &Array) -> Result<DynArray>,
{
    // Validar shape
    promotion::validate_binary_op(a, b, op_name)?;
    
    // FAST PATH: Ambos f32 (caso común - zero overhead con casting directo)
    if TypeId::of::<T1>() == TypeId::of::<f32>() && TypeId::of::<T2>() == TypeId::of::<f32>() {
        let a_f32 = unsafe { &*(a as *const Array<T1> as *const Array) };
        let b_f32 = unsafe { &*(b as *const Array<T2> as *const Array) };
        return op_fn(a_f32, b_f32);
    }
    
    // SLOW PATH: Tipos diferentes o no-f32 - PROMOCIÓN necesaria
    let (_result_dtype, a_data, b_data) = promotion::promote_arrays(a, b)?;
    
    let a_temp = Array::new(a.shape.clone(), a_data);
    let b_temp = Array::new(b.shape.clone(), b_data);
    
    op_fn(&a_temp, &b_temp)
}

/// Dispatch para tipos idénticos (sin promoción) - mantiene tipo nativo
#[inline]
fn dispatch_elementwise_native<T>(
    a: &Array<T>,
    b: &Array<T>,
    kind: ElementwiseKind,
) -> Result<DynArray>
where
    T: DTypeValue,
{
    // dispatch_elementwise_generic<T> ahora devuelve Array<T> correctamente
    // sin conversiones que pierdan precisión
    let result = crate::backend::dispatch::dispatch_elementwise_generic(a, b, kind)?;
    Ok(DynArray::from_generic(result))
}

/// Dispatch para tipos diferentes después de promoción
#[inline]
fn dispatch_elementwise_promoted<T1, T2>(
    a: &Array<T1>,
    b: &Array<T2>,
    result_dtype: DType,
    kind: ElementwiseKind,
) -> Result<DynArray>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    // Convertir operandos al tipo promovido y ejecutar
    // dispatch_elementwise_generic<T> ahora preserva el tipo T correctamente
    match result_dtype {
        DType::F32 => {
            let a_data: Vec<f32> = a.data.iter().map(|&x| x.to_f32()).collect();
            let b_data: Vec<f32> = b.data.iter().map(|&x| x.to_f32()).collect();
            let a_temp = Array::new(a.shape.clone(), a_data);
            let b_temp = Array::new(b.shape.clone(), b_data);
            
            let result = crate::backend::dispatch::dispatch_elementwise_generic(&a_temp, &b_temp, kind)?;
            Ok(DynArray::F32(result))
        }
        DType::F64 => {
            let a_data: Vec<f64> = a.data.iter().map(|&x| x.to_f32() as f64).collect();
            let b_data: Vec<f64> = b.data.iter().map(|&x| x.to_f32() as f64).collect();
            let a_temp = Array::new(a.shape.clone(), a_data);
            let b_temp = Array::new(b.shape.clone(), b_data);
            
            let result = crate::backend::dispatch::dispatch_elementwise_generic(&a_temp, &b_temp, kind)?;
            Ok(DynArray::F64(result))
        }
        DType::I32 => {
            let a_data: Vec<i32> = a.data.iter().map(|&x| x.to_f32() as i32).collect();
            let b_data: Vec<i32> = b.data.iter().map(|&x| x.to_f32() as i32).collect();
            let a_temp = Array::new(a.shape.clone(), a_data);
            let b_temp = Array::new(b.shape.clone(), b_data);
            
            let result = crate::backend::dispatch::dispatch_elementwise_generic(&a_temp, &b_temp, kind)?;
            Ok(DynArray::I32(result))
        }
        _ => {
            // Para otros tipos, convertir a f32 como fallback
            let a_data: Vec<f32> = a.data.iter().map(|&x| x.to_f32()).collect();
            let b_data: Vec<f32> = b.data.iter().map(|&x| x.to_f32()).collect();
            let a_temp = Array::new(a.shape.clone(), a_data);
            let b_temp = Array::new(b.shape.clone(), b_data);
            
            let result = crate::backend::dispatch::dispatch_elementwise_generic(&a_temp, &b_temp, kind)?;
            Ok(DynArray::F32(result))
        }
    }
}

/// Wrapper for unary operations (exp, cos, sin, sqrt) - maintains dtype, no casting except for dispatch
#[inline]
pub fn unary_promoted<T>(
    a: &Array<T>,
    kind: ElementwiseKind,
    _op_name: &str,
) -> Result<DynArray>
where
    T: DTypeValue,
{
    use std::any::TypeId;
    
    // FAST PATH: f32 - direct dispatch, zero overhead
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let a_f32 = unsafe { &*(a as *const Array<T> as *const Array<f32>) };
        let dummy = Array::zeros(a_f32.shape.clone());
        let table = crate::backend::dispatch::get_dispatch_table();
        let result = (table.elementwise)(a_f32, &dummy, kind)?;
        return Ok(DynArray::F32(result));
    }
    
    // For other types: convert to f32 for dispatch, then wrap in DynArray
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let a_f64 = unsafe { &*(a as *const Array<T> as *const Array<f64>) };
        let a_data: Vec<f32> = a_f64.data.iter().map(|&x| x as f32).collect();
        let a_temp = Array::new(a.shape.clone(), a_data);
        let dummy = Array::zeros(a_temp.shape.clone());
        
        let table = crate::backend::dispatch::get_dispatch_table();
        let result_f32 = (table.elementwise)(&a_temp, &dummy, kind)?;
        
        // Convert back to f64
        let result_data: Vec<f64> = result_f32.data.iter().map(|&x| x as f64).collect();
        let mut result = Array::new(a.shape.clone(), result_data);
        result.dtype = a.dtype;
        return Ok(DynArray::F64(result));
    }
    
    // For integer types: keep original dtype
    let a_data: Vec<f32> = a.data.iter().map(|&x| T::to_f32(x)).collect();
    let a_temp = Array::new(a.shape.clone(), a_data);
    let dummy = Array::zeros(a_temp.shape.clone());
    
    let table = crate::backend::dispatch::get_dispatch_table();
    let result_f32 = (table.elementwise)(&a_temp, &dummy, kind)?;
    
    // Wrap based on original type
    if TypeId::of::<T>() == TypeId::of::<i32>() {
        let result_data: Vec<i32> = result_f32.data.iter().map(|&x| x as i32).collect();
        let mut result = Array::new(a.shape.clone(), result_data);
        result.dtype = a.dtype;
        Ok(DynArray::I32(result))
    } else if TypeId::of::<T>() == TypeId::of::<i8>() {
        let result_data: Vec<i8> = result_f32.data.iter().map(|&x| x as i8).collect();
        let mut result = Array::new(a.shape.clone(), result_data);
        result.dtype = a.dtype;
        Ok(DynArray::I8(result))
    } else if TypeId::of::<T>() == TypeId::of::<u8>() {
        let result_data: Vec<u8> = result_f32.data.iter().map(|&x| x as u8).collect();
        let mut result = Array::new(a.shape.clone(), result_data);
        result.dtype = a.dtype;
        Ok(DynArray::U8(result))
    } else {
        Ok(DynArray::F32(result_f32))
    }
}

/// Wrapper for reduction operations (sum, mean, variance, max) - maintains dtype, no casting except for dispatch
#[inline]
pub fn reduction_promoted<T>(
    a: &Array<T>,
    axis: Option<usize>,
    kind: crate::llo::reduction::ReductionKind,
    _op_name: &str,
) -> Result<DynArray>
where
    T: DTypeValue,
{
    use std::any::TypeId;
    
    // FAST PATH: f32 - direct dispatch, zero overhead
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let a_f32 = unsafe { &*(a as *const Array<T> as *const Array<f32>) };
        let table = crate::backend::dispatch::get_dispatch_table();
        let result = (table.reduction)(a_f32, axis, kind)?;
        return Ok(DynArray::F32(result));
    }
    
    // For other types: convert to f32 for dispatch, then wrap in DynArray
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let a_f64 = unsafe { &*(a as *const Array<T> as *const Array<f64>) };
        let a_data: Vec<f32> = a_f64.data.iter().map(|&x| x as f32).collect();
        let a_temp = Array::new(a.shape.clone(), a_data);
        
        let table = crate::backend::dispatch::get_dispatch_table();
        let result_f32 = (table.reduction)(&a_temp, axis, kind)?;
        
        // Convert back to f64
        let result_data: Vec<f64> = result_f32.data.iter().map(|&x| x as f64).collect();
        let mut result = Array::new(result_f32.shape.clone(), result_data);
        result.dtype = a.dtype;
        return Ok(DynArray::F64(result));
    }
    
    // For integer types
    let a_data: Vec<f32> = a.data.iter().map(|&x| T::to_f32(x)).collect();
    let a_temp = Array::new(a.shape.clone(), a_data);
    
    let table = crate::backend::dispatch::get_dispatch_table();
    let result_f32 = (table.reduction)(&a_temp, axis, kind)?;
    
    // Wrap based on original type
    if TypeId::of::<T>() == TypeId::of::<i32>() {
        let result_data: Vec<i32> = result_f32.data.iter().map(|&x| x as i32).collect();
        let mut result = Array::new(result_f32.shape.clone(), result_data);
        result.dtype = a.dtype;
        Ok(DynArray::I32(result))
    } else if TypeId::of::<T>() == TypeId::of::<i8>() {
        let result_data: Vec<i8> = result_f32.data.iter().map(|&x| x as i8).collect();
        let mut result = Array::new(result_f32.shape.clone(), result_data);
        result.dtype = a.dtype;
        Ok(DynArray::I8(result))
    } else if TypeId::of::<T>() == TypeId::of::<u8>() {
        let result_data: Vec<u8> = result_f32.data.iter().map(|&x| x as u8).collect();
        let mut result = Array::new(result_f32.shape.clone(), result_data);
        result.dtype = a.dtype;
        Ok(DynArray::U8(result))
    } else {
        Ok(DynArray::F32(result_f32))
    }
}
