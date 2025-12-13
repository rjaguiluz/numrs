//! Dynamic (type-erased) array that can hold any dtype
//!
//! Este módulo proporciona `DynArray`, un enum que puede contener un Array<T>
//! de cualquier tipo concreto. Esto permite que funciones devuelvan arrays
//! con dtype determinado en runtime (como NumPy).

use crate::array::{Array, DType, DTypeValue};
use anyhow::{Result, bail};

/// Array con tipo borrado (type-erased) que puede contener cualquier dtype
///
/// Similar a como NumPy maneja arrays internamente: el dtype se conoce en runtime.
/// 
/// # Ejemplo
/// ```ignore
/// let a = DynArray::F32(Array::new(vec![2], vec![1.0, 2.0]));
/// let b = DynArray::I32(Array::new(vec![2], vec![3, 4]));
/// 
/// // Operaciones devuelven DynArray con tipo promovido
/// let result = ops::add_dyn(&a, &b)?; // -> DynArray::F32
/// 
/// // Pattern matching para extraer el tipo concreto
/// match result {
///     DynArray::F32(arr) => println!("f32: {:?}", arr.data),
///     DynArray::I32(arr) => println!("i32: {:?}", arr.data),
///     _ => {}
/// }
/// ```
#[derive(Debug, Clone)]
pub enum DynArray {
    F32(Array<f32>),
    F64(Array<f64>),
    I32(Array<i32>),
    I8(Array<i8>),
    U8(Array<u8>),
    Bool(Array<bool>),
}

impl DynArray {
    /// Obtener el DType de este array
    pub fn dtype(&self) -> DType {
        match self {
            DynArray::F32(_) => DType::F32,
            DynArray::F64(_) => DType::F64,
            DynArray::I32(_) => DType::I32,
            DynArray::I8(_) => DType::I8,
            DynArray::U8(_) => DType::U8,
            DynArray::Bool(_) => DType::Bool,
        }
    }
    
    /// Obtener el shape de este array
    pub fn shape(&self) -> &[usize] {
        match self {
            DynArray::F32(a) => &a.shape,
            DynArray::F64(a) => &a.shape,
            DynArray::I32(a) => &a.shape,
            DynArray::I8(a) => &a.shape,
            DynArray::U8(a) => &a.shape,
            DynArray::Bool(a) => &a.shape,
        }
    }
    
    /// Número de elementos
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }
    
    /// Si el array está vacío
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Convertir a Array<f32> (con conversión si es necesario)
    pub fn to_f32(&self) -> Array<f32> {
        match self {
            DynArray::F32(a) => a.clone(),
            DynArray::F64(a) => {
                let data: Vec<f32> = a.data.iter().map(|&x| x as f32).collect();
                Array::new(a.shape.clone(), data)
            }
            DynArray::I32(a) => {
                let data: Vec<f32> = a.data.iter().map(|&x| x as f32).collect();
                Array::new(a.shape.clone(), data)
            }
            DynArray::I8(a) => {
                let data: Vec<f32> = a.data.iter().map(|&x| x as f32).collect();
                Array::new(a.shape.clone(), data)
            }
            DynArray::U8(a) => {
                let data: Vec<f32> = a.data.iter().map(|&x| x as f32).collect();
                Array::new(a.shape.clone(), data)
            }
            DynArray::Bool(a) => {
                let data: Vec<f32> = a.data.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
                Array::new(a.shape.clone(), data)
            }
        }
    }
    
    /// Aplicar una función a los datos internos
    pub fn map_data<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&dyn std::any::Any) -> Result<R>,
    {
        match self {
            DynArray::F32(a) => f(&a.data),
            DynArray::F64(a) => f(&a.data),
            DynArray::I32(a) => f(&a.data),
            DynArray::I8(a) => f(&a.data),
            DynArray::U8(a) => f(&a.data),
            DynArray::Bool(a) => f(&a.data),
        }
    }
    
    /// Extract the inner Array<T> with zero-copy if types match, or cast if they don't.
    /// 
    /// # Automatic Casting
    /// If the internal dtype differs from T, this function will automatically cast/convert
    /// the data to T. This enables "automatic narrowing" (F64 -> F32) or normal promotion casting.
    pub fn into_typed<T: DTypeValue>(self) -> Result<Array<T>> {
        use std::any::TypeId;
        use std::mem;
        
        // Fast path: types match (zero copy)
        // We use TypeId check for safety before transmute
        match self {
            DynArray::F32(arr) if TypeId::of::<T>() == TypeId::of::<f32>() => {
                return Ok(unsafe { mem::transmute::<Array<f32>, Array<T>>(arr) });
            }
            DynArray::F64(arr) if TypeId::of::<T>() == TypeId::of::<f64>() => {
                return Ok(unsafe { mem::transmute::<Array<f64>, Array<T>>(arr) });
            }
            DynArray::I32(arr) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                return Ok(unsafe { mem::transmute::<Array<i32>, Array<T>>(arr) });
            }
            DynArray::I8(arr) if TypeId::of::<T>() == TypeId::of::<i8>() => {
                return Ok(unsafe { mem::transmute::<Array<i8>, Array<T>>(arr) });
            }
            DynArray::U8(arr) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                return Ok(unsafe { mem::transmute::<Array<u8>, Array<T>>(arr) });
            }
            DynArray::Bool(arr) if TypeId::of::<T>() == TypeId::of::<bool>() => {
                return Ok(unsafe { mem::transmute::<Array<bool>, Array<T>>(arr) });
            }
            _ => {
                // Slow path: Type mismatch -> Cast required
                match self {
                    DynArray::F32(arr) => Ok(crate::array::promotion::cast_array(&arr)),
                    DynArray::F64(arr) => Ok(crate::array::promotion::cast_array(&arr)),
                    DynArray::I32(arr) => Ok(crate::array::promotion::cast_array(&arr)),
                    DynArray::I8(arr) => Ok(crate::array::promotion::cast_array(&arr)),
                    DynArray::U8(arr) => Ok(crate::array::promotion::cast_array(&arr)),
                    DynArray::Bool(arr) => Ok(crate::array::promotion::cast_array(&arr)),
                }
            }
        }
    }
    
    /// Envolver un Array<T> genérico en DynArray basándose en su dtype
    /// 
    /// Esto usa unsafe para transmute, pero es seguro porque verificamos el dtype
    pub fn from_generic<T: DTypeValue>(arr: Array<T>) -> Self {
        use std::any::TypeId;
        use std::mem;
        
        // Verificar el tipo en compiletime si es posible
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let arr_f32 = unsafe { mem::transmute::<Array<T>, Array<f32>>(arr) };
            return DynArray::F32(arr_f32);
        }
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let arr_f64 = unsafe { mem::transmute::<Array<T>, Array<f64>>(arr) };
            return DynArray::F64(arr_f64);
        }
        if TypeId::of::<T>() == TypeId::of::<i32>() {
            let arr_i32 = unsafe { mem::transmute::<Array<T>, Array<i32>>(arr) };
            return DynArray::I32(arr_i32);
        }
        if TypeId::of::<T>() == TypeId::of::<i8>() {
            let arr_i8 = unsafe { mem::transmute::<Array<T>, Array<i8>>(arr) };
            return DynArray::I8(arr_i8);
        }
        if TypeId::of::<T>() == TypeId::of::<u8>() {
            let arr_u8 = unsafe { mem::transmute::<Array<T>, Array<u8>>(arr) };
            return DynArray::U8(arr_u8);
        }
        if TypeId::of::<T>() == TypeId::of::<bool>() {
            let arr_bool = unsafe { mem::transmute::<Array<T>, Array<bool>>(arr) };
            return DynArray::Bool(arr_bool);
        }
        
        // Fallback: no debería llegar aquí si DTypeValue está bien implementado
        panic!("Unsupported dtype for DynArray::from_generic");
    }
}

// Conversiones convenientes desde Array<T> a DynArray
impl From<Array<f32>> for DynArray {
    fn from(arr: Array<f32>) -> Self {
        DynArray::F32(arr)
    }
}

impl From<Array<f64>> for DynArray {
    fn from(arr: Array<f64>) -> Self {
        DynArray::F64(arr)
    }
}

impl From<Array<i32>> for DynArray {
    fn from(arr: Array<i32>) -> Self {
        DynArray::I32(arr)
    }
}

impl From<Array<i8>> for DynArray {
    fn from(arr: Array<i8>) -> Self {
        DynArray::I8(arr)
    }
}

impl From<Array<u8>> for DynArray {
    fn from(arr: Array<u8>) -> Self {
        DynArray::U8(arr)
    }
}

impl From<Array<bool>> for DynArray {
    fn from(arr: Array<bool>) -> Self {
        DynArray::Bool(arr)
    }
}

// Conversiones hacia Array<T> con TryFrom (puede fallar si el tipo no coincide)
impl TryFrom<DynArray> for Array<f32> {
    type Error = anyhow::Error;
    
    fn try_from(dyn_arr: DynArray) -> Result<Self> {
        match dyn_arr {
            DynArray::F32(a) => Ok(a),
            _ => bail!("Expected F32, got {:?}", dyn_arr.dtype()),
        }
    }
}

impl TryFrom<DynArray> for Array<f64> {
    type Error = anyhow::Error;
    
    fn try_from(dyn_arr: DynArray) -> Result<Self> {
        match dyn_arr {
            DynArray::F64(a) => Ok(a),
            _ => bail!("Expected F64, got {:?}", dyn_arr.dtype()),
        }
    }
}

impl TryFrom<DynArray> for Array<i32> {
    type Error = anyhow::Error;
    
    fn try_from(dyn_arr: DynArray) -> Result<Self> {
        match dyn_arr {
            DynArray::I32(a) => Ok(a),
            _ => bail!("Expected I32, got {:?}", dyn_arr.dtype()),
        }
    }
}

impl TryFrom<DynArray> for Array<i8> {
    type Error = anyhow::Error;
    
    fn try_from(dyn_arr: DynArray) -> Result<Self> {
        match dyn_arr {
            DynArray::I8(a) => Ok(a),
            _ => bail!("Expected I8, got {:?}", dyn_arr.dtype()),
        }
    }
}

impl TryFrom<DynArray> for Array<u8> {
    type Error = anyhow::Error;
    
    fn try_from(dyn_arr: DynArray) -> Result<Self> {
        match dyn_arr {
            DynArray::U8(a) => Ok(a),
            _ => bail!("Expected U8, got {:?}", dyn_arr.dtype()),
        }
    }
}

impl TryFrom<DynArray> for Array<bool> {
    type Error = anyhow::Error;
    
    fn try_from(dyn_arr: DynArray) -> Result<Self> {
        match dyn_arr {
            DynArray::Bool(a) => Ok(a),
            _ => bail!("Expected Bool, got {:?}", dyn_arr.dtype()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyn_array_creation() {
        let a = DynArray::F32(Array::new(vec![2], vec![1.0, 2.0]));
        assert_eq!(a.dtype(), DType::F32);
        assert_eq!(a.shape(), &[2]);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_dyn_array_conversions() {
        let arr_f32 = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
        let dyn_arr: DynArray = arr_f32.clone().into();
        
        assert_eq!(dyn_arr.dtype(), DType::F32);
        
        let back: Array<f32> = dyn_arr.try_into().unwrap();
        assert_eq!(back.data, arr_f32.data);
    }

    #[test]
    fn test_dyn_array_type_mismatch() {
        let dyn_arr = DynArray::I32(Array::new(vec![2], vec![1, 2]));
        
        let result: Result<Array<f32>> = dyn_arr.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn test_to_f32_conversion() {
        let i32_arr = DynArray::I32(Array::new(vec![3], vec![1, 2, 3]));
        let f32_arr = i32_arr.to_f32();
        
        assert_eq!(f32_arr.data, vec![1.0, 2.0, 3.0]);
    }
}
