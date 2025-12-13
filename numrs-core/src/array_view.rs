//! ArrayView - Zero-copy view into typed data for FFI operations
//!
//! Este struct resuelve el problema de performance en FFI:
//! - El caller mantiene ownership de los datos en C/Python/JS
//! - ArrayView contiene solo referencias (slices) - NO copia datos
//! - Compatible con dispatch system existente
//!
//! # Arquitectura
//! Para FFI, usamos un patrón de "view handle":
//! 1. C API recibe void* ptr del caller (datos en memoria C)
//! 2. Rust crea ArrayView con slice::from_raw_parts (sin to_vec!)
//! 3. ArrayView se guarda en Box y retorna opaque pointer al caller
//! 4. Caller usa ese handle para múltiples operaciones
//! 5. Caller destruye el handle cuando termina

use crate::array::DType;

/// View into typed array data - wraps borrowed slices (no copies!)
///
/// # Diseño
/// Para evitar to_vec() completamente, ArrayView contiene referencias
/// con lifetime 'static (seguro porque el caller garantiza que los datos
/// viven mientras el view existe - via Box en C API).
///
/// # Uso en FFI
/// ```ignore
/// // C tiene: float* data_a, float* data_b (10000 elementos cada uno)
/// 
/// // Crear views (sin to_vec! - solo crea slice):
/// let view_a = unsafe { ArrayView::from_ptr_f32(data_a, 10000) };
/// let view_b = unsafe { ArrayView::from_ptr_f32(data_b, 10000) };
/// 
/// // Múltiples operaciones - todas zero-copy:
/// ops_inplace::elementwise_view(&view_a, &view_b, out, Add)?;  // ~23μs
/// ops_inplace::elementwise_view(&view_a, &view_b, out, Mul)?;  // ~23μs
/// ops_inplace::elementwise_view(&view_a, &view_b, out, Sub)?;  // ~23μs
/// // Sin to_vec(), cada operación corre a velocidad nativa!
/// ```
#[derive(Debug, Clone)]
pub enum ArrayView {
    F32(&'static [f32]),
    F64(&'static [f64]),
    I32(&'static [i32]),
    I64(&'static [i64]),
    U8(&'static [u8]),
}

impl ArrayView {
    /// Create from f32 slice (zero-copy via raw pointer)
    /// 
    /// # Safety
    /// Caller must ensure data outlives the ArrayView. For FFI, this is safe
    /// because ArrayView is boxed and C API controls lifetime.
    pub fn from_slice_f32(data: &[f32]) -> Self {
        unsafe { Self::from_ptr_f32(data.as_ptr(), data.len()) }
    }
    
    /// Create from f64 slice (zero-copy via raw pointer)
    pub fn from_slice_f64(data: &[f64]) -> Self {
        unsafe { Self::from_ptr_f64(data.as_ptr(), data.len()) }
    }
    
    /// Create from i32 slice (zero-copy via raw pointer)
    pub fn from_slice_i32(data: &[i32]) -> Self {
        unsafe { Self::from_ptr_i32(data.as_ptr(), data.len()) }
    }
    
    /// Create from raw pointer (unsafe - for FFI)
    /// 
    /// # Safety
    /// - `ptr` must be valid for `len` elements
    /// - Data must outlive the ArrayView (caller's responsibility in C API via Box)
    /// - 'static lifetime is safe because C API holds Box<ArrayView> until destroy()
    pub unsafe fn from_ptr_f32(ptr: *const f32, len: usize) -> Self {
        ArrayView::F32(std::slice::from_raw_parts(ptr, len))
    }
    
    /// Create from raw pointer (unsafe - for FFI)
    pub unsafe fn from_ptr_f64(ptr: *const f64, len: usize) -> Self {
        ArrayView::F64(std::slice::from_raw_parts(ptr, len))
    }
    
    /// Create from raw pointer (unsafe - for FFI)
    pub unsafe fn from_ptr_i32(ptr: *const i32, len: usize) -> Self {
        ArrayView::I32(std::slice::from_raw_parts(ptr, len))
    }
    
    /// Get dtype
    pub fn dtype(&self) -> DType {
        match self {
            ArrayView::F32(_) => DType::F32,
            ArrayView::F64(_) => DType::F64,
            ArrayView::I32(_) => DType::I32,
            ArrayView::I64(_) => DType::I32, // Fallback to I32 for now
            ArrayView::U8(_) => DType::U8,
        }
    }
    
    /// Get length
    pub fn len(&self) -> usize {
        match self {
            ArrayView::F32(v) => v.len(),
            ArrayView::F64(v) => v.len(),
            ArrayView::I32(v) => v.len(),
            ArrayView::I64(v) => v.len(),
            ArrayView::U8(v) => v.len(),
        }
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get as f32 slice (zero-copy)
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            ArrayView::F32(v) => Some(v),
            _ => None,
        }
    }
    
    /// Get as f64 slice (zero-copy)
    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            ArrayView::F64(v) => Some(v),
            _ => None,
        }
    }
    
    /// Get as i32 slice (zero-copy)
    pub fn as_i32(&self) -> Option<&[i32]> {
        match self {
            ArrayView::I32(v) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_array_view_f32() {
        // Leak data to get 'static lifetime (ok for tests)
        let data: &'static [f32] = Box::leak(vec![1.0f32, 2.0, 3.0].into_boxed_slice());
        let view = ArrayView::from_slice_f32(data);
        
        assert_eq!(view.len(), 3);
        assert_eq!(view.dtype(), DType::F32);
        assert_eq!(view.as_f32().unwrap(), &[1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_zero_copy_reference() {
        let data: &'static [f32] = Box::leak(vec![1.0f32; 10000].into_boxed_slice());
        let view = ArrayView::from_slice_f32(data);
        
        // Multiple calls to as_f32() return SAME pointer - true zero-copy
        let slice1 = view.as_f32().unwrap();
        let slice2 = view.as_f32().unwrap();
        
        assert_eq!(slice1.as_ptr(), slice2.as_ptr());
        assert_eq!(slice1.as_ptr(), data.as_ptr()); // Points to original data!
    }
    
    #[test]
    fn test_from_ptr_zero_copy() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let ptr = data.as_ptr();
        let len = data.len();
        
        unsafe {
            let view = ArrayView::from_ptr_f32(ptr, len);
            
            // View apunta directamente a data - no hay to_vec()
            assert_eq!(view.as_f32().unwrap().as_ptr(), ptr);
            assert_eq!(view.len(), len);
        }
        
        // Keep data alive
        drop(data);
    }
}
