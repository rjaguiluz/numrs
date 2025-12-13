//! Data types (dtypes) supported by NumRs arrays
//!
//! Este módulo define los tipos de datos que pueden almacenarse en arrays
//! y proporciona utilities para trabajar con ellos.

use serde::{Deserialize, Serialize};

/// Tipos de datos soportados por NumRs arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    // Punto flotante
    /// 16-bit floating point (IEEE 754 half precision)
    F16,
    /// Brain float 16-bit (bfloat16, used in ML)
    BF16,
    /// 32-bit floating point (IEEE 754 single precision)
    F32,
    /// 64-bit floating point (IEEE 754 double precision)
    F64,
    
    // Enteros
    /// 8-bit unsigned integer (0 to 255, common for images)
    U8,
    /// 8-bit signed integer (-128 to 127)
    I8,
    /// 32-bit signed integer
    I32,
    
    // Booleano
    /// Boolean type (true/false)
    Bool,
}

impl DType {
    /// Size in bytes of this dtype
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            DType::U8 => 1,
            DType::I8 => 1,
            DType::I32 => 4,
            DType::Bool => 1,
        }
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::U8 => "u8",
            DType::I8 => "i8",
            DType::I32 => "i32",
            DType::Bool => "bool",
        }
    }

    /// Whether this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Whether this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(self, DType::U8 | DType::I8 | DType::I32)
    }
    
    /// Whether this is a signed integer type
    pub fn is_signed_int(&self) -> bool {
        matches!(self, DType::I8 | DType::I32)
    }
    
    /// Whether this is an unsigned integer type
    pub fn is_unsigned_int(&self) -> bool {
        matches!(self, DType::U8)
    }

    /// Whether this is a boolean type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_properties() {
        // Tamaños
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F64.size_bytes(), 8);
        assert_eq!(DType::U8.size_bytes(), 1);
        assert_eq!(DType::I8.size_bytes(), 1);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::Bool.size_bytes(), 1);
        
        // Float types
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
        
        // Integer types
        assert!(DType::I32.is_int());
        assert!(DType::I8.is_int());
        assert!(DType::U8.is_int());
        assert!(!DType::F32.is_int());
        
        // Signed/unsigned
        assert!(DType::I8.is_signed_int());
        assert!(DType::I32.is_signed_int());
        assert!(!DType::U8.is_signed_int());
        
        assert!(DType::U8.is_unsigned_int());
        assert!(!DType::I8.is_unsigned_int());
        
        // Bool
        assert!(DType::Bool.is_bool());
        assert!(!DType::F32.is_bool());
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(DType::F16.to_string(), "f16");
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(DType::F32.to_string(), "f32");
        assert_eq!(DType::F64.to_string(), "f64");
        assert_eq!(DType::U8.to_string(), "u8");
        assert_eq!(DType::I8.to_string(), "i8");
        assert_eq!(DType::I32.to_string(), "i32");
        assert_eq!(DType::Bool.to_string(), "bool");
    }
}

// ============================================================================
// DTypeValue trait - Marca tipos que pueden ser valores de array
// ============================================================================

/// Trait que deben implementar los tipos que pueden almacenarse en arrays.
/// Esto permite que Array<T> sea genérico sobre cualquier tipo soportado.
pub trait DTypeValue: Clone + Copy + Send + Sync + Default + 'static {
    /// El DType correspondiente a este tipo
    fn dtype() -> DType;
    
    /// Convertir a f32 (para compatibilidad con código existente)
    fn to_f32(self) -> f32;
    
    /// Crear desde f32 (para compatibilidad con código existente)
    fn from_f32(val: f32) -> Self;
    
    /// Valor cero del tipo
    fn zero() -> Self;
}

// Implementaciones para cada tipo soportado
impl DTypeValue for f32 {
    fn dtype() -> DType { DType::F32 }
    fn to_f32(self) -> f32 { self }
    fn from_f32(val: f32) -> Self { val }
    fn zero() -> Self { 0.0 }
}

impl DTypeValue for f64 {
    fn dtype() -> DType { DType::F64 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f32(val: f32) -> Self { val as f64 }
    fn zero() -> Self { 0.0 }
}

impl DTypeValue for i32 {
    fn dtype() -> DType { DType::I32 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f32(val: f32) -> Self { val as i32 }
    fn zero() -> Self { 0 }
}

impl DTypeValue for i8 {
    fn dtype() -> DType { DType::I8 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f32(val: f32) -> Self { val as i8 }
    fn zero() -> Self { 0 }
}

impl DTypeValue for u8 {
    fn dtype() -> DType { DType::U8 }
    fn to_f32(self) -> f32 { self as f32 }
    fn from_f32(val: f32) -> Self { val as u8 }
    fn zero() -> Self { 0 }
}

impl DTypeValue for bool {
    fn dtype() -> DType { DType::Bool }
    fn to_f32(self) -> f32 { if self { 1.0 } else { 0.0 } }
    fn from_f32(val: f32) -> Self { val != 0.0 }
    fn zero() -> Self { false }
}

// Para F16 y BF16, usamos u16 como representación
// (se almacena como bits, conversión se hace cuando sea necesario)
