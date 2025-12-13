//! Backend capabilities - Define qué dtypes soporta cada backend
//!
//! Cada backend declara explícitamente qué tipos de datos puede manejar.
//! Esto permite al sistema de dispatch seleccionar el backend apropiado
//! basado tanto en disponibilidad como en el dtype del array.

use crate::array::DType;

/// Capabilities de un backend - qué tipos soporta
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Nombre del backend
    pub name: &'static str,
    /// Tipos de datos soportados
    pub supported_dtypes: &'static [DType],
}

impl BackendCapabilities {
    /// Verifica si este backend soporta el dtype dado
    pub fn supports(&self, dtype: DType) -> bool {
        self.supported_dtypes.contains(&dtype)
    }

    /// Lista de dtypes soportados como strings (para logging)
    pub fn supported_types_str(&self) -> String {
        self.supported_dtypes
            .iter()
            .map(|dt| dt.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// ============================================================================
// Backend Capabilities Constants
// ============================================================================

/// BLAS capabilities: solo F32 y F64
/// BLAS tiene funciones específicas por tipo:
/// - sgemm (single precision, f32)
/// - dgemm (double precision, f64)
/// F16/BF16 no están soportados en BLAS estándar
pub const BLAS_CAPABILITIES: BackendCapabilities = BackendCapabilities {
    name: "blas",
    supported_dtypes: &[DType::F32, DType::F64],
};

/// CPU SIMD capabilities: F16, F32, F64, I32
/// AVX2/SSE soportan operaciones vectorizadas para estos tipos
/// F16 via conversión F16->F32->operación->F16
/// BF16 tiene soporte limitado (solo en AVX512_BF16)
pub const SIMD_CAPABILITIES: BackendCapabilities = BackendCapabilities {
    name: "cpu-simd",
    supported_dtypes: &[DType::F16, DType::F32, DType::F64, DType::I32],
};

/// CPU Scalar capabilities: todos los tipos
/// El backend escalar puede manejar cualquier tipo mediante loops simples
pub const SCALAR_CAPABILITIES: BackendCapabilities = BackendCapabilities {
    name: "cpu-scalar",
    supported_dtypes: &[
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::U8,
        DType::I8,
        DType::I32,
        DType::Bool,
    ],
};

/// WebGPU capabilities: F16, F32
/// WGSL (WebGPU Shading Language) soporta f16 (con extensión) y f32 nativamente
pub const WEBGPU_CAPABILITIES: BackendCapabilities = BackendCapabilities {
    name: "webgpu",
    supported_dtypes: &[DType::F16, DType::F32],
};

/// Metal capabilities: F16, F32, F64, I32, U8
/// Metal Shading Language tiene buen soporte para estos tipos
pub const METAL_CAPABILITIES: BackendCapabilities = BackendCapabilities {
    name: "metal",
    supported_dtypes: &[DType::F16, DType::F32, DType::F64, DType::I32, DType::U8],
};

/// CUDA capabilities: F16, BF16, F32, F64, I32, U8, I8
/// CUDA tiene excelente soporte para tipos ML (incluyendo BF16)
pub const CUDA_CAPABILITIES: BackendCapabilities = BackendCapabilities {
    name: "cuda",
    supported_dtypes: &[
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
        DType::U8,
        DType::I8,
        DType::I32,
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blas_capabilities() {
        assert!(BLAS_CAPABILITIES.supports(DType::F32));
        assert!(BLAS_CAPABILITIES.supports(DType::F64));
        assert!(!BLAS_CAPABILITIES.supports(DType::I32));
        assert!(!BLAS_CAPABILITIES.supports(DType::Bool));
    }

    #[test]
    fn test_simd_capabilities() {
        assert!(SIMD_CAPABILITIES.supports(DType::F16));
        assert!(SIMD_CAPABILITIES.supports(DType::F32));
        assert!(SIMD_CAPABILITIES.supports(DType::I32));
        assert!(!SIMD_CAPABILITIES.supports(DType::BF16));  // BF16 necesita AVX512_BF16
        assert!(!SIMD_CAPABILITIES.supports(DType::U8));
        assert!(!SIMD_CAPABILITIES.supports(DType::Bool));
    }

    #[test]
    fn test_scalar_capabilities() {
        // Scalar debería soportar TODOS los tipos
        assert!(SCALAR_CAPABILITIES.supports(DType::F16));
        assert!(SCALAR_CAPABILITIES.supports(DType::BF16));
        assert!(SCALAR_CAPABILITIES.supports(DType::F32));
        assert!(SCALAR_CAPABILITIES.supports(DType::F64));
        assert!(SCALAR_CAPABILITIES.supports(DType::U8));
        assert!(SCALAR_CAPABILITIES.supports(DType::I8));
        assert!(SCALAR_CAPABILITIES.supports(DType::I32));
        assert!(SCALAR_CAPABILITIES.supports(DType::Bool));
    }

    #[test]
    fn test_webgpu_capabilities() {
        assert!(WEBGPU_CAPABILITIES.supports(DType::F16));
        assert!(WEBGPU_CAPABILITIES.supports(DType::F32));
        assert!(!WEBGPU_CAPABILITIES.supports(DType::F64));
        assert!(!WEBGPU_CAPABILITIES.supports(DType::I32));
    }

    #[test]
    fn test_supported_types_str() {
        let s = BLAS_CAPABILITIES.supported_types_str();
        assert!(s.contains("f32"));
        assert!(s.contains("f64"));
    }
}
