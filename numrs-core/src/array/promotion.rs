//! Type promotion rules for operations between different dtypes
//!
//! Cuando dos arrays de tipos diferentes se operan, necesitamos determinar
//! el tipo del resultado siguiendo reglas de promoción (como NumPy).
//!
//! Jerarquía de promoción:
//! bool < u8 < i8 < i32 < f16 < bf16 < f32 < f64

use crate::array::{Array, DType, DTypeValue};
use anyhow::{Result, bail};

/// Determine el tipo de resultado cuando se combinan dos dtypes
/// 
/// Reglas:
/// - Operaciones entre el mismo tipo retornan ese tipo
/// - Operaciones entre tipos diferentes promueven al tipo "más grande"
/// - Float siempre gana sobre int
/// - Tipos más anchos (más bits) ganan sobre tipos más estrechos
pub fn promoted_dtype(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }

    // Lookup table para promoción
    match (a, b) {
        // F64 es el tipo más grande - siempre gana
        (DType::F64, _) | (_, DType::F64) => DType::F64,
        
        // F32 gana sobre todo excepto F64
        (DType::F32, DType::F16) | (DType::F16, DType::F32) => DType::F32,
        (DType::F32, DType::BF16) | (DType::BF16, DType::F32) => DType::F32,
        (DType::F32, _) | (_, DType::F32) => DType::F32,
        
        // BF16 vs F16 -> F32 (ambos son 16-bit, pero incompatibles)
        (DType::BF16, DType::F16) | (DType::F16, DType::BF16) => DType::F32,
        
        // F16 gana sobre enteros
        (DType::F16, _) | (_, DType::F16) => DType::F16,
        
        // BF16 gana sobre enteros
        (DType::BF16, _) | (_, DType::BF16) => DType::BF16,
        
        // I32 gana sobre tipos más pequeños
        (DType::I32, _) | (_, DType::I32) => DType::I32,
        
        // I8 vs U8 -> I32 (para evitar overflow)
        (DType::I8, DType::U8) | (DType::U8, DType::I8) => DType::I32,
        
        // I8 gana sobre Bool
        (DType::I8, DType::Bool) | (DType::Bool, DType::I8) => DType::I8,
        
        // U8 gana sobre Bool
        (DType::U8, DType::Bool) | (DType::Bool, DType::U8) => DType::U8,
        
        // Casos base (no deberían llegar aquí si la tabla está completa)
        _ => DType::F32, // Default seguro: F32
    }
}

/// Convierte un Array<T> a Array<U> (cast)
/// 
/// Esta función hace la conversión real de datos entre tipos.
pub fn cast_array<T, U>(arr: &Array<T>) -> Array<U>
where
    T: DTypeValue,
    U: DTypeValue,
{
    let data: Vec<U> = arr.data.iter().map(|&val| {
        U::from_f32(val.to_f32())
    }).collect();
    
    Array::new(arr.shape.clone(), data)
}

/// Promociona dos arrays al tipo común y retorna las versiones convertidas
/// 
/// Esta es la función principal que se usa en operaciones.
/// Si ambos arrays tienen el mismo tipo, no hace nada (zero-cost).
/// Si son diferentes, convierte ambos al tipo común.
pub fn promote_arrays<T1, T2>(
    a: &Array<T1>,
    b: &Array<T2>,
) -> Result<(DType, Vec<f32>, Vec<f32>)>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    let dtype_a = a.dtype;
    let dtype_b = b.dtype;
    
    if dtype_a == dtype_b {
        // Mismo tipo - solo convertir a f32 para procesamiento
        return Ok((dtype_a, a.data.iter().map(|&x| x.to_f32()).collect(), 
                            b.data.iter().map(|&x| x.to_f32()).collect()));
    }
    
    // Determinar tipo de resultado
    let result_dtype = promoted_dtype(dtype_a, dtype_b);
    
    // Convertir ambos arrays a f32 (forma intermedia común)
    let a_f32: Vec<f32> = a.data.iter().map(|&x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.data.iter().map(|&x| x.to_f32()).collect();
    
    Ok((result_dtype, a_f32, b_f32))
}

/// Valida que dos arrays puedan operarse juntos
/// 
/// Verifica:
/// - Que los shapes sean compatibles
/// - Que los dtypes sean compatibles
pub fn validate_binary_op<T1, T2>(
    a: &Array<T1>,
    b: &Array<T2>,
    op_name: &str,
) -> Result<()>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    // MATMUL es un caso especial - necesita validación diferente
    if op_name == "matmul" {
        return validate_matmul_shapes(a, b);
    }
    
    // Para operaciones elementwise, validar broadcasting
    if !shapes_are_broadcastable(&a.shape, &b.shape) {
        bail!(
            "{}: shape mismatch and not broadcastable: {:?} vs {:?}",
            op_name,
            a.shape,
            b.shape
        );
    }
    
    // Por ahora, todos los dtypes son compatibles (promoción automática)
    // En el futuro podríamos rechazar ciertas combinaciones
    
    Ok(())
}

/// Verifica si dos shapes son compatibles para broadcasting según reglas de NumPy:
/// - Los shapes se comparan desde el final hacia el inicio
/// - Cada dimensión debe ser igual O una de ellas debe ser 1
/// Ejemplos:
/// - [3, 4] y [4] -> broadcastable (se expande [4] a [1, 4])
/// - [3, 4] y [3, 1] -> broadcastable
/// - [3, 4] y [5] -> NO broadcastable
fn shapes_are_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    
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
        
        // Cada dimensión debe ser igual O una debe ser 1
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            return false;
        }
    }
    
    true
}

/// Valida shapes para matmul, soportando múltiples configuraciones como NumPy:
/// - 2D @ 2D: [M, K] @ [K, N] -> [M, N]
/// - 1D @ 2D: [K] @ [K, N] -> [N] (vector-matrix)
/// - 2D @ 1D: [M, K] @ [K] -> [M] (matrix-vector)
/// - 1D @ 1D: [K] @ [K] -> [] (dot product)
/// - nD @ mD: batched matmul con broadcasting
fn validate_matmul_shapes<T1, T2>(a: &Array<T1>, b: &Array<T2>) -> Result<()>
where
    T1: DTypeValue,
    T2: DTypeValue,
{
    let a_ndim = a.shape.len();
    let b_ndim = b.shape.len();
    
    // Casos soportados
    match (a_ndim, b_ndim) {
        // 2D @ 2D: Caso estándar de matriz
        (2, 2) => {
            let a_cols = a.shape[1];
            let b_rows = b.shape[0];
            if a_cols != b_rows {
                bail!(
                    "matmul: inner dimensions must match: [{}] @ [{}] incompatible",
                    a_cols, b_rows
                );
            }
            Ok(())
        }
        
        // 1D @ 2D: vector @ matrix -> vector
        (1, 2) => {
            let a_len = a.shape[0];
            let b_rows = b.shape[0];
            if a_len != b_rows {
                bail!(
                    "matmul: vector-matrix dimensions incompatible: [{}] @ [{}, {}]",
                    a_len, b_rows, b.shape[1]
                );
            }
            Ok(())
        }
        
        // 2D @ 1D: matrix @ vector -> vector
        (2, 1) => {
            let a_cols = a.shape[1];
            let b_len = b.shape[0];
            if a_cols != b_len {
                bail!(
                    "matmul: matrix-vector dimensions incompatible: [{}, {}] @ [{}]",
                    a.shape[0], a_cols, b_len
                );
            }
            Ok(())
        }
        
        // 1D @ 1D: dot product
        (1, 1) => {
            if a.shape[0] != b.shape[0] {
                bail!(
                    "matmul: vectors must have same length: [{}] @ [{}]",
                    a.shape[0], b.shape[0]
                );
            }
            Ok(())
        }
        
        // TODO: Batched matmul (3D+)
        _ => {
            bail!(
                "matmul: unsupported dimensions: {}D @ {}D (currently only 1D/2D supported)",
                a_ndim, b_ndim
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_type_promotion() {
        assert_eq!(promoted_dtype(DType::F32, DType::F32), DType::F32);
        assert_eq!(promoted_dtype(DType::I32, DType::I32), DType::I32);
    }

    #[test]
    fn test_float_hierarchy() {
        // F64 > F32 > F16/BF16
        assert_eq!(promoted_dtype(DType::F32, DType::F64), DType::F64);
        assert_eq!(promoted_dtype(DType::F16, DType::F32), DType::F32);
        assert_eq!(promoted_dtype(DType::BF16, DType::F32), DType::F32);
        assert_eq!(promoted_dtype(DType::F16, DType::F64), DType::F64);
    }

    #[test]
    fn test_float_vs_int() {
        // Float siempre gana
        assert_eq!(promoted_dtype(DType::F32, DType::I32), DType::F32);
        assert_eq!(promoted_dtype(DType::F16, DType::I32), DType::F16);
        assert_eq!(promoted_dtype(DType::I8, DType::F32), DType::F32);
    }

    #[test]
    fn test_int_promotion() {
        // I32 > I8/U8 > Bool
        assert_eq!(promoted_dtype(DType::I32, DType::I8), DType::I32);
        assert_eq!(promoted_dtype(DType::I32, DType::U8), DType::I32);
        assert_eq!(promoted_dtype(DType::I8, DType::Bool), DType::I8);
        assert_eq!(promoted_dtype(DType::U8, DType::Bool), DType::U8);
    }

    #[test]
    fn test_mixed_sign_ints() {
        // I8 + U8 -> I32 (para seguridad)
        assert_eq!(promoted_dtype(DType::I8, DType::U8), DType::I32);
    }

    #[test]
    fn test_f16_vs_bf16() {
        // F16 + BF16 -> F32 (incompatibles entre sí)
        assert_eq!(promoted_dtype(DType::F16, DType::BF16), DType::F32);
    }

    #[test]
    fn test_cast_array() {
        let a = Array::new(vec![3], vec![1.0_f32, 2.0, 3.0]);
        
        // F32 -> I32
        let b: Array<i32> = cast_array(&a);
        assert_eq!(b.dtype, DType::I32);
        assert_eq!(b.data, vec![1, 2, 3]);
        
        // F32 -> F64
        let c: Array<f64> = cast_array(&a);
        assert_eq!(c.dtype, DType::F64);
        assert_eq!(c.data, vec![1.0_f64, 2.0, 3.0]);
    }

    #[test]
    fn test_promote_same_type() -> Result<()> {
        let a = Array::new(vec![2], vec![1.0_f32, 2.0]);
        let b = Array::new(vec![2], vec![3.0_f32, 4.0]);
        
        let (dtype, a_data, b_data) = promote_arrays(&a, &b)?;
        
        assert_eq!(dtype, DType::F32);
        assert_eq!(a_data, vec![1.0, 2.0]);
        assert_eq!(b_data, vec![3.0, 4.0]);
        
        Ok(())
    }

    #[test]
    fn test_promote_different_types() -> Result<()> {
        let a = Array::new(vec![2], vec![1_i32, 2]);
        let b = Array::new(vec![2], vec![3.0_f32, 4.0]);
        
        let (dtype, a_data, b_data) = promote_arrays(&a, &b)?;
        
        assert_eq!(dtype, DType::F32);
        assert_eq!(a_data, vec![1.0, 2.0]);
        assert_eq!(b_data, vec![3.0, 4.0]);
        
        Ok(())
    }

    #[test]
    fn test_validate_binary_op_ok() -> Result<()> {
        let a = Array::new(vec![2, 3], vec![1.0_f32; 6]);
        let b = Array::new(vec![2, 3], vec![2.0_f64; 6]);
        
        validate_binary_op(&a, &b, "add")?;
        Ok(())
    }

    #[test]
    fn test_validate_binary_op_shape_mismatch() {
        let a = Array::new(vec![2], vec![1.0_f32, 2.0]);
        let b = Array::new(vec![3], vec![1.0_f32, 2.0, 3.0]);
        
        let result = validate_binary_op(&a, &b, "add");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("shape mismatch"));
    }
}
