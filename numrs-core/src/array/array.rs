use crate::array::{DType, DTypeValue};
use crate::ir::HloNode;
use anyhow::Result;

/// A lightweight n-dimensional array with generic dtype support.
///
/// Arrays are memory contiguous (row-major) and can hold different data types.
/// The type parameter `T` must implement `DTypeValue` trait.
///
/// Arrays now support zero-copy views with strides and offset:
/// - `strides`: Step size in each dimension (None = C-contiguous)
/// - `offset`: Starting offset in data buffer (0 = start at beginning)
#[derive(Clone, Debug)]
pub struct Array<T: DTypeValue = f32> {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<T>,
    /// Strides for each dimension (None = C-contiguous layout)
    pub strides: Option<Vec<isize>>,
    /// Offset into data buffer for view (0 = no offset)
    pub offset: usize,
}

impl<T: DTypeValue> Array<T> {
    /// Create a new array given a shape and a flat data vector.
    /// The dtype is automatically inferred from type T.
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape length mismatch"
        );
        Self {
            shape,
            dtype: T::dtype(),
            data,
            strides: None, // C-contiguous by default
            offset: 0,     // No offset
        }
    }

    /// Create a new array with implicit type conversion from any DTypeValue.
    ///
    /// This allows you to create arrays with automatic casting from input data
    /// to the target dtype using the promotion engine.
    ///
    /// # Examples
    /// ```
    /// use numrs::Array;
    ///
    /// // Create f32 array from i32 data - automatic conversion
    /// let arr = Array::<f32>::from_vec(vec![3], vec![1i32, 2, 3]);
    /// assert_eq!(arr.data, vec![1.0f32, 2.0, 3.0]);
    ///
    /// // Create i32 array from f32 data - automatic conversion
    /// let arr = Array::<i32>::from_vec(vec![2], vec![1.5f32, 2.7]);
    /// assert_eq!(arr.data, vec![1, 2]);
    /// ```
    pub fn from_vec<U: DTypeValue>(shape: Vec<usize>, data: Vec<U>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape length mismatch"
        );

        // Si los tipos son iguales, no hay conversión (zero-cost)
        if U::dtype() == T::dtype() {
            // Verificamos explícitamente tamaño y alineación para evitar UB.
            assert_eq!(std::mem::size_of::<U>(), std::mem::size_of::<T>());
            assert_eq!(std::mem::align_of::<U>(), std::mem::align_of::<T>());

            let mut data = std::mem::ManuallyDrop::new(data);
            let (ptr, len, cap) = (data.as_mut_ptr(), data.len(), data.capacity());

            let converted_data = unsafe { Vec::from_raw_parts(ptr as *mut T, len, cap) };
            return Self {
                shape,
                dtype: T::dtype(),
                data: converted_data,
                strides: None,
                offset: 0,
            };
        }

        // Si los tipos son diferentes, usar el motor de promoción
        let converted_data: Vec<T> = data.iter().map(|&val| T::from_f32(val.to_f32())).collect();

        Self {
            shape,
            dtype: T::dtype(),
            data: converted_data,
            strides: None,
            offset: 0,
        }
    }

    /// Helper to convert any array to f32 (useful for tests and interoperability)
    pub fn to_f32(&self) -> Array<f32> {
        let new_data: Vec<f32> = self.data.iter().map(|v| v.to_f32()).collect();
        Array {
            shape: self.shape.clone(),
            dtype: DType::F32,
            data: new_data,
            strides: self.strides.clone(),
            offset: 0,
        }
    }

    /// Create an array initialized with zeros for a shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            shape,
            dtype: T::dtype(),
            data: vec![T::default(); len],
            strides: None,
            offset: 0,
        }
    }

    /// Create an array initialized with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        let one = T::from_f32(1.0);
        Self {
            shape,
            dtype: T::dtype(),
            data: vec![one; len],
            strides: None,
            offset: 0,
        }
    }

    /// Return shape as a slice
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the dtype of this array
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Helper to get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if array is C-contiguous (no custom strides)
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_none() && self.offset == 0
    }

    /// Compute C-contiguous strides for current shape
    pub fn compute_default_strides(&self) -> Vec<isize> {
        let mut strides = vec![1isize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * (self.shape[i + 1] as isize);
        }
        strides
    }

    /// Get effective strides (either custom or computed default)
    pub fn get_strides(&self) -> Vec<isize> {
        self.strides
            .clone()
            .unwrap_or_else(|| self.compute_default_strides())
    }

    /// Create a broadcast view without copying data
    ///
    /// This creates a zero-copy view that logically broadcasts the array
    /// to a new shape by adjusting strides (setting stride to 0 for broadcast dims).
    ///
    /// # Arguments
    /// * `target_shape` - The shape to broadcast to
    ///
    /// # Returns
    /// A new Array that shares the same data but appears to have the target shape
    pub fn broadcast_view(&self, target_shape: &[usize]) -> Result<Self> {
        // Validar que el broadcast sea posible
        crate::ops::shape::broadcast_to::validate_broadcast_public(&self.shape, target_shape)?;

        let src_ndim = self.shape.len();
        let target_ndim = target_shape.len();

        // Calcular nuevos strides
        let src_strides = self.get_strides();
        let mut new_strides = vec![0isize; target_ndim];

        // Mapear desde el final (right-aligned)
        for i in 0..src_ndim {
            let src_idx = src_ndim - 1 - i;
            let target_idx = target_ndim - 1 - i;

            if self.shape[src_idx] == 1 {
                // Dimensión broadcast: stride = 0 (reutilizar mismo valor)
                new_strides[target_idx] = 0;
            } else {
                // Dimensión normal: usar stride original
                new_strides[target_idx] = src_strides[src_idx];
            }
        }

        // Dimensiones nuevas (padding izquierdo) tienen stride 0
        for i in 0..(target_ndim - src_ndim) {
            new_strides[i] = 0;
        }

        Ok(Self {
            shape: target_shape.to_vec(),
            dtype: self.dtype,
            data: self.data.clone(), // Compartir referencia (Rc en futuro?)
            strides: Some(new_strides),
            offset: self.offset,
        })
    }

    /// Materialize a view into a contiguous array
    ///
    /// If the array is already contiguous, returns a clone.
    /// Otherwise, creates a new contiguous array with the data arranged properly.
    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }

        let size: usize = self.shape.iter().product();
        let mut result = Vec::with_capacity(size);
        let strides = self.get_strides();

        // Iterar sobre todos los elementos en orden C-contiguous
        let mut indices = vec![0usize; self.shape.len()];

        for _ in 0..size {
            // Calcular offset con strides
            let mut flat_idx = self.offset as isize;
            for (i, &idx) in indices.iter().enumerate() {
                flat_idx += idx as isize * strides[i];
            }

            // Bounds check: asegurar que el índice es válido
            let flat_idx_usize = flat_idx as usize;
            if flat_idx_usize >= self.data.len() {
                // Fallback: usar módulo para wrap around (broadcasting)
                let safe_idx = flat_idx_usize % self.data.len().max(1);
                result.push(self.data[safe_idx]);
            } else {
                result.push(self.data[flat_idx_usize]);
            }

            // Incrementar índices (orden C)
            for i in (0..self.shape.len()).rev() {
                indices[i] += 1;
                if indices[i] < self.shape[i] {
                    break;
                }
                indices[i] = 0;
            }
        }

        Self {
            shape: self.shape.clone(),
            dtype: self.dtype,
            data: result,
            strides: None,
            offset: 0,
        }
    }

    /// Build an HLO graph node representing a constant array
    pub fn to_hlo_const(&self) -> HloNode {
        HloNode::const_node(self.shape.clone())
    }
}
