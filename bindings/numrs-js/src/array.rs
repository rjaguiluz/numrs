use napi::bindgen_prelude::*;
use napi_derive::napi;
use numrs::Array;

#[napi]
pub struct NumRsArray {
    pub(crate) inner: Array,
}

#[napi]
impl NumRsArray {
    /// Create new NumRsArray from Float32Array
    #[napi(constructor)]
    pub fn new(data: Float32Array, shape: Option<Vec<u32>>) -> Result<Self> {
        let data_slice = data.as_ref();
        let len = data_slice.len();
        
        let shape_vec: Vec<usize> = match shape {
            Some(s) => s.into_iter().map(|d| d as usize).collect(),
            None => vec![len],
        };
        
        // TODO: Handle other types (F64, I32) properly. Core Array is currently F32-centric in API but supports DynArray.
        // For now, we assume F32 for simplicity as per current Array::new signature in core (vec<f32>).
        // If core Array becomes generic or uses DynArray internally for storage, we adapt.
        let arr = Array::new(shape_vec, data_slice.to_vec());
        
        Ok(NumRsArray { inner: arr })
    }

    #[napi(factory)]
    pub fn from_array(data: Vec<f64>, shape: Option<Vec<u32>>) -> Result<Self> {
        let len = data.len();
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        
        let shape_vec: Vec<usize> = match shape {
            Some(s) => s.into_iter().map(|d| d as usize).collect(),
            None => vec![len],
        };
        
        let arr = Array::new(shape_vec, data_f32);
        Ok(NumRsArray { inner: arr })
    }

    #[napi(factory)]
    pub fn zeros(shape: Vec<u32>) -> Result<Self> {
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let arr = Array::zeros(shape_usize);
        Ok(NumRsArray { inner: arr })
    }
    
    #[napi(factory)]
    pub fn ones(shape: Vec<u32>) -> Result<Self> {
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let arr = Array::ones(shape_usize);
        Ok(NumRsArray { inner: arr })
    }

    #[napi]
    pub fn reshape(&self, shape: Vec<u32>) -> Result<Self> {
         // core reshape expects &[isize] (allowing -1)
         let shape_isize: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
         let res = numrs::ops::shape::reshape::reshape(&self.inner, &shape_isize)
             .map_err(|e| Error::from_reason(e.to_string()))?;
         Ok(NumRsArray { inner: res })
    }

    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.shape.iter().map(|&d| d as u32).collect()
    }

    #[napi(getter)]
    pub fn data(&self) -> Float32Array {
        Float32Array::new(self.inner.data.clone())
    }

    #[napi]
    pub fn to_string(&self) -> String {
        format!("{:?}", self.inner)
    }

    // --- Operations ---
    
    #[napi]
    pub fn add(&self, other: &NumRsArray) -> Result<NumRsArray> {
        let res = numrs::ops::add(&self.inner, &other.inner)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(NumRsArray { inner: res })
    }

    #[napi]
    pub fn sub(&self, other: &NumRsArray) -> Result<NumRsArray> {
        let res = numrs::ops::sub(&self.inner, &other.inner)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(NumRsArray { inner: res })
    }

    #[napi]
    pub fn mul(&self, other: &NumRsArray) -> Result<NumRsArray> {
        let res = numrs::ops::mul(&self.inner, &other.inner)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(NumRsArray { inner: res })
    }

    #[napi]
    pub fn div(&self, other: &NumRsArray) -> Result<NumRsArray> {
        let res = numrs::ops::div(&self.inner, &other.inner)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(NumRsArray { inner: res })
    }
}
