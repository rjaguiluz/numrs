use napi::bindgen_prelude::*;
use napi_derive::napi;
use numrs::Tensor as CoreTensor;
use crate::NumRsArray;
use std::rc::Rc;
use std::cell::RefCell;

#[napi]
#[derive(Clone)]
pub struct Tensor {
    pub(crate) inner: Rc<RefCell<CoreTensor>>,
}

#[napi]
impl Tensor {
    /// Create a new Tensor.
    #[napi(constructor)]
    pub fn new(data: Either<ClassInstance<NumRsArray>, Float32Array>, shape: Option<Vec<u32>>, requires_grad: Option<bool>) -> Result<Self> {
        let arr = match data {
            Either::A(a) => a.inner.clone(), 
            Either::B(b) => {
                 let data_slice = b.as_ref();
                 let len = data_slice.len();
                 let shape_vec: Vec<usize> = match shape {
                    Some(s) => s.into_iter().map(|d| d as usize).collect(),
                    None => vec![len],
                 };
                 numrs::Array::new(shape_vec, data_slice.to_vec())
            }
        };
        
        let tensor = CoreTensor::new(arr, requires_grad.unwrap_or(false));
        Ok(Tensor { inner: Rc::new(RefCell::new(tensor)) })
    }

    #[napi(factory)]
    pub fn from_array(data: Vec<f64>, shape: Option<Vec<u32>>, requires_grad: Option<bool>) -> Result<Self> {
         let len = data.len();
         let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
         
         let shape_vec: Vec<usize> = match shape {
            Some(s) => s.into_iter().map(|d| d as usize).collect(),
            None => vec![len],
         };
         
         let arr = numrs::Array::new(shape_vec, data_f32);
         let tensor = CoreTensor::new(arr, requires_grad.unwrap_or(false));
         Ok(Tensor { inner: Rc::new(RefCell::new(tensor)) })
    }

    // --- Attributes ---

    #[napi(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.inner.borrow().shape().iter().map(|&d| d as u32).collect()
    }

    #[napi(getter)]
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }
    
    #[napi(getter)]
    pub fn data(&self) -> NumRsArray {
        // Return NumRsArray view/copy
        NumRsArray { inner: self.inner.borrow().data.clone() }
    }

    // --- Autograd ---
    
    #[napi]
    pub fn backward(&self) -> Result<()> {
        self.inner.borrow().backward().map_err(|e| Error::from_reason(e.to_string()))
    }
    
    #[napi(getter)]
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.borrow().gradient().map(|g| Tensor { 
            inner: Rc::new(RefCell::new(CoreTensor::new(g, false)))
        })
    }

    // --- Ops ---

    #[napi]
    pub fn pow(&self, exponent: f64) -> Result<Tensor> {
        let res = self.inner.borrow().pow(exponent as f32).map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn neg(&self) -> Result<Tensor> {
        // -x = x * -1
        let neg_one_arr = numrs::Array::new(vec![1], vec![-1.0]);
        let neg_one = numrs::Tensor::new(neg_one_arr, false);
        
        let res = self.inner.borrow().mul(&neg_one).map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }


    #[napi]
    pub fn sqrt(&self) -> Result<Tensor> {
        let res = self.inner.borrow().sqrt().map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn relu(&self) -> Result<Tensor> {
        let res = self.inner.borrow().relu().map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn sigmoid(&self) -> Result<Tensor> {
        let res = self.inner.borrow().sigmoid().map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
    
    #[napi]
    pub fn exp(&self) -> Result<Tensor> {
        let res = self.inner.borrow().exp().map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
    
    #[napi]
    pub fn log(&self) -> Result<Tensor> {
        let res = self.inner.borrow().log().map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    // Missing: sin, cos, tan not in Autograd tensor ops yet.
    // If I add them here, I need to call Array ops and return detached tensor
    // OR wait for core update.
    // The user wants mapping. If mapped to detached, it's better than nothing.
    // Autograd supported
    #[napi]
    pub fn sin(&self) -> Result<Tensor> {
        let res = self.inner.borrow().sin()
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
    
    #[napi]
    pub fn cos(&self) -> Result<Tensor> {
        let res = self.inner.borrow().cos()
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }
    
    #[napi]
    pub fn tan(&self) -> Result<Tensor> {
        let res = self.inner.borrow().tan()
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn abs(&self) -> Result<Tensor> {
        let data = &self.inner.borrow().data;
        let res_arr = numrs::ops::elementwise::unary::abs(data)
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
    }

    #[napi]
    pub fn tanh(&self) -> Result<Tensor> {
        let res = self.inner.borrow().tanh()
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(res)) })
    }

    #[napi]
    pub fn asin(&self) -> Result<Tensor> {
        let data = &self.inner.borrow().data;
        let res_arr = numrs::ops::elementwise::unary::asin(data)
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
    }

    #[napi]
    pub fn acos(&self) -> Result<Tensor> {
        let data = &self.inner.borrow().data;
        let res_arr = numrs::ops::elementwise::unary::acos(data)
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
    }

    #[napi]
    pub fn atan(&self) -> Result<Tensor> {
        let data = &self.inner.borrow().data;
        let res_arr = numrs::ops::elementwise::unary::atan(data)
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
    }

    #[napi]
    pub fn softplus(&self) -> Result<Tensor> {
        let data = &self.inner.borrow().data;
        let res_arr = numrs::ops::elementwise::unary::softplus(data)
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
    }

    #[napi]
    pub fn leaky_relu(&self) -> Result<Tensor> {
        let data = &self.inner.borrow().data;
        let res_arr = numrs::ops::elementwise::unary::leaky_relu(data)
           .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Tensor { inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))) })
    }
    
    #[napi]
    pub fn zero_grad(&self) {
        self.inner.borrow().zero_grad();
    }
    
    #[napi]
    pub fn detach(&self) -> Tensor {
        Tensor { inner: Rc::new(RefCell::new(self.inner.borrow().detach())) }
    }
    
    /// Updates the data of this tensor with another tensor's data.
    /// This is used for in-place updates (e.g. optimizer steps).
    #[napi]
    pub fn assign(&self, other: &Tensor) {
        // In-place update of the data wrapped in RefCell
        let other_data = other.inner.borrow().data.clone();
        self.inner.borrow_mut().data = other_data;
    }
    
    #[napi]
    pub fn to_string(&self) -> String {
        format!("{}", self.inner.borrow())
    }
}
