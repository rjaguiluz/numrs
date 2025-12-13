use crate::NumRsArray;
use numrs::ops::{linalg, reduction, shape};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct Tensor {
    // Wrapped internally, not public to JS
    pub(crate) inner: Rc<RefCell<numrs::Tensor>>,
}

#[wasm_bindgen]
impl Tensor {
    // Constructor from Array
    #[wasm_bindgen(constructor)]
    pub fn new(data: &NumRsArray, requires_grad: bool) -> Tensor {
        // Clone the internal array data to create a new Tensor
        // Core Tensor::new takes ownership or clone
        Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(
                data.inner.clone(),
                requires_grad,
            ))),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape().to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn data(&self) -> NumRsArray {
        // Return a copy of the data as an Array wrapper
        NumRsArray {
            inner: self.inner.borrow().data.clone(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn grad(&self) -> Option<NumRsArray> {
        // Return gradient as Array if present
        self.inner
            .borrow()
            .gradient()
            .map(|g| NumRsArray { inner: g })
    }

    pub fn backward(&self) -> Result<(), JsValue> {
        self.inner
            .borrow()
            .backward()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // --- Ops ---

    pub fn add(&self, other: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .add(&*other.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .sub(&*other.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .mul(&*other.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .div(&*other.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .matmul(&*other.inner.borrow())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn pow(&self, exponent: f32) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .pow(exponent)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn neg(&self) -> Result<Tensor, JsValue> {
        // Core might not have direct tensor.neg(), check linkage later or implement via mul(-1)
        // Using -1 * self approach if needed, or if core supports unary neg op on tensor
        // Assuming core Tensor has or can support arithmetic ops.
        // Actually, core Tensor implements ops via traits.
        // Let's safe bet: mul by -1 constant tensor
        let neg_one_arr = numrs::Array::new(vec![1], vec![-1.0]);
        let neg_one = numrs::Tensor::new(neg_one_arr, false);
        let res = self
            .inner
            .borrow()
            .mul(&neg_one)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    // --- Elementwise Ops (delegated to Tensor) ---

    pub fn sqrt(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .sqrt()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn relu(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .relu()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn sigmoid(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .sigmoid()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn exp(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .exp()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn log(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .log()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn sin(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .sin()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn cos(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .cos()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn tan(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .tan()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn tanh(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .tanh()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    // --- Elementwise Ops using Array (Detached / No Autograd yet) ---

    pub fn abs(&self) -> Result<Tensor, JsValue> {
        let res_arr = numrs::ops::elementwise::unary::abs(&self.inner.borrow().data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    pub fn asin(&self) -> Result<Tensor, JsValue> {
        let res_arr = numrs::ops::elementwise::unary::asin(&self.inner.borrow().data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    pub fn acos(&self) -> Result<Tensor, JsValue> {
        let res_arr = numrs::ops::elementwise::unary::acos(&self.inner.borrow().data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    pub fn atan(&self) -> Result<Tensor, JsValue> {
        let res_arr = numrs::ops::elementwise::unary::atan(&self.inner.borrow().data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    pub fn softplus(&self) -> Result<Tensor, JsValue> {
        let res_arr = numrs::ops::elementwise::unary::softplus(&self.inner.borrow().data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    pub fn leaky_relu(&self) -> Result<Tensor, JsValue> {
        // Warning: might fail if leaky_relu signature changed, but matching JS
        let res_arr = numrs::ops::elementwise::unary::leaky_relu(&self.inner.borrow().data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    // --- Manipulation / Utils ---

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().zero_grad();
    }

    pub fn detach(&self) -> Tensor {
        Tensor {
            inner: Rc::new(RefCell::new(self.inner.borrow().detach())),
        }
    }

    pub fn transpose(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .transpose()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .reshape(shape)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn flatten(&self) -> Result<Tensor, JsValue> {
        let numel: usize = self.inner.borrow().shape().iter().product();
        let res = self
            .inner
            .borrow()
            .reshape(vec![numel])
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    // --- Reduction ---

    pub fn sum(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .sum()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn mean(&self) -> Result<Tensor, JsValue> {
        let res = self
            .inner
            .borrow()
            .mean()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(res)),
        })
    }

    pub fn max(&self, axis: Option<i32>) -> Result<Tensor, JsValue> {
        let ax = axis.map(|d| d as usize);
        let res_arr = reduction::max::max(&self.inner.borrow().data, ax)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    pub fn min(&self, axis: Option<i32>) -> Result<Tensor, JsValue> {
        let ax = axis.map(|d| d as usize);
        let res_arr = reduction::min::min(&self.inner.borrow().data, ax)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    pub fn argmax(&self, axis: Option<i32>) -> Result<Tensor, JsValue> {
        let ax = axis.map(|d| d as usize);
        let res_arr = reduction::argmax::argmax(&self.inner.borrow().data, ax)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    pub fn variance(&self, axis: Option<i32>) -> Result<Tensor, JsValue> {
        let ax = axis.map(|d| d as usize);
        let res_arr = reduction::variance::variance(&self.inner.borrow().data, ax)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    // --- Shape ---

    pub fn broadcast_to(&self, shape: Vec<usize>) -> Result<Tensor, JsValue> {
        let res_arr = shape::broadcast_to::broadcast_to(&self.inner.borrow().data, &shape)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    #[wasm_bindgen]
    pub fn concat(inputs: Vec<Tensor>, axis: usize) -> Result<Tensor, JsValue> {
        // Collect references to inner arrays
        // Note: inputs is Vec<Tensor>, so we own them.
        // We need Vec<&Array> for core concat.
        // We can get refs to inner.data.

        // We need to keep the borrowed cells alive.
        // Let's first collect the Inner Tensors then borrow data.
        let inners: Vec<Rc<RefCell<numrs::Tensor>>> =
            inputs.iter().map(|t| t.inner.clone()).collect();
        let borrows: Vec<std::cell::Ref<numrs::Tensor>> =
            inners.iter().map(|t| t.borrow()).collect();
        let arrays: Vec<numrs::Array> = borrows.iter().map(|t| t.data.clone()).collect();
        let array_refs: Vec<&numrs::Array> = arrays.iter().collect();

        let res_arr = shape::concat::concat(&array_refs, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }

    // --- Linear Algebra ---

    pub fn dot(&self, other: &Tensor) -> Result<Tensor, JsValue> {
        let res_arr = linalg::dot::dot(&self.inner.borrow().data, &other.inner.borrow().data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Tensor {
            inner: Rc::new(RefCell::new(numrs::Tensor::new(res_arr, false))),
        })
    }
}
