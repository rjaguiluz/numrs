//! Tensor - Array con autograd metadata
//! 
//! Un Tensor es un Array + información para backpropagation:
//! - data: Array con los valores
//! - grad: Gradiente acumulado (dL/d(este tensor))
//! - requires_grad: Si este tensor necesita gradientes
//! - compute_node: Operación que creó este tensor
//! - is_leaf: Si es un parámetro del modelo (no resultado de operación)

use crate::array::Array;
use crate::autograd::{ComputeNode, NodeId};
use std::rc::Rc;
use std::cell::RefCell;
use anyhow::Result;

/// Tensor con soporte para autograd
#[derive(Clone)]
pub struct Tensor {
    /// Datos del tensor
    pub data: Array,
    
    /// Gradiente acumulado (dL/d(este tensor))
    pub grad: Option<Rc<RefCell<Array>>>,
    
    /// Si este tensor requiere gradientes
    pub requires_grad: bool,
    
    /// Nodo del compute graph (None para leaf tensors)
    pub compute_node: Option<Rc<ComputeNode>>,
    
    /// Si es un leaf tensor (parámetro del modelo)
    pub is_leaf: bool,
}

impl Tensor {
    /// Crea un nuevo tensor desde un Array
    /// 
    /// # Argumentos
    /// - `data`: Array con los datos
    /// - `requires_grad`: Si se deben calcular gradientes para este tensor
    pub fn new(data: Array, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            // Inicializar gradiente a ceros con la misma forma
            Some(Rc::new(RefCell::new(Array::new(
                data.shape.clone(),
                vec![0.0; data.data.len()],
            ))))
        } else {
            None
        };
        
        Self {
            data,
            grad,
            requires_grad,
            compute_node: None,
            is_leaf: true,  // Por defecto es leaf (creado por el usuario)
        }
    }
    
    /// Crea tensor desde resultado de operación
    pub fn from_operation(
        data: Array,
        compute_node: ComputeNode,
        requires_grad: bool,
    ) -> Self {
        let grad = if requires_grad {
            Some(Rc::new(RefCell::new(Array::new(
                data.shape.clone(),
                vec![0.0; data.data.len()],
            ))))
        } else {
            None
        };
        
        Self {
            data,
            grad,
            requires_grad,
            compute_node: Some(Rc::new(compute_node)),
            is_leaf: false,  // No es leaf (creado por operación)
        }
    }
    
    /// Acceso directo a la forma
    pub fn shape(&self) -> &[usize] {
        &self.data.shape
    }
    
    /// Acceso directo a los datos
    pub fn values(&self) -> &[f32] {
        &self.data.data
    }
    
    /// Obtiene el gradiente actual (si existe)
    pub fn gradient(&self) -> Option<Array> {
        self.grad.as_ref().map(|g| g.borrow().clone())
    }

    /// Obtiene el valor escalar de un tensor de 1 elemento
    pub fn item(&self) -> f32 {
        if self.data.data.is_empty() {
            0.0
        } else {
            self.data.data[0]
        }
    }
    
    /// Establece el gradiente manualmente
    pub fn set_gradient(&self, grad: Array) -> Result<()> {
        if let Some(grad_cell) = &self.grad {
            *grad_cell.borrow_mut() = grad;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Tensor does not require gradients"))
        }
    }
    
    /// Acumula gradiente (grad += incoming_grad)
    pub fn accumulate_gradient(&self, incoming_grad: &Array) -> Result<()> {
        if let Some(grad_cell) = &self.grad {
            let mut grad = grad_cell.borrow_mut();
            
            // grad += incoming_grad
            for (g, &ig) in grad.data.iter_mut().zip(incoming_grad.data.iter()) {
                *g += ig;
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("Tensor does not require gradients"))
        }
    }
    
    /// Resetea el gradiente a cero
    pub fn zero_grad(&self) {
        if let Some(grad_cell) = &self.grad {
            let mut grad = grad_cell.borrow_mut();
            for g in grad.data.iter_mut() {
                *g = 0.0;
            }
        }
    }
    
    /// Desconecta del compute graph (detach)
    /// Retorna un nuevo tensor sin gradientes
    pub fn detach(&self) -> Self {
        Self {
            data: self.data.clone(),
            grad: None,
            requires_grad: false,
            compute_node: None,
            is_leaf: true,
        }
    }
    
    /// Convierte a Array (sin autograd)
    pub fn to_array(&self) -> Array {
        self.data.clone()
    }
    
    /// Obtiene el ID del nodo en el compute graph
    pub fn node_id(&self) -> Option<NodeId> {
        self.compute_node.as_ref().map(|node| node.id)
    }
    
    /// Backpropagation - Calcula gradientes desde este tensor hacia las hojas
    /// 
    /// Este es el algoritmo principal de autograd:
    /// 1. Inicializa grad = 1.0 para este tensor (dL/dL = 1)
    /// 2. Recorre el graph en orden topológico inverso
    /// 3. Para cada operación, llama su backward_fn
    /// 4. Acumula gradientes en los inputs
    pub fn backward(&self) -> Result<()> {
        if !self.requires_grad {
            return Err(anyhow::anyhow!("Tensor does not require gradients"));
        }
        
        // Inicializar gradiente de salida (dL/dL = 1)
        let initial_grad = Array::new(
            self.data.shape.clone(),
            vec![1.0; self.data.data.len()],
        );
        self.set_gradient(initial_grad)?;
        
        // Topological sort + backward pass
        let mut topo_order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.build_topo(self, &mut topo_order, &mut visited);
        
        // Reverse order para backward pass
        topo_order.reverse();
        
        // Ejecutar backward para cada nodo
        for tensor in topo_order {
            if let Some(node) = &tensor.compute_node {
                if let Some(backward_fn) = &node.backward_fn {
                    // Obtener gradiente de este tensor
                    let grad_output = tensor.gradient()
                        .ok_or_else(|| anyhow::anyhow!("Missing gradient"))?;
                    
                    // Calcular gradientes para los inputs
                    let grad_inputs = backward_fn(&grad_output, &node.inputs, &tensor)?;
                    
                    // Acumular gradientes en cada input
                    for (input, grad_input) in node.inputs.iter().zip(grad_inputs.iter()) {
                        if input.requires_grad {
                            input.accumulate_gradient(grad_input)?;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Construye orden topológico del compute graph (DFS)
    fn build_topo<'a>(
        &'a self,
        node: &'a Tensor,
        topo_order: &mut Vec<&'a Tensor>,
        visited: &mut std::collections::HashSet<NodeId>,
    ) {
        if let Some(id) = node.node_id() {
            if visited.contains(&id) {
                return;
            }
            visited.insert(id);
        }
        
        // Recursivamente visitar inputs
        if let Some(compute_node) = &node.compute_node {
            for input in &compute_node.inputs {
                self.build_topo(input, topo_order, visited);
            }
        }
        
        topo_order.push(node);
    }
    
    /// Transpose del tensor (invierte dimensiones)
    pub fn transpose(&self) -> Result<Tensor> {
        use crate::ops::transpose as op_transpose;
        let transposed_data = op_transpose(&self.data, None)?;
        
        // Si requiere gradientes, crear nodo en el grafo computacional
        if self.requires_grad {
            use crate::autograd::{backward::transpose_backward, OpKind};
            let backward_fn = Box::new(transpose_backward);
            let node = ComputeNode::new(
                OpKind::Transpose,
                vec![self.clone()],
                Some(backward_fn),
            );
            Ok(Tensor::from_operation(transposed_data, node, true))
        } else {
            Ok(Tensor::new(transposed_data, false))
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.data.shape)
            .field("requires_grad", &self.requires_grad)
            .field("is_leaf", &self.is_leaf)
            .field("has_grad", &self.grad.is_some())
            .field("node_id", &self.node_id())
            .finish()
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?}, requires_grad={}, data=[", 
               self.data.shape, self.requires_grad)?;
        
        let n = self.data.data.len().min(5);
        for i in 0..n {
            write!(f, "{:.4}", self.data.data[i])?;
            if i < n - 1 {
                write!(f, ", ")?;
            }
        }
        if self.data.data.len() > 5 {
            write!(f, ", ...")?;
        }
        write!(f, "])")
    }
}
