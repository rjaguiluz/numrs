//! Test super simple: verificar que los gradientes se calculan y aplican

use numrs::{Array, Tensor, Linear, Module};
use numrs::{Adam};
use numrs::autograd::Optimizer;
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔬 Test Minimal: Gradients Flow");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // Crear Linear simple
    let linear = Linear::new(1, 1)?;
    let params = linear.parameters();
    
    println!("📊 Pesos iniciales:");
    {
        let w = params[0].borrow();
        let b = params[1].borrow();
        println!("  Weight: {:.4}", w.data.data[0]);
        println!("  Bias:   {:.4}", b.data.data[0]);
    }
    println!();
    
    // Input simple
    let x = Tensor::new(Array::new(vec![1, 1], vec![2.0]), false);
    let y_true = Tensor::new(Array::new(vec![1, 1], vec![10.0]), false);
    
    println!("📊 Forward pass:");
    println!("  x = 2.0");
    println!("  y_true = 10.0");
    
    // Forward
    let y_pred = linear.forward(&x)?;
    println!("  y_pred = {:.4}", y_pred.data.data[0]);
    println!();
    
    // Loss = MSE
    let loss = y_pred.mse_loss(&y_true)?;
    
    println!("📊 Loss:");
    println!("  loss = {:.4}", loss.data.data[0]);
    println!();
    
    // Backward
    println!("📊 Calculando gradientes...");
    loss.backward()?;
    
    {
        let w = params[0].borrow();
        let b = params[1].borrow();
        
        println!("  Gradientes calculados:");
        if let Some(w_grad) = w.gradient() {
            println!("    dL/dW = {:.4}", w_grad.data[0]);
        } else {
            println!("    dL/dW = None ❌");
        }
        
        if let Some(b_grad) = b.gradient() {
            println!("    dL/dB = {:.4}", b_grad.data[0]);
        } else {
            println!("    dL/dB = None ❌");
        }
    }
    println!();
    
    // Optimizer step
    println!("📊 Aplicando optimizer (Adam, lr=0.1)...");
    let mut optimizer = Adam::with_lr(params.clone(), 0.1);
    optimizer.step()?;
    
    {
        let w = params[0].borrow();
        let b = params[1].borrow();
        println!("  Pesos después de step:");
        println!("    Weight: {:.4}", w.data.data[0]);
        println!("    Bias:   {:.4}", b.data.data[0]);
    }
    println!();
    
    println!("═══════════════════════════════════════════════════════════");
    println!("  Si Weight o Bias NO cambiaron → BUG en optimizer");
    println!("  Si gradientes son None → BUG en backward");
    println!("═══════════════════════════════════════════════════════════\n");
    
    Ok(())
}
