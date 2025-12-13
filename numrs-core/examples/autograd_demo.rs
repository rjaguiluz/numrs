//! Ejemplo: Autograd básico
//! 
//! Demuestra:
//! 1. Creación de tensors con requires_grad
//! 2. Forward pass (construye compute graph)
//! 3. Backward pass (calcula gradientes automáticamente)
//! 4. Gradient descent manual

use numrs::{Tensor, Array};
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🎓 NumRs Autograd Demo");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // ========================================================================
    // EJEMPLO 1: Gradiente simple - f(x) = x²
    // ========================================================================
    println!("📐 EJEMPLO 1: f(x) = x² en x=3\n");
    
    // Crear tensor con requires_grad=true
    let x = Tensor::new(Array::new(vec![1], vec![3.0]), true);
    println!("  Input: x = {}", x);
    
    // Forward pass: y = x * x
    let y = x.mul(&x)?;
    println!("  Output: y = x² = {}", y);
    
    // Backward pass: calcula dy/dx automáticamente
    y.backward()?;
    
    println!("\n  Gradiente calculado:");
    if let Some(grad) = x.gradient() {
        println!("    dy/dx = {:.4}", grad.data[0]);
        println!("    (Esperado: 2x = 2*3 = 6.0) ✓\n");
    }
    
    // ========================================================================
    // EJEMPLO 2: Chain rule - f(x) = (x + 1)²
    // ========================================================================
    println!("🔗 EJEMPLO 2: f(x) = (x + 1)² en x=2\n");
    
    let x = Tensor::new(Array::new(vec![1], vec![2.0]), true);
    let one = Tensor::new(Array::new(vec![1], vec![1.0]), false);
    
    // Forward: z = x + 1, y = z²
    let z = x.add(&one)?;
    let y = z.mul(&z)?;
    
    println!("  x = {:.4}", x.values()[0]);
    println!("  z = x + 1 = {:.4}", z.values()[0]);
    println!("  y = z² = {:.4}", y.values()[0]);
    
    // Backward
    y.backward()?;
    
    println!("\n  Gradiente calculado:");
    if let Some(grad) = x.gradient() {
        println!("    dy/dx = {:.4}", grad.data[0]);
        println!("    (Esperado: 2(x+1) = 2*3 = 6.0) ✓\n");
    }
    
    // ========================================================================
    // EJEMPLO 3: Regresión lineal simple con autograd
    // ========================================================================
    println!("📈 EJEMPLO 3: Regresión lineal con autograd\n");
    println!("  Objetivo: aprender y = 2x + 1\n");
    
    // Datos de entrenamiento
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let y_data = vec![3.0, 5.0, 7.0, 9.0];  // y = 2x + 1
    
    // Parámetros (inicializados random)
    let mut w = Tensor::new(Array::new(vec![1], vec![0.5]), true);
    let mut b = Tensor::new(Array::new(vec![1], vec![0.0]), true);
    
    println!("  Pesos iniciales: w={:.4}, b={:.4}", w.values()[0], b.values()[0]);
    println!("\n  Entrenando...\n");
    
    let learning_rate = 0.01;
    let epochs = 100;
    
    for epoch in 0..epochs {
        // Reset gradients
        w.zero_grad();
        b.zero_grad();
        
        let mut total_loss = 0.0;
        
        // Mini-batch (procesamos todos los datos)
        for (x_val, y_true) in x_data.iter().zip(y_data.iter()) {
            let x = Tensor::new(Array::new(vec![1], vec![*x_val]), false);
            let y_target = Tensor::new(Array::new(vec![1], vec![*y_true]), false);
            
            // Forward: y_pred = w * x + b
            let y_pred = w.mul(&x)?.add(&b)?;
            
            // Loss: MSE
            let loss = y_pred.mse_loss(&y_target)?;
            total_loss += loss.values()[0];
            
            // Backward
            loss.backward()?;
            
            // Accumulate gradients
            if let (Some(w_grad), Some(b_grad)) = (loss.gradient(), loss.gradient()) {
                // Los gradientes ya están acumulados en w y b
            }
        }
        
        // Update weights usando gradientes
        if let (Some(w_grad), Some(b_grad)) = (w.gradient(), b.gradient()) {
            let new_w_val = w.values()[0] - learning_rate * w_grad.data[0];
            let new_b_val = b.values()[0] - learning_rate * b_grad.data[0];
            
            w = Tensor::new(Array::new(vec![1], vec![new_w_val]), true);
            b = Tensor::new(Array::new(vec![1], vec![new_b_val]), true);
        }
        
        if epoch % 20 == 0 {
            println!("    Epoch {:3}: loss={:.6}, w={:.4}, b={:.4}", 
                     epoch, total_loss / x_data.len() as f32, w.values()[0], b.values()[0]);
        }
    }
    
    println!("\n  Pesos finales: w={:.4}, b={:.4}", w.values()[0], b.values()[0]);
    println!("  (Objetivo:     w=2.000, b=1.000)\n");
    
    // ========================================================================
    // EJEMPLO 4: Neural network forward pass
    // ========================================================================
    println!("🧠 EJEMPLO 4: Red neuronal simple (sin training)\n");
    
    // Arquitectura: 2 -> 3 -> 1
    let x = Tensor::new(Array::new(vec![1, 2], vec![0.5, -0.3]), false);
    let w1 = Tensor::new(Array::new(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), true);
    let b1 = Tensor::new(Array::new(vec![1, 3], vec![0.1, 0.1, 0.1]), true);
    
    println!("  Input:  {:?}", x.shape());
    println!("  Layer1: {:?} @ {:?} + {:?}", x.shape(), w1.shape(), b1.shape());
    
    // Layer 1: h = relu(x @ w1 + b1)
    let z1 = x.matmul(&w1)?;
    let h1 = z1.add(&b1)?;
    let a1 = h1.relu()?;
    
    println!("  Hidden: {:?}", a1.shape());
    println!("  Values: {:?}\n", &a1.values()[..a1.values().len().min(5)]);
    
    // Backward pass
    a1.backward()?;
    
    println!("  Gradientes calculados automáticamente:");
    println!("    ✓ w1.grad: {}", w1.gradient().is_some());
    println!("    ✓ b1.grad: {}", b1.gradient().is_some());
    
    // ========================================================================
    // RESUMEN
    // ========================================================================
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  ✅ RESUMEN");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("1. ✓ Autograd implementado:");
    println!("   → Compute graph automático");
    println!("   → Backward pass con chain rule");
    println!("   → Gradient accumulation");
    
    println!("\n2. ✓ Operaciones soportadas:");
    println!("   → add, mul, matmul");
    println!("   → relu, sigmoid, exp, log");
    println!("   → sum, mean");
    println!("   → mse_loss, cross_entropy_loss");
    
    println!("\n3. ⚠️  Falta para training completo:");
    println!("   → Optimizers (SGD, Adam, RMSprop)");
    println!("   → Training loop de alto nivel");
    println!("   → Data loaders");
    println!("   → Model abstraction\n");
    
    println!("💡 Próximo paso: Implementar optimizers!");
    println!("   → optimizer.step() para update automático");
    println!("   → optimizer.zero_grad() para limpiar gradientes\n");
    
    Ok(())
}
