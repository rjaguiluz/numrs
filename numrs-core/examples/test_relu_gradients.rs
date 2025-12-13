use numrs::{Array, Tensor, Linear, Sequential, ReLU, Module};
use numrs::{TrainerBuilder, Dataset, MSELoss};
use numrs::autograd::LossFunction;
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔬 Test: ReLU Gradients");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    
    // TEST 1: ReLU forward y backward básico
    println!("📊 TEST 1: ReLU básico");
    {
        let x = Tensor::new(Array::new(vec![3], vec![-1.0, 0.0, 2.0]), true);
        
        println!("  Input: {:?}", x.data.data);
        
        let y = x.relu()?;
        println!("  ReLU output: {:?}", y.data.data);
        println!("  Esperado: [0.0, 0.0, 2.0]");
        
        // Backward
        y.backward()?;
        
        let grad = x.gradient().unwrap();
        println!("  Gradients: {:?}", grad.data);
        println!("  Esperado: [0.0, 0.0, 1.0]");
        println!();
    }
    
    // TEST 2: Composición Linear → ReLU → Linear
    println!("📊 TEST 2: Linear → ReLU → Linear");
    {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4)?),
            Box::new(ReLU),
            Box::new(Linear::new(4, 1)?),
        ]);
        
        // Input simple
        let x = Tensor::new(Array::new(vec![1, 2], vec![1.0, 1.0]), false);
        
        println!("  Input: {:?}", x.data.data);
        
        let y_pred = model.forward(&x)?;
        println!("  Prediction: {:.4}", y_pred.data.data[0]);
        
        let y_true = Tensor::new(Array::new(vec![1, 1], vec![5.0]), false);
        
        let loss_fn = MSELoss;
        let loss = loss_fn.compute(&y_pred, &y_true)?;
        
        println!("  Loss: {:.4}", loss.data.data[0]);
        
        loss.backward()?;
        
        // Verificar que todos los parámetros tienen gradientes
        let params = model.parameters();
        println!("\n  Verificando gradientes de parámetros:");
        
        for (i, param) in params.iter().enumerate() {
            let p = param.borrow();
            if let Some(grad_cell) = &p.grad {
                let grad = grad_cell.borrow();
                let grad_sum: f32 = grad.data.iter().map(|x| x.abs()).sum();
                let has_nonzero = grad_sum > 1e-6;
                
                println!("    Param {}: grad_sum = {:.6} {}", 
                    i, 
                    grad_sum,
                    if has_nonzero { "✅" } else { "❌ ZERO!" }
                );
            } else {
                println!("    Param {}: ❌ NO GRADIENT!", i);
            }
        }
        println!();
    }
    
    // TEST 3: XOR super simple (4 samples, 1 epoch)
    println!("📊 TEST 3: XOR mínimo (1 epoch)");
    {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 4)?),
            Box::new(ReLU),
            Box::new(Linear::new(4, 1)?),
        ]);
        
        // XOR dataset
        let x_data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        
        let y_data = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
        ];
        
        let dataset = Dataset::new(x_data, y_data, 4);
        
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(0.1)
            .build_adam(Box::new(MSELoss));
        
        println!("  Loss inicial:");
        let metrics = trainer.train_epoch(&dataset)?;
        println!("    Epoch 1: loss = {:.6}", metrics.loss);
        
        println!("\n  Verificando que los pesos cambiaron:");
        let params = trainer.model.parameters();
        
        // Guardar pesos actuales
        let weights_after: Vec<f32> = params.iter()
            .flat_map(|p| p.borrow().data.data.clone())
            .collect();
        
        // Entrenar 1 epoch más
        let metrics2 = trainer.train_epoch(&dataset)?;
        println!("    Epoch 2: loss = {:.6}", metrics2.loss);
        
        // Verificar cambio
        let weights_after2: Vec<f32> = params.iter()
            .flat_map(|p| p.borrow().data.data.clone())
            .collect();
        
        let max_change = weights_after.iter().zip(weights_after2.iter())
            .map(|(w1, w2)| (w1 - w2).abs())
            .fold(0.0f32, |a, b| a.max(b));
        
        println!("    Max weight change: {:.6}", max_change);
        
        if max_change > 1e-6 {
            println!("    ✅ Pesos cambian");
        } else {
            println!("    ❌ Pesos NO cambian");
        }
        
        if (metrics2.loss - metrics.loss).abs() > 1e-6 {
            println!("    ✅ Loss cambia");
        } else {
            println!("    ❌ Loss NO cambia");
        }
    }
    
    println!();
    println!("═══════════════════════════════════════════════════════════");
    
    Ok(())
}
