//! Comparación de 13 optimizers en NumRs

use numrs::autograd::*;
use numrs::array::Array;
use anyhow::Result;

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════╗");
    println!("║  COMPARACIÓN DE OPTIMIZERS - NumRs       ║");
    println!("╚═══════════════════════════════════════════╝\n");
    
    println!("🎯 Problema: y = 2x (regresión simple)");
    println!("📐 Modelo: Linear(1 → 1)");
    println!("📊 50 muestras, 100 epochs\n");
    
    // Datos: y = 2x
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    for i in 0..50 {
        let x = (i as f32 - 25.0) / 10.0;
        x_data.push(vec![x]);
        y_data.push(2.0 * x);
    }
    
    let epochs = 100;
    let lr = 0.01;
    
    println!("Entrenando...\n");
    
    let mut results = Vec::new();
    
    // Optimizers originales
    results.push(("SGD", test_sgd(&x_data, &y_data, epochs, lr)?));
    results.push(("Adam", test_adam(&x_data, &y_data, epochs, lr)?));
    results.push(("AdamW", test_adamw(&x_data, &y_data, epochs, lr)?));
    results.push(("RMSprop", test_rmsprop(&x_data, &y_data, epochs, lr)?));
    results.push(("AdaGrad", test_adagrad(&x_data, &y_data, epochs, lr)?));
    results.push(("NAdam", test_nadam(&x_data, &y_data, epochs, lr)?));
    results.push(("RAdam", test_radam(&x_data, &y_data, epochs, lr)?));
    results.push(("AdaDelta", test_adadelta(&x_data, &y_data, epochs)?));
    
    // Nuevos optimizers
    results.push(("LAMB", test_lamb(&x_data, &y_data, epochs, lr)?));
    results.push(("AdaBound", test_adabound(&x_data, &y_data, epochs, lr)?));
    results.push(("LBFGS", test_lbfgs(&x_data, &y_data, epochs)?));
    results.push(("Rprop", test_rprop(&x_data, &y_data, epochs)?));
    results.push(("Lookahead", test_lookahead(&x_data, &y_data, epochs, lr)?));
    
    println!("\n╔═══════════════════════════════════════════╗");
    println!("║          RESULTADOS FINALES               ║");
    println!("╚═══════════════════════════════════════════╝\n");
    
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    for (i, (name, loss)) in results.iter().enumerate() {
        let medal = match i {
            0 => "🥇", 1 => "🥈", 2 => "🥉", _ => "  ",
        };
        println!("  {} #{} {:10} → loss = {:.6}", medal, i + 1, name, loss);
    }
    
    println!("\n✅ ¡{} optimizers probados!", results.len());
    
    Ok(())
}

fn train<O: Optimizer>(
    name: &str,
    model: &Linear,
    optimizer: &mut O,
    x_data: &[Vec<f32>],
    y_data: &[f32],
    epochs: usize,
) -> Result<f32> {
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        
        for (x, &y_true) in x_data.iter().zip(y_data.iter()) {
            let x_tensor = Tensor::new(Array::new(vec![1, 1], x.clone()), false);
            let output = model.forward(&x_tensor)?;
            
            let y_tensor = Tensor::new(Array::new(vec![1, 1], vec![y_true]), false);
            let diff: Vec<f32> = output.data.data.iter()
                .zip(y_tensor.data.data.iter())
                .map(|(a, b)| a - b)
                .collect();
            let loss: f32 = diff.iter().map(|x| x * x).sum();
            epoch_loss += loss;
            
            let grad = Array::new(output.data.shape.clone(), 
                diff.iter().map(|x| 2.0 * x).collect());
            model.backward(&x_tensor, &grad)?;
            
            optimizer.step()?;
            optimizer.zero_grad();
        }
        
        epoch_loss /= x_data.len() as f32;
        
        if epoch % 50 == 0 {
            println!("  {}: epoch {} → loss = {:.6}", name, epoch, epoch_loss);
        }
    }
    
    // Loss final
    let mut final_loss = 0.0;
    for (x, &y_true) in x_data.iter().zip(y_data.iter()) {
        let x_tensor = Tensor::new(Array::new(vec![1, 1], x.clone()), false);
        let output = model.forward(&x_tensor)?;
        let diff = output.data.data[0] - y_true;
        final_loss += diff * diff;
    }
    Ok(final_loss / x_data.len() as f32)
}

fn test_sgd(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = SGD::new(params, lr, 0.9, 0.0);
    train("SGD", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_adam(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = Adam::default(params, lr);
    train("Adam", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_adamw(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = AdamW::new(params, lr, 0.9, 0.999, 1e-8, 0.01);
    train("AdamW", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_rmsprop(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = RMSprop::default(params, lr);
    train("RMSprop", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_adagrad(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = AdaGrad::default(params, lr);
    train("AdaGrad", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_nadam(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = NAdam::default(params, lr);
    train("NAdam", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_radam(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = RAdam::default(params, lr);
    train("RAdam", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_adadelta(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = AdaDelta::default(params);
    train("AdaDelta", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_lamb(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = LAMB::default(params, lr);
    train("LAMB", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_adabound(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = AdaBound::default(params, lr);
    train("AdaBound", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_lbfgs(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = LBFGS::default(params);
    train("LBFGS", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_rprop(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    let mut optimizer = Rprop::default(params);
    train("Rprop", &model, &mut optimizer, x_data, y_data, epochs)
}

fn test_lookahead(x_data: &[Vec<f32>], y_data: &[f32], epochs: usize, lr: f32) -> Result<f32> {
    let model = Linear::new(1, 1)?;
    let params = model.parameters_mut();
    // Lookahead wrapping SGD
    let sgd = SGD::new(params.clone(), lr, 0.9, 0.0);
    let mut optimizer = Lookahead::new(sgd, params, 5, 0.5);
    train("Lookahead", &model, &mut optimizer, x_data, y_data, epochs)
}
