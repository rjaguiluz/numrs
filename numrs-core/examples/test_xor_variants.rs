use numrs::{Array, Tensor, Linear, Sequential, ReLU, Sigmoid, Module};
use numrs::{TrainerBuilder, Dataset, MSELoss};
use anyhow::Result;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔬 XOR: ReLU vs Sigmoid");
    println!("═══════════════════════════════════════════════════════════\n");
    
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
    
    println!("📊 Dataset: XOR");
    println!("  [0, 0] → 0");
    println!("  [0, 1] → 1");
    println!("  [1, 0] → 1");
    println!("  [1, 1] → 0\n");
    
    // TEST 1: Con ReLU
    println!("🔴 TEST 1: XOR con ReLU (2 → 8 → 1)");
    {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 8)?),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)?),
        ]);
        
        let dataset = Dataset::new(x_data.clone(), y_data.clone(), 4);
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(0.05)
            .build_adam(Box::new(MSELoss));
        
        println!("  Lr: 0.05, Epochs: 500");
        
        for epoch in 0..500 {
            let metrics = trainer.train_epoch(&dataset)?;
            
            if epoch % 100 == 0 || epoch == 499 {
                println!("    Epoch {:3}: loss = {:.6}", epoch + 1, metrics.loss);
            }
        }
        println!();
    }
    
    // TEST 2: Con Sigmoid
    println!("🟢 TEST 2: XOR con Sigmoid (2 → 8 → 1)");
    {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 8)?),
            Box::new(Sigmoid),
            Box::new(Linear::new(8, 1)?),
        ]);
        
        let dataset = Dataset::new(x_data.clone(), y_data.clone(), 4);
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(0.05)
            .build_adam(Box::new(MSELoss));
        
        println!("  Lr: 0.05, Epochs: 500");
        
        for epoch in 0..500 {
            let metrics = trainer.train_epoch(&dataset)?;
            
            if epoch % 100 == 0 || epoch == 499 {
                println!("    Epoch {:3}: loss = {:.6}", epoch + 1, metrics.loss);
            }
        }
        println!();
    }
    
    // TEST 3: Con ReLU y más neuronas
    println!("🔵 TEST 3: XOR con ReLU (2 → 16 → 8 → 1)");
    {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 16)?),
            Box::new(ReLU),
            Box::new(Linear::new(16, 8)?),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)?),
        ]);
        
        let dataset = Dataset::new(x_data.clone(), y_data.clone(), 4);
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(0.05)
            .build_adam(Box::new(MSELoss));
        
        println!("  Lr: 0.05, Epochs: 500");
        
        for epoch in 0..500 {
            let metrics = trainer.train_epoch(&dataset)?;
            
            if epoch % 100 == 0 || epoch == 499 {
                println!("    Epoch {:3}: loss = {:.6}", epoch + 1, metrics.loss);
            }
        }
        println!();
    }
    
    // TEST 4: Lr más alto
    println!("🟡 TEST 4: XOR con ReLU, lr alto (2 → 8 → 1)");
    {
        let model = Sequential::new(vec![
            Box::new(Linear::new(2, 8)?),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)?),
        ]);
        
        let dataset = Dataset::new(x_data.clone(), y_data.clone(), 4);
        let mut trainer = TrainerBuilder::new(model)
            .learning_rate(0.5)
            .build_adam(Box::new(MSELoss));
        
        println!("  Lr: 0.5, Epochs: 500");
        
        for epoch in 0..500 {
            let metrics = trainer.train_epoch(&dataset)?;
            
            if epoch % 100 == 0 || epoch == 499 {
                println!("    Epoch {:3}: loss = {:.6}", epoch + 1, metrics.loss);
            }
        }
        println!();
    }
    
    println!("═══════════════════════════════════════════════════════════");
    
    Ok(())
}
