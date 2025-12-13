use numrs::{Array, Tensor, Linear, Sequential, ReLU, Module};
use numrs::{TrainerBuilder, Dataset, MSELoss};
use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  ⚡ Training Performance Profiling");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // Dataset más grande para ver overhead
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    
    for i in 0..100 {
        let x = i as f32 / 10.0;
        x_data.push(vec![x]);
        y_data.push(vec![2.0 * x + 3.0]);
    }
    
    println!("📊 Dataset: 100 samples, y = 2x + 3");
    println!("📐 Model: Linear(1 → 8 → 1)\n");
    
    let model = Sequential::new(vec![
        Box::new(Linear::new(1, 8)?),
        Box::new(ReLU),
        Box::new(Linear::new(8, 1)?),
    ]);
    
    let dataset = Dataset::new(x_data, y_data, 32);  // batch_size=32
    
    let mut trainer = TrainerBuilder::new(model)
        .learning_rate(0.05)
        .build_adam(Box::new(MSELoss));
    
    println!("⏱️  Timing individual epochs:\n");
    
    for epoch in 0..10 {
        let start = Instant::now();
        let metrics = trainer.train_epoch(&dataset)?;
        let elapsed = start.elapsed();
        
        println!("  Epoch {:2}: loss = {:8.4}  time = {:6.2} ms", 
            epoch + 1, 
            metrics.loss,
            elapsed.as_secs_f64() * 1000.0
        );
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    
    // Benchmark componentes individuales
    println!("\n🔍 Component Breakdown:\n");
    
    // 1. Forward pass solo
    {
        let x = Tensor::new(Array::new(vec![32, 1], vec![1.0; 32]), false);
        let model = Sequential::new(vec![
            Box::new(Linear::new(1, 8)?),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)?),
        ]);
        
        let start = Instant::now();
        for _ in 0..100 {
            let _y = model.forward(&x)?;
        }
        let elapsed = start.elapsed();
        
        println!("  Forward pass (100x):     {:6.2} μs/iter", 
            elapsed.as_micros() as f64 / 100.0);
    }
    
    // 2. Forward + Backward
    {
        let x = Tensor::new(Array::new(vec![32, 1], vec![1.0; 32]), false);
        let model = Sequential::new(vec![
            Box::new(Linear::new(1, 8)?),
            Box::new(ReLU),
            Box::new(Linear::new(8, 1)?),
        ]);
        
        let start = Instant::now();
        for _ in 0..100 {
            let y = model.forward(&x)?;
            y.backward()?;
        }
        let elapsed = start.elapsed();
        
        println!("  Forward+Backward (100x): {:6.2} μs/iter", 
            elapsed.as_micros() as f64 / 100.0);
    }
    
    // 3. Matmul small (típico en training)
    {
        use numrs::ops::matmul;
        
        let a = Array::new(vec![32, 8], vec![1.0; 32*8]);
        let b = Array::new(vec![8, 1], vec![1.0; 8*1]);
        
        let start = Instant::now();
        for _ in 0..1000 {
            let _c = matmul(&a, &b)?;
        }
        let elapsed = start.elapsed();
        
        println!("  Matmul 32x8 @ 8x1 (1000x): {:6.2} μs/iter", 
            elapsed.as_micros() as f64 / 1000.0);
    }
    
    // 4. ReLU
    {
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        
        let start = Instant::now();
        for _ in 0..1000 {
            let _y = x.relu()?;
        }
        let elapsed = start.elapsed();
        
        println!("  ReLU 32x8 (1000x):       {:6.2} μs/iter", 
            elapsed.as_micros() as f64 / 1000.0);
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    
    Ok(())
}
