use numrs::{Array, Tensor, Linear, Sequential, ReLU, Module};
use numrs::{TrainerBuilder, Dataset, MSELoss};
use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  🔬 Detailed Performance Analysis");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // Test 1: Overhead de operaciones individuales
    println!("📊 Test 1: Operation Overhead\n");
    
    {
        let a = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        let b = Tensor::new(Array::new(vec![8, 4], vec![1.0; 8*4]), true);
        
        let start = Instant::now();
        for _ in 0..10000 {
            let _c = a.matmul(&b)?;
        }
        let elapsed = start.elapsed();
        println!("  Matmul 32x8 @ 8x4 (10k):  {:6.2} μs/iter", elapsed.as_micros() as f64 / 10000.0);
    }
    
    {
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        
        let start = Instant::now();
        for _ in 0..10000 {
            let _y = x.relu()?;
        }
        let elapsed = start.elapsed();
        println!("  ReLU 32x8 (10k):          {:6.2} μs/iter", elapsed.as_micros() as f64 / 10000.0);
    }
    
    {
        let a = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        let b = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        
        let start = Instant::now();
        for _ in 0..10000 {
            let _c = a.add(&b)?;
        }
        let elapsed = start.elapsed();
        println!("  Add 32x8 (10k):           {:6.2} μs/iter", elapsed.as_micros() as f64 / 10000.0);
    }
    
    {
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        
        let start = Instant::now();
        for _ in 0..10000 {
            let _t = x.transpose()?;
        }
        let elapsed = start.elapsed();
        println!("  Transpose 32x8 (10k):     {:6.2} μs/iter", elapsed.as_micros() as f64 / 10000.0);
    }
    
    // Test 2: Overhead de backward
    println!("\n📊 Test 2: Backward Pass Overhead\n");
    
    {
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        
        let start = Instant::now();
        for _ in 0..1000 {
            let y = x.relu()?;
            y.backward()?;
        }
        let elapsed = start.elapsed();
        println!("  ReLU forward+backward (1k): {:6.2} μs/iter", elapsed.as_micros() as f64 / 1000.0);
    }
    
    {
        let a = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        let b = Tensor::new(Array::new(vec![8, 4], vec![1.0; 8*4]), true);
        
        let start = Instant::now();
        for _ in 0..1000 {
            let c = a.matmul(&b)?;
            c.backward()?;
        }
        let elapsed = start.elapsed();
        println!("  Matmul forward+backward (1k): {:6.2} μs/iter", elapsed.as_micros() as f64 / 1000.0);
    }
    
    // Test 3: Linear layer breakdown
    println!("\n📊 Test 3: Linear Layer Breakdown\n");
    
    {
        let linear = Linear::new(8, 4)?;
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), false);
        
        let start = Instant::now();
        for _ in 0..10000 {
            let _y = linear.forward(&x)?;
        }
        let elapsed = start.elapsed();
        println!("  Linear(8,4) forward (10k): {:6.2} μs/iter", elapsed.as_micros() as f64 / 10000.0);
    }
    
    {
        let linear = Linear::new(8, 4)?;
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), false);
        
        let start = Instant::now();
        for _ in 0..1000 {
            let y = linear.forward(&x)?;
            y.backward()?;
        }
        let elapsed = start.elapsed();
        println!("  Linear(8,4) fwd+bwd (1k): {:6.2} μs/iter", elapsed.as_micros() as f64 / 1000.0);
    }
    
    // Test 4: Pattern Linear + ReLU
    println!("\n📊 Test 4: Linear+ReLU Pattern (fusionable)\n");
    
    {
        let linear = Linear::new(8, 4)?;
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), false);
        
        let start = Instant::now();
        for _ in 0..10000 {
            let y = linear.forward(&x)?;
            let _z = y.relu()?;
        }
        let elapsed = start.elapsed();
        println!("  Linear+ReLU forward (10k): {:6.2} μs/iter", elapsed.as_micros() as f64 / 10000.0);
    }
    
    {
        let linear = Linear::new(8, 4)?;
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), false);
        
        let start = Instant::now();
        for _ in 0..1000 {
            let y = linear.forward(&x)?;
            let z = y.relu()?;
            z.backward()?;
        }
        let elapsed = start.elapsed();
        println!("  Linear+ReLU fwd+bwd (1k): {:6.2} μs/iter", elapsed.as_micros() as f64 / 1000.0);
    }
    
    // Test 5: Clone overhead
    println!("\n📊 Test 5: Clone Overhead Analysis\n");
    
    {
        let x = Tensor::new(Array::new(vec![32, 8], vec![1.0; 32*8]), true);
        
        let start = Instant::now();
        for _ in 0..100000 {
            let _y = x.clone();
        }
        let elapsed = start.elapsed();
        println!("  Tensor clone 32x8 (100k): {:6.2} μs/iter", elapsed.as_micros() as f64 / 100000.0);
    }
    
    {
        let arr = Array::new(vec![32, 8], vec![1.0; 32*8]);
        
        let start = Instant::now();
        for _ in 0..100000 {
            let _y = arr.clone();
        }
        let elapsed = start.elapsed();
        println!("  Array clone 32x8 (100k):  {:6.2} μs/iter", elapsed.as_micros() as f64 / 100000.0);
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    
    Ok(())
}
