/// Demo of the 6 critical ML operations
use numrs::{Array, ops};
use anyhow::Result;

fn main() -> Result<()> {
    println!("========================================");
    println!("   NumRs - Critical ML Operations Demo");
    println!("========================================\n");
    
    // 1. RESHAPE
    println!("1. RESHAPE - Change tensor dimensions");
    let data = Array::new(vec![12], (0..12).map(|x| x as f32).collect());
    println!("   Original: {:?}", data.shape());
    let reshaped = ops::reshape(&data, &[3, 4])?;
    println!("   Reshaped to [3, 4]: {:?}\n", reshaped.shape());
    
    // 2. TRANSPOSE
    println!("2. TRANSPOSE - Swap dimensions");
    let matrix = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("   Original [2, 3]");
    let transposed = ops::transpose(&matrix, None)?;
    println!("   Transposed [3, 2]\n");
    
    // 3. CONCAT
    println!("3. CONCAT - Combine arrays");
    let batch1 = Array::new(vec![2, 4], vec![1.0; 8]);
    let batch2 = Array::new(vec![3, 4], vec![2.0; 12]);
    let combined = ops::concat(&[&batch1, &batch2], 0)?;
    println!("   Combined batches: {:?}\n", combined.shape());
    
    // 4. VARIANCE
    println!("4. VARIANCE - Statistical analysis");
    let data = Array::new(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let var = ops::variance(&data, None)?;
    println!("   Data variance: {:.4}\n", var.to_f32().data[0]);
    
    // 5. ARGMAX
    println!("5. ARGMAX - Find predictions");
    let probs = Array::new(vec![5], vec![0.05, 0.10, 0.60, 0.20, 0.05]);
    let predicted = ops::argmax(&probs, None)?;
    println!("   Predicted class: {}\n", predicted.to_f32().data[0] as usize);
    
    // 6. CROSS_ENTROPY
    println!("6. CROSS_ENTROPY - Training loss");
    let pred = Array::new(vec![3], vec![0.7, 0.2, 0.1]);
    let target = Array::new(vec![3], vec![1.0, 0.0, 0.0]);
    let loss = ops::cross_entropy(&pred, &target)?;
    println!("   Loss: {:.4}\n", loss.to_f32().data[0]);
    
    println!("========================================");
    println!("✅ ALL 6 OPERATIONS WORKING!");
    println!("   Progress: 34/64 ops (53.1%)");
    println!("========================================");
    
    Ok(())
}
