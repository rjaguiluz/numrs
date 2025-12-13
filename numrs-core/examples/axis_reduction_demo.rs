use numrs::{Array, ops};
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Axis-based Reduction Operations Demo ===\n");

    // Create a 2D array
    let data_2d = Array::new(vec![2, 3], vec![
        1.0, 5.0, 3.0,
        4.0, 2.0, 6.0,
    ]);
    
    println!("Input 2D array (2x3):");
    println!("[[1, 5, 3],");
    println!(" [4, 2, 6]]\n");

    // Mean with axis
    println!("--- MEAN ---");
    let mean_none = ops::mean(&data_2d, None)?;
    println!("mean(axis=None) = {} (global mean)", mean_none.data[0]);
    
    let mean_axis0 = ops::mean(&data_2d, Some(0))?;
    println!("mean(axis=0) = {:?} (mean across rows)", mean_axis0.data);
    
    let mean_axis1 = ops::mean(&data_2d, Some(1))?;
    println!("mean(axis=1) = {:?} (mean across columns)\n", mean_axis1.data);

    // Variance with axis
    println!("--- VARIANCE ---");
    let var_none = ops::variance(&data_2d, None)?;
    println!("variance(axis=None) = {:.4}", var_none.data[0]);
    
    let var_axis0 = ops::variance(&data_2d, Some(0))?;
    println!("variance(axis=0) = {:?}", var_axis0.data.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    
    let var_axis1 = ops::variance(&data_2d, Some(1))?;
    println!("variance(axis=1) = {:?}\n", var_axis1.data.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());

    // ArgMax with axis
    println!("--- ARGMAX ---");
    let argmax_none = ops::argmax(&data_2d, None)?;
    println!("argmax(axis=None) = {} (index of global max: 6.0 at position 5)", argmax_none.data[0] as usize);
    
    let argmax_axis0 = ops::argmax(&data_2d, Some(0))?;
    println!("argmax(axis=0) = {:?} (row indices with max values)", 
        argmax_axis0.data.iter().map(|x| *x as usize).collect::<Vec<_>>());
    
    let argmax_axis1 = ops::argmax(&data_2d, Some(1))?;
    println!("argmax(axis=1) = {:?} (column indices with max values)\n", 
        argmax_axis1.data.iter().map(|x| *x as usize).collect::<Vec<_>>());

    // 3D example
    println!("\n=== 3D Example ===\n");
    let data_3d = Array::new(vec![2, 2, 2], vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]);
    
    println!("Input 3D array (2x2x2):");
    println!("[[[1, 2], [3, 4]],");
    println!(" [[5, 6], [7, 8]]]\n");

    println!("--- MEAN on different axes ---");
    let mean_3d_ax0 = ops::mean(&data_3d, Some(0))?;
    println!("mean(axis=0) shape: {:?}, data: {:?}", mean_3d_ax0.shape, mean_3d_ax0.data);
    
    let mean_3d_ax1 = ops::mean(&data_3d, Some(1))?;
    println!("mean(axis=1) shape: {:?}, data: {:?}", mean_3d_ax1.shape, mean_3d_ax1.data);
    
    let mean_3d_ax2 = ops::mean(&data_3d, Some(2))?;
    println!("mean(axis=2) shape: {:?}, data: {:?}", mean_3d_ax2.shape, mean_3d_ax2.data);

    println!("\n--- ARGMAX on different axes ---");
    let argmax_3d_ax0 = ops::argmax(&data_3d, Some(0))?;
    println!("argmax(axis=0) shape: {:?}, indices: {:?}", 
        argmax_3d_ax0.shape, 
        argmax_3d_ax0.data.iter().map(|x| *x as usize).collect::<Vec<_>>());
    
    let argmax_3d_ax1 = ops::argmax(&data_3d, Some(1))?;
    println!("argmax(axis=1) shape: {:?}, indices: {:?}", 
        argmax_3d_ax1.shape,
        argmax_3d_ax1.data.iter().map(|x| *x as usize).collect::<Vec<_>>());
    
    let argmax_3d_ax2 = ops::argmax(&data_3d, Some(2))?;
    println!("argmax(axis=2) shape: {:?}, indices: {:?}", 
        argmax_3d_ax2.shape,
        argmax_3d_ax2.data.iter().map(|x| *x as usize).collect::<Vec<_>>());

    // ML use case: batch prediction
    println!("\n=== ML Use Case: Batch Classification ===\n");
    let batch_logits = Array::new(vec![3, 5], vec![
        0.1, 0.2, 0.8, 0.3, 0.1,  // Sample 1: class 2
        0.5, 0.1, 0.2, 0.1, 0.9,  // Sample 2: class 4
        0.3, 0.7, 0.4, 0.2, 0.1,  // Sample 3: class 1
    ]);
    
    println!("Batch of 3 samples with 5 classes each:");
    println!("Sample 1 logits: [0.1, 0.2, 0.8, 0.3, 0.1]");
    println!("Sample 2 logits: [0.5, 0.1, 0.2, 0.1, 0.9]");
    println!("Sample 3 logits: [0.3, 0.7, 0.4, 0.2, 0.1]\n");

    let predictions = ops::argmax(&batch_logits, Some(1))?;
    println!("Predicted classes (argmax along axis=1):");
    for (i, &class) in predictions.data.iter().enumerate() {
        println!("  Sample {}: Class {}", i + 1, class as usize);
    }

    println!("\n✅ All axis-based reductions working with backend support!");
    println!("📊 mean, variance, and argmax now support axis parameter");
    println!("🚀 Backend dispatch: CPU-Scalar always, CPU-SIMD for mean (axis=None)");

    Ok(())
}
