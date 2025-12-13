/// Comprehensive validation of all ops operations
/// 
/// This example validates:
/// - All ops functions (add, mul, sub, div, sum, matmul)
/// - Different array sizes (small, medium, large)
/// - Backend selection (SIMD, BLAS, Scalar)
/// - Correctness of results
/// - Performance characteristics

use numrs::{Array, ops};
use numrs::backend::get_dispatch_table;
use anyhow::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("\n🧪 NumRs Operations Comprehensive Validation\n");
    println!("{}", "=".repeat(60));
    
    // Print dispatch table info
    let table = get_dispatch_table();
    println!("\n📊 Dispatch Table Configuration:");
    println!("  - Elementwise: {}", table.elementwise_backend);
    println!("  - Reduction:   {}", table.reduction_backend);
    println!("  - MatMul:      {}", table.matmul_backend);
    println!();
    
    // Test 1: Elementwise operations
    println!("{}", "=".repeat(60));
    println!("\n✅ Test 1: Elementwise Operations\n");
    test_elementwise()?;
    
    // Test 2: Reduction operations
    println!("{}", "=".repeat(60));
    println!("\n✅ Test 2: Reduction Operations\n");
    test_reductions()?;
    
    // Test 3: MatMul operations
    println!("{}", "=".repeat(60));
    println!("\n✅ Test 3: MatMul Operations\n");
    test_matmul()?;
    
    // Test 4: Different sizes
    println!("{}", "=".repeat(60));
    println!("\n✅ Test 4: Size Scalability\n");
    test_sizes()?;
    
    // Test 5: Edge cases
    println!("{}", "=".repeat(60));
    println!("\n✅ Test 5: Edge Cases\n");
    test_edge_cases()?;
    
    println!("{}", "=".repeat(60));
    println!("\n✅ All tests passed! ops validation complete.\n");
    
    Ok(())
}

fn test_elementwise() -> Result<()> {
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![4], vec![2.0, 2.0, 2.0, 2.0]);
    
    // Test add
    let c = ops::add(&a, &b)?;
    assert_eq!(c.data, vec![3.0, 4.0, 5.0, 6.0]);
    println!("  ✓ add([1,2,3,4], [2,2,2,2]) = [3,4,5,6]");
    
    // Test mul
    let c = ops::mul(&a, &b)?;
    assert_eq!(c.data, vec![2.0, 4.0, 6.0, 8.0]);
    println!("  ✓ mul([1,2,3,4], [2,2,2,2]) = [2,4,6,8]");
    
    // Test sub
    let c = ops::sub(&a, &b)?;
    assert_eq!(c.data, vec![-1.0, 0.0, 1.0, 2.0]);
    println!("  ✓ sub([1,2,3,4], [2,2,2,2]) = [-1,0,1,2]");
    
    // Test div
    let c = ops::div(&a, &b)?;
    assert_eq!(c.data, vec![0.5, 1.0, 1.5, 2.0]);
    println!("  ✓ div([1,2,3,4], [2,2,2,2]) = [0.5,1,1.5,2]");
    
    Ok(())
}

fn test_reductions() -> Result<()> {
    let a = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    
    // Test sum
    let s = ops::sum(&a, None)?;
    assert_eq!(s.data, vec![10.0]);
    println!("  ✓ sum([1,2,3,4]) = 10");
    
    // Test with different sizes
    let b = Array::new(vec![100], vec![1.0; 100]);
    let s = ops::sum(&b, None)?;
    assert_eq!(s.data[0], 100.0);
    println!("  ✓ sum(100 ones) = 100");
    
    let c = Array::new(vec![1000], vec![0.5; 1000]);
    let s = ops::sum(&c, None)?;
    assert_eq!(s.data[0], 500.0);
    println!("  ✓ sum(1000 halves) = 500");
    
    Ok(())
}

fn test_matmul() -> Result<()> {
    // Test 2x2
    let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = ops::matmul(&a, &b)?;
    
    // Expected: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    println!("  ✓ matmul(2x2) = correct");
    
    // Test identity
    let a = Array::new(vec![3, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let identity = Array::new(vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let c = ops::matmul(&a, &identity)?;
    assert_eq!(c.data, a.data);
    println!("  ✓ matmul(A, I) = A");
    
    // Test larger matrix
    let n = 10;
    let a = Array::new(vec![n, n], vec![1.0; (n * n) as usize]);
    let b = Array::new(vec![n, n], vec![1.0; (n * n) as usize]);
    let c = ops::matmul(&a, &b)?;
    
    // Each element should be sum of 10 ones = 10.0
    assert!(c.data.iter().all(|&x| (x - 10.0).abs() < 1e-5));
    println!("  ✓ matmul(10x10 ones) = all 10s");
    
    Ok(())
}

fn test_sizes() -> Result<()> {
    let sizes = vec![4, 10, 100, 1000, 10_000];
    
    for size in sizes {
        let start = Instant::now();
        
        let a = Array::new(vec![size], vec![1.0; size as usize]);
        let b = Array::new(vec![size], vec![2.0; size as usize]);
        
        // Test elementwise
        let c = ops::add(&a, &b)?;
        assert_eq!(c.data.len(), size as usize);
        assert!(c.data.iter().all(|&x| (x - 3.0).abs() < 1e-5));
        
        // Test reduction
        let s = ops::sum(&a, None)?;
        assert!((s.data[0] - size as f32).abs() < 1e-3);
        
        let elapsed = start.elapsed();
        println!("  ✓ Size {}: {} µs (add + sum)", size, elapsed.as_micros());
    }
    
    // Test matmul sizes
    let mm_sizes = vec![2, 10, 50, 100];
    
    for n in mm_sizes {
        let start = Instant::now();
        
        let a = Array::new(vec![n, n], vec![1.0; (n * n) as usize]);
        let b = Array::new(vec![n, n], vec![1.0; (n * n) as usize]);
        let c = ops::matmul(&a, &b)?;
        
        assert_eq!(c.data.len(), (n * n) as usize);
        
        let elapsed = start.elapsed();
        let ops = 2.0 * (n as f64).powi(3);
        let gflops = ops / (elapsed.as_secs_f64() * 1e9);
        
        println!("  ✓ MatMul {}x{}: {} µs ({:.2} GFLOPS)", 
                 n, n, elapsed.as_micros(), gflops);
    }
    
    Ok(())
}

fn test_edge_cases() -> Result<()> {
    // Single element
    let a = Array::new(vec![1], vec![5.0]);
    let b = Array::new(vec![1], vec![3.0]);
    let c = ops::add(&a, &b)?;
    assert_eq!(c.data, vec![8.0]);
    println!("  ✓ Single element: [5] + [3] = [8]");
    
    // Zeros
    let a = Array::new(vec![10], vec![0.0; 10]);
    let s = ops::sum(&a, None)?;
    assert_eq!(s.data[0], 0.0);
    println!("  ✓ Sum of zeros = 0");
    
    // Negative numbers
    let a = Array::new(vec![4], vec![-1.0, -2.0, -3.0, -4.0]);
    let b = Array::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let c = ops::add(&a, &b)?;
    assert!(c.data.iter().all(|&x| x.abs() < 1e-5));
    println!("  ✓ Negative + positive = zeros");
    
    // Large values
    let a = Array::new(vec![4], vec![1e6, 2e6, 3e6, 4e6]);
    let s = ops::sum(&a, None)?;
    assert!((s.data[0] - 1e7).abs() / 1e7 < 1e-5);
    println!("  ✓ Large values: sum = 10M");
    
    // Small values
    let a = Array::new(vec![4], vec![1e-6, 2e-6, 3e-6, 4e-6]);
    let s = ops::sum(&a, None)?;
    assert!((s.data[0] - 1e-5).abs() / 1e-5 < 1e-3);
    println!("  ✓ Small values: sum = 1e-5");
    
    // 1x1 matmul
    let a = Array::new(vec![1, 1], vec![7.0]);
    let b = Array::new(vec![1, 1], vec![3.0]);
    let c = ops::matmul(&a, &b)?;
    assert_eq!(c.data, vec![21.0]);
    println!("  ✓ 1x1 matmul: [[7]] * [[3]] = [[21]]");
    
    // Non-square matmul (2x3 * 3x2)
    let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Array::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = ops::matmul(&a, &b)?;
    
    // Expected: 2x2 result
    assert_eq!(c.shape, vec![2, 2]);
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    println!("  ✓ Non-square matmul: 2x3 * 3x2 = 2x2");
    
    Ok(())
}
