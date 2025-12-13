use numrs::{Array, ops};
use anyhow::Result;

fn main() -> Result<()> {
    println!("🧪 Testing optimized tiny matmul...\n");
    
    // Test 1: Tiny matmul (< 128 elements)
    {
        let a = Array::new(vec![32, 8], (0..256).map(|i| i as f32).collect());
        let b = Array::new(vec![8, 4], (0..32).map(|i| (i % 7) as f32).collect());
        let c = ops::matmul(&a, &b)?;
        
        assert_eq!(c.shape(), &[32, 4]);
        
        // Verify first few elements manually
        // c[0,0] = sum(a[0,k] * b[k,0]) = 0*0 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5 + 6*6 + 7*0
        let expected_00 = 0.0*0.0 + 1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0 + 5.0*5.0 + 6.0*6.0 + 7.0*0.0;
        assert!((c.data[0] - expected_00).abs() < 1e-4, "c[0,0] mismatch: got {}, expected {}", c.data[0], expected_00);
        
        println!("✅ Test 1: Tiny matmul 32x8 @ 8x4 PASSED");
    }
    
    // Test 2: Very tiny (< 16 elements)
    {
        let a = Array::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Array::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = ops::matmul(&a, &b)?;
        
        assert_eq!(c.shape(), &[2, 2]);
        // c[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // c[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // c[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // c[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        assert_eq!(c.data, vec![22.0, 28.0, 49.0, 64.0]);
        println!("✅ Test 2: Very tiny matmul 2x3 @ 3x2 PASSED");
    }
    
    // Test 3: Boundary case (exactly 128 elements)
    {
        let a = Array::new(vec![16, 4], vec![1.0; 64]);
        let b = Array::new(vec![4, 2], vec![2.0; 8]);
        let c = ops::matmul(&a, &b)?;
        
        assert_eq!(c.shape(), &[16, 2]);
        // Each element should be 1*2 + 1*2 + 1*2 + 1*2 = 8
        assert!(c.data.iter().all(|&x| (x - 8.0).abs() < 1e-4));
        println!("✅ Test 3: Boundary case (128 elements) PASSED");
    }
    
    // Test 4: Edge case (1x1 @ 1x1)
    {
        let a = Array::new(vec![1, 1], vec![5.0]);
        let b = Array::new(vec![1, 1], vec![3.0]);
        let c = ops::matmul(&a, &b)?;
        
        assert_eq!(c.shape(), &[1, 1]);
        assert_eq!(c.data, vec![15.0]);
        println!("✅ Test 4: Edge case 1x1 @ 1x1 PASSED");
    }
    
    // Test 5: Column vector x row vector
    {
        let a = Array::new(vec![3, 1], vec![1.0, 2.0, 3.0]);
        let b = Array::new(vec![1, 4], vec![4.0, 5.0, 6.0, 7.0]);
        let c = ops::matmul(&a, &b)?;
        
        assert_eq!(c.shape(), &[3, 4]);
        // Outer product: c[i,j] = a[i] * b[j]
        assert_eq!(c.data[0], 4.0);  // 1*4
        assert_eq!(c.data[1], 5.0);  // 1*5
        assert_eq!(c.data[4], 8.0);  // 2*4
        assert_eq!(c.data[11], 21.0); // 3*7
        println!("✅ Test 5: Outer product 3x1 @ 1x4 PASSED");
    }
    
    // Test 6: Compare with reference implementation
    {
        let m = 8;
        let k = 5;
        let n = 6;
        
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.2).collect();
        
        let a = Array::new(vec![m, k], a_data.clone());
        let b = Array::new(vec![k, n], b_data.clone());
        let c = ops::matmul(&a, &b)?;
        
        // Reference implementation
        let mut c_ref = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    sum += a_data[i * k + kk] * b_data[kk * n + j];
                }
                c_ref[i * n + j] = sum;
            }
        }
        
        // Compare
        for i in 0..m*n {
            let diff = (c.data[i] - c_ref[i]).abs();
            assert!(diff < 1e-4, "Mismatch at index {}: got {}, expected {}", i, c.data[i], c_ref[i]);
        }
        println!("✅ Test 6: Reference comparison 8x5 @ 5x6 PASSED");
    }
    
    // Test 7: AVX2 path (process 8 columns)
    {
        let a = Array::new(vec![4, 10], vec![1.0; 40]);
        let b = Array::new(vec![10, 16], vec![0.5; 160]);
        let c = ops::matmul(&a, &b)?;
        
        assert_eq!(c.shape(), &[4, 16]);
        // Each element should be 1*0.5 * 10 = 5.0
        assert!(c.data.iter().all(|&x| (x - 5.0).abs() < 1e-4));
        println!("✅ Test 7: AVX2 path (16 columns) PASSED");
    }
    
    println!("\n🎉 All tiny matmul optimization tests PASSED!");
    Ok(())
}
