use numrs::{Array, ops};
use anyhow::Result;

fn main() -> Result<()> {
    println!("🧪 Testing optimized transpose...\n");
    
    // Test 1: Basic 2D transpose
    {
        let a = Array::new(vec![2, 3], vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]);
        let t = ops::transpose(&a, None)?;
        
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.data, vec![
            1.0, 4.0,
            2.0, 5.0,
            3.0, 6.0,
        ]);
        println!("✅ Test 1: Basic 2D transpose PASSED");
    }
    
    // Test 2: Large matrix transpose
    {
        let rows = 100;
        let cols = 50;
        let data: Vec<f32> = (0..(rows*cols)).map(|i| i as f32).collect();
        let a = Array::new(vec![rows, cols], data);
        let t = ops::transpose(&a, None)?;
        
        assert_eq!(t.shape(), &[cols, rows]);
        // Verify a few values
        assert_eq!(t.data[0], 0.0);  // t[0,0] = a[0,0]
        assert_eq!(t.data[1], 50.0); // t[0,1] = a[1,0]
        assert_eq!(t.data[rows], 1.0); // t[1,0] = a[0,1]
        println!("✅ Test 2: Large matrix transpose PASSED");
    }
    
    // Test 3: Square matrix
    {
        let n = 64;
        let data: Vec<f32> = (0..(n*n)).map(|i| (i % 13) as f32).collect();
        let a = Array::new(vec![n, n], data);
        let t = ops::transpose(&a, None)?;
        
        assert_eq!(t.shape(), &[n, n]);
        // Verify transpose property: t[i,j] = a[j,i]
        for i in 0..n {
            for j in 0..n {
                let a_val = a.data[i * n + j];
                let t_val = t.data[j * n + i];
                assert_eq!(a_val, t_val, "Mismatch at ({},{})", i, j);
            }
        }
        println!("✅ Test 3: Square matrix 64x64 transpose PASSED");
    }
    
    // Test 4: Edge cases
    {
        // 1x1
        let a = Array::new(vec![1, 1], vec![42.0]);
        let t = ops::transpose(&a, None)?;
        assert_eq!(t.data, vec![42.0]);
        
        // 1xN
        let a = Array::new(vec![1, 5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let t = ops::transpose(&a, None)?;
        assert_eq!(t.shape(), &[5, 1]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Nx1
        let a = Array::new(vec![5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let t = ops::transpose(&a, None)?;
        assert_eq!(t.shape(), &[1, 5]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        println!("✅ Test 4: Edge cases (1x1, 1xN, Nx1) PASSED");
    }
    
    // Test 5: Block boundaries (test BLOCK_SIZE = 32)
    {
        // Test matrix that spans multiple blocks
        let rows = 65; // > 2*BLOCK_SIZE
        let cols = 65;
        let data: Vec<f32> = (0..(rows*cols)).map(|i| i as f32).collect();
        let a = Array::new(vec![rows, cols], data);
        let t = ops::transpose(&a, None)?;
        
        assert_eq!(t.shape(), &[cols, rows]);
        // Verify corners and boundaries
        assert_eq!(t.data[0], 0.0);  // [0,0]
        assert_eq!(t.data[31 * rows + 31], 31.0 * cols as f32 + 31.0); // Block boundary
        assert_eq!(t.data[32 * rows + 32], 32.0 * cols as f32 + 32.0); // Next block
        println!("✅ Test 5: Block boundary cases PASSED");
    }
    
    println!("\n🎉 All transpose optimization tests PASSED!");
    Ok(())
}
