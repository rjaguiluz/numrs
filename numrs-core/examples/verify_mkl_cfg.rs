// Test to verify MKL is actually linked
use numrs::array::Array;
use numrs::backend;

fn main() {
    println!("=== Configuration Checks ===");
    println!("cfg!(numrs_has_blas) = {}", cfg!(numrs_has_blas));
    println!("cfg!(numrs_has_mkl) = {}", cfg!(numrs_has_mkl));
    println!("cfg!(numrs_kernel_matmul_blas) = {}", cfg!(numrs_kernel_matmul_blas));
    
    println!("\n=== Validation ===");
    let validation = backend::validate_backends();
    println!("BLAS available: {}", validation.blas_available);
    println!("BLAS validated: {}", validation.blas_validated);
    
    println!("\n=== Test BLAS Direct Call ===");
    let a = Array::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    
    #[cfg(numrs_has_blas)]
    {
        println!("Calling matmul_blas...");
        let result = numrs::backend::blas::matmul_blas(&a, &b);
        println!("Result: {:?}", result.data);
    }
    
    #[cfg(not(numrs_has_blas))]
    {
        println!("ERROR: numrs_has_blas cfg is NOT set!");
    }
}
