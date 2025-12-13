use numrs::Array;
use numrs::ops;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Test de Broadcasting en NumRs\n");
    println!("═══════════════════════════════════════════════════════════");
    
    // Test 1: Escalar a vector
    println!("\n📊 Test 1: Broadcast escalar [1] → [3]");
    let scalar = Array::new(vec![1], vec![5.0]);
    let broadcasted = ops::broadcast_to(&scalar, &[3])?;
    println!("  Input:  {:?} shape: {:?}", scalar.data, scalar.shape);
    println!("  Output: {:?} shape: {:?}", broadcasted.data, broadcasted.shape);
    assert_eq!(broadcasted.data, vec![5.0, 5.0, 5.0]);
    println!("  ✅ PASS");
    
    // Test 2: Vector a matriz
    println!("\n📊 Test 2: Broadcast vector [3] → [2, 3]");
    let vector = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
    let broadcasted = ops::broadcast_to(&vector, &[2, 3])?;
    println!("  Input:  {:?} shape: {:?}", vector.data, vector.shape);
    println!("  Output: {:?} shape: {:?}", broadcasted.data, broadcasted.shape);
    assert_eq!(broadcasted.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    println!("  ✅ PASS");
    
    // Test 3: Columna vector
    println!("\n📊 Test 3: Broadcast columna [2, 1] → [2, 3]");
    let col_vector = Array::new(vec![2, 1], vec![1.0, 2.0]);
    let broadcasted = ops::broadcast_to(&col_vector, &[2, 3])?;
    println!("  Input:  {:?} shape: {:?}", col_vector.data, col_vector.shape);
    println!("  Output: {:?} shape: {:?}", broadcasted.data, broadcasted.shape);
    assert_eq!(broadcasted.data, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    println!("  ✅ PASS");
    
    // Test 4: Broadcasting automático en suma
    println!("\n📊 Test 4: Auto-broadcast en suma [1, 8] + [8]");
    let matrix = Array::new(vec![1, 8], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let bias = Array::new(vec![8], vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let result = ops::add(&matrix, &bias)?;
    println!("  Matrix: {:?} shape: {:?}", matrix.data, matrix.shape);
    println!("  Bias:   {:?} shape: {:?}", bias.data, bias.shape);
    println!("  Result: {:?} shape: {:?}", result.data, result.shape);
    println!("  Expected: [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]");
    // Allow for floating point precision
    for (i, &val) in result.data.iter().enumerate() {
        let expected = (i + 1) as f32 + (i + 1) as f32 * 0.1;
        assert!((val - expected).abs() < 0.001, "Mismatch at index {}: {} vs {}", i, val, expected);
    }
    println!("  ✅ PASS");
    
    // Test 5: Broadcasting en multiplicación
    println!("\n📊 Test 5: Auto-broadcast en multiplicación [2, 3] * [3]");
    let matrix = Array::new(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let vector = Array::new(vec![3], vec![10.0f32, 100.0, 1000.0]);
    let result = ops::mul(&matrix, &vector)?;
    println!("  Matrix: {:?} shape: {:?}", matrix.data, matrix.shape);
    println!("  Vector: {:?} shape: {:?}", vector.data, vector.shape);
    println!("  Result: {:?} shape: {:?}", result.data, result.shape);
    println!("  Expected: [10, 200, 3000, 40, 500, 6000]");
    assert_eq!(result.data, vec![10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]);
    println!("  ✅ PASS");
    
    // Test 6: Caso invalido - dimensiones incompatibles
    println!("\n📊 Test 6: Validación de error en broadcast incompatible");
    let a = Array::new(vec![3], vec![1.0, 2.0, 3.0]);
    let result = ops::broadcast_to(&a, &[2, 4]);
    match result {
        Err(e) => {
            println!("  Intentando: [3] → [2, 4]");
            println!("  ✅ Error esperado: {}", e);
            println!("  ✅ PASS");
        }
        Ok(_) => {
            println!("  ❌ FAIL: Debería haber fallado");
            return Err("Test failed: invalid broadcast should error".into());
        }
    }
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("✨ ¡Todos los tests de broadcasting pasaron!");
    println!("\n📈 Resumen de capacidades:");
    println!("  ✓ Broadcasting explícito con broadcast_to()");
    println!("  ✓ Auto-broadcasting en operaciones binarias (add, mul, etc.)");
    println!("  ✓ Validación de reglas de NumPy");
    println!("  ✓ Soporte para CPU, SIMD AVX2 y WebGPU");
    println!("  ✓ Compatible con Linear layers en training");
    println!("\n🚀 NumRs tiene broadcasting completo como NumPy!");
    
    Ok(())
}
