use numrs::{Array, ops};

fn main() {
    println!("Testing integer dtype operations...\n");
    
    // Test i32 sqrt
    println!("1. Testing i32 sqrt:");
    let data_i32: Vec<i32> = vec![1, 4, 9, 16, 25];
    let a_i32 = Array::<i32>::new(vec![5], data_i32);
    println!("   Input: {:?}", a_i32.dtype);
    
    match ops::sqrt(&a_i32) {
        Ok(result) => {
            println!("   Result dtype: {:?}", result.dtype);
            println!("   Result data: {:?}\n", &result.data[..5]);
        },
        Err(e) => println!("   Error: {}\n", e),
    }
    
    // Test u8 exp
    println!("2. Testing u8 exp:");
    let data_u8: Vec<u8> = vec![0, 1, 2, 3, 4];
    let a_u8 = Array::<u8>::new(vec![5], data_u8);
    println!("   Input: {:?}", a_u8.dtype);
    
    match ops::exp(&a_u8) {
        Ok(result) => {
            println!("   Result dtype: {:?}", result.dtype);
            println!("   Result data: {:?}\n", &result.data[..5]);
        },
        Err(e) => println!("   Error: {}\n", e),
    }
    
    // Test i8 add
    println!("3. Testing i8 add:");
    let data_a: Vec<i8> = vec![1, 2, 3, 4, 5];
    let data_b: Vec<i8> = vec![10, 20, 30, 40, 50];
    let a_i8 = Array::<i8>::new(vec![5], data_a);
    let b_i8 = Array::<i8>::new(vec![5], data_b);
    println!("   Inputs: {:?} + {:?}", a_i8.dtype, b_i8.dtype);
    
    match ops::add(&a_i8, &b_i8) {
        Ok(result) => {
            println!("   Result dtype: {:?}", result.dtype);
            println!("   Result data: {:?}\n", &result.data[..5]);
        },
        Err(e) => println!("   Error: {}\n", e),
    }
    
    // Test cross-dtype: i32 + f32
    println!("4. Testing i32 + f32 (promotion):");
    let data_i32: Vec<i32> = vec![1, 2, 3, 4, 5];
    let data_f32: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let a_i32 = Array::<i32>::new(vec![5], data_i32);
    let b_f32 = Array::<f32>::new(vec![5], data_f32);
    println!("   Inputs: {:?} + {:?}", a_i32.dtype, b_f32.dtype);
    
    match ops::add(&a_i32, &b_f32) {
        Ok(result) => {
            println!("   Result dtype: {:?}", result.dtype);
            println!("   Result data: {:?}\n", &result.data[..5]);
        },
        Err(e) => println!("   Error: {}\n", e),
    }
}
