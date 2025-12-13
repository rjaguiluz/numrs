/// Compare Rust native vs C API performance
use std::time::Instant;
use numrs::{Array, ArrayView, ElementwiseKind};

fn main() {
    let size = 10_000;
    let iterations = 1000;
    
    println!("🔬 Performance Analysis (size: {}, iterations: {})\n", size, iterations);
    
    // Setup data
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    let mut output: Vec<f32> = vec![0.0; size];
    
    // ==== RUST NATIVE API (con promotion) ====
    println!("🦀 Rust Native API (numrs::ops::add - con promotion wrapper):");
    let a_arr = Array::new(vec![size], data_a.clone());
    let b_arr = Array::new(vec![size], data_b.clone());
    
    // Warmup
    for _ in 0..10 {
        let _ = numrs::ops::add(&a_arr, &b_arr).unwrap();
    }
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = numrs::ops::add(&a_arr, &b_arr).unwrap();
    }
    let elapsed = start.elapsed();
    let avg_rust = elapsed.as_micros() as f64 / iterations as f64;
    println!("  Average: {:.2} μs", avg_rust);
    
    // ==== DISPATCH DIRECTO (sin promotion) ====
    println!("\n⚡ Direct Dispatch (sin promotion wrapper):");
    let table = numrs::backend::dispatch::get_dispatch_table();
    
    // Warmup
    for _ in 0..10 {
        let _ = (table.elementwise)(&a_arr, &b_arr, ElementwiseKind::Add).unwrap();
    }
    
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = (table.elementwise)(&a_arr, &b_arr, ElementwiseKind::Add).unwrap();
    }
    let elapsed = start.elapsed();
    let avg_direct = elapsed.as_micros() as f64 / iterations as f64;
    println!("  Average: {:.2} μs", avg_direct);
    
    let elapsed = start.elapsed();
    let avg_direct = elapsed.as_micros() as f64 / iterations as f64;
    println!("  Average: {:.2} μs", avg_direct);
    
    // ==== C API (ops_inplace con to_vec) ====
    println!("\n🔌 C API (ops_inplace::elementwise_f32 - con to_vec()):");
    
    // Warmup
    for _ in 0..10 {
        numrs::ops_inplace::elementwise_f32(
            &data_a, 
            &data_b, 
            &mut output, 
            numrs::ElementwiseKind::Add
        ).unwrap();
    }
    
    let start = Instant::now();
    for _ in 0..iterations {
        numrs::ops_inplace::elementwise_f32(
            &data_a, 
            &data_b, 
            &mut output, 
            numrs::ElementwiseKind::Add
        ).unwrap();
    }
    let elapsed = start.elapsed();
    let avg_c_api = elapsed.as_micros() as f64 / iterations as f64;
    println!("  Average: {:.2} μs", avg_c_api);
    
    // ==== ARRAYVIEW (zero-copy real con pre-carga) ====
    println!("\n🚀 ArrayView desde Rust (zero-copy con pre-carga UNA VEZ):");
    
    // Crear ArrayViews UNA VEZ (aquí se hace el to_vec())
    let view_a = ArrayView::from_slice_f32(&data_a);
    let view_b = ArrayView::from_slice_f32(&data_b);
    
    // Warmup
    for _ in 0..10 {
        numrs::ops_inplace::elementwise_view(
            &view_a,
            &view_b,
            output.as_mut_ptr() as *mut std::ffi::c_void,
            size,
            ElementwiseKind::Add
        ).unwrap();
    }
    
    let start = Instant::now();
    for _ in 0..iterations {
        numrs::ops_inplace::elementwise_view(
            &view_a,
            &view_b,
            output.as_mut_ptr() as *mut std::ffi::c_void,
            size,
            ElementwiseKind::Add
        ).unwrap();
    }
    let elapsed = start.elapsed();
    let avg_view_rust = elapsed.as_micros() as f64 / iterations as f64;
    println!("  Average: {:.2} μs", avg_view_rust);
    
    // ==== ARRAYVIEW desde C API ====
    println!("\n🎯 ArrayView desde C API (simula FFI real - tipo agnóstico):");
    
    // Simular lo que haría el caller en C:
    // 1. Crear views UNA VEZ (esto copia datos a Rust)
    let view_a_c = unsafe { numrs_c::numrs_array_view_new_f32(data_a.as_ptr(), size) };
    let view_b_c = unsafe { numrs_c::numrs_array_view_new_f32(data_b.as_ptr(), size) };
    
    // 2. Warmup
    for _ in 0..10 {
        unsafe {
            numrs_c::numrs_add_view(
                view_a_c, 
                view_b_c, 
                output.as_mut_ptr() as *mut std::ffi::c_void, 
                size
            );
        }
    }
    
    // 3. Benchmark (no hay to_vec() aquí!)
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            numrs_c::numrs_add_view(
                view_a_c, 
                view_b_c, 
                output.as_mut_ptr() as *mut std::ffi::c_void, 
                size
            );
        }
    }
    let elapsed = start.elapsed();
    let avg_view_c = elapsed.as_micros() as f64 / iterations as f64;
    println!("  Average: {:.2} μs", avg_view_c);
    
    // 4. Cleanup
    unsafe {
        numrs_c::numrs_array_view_destroy(view_a_c as *mut _);
        numrs_c::numrs_array_view_destroy(view_b_c as *mut _);
    }
    
    // ==== COMPARISON ====
    println!("\n📊 Comparison:");
    println!("  Rust native (con promotion):    {:.2} μs", avg_rust);
    println!("  Direct dispatch (sin promotion): {:.2} μs", avg_direct);
    println!("  C API (ops_inplace + to_vec):    {:.2} μs", avg_c_api);
    println!("  ArrayView desde Rust:            {:.2} μs", avg_view_rust);
    println!("  ArrayView desde C API (FFI):     {:.2} μs", avg_view_c);
    
    let overhead_promotion = ((avg_rust / avg_direct) - 1.0) * 100.0;
    let overhead_c_api = ((avg_c_api / avg_direct) - 1.0) * 100.0;
    let overhead_view_rust = ((avg_view_rust / avg_direct) - 1.0) * 100.0;
    let overhead_view_c = ((avg_view_c / avg_direct) - 1.0) * 100.0;
    
    println!("\n🔍 Overhead Analysis:");
    println!("  Promotion wrapper:             {:.1}%", overhead_promotion);
    println!("  C API (to_vec cada vez):       {:.1}%", overhead_c_api);
    println!("  ArrayView Rust (sin re-copy):  {:.1}%", overhead_view_rust);
    println!("  ArrayView C API (FFI):         {:.1}%", overhead_view_c);
    
    if overhead_view_c < 10.0 {
        println!("\n✅ SUCCESS: ArrayView C API overhead < 10% - ¡Zero-copy FFI real!");
    } else if overhead_view_c < 50.0 {
        println!("\n✓ ArrayView C API overhead es aceptable (<50%)");
    } else {
        println!("\n⚠️  ArrayView todavía tiene overhead alto - investigar más");
    }
}
