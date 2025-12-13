use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: elementwise_micro <size> <reps>");
        std::process::exit(1);
    }
    let size: usize = args[1].parse().expect("invalid size");
    let reps: usize = args[2].parse().expect("invalid reps");

    println!("elementwise_micro: size={} reps={}", size, reps);

    // Create two arrays filled with values
    let a: Vec<f32> = vec![1.0f32; size];
    let b: Vec<f32> = vec![2.0f32; size];

    use numrs::array::Array;

    // warmup using library dispatch (respects NUMRS_BACKEND env var)
    let a_arr = Array::new(vec![size], a.clone());
    let b_arr = Array::new(vec![size], b.clone());
    let _ = numrs::add(&a_arr, &b_arr).expect("warmup add");

    let mut times = Vec::new();
    for rep in 0..reps {
        let a_arr = Array::new(vec![size], a.clone());
        let b_arr = Array::new(vec![size], b.clone());
        let start = Instant::now();
        let _out = numrs::add(&a_arr, &b_arr).expect("add failed");
        let elapsed = start.elapsed();
        let secs = elapsed.as_secs_f64();
        println!("rep: {} -> {:.6} s", rep, secs);
        times.push(secs);
    }

    let mean: f64 = times.iter().sum::<f64>() / (times.len() as f64);
    println!("mean: {:.6} s", mean);
}
