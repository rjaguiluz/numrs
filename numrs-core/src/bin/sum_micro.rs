use std::time::Instant;
use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: sum_micro <size> <reps>");
        std::process::exit(1);
    }
    let size: usize = args[1].parse().expect("invalid size");
    let reps: usize = args[2].parse().expect("invalid reps");

    println!("sum_micro: size={} reps={}", size, reps);

    // create data filled with 1.0
    let data = vec![1.0f32; size];
    let arr = numrs::array::Array::new(vec![size], data.clone());

    // warmup
    let _ = numrs::sum(&arr, None)?;

    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        let _res = numrs::sum(&arr, None)?;
        let dt = t0.elapsed();
        times.push(dt.as_secs_f64());
    }

    for (i, t) in times.iter().enumerate() {
        println!("rep: {} -> {:.6} s", i, t);
    }
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    println!("mean: {:.6} s", mean);

    Ok(())
}
