//! Pseudo-random number generation for internal use (Dropout, etc.)
//!
//! Uses a simple Xorshift128+ algorithm for speed.
//! Not cryptographically secure, but sufficient for ML randomness.

use std::cell::RefCell;


thread_local! {
    static RNG: RefCell<Xorshift128Plus> = RefCell::new(Xorshift128Plus::new_from_time());
}

struct Xorshift128Plus {
    state: [u64; 2],
}

impl Xorshift128Plus {
    fn new(seed: u64) -> Self {
        let mut rng = Self { state: [0, 0] };
        rng.seed(seed);
        rng
    }

    fn new_from_time() -> Self {
        // Simple seeding from address/stack pointer mix if strict time not available,
        // but for now let's use a fixed seed + some variation if possible.
        // In real impl, would use std::time or getrandom via syscall if allowed.
        // For no-std compat, simpler is better.
        // Here we just use a default seed to ensure reproducibility by default,
        // or mix with some pointer values.
        let seed = 123456789; 
        Self::new(seed)
    }

    fn seed(&mut self, seed: u64) {
        // SplitMix64 initialization
        let mut z = (seed).wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        let s0 = z ^ (z >> 31);

        let mut z = s0.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        let s1 = z ^ (z >> 31);
        
        self.state = [s0, s1];
    }

    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.state[0];
        let s0 = self.state[1];
        self.state[0] = s0;
        s1 ^= s1 << 23; // a
        self.state[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26); // b, c
        self.state[1].wrapping_add(s0)
    }

    fn next_f32(&mut self) -> f32 {
        // Generate uniform [0, 1) float
        // Method: generate u32, divide by 2^32
        let v = (self.next_u64() >> 32) as u32;
        (v as f32) / (u32::MAX as f32)
    }
}

/// Fills a buffer with random values from Uniform[0, 1)
pub fn rand_uniform(data: &mut [f32]) {
    RNG.with(|rng| {
        let mut r = rng.borrow_mut();
        for val in data.iter_mut() {
            *val = r.next_f32();
        }
    })
}

/// Generates a Bernoulli mask (1.0 with prob p, 0.0 with prob 1-p)
/// In dropout terms: 'p' usually means "probability of zeroing out" (Torch convention) 
/// or "probability of keeping" (TensorFlow convention).
/// PyTorch: p = prob of dropout (zeroing). 
/// Here we use p = prob of Dropout (Zeroing).
/// 
/// output[i] = 1 if keep, 0 if drop.
/// keep_prob = 1 - p
pub fn bernoulli_mask(data: &mut [f32], p: f32) {
    let threshold = 1.0 - p;
    RNG.with(|rng| {
        let mut r = rng.borrow_mut();
        for val in data.iter_mut() {
            let rnd = r.next_f32();
            *val = if rnd < threshold { 1.0 } else { 0.0 };
        }
    })
}

/// Seeds the thread-local RNG
pub fn seed(seed: u64) {
    RNG.with(|rng| {
        rng.borrow_mut().seed(seed);
    })
}
