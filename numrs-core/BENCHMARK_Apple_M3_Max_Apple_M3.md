# NumRs Performance Benchmarks

## System Configuration

- **CPU:** Apple M3 Max
- **GPU:** Apple M3 Max
- **RAM:** 36 GB
- **OS:** Darwin 15.1.1 (24.1.0)

## Benchmark Configuration

- **Generated:** 2025-12-16 20:22:57
- **Build:** Release mode with optimizations + MKL
- **Iterations:** 50 (after 10 warmup)

## Performance by Operation

| Operation | Backend | Size | DType1 | DType2 | DType_Result | Mean Time | Std Dev | Throughput |
|-----------|---------|------|--------|--------|--------------|-----------|---------|------------|
| add | Scalar | 10000 | f32 | f32 | f32 | 140.07 μs | 60.50 μs | 71.39 Mops/s |
| add | Scalar | 100000 | f32 | f32 | f32 | 151.94 μs | 44.24 μs | 658.16 Mops/s |
| add | Scalar | 10000 | i32 | f32 | f32 | 90.19 μs | 22.75 μs | 110.88 Mops/s |
| add | SIMD | 10000 | f32 | f32 | f32 | 94.57 μs | 42.78 μs | 105.74 Mops/s |
| add | SIMD | 100000 | f32 | f32 | f32 | 124.59 μs | 41.06 μs | 802.62 Mops/s |
| add | SIMD | 10000 | i32 | f32 | f32 | 111.28 μs | 57.99 μs | 89.86 Mops/s |
| add | Metal | 10000 | f32 | f32 | f32 | 365.26 μs | 218.13 μs | 27.38 Mops/s |
| add | Metal | 100000 | f32 | f32 | f32 | 210.24 μs | 46.37 μs | 475.65 Mops/s |
| add | Metal | 10000 | i32 | f32 | f32 | 198.50 μs | 47.50 μs | 50.38 Mops/s |
| add | WebGPU | 10000 | f32 | f32 | f32 | 1.92 ms | 128.90 μs | 5.21 Mops/s |
| add | WebGPU | 100000 | f32 | f32 | f32 | 2.38 ms | 307.95 μs | 42.10 Mops/s |
| add | WebGPU | 10000 | i32 | f32 | f32 | 2.24 ms | 552.07 μs | 4.46 Mops/s |
| add_u8_f32 | Scalar | 10000 | u8 | f32 | f32 | 89.88 μs | 38.77 μs | 111.26 Mops/s |
| add_u8_f32 | SIMD | 10000 | u8 | f32 | f32 | 117.61 μs | 54.73 μs | 85.02 Mops/s |
| add_u8_f32 | Metal | 10000 | u8 | f32 | f32 | 177.96 μs | 31.33 μs | 56.19 Mops/s |
| add_u8_f32 | WebGPU | 10000 | u8 | f32 | f32 | 2.20 ms | 169.73 μs | 4.55 Mops/s |
| cos | Scalar | 10000 | f32 | N/A | f32 | 86.82 μs | 28.73 μs | 115.18 Mops/s |
| cos | Scalar | 100000 | f32 | N/A | f32 | 141.39 μs | 29.93 μs | 707.27 Mops/s |
| cos | SIMD | 10000 | f32 | N/A | f32 | 107.85 μs | 53.98 μs | 92.72 Mops/s |
| cos | SIMD | 100000 | f32 | N/A | f32 | 167.74 μs | 62.17 μs | 596.18 Mops/s |
| cos | Metal | 10000 | f32 | N/A | f32 | 179.87 μs | 32.61 μs | 55.60 Mops/s |
| cos | Metal | 100000 | f32 | N/A | f32 | 192.55 μs | 34.30 μs | 519.34 Mops/s |
| cos | WebGPU | 10000 | f32 | N/A | f32 | 2.11 ms | 116.93 μs | 4.73 Mops/s |
| cos | WebGPU | 100000 | f32 | N/A | f32 | 2.36 ms | 222.34 μs | 42.38 Mops/s |
| div | Scalar | 10000 | f32 | f32 | f32 | 99.01 μs | 37.85 μs | 101.00 Mops/s |
| div | Scalar | 100000 | f32 | f32 | f32 | 129.14 μs | 38.76 μs | 774.34 Mops/s |
| div | SIMD | 10000 | f32 | f32 | f32 | 81.94 μs | 17.47 μs | 122.04 Mops/s |
| div | SIMD | 100000 | f32 | f32 | f32 | 113.62 μs | 29.33 μs | 880.11 Mops/s |
| div | Metal | 10000 | f32 | f32 | f32 | 180.15 μs | 48.69 μs | 55.51 Mops/s |
| div | Metal | 100000 | f32 | f32 | f32 | 214.50 μs | 36.16 μs | 466.20 Mops/s |
| div | WebGPU | 10000 | f32 | f32 | f32 | 2.02 ms | 66.07 μs | 4.94 Mops/s |
| div | WebGPU | 100000 | f32 | f32 | f32 | 2.29 ms | 59.55 μs | 43.68 Mops/s |
| dot | Scalar | 1000 | f32 | f32 | f32 | 141.60 ns | 39.95 ns | 14.12 Gops/s |
| dot | Scalar | 10000 | f32 | f32 | f32 | 486.64 ns | 54.25 ns | 41.10 Gops/s |
| dot | Scalar | 100000 | f32 | f32 | f32 | 1.55 μs | 124.28 ns | 128.68 Gops/s |
| dot | SIMD | 1000 | f32 | f32 | f32 | 118.28 ns | 15.40 ns | 16.91 Gops/s |
| dot | SIMD | 10000 | f32 | f32 | f32 | 431.72 ns | 24.74 ns | 46.33 Gops/s |
| dot | SIMD | 100000 | f32 | f32 | f32 | 1.51 μs | 48.56 ns | 132.82 Gops/s |
| dot | BLAS | 1000 | f32 | f32 | f32 | 141.64 ns | 47.79 ns | 14.12 Gops/s |
| dot | BLAS | 10000 | f32 | f32 | f32 | 423.36 ns | 22.57 ns | 47.24 Gops/s |
| dot | BLAS | 100000 | f32 | f32 | f32 | 1.54 μs | 63.45 ns | 130.22 Gops/s |
| dot | Metal | 1000 | f32 | f32 | f32 | 141.66 ns | 20.41 ns | 14.12 Gops/s |
| dot | Metal | 10000 | f32 | f32 | f32 | 450.82 ns | 37.89 ns | 44.36 Gops/s |
| dot | Metal | 100000 | f32 | f32 | f32 | 1.57 μs | 86.53 ns | 127.19 Gops/s |
| dot | WebGPU | 1000 | f32 | f32 | f32 | 148.86 ns | 21.94 ns | 13.44 Gops/s |
| dot | WebGPU | 10000 | f32 | f32 | f32 | 548.34 ns | 15.24 ns | 36.47 Gops/s |
| dot | WebGPU | 100000 | f32 | f32 | f32 | 1.94 μs | 42.06 ns | 102.83 Gops/s |
| exp | Scalar | 10000 | f32 | N/A | f32 | 92.44 μs | 29.46 μs | 108.18 Mops/s |
| exp | Scalar | 100000 | f32 | N/A | f32 | 132.08 μs | 30.10 μs | 757.13 Mops/s |
| exp | SIMD | 10000 | f32 | N/A | f32 | 114.17 μs | 59.56 μs | 87.59 Mops/s |
| exp | SIMD | 100000 | f32 | N/A | f32 | 172.51 μs | 74.02 μs | 579.67 Mops/s |
| exp | Metal | 10000 | f32 | N/A | f32 | 204.37 μs | 78.69 μs | 48.93 Mops/s |
| exp | Metal | 100000 | f32 | N/A | f32 | 211.74 μs | 92.44 μs | 472.27 Mops/s |
| exp | WebGPU | 10000 | f32 | N/A | f32 | 2.07 ms | 128.37 μs | 4.84 Mops/s |
| exp | WebGPU | 100000 | f32 | N/A | f32 | 2.23 ms | 130.70 μs | 44.81 Mops/s |
| exp_i8 | Scalar | 10000 | i8 | N/A | i8 | 89.72 μs | 15.07 μs | 111.46 Mops/s |
| exp_i8 | SIMD | 10000 | i8 | N/A | i8 | 163.27 μs | 82.61 μs | 61.25 Mops/s |
| exp_i8 | Metal | 10000 | i8 | N/A | i8 | 185.58 μs | 38.50 μs | 53.89 Mops/s |
| exp_i8 | WebGPU | 10000 | i8 | N/A | i8 | 2.25 ms | 144.72 μs | 4.44 Mops/s |
| matmul | Scalar | 128x128 | f32 | f32 | f32 | 536.58 μs | 25.62 μs | 7.82 Gops/s |
| matmul | Scalar | 256x256 | f32 | f32 | f32 | 2.15 ms | 54.38 μs | 15.61 Gops/s |
| matmul | Scalar | 512x512 | f32 | f32 | f32 | 10.29 ms | 0.00 ns | 26.08 Gops/s |
| matmul | Scalar | 1024x1024 | f32 | f32 | f32 | 51.61 ms | 0.00 ns | 41.61 Gops/s |
| matmul | Scalar | 2048x2048 | f32 | f32 | f32 | 383.75 ms | 0.00 ns | 44.77 Gops/s |
| matmul | SIMD | 128x128 | f32 | f32 | f32 | 515.17 μs | 2.57 μs | 8.14 Gops/s |
| matmul | SIMD | 256x256 | f32 | f32 | f32 | 2.15 ms | 70.82 μs | 15.60 Gops/s |
| matmul | SIMD | 512x512 | f32 | f32 | f32 | 8.80 ms | 0.00 ns | 30.50 Gops/s |
| matmul | SIMD | 1024x1024 | f32 | f32 | f32 | 51.66 ms | 0.00 ns | 41.57 Gops/s |
| matmul | SIMD | 2048x2048 | f32 | f32 | f32 | 371.90 ms | 0.00 ns | 46.20 Gops/s |
| matmul | BLAS | 128x128 | f32 | f32 | f32 | 4.23 μs | 81.32 ns | 990.76 Gops/s |
| matmul | BLAS | 256x256 | f32 | f32 | f32 | 25.76 μs | 1.52 μs | 1.30 Tops/s |
| matmul | BLAS | 512x512 | f32 | f32 | f32 | 281.04 μs | 0.00 ns | 955.14 Gops/s |
| matmul | BLAS | 1024x1024 | f32 | f32 | f32 | 910.79 μs | 0.00 ns | 2.36 Tops/s |
| matmul | BLAS | 2048x2048 | f32 | f32 | f32 | 7.09 ms | 0.00 ns | 2.42 Tops/s |
| matmul | Metal | 128x128 | f32 | f32 | f32 | 225.05 μs | 2.09 μs | 18.64 Gops/s |
| matmul | Metal | 256x256 | f32 | f32 | f32 | 483.07 μs | 109.18 μs | 69.46 Gops/s |
| matmul | Metal | 512x512 | f32 | f32 | f32 | 1.66 ms | 0.00 ns | 161.86 Gops/s |
| matmul | Metal | 1024x1024 | f32 | f32 | f32 | 4.22 ms | 0.00 ns | 509.16 Gops/s |
| matmul | Metal | 2048x2048 | f32 | f32 | f32 | 20.27 ms | 0.00 ns | 847.42 Gops/s |
| matmul | WebGPU | 128x128 | f32 | f32 | f32 | 1.38 ms | 25.74 μs | 3.03 Gops/s |
| matmul | WebGPU | 256x256 | f32 | f32 | f32 | 1.46 ms | 11.10 μs | 22.91 Gops/s |
| matmul | WebGPU | 512x512 | f32 | f32 | f32 | 1.82 ms | 0.00 ns | 147.39 Gops/s |
| matmul | WebGPU | 1024x1024 | f32 | f32 | f32 | 6.73 ms | 0.00 ns | 319.20 Gops/s |
| matmul | WebGPU | 2048x2048 | f32 | f32 | f32 | 23.74 ms | 0.00 ns | 723.67 Gops/s |
| matmul_i32 | Scalar | 128x128 | i32 | i32 | i32 | 541.52 μs | 11.76 μs | 7.75 Gops/s |
| matmul_i32 | SIMD | 128x128 | i32 | i32 | i32 | 583.46 μs | 96.17 μs | 7.19 Gops/s |
| matmul_i32 | BLAS | 128x128 | i32 | i32 | i32 | 8.62 μs | 4.00 μs | 486.77 Gops/s |
| matmul_i32 | Metal | 128x128 | i32 | i32 | i32 | 266.31 μs | 94.32 μs | 15.75 Gops/s |
| matmul_i32 | WebGPU | 128x128 | i32 | i32 | i32 | 1.43 ms | 74.40 μs | 2.93 Gops/s |
| max | Scalar | 10000 | f32 | N/A | f32 | 77.72 μs | 19.28 μs | 128.66 Mops/s |
| max | Scalar | 100000 | f32 | N/A | f32 | 124.32 μs | 31.91 μs | 804.35 Mops/s |
| max | Scalar | 1000000 | f32 | N/A | f32 | 174.47 μs | 49.76 μs | 5.73 Gops/s |
| max | SIMD | 10000 | f32 | N/A | f32 | 77.38 μs | 17.35 μs | 129.24 Mops/s |
| max | SIMD | 100000 | f32 | N/A | f32 | 133.13 μs | 44.53 μs | 751.12 Mops/s |
| max | SIMD | 1000000 | f32 | N/A | f32 | 189.28 μs | 49.85 μs | 5.28 Gops/s |
| max | BLAS | 10000 | f32 | N/A | f32 | 85.27 μs | 35.81 μs | 117.27 Mops/s |
| max | BLAS | 100000 | f32 | N/A | f32 | 114.71 μs | 35.81 μs | 871.74 Mops/s |
| max | BLAS | 1000000 | f32 | N/A | f32 | 201.48 μs | 50.31 μs | 4.96 Gops/s |
| max_axis1_i32 | Scalar | 500x500 | i32 | N/A | i32 | 132.37 μs | 47.14 μs | 1.89 Gops/s |
| mean | Scalar | 10000 | f32 | N/A | f32 | 74.12 μs | 17.31 μs | 134.92 Mops/s |
| mean | Scalar | 100000 | f32 | N/A | f32 | 109.69 μs | 33.93 μs | 911.65 Mops/s |
| mean | Scalar | 1000000 | f32 | N/A | f32 | 192.03 μs | 28.60 μs | 5.21 Gops/s |
| mean | SIMD | 10000 | f32 | N/A | f32 | 83.58 μs | 28.36 μs | 119.65 Mops/s |
| mean | SIMD | 100000 | f32 | N/A | f32 | 123.00 μs | 39.54 μs | 813.00 Mops/s |
| mean | SIMD | 1000000 | f32 | N/A | f32 | 206.98 μs | 59.21 μs | 4.83 Gops/s |
| mean | BLAS | 10000 | f32 | N/A | f32 | 78.19 μs | 26.12 μs | 127.90 Mops/s |
| mean | BLAS | 100000 | f32 | N/A | f32 | 120.55 μs | 46.60 μs | 829.53 Mops/s |
| mean | BLAS | 1000000 | f32 | N/A | f32 | 206.47 μs | 43.67 μs | 4.84 Gops/s |
| mean_axis1 | Scalar | 500x500 | f32 | N/A | f32 | 128.30 μs | 14.96 μs | 1.95 Gops/s |
| mean_axis1 | Scalar | 1000x1000 | f32 | N/A | f32 | 208.93 μs | 36.05 μs | 4.79 Gops/s |
| mean_axis1_u8 | Scalar | 500x500 | u8 | N/A | u8 | 152.11 μs | 25.20 μs | 1.64 Gops/s |
| mul | Scalar | 10000 | f32 | f32 | f32 | 112.30 μs | 59.59 μs | 89.05 Mops/s |
| mul | Scalar | 100000 | f32 | f32 | f32 | 114.35 μs | 20.17 μs | 874.47 Mops/s |
| mul | SIMD | 10000 | f32 | f32 | f32 | 89.92 μs | 38.35 μs | 111.20 Mops/s |
| mul | SIMD | 100000 | f32 | f32 | f32 | 122.26 μs | 35.85 μs | 817.90 Mops/s |
| mul | Metal | 10000 | f32 | f32 | f32 | 319.13 μs | 279.06 μs | 31.34 Mops/s |
| mul | Metal | 100000 | f32 | f32 | f32 | 221.92 μs | 36.07 μs | 450.61 Mops/s |
| mul | WebGPU | 10000 | f32 | f32 | f32 | 2.09 ms | 147.62 μs | 4.78 Mops/s |
| mul | WebGPU | 100000 | f32 | f32 | f32 | 2.26 ms | 135.96 μs | 44.29 Mops/s |
| mul_i8_i32 | Scalar | 10000 | i8 | i32 | i32 | 10.80 μs | 2.05 μs | 926.21 Mops/s |
| mul_i8_i32 | SIMD | 10000 | i8 | i32 | i32 | 12.94 μs | 4.41 μs | 772.69 Mops/s |
| mul_i8_i32 | Metal | 10000 | i8 | i32 | i32 | 7.06 μs | 404.30 ns | 1.42 Gops/s |
| mul_i8_i32 | WebGPU | 10000 | i8 | i32 | i32 | 118.76 μs | 207.82 μs | 84.20 Mops/s |
| sin | Scalar | 10000 | f32 | N/A | f32 | 76.17 μs | 13.68 μs | 131.28 Mops/s |
| sin | Scalar | 100000 | f32 | N/A | f32 | 140.70 μs | 33.35 μs | 710.76 Mops/s |
| sin | SIMD | 10000 | f32 | N/A | f32 | 119.35 μs | 120.99 μs | 83.79 Mops/s |
| sin | SIMD | 100000 | f32 | N/A | f32 | 186.93 μs | 73.18 μs | 534.97 Mops/s |
| sin | Metal | 10000 | f32 | N/A | f32 | 180.92 μs | 28.85 μs | 55.27 Mops/s |
| sin | Metal | 100000 | f32 | N/A | f32 | 207.29 μs | 81.62 μs | 482.42 Mops/s |
| sin | WebGPU | 10000 | f32 | N/A | f32 | 2.16 ms | 186.59 μs | 4.63 Mops/s |
| sin | WebGPU | 100000 | f32 | N/A | f32 | 2.38 ms | 182.84 μs | 42.00 Mops/s |
| sqrt | Scalar | 10000 | f32 | N/A | f32 | 63.43 μs | 34.26 μs | 157.65 Mops/s |
| sqrt | Scalar | 100000 | f32 | N/A | f32 | 128.95 μs | 57.24 μs | 775.50 Mops/s |
| sqrt | SIMD | 10000 | f32 | N/A | f32 | 108.54 μs | 100.43 μs | 92.13 Mops/s |
| sqrt | SIMD | 100000 | f32 | N/A | f32 | 202.02 μs | 141.85 μs | 495.01 Mops/s |
| sqrt | Metal | 10000 | f32 | N/A | f32 | 168.29 μs | 39.82 μs | 59.42 Mops/s |
| sqrt | Metal | 100000 | f32 | N/A | f32 | 216.47 μs | 51.71 μs | 461.95 Mops/s |
| sqrt | WebGPU | 10000 | f32 | N/A | f32 | 2.06 ms | 202.67 μs | 4.84 Mops/s |
| sqrt | WebGPU | 100000 | f32 | N/A | f32 | 2.37 ms | 164.41 μs | 42.12 Mops/s |
| sqrt_i32 | Scalar | 10000 | i32 | N/A | i32 | 89.07 μs | 24.77 μs | 112.27 Mops/s |
| sqrt_i32 | SIMD | 10000 | i32 | N/A | i32 | 169.59 μs | 70.98 μs | 58.97 Mops/s |
| sqrt_i32 | Metal | 10000 | i32 | N/A | i32 | 225.74 μs | 50.57 μs | 44.30 Mops/s |
| sqrt_i32 | WebGPU | 10000 | i32 | N/A | i32 | 2.24 ms | 269.87 μs | 4.46 Mops/s |
| sqrt_u8 | Scalar | 10000 | u8 | N/A | u8 | 92.34 μs | 24.99 μs | 108.30 Mops/s |
| sqrt_u8 | SIMD | 10000 | u8 | N/A | u8 | 147.09 μs | 66.34 μs | 67.99 Mops/s |
| sqrt_u8 | Metal | 10000 | u8 | N/A | u8 | 205.82 μs | 42.25 μs | 48.59 Mops/s |
| sqrt_u8 | WebGPU | 10000 | u8 | N/A | u8 | 2.19 ms | 207.66 μs | 4.56 Mops/s |
| sub | Scalar | 10000 | f32 | f32 | f32 | 118.81 μs | 56.62 μs | 84.17 Mops/s |
| sub | Scalar | 100000 | f32 | f32 | f32 | 128.95 μs | 30.06 μs | 775.49 Mops/s |
| sub | SIMD | 10000 | f32 | f32 | f32 | 78.76 μs | 21.34 μs | 126.97 Mops/s |
| sub | SIMD | 100000 | f32 | f32 | f32 | 138.19 μs | 50.40 μs | 723.64 Mops/s |
| sub | Metal | 10000 | f32 | f32 | f32 | 177.84 μs | 23.10 μs | 56.23 Mops/s |
| sub | Metal | 100000 | f32 | f32 | f32 | 216.46 μs | 34.40 μs | 461.98 Mops/s |
| sub | WebGPU | 10000 | f32 | f32 | f32 | 2.87 ms | 2.29 ms | 3.48 Mops/s |
| sub | WebGPU | 100000 | f32 | f32 | f32 | 2.13 ms | 105.51 μs | 46.97 Mops/s |
| sum | Scalar | 10000 | f32 | N/A | f32 | 76.67 μs | 21.40 μs | 130.44 Mops/s |
| sum | Scalar | 100000 | f32 | N/A | f32 | 114.18 μs | 26.35 μs | 875.83 Mops/s |
| sum | Scalar | 1000000 | f32 | N/A | f32 | 191.64 μs | 27.56 μs | 5.22 Gops/s |
| sum | SIMD | 10000 | f32 | N/A | f32 | 77.94 μs | 18.66 μs | 128.31 Mops/s |
| sum | SIMD | 100000 | f32 | N/A | f32 | 128.19 μs | 64.83 μs | 780.11 Mops/s |
| sum | SIMD | 1000000 | f32 | N/A | f32 | 193.96 μs | 48.62 μs | 5.16 Gops/s |
| sum | BLAS | 10000 | f32 | N/A | f32 | 79.94 μs | 25.74 μs | 125.09 Mops/s |
| sum | BLAS | 100000 | f32 | N/A | f32 | 133.97 μs | 51.41 μs | 746.42 Mops/s |
| sum | BLAS | 1000000 | f32 | N/A | f32 | 210.67 μs | 53.85 μs | 4.75 Gops/s |
| sum_axis0 | Scalar | 500x500 | f32 | N/A | f32 | 4.74 ms | 166.17 μs | 52.74 Mops/s |
| sum_axis0 | Scalar | 1000x1000 | f32 | N/A | f32 | 17.76 ms | 184.95 μs | 56.30 Mops/s |
| sum_axis0_i32 | Scalar | 500x500 | i32 | N/A | i32 | 4.46 ms | 60.59 μs | 56.11 Mops/s |
| sum_axis0_u8 | Scalar | 500x500 | u8 | N/A | u8 | 4.49 ms | 101.94 μs | 55.74 Mops/s |
| transpose | Scalar | 500x500 | f32 | N/A | f32 | 112.61 μs | 4.74 μs | 2.22 Gops/s |
| transpose | Scalar | 1000x1000 | f32 | N/A | f32 | 710.74 μs | 158.73 μs | 1.41 Gops/s |
| transpose | Scalar | 500x500 | f64 | N/A | f64 | 187.61 μs | 6.33 μs | 1.33 Gops/s |
| transpose | Scalar | 1000x1000 | f64 | N/A | f64 | 703.77 μs | 56.47 μs | 1.42 Gops/s |
| transpose_i32 | Scalar | 500x500 | i32 | N/A | i32 | 110.27 μs | 2.84 μs | 2.27 Gops/s |
| transpose_u8 | Scalar | 500x500 | u8 | N/A | u8 | 105.57 μs | 85.64 ns | 2.37 Gops/s |
| variance | Scalar | 10000 | f32 | N/A | f32 | 36.88 μs | 1.72 μs | 271.17 Mops/s |
| variance | Scalar | 100000 | f32 | N/A | f32 | 370.05 μs | 10.70 μs | 270.23 Mops/s |
| variance | Scalar | 1000000 | f32 | N/A | f32 | 3.71 ms | 56.77 μs | 269.23 Mops/s |
| variance | SIMD | 10000 | f32 | N/A | f32 | 38.45 μs | 3.63 μs | 260.10 Mops/s |
| variance | SIMD | 100000 | f32 | N/A | f32 | 377.54 μs | 11.75 μs | 264.87 Mops/s |
| variance | SIMD | 1000000 | f32 | N/A | f32 | 3.79 ms | 242.36 μs | 263.75 Mops/s |
| variance | BLAS | 10000 | f32 | N/A | f32 | 37.91 μs | 5.51 μs | 263.78 Mops/s |
| variance | BLAS | 100000 | f32 | N/A | f32 | 373.80 μs | 10.58 μs | 267.52 Mops/s |
| variance | BLAS | 1000000 | f32 | N/A | f32 | 3.73 ms | 53.45 μs | 267.81 Mops/s |

## Backend Comparison

**BLAS:** Fastest operation: dot (1000) - 141.64 ns
**Metal:** Fastest operation: dot (1000) - 141.66 ns
**SIMD:** Fastest operation: dot (1000) - 118.28 ns
**Scalar:** Fastest operation: dot (1000) - 141.60 ns
**WebGPU:** Fastest operation: dot (1000) - 148.86 ns

## Summary Statistics

**Total Benchmarks:** 183
**Backends Tested:** 5
**Operations Tested:** 29
**Highest Throughput:** matmul with BLAS (2048x2048) - 2.42 Tops/s

---
*Generated by `cargo run --bin numrs-bench --release` (BLAS/MKL enabled by default)*
