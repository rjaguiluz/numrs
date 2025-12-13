/**
 * ArrayView C API Benchmark
 * 
 * This benchmark measures real C FFI performance using ArrayView pattern:
 * - Create ArrayView ONCE (copies data into Rust)
 * - Reuse for multiple operations (zero-copy!)
 * - Compare with old API (to_vec every call)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "../include/numrs.h"

#define SIZE 10000
#define ITERATIONS 1000

// Helper: Get time in microseconds
double get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000.0 + ts.tv_nsec / 1000.0;
}

int main() {
    printf("🔬 NumRs C API Benchmark - ArrayView Performance\n");
    printf("=================================================\n\n");
    printf("Configuration:\n");
    printf("  Size: %d elements\n", SIZE);
    printf("  Iterations: %d\n", ITERATIONS);
    printf("  Data type: float (f32)\n\n");
    
    // Allocate input/output arrays
    float* data_a = (float*)malloc(SIZE * sizeof(float));
    float* data_b = (float*)malloc(SIZE * sizeof(float));
    float* output = (float*)malloc(SIZE * sizeof(float));
    
    if (!data_a || !data_b || !output) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }
    
    // Initialize data
    for (int i = 0; i < SIZE; i++) {
        data_a[i] = (float)i;
        data_b[i] = (float)(i * 2);
        output[i] = 0.0f;
    }
    
    printf("🔥 Starting benchmarks...\n\n");
    
    // =========================================================================
    // BENCHMARK 1: Old API (to_vec every call)
    // =========================================================================
    printf("📊 Benchmark 1: Old API (numrs_add - to_vec cada llamada)\n");
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        numrs_add((void*)data_a, (void*)data_b, (void*)output, SIZE, NumRsDType_Float32);
    }
    
    double start = get_time_us();
    for (int i = 0; i < ITERATIONS; i++) {
        numrs_add((void*)data_a, (void*)data_b, (void*)output, SIZE, NumRsDType_Float32);
    }
    double end = get_time_us();
    
    double avg_old_api = (end - start) / ITERATIONS;
    printf("  Average time: %.2f μs\n", avg_old_api);
    printf("  Total time: %.2f ms\n", (end - start) / 1000.0);
    printf("  Throughput: %.2f Mops/s\n\n", (SIZE / avg_old_api));
    
    // =========================================================================
    // BENCHMARK 2: ArrayView API (create once, reuse many times)
    // =========================================================================
    printf("📊 Benchmark 2: ArrayView API (crear UNA VEZ, reusar)\n");
    
    // Create ArrayViews ONCE (copy happens here)
    printf("  Creating ArrayViews...\n");
    NumRsArrayView* view_a = numrs_array_view_new_f32(data_a, SIZE);
    NumRsArrayView* view_b = numrs_array_view_new_f32(data_b, SIZE);
    
    if (!view_a || !view_b) {
        fprintf(stderr, "Failed to create ArrayViews!\n");
        return 1;
    }
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        numrs_add_view(view_a, view_b, (void*)output, SIZE);
    }
    
    // Benchmark (NO to_vec() here!)
    start = get_time_us();
    for (int i = 0; i < ITERATIONS; i++) {
        numrs_add_view(view_a, view_b, (void*)output, SIZE);
    }
    end = get_time_us();
    
    double avg_arrayview = (end - start) / ITERATIONS;
    printf("  Average time: %.2f μs\n", avg_arrayview);
    printf("  Total time: %.2f ms\n", (end - start) / 1000.0);
    printf("  Throughput: %.2f Mops/s\n\n", (SIZE / avg_arrayview));
    
    // Cleanup
    numrs_array_view_destroy(view_a);
    numrs_array_view_destroy(view_b);
    
    // =========================================================================
    // BENCHMARK 3: ArrayView with multiple operations (real-world scenario)
    // =========================================================================
    printf("📊 Benchmark 3: ArrayView con múltiples operaciones\n");
    printf("  (add + mul + sub en la misma iteración)\n");
    
    // Create views again
    view_a = numrs_array_view_new_f32(data_a, SIZE);
    view_b = numrs_array_view_new_f32(data_b, SIZE);
    
    float* temp1 = (float*)malloc(SIZE * sizeof(float));
    float* temp2 = (float*)malloc(SIZE * sizeof(float));
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        numrs_add_view(view_a, view_b, (void*)output, SIZE);
        numrs_mul_view(view_a, view_b, (void*)temp1, SIZE);
        numrs_sub_view(view_a, view_b, (void*)temp2, SIZE);
    }
    
    start = get_time_us();
    for (int i = 0; i < ITERATIONS; i++) {
        numrs_add_view(view_a, view_b, (void*)output, SIZE);
        numrs_mul_view(view_a, view_b, (void*)temp1, SIZE);
        numrs_sub_view(view_a, view_b, (void*)temp2, SIZE);
    }
    end = get_time_us();
    
    double avg_multi = (end - start) / ITERATIONS;
    printf("  Average time: %.2f μs (3 operaciones)\n", avg_multi);
    printf("  Per operation: %.2f μs\n", avg_multi / 3.0);
    printf("  Total time: %.2f ms\n", (end - start) / 1000.0);
    printf("  Speedup vs old API: %.2fx\n\n", (avg_old_api * 3.0) / avg_multi);
    
    // Cleanup
    numrs_array_view_destroy(view_a);
    numrs_array_view_destroy(view_b);
    free(temp1);
    free(temp2);
    
    // =========================================================================
    // SUMMARY
    // =========================================================================
    printf("📈 SUMMARY\n");
    printf("=================================================\n");
    printf("Old API (to_vec cada vez):       %.2f μs\n", avg_old_api);
    printf("ArrayView API (reusar):           %.2f μs\n", avg_arrayview);
    printf("ArrayView multi-op (per op):      %.2f μs\n", avg_multi / 3.0);
    printf("\n");
    
    double improvement = ((avg_old_api - avg_arrayview) / avg_old_api) * 100.0;
    printf("Mejora: %.1f%% más rápido con ArrayView\n", improvement);
    
    if (improvement > 30.0) {
        printf("✅ ArrayView API es significativamente más rápido!\n");
    } else if (improvement > 10.0) {
        printf("✓ ArrayView API muestra mejora\n");
    } else {
        printf("⚠️  Mejora marginal - investigar más\n");
    }
    
    // Cleanup
    free(data_a);
    free(data_b);
    free(output);
    
    return 0;
}
