use crate::array::Array;
/// SIMD implementation of Conv1D
pub fn conv1d_simd(
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    stride: usize,
    padding: usize,
) -> anyhow::Result<Array> {
    // Input: [Batch, InChannels, InLength]
    // Weight: [OutChannels, InChannels, KernelSize]
    // Output: [Batch, OutChannels, OutLength]

    let batch_size = input.shape[0];
    let in_channels = input.shape[1];
    let in_length = input.shape[2];

    let out_channels = weight.shape[0];
    let kernel_size = weight.shape[2];

    // Output length: (InLength + 2*Padding - KernelSize) / Stride + 1
    let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

    let _output_shape = vec![batch_size, out_channels, out_length];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                return conv1d_avx2_fma(
                    input,
                    weight,
                    bias,
                    batch_size,
                    in_channels,
                    in_length,
                    out_channels,
                    kernel_size,
                    out_length,
                    stride,
                    padding,
                );
            }
        }
    }

    // Fallback to naive if SIMD not available
    crate::backend::cpu::conv::conv1d_naive(input, weight, bias, stride, padding)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv1d_avx2_fma(
    input: &Array,
    weight: &Array,
    bias: Option<&Array>,
    batch_size: usize,
    in_channels: usize,
    in_length: usize,
    out_channels: usize,
    kernel_size: usize,
    out_length: usize,
    stride: usize,
    padding: usize,
) -> anyhow::Result<Array> {
    use rayon::prelude::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut output = Array::zeros(vec![batch_size, out_channels, out_length]);
    let out_ptr: *mut f32 = output.data.as_mut_ptr();

    // Pre-process bias if present
    let bias_data = if let Some(b) = bias {
        b.data.clone()
    } else {
        vec![0.0; out_channels]
    };

    // Wrapper to allow passing raw pointer to threads safely (we manage offsets manually)
    #[derive(Clone, Copy)]
    struct SendSyncPtr<T>(*mut T);
    unsafe impl<T> Send for SendSyncPtr<T> {}
    unsafe impl<T> Sync for SendSyncPtr<T> {}

    let out_ptr_wrapper = SendSyncPtr(out_ptr);

    // Parallelize over batch and output channels
    // Each thread handles one (batch, out_channel) slice outputting [out_length]
    (0..batch_size).into_par_iter().for_each(|b_idx| {
        (0..out_channels).into_par_iter().for_each(|oc| {
            let out_ptr = out_ptr_wrapper.0;
            // Get pointers relative to this task
            // Output start: b_idx * out_channels * out_length + oc * out_length
            let out_offset = b_idx * out_channels * out_length + oc * out_length;

            // Weight start: oc * in_channels * kernel_size
            let weight_offset_base = oc * in_channels * kernel_size;

            // Input start: b_idx * in_channels * in_length
            let input_offset_base = b_idx * in_channels * in_length;

            let bias_val = bias_data[oc];
            let v_bias = _mm256_set1_ps(bias_val);

            // Iterate over output sequence vectorized (8 elements at a time)
            let mut o_idx = 0;
            while o_idx + 8 <= out_length {
                let mut v_acc = v_bias;

                // Convolve over Kernel and InChannels
                for ic in 0..in_channels {
                    let w_start = weight_offset_base + ic * kernel_size;
                    let in_start = input_offset_base + ic * in_length;

                    for k in 0..kernel_size {
                        let w_val = *weight.data.get_unchecked(w_start + k);
                        let v_w = _mm256_set1_ps(w_val);

                        // Input indices for the 8 output positions
                        // input_idx = (o_idx + shift) * stride + k - padding
                        // We need to handle padding checks potentially.
                        // If padding is 0 and stride is 1 (common case), it simplifies.

                        // Vectorized load of input is tricky with stride/padding.
                        // For generic case, let's gather or scalar load into vector.

                        // Construct the 8 input values manually for FMA
                        // This is "vectorized intrisics" but partly scalar loads due to arbitrary stride
                        let mut loaded_inputs = [0.0f32; 8];
                        for lane in 0..8 {
                            let cur_out_idx = o_idx + lane;
                            let input_idx_signed =
                                (cur_out_idx * stride) as isize + k as isize - padding as isize;

                            if input_idx_signed >= 0 && input_idx_signed < in_length as isize {
                                loaded_inputs[lane] = *input
                                    .data
                                    .get_unchecked(in_start + input_idx_signed as usize);
                            } else {
                                loaded_inputs[lane] = 0.0;
                            }
                        }

                        let v_in = _mm256_loadu_ps(loaded_inputs.as_ptr());
                        v_acc = _mm256_fmadd_ps(v_in, v_w, v_acc);
                    }
                }

                // Store result
                // Safe because we are the only ones writing to this slice of output
                let dst_ptr = out_ptr.add(out_offset + o_idx);
                _mm256_storeu_ps(dst_ptr, v_acc);

                o_idx += 8;
            }

            // Handle remaining output elements
            for i in o_idx..out_length {
                let mut acc = bias_val;
                for ic in 0..in_channels {
                    let w_start = weight_offset_base + ic * kernel_size;
                    let in_start = input_offset_base + ic * in_length;

                    for k in 0..kernel_size {
                        let input_idx_signed =
                            (i * stride) as isize + k as isize - padding as isize;
                        if input_idx_signed >= 0 && input_idx_signed < in_length as isize {
                            let val = *input
                                .data
                                .get_unchecked(in_start + input_idx_signed as usize);
                            let w = *weight.data.get_unchecked(w_start + k);
                            acc += val * w;
                        }
                    }
                }
                *out_ptr.add(out_offset + i) = acc;
            }
        });
    });

    Ok(output)
}
