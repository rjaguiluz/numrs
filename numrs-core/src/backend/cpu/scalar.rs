use crate::array::Array;
use crate::llo::reduction::ReductionKind;
use crate::llo::ElementwiseKind;
use anyhow::{anyhow, Result};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Elementwise scalar implementation for add/mul.
/// Elementwise scalar implementation for add/mul.
#[cfg(not(target_arch = "wasm32"))]
pub fn elementwise_scalar(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array> {
    // simple broadcast rules not implemented — require same shape
    if a.shape != b.shape {
        return Err(anyhow!("shape mismatch in scalar elementwise (prototype)"));
    }
    let mut out = Array::zeros(a.shape.clone());
    match kind {
        ElementwiseKind::Add => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i] + b.data[i]);
        }
        ElementwiseKind::Mul => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i] * b.data[i]);
        }
        ElementwiseKind::Sub => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i] - b.data[i]);
        }
        ElementwiseKind::Div => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i] / b.data[i]);
        }
        ElementwiseKind::Sqrt => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].sqrt());
        }
        ElementwiseKind::Abs => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].abs());
        }
        ElementwiseKind::Neg => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = -a.data[i]);
        }
        ElementwiseKind::Exp => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].exp());
        }
        ElementwiseKind::Log => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].ln());
        }
        ElementwiseKind::Tan => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].tan());
        }
        ElementwiseKind::Sin => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].sin());
        }
        ElementwiseKind::Cos => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].cos());
        }
        ElementwiseKind::Pow => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].powf(b.data[i]));
        }
        ElementwiseKind::Asin => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].asin());
        }
        ElementwiseKind::Acos => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].acos());
        }
        ElementwiseKind::Atan => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].atan());
        }
        ElementwiseKind::Relu => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].max(0.0));
        }
        ElementwiseKind::LeakyRelu => {
            // leaky_relu(x) = x if x > 0 else 0.01 * x
            out.data.par_iter_mut().enumerate().for_each(|(i, o)| {
                let x = a.data[i];
                *o = if x > 0.0 { x } else { 0.01 * x };
            });
        }
        ElementwiseKind::Sigmoid => {
            // sigmoid(x) = 1 / (1 + exp(-x))
            out.data.par_iter_mut().enumerate().for_each(|(i, o)| {
                *o = 1.0 / (1.0 + (-a.data[i]).exp());
            });
        }
        ElementwiseKind::Tanh => {
            out.data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].tanh());
        }
        ElementwiseKind::Softplus => {
            // softplus(x) = ln(1 + exp(x))
            out.data.par_iter_mut().enumerate().for_each(|(i, o)| {
                *o = (1.0 + a.data[i].exp()).ln();
            });
        }
    }
    Ok(out)
}

/// Elementwise scalar implementation for add/mul (WASM Serial Version).
#[cfg(target_arch = "wasm32")]
pub fn elementwise_scalar(a: &Array, b: &Array, kind: ElementwiseKind) -> Result<Array> {
    // simple broadcast rules not implemented — require same shape
    if a.shape != b.shape {
        return Err(anyhow!("shape mismatch in scalar elementwise (prototype)"));
    }
    let mut out = Array::zeros(a.shape.clone());
    match kind {
        ElementwiseKind::Add => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i] + b.data[i]);
        }
        ElementwiseKind::Mul => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i] * b.data[i]);
        }
        ElementwiseKind::Sub => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i] - b.data[i]);
        }
        ElementwiseKind::Div => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i] / b.data[i]);
        }
        ElementwiseKind::Sqrt => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].sqrt());
        }
        ElementwiseKind::Abs => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].abs());
        }
        ElementwiseKind::Neg => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = -a.data[i]);
        }
        ElementwiseKind::Exp => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].exp());
        }
        ElementwiseKind::Log => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].ln());
        }
        ElementwiseKind::Tan => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].tan());
        }
        ElementwiseKind::Sin => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].sin());
        }
        ElementwiseKind::Cos => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].cos());
        }
        ElementwiseKind::Pow => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].powf(b.data[i]));
        }
        ElementwiseKind::Asin => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].asin());
        }
        ElementwiseKind::Acos => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].acos());
        }
        ElementwiseKind::Atan => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].atan());
        }
        ElementwiseKind::Relu => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].max(0.0));
        }
        ElementwiseKind::LeakyRelu => {
            out.data.iter_mut().enumerate().for_each(|(i, o)| {
                let x = a.data[i];
                *o = if x > 0.0 { x } else { 0.01 * x };
            });
        }
        ElementwiseKind::Sigmoid => {
            out.data.iter_mut().enumerate().for_each(|(i, o)| {
                *o = 1.0 / (1.0 + (-a.data[i]).exp());
            });
        }
        ElementwiseKind::Tanh => {
            out.data
                .iter_mut()
                .enumerate()
                .for_each(|(i, o)| *o = a.data[i].tanh());
        }
        ElementwiseKind::Softplus => {
            out.data.iter_mut().enumerate().for_each(|(i, o)| {
                *o = (1.0 + a.data[i].exp()).ln();
            });
        }
    }
    Ok(out)
}

/// Optimized reduction over the last axis (most cache-friendly case)
/// Data is laid out in row-major order, so reducing the last axis means
/// processing contiguous chunks of memory
/// Optimized reduction over the last axis (most cache-friendly case)
/// Data is laid out in row-major order, so reducing the last axis means
/// processing contiguous chunks of memory
#[cfg(not(target_arch = "wasm32"))]
pub fn reduce_last_axis_optimized(
    a: &Array,
    axis_size: usize,
    out_size: usize,
    out_shape: Vec<usize>,
    kind: ReductionKind,
) -> Result<Array> {
    let mut out_data = vec![0.0; out_size];

    match kind {
        ReductionKind::Sum | ReductionKind::Mean => {
            // Process in parallel chunks (each chunk reduces one row)
            out_data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    let start = i * axis_size;
                    let end = start + axis_size;
                    *out_val = a.data[start..end].iter().copied().sum();
                });

            // For mean, divide by axis size
            if kind == ReductionKind::Mean {
                out_data.par_iter_mut().for_each(|x| *x /= axis_size as f32);
            }
        }
        ReductionKind::Max => {
            out_data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    let start = i * axis_size;
                    let end = start + axis_size;
                    *out_val = a.data[start..end]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, |acc, x| acc.max(x));
                });
        }
        ReductionKind::Min => {
            out_data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    let start = i * axis_size;
                    let end = start + axis_size;
                    *out_val = a.data[start..end]
                        .iter()
                        .copied()
                        .fold(f32::INFINITY, |acc, x| acc.min(x));
                });
        }
        ReductionKind::ArgMax => {
            out_data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    let start = i * axis_size;
                    let end = start + axis_size;
                    let (max_idx, _) = a.data[start..end].iter().copied().enumerate().fold(
                        (0, f32::NEG_INFINITY),
                        |(idx, max_val), (j, val)| {
                            if val > max_val {
                                (j, val)
                            } else {
                                (idx, max_val)
                            }
                        },
                    );
                    *out_val = max_idx as f32;
                });
        }
        ReductionKind::Variance => {
            // First compute means
            let means: Vec<f32> = (0..out_size)
                .into_par_iter()
                .map(|i| {
                    let start = i * axis_size;
                    let end = start + axis_size;
                    let sum: f32 = a.data[start..end].iter().copied().sum();
                    sum / axis_size as f32
                })
                .collect();

            // Then compute variance
            out_data
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out_val)| {
                    let start = i * axis_size;
                    let end = start + axis_size;
                    let mean = means[i];
                    let var_sum: f32 = a.data[start..end]
                        .iter()
                        .copied()
                        .map(|x| {
                            let diff = x - mean;
                            diff * diff
                        })
                        .sum();
                    *out_val = var_sum / axis_size as f32;
                });
        }
    }

    Ok(Array::new(out_shape, out_data))
}

/// Optimized reduction over the last axis (WASM Serial Version)
#[cfg(target_arch = "wasm32")]
pub fn reduce_last_axis_optimized(
    a: &Array,
    axis_size: usize,
    out_size: usize,
    out_shape: Vec<usize>,
    kind: ReductionKind,
) -> Result<Array> {
    let mut out_data = vec![0.0; out_size];

    match kind {
        ReductionKind::Sum | ReductionKind::Mean => {
            // Process chunks serially
            out_data.iter_mut().enumerate().for_each(|(i, out_val)| {
                let start = i * axis_size;
                let end = start + axis_size;
                *out_val = a.data[start..end].iter().copied().sum();
            });

            // For mean, divide by axis size
            if kind == ReductionKind::Mean {
                out_data.iter_mut().for_each(|x| *x /= axis_size as f32);
            }
        }
        ReductionKind::Max => {
            out_data.iter_mut().enumerate().for_each(|(i, out_val)| {
                let start = i * axis_size;
                let end = start + axis_size;
                *out_val = a.data[start..end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, |acc, x| acc.max(x));
            });
        }
        ReductionKind::Min => {
            out_data.iter_mut().enumerate().for_each(|(i, out_val)| {
                let start = i * axis_size;
                let end = start + axis_size;
                *out_val = a.data[start..end]
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, |acc, x| acc.min(x));
            });
        }
        ReductionKind::ArgMax => {
            out_data.iter_mut().enumerate().for_each(|(i, out_val)| {
                let start = i * axis_size;
                let end = start + axis_size;
                let (max_idx, _) = a.data[start..end].iter().copied().enumerate().fold(
                    (0, f32::NEG_INFINITY),
                    |(idx, max_val), (j, val)| {
                        if val > max_val {
                            (j, val)
                        } else {
                            (idx, max_val)
                        }
                    },
                );
                *out_val = max_idx as f32;
            });
        }
        ReductionKind::Variance => {
            // Serial implementation of variance
            let means: Vec<f32> = (0..out_size)
                .map(|i| {
                    let start = i * axis_size;
                    let end = start + axis_size;
                    let sum: f32 = a.data[start..end].iter().copied().sum();
                    sum / axis_size as f32
                })
                .collect();

            out_data.iter_mut().enumerate().for_each(|(i, out_val)| {
                let start = i * axis_size;
                let end = start + axis_size;
                let mean = means[i];
                let var_sum: f32 = a.data[start..end]
                    .iter()
                    .copied()
                    .map(|x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum();
                *out_val = var_sum / axis_size as f32;
            });
        }
    }

    Ok(Array::new(out_shape, out_data))
}

/// Reduction operations (sum, max, min, mean) over all elements or axis.
pub fn reduce_scalar(a: &Array, axis: Option<usize>, kind: ReductionKind) -> Result<Array> {
    if axis.is_none() {
        // Full reduction -> scalar in shape [1]
        #[cfg(not(target_arch = "wasm32"))]
        let result: f32 = match kind {
            ReductionKind::Sum => a.data.par_iter().copied().sum(),
            ReductionKind::Max => a
                .data
                .par_iter()
                .copied()
                .fold(|| f32::NEG_INFINITY, |acc, x| acc.max(x))
                .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b)),
            ReductionKind::Min => a
                .data
                .par_iter()
                .copied()
                .fold(|| f32::INFINITY, |acc, x| acc.min(x))
                .reduce(|| f32::INFINITY, |a, b| a.min(b)),
            ReductionKind::Mean => {
                let sum: f32 = a.data.par_iter().copied().sum();
                sum / a.data.len() as f32
            }
            ReductionKind::ArgMax => {
                // Manual loop to ensure first occurrence on ties
                let mut max_idx = 0;
                let mut max_val = a.data[0];
                for (i, &val) in a.data.iter().enumerate().skip(1) {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                max_idx as f32
            }
            ReductionKind::Variance => {
                // Welford's online algorithm for numerical stability
                let n = a.data.len() as f32;
                let mut mean = 0.0f32;
                let mut m2 = 0.0f32;

                for (i, &x) in a.data.iter().enumerate() {
                    let delta = x - mean;
                    mean += delta / (i + 1) as f32;
                    let delta2 = x - mean;
                    m2 += delta * delta2;
                }

                m2 / n // population variance
            }
        };

        #[cfg(target_arch = "wasm32")]
        let result: f32 = match kind {
            ReductionKind::Sum => a.data.iter().copied().sum(),
            ReductionKind::Max => a
                .data
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, |acc, x| acc.max(x)),
            ReductionKind::Min => a
                .data
                .iter()
                .copied()
                .fold(f32::INFINITY, |acc, x| acc.min(x)),
            ReductionKind::Mean => {
                let sum: f32 = a.data.iter().copied().sum();
                sum / a.data.len() as f32
            }
            ReductionKind::ArgMax => {
                let mut max_idx = 0;
                let mut max_val = a.data[0];
                for (i, &val) in a.data.iter().enumerate().skip(1) {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                max_idx as f32
            }
            ReductionKind::Variance => {
                let n = a.data.len() as f32;
                let mut mean = 0.0f32;
                let mut m2 = 0.0f32;
                for (i, &x) in a.data.iter().enumerate() {
                    let delta = x - mean;
                    mean += delta / (i + 1) as f32;
                    let delta2 = x - mean;
                    m2 += delta * delta2;
                }
                m2 / n
            }
        };

        Ok(Array::new(vec![1], vec![result]))
    } else {
        // Axis-based reduction
        let axis = axis.unwrap();
        if axis >= a.shape.len() {
            return Err(anyhow!(
                "axis {} is out of bounds for array with {} dimensions",
                axis,
                a.shape.len()
            ));
        }

        // Compute output shape (remove the reduced axis)
        let mut out_shape: Vec<usize> = a
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect();

        // Handle edge case: reducing a 1D array produces scalar
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let axis_size = a.shape[axis];

        // OPTIMIZED PATH: Reducing over the last axis (most common and cache-friendly)
        if axis == a.shape.len() - 1 {
            return reduce_last_axis_optimized(a, axis_size, out_size, out_shape, kind);
        }

        let mut out_data = vec![0.0; out_size];

        // For ArgMax, we need to track both max values and indices
        if kind == ReductionKind::ArgMax {
            let mut max_vals = vec![f32::NEG_INFINITY; out_size];
            let mut max_indices = vec![0.0; out_size];

            // Compute strides for output shape
            let mut out_strides = vec![1; out_shape.len()];
            if out_shape.len() > 1 {
                for i in (0..out_shape.len() - 1).rev() {
                    out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                }
            }

            // Compute strides for input array (row-major)
            let mut in_strides = vec![1; a.shape.len()];
            for i in (0..a.shape.len() - 1).rev() {
                in_strides[i] = in_strides[i + 1] * a.shape[i + 1];
            }

            // Iterate over all elements
            for (linear_idx, &val) in a.data.iter().enumerate() {
                // Convert linear index to multi-dimensional index for input
                let mut in_idx = vec![0; a.shape.len()];
                let mut remaining = linear_idx;
                for (i, &stride) in in_strides.iter().enumerate() {
                    in_idx[i] = remaining / stride;
                    remaining %= stride;
                }

                // Map to output index (skip the reduced axis)
                let mut out_idx = 0;
                let mut out_pos = 0;
                for i in 0..a.shape.len() {
                    if i != axis {
                        out_idx += in_idx[i] * out_strides[out_pos];
                        out_pos += 1;
                    }
                }

                // Track max value and index along the reduced axis
                let axis_idx = in_idx[axis];
                if val > max_vals[out_idx] {
                    max_vals[out_idx] = val;
                    max_indices[out_idx] = axis_idx as f32;
                }
            }

            return Ok(Array::new(out_shape, max_indices));
        }

        // For Variance, we need two passes: first compute mean, then variance
        if kind == ReductionKind::Variance {
            // First pass: compute mean along axis
            let mean_result = reduce_scalar(a, Some(axis), ReductionKind::Mean)?;

            let mut out_data = vec![0.0; out_size];

            // Compute strides
            let mut out_strides = vec![1; out_shape.len()];
            if out_shape.len() > 1 {
                for i in (0..out_shape.len() - 1).rev() {
                    out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                }
            }

            let mut in_strides = vec![1; a.shape.len()];
            for i in (0..a.shape.len() - 1).rev() {
                in_strides[i] = in_strides[i + 1] * a.shape[i + 1];
            }

            // Second pass: compute sum of squared differences
            for (linear_idx, &val) in a.data.iter().enumerate() {
                let mut in_idx = vec![0; a.shape.len()];
                let mut remaining = linear_idx;
                for (i, &stride) in in_strides.iter().enumerate() {
                    in_idx[i] = remaining / stride;
                    remaining %= stride;
                }

                let mut out_idx = 0;
                let mut out_pos = 0;
                for i in 0..a.shape.len() {
                    if i != axis {
                        out_idx += in_idx[i] * out_strides[out_pos];
                        out_pos += 1;
                    }
                }

                let diff = val - mean_result.data[out_idx];
                out_data[out_idx] += diff * diff;
            }

            // Divide by axis size to get variance
            out_data.iter_mut().for_each(|x| *x /= axis_size as f32);

            return Ok(Array::new(out_shape, out_data));
        }

        // Initialize accumulator based on reduction kind
        let init_val = match kind {
            ReductionKind::Sum | ReductionKind::Mean => 0.0,
            ReductionKind::Max => f32::NEG_INFINITY,
            ReductionKind::Min => f32::INFINITY,
            ReductionKind::ArgMax | ReductionKind::Variance => unreachable!(), // Already handled above
        };

        out_data.iter_mut().for_each(|x| *x = init_val);

        // Compute strides for output shape
        let mut out_strides = vec![1; out_shape.len()];
        if out_shape.len() > 1 {
            for i in (0..out_shape.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }

        // Compute strides for input array (row-major)
        let mut in_strides = vec![1; a.shape.len()];
        for i in (0..a.shape.len() - 1).rev() {
            in_strides[i] = in_strides[i + 1] * a.shape[i + 1];
        }

        // Iterate over all elements
        for (linear_idx, &val) in a.data.iter().enumerate() {
            // Convert linear index to multi-dimensional index for input
            let mut in_idx = vec![0; a.shape.len()];
            let mut remaining = linear_idx;
            for (i, &stride) in in_strides.iter().enumerate() {
                in_idx[i] = remaining / stride;
                remaining %= stride;
            }

            // Map to output index (skip the reduced axis)
            let mut out_idx = 0;
            let mut out_pos = 0;
            for i in 0..a.shape.len() {
                if i != axis {
                    out_idx += in_idx[i] * out_strides[out_pos];
                    out_pos += 1;
                }
            }

            // Apply reduction operation
            match kind {
                ReductionKind::Sum | ReductionKind::Mean => {
                    out_data[out_idx] += val;
                }
                ReductionKind::Max => {
                    out_data[out_idx] = out_data[out_idx].max(val);
                }
                ReductionKind::Min => {
                    out_data[out_idx] = out_data[out_idx].min(val);
                }
                ReductionKind::ArgMax | ReductionKind::Variance => unreachable!(), // Already handled above
            }
        }

        // Finalize mean by dividing by axis size
        if kind == ReductionKind::Mean {
            out_data.iter_mut().for_each(|x| *x /= axis_size as f32);
        }

        Ok(Array::new(out_shape, out_data))
    }
}

/// Dot product implementation (single-pass fused multiply-add)
pub fn dot_scalar(a: &Array, b: &Array) -> Result<f32> {
    if a.shape.len() != 1 || b.shape.len() != 1 {
        return Err(anyhow!("dot_scalar: both inputs must be 1-D arrays"));
    }
    if a.shape[0] != b.shape[0] {
        return Err(anyhow!("dot_scalar: arrays must have same length"));
    }

    // Single-pass fused multiply-add
    let result: f32 = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).sum();

    Ok(result)
}
