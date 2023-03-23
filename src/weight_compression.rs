use crate::tensor::Tensor;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

pub fn quantize(tensor: &Tensor) -> Tensor {
    /*
     * This is a simplistic rounding quantizer. It splits each row in a tensor to 16 buckets and
     * takes the average value in said buckets as the quantized weight.
     */
    let mut allowed_values_by_row_block: Vec<Vec<Vec<f32>>> =
        Vec::with_capacity(tensor.rows() as usize);
    let col_blocks = (tensor.cols() + 511) / 512;
    for row in 0..tensor.rows() {
        let mut block_values: Vec<Vec<f32>> = Vec::with_capacity(col_blocks as usize);
        for block in 0..col_blocks {
            let start = block * 512;
            let end = std::cmp::min(start + 512, tensor.cols());

            let mut values: Vec<f32> = Vec::with_capacity(512);
            values.truncate(0);
            let mut mi: f32 = std::f32::MAX;
            let mut ma: f32 = std::f32::MIN;

            for col in start..end {
                let val = tensor.get_f32(row, col);
                if val < mi {
                    mi = val;
                }
                if val > ma {
                    ma = val;
                }
                values.push(val);
            }
            values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mut allowed_values: Vec<f32> = Vec::with_capacity(16);
            for i in 0..16 {
                let start_idx = i * values.len() / 16;
                let end_idx = (i + 1) * values.len() / 16;

                let mut avg = 0.0;
                for j in start_idx..end_idx {
                    avg += values[j];
                }
                avg /= (end_idx - start_idx) as f32;
                allowed_values.push(avg);
            }
            allowed_values[0] = mi;
            allowed_values[15] = ma;
            allowed_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            block_values.push(allowed_values);
        }
        allowed_values_by_row_block.push(block_values);
    }

    let result = Tensor::make_k4bit_from_fn(
        tensor.rows(),
        tensor.cols(),
        |row, col| {
            let col_block = col / 512;
            let allowed_values: &[f32] =
                &allowed_values_by_row_block[row as usize][col_block as usize];
            let val = tensor.get_f32(row, col);
            let mut best = 0;
            let mut best_dist = std::f32::MAX;
            for i in 0..16 {
                let dist = (val - allowed_values[i] as f32).abs();
                if dist < best_dist {
                    best = i;
                    best_dist = dist;
                }
            }
            best as u8
        },
        |row: i64, col: i64| -> [f32; 16] {
            let allowed_values: &[f32] =
                &allowed_values_by_row_block[row as usize][col as usize / 512];
            let mut result: [f32; 16] = [0.0; 16];
            for i in 0..16 {
                result[i] = allowed_values[i];
            }
            result
        },
    );
    result
}

// Same as quantize but doesn't actually change the type of the tensor. It just changes the tensor
// itself. Used to test new quantization schemes without writing support for them.
pub fn quantize_test(tensor: &Tensor) -> Tensor {
    let mut result = Tensor::zeros(tensor.rows(), tensor.cols(), tensor.dtype());
    for row in 0..tensor.rows() {
        let col_blocks = (tensor.cols() + 511) / 512;
        for block in 0..col_blocks {
            let mut values: Vec<f32> = Vec::with_capacity(tensor.cols() as usize);
            values.truncate(0);
            let mut mi: f32 = std::f32::MAX;
            let mut ma: f32 = std::f32::MIN;

            let start = block * 512;
            let end = std::cmp::min(start + 512, tensor.cols());

            for col in start..end {
                let val = tensor.get_f32(row, col);
                if val < mi {
                    mi = val;
                }
                if val > ma {
                    ma = val;
                }
                values.push(val);
            }
            values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let mut allowed_values: Vec<f32> = Vec::with_capacity(16);
            for i in 0..16 {
                let start_idx = i * values.len() / 16;
                let end_idx = (i + 1) * values.len() / 16;

                let mut avg = 0.0;
                for j in start_idx..end_idx {
                    avg += values[j];
                }
                avg /= (end_idx - start_idx) as f32;
                allowed_values.push(avg);
            }
            allowed_values[0] = mi;
            allowed_values[15] = ma;
            allowed_values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            for col in start..end {
                let val = tensor.get_f32(row, col);
                let mut best = 0;
                let mut best_dist = std::f32::MAX;
                for i in 0..16 {
                    let dist = (val - allowed_values[i] as f32).abs();
                    if dist < best_dist {
                        best = i;
                        best_dist = dist;
                    }
                }
                result.set_f32(row, col, allowed_values[best as usize]);
            }
        }
    }
    result
}
