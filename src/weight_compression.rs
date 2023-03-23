use crate::tensor::Tensor;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

pub fn quantize(tensor: &Tensor) -> Tensor {
    /*
     * This is a simplistic rounding quantizer. It splits each row in a tensor to 16 buckets and
     * takes the average value in said buckets as the quantized weight.
     */
    let mut allowed_values_by_row: Vec<Vec<f32>> = Vec::with_capacity(tensor.rows() as usize);
    for row in 0..tensor.rows() {
        let mut values: Vec<f32> = Vec::with_capacity(tensor.cols() as usize);
        values.truncate(0);
        let mut mi: f32 = std::f32::MAX;
        let mut ma: f32 = std::f32::MIN;

        for col in 0..tensor.cols() {
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
        let mut rng = thread_rng();
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
        allowed_values_by_row.push(allowed_values);
    }

    //let mut result = Tensor::zeros(tensor.rows(), tensor.cols(), tensor.dtype());
    let result = Tensor::make_k4bit_from_fn(
        tensor.rows(),
        tensor.cols(),
        |row, col| {
            let allowed_values: &[f32] = &allowed_values_by_row[row as usize];
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
        |row: i64| -> [f32; 16] {
            let allowed_values: &[f32] = &allowed_values_by_row[row as usize];
            let mut result: [f32; 16] = [0.0; 16];
            for i in 0..16 {
                result[i] = allowed_values[i];
            }
            result
        },
    );
    result
}
