extern crate rllama;

use rllama::tensor::{Tensor, TensorDType};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn tensor_benchmarks(c: &mut Criterion) {
    let orig16_1 = Tensor::full(16, 32, TensorDType::Float16, 3.0);
    let orig16_2 = Tensor::full(32, 512, TensorDType::Float16, -1.33);

    let orig32_1 = Tensor::full(16, 32, TensorDType::Float32, 3.0);
    let orig32_2 = Tensor::full(32, 512, TensorDType::Float32, -1.33);
    let orig32_2_transposed = orig32_2.transpose();

    let mut result_16 = Tensor::zeros(16, 512, TensorDType::Float16);
    let mut result_32 = Tensor::zeros(16, 512, TensorDType::Float32);

    let orig_84096_1 = Tensor::zeros(8, 4096, TensorDType::Float32);
    let orig_84096_2 = Tensor::zeros(4096, 4096, TensorDType::Float32);
    let mut result_84096 = Tensor::zeros(8, 4096, TensorDType::Float32);

    c.bench_function(
        "matrix multiplication 8x4096 @ 4096x4096 f32 in-place",
        |b| {
            b.iter(|| {
                let _ = result_84096
                    .matrix_mul_inplace(black_box(&orig_84096_1), black_box(&orig_84096_2));
            })
        },
    );

    c.bench_function(
        "matrix multiplication 8x4096 @ 4096x4096 f32 in-place, transposed",
        |b| {
            b.iter(|| {
                let _ = result_84096.matrix_mul_inplace_transposed(
                    black_box(&orig_84096_1),
                    black_box(&orig_84096_2),
                );
            })
        },
    );

    c.bench_function("matrix multiplication f32 not in-place", |b| {
        b.iter(|| {
            let _ = black_box(&orig32_1).matrix_mul(black_box(&orig32_2));
        })
    });
    c.bench_function("matrix multiplication f32 naive", |b| {
        b.iter(|| {
            let _ = black_box(&orig32_1).matrix_mul_naive(black_box(&orig32_2));
        })
    });
    c.bench_function("matrix multiplication f16 not in-place", |b| {
        b.iter(|| {
            let _ = black_box(&orig16_1).matrix_mul(black_box(&orig16_2));
        })
    });
    c.bench_function("matrix multiplication f16 naive", |b| {
        b.iter(|| {
            let _ = black_box(&orig16_1).matrix_mul_naive(black_box(&orig16_2));
        })
    });
    c.bench_function("matrix multiplication f16 in-place", |b| {
        b.iter(|| {
            let _ = result_16.matrix_mul_inplace(black_box(&orig16_1), black_box(&orig16_2));
        })
    });
    c.bench_function("matrix multiplication f32 in-place", |b| {
        b.iter(|| {
            let _ = result_32.matrix_mul_inplace(black_box(&orig32_1), black_box(&orig32_2));
        })
    });
    c.bench_function("matrix multiplication f32 in-place, transposed", |b| {
        b.iter(|| {
            let _ = result_32.matrix_mul_inplace_transposed(
                black_box(&orig32_1),
                black_box(&orig32_2_transposed),
            );
        })
    });
}

criterion_group!(benches, tensor_benchmarks);
criterion_main!(benches);
