extern crate rllama;
#[cfg(feature = "opencl")]
use rllama::tensor_opencl_support::OpenCL;

use rllama::tensor::{Tensor, TensorDType};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[cfg(feature = "opencl")]
pub fn opencl_benchmarks(c: &mut Criterion) {
    let mut orig1 = Tensor::random(1, 1, TensorDType::Float16);
    let mut orig16 = Tensor::random(1024, 1024, TensorDType::Float16);
    let mut orig32 = Tensor::random(4096, 4096, TensorDType::Float16);
    let cl = OpenCL::new(false, 0).unwrap();

    let mut mul_left = Tensor::random(1024, 1024, TensorDType::Float16);
    mul_left.to_gpu_inplace(&cl).unwrap();
    let mut mul_right = Tensor::random(1024, 1024, TensorDType::Float16);
    mul_right.to_gpu_inplace(&cl).unwrap();
    let mut mul_target = Tensor::zeros(1024, 1024, TensorDType::Float16);
    mul_target.to_gpu_inplace(&cl).unwrap();

    let mut mul_left_cpu = Tensor::random(1024, 1024, TensorDType::Float32);
    let mut mul_right_cpu = Tensor::random(1024, 1024, TensorDType::Float32);
    let mut mul_target_cpu = Tensor::random(1024, 1024, TensorDType::Float32);

    let mut mul_left1 = Tensor::random(4096, 11000, TensorDType::Float16);
    let mut mul_right1 = Tensor::random(1, 11000, TensorDType::Float16);
    let mut mul_target1 = Tensor::zeros(4096, 1, TensorDType::Float16);
    mul_left1.to_gpu_inplace(&cl).unwrap();
    mul_right1.to_gpu_inplace(&cl).unwrap();
    mul_target1.to_gpu_inplace(&cl).unwrap();

    c.bench_function(
        "4096x11000 to 1x11000 matrix multiplication transposed on OpenCL",
        |b| {
            b.iter(|| {
                mul_target1
                    .matrix_mul_inplace_transposed(black_box(&mul_left1), black_box(&mul_right1));
                mul_target1.finish();
            })
        },
    );

    c.bench_function(
        "1024x1024 matrix multiplication transposed on OpenCL",
        |b| {
            b.iter(|| {
                mul_target
                    .matrix_mul_inplace_transposed(black_box(&mul_left), black_box(&mul_right));
                mul_target.finish();
            })
        },
    );

    c.bench_function("1024x1024 matrix multiplication transposed on CPU", |b| {
        b.iter(|| {
            let _ = mul_target_cpu.matrix_mul_inplace_transposed(&mul_left_cpu, &mul_right_cpu);
        })
    });

    c.bench_function("1x1 matrix from CPU to OpenCL device and back", |b| {
        b.iter(|| {
            let _ = orig1.to_gpu_inplace(&cl).unwrap();
            let _ = orig1.to_cpu_inplace();
            orig1.finish();
        })
    });

    c.bench_function("1024x1024 matrix from CPU to OpenCL device and back", |b| {
        b.iter(|| {
            let _ = orig16.to_gpu_inplace(&cl).unwrap();
            let _ = orig16.to_cpu_inplace();
            orig16.finish();
        })
    });

    c.bench_function("4096x4096 matrix from CPU to OpenCL device and back", |b| {
        b.iter(|| {
            let _ = orig32.to_gpu_inplace(&cl).unwrap();
            let _ = orig32.to_cpu_inplace();
            orig32.finish();
        })
    });
}

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
    let orig_84096_quant = orig_84096_1.quantize();
    let mut result_84096 = Tensor::zeros(8, 4096, TensorDType::Float32);

    let orig_84096_1_f16 = Tensor::zeros(8, 4096, TensorDType::Float16);
    let orig_84096_2_f16 = Tensor::zeros(4096, 4096, TensorDType::Float16);
    let mut result_84096_f16 = Tensor::zeros(8, 4096, TensorDType::Float16);

    let orig_f32 = Tensor::zeros(1024, 1024, TensorDType::Float32);
    let orig_f16 = Tensor::zeros(1024, 1024, TensorDType::Float16);

    let m1 = Tensor::random(1024, 128, TensorDType::Float32);
    let m2 = Tensor::random(1, 128, TensorDType::Float32);
    let m1_f16 = m1.to_f16();
    let m2_f16 = m2.to_f16();

    let quant = m1.quantize();
    let quant2 = m2.quantize();

    c.bench_function(
        "1024x128 * 1x128 matrix vector transposed multiplication, k4 quantized * f32",
        |b| {
            b.iter(|| {
                let _ = quant.matrix_vector_mul_transposed(black_box(&m2));
            })
        },
    );
    c.bench_function(
        "1024x128 * 1x128 matrix vector transposed multiplication, f32 quantized * k4",
        |b| {
            b.iter(|| {
                let _ = m1.matrix_vector_mul_transposed(black_box(&quant2));
            })
        },
    );

    c.bench_function(
        "matrix multiplication 8x4096 @ 4096x4096 k4 quantized * f32 in-place, transposed",
        |b| {
            b.iter(|| {
                let _ = result_84096.matrix_mul_inplace_transposed(
                    black_box(&orig_84096_quant),
                    black_box(&orig_84096_2),
                );
            })
        },
    );

    c.bench_function(
        "1024x128 * 1x128 matrix vector transposed multiplication, f32",
        |b| {
            b.iter(|| {
                let _ = m1.matrix_vector_mul_transposed(black_box(&m2));
            })
        },
    );

    c.bench_function(
        "1024x128 * 1x128 matrix vector transposed multiplication, f16",
        |b| {
            b.iter(|| {
                let _ = m1_f16.matrix_vector_mul_transposed(black_box(&m2_f16));
            })
        },
    );

    c.bench_function(
        "matrix multiplication 8x4096 @ 4096x4096 f16 in-place, transposed",
        |b| {
            b.iter(|| {
                let _ = result_84096_f16.matrix_mul_inplace_transposed(
                    black_box(&orig_84096_1_f16),
                    black_box(&orig_84096_2_f16),
                );
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

    c.bench_function(
        "matrix multiplication 8x4096 @ 4096x4096 f32 in-place",
        |b| {
            b.iter(|| {
                let _ = result_84096
                    .matrix_mul_inplace(black_box(&orig_84096_1), black_box(&orig_84096_2));
            })
        },
    );

    c.bench_function("1024x1024 matrix from f32->f16", |b| {
        b.iter(|| {
            let _ = black_box(&orig_f32).to_f16();
        })
    });

    c.bench_function("1024x1024 matrix from f16->f32", |b| {
        b.iter(|| {
            let _ = black_box(&orig_f16).to_f32();
        })
    });

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

#[cfg(feature = "opencl")]
criterion_group!(benches, opencl_benchmarks, tensor_benchmarks);
#[cfg(not(feature = "opencl"))]
criterion_group!(benches, tensor_benchmarks);
criterion_main!(benches);
