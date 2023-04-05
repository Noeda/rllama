#![feature(stdsimd)]
#![feature(decl_macro)]

pub mod data_source;
pub mod embedding;
pub mod huggingface_loader;
pub mod model_params;
pub mod protomodels;
pub mod rllama_main;
pub mod semaphore;
pub mod simd_support;
pub mod tensor;
#[cfg(feature = "opencl")]
pub mod tensor_opencl_support;
pub mod token_sampler;
pub mod tokenizer;
pub mod transformer;
pub mod unpickler;
pub mod weight_compression;
#[cfg(feature = "server")]
#[macro_use]
extern crate rocket;
