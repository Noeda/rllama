/*
 *
 * Tensors for RLLaMA
 *
 * This is not a general Tensor library; but it has just enough to run the transformers in LLaMA
 * model.
 *
 *
 * The main structure you work with is Tensor, which is a 2D matrix. All Tensors here are 2D
 * matrices with no flexibility.
 *
 * Tensors can be 16-bit, 32-bit and they can be on OpenCL or on the CPU.
 *
 * Operations have this naming convention:
 *
 *   If it's "to_XXX", then it returns a new tensor in the specified format.
 *   If it's "XXX_inplace", then it has a &mut self and it modifies the tensor in place.
 */

#[cfg(feature = "opencl")]
use crate::tensor_opencl_support::{OpenCL, OpenCLError, OpenCLEvent, OpenCLTensor};
use crate::unpickler;
use crate::unpickler::UnpicklingError;
use half::f16;
use rand::Rng;
use rayon::prelude::*;
use std::alloc::Layout;
use std::arch::x86_64::*;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};
#[cfg(feature = "opencl")]
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TensorBuilder {
    pub(crate) src_path: PathBuf,
    pub(crate) dtype: TensorDType,
    pub(crate) stride: i64,
    pub(crate) rows: i64,
    pub(crate) cols: i64,
    pub(crate) nitems: i64,
    pub(crate) offset: i64,
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TensorDType {
    Float16,
    Float32,
}

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("IOError while reading tensor: {0} {1}")]
    TensorBuilderReadError(std::io::Error, String),
    #[error("Invalid stride: {0}")]
    InvalidStride(i64),
    #[error("Tried to build a tensor from zero files")]
    TensorBuilderEmpty,
    #[error("Tried to build a tensor from multiple files but the number of rows do not agree between the files. {0} != {1}")]
    TensorBuilderRowsMismatch(i64, i64),
    #[error("Tried to build a tensor from multiple files but the data types do not agree between the files. {0:?} != {1:?}")]
    TensorBuilderDTypeMismatch(TensorDType, TensorDType),
    #[cfg(feature = "opencl")]
    #[error("OpenCL error")]
    OpenCLError(#[from] OpenCLError),
}

impl TensorDType {
    pub fn bytes_per_item(&self) -> usize {
        match self {
            Self::Float16 => 2,
            Self::Float32 => 4,
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    data: *mut u8,
    #[cfg(feature = "opencl")]
    opencl_data: Arc<RwLock<Option<OpenCLTensor>>>,
    #[cfg(feature = "opencl")]
    waiting_for_data: Option<OpenCLEvent>, // Is OpenCL in process of sending data back to CPU?

    dtype: TensorDType,
    layout: Layout,
    rows: i64,
    cols: i64,
    // Every matrix is allocated so that cols are rounded to the next multiple of 32.
    // This lets us write AVX2 code without complicated checks.
    capacity_cols: i64,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        #[cfg(feature = "opencl")]
        {
            if let Some(ref wfd) = self.waiting_for_data {
                wfd.wait();
                let mut od = self.opencl_data.write().unwrap();
                *od = None;
            }
            let od = self.opencl_data.read().unwrap();
            if od.is_some() {
                panic!("Tried to clone a tensor that is on the GPU");
            }
        }
        unsafe {
            let new_tensor = Tensor::uninitialized(self.rows, self.cols, self.dtype);
            std::ptr::copy_nonoverlapping(
                self.data,
                new_tensor.data,
                (self.rows * self.capacity_cols * self.dtype.bytes_per_item() as i64) as usize,
            );
            new_tensor
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        #[cfg(feature = "opencl")]
        self.process_waiting_for_data_mut();
        unsafe {
            if !self.data.is_null() {
                std::alloc::dealloc(self.data, self.layout);
            }
        }
    }
}

fn compute_capacity_cols(dtype: TensorDType, cols: i64) -> i64 {
    match dtype {
        TensorDType::Float16 => compute_capacity_cols_f16(cols),
        TensorDType::Float32 => compute_capacity_cols_f32(cols),
    }
}

fn compute_capacity_cols_f32(cols: i64) -> i64 {
    if cols % 8 == 0 {
        cols
    } else {
        cols + 8 - cols % 8
    }
}

fn compute_capacity_cols_f16(cols: i64) -> i64 {
    if cols % 16 == 0 {
        cols
    } else {
        cols + 16 - cols % 16
    }
}

#[inline]
fn horizontal_sum(mut ymm: __m256) -> f32 {
    unsafe {
        let ymm2 = _mm256_permute2f128_ps(ymm, ymm, 1);
        ymm = _mm256_add_ps(ymm, ymm2);
        ymm = _mm256_hadd_ps(ymm, ymm);
        ymm = _mm256_hadd_ps(ymm, ymm);
        _mm256_cvtss_f32(ymm)
    }
}

#[inline]
fn horizontal_sum_f32_to_f16(mut ymm: __m256) -> f16 {
    unsafe {
        let ymm2 = _mm256_permute2f128_ps(ymm, ymm, 1);
        ymm = _mm256_add_ps(ymm, ymm2);
        ymm = _mm256_hadd_ps(ymm, ymm);
        ymm = _mm256_hadd_ps(ymm, ymm);
        f16::from_f32(_mm256_cvtss_f32(ymm))
    }
}

impl Tensor {
    #[inline]
    pub fn assume_on_gpu(&self) {
        #[cfg(feature = "opencl")]
        {
            self.process_waiting_for_data();
            let od = self.opencl_data.read().unwrap();
            if !od.is_some() {
                panic!("Tried to assume_on_gpu on a tensor that is on the CPU");
            }
        }
    }

    #[inline]
    pub fn assume_on_cpu(&self) {
        #[cfg(feature = "opencl")]
        {
            self.process_waiting_for_data();
            let od = self.opencl_data.read().unwrap();
            if od.is_some() {
                panic!("Tried to assume_on_cpu on a tensor that is on the GPU");
            }
        }
    }

    pub fn from_unpickled<P: AsRef<Path>, S: AsRef<str>>(
        unpickled: &unpickler::Value,
        name: S,
        data_dir: P,
    ) -> Result<Tensor, UnpicklingError> {
        let data_dir: &Path = data_dir.as_ref();
        let name: &str = name.as_ref();
        let val = unpickled
            .get_str_key(name)
            .ok_or(UnpicklingError::MissingField(name.to_string()))?;
        let val = val
            .to_tensor_builder()
            .ok_or(UnpicklingError::InvalidTensorData)?;
        let val = val.load(data_dir)?;
        Ok(val)
    }

    pub fn from_unpickled_pieces<P: AsRef<Path>, S: AsRef<str>>(
        unpickled: &[unpickler::Value],
        name: S,
        data_dir: P,
        direction: FromPiecesDirection,
    ) -> Result<Tensor, UnpicklingError> {
        let data_dir: &Path = data_dir.as_ref();
        let name: &str = name.as_ref();
        let mut builders = Vec::new();
        for unpickle in unpickled.iter() {
            let val = unpickle
                .get_str_key(name)
                .ok_or(UnpicklingError::MissingField(name.to_string()))?;
            let val = val
                .to_tensor_builder()
                .ok_or(UnpicklingError::InvalidTensorData)?;
            builders.push(val);
        }
        let val = TensorBuilder::load_from_pieces(&builders, data_dir, direction)?;
        Ok(val)
    }

    pub fn rows(&self) -> i64 {
        self.rows
    }

    pub fn cols(&self) -> i64 {
        self.cols
    }

    // Gets a value as f32 from the tensor.
    #[inline]
    pub fn get_f32(&self, row: i64, col: i64) -> f32 {
        self.assume_on_cpu();
        assert!(
            row >= 0 && col >= 0 && row < self.rows && col < self.cols,
            "Invalid index: {}, {} Size: {}, {}",
            row,
            col,
            self.rows,
            self.cols
        );

        let idx = row * self.capacity_cols + col;
        match self.dtype {
            TensorDType::Float16 => {
                let val: f16 = unsafe { *(self.data.add(idx as usize * 2) as *const f16) };
                val.to_f32()
            }
            TensorDType::Float32 => {
                let val: f32 = unsafe { *(self.data.add(idx as usize * 4) as *const f32) };
                val
            }
        }
    }

    // Sets a value from f32. The value is cast into whatever the tensor's dtype is.
    #[inline]
    pub fn set_f32(&mut self, row: i64, col: i64, val: f32) {
        self.assume_on_cpu();
        let idx = row * self.capacity_cols + col;
        match self.dtype {
            TensorDType::Float16 => {
                let val: f16 = f16::from_f32(val);
                unsafe { *(self.data.add(idx as usize * 2) as *mut f16) = val };
            }
            TensorDType::Float32 => {
                unsafe { *(self.data.add(idx as usize * 4) as *mut f32) = val };
            }
        }
    }

    // Converts the tensor to two-dimensional Vec<f32>.
    // Meant for debugging and making it easy to print tensors.
    pub fn to_vec(&self) -> Vec<Vec<f32>> {
        self.assume_on_cpu();
        let mut result = Vec::new();
        for row in 0..self.rows {
            let mut row_vec = Vec::new();
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                row_vec.push(val);
            }
            result.push(row_vec);
        }
        result
    }

    pub fn empty() -> Self {
        Self {
            data: std::ptr::null_mut(),
            #[cfg(feature = "opencl")]
            opencl_data: Arc::new(RwLock::new(None)),
            #[cfg(feature = "opencl")]
            waiting_for_data: None,
            dtype: TensorDType::Float16,
            layout: Layout::from_size_align(0, 0).unwrap(),
            rows: 0,
            cols: 0,
            capacity_cols: 0,
        }
    }

    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn uninitialized(rows: i64, cols: i64, dtype: TensorDType) -> Self {
        if rows == 0 || cols == 0 {
            let mut tensor = Self::empty();
            tensor.rows = rows;
            tensor.cols = cols;
            return tensor;
        }
        // Rouns up cols to 8
        let capacity_cols = compute_capacity_cols(dtype, cols);
        let nitems = rows * capacity_cols;
        let layout =
            Layout::from_size_align((nitems as usize) * dtype.bytes_per_item(), 32).unwrap();
        let data = unsafe { std::alloc::alloc(layout) };
        if data.is_null() {
            panic!("Failed to allocate tensor");
        }
        // Even though we are uninitialized, we should zero out the extra space between the
        // columns.
        // Otherwise there might be problems later as other operations assume it is zeroed.
        for extra_col in cols..capacity_cols {
            for row in 0..rows {
                let idx = row * capacity_cols + extra_col;
                match dtype {
                    TensorDType::Float16 => {
                        let val: f16 = f16::from_f32(0.0);
                        unsafe { *(data.add(idx as usize * 2) as *mut f16) = val };
                    }
                    TensorDType::Float32 => {
                        unsafe { *(data.add(idx as usize * 4) as *mut f32) = 0.0 };
                    }
                }
            }
        }

        Self {
            data,
            #[cfg(feature = "opencl")]
            opencl_data: Arc::new(RwLock::new(None)),
            #[cfg(feature = "opencl")]
            waiting_for_data: None,
            dtype,
            rows,
            cols,
            capacity_cols,
            layout,
        }
    }

    pub fn full(rows: i64, cols: i64, dtype: TensorDType, value: f32) -> Self {
        let mut tensor = unsafe { Tensor::uninitialized(rows, cols, dtype) };
        for row in 0..rows {
            for col in 0..cols {
                tensor.set_f32(row, col, value);
            }
        }
        tensor
    }

    // Runs softmax on row dimension.
    pub fn softmax(&self) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            let mut sum = 0.0;
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                sum += val.exp();
            }
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                result.set_f32(row, col, val.exp() / sum);
            }
        }
        result
    }

    pub fn full_triu(rows: i64, cols: i64, start_pos: i64, dtype: TensorDType, value: f32) -> Self {
        let mut tensor = unsafe { Tensor::uninitialized(rows, cols, dtype) };
        for row in 0..rows {
            for col in 0..cols {
                if col >= row + start_pos {
                    tensor.set_f32(row, col, value);
                } else {
                    tensor.set_f32(row, col, 0.0);
                }
            }
        }
        tensor
    }

    // Computes mean for each row, so that columns become 1.
    pub fn mean_cols(&self) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.rows, 1, self.dtype) };
        for row in 0..self.rows {
            let mut sum = 0.0;
            for col in 0..self.cols {
                sum += self.get_f32(row, col);
            }
            result.set_f32(row, 0, sum / self.cols as f32);
        }
        result
    }

    pub fn mean(&self) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(1, 1, self.dtype) };
        let mut sum = 0.0;
        for row in 0..self.rows {
            for col in 0..self.cols {
                sum += self.get_f32(row, col);
            }
        }
        result.set_f32(0, 0, sum / (self.rows * self.cols) as f32);
        result
    }

    pub fn pow(&self, power: f32) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                result.set_f32(row, col, val.powf(power));
            }
        }
        result
    }

    pub fn sqrt(&self) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                result.set_f32(row, col, val.sqrt());
            }
        }
        result
    }

    pub fn rsqrt(&self) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                result.set_f32(row, col, 1.0 / val.sqrt());
            }
        }
        result
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!(
                "add: Tensors must have the same shape, left: {}x{} right: {}x{}",
                self.rows(),
                self.cols(),
                other.rows(),
                other.cols()
            );
        }
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col) + other.get_f32(row, col);
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col) + scalar;
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn scalar_multiply_f32(&self, scalar: f32) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col) * scalar;
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn scalar_multiply_broadcast(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        if other.cols != 1 {
            panic!("Invalid scalar broadcast");
        }
        if other.rows != self.rows {
            panic!("Invalid scalar broadcast");
        }
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            let scalar = other.get_f32(row, 0);
            for col in 0..self.cols {
                let val = self.get_f32(row, col) * scalar;
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn scalar_product(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        if other.cols != 1 || other.rows != 1 {
            panic!("Invalid scalar product");
        }
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        let scalar = other.get_f32(0, 0);
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col) * scalar;
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn hadamard_product_broadcast(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        if self.cols != other.cols {
            panic!(
                "Invalid hadamard product broadcast: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        if other.rows != 1 {
            panic!(
                "Invalid hadamard product broadcast: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col) * other.get_f32(0, col);
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn hadamard_product(&self, other: &Tensor) -> Tensor {
        if self.cols != other.cols || self.rows != other.rows {
            panic!(
                "Invalid hadamard product: incompatible shapes, {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        #[cfg(feature = "opencl")]
        {
            if self.is_on_gpu() {
                self.hadamard_product_gpu(other)
            } else {
                self.hadamard_product_cpu(other)
            }
        }
        #[cfg(not(feature = "opencl"))]
        {
            self.hadamard_product_cpu(other)
        }
    }

    #[cfg(feature = "opencl")]
    fn hadamard_product_gpu(&self, other: &Tensor) -> Tensor {
        // Assume: sizes have been checked already
        self.assume_on_gpu();
        other.assume_on_gpu();

        self.with_opencl_data(|self_tensor| {
            let cl = self_tensor.cl();
            // TODO: do not create a CPU-side copy
            let result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
            let mut result = result.to_f16();
            result.to_gpu_inplace(&cl).unwrap();
            result.with_opencl_data_mut(|tgt_tensor| {
                tgt_tensor.copy_inplace(self_tensor).unwrap();
                other.with_opencl_data(|other_tensor| {
                    tgt_tensor.hadamard_product_inplace(other_tensor).unwrap();
                });
            });
            result
        })
    }

    fn hadamard_product_cpu(&self, other: &Tensor) -> Tensor {
        // Assume: sizes have been checked already
        self.assume_on_cpu();
        other.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col) * other.get_f32(row, col);
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn concat(pieces: &[&Tensor]) -> Tensor {
        if pieces.is_empty() {
            return Tensor::empty();
        }
        let mut total_rows: i64 = 0;
        let expected_cols: i64 = pieces[0].cols;
        let expected_dtype: TensorDType = pieces[0].dtype;
        for piece in pieces {
            if piece.cols != expected_cols {
                panic!("Invalid tensor concatenation, wrong number of columns");
            }
            if piece.dtype != expected_dtype {
                panic!("Invalid tensor concatenation, wrong dtype");
            }
            total_rows += piece.rows;
        }
        let mut result =
            unsafe { Tensor::uninitialized(total_rows, expected_cols, pieces[0].dtype) };
        let mut row_offset = 0;
        for piece in pieces {
            piece.assume_on_cpu();
            for row in 0..piece.rows {
                for col in 0..piece.cols {
                    let val = piece.get_f32(row, col);
                    result.set_f32(row_offset + row, col, val);
                }
            }
            row_offset += piece.rows;
        }
        result
    }

    pub fn silu(&self) -> Tensor {
        #[cfg(feature = "opencl")]
        {
            if self.is_on_gpu() {
                self.silu_gpu()
            } else {
                self.silu_cpu()
            }
        }
        #[cfg(not(feature = "opencl"))]
        {
            self.silu_cpu()
        }
    }

    // with_opencl_data & with_opencl_data_mut are utilities to get access to the underlying
    // OpenCLTensor, if the tensor is on gpu. Panics if they are not on GPU.
    #[cfg(feature = "opencl")]
    fn with_opencl_data<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&OpenCLTensor) -> R,
    {
        let opencl_data = self.opencl_data.read().unwrap();
        let opencl_data = opencl_data.as_ref();
        f(opencl_data.unwrap())
    }

    #[cfg(feature = "opencl")]
    fn with_opencl_data_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut OpenCLTensor) -> R,
    {
        let mut opencl_data = self.opencl_data.write().unwrap();
        let opencl_data = opencl_data.as_mut();
        f(opencl_data.unwrap())
    }

    #[cfg(feature = "opencl")]
    fn silu_gpu(&self) -> Tensor {
        self.assume_on_gpu();
        self.with_opencl_data(|src_tensor| {
            let cl: OpenCL = src_tensor.cl();
            // TODO: don't generate a CPU-side copy, create the result directly on OpenCL side
            let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
            result = result.to_f16();
            result.to_gpu_inplace(&cl).unwrap();
            result.with_opencl_data_mut(|tgt_tensor| {
                tgt_tensor.copy_inplace(src_tensor).unwrap();
                tgt_tensor.silu_inplace().unwrap();
            });
            result
        })
    }

    fn silu_cpu(&self) -> Tensor {
        let mut result = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                let val = val / (1.0 + (-val).exp());
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn transpose(&self) -> Tensor {
        #[cfg(feature = "opencl")]
        {
            if self.is_on_gpu() {
                self.transpose_gpu()
            } else {
                self.transpose_cpu()
            }
        }
        #[cfg(not(feature = "opencl"))]
        {
            self.transpose_cpu()
        }
    }

    #[cfg(feature = "opencl")]
    fn transpose_gpu(&self) -> Tensor {
        self.assume_on_gpu();
        self.with_opencl_data(|src_tensor| {
            let cl: OpenCL = src_tensor.cl();
            // TODO: don't generate a CPU-side copy, create the result directly on OpenCL side
            let mut result = unsafe { Tensor::uninitialized(self.cols, self.rows, self.dtype) };
            result = result.to_f16();
            result.to_gpu_inplace(&cl).unwrap();
            result.with_opencl_data_mut(|tgt_tensor| {
                tgt_tensor.transpose_from(src_tensor).unwrap();
            });
            result
        })
    }

    fn transpose_cpu(&self) -> Tensor {
        self.assume_on_cpu();
        let mut result = unsafe { Tensor::uninitialized(self.cols, self.rows, self.dtype) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                result.set_f32(col, row, val);
            }
        }
        result
    }

    /// Slow, naive matrix multiplication.
    ///
    /// This is used as a reference to test correctness of other matrix multiplications.
    pub fn matrix_mul_naive(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        if self.cols != other.rows {
            panic!(
                "Invalid matrix multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut result = unsafe { Tensor::uninitialized(self.rows, other.cols, self.dtype) };
        for row in 0..self.rows {
            for col in 0..other.cols {
                let mut sum = 0.0;
                for i in 0..self.cols {
                    sum += self.get_f32(row, i) * other.get_f32(i, col);
                }
                result.set_f32(row, col, sum);
            }
        }
        result
    }

    pub fn matrix_mul(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        if self.cols != other.rows {
            panic!(
                "Invalid matrix multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        if self.rows == 1 {
            return self.vector_matrix_mul(other);
        }
        if other.cols == 1 {
            return self.matrix_vector_mul(other);
        }
        let mut result = unsafe { Tensor::uninitialized(self.rows, other.cols, self.dtype) };
        result.matrix_mul_inplace(self, other);
        result
    }

    pub fn matrix_mul_transposed(&self, other: &Tensor) -> Tensor {
        if self.cols != other.cols {
            panic!(
                "Invalid matrix transposed multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.cols, other.rows
            );
        }
        // We don't have implementation for f16, so don't use the vector function if we have
        // f16
        #[cfg(not(feature = "opencl"))]
        if other.rows == 1 && other.dtype != TensorDType::Float16 {
            return self.matrix_vector_mul_transposed(other);
        }
        #[cfg(feature = "opencl")]
        if other.rows == 1 && self.is_on_cpu() {
            return self.matrix_vector_mul_transposed(other);
        }
        let mut result = unsafe { Tensor::uninitialized(self.rows, other.rows, self.dtype) };
        #[cfg(feature = "opencl")]
        if self.is_on_gpu() {
            let od = self.opencl_data.write().unwrap();
            result.to_gpu_inplace(&od.as_ref().unwrap().cl()).unwrap();
        }

        result.matrix_mul_inplace_transposed(self, other);
        result
    }

    /// Matrix multiplication done in-place
    pub fn matrix_mul_inplace(&mut self, src: &Tensor, other: &Tensor) {
        self.assume_on_cpu();
        src.assume_on_cpu();
        other.assume_on_cpu();
        if src.cols != other.rows {
            panic!(
                "Invalid matrix multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        if src.dtype != other.dtype {
            panic!("Invalid matrix multiplication, different dtypes");
        }
        if self.rows != src.rows {
            panic!("Invalid matrix multiplication, different number of rows");
        }
        if self.cols != other.cols {
            panic!("Invalid matrix multiplication, different number of cols");
        }

        match src.dtype {
            TensorDType::Float32 => {
                // not actual cache line size, but this represents 8 floats which is the number we can
                // operate with AVX2
                const CACHE_LINE_SIZE: usize = 32;
                const ITEMS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / std::mem::size_of::<f32>();

                let tgt_data: *mut f32 = self.data as *mut f32;
                unsafe {
                    std::ptr::write_bytes(
                        tgt_data,
                        0,
                        self.rows as usize * self.capacity_cols as usize,
                    );
                }
                let src_data: *const f32 = src.data as *const f32;
                let other_data: *const f32 = other.data as *const f32;

                let src_rows: usize = src.rows as usize;
                let other_cols: usize = other.cols as usize;
                let src_cols: usize = src.cols as usize;
                let other_cols_capacity: usize = other.capacity_cols as usize;
                let src_cols_capacity: usize = src.capacity_cols as usize;
                let self_cols_capacity: usize = self.capacity_cols as usize;

                let mut row: usize = 0;
                let mut col: usize;
                let mut k: usize;

                unsafe {
                    while row < src_rows {
                        col = 0;
                        while col < other_cols {
                            k = 0;
                            while k < src_cols {
                                for i2 in row..std::cmp::min(row + ITEMS_PER_CACHE_LINE, src_rows) {
                                    let i2_self_cols = i2 * self_cols_capacity;
                                    let i2_src_cols = i2 * src_cols_capacity;
                                    for k2 in k..std::cmp::min(k + ITEMS_PER_CACHE_LINE, src_cols) {
                                        let other_value8: __m256 = _mm256_loadu_ps(
                                            other_data.add(k2 * other_cols_capacity + col),
                                        );
                                        let src_value8_broadcast: __m256 =
                                            _mm256_broadcast_ss(&*src_data.add(i2_src_cols + k2));
                                        let tgt_value8: __m256 =
                                            _mm256_loadu_ps(tgt_data.add(i2_self_cols + col));
                                        let result8: __m256 = _mm256_fmadd_ps(
                                            src_value8_broadcast,
                                            other_value8,
                                            tgt_value8,
                                        );
                                        _mm256_storeu_ps(tgt_data.add(i2_self_cols + col), result8);
                                    }
                                }
                                k += ITEMS_PER_CACHE_LINE;
                            }
                            col += ITEMS_PER_CACHE_LINE;
                        }
                        row += ITEMS_PER_CACHE_LINE;
                    }
                }
            }
            TensorDType::Float16 => unsafe {
                // Even with conversion, float16 is much slower than float32
                const CACHE_LINE_SIZE: usize = 16;
                const ITEMS_PER_CACHE_LINE: usize = CACHE_LINE_SIZE / std::mem::size_of::<f16>();
                assert!(src.rows as usize % ITEMS_PER_CACHE_LINE == 0);
                assert!(src.cols as usize % ITEMS_PER_CACHE_LINE == 0);
                assert!(other.cols as usize % ITEMS_PER_CACHE_LINE == 0);
                assert!(other.rows as usize % ITEMS_PER_CACHE_LINE == 0);

                let tgt_data: *mut f16 = self.data as *mut f16;
                std::ptr::write_bytes(tgt_data, 0, self.rows as usize * self.cols as usize);
                let src_data: *const f16 = src.data as *const f16;
                let other_data: *const f16 = other.data as *const f16;

                let src_rows: usize = src.rows as usize;
                let other_cols: usize = other.cols as usize;
                let src_cols: usize = src.cols as usize;
                let self_cols: usize = self.cols as usize;

                let mut row: usize = 0;
                let mut col: usize;
                let mut k: usize;

                while row < src_rows {
                    col = 0;
                    while col < other_cols {
                        k = 0;
                        while k < src_cols {
                            for i2 in row..row + ITEMS_PER_CACHE_LINE {
                                let i2_self_cols = i2 * self_cols;
                                let i2_src_cols = i2 * src_cols;
                                for k2 in k..k + ITEMS_PER_CACHE_LINE {
                                    let other_value8: __m256 = _mm256_cvtph_ps(_mm_loadu_si128(
                                        other_data.add(k2 * other_cols + col) as *const _,
                                    ));
                                    let src_value8: f16 = *src_data.add(i2_src_cols + k2);
                                    let src_value8_broadcast: __m256 =
                                        _mm256_broadcast_ss(&src_value8.to_f32());
                                    let tgt_value8: __m256 = _mm256_cvtph_ps(_mm_loadu_si128(
                                        tgt_data.add(i2_self_cols + col) as *const _,
                                    ));
                                    let result8: __m256 = _mm256_fmadd_ps(
                                        src_value8_broadcast,
                                        other_value8,
                                        tgt_value8,
                                    );
                                    let result8_packed: __m128i = _mm256_cvtps_ph(result8, 0);
                                    _mm_storeu_si128(
                                        tgt_data.add(i2_self_cols + col) as *mut _,
                                        result8_packed,
                                    );
                                }
                            }
                            k += ITEMS_PER_CACHE_LINE;
                        }
                        col += ITEMS_PER_CACHE_LINE;
                    }
                    row += ITEMS_PER_CACHE_LINE;
                }
            },
        }
    }

    #[cfg(feature = "opencl")]
    pub fn is_on_gpu(&self) -> bool {
        if self.waiting_for_data.is_some() {
            return false;
        }
        let od = self.opencl_data.read().unwrap();
        if od.is_some() {
            return true;
        }
        false
    }

    #[cfg(feature = "opencl")]
    pub fn is_on_cpu(&self) -> bool {
        return !self.is_on_gpu();
    }

    #[cfg(feature = "opencl")]
    fn matrix_mul_inplace_transposed_gpu(&mut self, src: &Tensor, other: &Tensor) {
        let mut self_od = self.opencl_data.write().unwrap();
        let src_od = src.opencl_data.read().unwrap();
        let other_od = other.opencl_data.read().unwrap();
        let self_od: &mut OpenCLTensor = self_od.as_mut().unwrap();
        let src_od: &OpenCLTensor = src_od.as_ref().unwrap();
        let other_od: &OpenCLTensor = other_od.as_ref().unwrap();

        // TODO: if this fails, we panic. Think about if this is alright. I think for now it's
        // alright.
        self_od
            .matrix_mul_inplace_transposed(src_od, other_od)
            .unwrap();
        std::mem::drop(self_od);
        std::mem::drop(src_od);
        std::mem::drop(other_od);
    }

    /// Matrix multiplication done in-place, but the second matrix is transposed.
    /// With this, you can avoid using .transpose() on the second matrix.
    pub fn matrix_mul_inplace_transposed(&mut self, src: &Tensor, other: &Tensor) {
        #[cfg(feature = "opencl")]
        if self.is_on_gpu() && src.is_on_gpu() && other.is_on_gpu() {
            self.matrix_mul_inplace_transposed_gpu(src, other);
            return;
        }
        self.assume_on_cpu();
        src.assume_on_cpu();
        other.assume_on_cpu();
        if src.cols != other.cols {
            panic!(
                "Invalid matrix multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        if src.dtype != other.dtype {
            panic!("Invalid matrix multiplication, different dtypes");
        }
        if self.rows != src.rows {
            panic!("Invalid matrix multiplication, different number of rows");
        }
        if self.cols != other.rows {
            panic!("Invalid matrix multiplication, different number of cols");
        }

        match src.dtype {
            TensorDType::Float32 => {
                const ITEMS_PER_LINE: usize = 8;

                let tgt_data: *mut f32 = self.data as *mut f32;
                unsafe {
                    std::ptr::write_bytes(
                        tgt_data,
                        0,
                        self.rows as usize * self.capacity_cols as usize,
                    );
                }
                let src_data: *const f32 = src.data as *const f32;
                let other_data: *const f32 = other.data as *const f32;

                let src_rows: usize = src.rows as usize;
                let src_cols: usize = src.cols as usize;
                let self_rows: usize = self.rows as usize;
                let self_cols: usize = self.cols as usize;
                let _other_cols: usize = other.cols as usize;
                let other_rows: usize = other.rows as usize;
                let other_cols_capacity: usize = other.capacity_cols as usize;
                let src_cols_capacity: usize = src.capacity_cols as usize;
                let self_cols_capacity: usize = self.capacity_cols as usize;

                let src_cols_its = if src_cols % ITEMS_PER_LINE == 0 {
                    src_cols / ITEMS_PER_LINE
                } else {
                    src_cols / ITEMS_PER_LINE + 1
                };
                let row_its = if self_rows % 4 == 0 {
                    self_rows / 4
                } else {
                    self_rows / 4 + 1
                };
                let self_cols_its = if self_cols % 4 == 0 {
                    self_cols / 4
                } else {
                    self_cols / 4 + 1
                };

                unsafe {
                    for row in 0..row_its {
                        let row0 = row * 4;
                        let row1 = row * 4 + 1;
                        let row2 = row * 4 + 2;
                        let row3 = row * 4 + 3;
                        for col in 0..self_cols_its {
                            let col0 = col * 4;
                            let col1 = col * 4 + 1;
                            let col2 = col * 4 + 2;
                            let col3 = col * 4 + 3;
                            let mut targets8: [[__m256; 4]; 4] = [
                                [
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                ],
                                [
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                ],
                                [
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                ],
                                [
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                ],
                            ];
                            for p in 0..src_cols_its {
                                let other8_0: __m256 = _mm256_loadu_ps(
                                    other_data.add(col0 * other_cols_capacity + p * ITEMS_PER_LINE),
                                );
                                let other8_1: __m256 = if col1 < other_rows {
                                    _mm256_loadu_ps(
                                        other_data
                                            .add(col1 * other_cols_capacity + p * ITEMS_PER_LINE),
                                    )
                                } else {
                                    _mm256_setzero_ps()
                                };
                                let other8_2: __m256 = if col2 < other_rows {
                                    _mm256_loadu_ps(
                                        other_data
                                            .add(col2 * other_cols_capacity + p * ITEMS_PER_LINE),
                                    )
                                } else {
                                    _mm256_setzero_ps()
                                };
                                let other8_3: __m256 = if col3 < other_rows {
                                    _mm256_loadu_ps(
                                        other_data
                                            .add(col3 * other_cols_capacity + p * ITEMS_PER_LINE),
                                    )
                                } else {
                                    _mm256_setzero_ps()
                                };
                                let src8_0: __m256 = _mm256_loadu_ps(
                                    src_data.add(row0 * src_cols_capacity + p * ITEMS_PER_LINE),
                                );
                                let src8_1: __m256 = if row1 < src_rows {
                                    _mm256_loadu_ps(
                                        src_data.add(row1 * src_cols_capacity + p * ITEMS_PER_LINE),
                                    )
                                } else {
                                    _mm256_setzero_ps()
                                };
                                let src8_2: __m256 = if row2 < src_rows {
                                    _mm256_loadu_ps(
                                        src_data.add(row2 * src_cols_capacity + p * ITEMS_PER_LINE),
                                    )
                                } else {
                                    _mm256_setzero_ps()
                                };
                                let src8_3: __m256 = if row3 < src_rows {
                                    _mm256_loadu_ps(
                                        src_data.add(row3 * src_cols_capacity + p * ITEMS_PER_LINE),
                                    )
                                } else {
                                    _mm256_setzero_ps()
                                };
                                targets8[0][0] = _mm256_fmadd_ps(src8_0, other8_0, targets8[0][0]);
                                targets8[0][1] = _mm256_fmadd_ps(src8_1, other8_0, targets8[0][1]);
                                targets8[0][2] = _mm256_fmadd_ps(src8_2, other8_0, targets8[0][2]);
                                targets8[0][3] = _mm256_fmadd_ps(src8_3, other8_0, targets8[0][3]);
                                targets8[1][0] = _mm256_fmadd_ps(src8_0, other8_1, targets8[1][0]);
                                targets8[1][1] = _mm256_fmadd_ps(src8_1, other8_1, targets8[1][1]);
                                targets8[1][2] = _mm256_fmadd_ps(src8_2, other8_1, targets8[1][2]);
                                targets8[1][3] = _mm256_fmadd_ps(src8_3, other8_1, targets8[1][3]);
                                targets8[2][0] = _mm256_fmadd_ps(src8_0, other8_2, targets8[2][0]);
                                targets8[2][1] = _mm256_fmadd_ps(src8_1, other8_2, targets8[2][1]);
                                targets8[2][2] = _mm256_fmadd_ps(src8_2, other8_2, targets8[2][2]);
                                targets8[2][3] = _mm256_fmadd_ps(src8_3, other8_2, targets8[2][3]);
                                targets8[3][0] = _mm256_fmadd_ps(src8_0, other8_3, targets8[3][0]);
                                targets8[3][1] = _mm256_fmadd_ps(src8_1, other8_3, targets8[3][1]);
                                targets8[3][2] = _mm256_fmadd_ps(src8_2, other8_3, targets8[3][2]);
                                targets8[3][3] = _mm256_fmadd_ps(src8_3, other8_3, targets8[3][3]);
                            }
                            let target00: f32 = horizontal_sum(targets8[0][0]);
                            let target01: f32 = horizontal_sum(targets8[0][1]);
                            let target02: f32 = horizontal_sum(targets8[0][2]);
                            let target03: f32 = horizontal_sum(targets8[0][3]);
                            let target10: f32 = horizontal_sum(targets8[1][0]);
                            let target11: f32 = horizontal_sum(targets8[1][1]);
                            let target12: f32 = horizontal_sum(targets8[1][2]);
                            let target13: f32 = horizontal_sum(targets8[1][3]);
                            let target20: f32 = horizontal_sum(targets8[2][0]);
                            let target21: f32 = horizontal_sum(targets8[2][1]);
                            let target22: f32 = horizontal_sum(targets8[2][2]);
                            let target23: f32 = horizontal_sum(targets8[2][3]);
                            let target30: f32 = horizontal_sum(targets8[3][0]);
                            let target31: f32 = horizontal_sum(targets8[3][1]);
                            let target32: f32 = horizontal_sum(targets8[3][2]);
                            let target33: f32 = horizontal_sum(targets8[3][3]);

                            *tgt_data.add(row0 * self_cols_capacity + col0) += target00;
                            *tgt_data.add(row0 * self_cols_capacity + col1) += target10;
                            *tgt_data.add(row0 * self_cols_capacity + col2) += target20;
                            *tgt_data.add(row0 * self_cols_capacity + col3) += target30;
                            if row1 < self_rows {
                                *tgt_data.add(row1 * self_cols_capacity + col0) += target01;
                                *tgt_data.add(row1 * self_cols_capacity + col1) += target11;
                                *tgt_data.add(row1 * self_cols_capacity + col2) += target21;
                                *tgt_data.add(row1 * self_cols_capacity + col3) += target31;
                            }
                            if row2 < self_rows {
                                *tgt_data.add(row2 * self_cols_capacity + col0) += target02;
                                *tgt_data.add(row2 * self_cols_capacity + col1) += target12;
                                *tgt_data.add(row2 * self_cols_capacity + col2) += target22;
                                *tgt_data.add(row2 * self_cols_capacity + col3) += target32;
                            }
                            if row3 < self_rows {
                                *tgt_data.add(row3 * self_cols_capacity + col0) += target03;
                                *tgt_data.add(row3 * self_cols_capacity + col1) += target13;
                                *tgt_data.add(row3 * self_cols_capacity + col2) += target23;
                                *tgt_data.add(row3 * self_cols_capacity + col3) += target33;
                            }
                        }
                    }
                }
            }
            TensorDType::Float16 => {
                const ITEMS_PER_LINE: usize = 8;

                let tgt_data: *mut f16 = self.data as *mut f16;
                unsafe {
                    std::ptr::write_bytes(
                        tgt_data,
                        0,
                        self.rows as usize * self.capacity_cols as usize,
                    );
                }
                let src_data: *const f16 = src.data as *const f16;
                let other_data: *const f16 = other.data as *const f16;

                let src_rows: usize = src.rows as usize;
                let src_cols: usize = src.cols as usize;
                let self_rows: usize = self.rows as usize;
                let self_cols: usize = self.cols as usize;
                let _other_cols: usize = other.cols as usize;
                let other_rows: usize = other.rows as usize;
                let other_cols_capacity: usize = other.capacity_cols as usize;
                let src_cols_capacity: usize = src.capacity_cols as usize;
                let self_cols_capacity: usize = self.capacity_cols as usize;

                let src_cols_its = if src_cols % ITEMS_PER_LINE == 0 {
                    src_cols / ITEMS_PER_LINE
                } else {
                    src_cols / ITEMS_PER_LINE + 1
                };
                let row_its = if self_rows % 4 == 0 {
                    self_rows / 4
                } else {
                    self_rows / 4 + 1
                };
                let self_cols_its = if self_cols % 4 == 0 {
                    self_cols / 4
                } else {
                    self_cols / 4 + 1
                };

                unsafe {
                    for row in 0..row_its {
                        let row0 = row * 4;
                        let row1 = row * 4 + 1;
                        let row2 = row * 4 + 2;
                        let row3 = row * 4 + 3;
                        for col in 0..self_cols_its {
                            let col0 = col * 4;
                            let col1 = col * 4 + 1;
                            let col2 = col * 4 + 2;
                            let col3 = col * 4 + 3;
                            let mut targets8: [[__m256; 4]; 4] = [
                                [
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                ],
                                [
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                ],
                                [
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                ],
                                [
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                    _mm256_setzero_ps(),
                                ],
                            ];
                            // Loads from (row, column..column+8) and (row+1, column..column+8)
                            #[inline]
                            fn load2_rows(
                                ptr: *const f16,
                                row: usize,
                                column: usize,
                                cols_capacity: usize,
                                nrows: usize,
                            ) -> (__m256, __m256) {
                                unsafe {
                                    let (left, right) = if row + 1 < nrows {
                                        (
                                            _mm_loadu_si128(ptr.add(row * cols_capacity + column)
                                                as *const __m128i),
                                            _mm_loadu_si128(
                                                ptr.add((row + 1) * cols_capacity + column)
                                                    as *const __m128i,
                                            ),
                                        )
                                    } else {
                                        (
                                            _mm_loadu_si128(ptr.add(row * cols_capacity + column)
                                                as *const __m128i),
                                            _mm_setzero_si128(),
                                        )
                                    };
                                    let left: __m256 = _mm256_cvtph_ps(left);
                                    let right: __m256 = _mm256_cvtph_ps(right);
                                    (left, right)
                                }
                            }
                            for p in 0..src_cols_its {
                                let (other8_0, other8_1) = load2_rows(
                                    other_data,
                                    col0,
                                    p * ITEMS_PER_LINE,
                                    other_cols_capacity,
                                    other_rows,
                                );
                                let (other8_2, other8_3) = load2_rows(
                                    other_data,
                                    col2,
                                    p * ITEMS_PER_LINE,
                                    other_cols_capacity,
                                    other_rows,
                                );
                                let (src8_0, src8_1) = load2_rows(
                                    src_data,
                                    row0,
                                    p * ITEMS_PER_LINE,
                                    src_cols_capacity,
                                    src_rows,
                                );
                                let (src8_2, src8_3) = load2_rows(
                                    src_data,
                                    row2,
                                    p * ITEMS_PER_LINE,
                                    src_cols_capacity,
                                    src_rows,
                                );
                                targets8[0][0] = _mm256_fmadd_ps(src8_0, other8_0, targets8[0][0]);
                                targets8[0][1] = _mm256_fmadd_ps(src8_1, other8_0, targets8[0][1]);
                                targets8[0][2] = _mm256_fmadd_ps(src8_2, other8_0, targets8[0][2]);
                                targets8[0][3] = _mm256_fmadd_ps(src8_3, other8_0, targets8[0][3]);
                                targets8[1][0] = _mm256_fmadd_ps(src8_0, other8_1, targets8[1][0]);
                                targets8[1][1] = _mm256_fmadd_ps(src8_1, other8_1, targets8[1][1]);
                                targets8[1][2] = _mm256_fmadd_ps(src8_2, other8_1, targets8[1][2]);
                                targets8[1][3] = _mm256_fmadd_ps(src8_3, other8_1, targets8[1][3]);
                                targets8[2][0] = _mm256_fmadd_ps(src8_0, other8_2, targets8[2][0]);
                                targets8[2][1] = _mm256_fmadd_ps(src8_1, other8_2, targets8[2][1]);
                                targets8[2][2] = _mm256_fmadd_ps(src8_2, other8_2, targets8[2][2]);
                                targets8[2][3] = _mm256_fmadd_ps(src8_3, other8_2, targets8[2][3]);
                                targets8[3][0] = _mm256_fmadd_ps(src8_0, other8_3, targets8[3][0]);
                                targets8[3][1] = _mm256_fmadd_ps(src8_1, other8_3, targets8[3][1]);
                                targets8[3][2] = _mm256_fmadd_ps(src8_2, other8_3, targets8[3][2]);
                                targets8[3][3] = _mm256_fmadd_ps(src8_3, other8_3, targets8[3][3]);
                            }
                            let target00: f16 = horizontal_sum_f32_to_f16(targets8[0][0]);
                            let target01: f16 = horizontal_sum_f32_to_f16(targets8[0][1]);
                            let target02: f16 = horizontal_sum_f32_to_f16(targets8[0][2]);
                            let target03: f16 = horizontal_sum_f32_to_f16(targets8[0][3]);
                            let target10: f16 = horizontal_sum_f32_to_f16(targets8[1][0]);
                            let target11: f16 = horizontal_sum_f32_to_f16(targets8[1][1]);
                            let target12: f16 = horizontal_sum_f32_to_f16(targets8[1][2]);
                            let target13: f16 = horizontal_sum_f32_to_f16(targets8[1][3]);
                            let target20: f16 = horizontal_sum_f32_to_f16(targets8[2][0]);
                            let target21: f16 = horizontal_sum_f32_to_f16(targets8[2][1]);
                            let target22: f16 = horizontal_sum_f32_to_f16(targets8[2][2]);
                            let target23: f16 = horizontal_sum_f32_to_f16(targets8[2][3]);
                            let target30: f16 = horizontal_sum_f32_to_f16(targets8[3][0]);
                            let target31: f16 = horizontal_sum_f32_to_f16(targets8[3][1]);
                            let target32: f16 = horizontal_sum_f32_to_f16(targets8[3][2]);
                            let target33: f16 = horizontal_sum_f32_to_f16(targets8[3][3]);

                            *tgt_data.add(row0 * self_cols_capacity + col0) += target00;
                            *tgt_data.add(row0 * self_cols_capacity + col1) += target10;
                            *tgt_data.add(row0 * self_cols_capacity + col2) += target20;
                            *tgt_data.add(row0 * self_cols_capacity + col3) += target30;
                            if row1 < self_rows {
                                *tgt_data.add(row1 * self_cols_capacity + col0) += target01;
                                *tgt_data.add(row1 * self_cols_capacity + col1) += target11;
                                *tgt_data.add(row1 * self_cols_capacity + col2) += target21;
                                *tgt_data.add(row1 * self_cols_capacity + col3) += target31;
                            }
                            if row2 < self_rows {
                                *tgt_data.add(row2 * self_cols_capacity + col0) += target02;
                                *tgt_data.add(row2 * self_cols_capacity + col1) += target12;
                                *tgt_data.add(row2 * self_cols_capacity + col2) += target22;
                                *tgt_data.add(row2 * self_cols_capacity + col3) += target32;
                            }
                            if row3 < self_rows {
                                *tgt_data.add(row3 * self_cols_capacity + col0) += target03;
                                *tgt_data.add(row3 * self_cols_capacity + col1) += target13;
                                *tgt_data.add(row3 * self_cols_capacity + col2) += target23;
                                *tgt_data.add(row3 * self_cols_capacity + col3) += target33;
                            }
                        }
                    }
                }
            }
        }
    }

    // Computes matrix multiplication assuming that the number of rows on the latter matrix is 1.
    //
    // AxB @ Cx1 = Ax1
    pub fn matrix_vector_mul(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        // TODO: this function is not optimized.
        if self.cols != other.rows {
            panic!(
                "Invalid matrix-vector multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        assert_eq!(other.cols, 1);
        assert_eq!(other.dtype, self.dtype);
        assert_eq!(self.dtype, TensorDType::Float32);

        let mut result = unsafe { Tensor::uninitialized(self.rows, 1, self.dtype) };
        for row in 0..self.rows {
            let mut sum = 0.0;
            for col in 0..self.cols {
                sum += self.get_f32(row, col) * other.get_f32(col, 0);
            }
            result.set_f32(row, 0, sum);
        }
        result
    }

    /// Same as matrix_vector_mul, but right side is assumed to be transposed.
    pub fn matrix_vector_mul_transposed(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        if self.cols != other.cols {
            panic!(
                "Invalid matrix-vector transposed multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        assert_eq!(other.rows, 1);
        assert_eq!(other.dtype, self.dtype);

        match self.dtype {
            TensorDType::Float32 => self.matrix_vector_mul_transposed_f32(other),
            TensorDType::Float16 => self.matrix_vector_mul_transposed_f16(other),
            _ => panic!("Unsupported dtype"),
        }
    }

    fn matrix_vector_mul_transposed_f16(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        unsafe {
            let mut result = Tensor::uninitialized(self.rows, 1, self.dtype);
            let col_its: usize = if self.cols % 16 == 0 {
                (self.cols / 16) as usize
            } else {
                (self.cols / 16 + 1) as usize
            };
            let row_its: usize = if self.rows % 4 == 0 {
                (self.rows / 4) as usize
            } else {
                (self.rows / 4 + 1) as usize
            };
            let mut sum8s: [[__m256; 4]; 2] = [
                [
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                ],
                [
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                ],
            ];
            let self_data: *const f16 = self.data as *const f16;
            let other_data: *const f16 = other.data as *const f16;
            let _ncols_capacity: usize = result.capacity_cols as usize;
            for row in 0..row_its {
                let row: i64 = row as i64;
                sum8s[0][0] = _mm256_setzero_ps();
                sum8s[0][1] = _mm256_setzero_ps();
                sum8s[0][2] = _mm256_setzero_ps();
                sum8s[0][3] = _mm256_setzero_ps();
                sum8s[1][0] = _mm256_setzero_ps();
                sum8s[1][1] = _mm256_setzero_ps();
                sum8s[1][2] = _mm256_setzero_ps();
                sum8s[1][3] = _mm256_setzero_ps();
                let row4_0 = row * 4;
                let row4_1 = row * 4 + 1;
                let row4_2 = row * 4 + 2;
                let row4_3 = row * 4 + 3;

                // Loads from (0, column..column+8)
                #[inline]
                fn load2(ptr: *const f16, col: usize) -> __m256 {
                    unsafe { _mm256_cvtph_ps(_mm_loadu_si128(ptr.add(col) as *const __m128i)) }
                }
                // Loads from (row, column..column+8)
                #[inline]
                fn load2row(
                    ptr: *const f16,
                    row: i64,
                    col: usize,
                    cols_capacity: i64,
                    nrows: i64,
                ) -> __m256 {
                    unsafe {
                        if row < nrows {
                            _mm256_cvtph_ps(_mm_loadu_si128(
                                ptr.add(row as usize * cols_capacity as usize + col)
                                    as *const __m128i,
                            ))
                        } else {
                            _mm256_setzero_ps()
                        }
                    }
                }

                for col in 0..col_its {
                    let col = col * 16;
                    let col2 = col + 8;
                    let right_side8_0 = load2(other_data, col);
                    let left_side8_00 =
                        load2row(self_data, row4_0, col, self.capacity_cols, self.rows);
                    let left_side8_10 =
                        load2row(self_data, row4_1, col, self.capacity_cols, self.rows);
                    let left_side8_20 =
                        load2row(self_data, row4_2, col, self.capacity_cols, self.rows);
                    let left_side8_30 =
                        load2row(self_data, row4_3, col, self.capacity_cols, self.rows);
                    sum8s[0][0] = _mm256_fmadd_ps(left_side8_00, right_side8_0, sum8s[0][0]);
                    sum8s[0][1] = _mm256_fmadd_ps(left_side8_10, right_side8_0, sum8s[0][1]);
                    sum8s[0][2] = _mm256_fmadd_ps(left_side8_20, right_side8_0, sum8s[0][2]);
                    sum8s[0][3] = _mm256_fmadd_ps(left_side8_30, right_side8_0, sum8s[0][3]);
                    let right_side8_1 = load2(other_data, col2);
                    let left_side8_01 =
                        load2row(self_data, row4_0, col2, self.capacity_cols, self.rows);
                    let left_side8_11 =
                        load2row(self_data, row4_1, col2, self.capacity_cols, self.rows);
                    let left_side8_21 =
                        load2row(self_data, row4_2, col2, self.capacity_cols, self.rows);
                    let left_side8_31 =
                        load2row(self_data, row4_3, col2, self.capacity_cols, self.rows);
                    sum8s[1][0] = _mm256_fmadd_ps(left_side8_01, right_side8_1, sum8s[1][0]);
                    sum8s[1][1] = _mm256_fmadd_ps(left_side8_11, right_side8_1, sum8s[1][1]);
                    sum8s[1][2] = _mm256_fmadd_ps(left_side8_21, right_side8_1, sum8s[1][2]);
                    sum8s[1][3] = _mm256_fmadd_ps(left_side8_31, right_side8_1, sum8s[1][3]);
                }
                let sum_0: f32 = horizontal_sum(sum8s[0][0]) + horizontal_sum(sum8s[1][0]);
                let sum_1: f32 = horizontal_sum(sum8s[0][1]) + horizontal_sum(sum8s[1][1]);
                let sum_2: f32 = horizontal_sum(sum8s[0][2]) + horizontal_sum(sum8s[1][2]);
                let sum_3: f32 = horizontal_sum(sum8s[0][3]) + horizontal_sum(sum8s[1][3]);
                if row4_0 < result.rows {
                    result.set_f32(row4_0, 0, sum_0);
                }
                if row4_1 < result.rows {
                    result.set_f32(row4_1, 0, sum_1);
                }
                if row4_2 < result.rows {
                    result.set_f32(row4_2, 0, sum_2);
                }
                if row4_3 < result.rows {
                    result.set_f32(row4_3, 0, sum_3);
                }
            }
            result
        }
    }

    fn matrix_vector_mul_transposed_f32(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        unsafe {
            let mut result = Tensor::uninitialized(self.rows, 1, self.dtype);
            let col_its: usize = if self.cols % 8 == 0 {
                (self.cols / 8) as usize
            } else {
                (self.cols / 8 + 1) as usize
            };
            let row_its: usize = if self.rows % 4 == 0 {
                (self.rows / 4) as usize
            } else {
                (self.rows / 4 + 1) as usize
            };
            let mut sum8s: [__m256; 4] = [
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            ];
            let self_data: *const f32 = self.data as *const f32;
            let other_data: *const f32 = other.data as *const f32;
            let _tgt_data: *mut f32 = result.data as *mut f32;
            let _ncols_capacity: usize = result.capacity_cols as usize;
            for row in 0..row_its {
                let row: i64 = row as i64;
                sum8s[0] = _mm256_setzero_ps();
                sum8s[1] = _mm256_setzero_ps();
                sum8s[2] = _mm256_setzero_ps();
                sum8s[3] = _mm256_setzero_ps();
                let row4_0 = row * 4;
                let row4_1 = row * 4 + 1;
                let row4_2 = row * 4 + 2;
                let row4_3 = row * 4 + 3;

                for col in 0..col_its {
                    let col = col * 8;
                    let right_side8 = _mm256_loadu_ps(other_data.add(col));
                    let left_side8_0 = _mm256_loadu_ps(
                        self_data.add((row4_0 * self.capacity_cols) as usize + col),
                    );
                    let left_side8_1 = if row4_1 < self.rows {
                        _mm256_loadu_ps(self_data.add((row4_1 * self.capacity_cols) as usize + col))
                    } else {
                        _mm256_setzero_ps()
                    };
                    let left_side8_2 = if row4_2 < self.rows {
                        _mm256_loadu_ps(self_data.add((row4_2 * self.capacity_cols) as usize + col))
                    } else {
                        _mm256_setzero_ps()
                    };
                    let left_side8_3 = if row4_3 < self.rows {
                        _mm256_loadu_ps(self_data.add((row4_3 * self.capacity_cols) as usize + col))
                    } else {
                        _mm256_setzero_ps()
                    };
                    sum8s[0] = _mm256_fmadd_ps(left_side8_0, right_side8, sum8s[0]);
                    sum8s[1] = _mm256_fmadd_ps(left_side8_1, right_side8, sum8s[1]);
                    sum8s[2] = _mm256_fmadd_ps(left_side8_2, right_side8, sum8s[2]);
                    sum8s[3] = _mm256_fmadd_ps(left_side8_3, right_side8, sum8s[3]);
                }
                let sum_0: f32 = horizontal_sum(sum8s[0]);
                let sum_1: f32 = horizontal_sum(sum8s[1]);
                let sum_2: f32 = horizontal_sum(sum8s[2]);
                let sum_3: f32 = horizontal_sum(sum8s[3]);
                if row4_0 < result.rows {
                    result.set_f32(row4_0, 0, sum_0);
                }
                if row4_1 < result.rows {
                    result.set_f32(row4_1, 0, sum_1);
                }
                if row4_2 < result.rows {
                    result.set_f32(row4_2, 0, sum_2);
                }
                if row4_3 < result.rows {
                    result.set_f32(row4_3, 0, sum_3);
                }
            }
            result
        }
    }

    /// Same as matrix_vector_mul but uses threading.
    pub fn matrix_vector_mul_transposed_multithreaded(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        if self.cols != other.cols {
            panic!(
                "Invalid matrix-vector transposed multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        assert_eq!(other.rows, 1);
        assert_eq!(other.dtype, self.dtype);
        assert_eq!(self.dtype, TensorDType::Float32);

        // Use this to smuggle pointers to threads without Rust getting so goddamn mad
        //
        // Assumption usize = pointer size.
        #[derive(Copy, Clone)]
        struct WrappedPtr {
            ptr: usize,
        }
        impl WrappedPtr {
            fn wrap(ptr: *const u8) -> WrappedPtr {
                WrappedPtr { ptr: ptr as usize }
            }

            fn unwrap(self) -> *const u8 {
                self.ptr as *const u8
            }
        }

        unsafe {
            let result = Tensor::uninitialized(self.rows, 1, self.dtype);
            let capacity_cols: i64 = self.capacity_cols;
            let result_capacity_cols: i64 = result.capacity_cols;
            let col_its: usize = if self.cols % 8 == 0 {
                (self.cols / 8) as usize
            } else {
                (self.cols / 8 + 1) as usize
            };
            let self_data = WrappedPtr::wrap(self.data);
            let other_data = WrappedPtr::wrap(other.data);
            let result_data = WrappedPtr::wrap(result.data);
            (0..self.rows as usize)
                .into_par_iter()
                .with_min_len(64)
                .for_each(|row| {
                    let row = row as i64;
                    let self_data: *const f32 = self_data.unwrap() as *const f32;
                    let other_data: *const f32 = other_data.unwrap() as *const f32;
                    let result_data: *mut f32 = result_data.unwrap() as *mut f32;

                    let mut sum8: __m256 = _mm256_setzero_ps();
                    for col in 0..col_its {
                        let col = col * 8;
                        let left_side8 =
                            _mm256_loadu_ps(self_data.add((row * capacity_cols) as usize + col));
                        let right_side8 = _mm256_loadu_ps(other_data.add(col));
                        sum8 = _mm256_fmadd_ps(left_side8, right_side8, sum8);
                    }
                    let sum: f32 = horizontal_sum(sum8);
                    result_data
                        .add((row * result_capacity_cols) as usize)
                        .write(sum);
                });
            result
        }
    }

    // Computes matrix multiplication assuming left side has number of rows as 1
    #[allow(clippy::erasing_op)]
    #[allow(clippy::identity_op)]
    pub fn vector_matrix_mul(&self, other: &Tensor) -> Tensor {
        self.assume_on_cpu();
        other.assume_on_cpu();
        if self.cols != other.rows {
            panic!(
                "Invalid matrix-vector multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        assert_eq!(self.rows, 1);
        unsafe {
            let result = Tensor::uninitialized(1, other.cols, self.dtype);
            let col_its: usize = if other.rows % 8 == 0 {
                (other.rows / 8) as usize
            } else {
                (other.rows / 8 + 1) as usize
            };
            let left_data: *const f32 = self.data as *const f32;
            let right_data: *const f32 = other.data as *const f32;
            let tgt_data: *mut f32 = result.data as *mut f32;
            let other_capacity_cols = other.capacity_cols as usize;

            let o0: i32 = other_capacity_cols as i32 * 0 * 4;
            let o1: i32 = other_capacity_cols as i32 * 1 * 4;
            let o2: i32 = other_capacity_cols as i32 * 2 * 4;
            let o3: i32 = other_capacity_cols as i32 * 3 * 4;
            let o4: i32 = other_capacity_cols as i32 * 4 * 4;
            let o5: i32 = other_capacity_cols as i32 * 5 * 4;
            let o6: i32 = other_capacity_cols as i32 * 6 * 4;
            let o7: i32 = other_capacity_cols as i32 * 7 * 4;

            for col in 0..other.cols {
                let col = col as usize;
                let mut sum8: __m256 = _mm256_setzero_ps();
                for row8 in 0..col_its {
                    let row = row8 * 8;
                    let left = _mm256_loadu_ps(left_data.add(row));
                    let mut r = [0.0f32; 8];
                    // i hate you clippy because you ask me
                    // to make code more unreadable
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..8 {
                        if row + i < other.rows as usize {
                            r[i] = *right_data.add((row + i) * other_capacity_cols + col);
                        }
                    }
                    let right = if row + 8 <= other.rows as usize {
                        _mm256_i32gather_ps(
                            right_data.add(row * other_capacity_cols + col),
                            _mm256_set_epi32(o7, o6, o5, o4, o3, o2, o1, o0),
                            1,
                        )
                    } else {
                        _mm256_loadu_ps(r.as_ptr())
                    };
                    sum8 = _mm256_fmadd_ps(left, right, sum8);
                }
                *tgt_data.add(col) = horizontal_sum(sum8);
            }
            result
        }
    }

    pub fn random(rows: i64, cols: i64, dtype: TensorDType) -> Self {
        let mut result = unsafe { Tensor::uninitialized(rows, cols, dtype) };
        let mut rng = rand::thread_rng();
        for row in 0..rows {
            for col in 0..cols {
                result.set_f32(row, col, rng.gen_range(-1.0..1.0));
            }
        }
        result
    }

    pub fn eye(sz: i64, dtype: TensorDType) -> Self {
        let mut result = unsafe { Tensor::uninitialized(sz, sz, dtype) };
        for row in 0..sz {
            for col in 0..sz {
                result.set_f32(row, col, if row == col { 1.0 } else { 0.0 });
            }
        }
        result
    }

    pub fn zeros(rows: i64, cols: i64, dtype: TensorDType) -> Self {
        if rows == 0 || cols == 0 {
            let mut tensor = Self::empty();
            tensor.rows = rows;
            tensor.cols = cols;
            return tensor;
        }
        let capacity_cols = compute_capacity_cols(dtype, cols);
        let nitems = rows * capacity_cols;
        let layout =
            Layout::from_size_align((nitems as usize) * dtype.bytes_per_item(), 32).unwrap();
        let data = unsafe { std::alloc::alloc_zeroed(layout) };
        if data.is_null() {
            panic!("Failed to allocate tensor");
        }
        Self {
            data,
            #[cfg(feature = "opencl")]
            opencl_data: Arc::new(RwLock::new(None)),
            #[cfg(feature = "opencl")]
            waiting_for_data: None,
            dtype,
            rows,
            cols,
            capacity_cols,
            layout,
        }
    }

    pub fn clip_cols(&self, cols: usize) -> Tensor {
        self.assume_on_cpu();
        if cols == 0 {
            return Self::empty();
        }
        assert!(cols as i64 <= self.cols);

        let result = unsafe { Tensor::uninitialized(self.rows, cols as i64, self.dtype) };
        for row in 0..self.rows {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.data.add(
                        (row * self.capacity_cols * self.dtype.bytes_per_item() as i64) as usize,
                    ),
                    result.data.add(
                        (row * result.capacity_cols * self.dtype.bytes_per_item() as i64) as usize,
                    ),
                    cols * self.dtype.bytes_per_item(),
                );
            }
        }
        result
    }

    pub fn view(&self, rows: i64, cols: i64) -> Tensor {
        self.assume_on_cpu();
        if rows * cols != self.rows * self.cols {
            panic!(
                "Invalid tensor view, requested {}x{} but tensor is {}x{}",
                rows, cols, self.rows, self.cols
            );
        }
        if rows == self.rows {
            return self.clone();
        }
        unsafe {
            let mut result = Self::zeros(rows, cols, self.dtype);
            result.rows = rows;
            result.cols = cols;
            match self.dtype {
                TensorDType::Float16 => {
                    let mut tgt_row: usize = 0;
                    let mut tgt_col: usize = 0;
                    for src_row in 0..self.rows {
                        for src_col in 0..self.cols {
                            let idx = (src_row * self.capacity_cols + src_col) as usize;
                            let v: f16 = *(self.data.add(idx * 2) as *const f16);
                            *(result
                                .data
                                .add((tgt_row * result.capacity_cols as usize + tgt_col) * 2)
                                as *mut f16) = v;
                            tgt_col += 1;
                            if tgt_col == cols as usize {
                                tgt_col = 0;
                                tgt_row += 1;
                            }
                        }
                    }
                }
                TensorDType::Float32 => {
                    let mut tgt_row: usize = 0;
                    let mut tgt_col: usize = 0;
                    for src_row in 0..self.rows {
                        for src_col in 0..self.cols {
                            let idx = (src_row * self.capacity_cols + src_col) as usize;
                            let v: f32 = *(self.data.add(idx * 4) as *const f32);
                            *(result
                                .data
                                .add((tgt_row * result.capacity_cols as usize + tgt_col) * 4)
                                as *mut f32) = v;
                            tgt_col += 1;
                            if tgt_col == cols as usize {
                                tgt_col = 0;
                                tgt_row += 1;
                            }
                        }
                    }
                }
            }
            result
        }
    }

    /// Sends a tensor to the GPU. This is a no-op if the tensor is already on the GPU.
    ///
    /// The tensor is moved asynchronously.
    #[cfg(feature = "opencl")]
    pub fn to_gpu_inplace(&mut self, cl: &OpenCL) -> Result<(), TensorError> {
        self.process_waiting_for_data_mut();
        let mut od = self.opencl_data.write().unwrap();
        if od.is_some() {
            return Ok(());
        }
        if self.dtype != TensorDType::Float16 {
            panic!("to_gpu_inplace: Only float16 tensors are supported on the GPU");
        }
        let cl_tensor = cl.data_u16_to_gpu(
            self.data as *const u16,
            self.layout,
            (self.rows * self.capacity_cols) as usize,
            self.rows,
            self.cols,
            self.capacity_cols,
        )?;
        self.data = std::ptr::null_mut();
        *od = Some(cl_tensor);
        Ok(())
    }

    #[cfg(feature = "opencl")]
    fn process_waiting_for_data_mut(&mut self) {
        if let Some(ref wfd) = self.waiting_for_data {
            wfd.wait();
            let mut od = self.opencl_data.write().unwrap();
            *od = None;
        }
        self.waiting_for_data = None;
    }

    #[cfg(feature = "opencl")]
    fn process_waiting_for_data(&self) {
        if let Some(ref wfd) = self.waiting_for_data {
            wfd.wait();
            let mut od = self.opencl_data.write().unwrap();
            *od = None;
        }
    }

    /// Waits until asynchronous all operations on this tensor are done
    #[cfg(feature = "opencl")]
    pub fn finish(&mut self) {
        self.process_waiting_for_data_mut();
        let mut od = self.opencl_data.write().unwrap();
        if od.is_some() {
            od.as_mut().unwrap().wait_until_ready();
        }
    }

    /// Sends a tensor from the GPU to the CPU. This is a no-op if the tensor is already on the
    /// CPU.
    #[cfg(feature = "opencl")]
    pub fn to_cpu_inplace(&mut self) -> Result<(), TensorError> {
        self.process_waiting_for_data_mut();
        let mut od = self.opencl_data.write().unwrap();
        if od.is_none() {
            return Ok(());
        }
        let data = unsafe { std::alloc::alloc(self.layout) };
        if data.is_null() {
            panic!("to_cpu_inplace: Failed to allocate tensor");
        }
        let ev = od.as_mut().unwrap().data_u16_from_gpu(data as *mut u16)?;
        self.data = data as *mut u16 as *mut u8;
        self.waiting_for_data = Some(ev);
        Ok(())
    }

    /// Make sure that the tensor has finished going to GPU. Used mostly for benchmarking.
    #[cfg(feature = "opencl")]
    pub fn wait_until_on_gpu(&mut self) {
        let mut od = self.opencl_data.write().unwrap();
        if od.is_none() {
            panic!("wait_until_on_gpu: Tensor is not on GPU");
        }
        od.as_mut().unwrap().wait_until_ready();
    }

    /// Naive implementation of to_f32, used for testing that the faster methods are correct.
    pub fn to_f32_naive(&self) -> Tensor {
        self.assume_on_cpu();
        if self.dtype == TensorDType::Float32 {
            return self.clone();
        }

        let mut result =
            unsafe { Tensor::uninitialized(self.rows, self.cols, TensorDType::Float32) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn to_f32(&self) -> Tensor {
        self.assume_on_cpu();
        if self.dtype == TensorDType::Float32 {
            return self.clone();
        }

        assert_eq!(self.dtype, TensorDType::Float16);

        unsafe {
            let cols_it = if self.cols % 8 == 0 {
                self.cols / 8
            } else {
                self.cols / 8 + 1
            };
            let result = Tensor::uninitialized(self.rows, self.cols, TensorDType::Float32);

            let self_data: *const f16 = self.data as *const f16;
            let tgt_data: *mut f32 = result.data as *mut f32;
            let tgt_capacity_cols = result.capacity_cols;
            let self_capacity_cols = self.capacity_cols;
            for row in 0..self.rows {
                for col in 0..cols_it {
                    let col = col * 8;
                    let val8: __m128i =
                        _mm_loadu_si128(self_data.add((row * self_capacity_cols + col) as usize)
                            as *const __m128i);
                    let val8: __m256 = _mm256_cvtph_ps(val8);
                    _mm256_storeu_ps(tgt_data.add((row * tgt_capacity_cols + col) as usize), val8);
                }
            }
            result
        }
    }

    /// Naive implementation of to_f16, used for testing that the faster methods are correct.
    pub fn to_f16_naive(&self) -> Tensor {
        self.assume_on_cpu();
        if self.dtype == TensorDType::Float16 {
            return self.clone();
        }

        let mut result =
            unsafe { Tensor::uninitialized(self.rows, self.cols, TensorDType::Float16) };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let val = self.get_f32(row, col);
                result.set_f32(row, col, val);
            }
        }
        result
    }

    pub fn to_f16(&self) -> Tensor {
        self.assume_on_cpu();
        if self.dtype == TensorDType::Float16 {
            return self.clone();
        }

        unsafe {
            let cols_it = if self.cols % 8 == 0 {
                self.cols / 8
            } else {
                self.cols / 8 + 1
            };
            let result = Tensor::uninitialized(self.rows, self.cols, TensorDType::Float16);
            let self_data: *const f32 = self.data as *const f32;
            let tgt_data: *mut f16 = result.data as *mut f16;
            let tgt_capacity_cols = result.capacity_cols;
            let self_capacity_cols = self.capacity_cols;

            for row in 0..self.rows {
                for col in 0..cols_it {
                    let col = col * 8;
                    let val8: __m256 =
                        _mm256_loadu_ps(self_data.add((row * self_capacity_cols + col) as usize));
                    let val8: __m128i = _mm256_cvtps_ph(val8, 0);
                    _mm_storeu_si128(
                        tgt_data.add((row * tgt_capacity_cols + col) as usize) as *mut __m128i,
                        val8,
                    );
                }
            }
            result
        }
    }

    pub fn row(&self, row: i64) -> Tensor {
        self.assume_on_cpu();
        if row < 0 || row > self.rows {
            panic!("Invalid row index");
        }

        let result = unsafe { Tensor::uninitialized(1, self.cols, self.dtype) };
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data
                    .add((row * self.capacity_cols) as usize * self.dtype.bytes_per_item()),
                result.data,
                self.cols as usize * self.dtype.bytes_per_item(),
            );
        }
        result
    }
}

/// When we load multiple tensors, should we slap them together row by row, or column by column?
///
/// E.g. If we have 32x4 and 32x4   then Rows  --> 64x4
///      If we have 32x4 and 32x4   then Cols  --> 32x8
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Debug)]
pub enum FromPiecesDirection {
    Rows,
    Cols,
}

impl TensorBuilder {
    pub fn load<P: AsRef<Path>>(&self, data_dir: P) -> Result<Tensor, TensorError> {
        let data_dir: &Path = data_dir.as_ref();
        if self.stride < 1 {
            return Err(TensorError::InvalidStride(self.stride));
        }
        let tensor = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        let path = data_dir
            .join(format!("consolidated.{:02}", 0))
            .join("data")
            .join(&self.src_path);

        let mut f = std::fs::File::open(&path).unwrap();
        f.seek(std::io::SeekFrom::Start(
            (self.offset as u64) * self.dtype.bytes_per_item() as u64,
        ))?;
        let mut cursor: usize = 0;
        let mut buf: Vec<u8> = vec![0; self.cols as usize * self.dtype.bytes_per_item()];
        for _row in 0..self.rows {
            f.read_exact(&mut buf)?;
            unsafe {
                std::ptr::copy_nonoverlapping(buf.as_ptr(), tensor.data.add(cursor), buf.len());
            }
            cursor += tensor.capacity_cols as usize * self.dtype.bytes_per_item();
        }
        Ok(tensor.to_f32())
    }

    /// Loads a tensor from multiple TensorBuilders; used to load a tensor from multiple files
    /// which is what the larger LLaMA models do.
    pub fn load_from_pieces<P: AsRef<Path>>(
        builders: &[Self],
        data_dir: P,
        direction: FromPiecesDirection,
    ) -> Result<Tensor, TensorError> {
        let data_dir: &Path = data_dir.as_ref();
        if builders.is_empty() {
            return Err(TensorError::TensorBuilderEmpty);
        }

        fn load_from_pieces_cols(
            builders: &[TensorBuilder],
            data_dir: &Path,
        ) -> Result<Tensor, TensorError> {
            let mut total_cols: i64 = 0;
            let expected_rows: i64 = builders[0].rows;
            let expected_dtype: TensorDType = builders[0].dtype;

            // Do some checking before we attempt loading.
            for builder in builders.iter() {
                total_cols += builder.cols;
                if builder.stride < 1 {
                    return Err(TensorError::InvalidStride(builder.stride));
                }
                if builder.rows != expected_rows {
                    return Err(TensorError::TensorBuilderRowsMismatch(
                        builder.rows,
                        expected_rows,
                    ));
                }
                if builder.dtype != expected_dtype {
                    return Err(TensorError::TensorBuilderDTypeMismatch(
                        builder.dtype,
                        expected_dtype,
                    ));
                }
            }

            let tensor =
                unsafe { Tensor::uninitialized(expected_rows, total_cols, builders[0].dtype) };
            let mut buf: Vec<u8> = vec![];
            let mut col_offset = 0;
            for (idx, builder) in builders.iter().enumerate() {
                let path = data_dir
                    .join(format!("consolidated.{:02}", idx))
                    .join("data")
                    .join(&builder.src_path);
                buf.truncate(0);
                buf.resize(builder.cols as usize * builder.dtype.bytes_per_item(), 0);
                let mut f = std::fs::File::open(&path).unwrap();
                f.seek(std::io::SeekFrom::Start(
                    (builder.offset as u64) * builder.dtype.bytes_per_item() as u64,
                ))?;
                for row in 0..builder.rows {
                    match f.read_exact(&mut buf) {
                        Ok(_) => {}
                        Err(err) => {
                            return Err(TensorError::TensorBuilderReadError(
                                err,
                                format!(
                                    "path={:?} row={} expected_len={} offset={}",
                                    path,
                                    row,
                                    buf.len(),
                                    builder.offset
                                ),
                            ));
                        }
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            buf.as_ptr(),
                            tensor.data.add(
                                ((row * tensor.capacity_cols + col_offset) as usize)
                                    * builder.dtype.bytes_per_item(),
                            ),
                            buf.len(),
                        );
                    }
                }
                col_offset += builder.cols;
            }
            Ok(tensor.to_f32())
        }

        fn load_from_pieces_rows(
            builders: &[TensorBuilder],
            data_dir: &Path,
        ) -> Result<Tensor, TensorError> {
            let mut total_rows: i64 = 0;
            let expected_cols: i64 = builders[0].cols;
            let expected_dtype: TensorDType = builders[0].dtype;

            // Do some checking before we attempt loading.
            for builder in builders.iter() {
                total_rows += builder.rows;
                if builder.stride < 1 {
                    return Err(TensorError::InvalidStride(builder.stride));
                }
                if builder.cols != expected_cols {
                    return Err(TensorError::TensorBuilderRowsMismatch(
                        builder.cols,
                        expected_cols,
                    ));
                }
                if builder.dtype != expected_dtype {
                    return Err(TensorError::TensorBuilderDTypeMismatch(
                        builder.dtype,
                        expected_dtype,
                    ));
                }
            }

            let tensor =
                unsafe { Tensor::uninitialized(total_rows, expected_cols, builders[0].dtype) };
            let mut buf: Vec<u8> = vec![];
            let mut row_offset: i64 = 0;
            for (idx, builder) in builders.iter().enumerate() {
                let path = data_dir
                    .join(format!("consolidated.{:02}", idx))
                    .join("data")
                    .join(&builder.src_path);
                buf.truncate(0);
                buf.resize(builder.cols as usize * builder.dtype.bytes_per_item(), 0);
                let mut f = std::fs::File::open(&path).unwrap();
                f.seek(std::io::SeekFrom::Start(
                    (builder.offset as u64) * builder.dtype.bytes_per_item() as u64,
                ))?;
                for row in 0..builder.rows {
                    match f.read_exact(&mut buf) {
                        Ok(_) => {}
                        Err(err) => {
                            return Err(TensorError::TensorBuilderReadError(
                                err,
                                format!(
                                    "path={:?} row={} expected_len={} offset={}",
                                    path,
                                    row,
                                    buf.len(),
                                    builder.offset
                                ),
                            ));
                        }
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            buf.as_ptr(),
                            tensor.data.add(
                                (((row + row_offset) * tensor.capacity_cols) as usize)
                                    * builder.dtype.bytes_per_item(),
                            ),
                            buf.len(),
                        );
                    }
                }
                row_offset += builder.rows;
            }
            Ok(tensor.to_f32())
        }

        match direction {
            FromPiecesDirection::Rows => load_from_pieces_rows(builders, data_dir),
            FromPiecesDirection::Cols => load_from_pieces_cols(builders, data_dir),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mat_mul_transposed_agrees_with_regular_mat_mul() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a = rng.gen_range(1..=128);
            let b = rng.gen_range(1..=128);
            let r = rng.gen_range(1..=128);

            // Make matrixes AxR and RxB
            let a = Tensor::random(a, r, TensorDType::Float32);
            let b = Tensor::random(r, b, TensorDType::Float32);
            let b_transposed = b.transpose();

            let c = a.matrix_mul(&b);
            let c2 = a.matrix_mul_transposed(&b_transposed);

            assert_eq!(c.rows, c2.rows);
            assert_eq!(c.cols, c2.cols);

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-3);
                }
            }
        }
    }

    #[test]
    fn mat_mul_transposed_f32_agrees_mat_mul_transposed_f16() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a = rng.gen_range(1..=128);
            let b = rng.gen_range(1..=128);
            let r = rng.gen_range(1..=128);

            // Make matrixes AxR and RxB
            let a = Tensor::random(a, r, TensorDType::Float32);
            let b = Tensor::random(r, b, TensorDType::Float32);
            let a2 = a.clone().to_f16();
            let b2 = b.clone().to_f16();
            let b_transposed = b.transpose();
            let b2_transposed = b2.transpose();

            let c = a.matrix_mul_transposed(&b_transposed);
            let c2 = a2.matrix_mul_transposed(&b2_transposed);

            assert_eq!(c.rows, c2.rows);
            assert_eq!(c.cols, c2.cols);

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-1);
                }
            }
        }
    }

    #[test]
    fn mat_vector_mul_transposed_f32_agrees_mat_vector_mul_transposed_f16() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a = rng.gen_range(1..=128);
            let r = rng.gen_range(1..=128);

            // Make matrixes AxR and Rx1
            let a = Tensor::random(a, r, TensorDType::Float32);
            let b = Tensor::random(r, 1, TensorDType::Float32);
            let a2 = a.clone().to_f16();
            let b2 = b.clone().to_f16();
            let b_transposed = b.transpose();
            let b2_transposed = b2.transpose();

            let c = a.matrix_vector_mul_transposed(&b_transposed);
            let c2 = a2.matrix_vector_mul_transposed(&b2_transposed);

            assert_eq!(c.rows, c2.rows);
            assert_eq!(c.cols, c2.cols);

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-1);
                }
            }
        }
    }

    #[test]
    fn view_preserves_values() {
        fn test_with_type(dtype: TensorDType) {
            let mut rng = rand::thread_rng();

            for _ in 0..1000 {
                let mut a: i64;
                let mut b: i64;
                let mut c: i64;
                let d: i64;
                loop {
                    a = rng.gen_range(8..64);
                    b = rng.gen_range(8..64);
                    c = rng.gen_range(8..64);
                    if (a * b) % c != 0 {
                        continue;
                    }
                    d = (a * b) / c;
                    break;
                }

                let tensor_left = Tensor::random(a, b, dtype);
                let tensor_right = tensor_left.view(c, d);

                assert_eq!(
                    tensor_left.cols() * tensor_left.rows(),
                    tensor_right.cols() * tensor_right.rows()
                );

                let mut cursor: usize = 0;
                let mut left_row: usize = 0;
                let mut left_col: usize = 0;
                let mut right_row: usize = 0;
                let mut right_col: usize = 0;

                while cursor < tensor_left.cols() as usize * tensor_left.rows() as usize {
                    let left_value = tensor_left.get_f32(left_row as i64, left_col as i64);
                    let right_value = tensor_right.get_f32(right_row as i64, right_col as i64);
                    assert_eq!(
                        left_value, right_value,
                        "left: {:?}, right: {:?} dtype {:?}",
                        tensor_left, tensor_right, dtype
                    );
                    left_col += 1;
                    if left_col == tensor_left.cols() as usize {
                        left_col = 0;
                        left_row += 1;
                    }
                    right_col += 1;
                    if right_col == tensor_right.cols() as usize {
                        right_col = 0;
                        right_row += 1;
                    }
                    cursor += 1;
                }
            }
        }
        test_with_type(TensorDType::Float32);
        test_with_type(TensorDType::Float16);
    }

    #[test]
    fn mat_vector_mul_matches_naive_mat_mul() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let r = rng.gen_range(1..100);
            let r2 = rng.gen_range(1..100);

            let a = Tensor::random(r, r2, TensorDType::Float32);
            let b = Tensor::random(r2, 1, TensorDType::Float32);

            let c = a.matrix_mul_naive(&b);
            let c2 = a.matrix_vector_mul(&b);

            assert_eq!(c.rows(), c2.rows());
            assert_eq!(c.cols(), c2.cols());

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn mat_vector_transposed_mul_matches_naive_mat_mul() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let r = rng.gen_range(1..100);
            let r2 = rng.gen_range(1..100);

            let a = Tensor::random(r, r2, TensorDType::Float32);
            let b = Tensor::random(1, r2, TensorDType::Float32);

            let c = a.matrix_mul_naive(&b.transpose());
            let c2 = a.matrix_vector_mul_transposed(&b);

            assert_eq!(c.rows(), c2.rows());
            assert_eq!(c.cols(), c2.cols());

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn naive_mat_mul_and_fast_are_same_f32_random_sizes() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let left_rows = rng.gen_range(1..100);
            let right_cols = rng.gen_range(1..100);
            let shared_len = rng.gen_range(1..100);

            let a = Tensor::random(left_rows, shared_len, TensorDType::Float32);
            let b = Tensor::random(shared_len, right_cols, TensorDType::Float32);

            let c = a.matrix_mul_naive(&b);
            let c2 = a.matrix_mul(&b);

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn naive_mat_mul_and_fast_are_same_f32() {
        for _ in 0..50 {
            let a = Tensor::random(16, 32, TensorDType::Float32);
            let b = Tensor::random(32, 16, TensorDType::Float32);

            let c = a.matrix_mul_naive(&b);
            let c2 = a.matrix_mul(&b);

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn mat_mul_with_itself_is_correct_f32() {
        for _ in 0..50 {
            let a = Tensor::random(16, 16, TensorDType::Float32);
            let c = a.matrix_mul_naive(&a);
            let c2 = a.matrix_mul(&a);

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn vector_mat_mul_and_naive_mat_mul_agree() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let a = rng.gen_range(1..100);
            let b = rng.gen_range(1..100);

            let m1 = Tensor::random(1, a, TensorDType::Float32);
            let m2 = Tensor::random(a, b, TensorDType::Float32);

            let c = m1.matrix_mul_naive(&m2);
            let c2 = m1.vector_matrix_mul(&m2);

            assert_eq!(c.rows(), c2.rows());
            assert_eq!(c.cols(), c2.cols());

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn naive_mat_mul_and_fast_are_same_f16() {
        for _ in 0..50 {
            let a = Tensor::random(16, 32, TensorDType::Float16);
            let b = Tensor::random(32, 16, TensorDType::Float16);

            let c = a.matrix_mul_naive(&b);
            let c2 = a.matrix_mul(&b);

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-1);
                }
            }
        }
    }

    #[test]
    fn mat_mul_with_itself_is_correct_f16() {
        for _ in 0..50 {
            let a = Tensor::random(16, 16, TensorDType::Float16);
            let c = a.matrix_mul_naive(&a);
            let c2 = a.matrix_mul(&a);

            for row in 0..c.rows {
                for col in 0..c.cols {
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-1);
                }
            }
        }
    }

    #[test]
    fn clip_cols_works() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let rows = rng.gen_range(1..100);
            let cols = rng.gen_range(2..100);
            let new_cols = rng.gen_range(1..=cols);

            let a = Tensor::random(rows, cols, TensorDType::Float32);
            let a_clipped = a.clip_cols(new_cols as usize);

            assert_eq!(a.rows(), a_clipped.rows());
            assert_eq!(a_clipped.cols(), new_cols);

            for row in 0..a_clipped.rows {
                for col in 0..a_clipped.cols {
                    assert_eq!(a.get_f32(row, col), a_clipped.get_f32(row, col));
                }
            }
        }
    }

    #[test]
    fn conversion_from_f16_tensor_to_f32_tensor_agrees_with_naive() {
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let rows = rng.gen_range(1..100);
            let cols = rng.gen_range(1..100);

            let src = Tensor::random(rows, cols, TensorDType::Float16);
            let tgt1 = src.to_f32_naive();
            let tgt2 = src.to_f32();

            assert_eq!(tgt1.rows(), tgt2.rows());
            assert_eq!(tgt1.cols(), tgt2.cols());
            for row in 0..tgt1.rows {
                for col in 0..tgt1.cols {
                    assert_eq!(tgt1.get_f32(row, col), tgt2.get_f32(row, col));
                }
            }
        }
    }

    #[test]
    fn conversion_from_f32_tensor_to_f16_tensor_agrees_with_naive() {
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let rows = rng.gen_range(1..100);
            let cols = rng.gen_range(1..100);

            let src = Tensor::random(rows, cols, TensorDType::Float32);
            let tgt1 = src.to_f16_naive();
            let tgt2 = src.to_f16();

            assert_eq!(tgt1.rows(), tgt2.rows());
            assert_eq!(tgt1.cols(), tgt2.cols());
            for row in 0..tgt1.rows {
                for col in 0..tgt1.cols {
                    assert_eq!(tgt1.get_f32(row, col), tgt2.get_f32(row, col));
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_transposed_is_close_to_cpu_matrix_mul_transposed_512x1024() {
        let cl = OpenCL::new(false, 0).unwrap();
        let a = Tensor::random(512, 1024, TensorDType::Float32);
        let b = Tensor::random(768, 1024, TensorDType::Float32);
        let mut a2 = a.to_f16();
        let mut b2 = b.to_f16();
        let mut c = Tensor::random(512, 768, TensorDType::Float32);
        let mut c2 = Tensor::zeros(512, 768, TensorDType::Float32).to_f16();
        a2.to_gpu_inplace(&cl).unwrap();
        b2.to_gpu_inplace(&cl).unwrap();
        c2.to_gpu_inplace(&cl).unwrap();
        c.matrix_mul_inplace_transposed(&a, &b);
        c2.matrix_mul_inplace_transposed(&a2, &b2);
        c2.to_cpu_inplace().unwrap();

        assert_eq!(c.rows(), c2.rows());
        assert_eq!(c.cols(), c2.cols());

        for row in 0..c.rows {
            for col in 0..c.cols {
                assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-1);
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_transposed_is_close_to_cpu_matrix_mul_transposed_1024x1024() {
        let cl = OpenCL::new(false, 0).unwrap();
        let a = Tensor::random(1024, 1024, TensorDType::Float32);
        let b = Tensor::random(1024, 1024, TensorDType::Float32);
        let mut a2 = a.to_f16();
        let mut b2 = b.to_f16();
        let mut c = Tensor::random(1024, 1024, TensorDType::Float32);
        let mut c2 = Tensor::zeros(1024, 1024, TensorDType::Float32).to_f16();
        a2.to_gpu_inplace(&cl).unwrap();
        b2.to_gpu_inplace(&cl).unwrap();
        c2.to_gpu_inplace(&cl).unwrap();
        c.matrix_mul_inplace_transposed(&a, &b);
        c2.matrix_mul_inplace_transposed(&a2, &b2);
        c2.to_cpu_inplace().unwrap();

        assert_eq!(c.rows(), c2.rows());
        assert_eq!(c.cols(), c2.cols());

        for row in 0..c.rows {
            for col in 0..c.cols {
                assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-1);
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_silu_and_cpu_silu_agree() {
        let cl = OpenCL::new(false, 0).unwrap();

        for _trial in 0..300 {
            let mut rng = rand::thread_rng();
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);
            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = mat1.clone();
            let mut mat2 = mat2.to_f16();
            mat2.to_gpu_inplace(&cl).unwrap();

            let mat1_result = mat1.silu();
            let mut mat2_result = mat2.silu();
            mat2_result.to_cpu_inplace().unwrap();

            assert_eq!(mat1_result.rows(), mat2_result.rows());
            assert_eq!(mat1_result.cols(), mat2_result.cols());

            for row in 0..mat1_result.rows {
                for col in 0..mat1_result.cols {
                    assert_relative_eq!(
                        mat1_result.get_f32(row, col),
                        mat2_result.get_f32(row, col),
                        epsilon = 1e-2
                    );
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_hadamard_product_and_cpu_hadamard_product_agree() {
        let cl = OpenCL::new(false, 0).unwrap();

        for _trial in 0..300 {
            let mut rng = rand::thread_rng();
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);
            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = Tensor::random(a, b, TensorDType::Float16);

            let mut mat1_gpu = mat1.to_f16();
            let mut mat2_gpu = mat2.to_f16();
            mat1_gpu.to_gpu_inplace(&cl).unwrap();
            mat2_gpu.to_gpu_inplace(&cl).unwrap();

            let result1 = mat1.hadamard_product(&mat2);
            let mut result2 = mat1_gpu.hadamard_product(&mat2_gpu);
            result2.to_cpu_inplace().unwrap();

            assert_eq!(result1.rows(), result2.rows());
            assert_eq!(result1.cols(), result2.cols());

            for row in 0..result1.rows() {
                for col in 0..result2.cols() {
                    assert_relative_eq!(
                        result1.get_f32(row, col),
                        result2.get_f32(row, col),
                        epsilon = 1e-2
                    );
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_transpose_and_cpu_transpose_agree() {
        let cl = OpenCL::new(false, 0).unwrap();
        let mut rng = rand::thread_rng();
        for _trial in 0..300 {
            let a = rng.gen_range(1..=100);
            let b = rng.gen_range(1..=100);
            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mut mat1_gpu = mat1.to_f16();
            mat1_gpu.to_gpu_inplace(&cl).unwrap();

            let mat1_transposed = mat1.transpose();
            let mut mat1_gpu_transposed = mat1_gpu.transpose();
            mat1_gpu_transposed.to_cpu_inplace().unwrap();

            assert_eq!(mat1_transposed.rows(), mat1_gpu_transposed.rows());
            assert_eq!(mat1_transposed.cols(), mat1_gpu_transposed.cols());

            for row in 0..mat1_transposed.rows {
                for col in 0..mat1_transposed.cols {
                    assert_relative_eq!(
                        mat1_transposed.get_f32(row, col),
                        mat1_gpu_transposed.get_f32(row, col),
                        epsilon = 1e-2,
                    );
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_transposed_is_close_to_cpu_matrix_mul_transposed() {
        let cl = OpenCL::new(false, 0).unwrap();
        let mut rng = rand::thread_rng();

        for _trial in 0..300 {
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);
            let c = rng.gen_range(1..=300);

            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = Tensor::random(c, b, TensorDType::Float16);
            let mat3 = Tensor::random(a, c, TensorDType::Float16);
            let mut mat1_gpu = mat1.clone();
            let mut mat2_gpu = mat2.clone();
            let mut mat3_gpu = mat3.clone();
            mat1_gpu.to_gpu_inplace(&cl).unwrap();
            mat2_gpu.to_gpu_inplace(&cl).unwrap();
            mat3_gpu.to_gpu_inplace(&cl).unwrap();

            let mat1 = mat1.to_f32();
            let mat2 = mat2.to_f32();
            let mut mat3 = mat3.to_f32();

            mat3.matrix_mul_inplace_transposed(&mat1, &mat2);
            mat3_gpu.matrix_mul_inplace_transposed(&mat1_gpu, &mat2_gpu);
            mat3_gpu.to_cpu_inplace().unwrap();

            assert_eq!(mat3.rows(), mat3_gpu.rows());
            assert_eq!(mat3.cols(), mat3_gpu.cols());

            for row in 0..mat3.rows {
                for col in 0..mat3.cols {
                    assert_relative_eq!(
                        mat3.get_f32(row, col),
                        mat3_gpu.get_f32(row, col),
                        epsilon = 1e-2,
                    );
                }
            }
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn gpu_matrix_mul_vector_transposed_is_close_to_cpu_matrix_mul_vector_transposed() {
        let cl = OpenCL::new(false, 0).unwrap();
        let mut rng = rand::thread_rng();

        for _trial in 0..300 {
            let a = rng.gen_range(1..=300);
            let b = rng.gen_range(1..=300);

            let mat1 = Tensor::random(a, b, TensorDType::Float16);
            let mat2 = Tensor::random(1, b, TensorDType::Float16);
            let mat3 = Tensor::random(a, 1, TensorDType::Float16);
            let mut mat1_gpu = mat1.clone();
            let mut mat2_gpu = mat2.clone();
            let mut mat3_gpu = mat3.clone();
            mat1_gpu.to_gpu_inplace(&cl).unwrap();
            mat2_gpu.to_gpu_inplace(&cl).unwrap();
            mat3_gpu.to_gpu_inplace(&cl).unwrap();

            let mat1 = mat1.to_f32();
            let mat2 = mat2.to_f32();
            let mut mat3 = mat3.to_f32();

            mat3.matrix_mul_inplace_transposed(&mat1, &mat2);
            mat3_gpu.matrix_mul_inplace_transposed(&mat1_gpu, &mat2_gpu);
            mat3_gpu.to_cpu_inplace().unwrap();

            assert_eq!(mat3.rows(), mat3_gpu.rows());
            assert_eq!(mat3.cols(), mat3_gpu.cols());

            for row in 0..mat3.rows {
                for col in 0..mat3.cols {
                    assert_relative_eq!(
                        mat3.get_f32(row, col),
                        mat3_gpu.get_f32(row, col),
                        epsilon = 1e-2,
                    );
                }
            }
        }
    }
}
