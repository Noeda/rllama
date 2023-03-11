use crate::unpickler;
use crate::unpickler::UnpicklingError;
use half::f16;
use rand::Rng;
use std::alloc::Layout;
use std::arch::x86_64::*;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TensorBuilder {
    pub(crate) src_path: PathBuf,
    pub(crate) dtype: TensorDType,
    pub(crate) stride: i64,
    pub(crate) rows: i64,
    pub(crate) cols: i64,
    pub(crate) nitems: i64,
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
    #[error("Invalid stride: {0}")]
    InvalidStride(i64),
}

impl TensorDType {
    fn bytes_per_item(&self) -> usize {
        match self {
            Self::Float16 => 2,
            Self::Float32 => 4,
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    data: *mut u8,
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
        unsafe {
            if self.data != std::ptr::null_mut() {
                std::alloc::dealloc(self.data, self.layout);
            }
        }
    }
}

fn compute_capacity_cols(cols: i64) -> i64 {
    if cols % 8 == 0 {
        cols
    } else {
        cols + 8 - cols % 8
    }
}

#[inline]
fn horizontal_sum(mut ymm: __m256) -> f32 {
    unsafe {
        let ymm2 = _mm256_permute2f128_ps(ymm, ymm, 1);
        ymm = _mm256_add_ps(ymm, ymm2);
        ymm = _mm256_hadd_ps(ymm, ymm);
        ymm = _mm256_hadd_ps(ymm, ymm);
        return _mm256_cvtss_f32(ymm);
    }
}

impl Tensor {
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

    pub fn rows(&self) -> i64 {
        self.rows
    }

    pub fn cols(&self) -> i64 {
        self.cols
    }

    // Gets a value as f32 from the tensor.
    #[inline]
    pub fn get_f32(&self, row: i64, col: i64) -> f32 {
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
            dtype: TensorDType::Float16,
            layout: Layout::from_size_align(0, 0).unwrap(),
            rows: 0,
            cols: 0,
            capacity_cols: 0,
        }
    }

    pub unsafe fn uninitialized(rows: i64, cols: i64, dtype: TensorDType) -> Self {
        if rows == 0 || cols == 0 {
            let mut tensor = Self::empty();
            tensor.rows = rows;
            tensor.cols = cols;
            return tensor;
        }
        // Rouns up cols to 8
        let capacity_cols = compute_capacity_cols(cols);
        let nitems = rows * capacity_cols;
        let layout =
            Layout::from_size_align((nitems as usize) * dtype.bytes_per_item(), 32).unwrap();
        let data = unsafe { std::alloc::alloc(layout) };
        if data == std::ptr::null_mut() {
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
        if self.cols != other.cols {
            panic!("Invalid hadamard product broadcast");
        }
        if other.rows != 1 {
            panic!("Invalid hadamard product broadcast");
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
        if pieces.len() == 0 {
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
        if other.rows == 1 {
            return self.matrix_vector_mul_transposed(other);
        }
        let mut result = unsafe { Tensor::uninitialized(self.rows, other.rows, self.dtype) };
        result.matrix_mul_inplace_transposed(self, other);
        result
    }

    /// Matrix multiplication done in-place
    pub fn matrix_mul_inplace(&mut self, src: &Tensor, other: &Tensor) {
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

    /// Matrix multiplication done in-place, but the second matrix is transposed.
    /// With this, you can avoid using .transpose() on the second matrix.
    pub fn matrix_mul_inplace_transposed(&mut self, src: &Tensor, other: &Tensor) {
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

                let src_cols: usize = src.cols as usize;
                let self_rows: usize = self.rows as usize;
                let self_cols: usize = self.cols as usize;
                let other_cols_capacity: usize = other.capacity_cols as usize;
                let src_cols_capacity: usize = src.capacity_cols as usize;
                let self_cols_capacity: usize = self.capacity_cols as usize;

                let src_cols_its = if src_cols % ITEMS_PER_CACHE_LINE == 0 {
                    src_cols / ITEMS_PER_CACHE_LINE
                } else {
                    src_cols / ITEMS_PER_CACHE_LINE + 1
                };

                unsafe {
                    for row in 0..self_rows {
                        let row = row as usize;
                        for col in 0..self_cols {
                            let mut target8: __m256 = _mm256_setzero_ps();
                            for p in 0..src_cols_its {
                                let src8: __m256 = _mm256_loadu_ps(
                                    src_data
                                        .add(row * src_cols_capacity + p * ITEMS_PER_CACHE_LINE),
                                );
                                let other8: __m256 = _mm256_loadu_ps(
                                    other_data
                                        .add(col * other_cols_capacity + p * ITEMS_PER_CACHE_LINE),
                                );
                                target8 = _mm256_fmadd_ps(src8, other8, target8);
                            }
                            let target: f32 = horizontal_sum(target8);
                            *tgt_data.add(row * self_cols_capacity + col) = target;
                        }
                    }
                }
            }
            TensorDType::Float16 => unimplemented!(),
        }
    }

    // Computes matrix multiplication assuming that the number of rows on the latter matrix is 1.
    //
    // AxB @ Cx1 = Ax1
    pub fn matrix_vector_mul(&self, other: &Tensor) -> Tensor {
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
        if self.cols != other.cols {
            panic!(
                "Invalid matrix-vector transposed multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        assert_eq!(other.rows, 1);
        assert_eq!(other.dtype, self.dtype);
        assert_eq!(self.dtype, TensorDType::Float32);

        unsafe {
            let mut result = Tensor::uninitialized(self.rows, 1, self.dtype);
            let col_its: usize = if self.cols % 8 == 0 {
                (self.cols / 8) as usize
            } else {
                (self.cols / 8 + 1) as usize
            };
            let self_data: *const f32 = self.data as *const f32;
            let other_data: *const f32 = other.data as *const f32;
            for row in 0..self.rows {
                let mut sum8: __m256 = _mm256_setzero_ps();
                for col in 0..col_its {
                    let col = (col * 8) as usize;
                    let left_side8 =
                        _mm256_loadu_ps(self_data.add((row * self.capacity_cols) as usize + col));
                    let right_side8 = _mm256_loadu_ps(other_data.add(col));
                    sum8 = _mm256_fmadd_ps(left_side8, right_side8, sum8);
                }
                let sum: f32 = horizontal_sum(sum8);
                result.set_f32(row, 0, sum);
            }
            result
        }
    }

    // Computes matrix multiplication assuming left side has number of rows as 1
    pub fn vector_matrix_mul(&self, other: &Tensor) -> Tensor {
        if self.cols != other.rows {
            panic!(
                "Invalid matrix-vector multiplication {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        assert_eq!(self.rows, 1);
        let mut result = unsafe { Tensor::uninitialized(1, other.cols, self.dtype) };
        for col in 0..other.cols {
            let mut sum = 0.0;
            for row in 0..self.cols {
                sum += self.get_f32(0, row) * other.get_f32(row, col);
            }
            result.set_f32(0, col, sum);
        }
        result
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
        let capacity_cols = compute_capacity_cols(cols);
        let nitems = rows * capacity_cols;
        let layout =
            Layout::from_size_align((nitems as usize) * dtype.bytes_per_item(), 32).unwrap();
        let data = unsafe { std::alloc::alloc_zeroed(layout) };
        if data == std::ptr::null_mut() {
            panic!("Failed to allocate tensor");
        }
        Self {
            data,
            dtype,
            rows,
            cols,
            capacity_cols,
            layout,
        }
    }

    pub fn clip_cols(&self, cols: usize) -> Tensor {
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
        if rows * cols != self.rows * self.cols {
            panic!("Invalid tensor view");
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

    pub fn to_f32(&self) -> Tensor {
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

    pub fn to_f16(&self) -> Tensor {
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

    pub fn row(&self, row: i64) -> Tensor {
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

impl TensorBuilder {
    pub fn load<P: AsRef<Path>>(&self, data_dir: P) -> Result<Tensor, TensorError> {
        let data_dir: &Path = data_dir.as_ref();
        if self.stride < 1 {
            return Err(TensorError::InvalidStride(self.stride));
        }
        let tensor = unsafe { Tensor::uninitialized(self.rows, self.cols, self.dtype) };
        assert_eq!(self.dtype, TensorDType::Float16);
        let path = data_dir.join(&self.src_path);

        let mut f = std::fs::File::open(&path).unwrap();
        let mut cursor: usize = 0;
        let mut buf: Vec<u8> = vec![0; self.cols as usize * 2];
        for _row in 0..self.rows {
            f.read_exact(&mut buf)?;
            unsafe {
                std::ptr::copy_nonoverlapping(buf.as_ptr(), tensor.data.add(cursor), buf.len());
            }
            cursor = cursor + (tensor.capacity_cols as usize * 2);
        }
        Ok(tensor.to_f32())
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
            let a = rng.gen_range(8..64);
            let b = rng.gen_range(8..64);
            let r = rng.gen_range(8..64);

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
                    assert_relative_eq!(c.get_f32(row, col), c2.get_f32(row, col), epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn view_preserves_values() {
        fn test_with_type(dtype: TensorDType) {
            let mut rng = rand::thread_rng();

            for _ in 0..1000 {
                let mut a: i64 = 0;
                let mut b: i64 = 0;
                let mut c: i64 = 0;
                let mut d: i64 = 0;
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
}
