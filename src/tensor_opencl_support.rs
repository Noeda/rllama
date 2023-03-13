/*
 * OpenCL stuff to run (some) of the tensor operations.
 */

use ocl::{Buffer, Context, Device, Event, Kernel, Platform, Program, Queue};
use std::alloc::Layout;
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Debug)]
#[allow(dead_code)]
struct Programs {
    matrix_mul_transposed_by_row_f16_program: Program,
    matrix_mul_transposed_by_row_f16: Kernel,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct OpenCL {
    ctx: Context,
    queue: Queue,
    programs: Arc<RwLock<Programs>>,
}

#[derive(Debug)]
pub struct OpenCLTensor {
    buf: Buffer<u16>, // really is f16
    initial_write_event: Option<ocl::Event>,
    last_event: Option<ocl::Event>,
    data: *const u16,
    data_layout: Layout,
    nitems: usize,
    rows: i64,
    cols: i64,
    cols_capacity: i64,
    queue: Queue,
    programs: Arc<RwLock<Programs>>,
}

#[derive(Debug)]
pub struct OpenCLEvent {
    event: ocl::Event,
}

impl Drop for OpenCLTensor {
    fn drop(&mut self) {
        if self.initial_write_event.is_some() {
            self.initial_write_event
                .as_ref()
                .unwrap()
                .wait_for()
                .unwrap();
        }
        self.initial_write_event = None;
        if !self.data.is_null() {
            unsafe {
                std::alloc::dealloc(self.data as *mut u8, self.data_layout);
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum OpenCLError {
    #[error("OpenCL error: {0}")]
    OpenCL(#[from] ocl::Error),
    #[error("Cannot select device")]
    OpenCLDeviceSelection,
}

impl OpenCL {
    pub fn new(verbose: bool, nth_device: usize) -> Result<OpenCL, OpenCLError> {
        let platforms = Platform::list();
        let mut devices: Vec<(Platform, Device)> = Vec::new();
        for platform in platforms {
            for device in Device::list_all(platform)? {
                devices.push((platform, device));
            }
        }
        if verbose {
            println!("Enumerating OpenCL devices:");
        }
        for (idx, (_plat, device)) in devices.iter().enumerate() {
            if verbose {
                println!("OpenCL {} device: {}", idx, device.name()?,);
            }
        }
        if nth_device > devices.len() {
            return Err(OpenCLError::OpenCLDeviceSelection);
        }
        if verbose {
            println!("---");
            println!("Selected OpenCL device: {}", devices[nth_device].1.name()?);
        }

        let ctx = Context::builder()
            .platform(devices[nth_device].0)
            .devices(devices[nth_device].1)
            .build()?;

        let queue = Queue::new(&ctx, devices[nth_device].1, None)?;
        let programs = make_programs(&ctx, &queue)?;
        Ok(OpenCL {
            ctx: ctx,
            queue: queue,
            programs: Arc::new(RwLock::new(programs)),
        })
    }

    pub fn flush(&self) {
        let _ = self.queue.flush();
    }

    pub fn data_u16_to_gpu(
        &self,
        data: *const u16,
        data_layout: Layout,
        nitems: usize,
        rows: i64,
        cols: i64,
        cols_capacity: i64,
    ) -> Result<OpenCLTensor, OpenCLError> {
        unsafe {
            let buf = Buffer::builder()
                .queue(self.queue.clone())
                .len(nitems)
                .build()?;
            let mut event = Event::empty();
            let data_slice: &[u16] = std::slice::from_raw_parts(data, nitems);
            buf.cmd()
                .write(data_slice)
                .block(false)
                .enew(&mut event)
                .enq()?;
            Ok(OpenCLTensor {
                buf,
                initial_write_event: Some(event),
                last_event: None,
                data,
                data_layout,
                nitems,
                rows,
                cols,
                cols_capacity,
                queue: self.queue.clone(),
                programs: self.programs.clone(),
            })
        }
    }
}

impl OpenCLTensor {
    pub fn wait_until_ready(&mut self) {
        if self.last_event.is_some() {
            self.last_event.as_ref().unwrap().wait_for().unwrap();
            self.last_event = None;
        }
        if self.initial_write_event.is_some() {
            self.initial_write_event
                .as_ref()
                .unwrap()
                .wait_for()
                .unwrap();
            self.initial_write_event = None;
        }
        if !self.data.is_null() {
            unsafe {
                std::alloc::dealloc(self.data as *mut u8, self.data_layout);
            }
            self.data = std::ptr::null();
        }
    }

    pub fn data_u16_from_gpu(&mut self, data: *mut u16) -> Result<OpenCLEvent, OpenCLError> {
        unsafe {
            let mut event = Event::empty();
            let data_slice: &mut [u16] = std::slice::from_raw_parts_mut(data, self.nitems);
            let b = self
                .buf
                .cmd()
                .read(data_slice)
                .block(false)
                .enew(&mut event);
            b.enq()?;
            self.last_event = Some(event.clone());
            return Ok(OpenCLEvent { event });
        }
    }

    pub fn matrix_mul_inplace_transposed(
        &mut self,
        src: &OpenCLTensor,
        other: &OpenCLTensor,
    ) -> Result<OpenCLEvent, OpenCLError> {
        if src.cols != other.cols {
            panic!(
                "OpenCL matrix_mul_inplace_transposed: src.cols must equal other.cols: {}x{} vs {}x{}",
                src.rows, src.cols, other.rows, other.cols
            );
        }
        if self.rows != src.rows || self.cols != other.rows {
            panic!(
                "OpenCL matrix_mul_inplace_transposed: self.rows must equal src.rows and self.cols must equal other.cols: {}x{} vs {}x{} vs {}x{}",
                self.rows, self.cols, src.rows, src.cols, other.rows, other.cols
            );
        }

        // Clear out the target memory
        unsafe { self.buf.cmd().fill(0u16, None).block(false).enq()? };

        let prg = self.programs.write().unwrap();
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(0, self.buf.clone())?;
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(1, src.buf.clone())?;
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(2, other.buf.clone())?;
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(3, src.cols_capacity as i32)?;
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(4, other.cols_capacity as i32)?;
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(5, self.cols_capacity as i32)?;
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(6, self.rows as i32)?;
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(7, self.cols as i32)?;
        prg.matrix_mul_transposed_by_row_f16
            .set_arg(8, src.cols as i32)?;
        let mut event = Event::empty();

        unsafe {
            let b = prg
                .matrix_mul_transposed_by_row_f16
                .cmd()
                .queue(&self.queue)
                .global_work_size([self.rows as usize, self.cols_capacity as usize])
                .enew(&mut event);
            b.enq()?;
        }
        self.last_event = Some(event.clone());
        Ok(OpenCLEvent { event })
    }
}

impl OpenCLEvent {
    #[inline]
    pub fn wait(&self) {
        self.event.wait_for().unwrap();
    }
}

fn make_programs(ctx: &Context, queue: &Queue) -> Result<Programs, OpenCLError> {
    let mut last_err: Option<OpenCLError> = None;
    // There used to be more sources here but now it's just one. This can go through programs and
    // accept first one that compiles
    for src in &[MATRIX_MUL_TRANSPOSED_BY_ROW_F16_SRC] {
        fn make_programs_with_src(
            ctx: &Context,
            queue: &Queue,
            src: &str,
        ) -> Result<Programs, OpenCLError> {
            let program = Program::builder().src(src).build(&ctx)?;
            let kernel = Kernel::builder()
                .program(&program)
                .name("matrix_mul_transposed_by_row_f16")
                .arg(None::<&Buffer<u16>>)
                .arg(None::<&Buffer<u16>>)
                .arg(None::<&Buffer<u16>>)
                .arg(&0)
                .arg(&0)
                .arg(&0)
                .arg(&0)
                .arg(&0)
                .arg(&0)
                .queue(queue.clone())
                .build()?;
            Ok(Programs {
                matrix_mul_transposed_by_row_f16_program: program,
                matrix_mul_transposed_by_row_f16: kernel,
            })
        }
        match make_programs_with_src(ctx, queue, src) {
            Err(e) => {
                last_err = Some(e);
                continue;
            }
            Ok(p) => return Ok(p),
        }
    }
    if last_err.is_none() {
        unreachable!();
    }
    Err(last_err.unwrap())
}

const MATRIX_MUL_TRANSPOSED_BY_ROW_F16_SRC: &str = r#"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

/*
 * Matrix multiplication with a transposed second matrix, using 16-bit floats.
 *
 * One work unit per row.
 *
 * Assumes that each row in the matrices are zero-padded so that there's space for 32 bytes (or 16
 * halfs) of data and we don't need to care if our loops go over the bounds.
 *
 * Operations are done in float32.
 *
 * This thing is not very fast right now. I compared with PyTorch and this is like 20x slower. It
 * is still much faster than CPU. Not sure PyTorch uses cuBlas but if we could get at least
 * somewhere like 50% of that speed I would be happy.
 *
 * The OpenCL on CPU for Ryzen 3950X seems to easily beat my own AVX2 operations.
 *
 * TODO: need to read resources like https://cnugteren.github.io/tutorial/pages/page1.html to
 * figure out how matrix multiply faster.
 */
__kernel void matrix_mul_transposed_by_row_f16(
    __global half *tgt,
    __global const half *left,
    __global const half *right,
    const int left_cols_capacity,
    const int right_cols_capacity,
    const int ncols_capacity,
    const int nrows,
    const int ncols,  // size of target
    const int shared_sz
) {
    int col_iterations = shared_sz / 16;
    if (shared_sz % 16 != 0) {
      col_iterations = col_iterations + 1;
    }

    const int tgt_row = get_global_id(0);
    const int tgt_col = get_global_id(1);

    float16 sum = 0;
    for (int col16 = 0; col16 < col_iterations; col16++) {
        const float16 left8 = vload_half16((tgt_row * left_cols_capacity)/16 + col16, (__global const half*) left);
        const float16 right8 = vload_half16((tgt_col * right_cols_capacity)/16 + col16, (__global const half*) right);
        // hadamard product FMA add it to sum
        // const float16 result8 = left8 * right8;
        // sum += result8;
        sum = fma(left8, right8, sum);
    }
    // Reduce as accurately as possible
    float sum1 = sum.s0 + sum.s1;
    float sum2 = sum.s2 + sum.s3;
    float sum3 = sum.s4 + sum.s5;
    float sum4 = sum.s6 + sum.s7;
    float sum5 = sum.s8 + sum.s9;
    float sum6 = sum.sa + sum.sb;
    float sum7 = sum.sc + sum.sd;
    float sum8 = sum.se + sum.sf;

    float sum11 = sum1 + sum2;
    float sum12 = sum3 + sum4;
    float sum13 = sum5 + sum6;
    float sum14 = sum7 + sum8;

    float sum21 = sum11 + sum12;
    float sum22 = sum13 + sum14;

    float total = sum21 + sum22;

    vstore_half(total, 0, (__global half*) &tgt[tgt_row * ncols_capacity + tgt_col]);
}
"#;
