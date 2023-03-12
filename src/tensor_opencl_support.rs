/*
 * OpenCL stuff to run (some) of the tensor operations.
 */

use ocl::{Buffer, Context, Device, Event, Platform, Queue};
use std::alloc::Layout;
use thiserror::Error;

#[derive(Debug)]
pub struct OpenCL {
    ctx: Context,
    queue: Queue,
}

#[derive(Debug)]
pub struct OpenCLTensor {
    buf: Buffer<u16>,                // really is f16
    write_event: Option<ocl::Event>, // if Some, the buffer is being written to
    data: *const u16,                // if non-null, is host pointer that should be freed
    data_layout: Layout,
}

impl Drop for OpenCLTensor {
    fn drop(&mut self) {
        if !self.data.is_null() {
            if self.write_event.is_some() {
                self.write_event.as_ref().unwrap().wait_for().unwrap();
            }
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
        for (idx, (_, device)) in devices.iter().enumerate() {
            if verbose {
                println!("OpenCL {} device: {}", idx, device.name()?);
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

        Ok(OpenCL { ctx, queue })
    }

    pub fn data_u16_to_gpu(
        &self,
        data: *const u16,
        data_layout: Layout,
        nitems: usize,
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
                write_event: Some(event),
                data,
                data_layout,
            })
        }
    }
}

impl OpenCLTensor {
    pub fn wait_until_ready(&mut self) {
        if self.write_event.is_none() {
            return;
        }
        self.write_event.as_ref().unwrap().wait_for().unwrap();
        self.write_event = None;
        if !self.data.is_null() {
            if self.write_event.is_some() {
                self.write_event.as_ref().unwrap().wait_for().unwrap();
            }
            unsafe {
                std::alloc::dealloc(self.data as *mut u8, self.data_layout);
            }
        }
    }
}
