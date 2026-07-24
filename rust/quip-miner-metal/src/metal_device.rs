//! Metal device + runtime-compiled SA/Gibbs pipelines for one Apple GPU.
//!
//! One process owns one device (`[metal.N]` → device N / miner id `metal-N`).
//! Kernels are JIT-compiled from `.metal` source via
//! `new_library_with_source` (analogous to NVRTC in the CUDA crate).
//!
//! # Send / Sync
//!
//! The `metal` crate's owned wrapper types (`Device`, `CommandQueue`,
//! `ComputePipelineState`, `Buffer`, ...) are declared `Send + Sync` by the
//! crate itself (`foreign_type! { pub unsafe type ...: Sync + Send { ... } }`
//! in `metal::lib`), so `Arc<MetalDevice>` would in fact compile. This crate
//! still owns `MetalDevice` directly (no `Arc`) and drives `sample_ising`
//! synchronously inside `run_session`'s single `block_on` future — matching
//! the CUDA crate's one-thread-per-job model and keeping every GPU call on
//! the same OS thread for the life of the process, which is the simpler and
//! more conservative choice given Apple's own guidance that `MTLDevice` /
//! `MTLCommandQueue` usage from multiple threads needs care.

use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions};
use thiserror::Error;

const SA_SRC: &str = include_str!("../kernels/sa.metal");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.metal");

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Metal driver: {0}")]
    Driver(String),
    #[error("Metal compile: {0}")]
    Compile(String),
    #[error("no Metal device at index {0}")]
    NoDevice(usize),
}

/// Loaded pipelines + queue bound to a single device.
///
/// Kept on one OS thread by convention, not by type constraint — see the
/// module-level Send / Sync note above.
pub struct MetalDevice {
    pub device_index: usize,
    pub device: Device,
    pub queue: metal::CommandQueue,
    pub sa: ComputePipelineState,
    pub gibbs: ComputePipelineState,
    /// Chromatic (node-parallel) Gibbs: one threadgroup per sample, threads
    /// split the nodes of each color, `threadgroup`-shared state. Same buffer
    /// layout as `gibbs`, different dispatch geometry.
    pub gibbs_parallel: ComputePipelineState,
}

impl MetalDevice {
    /// Open device `device_index` and compile both SA and Gibbs kernels.
    ///
    /// Indexing: `Device::all()` order. Index 0 is typically the system
    /// default (Apple Silicon integrated GPU). Higher indices map into
    /// `all()` when multiple Metal devices are present.
    pub fn open(device_index: usize) -> Result<Self, MetalError> {
        let devices = Device::all();
        if devices.is_empty() {
            return Err(MetalError::NoDevice(device_index));
        }
        let device = devices
            .into_iter()
            .nth(device_index)
            .ok_or(MetalError::NoDevice(device_index))?;

        let sa = compile_pipeline(&device, SA_SRC, "pure_simulated_annealing")?;
        let gibbs = compile_pipeline(&device, GIBBS_SRC, "block_gibbs_sampler")?;
        let gibbs_parallel = compile_pipeline(&device, GIBBS_SRC, "block_gibbs_parallel")?;
        let queue = device.new_command_queue();

        Ok(Self {
            device_index,
            device,
            queue,
            sa,
            gibbs,
            gibbs_parallel,
        })
    }

    /// Number of Metal devices visible to this process.
    pub fn device_count() -> Result<usize, MetalError> {
        Ok(Device::all().len())
    }

    /// Probe that a device can open and compile kernels (`--check`).
    pub fn check(device_index: usize) -> Result<(), MetalError> {
        let _ = Self::open(device_index)?;
        Ok(())
    }

    /// Shared-storage buffer of `len` zeroed bytes.
    pub fn new_zeroed_buffer(&self, len: u64) -> metal::Buffer {
        self.device
            .new_buffer(len, MTLResourceOptions::StorageModeShared)
    }

    /// Shared-storage buffer filled from host slice bytes.
    pub fn new_buffer_from_slice<T: Copy>(&self, data: &[T]) -> metal::Buffer {
        let byte_len = std::mem::size_of_val(data) as u64;
        if byte_len == 0 {
            // Metal rejects zero-length buffers; allocate a tiny stub.
            return self.new_zeroed_buffer(4);
        }
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

fn compile_pipeline(
    device: &Device,
    source: &str,
    entry: &str,
) -> Result<ComputePipelineState, MetalError> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
        .map_err(|e| MetalError::Compile(format!("{entry}: {e}")))?;
    let function = library
        .get_function(entry, None)
        .map_err(|e| MetalError::Compile(format!("{entry} function: {e}")))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| MetalError::Driver(format!("{entry} pipeline: {e}")))
}
