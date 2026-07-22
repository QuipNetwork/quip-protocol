//! CUDA context + NVRTC-compiled kernels for one physical GPU.
//!
//! One process owns one device (`[cuda.N]` → device N / miner id `cuda-N`).

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use thiserror::Error;

const SA_SRC: &str = include_str!("../kernels/sa.cu");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.cu");
const ENERGY_SRC: &str = include_str!("../kernels/energy.cu");

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA driver: {0}")]
    Driver(String),
    #[error("NVRTC compile: {0}")]
    Compile(String),
    #[error("no CUDA device at index {0}")]
    NoDevice(usize),
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        CudaError::Driver(e.to_string())
    }
}

/// Loaded kernels + stream bound to a single device.
pub struct CudaDevice {
    pub device_index: usize,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub sa: CudaFunction,
    pub gibbs: CudaFunction,
    pub energy: CudaFunction,
    _sa_mod: Arc<CudaModule>,
    _gibbs_mod: Arc<CudaModule>,
    _energy_mod: Arc<CudaModule>,
}

impl CudaDevice {
    /// Create a context on `device_index` and NVRTC-compile the three kernels.
    pub fn open(device_index: usize) -> Result<Self, CudaError> {
        let n = CudaContext::device_count().map_err(CudaError::from)? as usize;
        if device_index >= n {
            return Err(CudaError::NoDevice(device_index));
        }
        let ctx = CudaContext::new(device_index).map_err(CudaError::from)?;
        let stream = ctx.default_stream();

        let sa_ptx = compile_ptx(SA_SRC).map_err(|e| CudaError::Compile(e.to_string()))?;
        let gibbs_ptx = compile_ptx(GIBBS_SRC).map_err(|e| CudaError::Compile(e.to_string()))?;
        let energy_ptx = compile_ptx(ENERGY_SRC).map_err(|e| CudaError::Compile(e.to_string()))?;

        let sa_mod = ctx.load_module(sa_ptx).map_err(CudaError::from)?;
        let gibbs_mod = ctx.load_module(gibbs_ptx).map_err(CudaError::from)?;
        let energy_mod = ctx.load_module(energy_ptx).map_err(CudaError::from)?;

        let sa = sa_mod
            .load_function("cuda_sa_sample")
            .map_err(CudaError::from)?;
        let gibbs = gibbs_mod
            .load_function("cuda_gibbs_sample")
            .map_err(CudaError::from)?;
        let energy = energy_mod
            .load_function("cuda_energy_milli")
            .map_err(CudaError::from)?;

        Ok(Self {
            device_index,
            ctx,
            stream,
            sa,
            gibbs,
            energy,
            _sa_mod: sa_mod,
            _gibbs_mod: gibbs_mod,
            _energy_mod: energy_mod,
        })
    }

    /// Number of CUDA devices visible to this process.
    pub fn device_count() -> Result<usize, CudaError> {
        Ok(CudaContext::device_count().map_err(CudaError::from)? as usize)
    }

    /// Probe that a device can open and compile kernels (`--check`).
    pub fn check(device_index: usize) -> Result<(), CudaError> {
        let _ = Self::open(device_index)?;
        Ok(())
    }
}
