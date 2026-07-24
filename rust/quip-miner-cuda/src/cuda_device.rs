//! CUDA context + NVRTC-compiled self-feeding kernels for one physical GPU.
//!
//! One process owns one device (`[cuda.N]` → device N / miner id `cuda-N`).

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use std::sync::Arc;
use thiserror::Error;

const SA_SRC: &str = include_str!("../kernels/sa.cu");
const GIBBS_SRC: &str = include_str!("../kernels/gibbs.cu");

/// Minimum CUDA driver version (`cuDriverGetVersion` encoding: `major*1000 +
/// minor*10`) NVRTC needs to natively target each GPU arch. Port of
/// `GPU/base_cuda_sampler.py::_CUDA_ARCH_MIN_VERSION`.
const CUDA_ARCH_MIN_VERSION: &[(i32, i32)] = &[
    (121, 12090),
    (120, 12080),
    (103, 12090),
    (101, 12080),
    (100, 12080),
    (90, 12000),
    (89, 11080),
    (86, 11010),
    (80, 11000),
];

/// Highest GPU arch the given driver version supports. Port of
/// `_best_fallback_arch`.
fn best_fallback_arch(driver_version: i32) -> i32 {
    CUDA_ARCH_MIN_VERSION
        .iter()
        .filter(|&&(_, min)| min <= driver_version)
        .map(|&(arch, _)| arch)
        .max()
        .unwrap_or(80)
}

/// `cuDriverGetVersion`, wrapped safely (cudarc exposes only the raw sys fn).
fn driver_version() -> Result<i32, CudaError> {
    let mut v: std::ffi::c_int = 0;
    unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut v) }
        .result()
        .map_err(|e| CudaError::Driver(e.to_string()))?;
    Ok(v)
}

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

/// Compile CUDA source with NVRTC, retrying with an explicit
/// `--gpu-architecture=compute_N` PTX fallback if the default (portable,
/// arch-unspecified) compile fails — e.g. a GPU newer than this NVRTC knows
/// natively. The fallback arch is the highest one the installed driver
/// supports; the driver JIT-compiles that PTX up to the real SM at module
/// load time. Port of `_compile_module`.
fn compile_with_fallback(src: &str) -> Result<Ptx, CudaError> {
    let base = CompileOptions {
        use_fast_math: Some(true),
        ..Default::default()
    };
    match compile_ptx_with_opts(src, base.clone()) {
        Ok(ptx) => Ok(ptx),
        Err(first_err) => {
            let ver = driver_version()?;
            let fb = best_fallback_arch(ver);
            let opts = CompileOptions {
                use_fast_math: Some(true),
                options: vec![format!("--gpu-architecture=compute_{fb}")],
                ..Default::default()
            };
            compile_ptx_with_opts(src, opts).map_err(|e| {
                CudaError::Compile(format!(
                    "default compile failed ({first_err}); compute_{fb} fallback also failed: {e}"
                ))
            })
        }
    }
}

/// Loaded kernels + streams bound to a single device.
pub struct CudaDevice {
    pub device_index: usize,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// `cuda_sa_self_feeding` — persistent kernel, 1 block (1 SM) per nonce.
    pub sa: CudaFunction,
    /// `cuda_gibbs_self_feeding` — persistent kernel, `sms_per_nonce` blocks
    /// per nonce.
    pub gibbs: CudaFunction,
    /// SMs on this device (`launch_self_feeding`'s `num_kernels` budget).
    pub max_sms: usize,
    _sa_mod: Arc<CudaModule>,
    _gibbs_mod: Arc<CudaModule>,
}

impl CudaDevice {
    /// Create a context on `device_index` and NVRTC-compile the kernels.
    pub fn open(device_index: usize) -> Result<Self, CudaError> {
        let n = CudaContext::device_count().map_err(CudaError::from)? as usize;
        if device_index >= n {
            return Err(CudaError::NoDevice(device_index));
        }
        let ctx = CudaContext::new(device_index).map_err(CudaError::from)?;

        // The self-feeding streaming session runs a persistent kernel on one
        // stream while a second stream concurrently uploads/downloads slot
        // data the kernel is still reading/writing (by design: the kernel's
        // own volatile ctrl protocol + __threadfence calls are the
        // synchronization, matching the reference CuPy driver's raw async
        // streams). cudarc's default per-CudaSlice read/write event
        // tracking would instead insert a wait for the (never-until-exit
        // signaled) kernel completion event on the transfer stream, which
        // would deadlock the self-feeding protocol. Safety: every buffer the
        // persistent kernel touches is torn down only after `signal_exit` +
        // `stream_compute.synchronize()` (see `streaming::SelfFeedingSession`
        // drop), so no CudaSlice is freed while still in use.
        unsafe { ctx.disable_event_tracking() };

        let stream = ctx.default_stream();

        let max_sms = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(CudaError::from)? as usize;

        let sa_ptx = compile_with_fallback(SA_SRC)?;
        let gibbs_ptx = compile_with_fallback(GIBBS_SRC)?;

        let sa_mod = ctx.load_module(sa_ptx).map_err(CudaError::from)?;
        let gibbs_mod = ctx.load_module(gibbs_ptx).map_err(CudaError::from)?;

        let sa = sa_mod
            .load_function("cuda_sa_self_feeding")
            .map_err(CudaError::from)?;
        let gibbs = gibbs_mod
            .load_function("cuda_gibbs_self_feeding")
            .map_err(CudaError::from)?;

        Ok(Self {
            device_index,
            ctx,
            stream,
            sa,
            gibbs,
            max_sms: max_sms.max(1),
            _sa_mod: sa_mod,
            _gibbs_mod: gibbs_mod,
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
