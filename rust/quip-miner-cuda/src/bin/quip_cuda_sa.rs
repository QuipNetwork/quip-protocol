//! CUDA simulated-annealing miner (`quip-cuda-sa`).
//!
//! One process per GPU: `--device N` binds CUDA device N and defaults
//! `--miner-id` to `cuda-N` (matching `[cuda.N]` config sections).

use clap::Parser;
use quip_miner_core::{run, CommonArgs, OpenError};
use quip_miner_cuda::cuda_device::CudaDevice;
use quip_miner_cuda::nvml_gov::UtilGovernor;
use quip_miner_cuda::{Algorithm, CudaSampler, CUDA_SA_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
    /// CUDA device index (one process per GPU). Default 0 → miner id `cuda-0`.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Target GPU utilization ceiling percent (1–100). Used by NVML governor.
    #[arg(long, default_value_t = 100)]
    utilization: u32,
    /// Yield to other GPU users when NVML util exceeds 90%.
    #[arg(long, default_value_t = false)]
    yielding: bool,
}

fn main() -> ExitCode {
    let mut cli = Cli::parse();
    if cli.common.miner_id.is_none() {
        cli.common.miner_id = Some(format!("cuda-{}", cli.device));
    }
    run(CUDA_SA_IDENTITY, &cli.common, || {
        let device = CudaDevice::open(cli.device)
            .map_err(|e| OpenError(format!("device {}: {e}", cli.device)))?;
        let gov = UtilGovernor::start(cli.device as u32, cli.utilization, cli.yielding);
        Ok(CudaSampler::new(device, gov, Algorithm::Sa))
    })
}
