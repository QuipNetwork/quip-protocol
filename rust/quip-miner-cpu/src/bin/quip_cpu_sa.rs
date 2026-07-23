//! CPU simulated-annealing miner (`quip-cpu-sa`).

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{Algorithm, CpuSampler, CPU_SA_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_SA_IDENTITY, &cli.common, || {
        Ok(CpuSampler::new(Algorithm::Sa))
    })
}
