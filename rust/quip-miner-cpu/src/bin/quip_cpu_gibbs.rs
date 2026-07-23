//! CPU heat-bath Gibbs miner (`quip-cpu-gibbs`).

use clap::Parser;
use quip_miner_core::{run, CommonArgs};
use quip_miner_cpu::{Algorithm, CpuSampler, CPU_GIBBS_IDENTITY};
use std::process::ExitCode;

#[derive(Parser)]
#[command(version = concat!(env!("CARGO_PKG_VERSION"), " protocol 1"))]
struct Cli {
    #[command(flatten)]
    common: CommonArgs,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    run(CPU_GIBBS_IDENTITY, &cli.common, || {
        Ok(CpuSampler::new(Algorithm::Gibbs))
    })
}
