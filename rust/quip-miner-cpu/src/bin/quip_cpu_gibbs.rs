//! CPU heat-bath Gibbs miner (`quip-cpu-gibbs`).

use quip_miner_cpu::sampler_core::Algorithm;
use quip_miner_cpu::session::{run_cli, AlgorithmIdentity};
use std::process::ExitCode;

fn main() -> ExitCode {
    run_cli(AlgorithmIdentity {
        algorithm: "gibbs",
        sampler: Algorithm::Gibbs,
    })
}
