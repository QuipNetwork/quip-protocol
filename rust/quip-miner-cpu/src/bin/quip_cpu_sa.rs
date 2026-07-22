//! CPU simulated-annealing miner (`quip-cpu-sa`).

use quip_miner_cpu::sampler_core::Algorithm;
use quip_miner_cpu::session::{run_cli, AlgorithmIdentity};
use std::process::ExitCode;

fn main() -> ExitCode {
    run_cli(AlgorithmIdentity {
        algorithm: "sa",
        sampler: Algorithm::Sa,
    })
}
