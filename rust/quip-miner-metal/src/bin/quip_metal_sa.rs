//! Metal simulated-annealing miner (`quip-metal-sa`).

use quip_miner_metal::sampler::Algorithm;
use quip_miner_metal::session::{run_cli, AlgorithmIdentity};
use std::process::ExitCode;

fn main() -> ExitCode {
    run_cli(AlgorithmIdentity {
        algorithm: "sa",
        sampler: Algorithm::Sa,
    })
}
