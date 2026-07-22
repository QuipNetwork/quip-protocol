//! Metal heat-bath Gibbs miner (`quip-metal-gibbs`).

use quip_miner_metal::sampler::Algorithm;
use quip_miner_metal::session::{run_cli, AlgorithmIdentity};
use std::process::ExitCode;

fn main() -> ExitCode {
    run_cli(AlgorithmIdentity {
        algorithm: "gibbs",
        sampler: Algorithm::Gibbs,
    })
}
