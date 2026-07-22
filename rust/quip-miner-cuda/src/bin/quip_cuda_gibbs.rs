//! CUDA heat-bath Gibbs miner (`quip-cuda-gibbs`).

use quip_miner_cuda::sampler::Algorithm;
use quip_miner_cuda::session::{run_cli, AlgorithmIdentity};
use std::process::ExitCode;

fn main() -> ExitCode {
    run_cli(AlgorithmIdentity {
        algorithm: "gibbs",
        sampler: Algorithm::Gibbs,
    })
}
