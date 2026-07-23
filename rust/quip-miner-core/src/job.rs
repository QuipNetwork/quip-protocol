//! Job validation and dispatch: wire fields → [`IsingGraph`] → [`Sampler`] →
//! `Result`/`Reject`.

use crate::ising::{IsingGraph, SampleParams};
use crate::session::BackendIdentity;
use crate::Sampler;
use quip_proto::v1::{
    ising_problem, miner_msg, IsingProblem, Job, JobKind, JobRequest, MinerMsg, Reject,
    RejectReason, Result as JobResult, SamplerMeta, Solution, Status,
};
use quip_protocol::wire::{decode_i32_le, encode_spins, WireError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(crate) const DEFAULT_NUM_SWEEPS: usize = 64;

pub(crate) fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(crate) fn miner(msg: miner_msg::Msg) -> MinerMsg {
    MinerMsg { msg: Some(msg) }
}

pub(crate) fn status_msg(miner_id: &str, jobs_done: u64, utilization: f64) -> MinerMsg {
    miner(miner_msg::Msg::Status(Status {
        miner_id: miner_id.into(),
        utilization,
        jobs_done,
        abandoned_generation: 0,
        sampler_stats: Default::default(),
    }))
}

pub(crate) fn reject(job_id: Vec<u8>, reason: RejectReason) -> MinerMsg {
    miner(miner_msg::Msg::Reject(Reject {
        job_id,
        reason: reason as i32,
    }))
}

/// Milli-int-encoded field → float vector.
fn decode_milli_f64(bytes: &[u8]) -> Result<Vec<f64>, WireError> {
    Ok(decode_i32_le(bytes)?
        .iter()
        .map(|&v| v as f64 / 1000.0)
        .collect())
}

fn edges_of(ising: &IsingProblem) -> Result<Vec<(usize, usize)>, RejectReason> {
    match &ising.graph {
        Some(ising_problem::Graph::Edges(e)) => {
            if e.u.len() != e.v.len() {
                return Err(RejectReason::Malformed);
            }
            Ok(e.u
                .iter()
                .zip(&e.v)
                .map(|(&u, &v)| (u as usize, v as usize))
                .collect())
        }
        // Topology-hash graphs are not resolved by a miner alone.
        Some(ising_problem::Graph::TopologyHash(_)) => Err(RejectReason::Malformed),
        None => Ok(Vec::new()),
    }
}

/// Validate wire fields and build the base Ising graph, or a reject reason.
fn parse_ising(
    ising: &IsingProblem,
    max_nodes: u32,
    max_edges: u32,
) -> Result<IsingGraph, RejectReason> {
    let h = decode_milli_f64(&ising.h_milli_le32).map_err(|_| RejectReason::Malformed)?;
    let j = decode_milli_f64(&ising.j_milli_le32).map_err(|_| RejectReason::Malformed)?;
    let edges = edges_of(ising)?;
    if !edges.is_empty() && j.len() != edges.len() {
        return Err(RejectReason::Malformed);
    }
    let n = h.len();
    if n > max_nodes as usize || edges.len() > max_edges as usize {
        return Err(RejectReason::TooLarge);
    }
    for &(u, v) in &edges {
        if u >= n || v >= n {
            return Err(RejectReason::Malformed);
        }
    }
    Ok(IsingGraph::new(h, j, edges))
}

/// Best-effort parse of `num_sweeps = N` from `Configure.backend_toml`.
pub(crate) fn num_sweeps_from_toml(backend_toml: &str) -> usize {
    for line in backend_toml.lines() {
        let line = line.split('#').next().unwrap_or(line).trim();
        if let Some(rest) = line.strip_prefix("num_sweeps") {
            let rest = rest.trim().trim_start_matches('=').trim();
            if let Ok(n) = rest.parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    DEFAULT_NUM_SWEEPS
}

/// Handle one job: validate, sample, return `Result` + `JobRequest` (or `Reject`).
pub(crate) fn handle_job<S: Sampler>(
    job: Job,
    sampler: &S,
    id: &BackendIdentity,
    num_sweeps: usize,
    jobs_done: &mut u64,
) -> Vec<MinerMsg> {
    let job_id = job.job_id.clone();

    if job.kind != JobKind::IsingSample as i32 {
        return vec![reject(job_id, RejectReason::UnsupportedKind)];
    }
    if job.deadline_ms < now_unix_ms() {
        return vec![reject(job_id, RejectReason::Expired)];
    }
    let ising = match job.ising {
        Some(i) => i,
        None => return vec![reject(job_id, RejectReason::Malformed)],
    };

    if ising.num_reads > sampler.max_reads() {
        return vec![reject(job_id, RejectReason::TooLarge)];
    }

    let graph = match parse_ising(&ising, id.max_nodes, id.max_edges) {
        Ok(g) => g,
        Err(reason) => return vec![reject(job_id, reason)],
    };

    if sampler.should_throttle() {
        std::thread::sleep(Duration::from_millis(50));
    }

    let num_reads = if ising.num_reads == 0 {
        1
    } else {
        ising.num_reads as usize
    };
    let params = SampleParams {
        num_reads,
        num_sweeps,
        seed: now_unix_ms(),
        ..Default::default()
    };

    let t0 = Instant::now();
    let samples = match sampler.sample(&graph, &params) {
        Ok(s) => s,
        Err(reason) => return vec![reject(job_id, reason)],
    };
    let elapsed_us = t0.elapsed().as_micros() as u64;

    let solutions: Vec<Solution> = samples
        .into_iter()
        .map(|r| Solution {
            spins_bytes: encode_spins(&r.spins),
            energy_milli: r.energy_milli,
        })
        .collect();

    *jobs_done = jobs_done.saturating_add(1);

    let result = JobResult {
        job_id,
        solutions,
        meta: Some(SamplerMeta {
            reads: num_reads as u32,
            sweeps: num_sweeps as u32,
            device_access_time_us: elapsed_us,
            qpu_access_us: 0,
            extra: Default::default(),
        }),
    };

    vec![
        miner(miner_msg::Msg::Result(result)),
        miner(miner_msg::Msg::JobRequest(JobRequest { credits: 1 })),
    ]
}
