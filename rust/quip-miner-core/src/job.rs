//! Job validation and dispatch: wire fields → [`IsingGraph`] → [`Sampler`] →
//! `Result`/`Reject`.

use crate::ising::{IsingGraph, SampleParams};
use crate::session::BackendIdentity;
use crate::Sampler;
use quip_proto::v1::{
    ising_problem, miner_msg, IsingProblem, Job, JobKind, JobRequest, MinerMsg, Reject,
    RejectReason, Result as JobResult, SamplerMeta, Solution, Status, Topology,
};
use quip_protocol::wire::{decode_i32_le, encode_spins, WireError};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Session-cached topology, used to resolve `TopologyHash` jobs to dense edges.
///
/// Built once from the `Topology` message at Configure. `pos` maps each native
/// node id to its dense position (id → index in received order), so sparse
/// D-Wave qubit ids resolve without a per-job remap.
pub(crate) struct TopologyCache {
    hash: Vec<u8>,
    edges: Vec<(u32, u32)>,
    pos: HashMap<u32, usize>,
}

impl TopologyCache {
    /// Build from a `Topology` message. Node ids map to their received-order
    /// position; edges keep received order (the consensus `j`-zip invariant).
    pub(crate) fn from_proto(t: &Topology) -> Self {
        let mut pos = HashMap::with_capacity(t.nodes.len());
        for (i, &node) in t.nodes.iter().enumerate() {
            let _ = pos.insert(node, i);
        }
        let edges = t.edges.as_ref().map_or_else(Vec::new, |e| {
            e.u.iter().zip(&e.v).map(|(&u, &v)| (u, v)).collect()
        });
        Self {
            hash: t.hash.clone(),
            edges,
            pos,
        }
    }
}

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

fn resolve_edges(
    ising: &IsingProblem,
    cache: Option<&TopologyCache>,
) -> Result<Vec<(usize, usize)>, RejectReason> {
    match &ising.graph {
        // Inline edges use dense 0..n-1 ids straight off the wire.
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
        // Topology-hash jobs resolve against the session-cached topology,
        // mapping native (possibly sparse) node ids to dense positions.
        Some(ising_problem::Graph::TopologyHash(h)) => {
            let cache = cache.ok_or(RejectReason::TopologyMissing)?;
            if *h != cache.hash {
                return Err(RejectReason::TopologyMismatch);
            }
            let mut out = Vec::with_capacity(cache.edges.len());
            for &(u, v) in &cache.edges {
                let pu = *cache.pos.get(&u).ok_or(RejectReason::Malformed)?;
                let pv = *cache.pos.get(&v).ok_or(RejectReason::Malformed)?;
                out.push((pu, pv));
            }
            Ok(out)
        }
        None => Ok(Vec::new()),
    }
}

/// Validate wire fields and build the base Ising graph, or a reject reason.
fn parse_ising(
    ising: &IsingProblem,
    max_nodes: u32,
    max_edges: u32,
    cache: Option<&TopologyCache>,
) -> Result<IsingGraph, RejectReason> {
    let h = decode_milli_f64(&ising.h_milli_le32).map_err(|_| RejectReason::Malformed)?;
    let j = decode_milli_f64(&ising.j_milli_le32).map_err(|_| RejectReason::Malformed)?;
    let edges = resolve_edges(ising, cache)?;
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
    cache: Option<&TopologyCache>,
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

    let graph = match parse_ising(&ising, id.max_nodes, id.max_edges, cache) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::{ising_problem::Graph, EdgeList, Topology};
    use quip_protocol::wire::encode_i32_le;

    fn hash_job(hash: Vec<u8>, h_milli: &[i32], j_milli: &[i32]) -> IsingProblem {
        IsingProblem {
            graph: Some(Graph::TopologyHash(hash)),
            h_milli_le32: encode_i32_le(h_milli),
            j_milli_le32: encode_i32_le(j_milli),
            num_reads: 0,
            gates: None,
        }
    }

    #[test]
    fn topology_cache_maps_sparse_ids_to_positions() {
        let topo = Topology {
            hash: vec![0xAB; 32],
            nodes: vec![0, 12, 2400],
            edges: Some(EdgeList {
                u: vec![0, 12],
                v: vec![12, 2400],
            }),
        };
        let c = TopologyCache::from_proto(&topo);
        assert_eq!(c.pos.get(&0), Some(&0));
        assert_eq!(c.pos.get(&12), Some(&1));
        assert_eq!(c.pos.get(&2400), Some(&2));
        // received order, unsorted
        assert_eq!(c.edges, vec![(0, 12), (12, 2400)]);
        assert_eq!(c.hash, vec![0xAB; 32]);
    }

    #[test]
    fn hash_job_resolves_sparse_edges_to_positions() {
        let topo = Topology {
            hash: vec![7; 32],
            nodes: vec![0, 12, 2400],
            edges: Some(EdgeList {
                u: vec![0, 12],
                v: vec![12, 2400],
            }),
        };
        let cache = TopologyCache::from_proto(&topo);
        let ising = hash_job(vec![7; 32], &[1000, -1000, 1000], &[1000, -1000]);
        let g = parse_ising(&ising, 100_000, 1_000_000, Some(&cache)).unwrap();
        assert_eq!(g.edges, vec![(0, 1), (1, 2)]);
        assert_eq!(g.h, vec![1.0, -1.0, 1.0]);
        assert_eq!(g.j, vec![1.0, -1.0]);
    }

    #[test]
    fn hash_job_without_cache_rejects_missing() {
        let ising = hash_job(vec![7; 32], &[1000], &[]);
        let err = parse_ising(&ising, 100_000, 1_000_000, None).unwrap_err();
        assert_eq!(err, RejectReason::TopologyMissing);
    }

    #[test]
    fn hash_job_wrong_hash_rejects_mismatch() {
        let topo = Topology {
            hash: vec![1; 32],
            nodes: vec![0],
            edges: None,
        };
        let cache = TopologyCache::from_proto(&topo);
        let ising = hash_job(vec![2; 32], &[1000], &[]);
        let err = parse_ising(&ising, 100_000, 1_000_000, Some(&cache)).unwrap_err();
        assert_eq!(err, RejectReason::TopologyMismatch);
    }

    #[test]
    fn hash_job_edge_id_absent_from_map_rejects_malformed() {
        // edge references id 999, which is not in nodes → no position.
        let topo = Topology {
            hash: vec![7; 32],
            nodes: vec![0, 1],
            edges: Some(EdgeList {
                u: vec![0],
                v: vec![999],
            }),
        };
        let cache = TopologyCache::from_proto(&topo);
        let ising = hash_job(vec![7; 32], &[1000, 1000], &[1000]);
        let err = parse_ising(&ising, 100_000, 1_000_000, Some(&cache)).unwrap_err();
        assert_eq!(err, RejectReason::Malformed);
    }
}
