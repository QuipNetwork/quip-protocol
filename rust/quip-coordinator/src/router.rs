//! Capability index, per-miner staged queue, credit accounting, cancel, reject.

use quip_proto::v1::{Job, RejectReason};
use std::collections::{HashMap, HashSet, VecDeque};

/// Capability envelope advertised in `Hello`.
#[derive(Debug, Clone)]
pub struct MinerCaps {
    pub backend: String,
    pub algorithm: String,
    pub supported_kinds: Vec<i32>,
    pub max_nodes: u32,
    pub max_edges: u32,
}

#[derive(Debug)]
struct MinerQueue {
    caps: MinerCaps,
    staged: VecDeque<Job>,
    /// Credits granted by the miner via `JobRequest` (cumulative capacity).
    granted_credits: u32,
    /// Jobs dispatched and not yet acked by Result/Reject.
    outstanding: u32,
    unsupported_kinds: HashSet<i32>,
}

/// Routes jobs to miners by capability, stages them, and gates dispatch on credits.
#[derive(Debug, Default)]
pub struct Router {
    miners: HashMap<String, MinerQueue>,
    /// Jobs that could not be routed (no capable miner).
    pub unroutable: Vec<Job>,
}

impl Router {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_miner(&mut self, miner_id: impl Into<String>, caps: MinerCaps) {
        let id = miner_id.into();
        self.miners.insert(
            id,
            MinerQueue {
                caps,
                staged: VecDeque::new(),
                granted_credits: 0,
                outstanding: 0,
                unsupported_kinds: HashSet::new(),
            },
        );
    }

    /// Stage `job` on a capable miner; returns the chosen `miner_id`, or `None`.
    pub fn route(&mut self, job: Job) -> Option<String> {
        let n_nodes = job_node_count(&job);
        let n_edges = job_edge_count(&job);
        let kind = job.kind;

        let mut candidates: Vec<&String> = self
            .miners
            .iter()
            .filter(|(_, q)| capable(q, kind, n_nodes, n_edges))
            .map(|(id, _)| id)
            .collect();
        candidates.sort(); // deterministic pick

        if let Some(id) = candidates.first().copied() {
            let id = id.clone();
            if let Some(q) = self.miners.get_mut(&id) {
                q.staged.push_back(job);
            }
            Some(id)
        } else {
            self.unroutable.push(job);
            None
        }
    }

    pub fn grant_credits(&mut self, miner_id: &str, credits: u32) {
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.granted_credits = q.granted_credits.saturating_add(credits);
        }
    }

    /// Pop the next staged job if credits allow (`outstanding < granted`).
    pub fn next_job(&mut self, miner_id: &str) -> Option<Job> {
        let q = self.miners.get_mut(miner_id)?;
        if q.outstanding >= q.granted_credits {
            return None;
        }
        let job = q.staged.pop_front()?;
        q.outstanding += 1;
        Some(job)
    }

    /// Decrement outstanding on Result/Reject.
    pub fn ack(&mut self, miner_id: &str) {
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.outstanding = q.outstanding.saturating_sub(1);
        }
    }

    /// Handle a miner reject: mark unsupported kinds, re-route when possible.
    pub fn on_reject(&mut self, miner_id: &str, job: Job, reason: i32) {
        self.ack(miner_id);
        if reason == RejectReason::UnsupportedKind as i32 {
            if let Some(q) = self.miners.get_mut(miner_id) {
                q.unsupported_kinds.insert(job.kind);
            }
        }
        // Re-route to another miner if possible.
        let _ = self.route(job);
    }

    /// Drop staged PoW jobs with `0 < generation <= max_generation`.
    /// Mempool jobs (`generation == 0`) are preserved.
    pub fn cancel(&mut self, max_generation: u64) {
        for q in self.miners.values_mut() {
            q.staged
                .retain(|j| j.generation == 0 || j.generation > max_generation);
        }
    }

    /// Return all outstanding + staged jobs for a miner (e.g. on crash re-queue).
    pub fn reclaim(&mut self, miner_id: &str) -> Vec<Job> {
        if let Some(q) = self.miners.get_mut(miner_id) {
            q.outstanding = 0;
            q.granted_credits = 0;
            return q.staged.drain(..).collect();
        }
        Vec::new()
    }

    pub fn caps(&self, miner_id: &str) -> Option<&MinerCaps> {
        self.miners.get(miner_id).map(|q| &q.caps)
    }

    pub fn staged_len(&self, miner_id: &str) -> usize {
        self.miners
            .get(miner_id)
            .map(|q| q.staged.len())
            .unwrap_or(0)
    }
}

fn capable(q: &MinerQueue, kind: i32, n_nodes: u32, n_edges: u32) -> bool {
    if q.unsupported_kinds.contains(&kind) {
        return false;
    }
    if !q.caps.supported_kinds.is_empty() && !q.caps.supported_kinds.contains(&kind) {
        return false;
    }
    // max_nodes/max_edges of 0 means unlimited (Hello default when unset).
    if q.caps.max_nodes > 0 && n_nodes > q.caps.max_nodes {
        return false;
    }
    if q.caps.max_edges > 0 && n_edges > q.caps.max_edges {
        return false;
    }
    true
}

fn job_node_count(job: &Job) -> u32 {
    let Some(ising) = job.ising.as_ref() else {
        return 0;
    };
    // h field length / 4 = node count
    (ising.h_milli_le32.len() / 4) as u32
}

fn job_edge_count(job: &Job) -> u32 {
    let Some(ising) = job.ising.as_ref() else {
        return 0;
    };
    use quip_proto::v1::ising_problem;
    match &ising.graph {
        Some(ising_problem::Graph::Edges(e)) => e.u.len() as u32,
        _ => (ising.j_milli_le32.len() / 4) as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::{IsingProblem, JobKind, Provenance};

    fn caps_ising() -> MinerCaps {
        MinerCaps {
            backend: "mock".into(),
            algorithm: "sa".into(),
            supported_kinds: vec![JobKind::IsingSample as i32],
            max_nodes: 1000,
            max_edges: 10000,
        }
    }

    fn make_job(generation: u64, kind: JobKind) -> Job {
        Job {
            job_id: format!("g{generation}").into_bytes(),
            kind: kind as i32,
            generation,
            deadline_ms: 9_999_999,
            ising: Some(IsingProblem {
                graph: None,
                h_milli_le32: vec![0; 8], // 2 nodes
                j_milli_le32: vec![0; 4], // 1 edge
                num_reads: 0,
                num_sweeps: 0,
                anneal_time_us: 0,
                gates: None,
            }),
            provenance: Some(Provenance {
                is_pow: generation != 0,
                order_id: vec![],
            }),
        }
    }

    #[test]
    fn route_picks_capable_miner_and_stages() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let id = r.route(make_job(1, JobKind::IsingSample));
        assert_eq!(id.as_deref(), Some("cpu-0"));
        assert_eq!(r.staged_len("cpu-0"), 1);
    }

    #[test]
    fn next_job_respects_credits() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.route(make_job(1, JobKind::IsingSample));
        r.route(make_job(2, JobKind::IsingSample));
        assert!(r.next_job("cpu-0").is_none()); // no credits
        r.grant_credits("cpu-0", 1);
        assert!(r.next_job("cpu-0").is_some());
        assert!(r.next_job("cpu-0").is_none()); // outstanding == granted
        r.ack("cpu-0");
        // still only 1 credit total, already used once → still None until re-grant
        // Actually: outstanding=0, granted=1, staged has 1 → should yield
        assert!(r.next_job("cpu-0").is_some());
    }

    #[test]
    fn never_exceeds_credits() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.route(make_job(1, JobKind::IsingSample));
        r.route(make_job(2, JobKind::IsingSample));
        r.grant_credits("cpu-0", 1);
        assert!(r.next_job("cpu-0").is_some());
        assert!(r.next_job("cpu-0").is_none());
    }

    #[test]
    fn cancel_drops_pow_keeps_mempool_and_newer() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.route(make_job(0, JobKind::IsingSample)); // mempool
        r.route(make_job(3, JobKind::IsingSample));
        r.route(make_job(5, JobKind::IsingSample));
        r.route(make_job(6, JobKind::IsingSample));
        r.cancel(5);
        // keep gen 0 and gen 6
        assert_eq!(r.staged_len("cpu-0"), 2);
        let mut gens: Vec<u64> = Vec::new();
        r.grant_credits("cpu-0", 10);
        while let Some(j) = r.next_job("cpu-0") {
            gens.push(j.generation);
        }
        assert_eq!(gens, vec![0, 6]);
    }

    #[test]
    fn on_reject_unsupported_stops_routing_kind() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.register_miner(
            "cpu-1",
            MinerCaps {
                backend: "mock".into(),
                algorithm: "sa".into(),
                supported_kinds: vec![JobKind::IsingSample as i32],
                max_nodes: 1000,
                max_edges: 10000,
            },
        );
        let job = make_job(1, JobKind::IsingSample);
        r.route(job.clone());
        r.grant_credits("cpu-0", 1);
        let dispatched = r.next_job("cpu-0").unwrap();
        r.on_reject("cpu-0", dispatched, RejectReason::UnsupportedKind as i32);
        // cpu-0 marked unsupported; re-route should land on cpu-1
        assert_eq!(r.staged_len("cpu-0"), 0);
        assert_eq!(r.staged_len("cpu-1"), 1);
        // further routes skip cpu-0
        r.route(make_job(2, JobKind::IsingSample));
        assert_eq!(r.staged_len("cpu-0"), 0);
        assert_eq!(r.staged_len("cpu-1"), 2);
    }
}
