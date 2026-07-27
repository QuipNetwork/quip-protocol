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
    /// Consumable credit pool: each `JobRequest` adds credits, each dispatch
    /// spends one. The miner grants `width` up front and one per terminal
    /// event (Result or Reject), so dispatch stays 1:1 with completion and
    /// in-flight is bounded — never a cumulative ceiling that ack also frees
    /// (that double-count let in-flight grow without bound and deadlocked the
    /// miner's channels). In-flight itself is tracked by the coordinator's
    /// `inflight` map, so the router needs no separate outstanding counter.
    granted_credits: u32,
    /// Jobs dispatched (drained from `staged`) since the last `take_consumed`.
    /// The feeder reads-and-resets this each poll to size the adaptive staging
    /// window from the miner's observed drain rate (see `feeder_loop`).
    consumed_since_poll: u32,
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
        match self.miners.get_mut(&id) {
            // Re-registration (e.g. a supervised restart of the same id): keep
            // any jobs re-staged onto this id while it was down — otherwise the
            // reclaim→re-route→re-handshake path silently drops them. Reset only
            // the volatile per-connection state and refresh caps.
            Some(q) => {
                q.caps = caps;
                q.granted_credits = 0;
                q.consumed_since_poll = 0;
                q.unsupported_kinds.clear();
            }
            None => {
                self.miners.insert(
                    id,
                    MinerQueue {
                        caps,
                        staged: VecDeque::new(),
                        granted_credits: 0,
                        consumed_since_poll: 0,
                        unsupported_kinds: HashSet::new(),
                    },
                );
            }
        }
    }

    /// Stage `job` on a capable miner; returns the chosen `miner_id`, or `None`.
    pub fn route(&mut self, job: Job) -> Option<String> {
        self.route_excluding(job, None)
    }

    /// Like [`Router::route`], but never selects `exclude`. Used by
    /// [`Router::on_reject`] so a rejected job is not re-staged onto the miner
    /// that just rejected it.
    fn route_excluding(&mut self, job: Job, exclude: Option<&str>) -> Option<String> {
        let n_nodes = job_node_count(&job);
        let n_edges = job_edge_count(&job);
        let kind = job.kind;

        let mut candidates: Vec<&String> = self
            .miners
            .iter()
            .filter(|(id, q)| exclude != Some(id.as_str()) && capable(q, kind, n_nodes, n_edges))
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

    /// Pop the next staged job, spending one credit. Returns `None` when the
    /// miner has no credits left or nothing is staged.
    pub fn next_job(&mut self, miner_id: &str) -> Option<Job> {
        let q = self.miners.get_mut(miner_id)?;
        if q.granted_credits == 0 {
            return None;
        }
        let job = q.staged.pop_front()?;
        q.granted_credits -= 1;
        q.consumed_since_poll = q.consumed_since_poll.saturating_add(1);
        Some(job)
    }

    /// Read and reset the jobs-consumed counter for `miner_id` — the feeder's
    /// per-poll drain sample. Returns 0 for an unknown miner or an idle poll.
    pub fn take_consumed(&mut self, miner_id: &str) -> u32 {
        self.miners
            .get_mut(miner_id)
            .map(|q| std::mem::take(&mut q.consumed_since_poll))
            .unwrap_or(0)
    }

    /// Handle a miner reject: mark unsupported kinds, re-route to a *different*
    /// capable miner.
    ///
    /// Re-routing excludes the rejecting miner: for a deterministic reject
    /// (`Malformed`, `Expired`, `TopologyMissing`/`Mismatch`) the same job on the
    /// same miner fails again, and since the rejection grants its own replacement
    /// credit it would loop forever with a single miner. If no other miner can
    /// take it, the job lands in `unroutable` (dropped) rather than spinning.
    pub fn on_reject(&mut self, miner_id: &str, job: Job, reason: i32) {
        if reason == RejectReason::UnsupportedKind as i32 {
            if let Some(q) = self.miners.get_mut(miner_id) {
                q.unsupported_kinds.insert(job.kind);
            }
        }
        if self.route_excluding(job, Some(miner_id)).is_none() {
            tracing::warn!(
                miner = %miner_id,
                reason,
                "rejected job has no alternative capable miner; dropping"
            );
        }
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

    /// Registered miner ids, sorted for deterministic iteration (the feeder
    /// walks them to top each up to buffer depth).
    pub fn miner_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.miners.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Stage `job` directly on `miner_id` if it is registered and capable —
    /// the feeder tops up a specific miner rather than first-fit `route`. Returns
    /// whether the job was staged.
    pub fn stage_on(&mut self, miner_id: &str, job: Job) -> bool {
        let n_nodes = job_node_count(&job);
        let n_edges = job_edge_count(&job);
        let kind = job.kind;
        if let Some(q) = self.miners.get_mut(miner_id) {
            if capable(q, kind, n_nodes, n_edges) {
                q.staged.push_back(job);
                return true;
            }
        }
        false
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
    fn next_job_consumes_credits() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.route(make_job(1, JobKind::IsingSample));
        r.route(make_job(2, JobKind::IsingSample));
        assert!(r.next_job("cpu-0").is_none()); // no credits
        r.grant_credits("cpu-0", 1);
        assert!(r.next_job("cpu-0").is_some()); // spends the one credit
        assert!(r.next_job("cpu-0").is_none()); // pool empty — a completion is
                                                // what refills it, not dispatch
        r.grant_credits("cpu-0", 1); // miner grants one per terminal event
        assert!(r.next_job("cpu-0").is_some()); // dispatches the replacement
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
    fn take_consumed_counts_dispatches_then_resets() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.route(make_job(1, JobKind::IsingSample));
        r.route(make_job(2, JobKind::IsingSample));
        r.grant_credits("cpu-0", 2);
        assert!(r.next_job("cpu-0").is_some());
        assert!(r.next_job("cpu-0").is_some());
        assert_eq!(r.take_consumed("cpu-0"), 2); // two dispatched this interval
        assert_eq!(r.take_consumed("cpu-0"), 0); // reset after read
        assert_eq!(r.take_consumed("unknown"), 0); // unknown miner
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
    fn miner_ids_are_sorted() {
        let mut r = Router::new();
        r.register_miner("cpu-1", caps_ising());
        r.register_miner("cpu-0", caps_ising());
        assert_eq!(
            r.miner_ids(),
            vec!["cpu-0".to_string(), "cpu-1".to_string()]
        );
    }

    #[test]
    fn stage_on_targets_specific_capable_miner() {
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.register_miner("cpu-1", caps_ising());
        // Directly stage on cpu-1 (first-fit `route` would have picked cpu-0).
        assert!(r.stage_on("cpu-1", make_job(1, JobKind::IsingSample)));
        assert_eq!(r.staged_len("cpu-0"), 0);
        assert_eq!(r.staged_len("cpu-1"), 1);
        // Unknown miner → not staged.
        assert!(!r.stage_on("nope", make_job(2, JobKind::IsingSample)));
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

    #[test]
    fn on_reject_terminal_does_not_reroute_onto_same_single_miner() {
        // F6: with one capable miner, a deterministic reject must NOT re-stage
        // onto the rejecting miner (which would loop forever); it drops instead.
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        let job = make_job(1, JobKind::IsingSample);
        r.route(job);
        r.grant_credits("cpu-0", 1);
        let dispatched = r.next_job("cpu-0").unwrap();
        assert_eq!(r.staged_len("cpu-0"), 0);
        r.on_reject("cpu-0", dispatched, RejectReason::Malformed as i32);
        // Not re-staged onto the only (rejecting) miner.
        assert_eq!(r.staged_len("cpu-0"), 0);
    }

    #[test]
    fn reregister_preserves_staged_queue() {
        // F8: a supervised restart of the same id must keep jobs re-staged onto
        // it while it was down (reclaim → re-route → re-handshake path).
        let mut r = Router::new();
        r.register_miner("cpu-0", caps_ising());
        r.route(make_job(0, JobKind::IsingSample)); // mempool order (generation 0)
        assert_eq!(r.staged_len("cpu-0"), 1);
        // Re-handshake with the same id: queue survives, credits reset.
        r.register_miner("cpu-0", caps_ising());
        assert_eq!(r.staged_len("cpu-0"), 1);
    }
}
