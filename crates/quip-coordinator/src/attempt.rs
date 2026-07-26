//! Mining-attempt recording (agm.4).
//!
//! Every model the ising solver sends back a result for is recorded as one JSON
//! line under `<data_dir>/<qblock_id>/attempts.jsonl`. A single writer thread
//! drains a channel and appends, so concurrent miner sessions never interleave
//! partial lines. The REST dashboard ([`crate::dashboard`]) serves these files
//! statically — the coordinator does no query work.

use crate::validate::Validated;
use quip_proto::v1::Job;
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

/// One recorded mining attempt: a model that was sent to a miner and solved.
#[derive(Debug, Clone, Serialize)]
pub struct AttemptRecord {
    /// Unix-epoch milliseconds when the coordinator validated the result.
    pub ts_ms: u64,
    /// Chain quantum-block id (`QuantumPowApi_latest_qblock_id`). `None` before
    /// the chain assigns one (e.g. drive mode) — those land under `pending/`.
    pub qblock_id: Option<u64>,
    /// PoW cancellation generation of the job (0 for mempool).
    pub generation: u64,
    /// Miner that solved the model.
    pub miner_id: String,
    /// Job id (nonce) in hex.
    pub job_id: String,
    pub is_pow: bool,
    /// Mempool order id in hex; empty for PoW.
    pub order_id: String,
    pub best_energy_milli: i64,
    pub diversity_milli: u32,
    pub n_valid: u32,
    /// Met the acceptance gate (energy + diversity + min_solutions).
    pub accepted: bool,
    /// The coordinator submitted this attempt as a proof.
    pub submitted: bool,
    pub device_access_time_us: u64,
}

impl AttemptRecord {
    /// Build a record from a validated result. `miner_id` is the solving miner;
    /// `submitted` is whether the coordinator submitted it as a proof.
    pub fn new(
        qblock_id: Option<u64>,
        miner_id: &str,
        job_id: &[u8],
        job: &Job,
        v: &Validated,
        submitted: bool,
        device_access_time_us: u64,
    ) -> Self {
        let (is_pow, order_id) = job
            .provenance
            .as_ref()
            .map(|p| (p.is_pow, hex(&p.order_id)))
            .unwrap_or((false, String::new()));
        Self {
            ts_ms: now_ms(),
            qblock_id,
            generation: job.generation,
            miner_id: miner_id.to_string(),
            job_id: hex(job_id),
            is_pow,
            order_id,
            best_energy_milli: v.best_energy_milli,
            diversity_milli: v.diversity_milli,
            n_valid: v.n_valid,
            accepted: v.accepted,
            submitted,
            device_access_time_us,
        }
    }

    /// Directory segment: the qblock id, or `pending` before the chain assigns
    /// one.
    fn qblock_dir(&self) -> String {
        self.qblock_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "pending".to_string())
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Append one record as a JSON line to `<data_dir>/<qblock_id>/attempts.jsonl`,
/// creating the per-qblock directory on first write.
pub fn append_record(data_dir: &Path, rec: &AttemptRecord) -> std::io::Result<()> {
    let dir = data_dir.join(rec.qblock_dir());
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("attempts.jsonl");
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let line = serde_json::to_string(rec).map_err(std::io::Error::other)?;
    writeln!(f, "{line}")
}

/// Spawn the single writer thread and return the sender the sessions push to.
/// Blocking file I/O runs on this dedicated thread, off the async runtime.
pub fn spawn_writer(data_dir: PathBuf) -> mpsc::Sender<AttemptRecord> {
    let (tx, rx) = mpsc::channel::<AttemptRecord>();
    std::thread::spawn(move || {
        while let Ok(rec) = rx.recv() {
            if let Err(e) = append_record(&data_dir, &rec) {
                tracing::warn!(qblock = ?rec.qblock_id, "attempt-log write failed: {e}");
            }
        }
    });
    tx
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::Provenance;

    fn rec(qblock_id: Option<u64>, job_id: &[u8], is_pow: bool) -> AttemptRecord {
        let job = Job {
            job_id: job_id.to_vec(),
            generation: 7,
            provenance: Some(Provenance {
                is_pow,
                order_id: if is_pow { vec![] } else { vec![0xab, 0xcd] },
            }),
            ..Default::default()
        };
        let v = Validated {
            best_energy_milli: -14_200,
            diversity_milli: 250,
            n_valid: 6,
            accepted: true,
        };
        AttemptRecord::new(qblock_id, "cpu-0", job_id, &job, &v, is_pow, 1234)
    }

    #[test]
    fn record_serializes_hex_ids_and_fields() {
        let r = rec(Some(42), &[0x01, 0xff], true);
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(v["qblock_id"], 42);
        assert_eq!(v["job_id"], "01ff");
        assert_eq!(v["miner_id"], "cpu-0");
        assert_eq!(v["is_pow"], true);
        assert_eq!(v["order_id"], "");
        assert_eq!(v["best_energy_milli"], -14_200);
        assert_eq!(v["n_valid"], 6);
        assert_eq!(v["submitted"], true);
    }

    #[test]
    fn mempool_record_hex_encodes_order_id() {
        let r = rec(Some(1), &[0x09], false);
        assert_eq!(r.order_id, "abcd");
        assert!(!r.is_pow);
    }

    #[test]
    fn append_writes_jsonl_under_qblock_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let r1 = rec(Some(5), &[0xaa], true);
        let r2 = rec(Some(5), &[0xbb], true);
        append_record(tmp.path(), &r1).unwrap();
        append_record(tmp.path(), &r2).unwrap();
        let path = tmp.path().join("5").join("attempts.jsonl");
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        // Each line is a standalone JSON object.
        for line in lines {
            let _: serde_json::Value = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn none_qblock_lands_under_pending() {
        let tmp = tempfile::tempdir().unwrap();
        append_record(tmp.path(), &rec(None, &[0x01], true)).unwrap();
        assert!(tmp.path().join("pending").join("attempts.jsonl").exists());
    }
}
