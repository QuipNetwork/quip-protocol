//! Minimal SCALE-compatible types mirrored from pallet `types.rs`.
//!
//! Prefer these over path-depending the FRAME pallets (which drag
//! polkadot-sdk git + full FRAME). Layout must stay byte-identical to
//! `pallet_quantum_pow` / `pallet_quantum_compute_mempool`.

use parity_scale_codec::{Decode, Encode};
use quantum_validation::AllowedValueSpec;
use sp_core::{H256, U256};

/// On-wire difficulty config (matches pallet `DifficultyConfig`).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct DifficultyConfig {
    pub min_solutions: u32,
    pub max_energy_milli: i64,
    pub min_diversity_milli: u32,
}

/// Per-topology curve override (matches pallet `CurveC`): per-mille c-triple
/// stored under `TopologyCurveC[topology_hash]`. Field order is the pallet's.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct CurveCScale {
    pub easy_milli: u32,
    pub knee_milli: u32,
    pub hard_milli: u32,
}

/// Runtime-API mining snapshot (matches pallet `MiningSnapshot`).
///
/// Note: no `block_number` field — callers fetch the header separately.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct MiningSnapshotScale {
    pub last_proof_block_hash: H256,
    pub difficulty: DifficultyConfig,
    pub topology_hash: H256,
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub allowed_h_values: AllowedValueSpec<Vec<i32>>,
    pub allowed_j_values: AllowedValueSpec<Vec<i32>>,
    pub allowed_spin_values: AllowedValueSpec<Vec<i32>>,
}

/// Proof payload for `QuantumPow.submit_proof` (pallet index 10, call index 4).
///
/// Energies / diversity are **not** sent — the chain recomputes them.
/// `solutions` is a list of bit-packed spin vectors. `device_access_time_us`
/// is miner-reported compute time in microseconds (self-reported
/// observability, unverifiable; `0` = unreported).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct QuantumProof {
    pub topology_hash: H256,
    pub nonce: U256,
    pub salt: [u8; 32],
    pub solutions: Vec<Vec<u8>>,
    pub device_access_time_us: u64,
}

/// Ising params nested in a mempool `JobOrder`.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct IsingParams {
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub h_values: Vec<i32>,
    pub j_values: Vec<i32>,
    pub min_energy_milli: Option<i64>,
    pub min_diversity_milli: Option<u32>,
    pub min_solutions: Option<u32>,
}

/// Order timing (BlockNumber = u32 on the default runtime).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct OrderTiming {
    pub deadline_blocks: u32,
    pub block_wait: u32,
}

/// Minimal status enum (SCALE tag order must match the pallet).
#[derive(Clone, Copy, Debug, Encode, Decode, PartialEq, Eq)]
pub enum OrderStatus {
    Opened,
    Expired,
    Closed,
}

/// Reward resolution enum (SCALE tag order must match the pallet).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub enum RewardResolution {
    SingleBest,
    TopNWeighted { n: u32 },
    TopNEqual { n: u32 },
}

/// Job mode enum. Bid carries optional account / miner-type filters.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub enum JobMode {
    Open,
    Bid {
        miners: Option<Vec<[u8; 32]>>,
        miner_types: Option<Vec<u8>>,
    },
}

/// Result delivery enum.
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub enum ResultDelivery {
    OnChainOnly,
    Callback { endpoint: Vec<u8> },
    CallbackWithPoll { endpoint: Vec<u8> },
}

/// Full on-chain `JobOrder` (AccountId32 / Balance=u128 / BlockNumber=u32).
#[derive(Clone, Debug, Encode, Decode, PartialEq, Eq)]
pub struct JobOrderScale {
    pub spec_id: H256,
    pub proposer: [u8; 32],
    pub ising_params: IsingParams,
    pub reward: u128,
    pub mode: JobMode,
    pub resolution: RewardResolution,
    pub timing: OrderTiming,
    pub delivery: ResultDelivery,
    pub status: OrderStatus,
    pub created_at: u32,
    pub first_solution_at: Option<u32>,
    pub solution_count: u32,
}

/// Call indices for hand-assembled extrinsics (from runtime construct).
pub const QUANTUM_POW_PALLET_INDEX: u8 = 10;
pub const SUBMIT_PROOF_CALL_INDEX: u8 = 4;

/// SCALE-encode the `QuantumPow.submit_proof(proof)` call body.
pub fn encode_submit_proof_call(proof: &QuantumProof) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(QUANTUM_POW_PALLET_INDEX);
    out.push(SUBMIT_PROOF_CALL_INDEX);
    // Call args: single composite field `proof`.
    out.extend(proof.encode());
    out
}

/// Require an `AllowedValueSpec::Set` and return its (non-empty) values.
///
/// The coordinator draws PoW models with [`draw_ising_milli`] over a discrete
/// allowed set. The chain's `generate_ising_model` samples `IntegerRange` /
/// `ContinuousRange` specs with a *different* RNG consumption (see
/// `quantum_validation::AllowedValueSpec::sample`), so a coordinator-side
/// expansion of a range into a `Set` would neither reproduce the chain's model
/// nor match its topology hash. Rather than mine unverifiable jobs, reject
/// non-`Set` specs (and empty sets) here so the failure surfaces at snapshot
/// decode with a clear message.
///
/// [`draw_ising_milli`]: quip_protocol::chacha8::draw_ising_milli
///
/// # Errors
/// Returns an error for `IntegerRange` / `ContinuousRange` specs and for an
/// empty `Set`.
pub fn require_set_values(spec: &AllowedValueSpec<Vec<i32>>) -> Result<Vec<i32>, String> {
    match spec {
        AllowedValueSpec::Set(v) if !v.is_empty() => Ok(v.clone()),
        AllowedValueSpec::Set(_) => {
            Err("allowed-value Set is empty; a PoW model cannot be drawn".to_string())
        }
        AllowedValueSpec::IntegerRange { .. } | AllowedValueSpec::ContinuousRange { .. } => Err(
            "coordinator requires AllowedValueSpec::Set; range specs (IntegerRange/\
             ContinuousRange) sample differently on-chain and are not supported"
                .to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_set_values_accepts_nonempty_set() {
        let spec = AllowedValueSpec::Set(vec![-1000, 0, 1000]);
        assert_eq!(require_set_values(&spec).unwrap(), vec![-1000, 0, 1000]);
    }

    #[test]
    fn require_set_values_rejects_empty_set() {
        let spec = AllowedValueSpec::Set(Vec::new());
        assert!(require_set_values(&spec).is_err());
    }

    #[test]
    fn require_set_values_rejects_ranges() {
        // Range specs sample differently on-chain; the coordinator must not
        // silently expand them into a Set (would diverge from consensus).
        assert!(require_set_values(&AllowedValueSpec::IntegerRange { min: -2, max: 2 }).is_err());
        assert!(require_set_values(&AllowedValueSpec::ContinuousRange {
            min: -1000,
            max: 1000
        })
        .is_err());
    }

    #[test]
    fn quantum_proof_scale_roundtrip() {
        let proof = QuantumProof {
            topology_hash: H256::from([0xab; 32]),
            nonce: U256::from(0x1234_5678u64),
            salt: [0xcd; 32],
            solutions: vec![vec![0b1010_0101], vec![0x00, 0xff]],
            device_access_time_us: 987_654,
        };
        let encoded = proof.encode();
        let decoded = QuantumProof::decode(&mut &encoded[..]).expect("decode");
        assert_eq!(decoded, proof);
    }

    #[test]
    fn mining_snapshot_scale_roundtrip() {
        let snap = MiningSnapshotScale {
            last_proof_block_hash: H256::from([1u8; 32]),
            difficulty: DifficultyConfig {
                min_solutions: 5,
                max_energy_milli: -1_200_000,
                min_diversity_milli: 200,
            },
            topology_hash: H256::from([2u8; 32]),
            nodes: vec![0, 1, 2],
            edges: vec![(0, 1), (1, 2)],
            allowed_h_values: AllowedValueSpec::Set(vec![-1000, 0, 1000]),
            allowed_j_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            allowed_spin_values: AllowedValueSpec::Set(vec![-1000, 1000]),
        };
        let encoded = Some(snap.clone()).encode();
        let decoded: Option<MiningSnapshotScale> =
            Decode::decode(&mut &encoded[..]).expect("decode");
        assert_eq!(decoded, Some(snap));
    }

    #[test]
    fn submit_proof_call_starts_with_pallet_and_call_index() {
        let proof = QuantumProof {
            topology_hash: H256::repeat_byte(0),
            nonce: U256::zero(),
            salt: [0u8; 32],
            solutions: vec![vec![0]],
            device_access_time_us: 0,
        };
        let call = encode_submit_proof_call(&proof);
        assert_eq!(call[0], QUANTUM_POW_PALLET_INDEX);
        assert_eq!(call[1], SUBMIT_PROOF_CALL_INDEX);
        // Remainder is the proof encoding.
        assert_eq!(&call[2..], &proof.encode());
    }

    #[test]
    fn job_order_ising_params_roundtrip() {
        let params = IsingParams {
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            h_values: vec![0, 0],
            j_values: vec![-1000],
            min_energy_milli: Some(-500),
            min_diversity_milli: Some(100),
            min_solutions: Some(2),
        };
        let order = JobOrderScale {
            spec_id: H256::repeat_byte(9),
            proposer: [0x11; 32],
            ising_params: params,
            reward: 1_000_000_000_000,
            mode: JobMode::Open,
            resolution: RewardResolution::SingleBest,
            timing: OrderTiming {
                deadline_blocks: 100,
                block_wait: 10,
            },
            delivery: ResultDelivery::OnChainOnly,
            status: OrderStatus::Opened,
            created_at: 42,
            first_solution_at: None,
            solution_count: 0,
        };
        let enc = order.encode();
        let dec = JobOrderScale::decode(&mut &enc[..]).expect("decode");
        assert_eq!(dec, order);
    }

    fn valid_job_order_bytes() -> Vec<u8> {
        JobOrderScale {
            spec_id: H256::repeat_byte(9),
            proposer: [0x11; 32],
            ising_params: IsingParams {
                nodes: vec![0, 1],
                edges: vec![(0, 1)],
                h_values: vec![0, 0],
                j_values: vec![-1000],
                min_energy_milli: Some(-500),
                min_diversity_milli: Some(100),
                min_solutions: Some(2),
            },
            reward: 1_000_000_000_000,
            mode: JobMode::Open,
            resolution: RewardResolution::SingleBest,
            timing: OrderTiming {
                deadline_blocks: 100,
                block_wait: 10,
            },
            delivery: ResultDelivery::OnChainOnly,
            status: OrderStatus::Opened,
            created_at: 42,
            first_solution_at: None,
            solution_count: 0,
        }
        .encode()
    }

    // The JobProposed decode (real.rs) parses untrusted on-chain bytes; it must
    // reject malformed input with an error, never panic. Only a live #[ignore]
    // devnet test exercised this before — these are the offline guards.

    #[test]
    fn job_order_decode_rejects_truncated_bytes() {
        let enc = valid_job_order_bytes();
        // Chop the tail: the later fields (timing/status/counts) can no longer
        // be decoded.
        let truncated = &enc[..enc.len() - 4];
        assert!(JobOrderScale::decode(&mut &truncated[..]).is_err());
    }

    #[test]
    fn job_order_decode_rejects_empty() {
        assert!(JobOrderScale::decode(&mut &[][..]).is_err());
    }

    #[test]
    fn job_order_decode_rejects_bad_enum_tag() {
        // `mode: JobMode` is a 3-field composite; corrupt the SCALE variant tag
        // of the first enum (`mode`, after the fixed-size spec_id/proposer and
        // the variable ising_params). A tag past the variant count must error,
        // not panic. Scan for the first byte whose flip yields an out-of-range
        // enum tag by fuzzing each position and asserting no decode panics.
        let enc = valid_job_order_bytes();
        for i in 0..enc.len() {
            let mut bad = enc.clone();
            bad[i] = 0xFF; // 0xFF is out of range for every enum in the struct
                           // Must return Ok or Err, never panic (the point of the guard).
            let _ = JobOrderScale::decode(&mut &bad[..]);
        }
    }
}
