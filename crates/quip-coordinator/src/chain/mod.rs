//! Chain access behind `ChainClient`. Real impl talks Substrate JSON-RPC;
//! tests use `FakeChain`.

pub mod extrinsic;
pub mod fake;
pub mod mempool;
pub mod outcome;
pub mod preflight;
pub mod proof_encode;
pub mod real;
pub mod scale_types;
pub mod snapshot;
pub mod submit;
pub mod sync;
pub mod transport;
pub mod transport_jsonrpsee;

pub use fake::FakeChain;
pub use mempool::JobOrder;
pub use outcome::{SubmitLedger, QBLOCK_RETENTION};
pub use real::RealChainClient;
pub use scale_types::{MinerKind, MinerSpecScale, NodeDescriptorV2Input, NodeLogLevel};
pub use snapshot::{head_state_key, DecayParams, MiningSnapshot};
pub use submit::{
    classify_descriptor, classify_participation, classify_receipt, DescriptorOutcome,
    ParticipationOutcome, Proof, SubmitAction,
};
pub use transport::{BoxStream, RpcTransport};
pub use transport_jsonrpsee::JsonrpseeTransport;

use async_trait::async_trait;

/// Errors from chain I/O (RPC, decode, submit).
#[derive(Debug)]
pub enum ChainError {
    /// Node / transport unreachable or misconfigured.
    Unavailable(String),
    /// SCALE or hex decode of a chain response failed.
    Decode(String),
    /// Extrinsic submit or proof-encode path failed.
    Submit(String),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(s) => write!(f, "chain unavailable: {s}"),
            Self::Decode(s) => write!(f, "decode error: {s}"),
            Self::Submit(s) => write!(f, "submit error: {s}"),
        }
    }
}

impl std::error::Error for ChainError {}

/// Async chain seam: snapshot fetch, mempool orders, extrinsic submit.
#[async_trait]
pub trait ChainClient: Send + Sync {
    /// Fetch the mining snapshot at `at` (or best head). `None` if not ready.
    async fn fetch_mining_snapshot(
        &self,
        at: Option<[u8; 32]>,
        miner_account: [u8; 32],
        topology_hash: Option<[u8; 32]>,
    ) -> Result<Option<MiningSnapshot>, ChainError>;

    /// Fetch open mempool orders eligible for this miner.
    async fn fetch_mempool_orders(
        &self,
        miner_account: [u8; 32],
    ) -> Result<Vec<JobOrder>, ChainError>;

    /// Hybrid-sign and submit a proof extrinsic; classify the receipt.
    async fn submit_proof(&self, proof: &Proof) -> Result<SubmitAction, ChainError>;

    /// Hybrid-sign and submit `MinerRegistry.set_descriptor`.
    async fn file_descriptor(
        &self,
        descriptor: &NodeDescriptorV2Input,
    ) -> Result<DescriptorOutcome, ChainError>;

    /// Hybrid-sign and submit `MinerRegistry.participate` for `qblock_id`.
    async fn declare_participation(
        &self,
        qblock_id: u64,
    ) -> Result<ParticipationOutcome, ChainError>;

    /// Current quantum-block id (`QuantumPowApi_latest_qblock_id`). `None` when
    /// the chain hasn't started a round or doesn't expose one; used to key the
    /// per-qblock mining-attempt logs.
    async fn fetch_latest_qblock_id(&self) -> Result<Option<u64>, ChainError>;

    /// Fetch decay-projection inputs for `topology_hash`: base (un-decayed)
    /// difficulty, last-proof block, epoch length, and the curve c-triple — so
    /// the coordinator can locally project when a candidate becomes viable as
    /// difficulty eases, without polling the chain every block. `None` if the
    /// read fails.
    async fn fetch_decay_params(
        &self,
        topology_hash: [u8; 32],
    ) -> Result<Option<DecayParams>, ChainError>;
}
