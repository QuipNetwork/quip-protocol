//! Chain access behind `ChainClient`. Real impl talks Substrate JSON-RPC;
//! tests use `FakeChain`.

pub mod extrinsic;
pub mod fake;
pub mod mempool;
pub mod proof_encode;
pub mod real;
pub mod scale_types;
pub mod snapshot;
pub mod submit;

pub use fake::FakeChain;
pub use mempool::JobOrder;
pub use real::RealChainClient;
pub use snapshot::{head_state_key, MiningSnapshot};
pub use submit::{classify_receipt, Proof, SubmitAction};

use async_trait::async_trait;

/// Errors from chain I/O (RPC, decode, submit).
#[derive(Debug)]
pub enum ChainError {
    Unavailable(String),
    Decode(String),
    Submit(String),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::Unavailable(s) => write!(f, "chain unavailable: {s}"),
            ChainError::Decode(s) => write!(f, "decode error: {s}"),
            ChainError::Submit(s) => write!(f, "submit error: {s}"),
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
}
