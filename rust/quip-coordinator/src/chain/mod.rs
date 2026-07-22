//! Chain access behind `ChainClient`. Real impl is CONFIRM-isolated; tests use `FakeChain`.

pub mod fake;
pub mod mempool;
pub mod real;
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
    /// External-crate API not yet wired (CONFIRM markers).
    ConfirmPending(&'static str),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::Unavailable(s) => write!(f, "chain unavailable: {s}"),
            ChainError::Decode(s) => write!(f, "decode error: {s}"),
            ChainError::Submit(s) => write!(f, "submit error: {s}"),
            ChainError::ConfirmPending(s) => write!(f, "CONFIRM crate API: {s}"),
        }
    }
}

impl std::error::Error for ChainError {}

/// Async chain seam: snapshot fetch, mempool orders, extrinsic submit.
///
/// The real impl (`RealChainClient`) depends on external crates
/// (`quip-protocol-rs`, `hybrid-sig`) that are not available in this workspace;
/// its methods are `todo!("CONFIRM crate API: …")`. `FakeChain` is the tested path.
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
