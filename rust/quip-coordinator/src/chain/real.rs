//! Real chain client: method bodies are CONFIRM-isolated `todo!`s.
//!
//! Depends on external crates not available in this workspace:
//! `quip-protocol-rs`, `hybrid-sig`, `quantum-validation`.

use super::{ChainClient, ChainError, JobOrder, MiningSnapshot, Proof, SubmitAction};
use async_trait::async_trait;

/// Production chain client (not functional until external crates are wired).
pub struct RealChainClient {
    pub validators: Vec<String>,
    pub signer_key: String,
}

impl RealChainClient {
    pub fn new(validators: Vec<String>, signer_key: String) -> Self {
        Self {
            validators,
            signer_key,
        }
    }
}

#[async_trait]
impl ChainClient for RealChainClient {
    async fn fetch_mining_snapshot(
        &self,
        _at: Option<[u8; 32]>,
        _miner_account: [u8; 32],
        _topology_hash: Option<[u8; 32]>,
    ) -> Result<Option<MiningSnapshot>, ChainError> {
        // CONFIRM crate API: quip-protocol-rs runtime call
        // QuantumPowApi_mining_snapshot + SCALE decode of:
        // last_proof_block_hash, difficulty, topology_hash, nodes, edges, allowed_*
        todo!("CONFIRM crate API: quip-protocol-rs mining_snapshot runtime API + SCALE decode");
    }

    async fn fetch_mempool_orders(
        &self,
        _miner_account: [u8; 32],
    ) -> Result<Vec<JobOrder>, ChainError> {
        // CONFIRM crate API: block event scan (JobProposed/OrderExpired) +
        // query_job_order storage read via quip-protocol-rs
        todo!("CONFIRM crate API: quip-protocol-rs JobProposed events + JobOrder storage decode");
    }

    async fn submit_proof(&self, _proof: &Proof) -> Result<SubmitAction, ChainError> {
        // CONFIRM crate API: hybrid-sig HybridTxSignature + quip-protocol-rs
        // extrinsic assembly/submit + receipt classification
        todo!("CONFIRM crate API: hybrid-sig sign + quip-protocol-rs submit_proof extrinsic");
    }
}
