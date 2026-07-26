//! Scripted chain for tests: fixed snapshot, optional mempool order, captured submits.

use super::{ChainClient, ChainError, JobOrder, MiningSnapshot, Proof, SubmitAction};
use async_trait::async_trait;
use std::sync::Mutex;

/// Test double: returns a scripted snapshot / order and records submits.
pub struct FakeChain {
    snapshot: Mutex<Option<MiningSnapshot>>,
    orders: Mutex<Vec<JobOrder>>,
    pub submitted: Mutex<Vec<Proof>>,
    /// Optional scripted submit result (default Success).
    submit_result: Mutex<Result<SubmitAction, ChainError>>,
    /// Scripted `latest_qblock_id` (default `None`).
    qblock_id: Mutex<Option<u64>>,
}

impl FakeChain {
    pub fn new(snapshot: MiningSnapshot, order: Option<JobOrder>) -> Self {
        Self {
            snapshot: Mutex::new(Some(snapshot)),
            orders: Mutex::new(order.into_iter().collect()),
            submitted: Mutex::new(Vec::new()),
            submit_result: Mutex::new(Ok(SubmitAction::Success)),
            qblock_id: Mutex::new(None),
        }
    }

    pub fn set_qblock_id(&self, id: Option<u64>) {
        *self.qblock_id.lock().unwrap() = id;
    }

    pub fn set_snapshot(&self, snap: Option<MiningSnapshot>) {
        *self.snapshot.lock().unwrap() = snap;
    }

    pub fn set_orders(&self, orders: Vec<JobOrder>) {
        *self.orders.lock().unwrap() = orders;
    }

    pub fn submitted_count(&self) -> usize {
        self.submitted.lock().unwrap().len()
    }

    pub fn take_submitted(&self) -> Vec<Proof> {
        std::mem::take(&mut *self.submitted.lock().unwrap())
    }
}

#[async_trait]
impl ChainClient for FakeChain {
    async fn fetch_mining_snapshot(
        &self,
        _at: Option<[u8; 32]>,
        _miner_account: [u8; 32],
        _topology_hash: Option<[u8; 32]>,
    ) -> Result<Option<MiningSnapshot>, ChainError> {
        Ok(self.snapshot.lock().unwrap().clone())
    }

    async fn fetch_mempool_orders(
        &self,
        _miner_account: [u8; 32],
    ) -> Result<Vec<JobOrder>, ChainError> {
        Ok(self.orders.lock().unwrap().clone())
    }

    async fn submit_proof(&self, proof: &Proof) -> Result<SubmitAction, ChainError> {
        self.submitted.lock().unwrap().push(proof.clone());
        // Can't move out of Mutex guard for Result with ChainError (not Clone);
        // reconstruct Success/Retry/etc from a stored pattern.
        match &*self.submit_result.lock().unwrap() {
            Ok(a) => Ok(*a),
            Err(ChainError::Unavailable(s)) => Err(ChainError::Unavailable(s.clone())),
            Err(ChainError::Decode(s)) => Err(ChainError::Decode(s.clone())),
            Err(ChainError::Submit(s)) => Err(ChainError::Submit(s.clone())),
        }
    }

    async fn fetch_latest_qblock_id(&self) -> Result<Option<u64>, ChainError> {
        Ok(*self.qblock_id.lock().unwrap())
    }
}
