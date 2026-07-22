//! Mempool job-order types.
//!
//! > **CONFIRM crate API:** order discovery via block events + storage
//! > query through `quip-protocol-rs`. The decode of on-chain `JobOrder`
//! > is the CONFIRM point; conversion to wire `Job` is groundable.

/// A mempool job order ready for conversion to a wire `Job`.
#[derive(Debug, Clone)]
pub struct JobOrder {
    pub order_id: Vec<u8>,
    pub nodes: Vec<u32>,
    pub edges: Vec<(u32, u32)>,
    pub h_milli: Vec<i32>,
    pub j_milli: Vec<i32>,
    pub min_energy_milli: Option<i64>,
    pub min_diversity_milli: Option<u32>,
    pub min_solutions: Option<u32>,
    pub deadline_ms: u64,
}
