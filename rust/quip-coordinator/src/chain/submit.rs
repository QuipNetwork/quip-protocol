//! Extrinsic submit + hybrid sign (CONFIRM) and groundable receipt classification.

/// Action the fire-loop should take after a submit attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitAction {
    Success,
    Retry,
    StopRoundStale,
    StopFatal,
}

/// A validated proof ready for chain submission.
#[derive(Debug, Clone)]
pub struct Proof {
    pub job_id: Vec<u8>,
    pub best_energy_milli: i64,
    pub diversity_milli: u32,
    pub n_valid: u32,
    pub solutions: Vec<quip_proto::v1::Solution>,
    pub is_pow: bool,
    pub order_id: Vec<u8>,
    pub generation: u64,
    /// PoW salt (32 bytes). Required for live `submit_proof`; empty for
    /// FakeChain paths that only record the proof.
    pub salt: Vec<u8>,
}

/// Classify a pallet/dispatch error string into a fire-loop action.
///
/// Mirrors `substrate/submitter.py:_classify_receipt`. Unknown → fail loud.
pub fn classify_receipt(error: Option<&str>) -> SubmitAction {
    let Some(e) = error else {
        return SubmitAction::Success;
    };
    const RETRY: [&str; 2] = ["InsufficientEnergy", "ProofLimitReached"];
    const STALE: [&str; 3] = ["InvalidNonce", "TopologyNotRegistered", "InvalidTopology"];
    const FATAL: [&str; 5] = [
        "InsufficientSolutions",
        "InsufficientDiversity",
        "MinerNotRegistered",
        "BadSignature",
        "BadProof",
    ];
    if RETRY.iter().any(|s| e.contains(s)) {
        return SubmitAction::Retry;
    }
    if STALE.iter().any(|s| e.contains(s)) {
        return SubmitAction::StopRoundStale;
    }
    if FATAL.iter().any(|s| e.contains(s)) {
        return SubmitAction::StopFatal;
    }
    SubmitAction::StopFatal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_pallet_errors() {
        assert!(matches!(classify_receipt(None), SubmitAction::Success));
        assert!(matches!(
            classify_receipt(Some("InsufficientEnergy")),
            SubmitAction::Retry
        ));
        assert!(matches!(
            classify_receipt(Some("ProofLimitReached")),
            SubmitAction::Retry
        ));
        assert!(matches!(
            classify_receipt(Some("InvalidNonce")),
            SubmitAction::StopRoundStale
        ));
        assert!(matches!(
            classify_receipt(Some("TopologyNotRegistered")),
            SubmitAction::StopRoundStale
        ));
        assert!(matches!(
            classify_receipt(Some("InsufficientSolutions")),
            SubmitAction::StopFatal
        ));
        assert!(matches!(
            classify_receipt(Some("BadProof")),
            SubmitAction::StopFatal
        ));
        assert!(matches!(
            classify_receipt(Some("SomethingUnknown")),
            SubmitAction::StopFatal
        ));
    }
}
