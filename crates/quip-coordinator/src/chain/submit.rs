//! Extrinsic submit + hybrid sign (CONFIRM) and groundable receipt classification.

/// Action the fire-loop should take after a submit attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitAction {
    /// Extrinsic accepted (or no error string to classify).
    Success,
    /// Transient / retriable pallet rejection (energy, proof limit).
    Retry,
    /// Round-stale rejection; abandon the current generation.
    StopRoundStale,
    /// Fatal rejection or unknown error; stop submitting this proof.
    StopFatal,
}

/// A validated proof ready for chain submission.
#[derive(Debug, Clone)]
pub struct Proof {
    /// Wire job id this proof answers.
    pub job_id: Vec<u8>,
    /// Best solution energy in milli units.
    pub best_energy_milli: i64,
    /// Diversity of the solution set in milli units.
    pub diversity_milli: u32,
    /// Count of solutions that passed local gates.
    pub n_valid: u32,
    /// Valid solutions (spins + reported energies).
    pub solutions: Vec<quip_proto::v1::Solution>,
    /// `true` when this is a `PoW` proof (not a mempool order).
    pub is_pow: bool,
    /// Mempool order id bytes; empty for pure `PoW`.
    pub order_id: Vec<u8>,
    /// Producer generation this proof belongs to.
    pub generation: u64,
    /// `PoW` salt (32 bytes). Required for live `submit_proof`; empty for
    /// `FakeChain` paths that only record the proof.
    pub salt: Vec<u8>,
    /// Miner-reported compute time in microseconds, from the result's
    /// `SamplerMeta.device_access_time_us`. Plumbed into the on-chain
    /// `QuantumProof`; `0` = unreported.
    pub device_access_time_us: u64,
}

/// Classify a pallet/dispatch error string into a fire-loop action.
///
/// Mirrors `substrate/submitter.py:_classify_receipt`. Unknown → fail loud.
#[must_use]
pub fn classify_receipt(error: Option<&str>) -> SubmitAction {
    const RETRY: [&str; 2] = ["InsufficientEnergy", "ProofLimitReached"];
    const STALE: [&str; 3] = ["InvalidNonce", "TopologyNotRegistered", "InvalidTopology"];
    const FATAL: [&str; 5] = [
        "InsufficientSolutions",
        "InsufficientDiversity",
        "MinerNotRegistered",
        "BadSignature",
        "BadProof",
    ];
    let Some(e) = error else {
        return SubmitAction::Success;
    };
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
