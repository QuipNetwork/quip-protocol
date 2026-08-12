//! Startup driver for the round state machine.
//!
//! Process start has no miners to stop and does not stage work. This module
//! drives [`crate::round::RoundState`] through validator-synced, account-funded,
//! and requirements-downloaded. The feeder drives the same machine, including
//! stop-mining and start-mining, on every later round.
//!
//! A funding failure at startup is fatal. A missing snapshot is not: the
//! caller warns and the feeder retries after miners connect.

use crate::chain::extrinsic::hex_encode;
use crate::chain::sync::{wait_until_synced, SyncOutcome, SyncSource};
use crate::chain::{ChainClient, MiningSnapshot, ParticipationOutcome};
use crate::funding::{
    ensure_funded, BalanceSource, Faucet, FundingError, FundingParams, HttpFaucet,
};
use crate::round::{RoundEvent, RoundState};
use std::time::Duration;

/// Why the coordinator is not ready to mine.
#[derive(Debug)]
pub enum ReadinessError {
    /// The miner account cannot pay submit fees.
    Funding(FundingError),
    /// The next qblock's requirements could not be loaded.
    Snapshot(String),
}

impl std::fmt::Display for ReadinessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Funding(e) => write!(f, "{e}"),
            Self::Snapshot(e) => write!(f, "cannot load next qblock requirements: {e}"),
        }
    }
}

impl std::error::Error for ReadinessError {}

/// Build a faucet client from a URL. A URL that will not even construct a
/// client is reported and then treated as absent.
#[must_use]
pub fn build_faucet(url: Option<&str>) -> Option<HttpFaucet> {
    let url = url?;
    match HttpFaucet::new(url) {
        Ok(f) => Some(f),
        Err(e) => {
            tracing::error!(url = %url, error = %e, "cannot build faucet client");
            None
        }
    }
}

/// Drive the round machine through validator-synced, account-funded, and
/// requirements-downloaded. Used at process start. After the snapshot lands
/// the walk declares participation for the candidate qblock. That declaration
/// never fails the walk. The feeder walks the same states, plus stop-mining
/// and start-mining, on every later round.
///
/// `sleep` is injected so tests do not wait on the real clock. A validator
/// that never answers sync is not a failure here: that matches startup, which
/// warns and continues.
///
/// # Errors
/// Returns [`ReadinessError::Funding`] when the account cannot be brought up
/// to the floor, or [`ReadinessError::Snapshot`] when the next round's
/// requirements cannot be read.
pub async fn prepare_round<C, F, S, Fut>(
    chain: &C,
    faucet: Option<&F>,
    account: [u8; 32],
    funding: &FundingParams,
    sleep: S,
) -> Result<MiningSnapshot, ReadinessError>
where
    C: ChainClient + SyncSource + BalanceSource,
    F: Faucet + ?Sized,
    S: FnMut(Duration) -> Fut + Send + Clone,
    Fut: std::future::Future<Output = ()> + Send,
{
    let mut state = RoundState::ValidatorSynced;
    state.log_entry(0);

    match wait_until_synced(chain, sleep.clone()).await {
        SyncOutcome::Synced => {}
        SyncOutcome::Unknown(reason) => {
            tracing::warn!(
                reason = %reason,
                "cannot confirm the validator has caught up; continuing, but funding and \
                 mining may fail until it does"
            );
        }
    }
    state = match state.transition(RoundEvent::Succeeded) {
        Some(next) => {
            next.log_entry(0);
            next
        }
        None => state,
    };

    let _ = ensure_funded(chain, faucet, account, funding, sleep)
        .await
        .map_err(ReadinessError::Funding)?;
    state = match state.transition(RoundEvent::Succeeded) {
        Some(next) => {
            next.log_entry(0);
            next
        }
        None => state,
    };

    let snap = chain
        .fetch_mining_snapshot(None, account, None)
        .await
        .map_err(|e| ReadinessError::Snapshot(e.to_string()))?
        .ok_or_else(|| ReadinessError::Snapshot("chain has no mining snapshot".into()))?;
    let mut last_declared = None;
    declare_round_participation(chain, &mut last_declared, account).await;
    let _ = state.transition(RoundEvent::Succeeded);
    Ok(snap)
}

/// Candidate qblock id the pallet accepts: one past the last minted qblock.
#[must_use]
pub(crate) fn candidate_qblock_id(latest: Option<u64>) -> u64 {
    latest.unwrap_or(0).saturating_add(1)
}

/// Submit `MinerRegistry.participate` for the current candidate qblock.
///
/// Mining does not wait on the result. A second call for the same qblock does
/// not submit again. Pallet errors never become a readiness failure.
pub(crate) async fn declare_round_participation<C: ChainClient>(
    chain: &C,
    last_declared: &mut Option<u64>,
    account: [u8; 32],
) {
    let latest = match chain.fetch_latest_qblock_id().await {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "cannot read qblock id; will retry participation next round"
            );
            return;
        }
    };
    let qblock_id = candidate_qblock_id(latest);
    if *last_declared == Some(qblock_id) {
        return;
    }
    match chain.declare_participation(qblock_id).await {
        Ok(ParticipationOutcome::Declared | ParticipationOutcome::AlreadyDeclared) => {
            *last_declared = Some(qblock_id);
        }
        Ok(ParticipationOutcome::StaleQBlock) => {
            tracing::debug!(
                qblock_id,
                "candidate qblock moved; will declare the next one"
            );
            *last_declared = Some(qblock_id);
        }
        Ok(ParticipationOutcome::DescriptorMissing) => {
            tracing::warn!(
                account = %hex_encode(&account),
                qblock_id,
                "MinerRegistry has no descriptor for this account; participation cannot be recorded until one is set"
            );
            *last_declared = Some(qblock_id);
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                qblock_id,
                "participation submit failed; will retry next round"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{candidate_qblock_id, declare_round_participation, prepare_round, HttpFaucet};
    use crate::chain::{ChainError, FakeChain, MiningSnapshot, ParticipationOutcome};
    use crate::funding::FundingParams;
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tracing_subscriber::fmt::MakeWriter;

    fn snap() -> MiningSnapshot {
        MiningSnapshot {
            last_proof_block_hash: [1u8; 32],
            topology_hash: vec![0u8; 32],
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_milli: vec![0],
            allowed_j_milli: vec![0],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 1,
            max_energy_milli: 0,
            min_diversity_milli: 0,
            block_number: 1,
        }
    }

    async fn no_sleep(_d: Duration) {}

    fn account() -> [u8; 32] {
        [0xab; 32]
    }

    #[test]
    fn candidate_is_one_past_latest() {
        assert_eq!(candidate_qblock_id(None), 1);
        assert_eq!(candidate_qblock_id(Some(0)), 1);
        assert_eq!(candidate_qblock_id(Some(4581)), 4582);
    }

    #[tokio::test]
    async fn same_qblock_declares_once() {
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(10));
        let mut last = None;
        declare_round_participation(&chain, &mut last, account()).await;
        declare_round_participation(&chain, &mut last, account()).await;
        assert_eq!(chain.participation_calls(), 1);
        assert_eq!(chain.take_participations(), vec![11]);
        assert_eq!(last, Some(11));
    }

    #[tokio::test]
    async fn different_qblocks_declare_twice() {
        let chain = FakeChain::new(snap(), None);
        let mut last = None;
        chain.set_qblock_id(Some(10));
        declare_round_participation(&chain, &mut last, account()).await;
        chain.set_qblock_id(Some(11));
        declare_round_participation(&chain, &mut last, account()).await;
        assert_eq!(chain.participation_calls(), 2);
        assert_eq!(chain.take_participations(), vec![11, 12]);
    }

    #[tokio::test]
    async fn prepare_round_declares_the_candidate() {
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(20));
        let _snap = prepare_round(
            &chain,
            None::<&HttpFaucet>,
            account(),
            &FundingParams::default(),
            no_sleep,
        )
        .await
        .expect("prepare_round");
        assert_eq!(chain.participation_calls(), 1);
        assert_eq!(chain.take_participations(), vec![21]);
    }

    #[derive(Clone)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("capture lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Capture {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn drain(buf: &Arc<Mutex<Vec<u8>>>) -> String {
        let mut g = buf.lock().expect("capture lock");
        let text = String::from_utf8_lossy(&g).into_owned();
        g.clear();
        text
    }

    #[tokio::test]
    async fn only_descriptor_required_warns_among_pallet_errors() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_writer(Capture(Arc::clone(&buf)))
            .with_ansi(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(4));

        chain.set_participation_result(Ok(ParticipationOutcome::AlreadyDeclared));
        let mut last = None;
        declare_round_participation(&chain, &mut last, account()).await;
        let dup = drain(&buf);
        assert!(
            !dup.contains("WARN") && !dup.to_ascii_lowercase().contains("warn"),
            "DuplicateParticipation must not warn, got {dup:?}"
        );

        chain.set_qblock_id(Some(5));
        chain.set_participation_result(Ok(ParticipationOutcome::StaleQBlock));
        declare_round_participation(&chain, &mut last, account()).await;
        let stale = drain(&buf);
        assert!(
            stale.is_empty(),
            "InvalidQBlockId must not warn, got {stale:?}"
        );

        chain.set_qblock_id(Some(6));
        chain.set_participation_result(Ok(ParticipationOutcome::DescriptorMissing));
        declare_round_participation(&chain, &mut last, account()).await;
        let missing = drain(&buf);
        assert!(
            missing.contains("no descriptor"),
            "DescriptorRequired must warn, got {missing:?}"
        );
        assert!(
            missing.contains("0xabababab"),
            "DescriptorRequired must name the account, got {missing:?}"
        );
    }

    #[tokio::test]
    async fn transient_error_does_not_record_the_qblock() {
        let chain = FakeChain::new(snap(), None);
        chain.set_qblock_id(Some(7));
        chain.set_participation_result(Err(ChainError::Unavailable("rpc down".into())));
        let mut last = None;
        declare_round_participation(&chain, &mut last, account()).await;
        assert_eq!(last, None);
        chain.set_participation_result(Ok(ParticipationOutcome::Declared));
        declare_round_participation(&chain, &mut last, account()).await;
        assert_eq!(chain.participation_calls(), 2);
        assert_eq!(last, Some(8));
    }
}
