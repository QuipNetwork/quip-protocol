//! Startup driver for the round state machine.
//!
//! Process start has no miners to stop and does not stage work. This module
//! drives [`crate::round::RoundState`] through validator-synced, account-funded,
//! and requirements-downloaded. The feeder drives the same machine, including
//! stop-mining and start-mining, on every later round.
//!
//! A funding failure at startup is fatal. A missing snapshot is not: the
//! caller warns and the feeder retries after miners connect.

use crate::chain::sync::{wait_until_synced, SyncOutcome, SyncSource};
use crate::chain::{ChainClient, MiningSnapshot};
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
/// requirements-downloaded. Used at process start. The feeder walks the same
/// states, plus stop-mining and start-mining, on every later round.
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
    let _ = state.transition(RoundEvent::Succeeded);
    Ok(snap)
}
