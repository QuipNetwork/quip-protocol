//! The readiness walk used at startup and at every round transition.
//!
//! Winning a qblock returns the coordinator to the same not-yet-mining state
//! it has at process start. Before it mines again it must:
//!
//! 1. Stop mining (the caller broadcasts `Cancel` and drops staged work).
//! 2. Wait until the validator is synced with the chain head.
//! 3. Confirm the miner account can pay submit fees.
//! 4. Download the next qblock's requirements.
//!
//! Only after this returns [`Ok`] does the caller stage new work. A mid-run
//! failure is not fatal: the caller warns, holds off mining, and retries.

use crate::chain::sync::{wait_until_synced, SyncOutcome, SyncSource};
use crate::chain::{ChainClient, MiningSnapshot};
use crate::funding::{
    ensure_funded, BalanceSource, Faucet, FundingError, FundingParams, HttpFaucet,
};
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

/// Steps 2–4 of the readiness walk: validator synced, account funded, next
/// qblock requirements loaded. The caller stops mining before this and
/// starts mining only after this returns the snapshot.
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
    let _ = ensure_funded(chain, faucet, account, funding, sleep)
        .await
        .map_err(ReadinessError::Funding)?;
    chain
        .fetch_mining_snapshot(None, account, None)
        .await
        .map_err(|e| ReadinessError::Snapshot(e.to_string()))?
        .ok_or_else(|| ReadinessError::Snapshot("chain has no mining snapshot".into()))
}
