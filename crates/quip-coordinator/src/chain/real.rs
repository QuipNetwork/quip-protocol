//! Production chain client over Substrate JSON-RPC.
//!
//! Live end-to-end verification requires a running node. Offline unit tests
//! cover SCALE encode/decode, hybrid extrinsic assembly, and proof packing.
//! Methods return transport errors when no validator is reachable.

use super::extrinsic::{
    build_hybrid_signed_extrinsic, default_topology_storage_key, difficulties_storage_key,
    hex_decode, hex_encode, job_orders_storage_key, last_proof_block_storage_key, load_hybrid_pair,
    miner_identity_bytes, topology_curve_c_storage_key, SignedExtensionContext,
};
use super::proof_encode::{build_quantum_proof, ProofBuildContext};
use super::scale_types::{
    encode_participate_call, encode_set_descriptor_call, encode_submit_proof_call,
    require_set_values, CurveCScale, DifficultyConfig, JobOrderScale, MinerKind,
    MiningSnapshotScale, NodeDescriptorV2Input, OrderStatus,
};
use super::submit::{
    classify_descriptor, classify_participation, classify_receipt, DescriptorOutcome,
    ParticipationOutcome, Proof, SubmitAction,
};
use super::transport_jsonrpsee::{rpc_request, RPC_CONNECT_TIMEOUT, RPC_REQUEST_TIMEOUT};
use super::{ChainClient, ChainError, DecayParams, JobOrder, MiningSnapshot};
use crate::decay::{
    DEFAULT_BASE_MAX_ENERGY_MILLI, DEFAULT_C_EASY_MILLI, DEFAULT_C_HARD_MILLI,
    DEFAULT_C_KNEE_MILLI, EPOCH_LENGTH_BLOCKS,
};
use async_trait::async_trait;
use parity_scale_codec::Decode;
use quantum_validation::AllowedValueSpec;
use quip_transaction_crypto::HybridPair;
use serde_json::Value;
use sp_core::crypto::Ss58Codec;
use sp_core::Pair as _;
use std::sync::Mutex;
use std::time::Duration;
use subxt::rpcs::client::reconnecting_rpc_client::{ExponentialBackoff, PingConfig};
use subxt::rpcs::client::ReconnectingRpcClient;
use subxt::transactions::TransactionStatus;
use tokio::sync::OnceCell;

/// Interval between WebSocket pings on the cached subxt client.
///
/// A half-open socket, which a NAT or firewall idle timeout produces, raises
/// no error. Without a ping the coordinator does not learn the link is dead
/// until something tries to use it. Ten seconds is frequent enough to notice
/// a drop long before a later submit needs the socket. The builder default
/// also enables a ping, but that default is not ours to depend on.
const RPC_WS_PING_INTERVAL: Duration = Duration::from_secs(10);

/// Time without a pong before the reconnecting client treats the socket as dead.
///
/// This must stay well above [`RPC_WS_PING_INTERVAL`] so one slow pong does
/// not force a needless reconnect. Thirty seconds gives the peer two extra
/// ping intervals after the first miss. `max_failures` stays at its default
/// of 1.
const RPC_WS_PING_INACTIVE_LIMIT: Duration = Duration::from_secs(30);

/// Keys requested per `state_getKeysPaged` call when walking the order map.
///
/// The mempool holds tens of orders in practice. A page of 200 fetches them in
/// one round trip while staying far below any node response limit.
const ORDER_PAGE_SIZE: u32 = 200;

/// Cached subxt client. Submit still uses it for `submit_and_watch`.
type SubxtClient = subxt::OnlineClient<subxt::SubstrateConfig>;

/// Production chain client (RPC + hybrid-signed submit).
pub struct RealChainClient {
    /// Validator WebSocket / HTTP RPC URLs (primary first).
    pub validators: Vec<String>,
    /// Hybrid keystore path, any substrate secret URI, or 32-byte hex seed.
    pub signer_key: String,
    /// Cached hybrid pair (loaded lazily from `signer_key`).
    pair: Mutex<Option<HybridPair>>,
    /// Last fetched mining snapshot — used by `submit_proof` to pack spins
    /// and derive the nonce. Updated by `fetch_mining_snapshot`.
    last_snapshot: Mutex<Option<MiningSnapshot>>,
    /// Last `allowed_spin` spec (full `AllowedValueSpec`, not just Set values).
    last_spin_spec: Mutex<Option<AllowedValueSpec<Vec<i32>>>>,
    /// Lazily-connected subxt client for `submit_and_watch`.
    subxt: OnceCell<SubxtClient>,
    /// Whether the last RPC round-trip reached the validator. `None` before the
    /// first call. Every RPC builds a fresh client (see `rpc_http` / `rpc_ws`),
    /// so a per-call "connected" line would run to thousands an hour; this
    /// tracks reachability so only the *transitions* are reported.
    reachable: Mutex<Option<bool>>,
    /// Kind declared on `participate`. Derived from the miners this process starts.
    participate_kind: MinerKind,
}

impl RealChainClient {
    /// Construct a client over the given validators and signer material.
    #[must_use]
    pub fn new(validators: Vec<String>, signer_key: String, participate_kind: MinerKind) -> Self {
        Self {
            validators,
            signer_key,
            pair: Mutex::new(None),
            last_snapshot: Mutex::new(None),
            last_spin_spec: Mutex::new(None),
            subxt: OnceCell::new(),
            reachable: Mutex::new(None),
            participate_kind,
        }
    }

    /// Record the outcome of one RPC round-trip and log reachability changes.
    ///
    /// A validator that goes away (or comes back) is the single most common
    /// reason a coordinator stops mining, and v0.2.1 narrated it on every
    /// reconnect. Reporting only the edges keeps that signal without the
    /// per-call volume.
    fn note_reachability(&self, url: &str, outcome: Result<(), &ChainError>) {
        let now = outcome.is_ok();
        // Poisoned lock: reachability is advisory, never fail an RPC over it.
        let Ok(mut guard) = self.reachable.lock() else {
            return;
        };
        if *guard == Some(now) {
            return;
        }
        match outcome {
            Ok(()) => tracing::info!(url = %url, "validator RPC reachable"),
            Err(e) => {
                tracing::warn!(url = %url, error = %e, "validator RPC unreachable");
            }
        }
        *guard = Some(now);
    }

    /// Return the cached subxt client, connecting on the first call.
    ///
    /// The client sits on a reconnecting WebSocket. The socket reopens after
    /// a validator restart. Runtime metadata stays cached in the `OnceCell`.
    ///
    /// Method calls resume on the new socket. Subscriptions do not replay.
    /// A `submit_and_watch` that is in flight when the socket drops fails
    /// that one submission. The caller retries on the next round.
    async fn subxt_client(&self) -> Result<&SubxtClient, ChainError> {
        let url = self.primary_url()?.to_string();
        self.subxt
            .get_or_try_init(|| async move {
                let rpc = ReconnectingRpcClient::builder()
                    .retry_policy(
                        ExponentialBackoff::from_millis(200).max_delay(Duration::from_secs(10)),
                    )
                    .connection_timeout(RPC_CONNECT_TIMEOUT)
                    .request_timeout(RPC_REQUEST_TIMEOUT)
                    .enable_ws_ping(
                        PingConfig::new()
                            .ping_interval(RPC_WS_PING_INTERVAL)
                            .inactive_limit(RPC_WS_PING_INACTIVE_LIMIT),
                    )
                    .build(&url)
                    .await
                    .map_err(|e| ChainError::Unavailable(format!("subxt connect: {e}")))?;
                SubxtClient::from_rpc_client(rpc)
                    .await
                    .map_err(|e| ChainError::Unavailable(format!("subxt connect: {e}")))
            })
            .await
    }

    fn pair(&self) -> Result<HybridPair, ChainError> {
        let mut guard = self
            .pair
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))?;
        if guard.is_none() {
            let p = load_hybrid_pair(&self.signer_key)
                .map_err(|e| ChainError::Unavailable(format!("load signer: {e}")))?;
            *guard = Some(p);
        }
        // HybridPair is not Clone in all builds — re-load from key each time
        // if needed. Prefer re-load for simplicity.
        load_hybrid_pair(&self.signer_key)
            .map_err(|e| ChainError::Unavailable(format!("load signer: {e}")))
    }

    fn primary_url(&self) -> Result<&str, ChainError> {
        self.validators
            .first()
            .map(String::as_str)
            .ok_or_else(|| ChainError::Unavailable("no validators configured".into()))
    }

    /// Issue `method` against the configured validators in order, returning the
    /// first success.
    ///
    /// `validators` is documented as an ordered failover list, and the default
    /// pair leads with a container-network name that does not resolve on a host
    /// install. Trying only the first entry makes that default unusable off
    /// Docker, so every entry gets a turn before the call is called failed.
    async fn rpc_call(&self, method: &str, params: Value) -> Result<Value, ChainError> {
        if self.validators.is_empty() {
            return Err(ChainError::Unavailable("no validators configured".into()));
        }
        let mut last: Option<ChainError> = None;
        for url in &self.validators {
            let out = rpc_request(url, method, params.clone()).await;
            tracing::trace!(url = %url, method = %method, ok = out.is_ok(), "rpc call");
            match out {
                Ok(v) => {
                    // Lock only after the await: `reachable` is a std Mutex and
                    // must never be held across a suspension point.
                    self.note_reachability(url, Ok(()));
                    return Ok(v);
                }
                Err(e) => {
                    tracing::debug!(url = %url, method = %method, error = %e, "validator failed; trying next");
                    last = Some(e);
                }
            }
        }
        // Every endpoint failed: report against the primary, which is the one an
        // operator will look at first.
        let err =
            last.unwrap_or_else(|| ChainError::Unavailable("no validators configured".into()));
        if let Some(primary) = self.validators.first() {
            self.note_reachability(primary, Err(&err));
        }
        Err(err)
    }

    /// Raw `state_getRuntimeVersion` response, for the startup compatibility
    /// check in [`super::preflight`].
    ///
    /// # Errors
    /// Returns a transport error when the validator cannot be reached.
    pub(crate) async fn runtime_version_raw(&self) -> Result<Value, ChainError> {
        self.rpc_call("state_getRuntimeVersion", Value::Array(vec![]))
            .await
    }

    /// Raw `system_health` response, for the startup sync gate in
    /// [`super::sync`].
    ///
    /// # Errors
    /// Returns a transport error when the validator cannot be reached.
    pub(crate) async fn system_health_raw(&self) -> Result<Value, ChainError> {
        self.rpc_call("system_health", Value::Array(vec![])).await
    }

    /// Raw `system_syncState` response: the block heights the sync gate reports.
    ///
    /// # Errors
    /// Returns a transport error when the validator cannot be reached, or when
    /// it does not serve this method.
    pub(crate) async fn sync_state_raw(&self) -> Result<Value, ChainError> {
        self.rpc_call("system_syncState", Value::Array(vec![]))
            .await
    }

    /// Read + SCALE-decode a storage value at `at_hex`. `Ok(None)` when the key
    /// is unset (null result).
    async fn read_storage<T: Decode>(
        &self,
        key: &[u8],
        at_hex: &str,
    ) -> Result<Option<T>, ChainError> {
        let raw = self
            .rpc_call(
                "state_getStorage",
                Value::Array(vec![
                    Value::String(hex_encode(key)),
                    Value::String(at_hex.to_string()),
                ]),
            )
            .await?;
        let Some(hex) = raw.as_str() else {
            return Ok(None);
        };
        let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
        T::decode(&mut &bytes[..])
            .map(Some)
            .map_err(|e| ChainError::Decode(e.to_string()))
    }

    /// Read `QuantumPow.DefaultTopology` at the current head.
    ///
    /// `Ok(None)` means no topology is registered, which is the state a fresh
    /// chain starts in and the only state the seed path accepts.
    ///
    /// # Errors
    /// Returns a transport error when the validator cannot be reached, or a
    /// decode error when the stored value is not a 32-byte hash.
    pub(crate) async fn default_topology(&self) -> Result<Option<[u8; 32]>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;
        self.read_storage::<[u8; 32]>(&default_topology_storage_key(), at)
            .await
    }

    /// Nonce, genesis hash, and runtime versions for a signed extrinsic.
    async fn signed_extension_context(
        &self,
        pair: &HybridPair,
    ) -> Result<SignedExtensionContext, ChainError> {
        let account_ss58 =
            quip_transaction_crypto::account_id_from_public(&pair.public()).to_ss58check();
        let nonce_val = self
            .rpc_call(
                "system_accountNextIndex",
                Value::Array(vec![Value::String(account_ss58)]),
            )
            .await?;
        let account_nonce = u32::try_from(
            nonce_val
                .as_u64()
                .ok_or_else(|| ChainError::Decode("system_accountNextIndex not a u64".into()))?,
        )
        .map_err(|_| ChainError::Decode("account nonce exceeds u32".into()))?;

        let genesis = self
            .rpc_call(
                "chain_getBlockHash",
                Value::Array(vec![Value::Number(0.into())]),
            )
            .await?;
        let genesis_hex = genesis
            .as_str()
            .ok_or_else(|| ChainError::Decode("genesis hash not a string".into()))?;
        let genesis_bytes = hex_decode(genesis_hex).map_err(ChainError::Decode)?;
        let mut genesis_hash = [0u8; 32];
        if genesis_bytes.len() != 32 {
            return Err(ChainError::Decode("genesis hash length".into()));
        }
        genesis_hash.copy_from_slice(&genesis_bytes);

        let rv = self
            .rpc_call("state_getRuntimeVersion", Value::Array(vec![]))
            .await?;
        let spec_version = u32::try_from(
            rv.get("specVersion")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    ChainError::Decode("runtime specVersion missing/not a u64".into())
                })?,
        )
        .map_err(|_| ChainError::Decode("specVersion exceeds u32".into()))?;
        let transaction_version = u32::try_from(
            rv.get("transactionVersion")
                .and_then(Value::as_u64)
                .ok_or_else(|| {
                    ChainError::Decode("runtime transactionVersion missing/not a u64".into())
                })?,
        )
        .map_err(|_| ChainError::Decode("transactionVersion exceeds u32".into()))?;

        Ok(SignedExtensionContext {
            account_nonce,
            genesis_hash,
            spec_version,
            transaction_version,
            tip: 0,
        })
    }

    /// Hybrid-sign `call` and watch inclusion. Callers classify the outcome.
    ///
    /// Submits via subxt's `author_submitAndWatchExtrinsic` and CONFIRMS the
    /// on-chain outcome, rather than treating pool acceptance as success. The
    /// status stream is watched to in-best-block, not to finality, to stay
    /// responsive: a re-org after that point is rare on the local validator,
    /// and the per-generation `current_best` reset bounds any wrong advance.
    pub(crate) async fn submit_signed_call(
        &self,
        call: &[u8],
    ) -> Result<SignedCallOutcome, ChainError> {
        let pair = self.pair()?;
        let signed_ctx = self.signed_extension_context(&pair).await?;
        let ext = build_hybrid_signed_extrinsic(&pair, call, &signed_ctx);

        let client = self.subxt_client().await?;
        let tx_client = client
            .tx()
            .await
            .map_err(|e| ChainError::Unavailable(format!("subxt tx client: {e}")))?;
        let mut progress = tx_client
            .from_bytes(ext)
            .submit_and_watch()
            .await
            .map_err(|e| ChainError::Submit(format!("submit_and_watch: {e}")))?;
        loop {
            let status = progress
                .next()
                .await
                .ok_or_else(|| ChainError::Unavailable("tx status stream ended".into()))?
                .map_err(|e| ChainError::Unavailable(format!("tx progress: {e}")))?;
            match status {
                TransactionStatus::InBestBlock(in_block)
                | TransactionStatus::InFinalizedBlock(in_block) => {
                    let block = in_block.block_hash().to_string();
                    return match in_block.wait_for_success().await {
                        Ok(_events) => Ok(SignedCallOutcome::Success { block }),
                        Err(e) => {
                            let error = match &e {
                                subxt::error::TransactionEventsError::ExtrinsicFailed(
                                    subxt::error::DispatchError::Module(m),
                                ) => m.details_string(),
                                other => other.to_string(),
                            };
                            Ok(SignedCallOutcome::DispatchFailed { error })
                        }
                    };
                }
                TransactionStatus::Invalid { message } => {
                    return Ok(SignedCallOutcome::Invalid { message });
                }
                TransactionStatus::Dropped { message } | TransactionStatus::Error { message } => {
                    return Ok(SignedCallOutcome::Dropped { message });
                }
                _ => {}
            }
        }
    }
}

/// On-chain result of a hybrid-signed extrinsic, before pallet-specific classify.
pub(crate) enum SignedCallOutcome {
    /// Included and the dispatch succeeded.
    Success { block: String },
    /// Included but the dispatch failed.
    DispatchFailed { error: String },
    /// The transaction pool rejected the extrinsic.
    Invalid { message: String },
    /// Dropped or errored before inclusion.
    Dropped { message: String },
}

#[async_trait]
impl crate::funding::BalanceSource for RealChainClient {
    async fn free_balance(&self, account: [u8; 32]) -> Result<u128, String> {
        let key = super::account::system_account_storage_key(&account);
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await
            .map_err(|e| e.to_string())?;
        let at = head
            .as_str()
            .ok_or_else(|| "chain_getBlockHash did not return a string".to_string())?;
        let raw = self
            .rpc_call(
                "state_getStorage",
                Value::Array(vec![
                    Value::String(hex_encode(&key)),
                    Value::String(at.to_string()),
                ]),
            )
            .await
            .map_err(|e| e.to_string())?;
        // A null result means the account has never been touched on chain,
        // which is a zero balance rather than a read failure.
        let Some(hex) = raw.as_str() else {
            return Ok(0);
        };
        let bytes = hex_decode(hex)?;
        super::account::free_from_account_bytes(&bytes)
    }
}

#[async_trait]
impl ChainClient for RealChainClient {
    async fn fetch_mining_snapshot(
        &self,
        at: Option<[u8; 32]>,
        _miner_account: [u8; 32],
        topology_hash: Option<[u8; 32]>,
    ) -> Result<Option<MiningSnapshot>, ChainError> {
        let block_hash = if let Some(h) = at {
            h
        } else {
            let head = self
                .rpc_call("chain_getBlockHash", Value::Array(vec![]))
                .await?;
            let hex = head
                .as_str()
                .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;
            let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
            if bytes.len() != 32 {
                return Err(ChainError::Decode(format!(
                    "block hash len {}",
                    bytes.len()
                )));
            }
            let mut h = [0u8; 32];
            h.copy_from_slice(&bytes);
            h
        };

        // Parameter: Option<H256> SCALE-encoded.
        let param = match topology_hash {
            None => vec![0u8],
            Some(th) => {
                let mut p = vec![1u8];
                p.extend_from_slice(&th);
                p
            }
        };

        let result = self
            .rpc_call(
                "state_call",
                Value::Array(vec![
                    Value::String("QuantumPowApi_mining_snapshot".into()),
                    Value::String(hex_encode(&param)),
                    Value::String(hex_encode(&block_hash)),
                ]),
            )
            .await?;

        let Some(hex) = result.as_str() else {
            // Null result → no topology / empty response.
            return Ok(None);
        };
        let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
        let decoded: Option<MiningSnapshotScale> =
            Decode::decode(&mut &bytes[..]).map_err(|e| ChainError::Decode(e.to_string()))?;
        let Some(scale) = decoded else {
            return Ok(None);
        };

        // Block number from header (snapshot has no block_number field).
        let header = self
            .rpc_call(
                "chain_getHeader",
                Value::Array(vec![Value::String(hex_encode(&block_hash))]),
            )
            .await?;
        let block_number = parse_block_number(&header)?;

        let snap = MiningSnapshot {
            last_proof_block_hash: scale.last_proof_block_hash.0,
            topology_hash: scale.topology_hash.0.to_vec(),
            nodes: scale.nodes,
            edges: scale.edges,
            allowed_h_milli: require_set_values(&scale.allowed_h_values)
                .map_err(ChainError::Decode)?,
            allowed_j_milli: require_set_values(&scale.allowed_j_values)
                .map_err(ChainError::Decode)?,
            allowed_spin_milli: require_set_values(&scale.allowed_spin_values)
                .map_err(ChainError::Decode)?,
            min_solutions: scale.difficulty.min_solutions,
            max_energy_milli: scale.difficulty.max_energy_milli,
            min_diversity_milli: scale.difficulty.min_diversity_milli,
            block_number,
        };

        if let Ok(mut g) = self.last_snapshot.lock() {
            *g = Some(snap.clone());
        }
        if let Ok(mut g) = self.last_spin_spec.lock() {
            *g = Some(scale.allowed_spin_values);
        }
        Ok(Some(snap))
    }

    async fn fetch_latest_qblock_id(&self) -> Result<Option<u64>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let block_hash = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))
            .and_then(|hex| hex_decode(hex).map_err(ChainError::Decode))?;

        // `QuantumPowApi_latest_qblock_id()` takes no args → empty SCALE params.
        let result = self
            .rpc_call(
                "state_call",
                Value::Array(vec![
                    Value::String("QuantumPowApi_latest_qblock_id".into()),
                    Value::String(hex_encode(&[])),
                    Value::String(hex_encode(&block_hash)),
                ]),
            )
            .await?;

        let Some(hex) = result.as_str() else {
            return Ok(None);
        };
        let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
        Decode::decode(&mut &bytes[..]).map_err(|e| ChainError::Decode(e.to_string()))
    }

    async fn fetch_decay_params(
        &self,
        topology_hash: [u8; 32],
    ) -> Result<Option<DecayParams>, ChainError> {
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let at = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?
            .to_string();

        // Base (un-decayed) difficulty; unset topology → chain genesis default.
        let base: Option<DifficultyConfig> = self
            .read_storage(&difficulties_storage_key(&topology_hash), &at)
            .await?;
        // LastProofBlock is a u32 BlockNumber (ValueQuery → 0 when unset).
        let last_proof_block: u32 = self
            .read_storage(&last_proof_block_storage_key(), &at)
            .await?
            .unwrap_or(0);
        // Per-topology curve override; unset → the runtime CurveC* constants.
        let curve: Option<CurveCScale> = self
            .read_storage(&topology_curve_c_storage_key(&topology_hash), &at)
            .await?;
        let (c_easy_milli, c_knee_milli, c_hard_milli) = curve.map_or(
            (
                DEFAULT_C_EASY_MILLI,
                DEFAULT_C_KNEE_MILLI,
                DEFAULT_C_HARD_MILLI,
            ),
            |c| (c.easy_milli, c.knee_milli, c.hard_milli),
        );

        Ok(Some(DecayParams {
            base_max_energy_milli: base
                .map_or(DEFAULT_BASE_MAX_ENERGY_MILLI, |d| d.max_energy_milli),
            last_proof_block: u64::from(last_proof_block),
            epoch_length: EPOCH_LENGTH_BLOCKS,
            c_easy_milli,
            c_knee_milli,
            c_hard_milli,
        }))
    }

    async fn fetch_mempool_orders(
        &self,
        _miner_account: [u8; 32],
    ) -> Result<Vec<JobOrder>, ChainError> {
        // Discover order ids by walking the JobOrders map at head, then
        // storage-read each order. Without a live node this returns transport
        // errors; with a node, empty open-order sets yield Ok(vec![]).
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let head_hex = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;

        let head_bytes = hex_decode(head_hex).map_err(ChainError::Decode)?;
        if head_bytes.len() != 32 {
            return Err(ChainError::Decode(format!(
                "head hash is {} bytes, expected 32",
                head_bytes.len()
            )));
        }

        // Walk the JobOrders map rather than decoding JobProposed events. The
        // map holds every open order, not only those proposed in the head
        // block, and it needs no runtime metadata.
        let prefix = hex_encode(&super::orders::job_orders_prefix());
        let mut order_ids: Vec<u64> = Vec::new();
        let mut start_key: Option<String> = None;
        loop {
            let mut params = vec![
                Value::String(prefix.clone()),
                Value::Number(ORDER_PAGE_SIZE.into()),
            ];
            // state_getKeysPaged takes the previous page's last key as the
            // resume point. The block hash is always the final argument, so a
            // missing start key must still be sent as null.
            params.push(match &start_key {
                Some(k) => Value::String(k.clone()),
                None => Value::Null,
            });
            params.push(Value::String(head_hex.to_string()));

            let page = self
                .rpc_call("state_getKeysPaged", Value::Array(params))
                .await?;
            let Some(keys) = page.as_array() else {
                break;
            };
            if keys.is_empty() {
                break;
            }
            for k in keys {
                let Some(hex) = k.as_str() else { continue };
                let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
                if let Some(oid) = super::orders::order_id_from_key(&bytes) {
                    order_ids.push(oid);
                }
            }
            let full_page = keys.len() == ORDER_PAGE_SIZE as usize;
            start_key = keys.last().and_then(|k| k.as_str()).map(String::from);
            if !full_page {
                break;
            }
        }

        let mut orders = Vec::new();
        for oid in order_ids {
            let key = job_orders_storage_key(oid);
            let raw = self
                .rpc_call(
                    "state_getStorage",
                    Value::Array(vec![
                        Value::String(hex_encode(&key)),
                        Value::String(head_hex.to_string()),
                    ]),
                )
                .await?;
            let Some(hex) = raw.as_str() else {
                continue;
            };
            let bytes = hex_decode(hex).map_err(ChainError::Decode)?;
            let scale = JobOrderScale::decode(&mut &bytes[..])
                .map_err(|e| ChainError::Decode(format!("JobOrder {oid}: {e}")))?;
            if scale.status != OrderStatus::Opened {
                continue;
            }
            orders.push(JobOrder {
                order_id: oid.to_le_bytes().to_vec(),
                nodes: scale.ising_params.nodes,
                edges: scale.ising_params.edges,
                h_milli: scale.ising_params.h_values,
                j_milli: scale.ising_params.j_values,
                min_energy_milli: scale.ising_params.min_energy_milli,
                min_diversity_milli: scale.ising_params.min_diversity_milli,
                min_solutions: scale.ising_params.min_solutions,
                // Convert deadline_blocks → a soft ms deadline using a 6s
                // block time estimate. Callers that need exact expiry should
                // re-query chain head.
                deadline_ms: u64::from(scale.timing.deadline_blocks).saturating_mul(6_000),
            });
        }
        Ok(orders)
    }

    async fn submit_proof(&self, proof: &Proof) -> Result<SubmitAction, ChainError> {
        let pair = self.pair()?;
        let snap = self
            .last_snapshot
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))?
            .clone()
            .ok_or_else(|| {
                ChainError::Unavailable(
                    "submit_proof requires a prior fetch_mining_snapshot".into(),
                )
            })?;
        let spin_spec = self
            .last_spin_spec
            .lock()
            .map_err(|e| ChainError::Unavailable(e.to_string()))?
            .clone()
            .unwrap_or_else(|| AllowedValueSpec::Set(snap.allowed_spin_milli.clone()));

        let salt = extract_salt(proof).ok_or_else(|| {
            ChainError::Submit(
                "proof.salt must be 32 bytes for live submit_proof (session does not yet thread salt through Job)".into(),
            )
        })?;

        let mut topo = [0u8; 32];
        if snap.topology_hash.len() == 32 {
            topo.copy_from_slice(&snap.topology_hash);
        }

        let ctx = ProofBuildContext {
            topology_hash: topo,
            last_proof_block_hash: snap.last_proof_block_hash,
            miner_identity: miner_identity_bytes(&pair),
            salt,
            num_nodes: snap.nodes.len(),
            allowed_spin: spin_spec,
        };
        let quantum = build_quantum_proof(proof, &ctx)
            .map_err(|e| ChainError::Submit(format!("encode proof: {e}")))?;
        let call = encode_submit_proof_call(&quantum);

        match self.submit_signed_call(&call).await? {
            SignedCallOutcome::Success { block } => {
                tracing::info!(block = %block, "proof included and dispatched successfully");
                Ok(SubmitAction::Success)
            }
            SignedCallOutcome::DispatchFailed { error, .. } => {
                tracing::warn!(error = %error, "proof included but ExtrinsicFailed");
                Ok(classify_receipt(Some(&error)))
            }
            SignedCallOutcome::Invalid { message } => {
                tracing::warn!(%message, "proof rejected as invalid before inclusion");
                Ok(classify_receipt(Some(&message)))
            }
            SignedCallOutcome::Dropped { message } => {
                tracing::warn!(%message, "proof dropped by node before inclusion");
                Ok(SubmitAction::Retry)
            }
        }
    }

    async fn file_descriptor(
        &self,
        descriptor: &NodeDescriptorV2Input,
    ) -> Result<DescriptorOutcome, ChainError> {
        let call = encode_set_descriptor_call(descriptor);
        match self.submit_signed_call(&call).await? {
            SignedCallOutcome::Success { .. } => Ok(DescriptorOutcome::Filed),
            SignedCallOutcome::DispatchFailed { error, .. }
            | SignedCallOutcome::Invalid { message: error } => {
                classify_descriptor(Some(&error)).ok_or(ChainError::Submit(error))
            }
            SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(message)),
        }
    }

    async fn declare_participation(
        &self,
        qblock_id: u64,
    ) -> Result<ParticipationOutcome, ChainError> {
        let call = encode_participate_call(qblock_id, self.participate_kind, None);
        match self.submit_signed_call(&call).await? {
            SignedCallOutcome::Success { .. } => Ok(ParticipationOutcome::Declared),
            SignedCallOutcome::DispatchFailed { error, .. }
            | SignedCallOutcome::Invalid { message: error } => {
                classify_participation(Some(&error)).ok_or(ChainError::Submit(error))
            }
            SignedCallOutcome::Dropped { message } => Err(ChainError::Submit(message)),
        }
    }
}

fn extract_salt(proof: &Proof) -> Option<[u8; 32]> {
    if proof.salt.len() != 32 {
        return None;
    }
    let mut s = [0u8; 32];
    s.copy_from_slice(&proof.salt);
    Some(s)
}

fn parse_block_number(header: &Value) -> Result<u64, ChainError> {
    let number = header
        .get("number")
        .ok_or_else(|| ChainError::Decode("header missing number".into()))?;
    if let Some(n) = number.as_u64() {
        return Ok(n);
    }
    if let Some(s) = number.as_str() {
        let s = s.strip_prefix("0x").unwrap_or(s);
        return u64::from_str_radix(s, 16).map_err(|e| ChainError::Decode(e.to_string()));
    }
    Err(ChainError::Decode("header.number unparseable".into()))
}

#[cfg(test)]
mod tests {
    use super::{rpc_request, RealChainClient, RPC_CONNECT_TIMEOUT, RPC_REQUEST_TIMEOUT};
    use crate::chain::scale_types::MinerKind;
    use crate::chain::ChainError;
    use serde_json::Value as JsonValue;
    use std::time::{Duration, Instant};
    use tokio::net::TcpListener;

    /// Accept connections and never write a response byte. Models a wedged peer
    /// that passes TCP but blocks the client forever without timeouts.
    async fn spawn_blackhole_listener() -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind blackhole listener");
        let addr = listener.local_addr().expect("blackhole local addr");
        drop(tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else {
                    break;
                };
                // Hold the accepted socket open and never write. The peer must
                // time out on its own; that is the behaviour under test.
                drop(tokio::spawn(async move {
                    let _sock = sock;
                    std::future::pending::<()>().await;
                }));
            }
        }));
        format!("{addr}")
    }

    /// Upper bound for a single-endpoint blackhole: connect budget plus slack
    /// for scheduler jitter on a loaded CI host. Must stay well under the
    /// test's outer timeout so a hang fails the test, not the harness.
    fn single_endpoint_budget() -> Duration {
        RPC_CONNECT_TIMEOUT
            .saturating_mul(2)
            .saturating_add(Duration::from_secs(5))
    }

    /// Upper bound when the first endpoint blackholes and the second refuses:
    /// one connect timeout plus a refused connect, with CI slack.
    fn failover_budget() -> Duration {
        RPC_CONNECT_TIMEOUT
            .saturating_mul(3)
            .saturating_add(RPC_REQUEST_TIMEOUT)
            .saturating_add(Duration::from_secs(5))
    }

    #[tokio::test]
    async fn ws_blackhole_endpoint_errors_within_connect_budget() {
        let addr = spawn_blackhole_listener().await;
        let url = format!("ws://{addr}");
        let started = Instant::now();
        let outcome = tokio::time::timeout(
            single_endpoint_budget(),
            rpc_request(&url, "system_health", JsonValue::Array(vec![])),
        )
        .await;
        assert!(
            outcome.is_ok(),
            "ws blackhole call hung past {:?}",
            single_endpoint_budget()
        );
        let err = outcome
            .expect("outer timeout")
            .expect_err("blackhole must not succeed");
        assert!(
            matches!(err, ChainError::Unavailable(_)),
            "timeout must classify as Unavailable for failover, got {err}"
        );
        assert!(
            started.elapsed() < single_endpoint_budget(),
            "elapsed {:?} exceeds budget",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn http_blackhole_endpoint_errors_within_request_budget() {
        let addr = spawn_blackhole_listener().await;
        let url = format!("http://{addr}");
        // HTTP has no separate connect timeout; request_timeout covers the hang.
        let budget = RPC_REQUEST_TIMEOUT.saturating_add(Duration::from_secs(10));
        let started = Instant::now();
        let outcome = tokio::time::timeout(
            budget,
            rpc_request(&url, "system_health", JsonValue::Array(vec![])),
        )
        .await;
        assert!(outcome.is_ok(), "http blackhole call hung past {budget:?}");
        let err = outcome
            .expect("outer timeout")
            .expect_err("blackhole must not succeed");
        assert!(
            matches!(err, ChainError::Unavailable(_)),
            "timeout must classify as Unavailable for failover, got {err}"
        );
        assert!(
            started.elapsed() < budget,
            "elapsed {:?} exceeds budget",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn rpc_call_fails_over_past_blackhole_first_validator() {
        let addr = spawn_blackhole_listener().await;
        // Primary accepts and stays silent. Secondary refuses immediately so
        // the call ends after one connect timeout plus a fast connection error.
        let client = RealChainClient::new(
            vec![format!("ws://{addr}"), "ws://127.0.0.1:1".to_string()],
            "//Alice".to_string(),
            MinerKind::Cpu,
        );
        let started = Instant::now();
        let outcome = tokio::time::timeout(failover_budget(), client.runtime_version_raw()).await;
        assert!(
            outcome.is_ok(),
            "failover hung past {:?}",
            failover_budget()
        );
        let err = outcome
            .expect("outer timeout")
            .expect_err("no working validator in the list");
        assert!(
            matches!(err, ChainError::Unavailable(_)),
            "all-endpoint failure must be Unavailable, got {err}"
        );
        // Must have left the blackhole: elapsed is under one connect budget
        // times a small factor, not an unbounded hang.
        assert!(
            started.elapsed() < failover_budget(),
            "elapsed {:?} exceeds failover budget",
            started.elapsed()
        );
        // And it must have waited long enough that the connect timeout fired
        // (not instant success / skip). A refused-only pair would be near 0.
        assert!(
            started.elapsed() >= RPC_CONNECT_TIMEOUT,
            "expected at least one connect timeout ({RPC_CONNECT_TIMEOUT:?}), got {:?}",
            started.elapsed()
        );
    }
}
