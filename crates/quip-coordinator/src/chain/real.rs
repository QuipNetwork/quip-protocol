//! Production chain client over Substrate JSON-RPC.
//!
//! Live end-to-end verification requires a running node. Offline unit tests
//! cover SCALE encode/decode, hybrid extrinsic assembly, and proof packing.
//! Methods return transport errors when no validator is reachable.

use super::extrinsic::{
    build_hybrid_signed_extrinsic, difficulties_storage_key, hex_decode, hex_encode,
    job_orders_storage_key, last_proof_block_storage_key, load_hybrid_pair, miner_identity_bytes,
    topology_curve_c_storage_key, SignedExtensionContext,
};
use super::proof_encode::{build_quantum_proof, ProofBuildContext};
use super::scale_types::{
    encode_submit_proof_call, set_values, CurveCScale, DifficultyConfig, JobOrderScale,
    MiningSnapshotScale, OrderStatus,
};
use super::submit::{classify_receipt, Proof, SubmitAction};
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
use subxt::config::substrate::H256;
use subxt::ext::scale_value::{Composite, Primitive, ValueDef};
use tokio::sync::OnceCell;

/// subxt client for read-only, metadata-aware event decoding. Submit stays on
/// the hybrid-signed jsonrpsee path — subxt is used only to decode mempool
/// `JobProposed` events, which needs the runtime type registry.
type SubxtClient = subxt::OnlineClient<subxt::SubstrateConfig>;

/// Production chain client (RPC + hybrid-signed submit).
pub struct RealChainClient {
    pub validators: Vec<String>,
    pub signer_key: String,
    /// Cached hybrid pair (loaded lazily from `signer_key`).
    pair: Mutex<Option<HybridPair>>,
    /// Last fetched mining snapshot — used by `submit_proof` to pack spins
    /// and derive the nonce. Updated by `fetch_mining_snapshot`.
    last_snapshot: Mutex<Option<MiningSnapshot>>,
    /// Last allowed_spin spec (full AllowedValueSpec, not just Set values).
    last_spin_spec: Mutex<Option<AllowedValueSpec<Vec<i32>>>>,
    /// Lazily-connected subxt client for mempool `JobProposed` event decoding.
    subxt: OnceCell<SubxtClient>,
}

impl RealChainClient {
    pub fn new(validators: Vec<String>, signer_key: String) -> Self {
        Self {
            validators,
            signer_key,
            pair: Mutex::new(None),
            last_snapshot: Mutex::new(None),
            last_spin_spec: Mutex::new(None),
            subxt: OnceCell::new(),
        }
    }

    /// Lazily connect the read-only subxt client to the primary validator.
    async fn subxt_client(&self) -> Result<&SubxtClient, ChainError> {
        let url = self.primary_url()?.to_string();
        self.subxt
            .get_or_try_init(|| async move {
                SubxtClient::from_url(&url)
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

    async fn rpc_call(&self, method: &str, params: Value) -> Result<Value, ChainError> {
        let url = self.primary_url()?;
        rpc_request(url, method, params).await
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
}

#[async_trait]
impl ChainClient for RealChainClient {
    async fn fetch_mining_snapshot(
        &self,
        at: Option<[u8; 32]>,
        _miner_account: [u8; 32],
        topology_hash: Option<[u8; 32]>,
    ) -> Result<Option<MiningSnapshot>, ChainError> {
        let block_hash = match at {
            Some(h) => h,
            None => {
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
            }
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
            allowed_h_milli: set_values(&scale.allowed_h_values),
            allowed_j_milli: set_values(&scale.allowed_j_values),
            allowed_spin_milli: set_values(&scale.allowed_spin_values),
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
        let (c_easy_milli, c_knee_milli, c_hard_milli) = curve
            .map(|c| (c.easy_milli, c.knee_milli, c.hard_milli))
            .unwrap_or((
                DEFAULT_C_EASY_MILLI,
                DEFAULT_C_KNEE_MILLI,
                DEFAULT_C_HARD_MILLI,
            ));

        Ok(Some(DecayParams {
            base_max_energy_milli: base
                .map(|d| d.max_energy_milli)
                .unwrap_or(DEFAULT_BASE_MAX_ENERGY_MILLI),
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
        // Discover recent order ids from system events at head, then storage-read
        // each JobOrders(order_id). Without a live node this returns transport
        // errors; with a node, empty open-order sets yield Ok(vec![]).
        let head = self
            .rpc_call("chain_getBlockHash", Value::Array(vec![]))
            .await?;
        let head_hex = head
            .as_str()
            .ok_or_else(|| ChainError::Decode("chain_getBlockHash not a string".into()))?;

        // Decode System.Events at head via subxt (metadata-aware) and collect
        // the order_id of every QuantumComputeMempool::JobProposed. This finds
        // orders proposed in the head block; still-open orders from earlier
        // blocks are re-surfaced as they are re-proposed or by the storage-status
        // filter below. Matches the Python reference (get_events_at → filter
        // module_id/event_id → attributes.order_id).
        let head_bytes = hex_decode(head_hex).map_err(ChainError::Decode)?;
        if head_bytes.len() != 32 {
            return Err(ChainError::Decode(format!(
                "head hash is {} bytes, expected 32",
                head_bytes.len()
            )));
        }
        let head_hash = H256::from_slice(&head_bytes);

        let client = self.subxt_client().await?;
        let at = client
            .at_block(head_hash)
            .await
            .map_err(|e| ChainError::Unavailable(format!("subxt at_block: {e}")))?;
        let events = at
            .events()
            .fetch()
            .await
            .map_err(|e| ChainError::Unavailable(format!("subxt events fetch: {e}")))?;

        let mut order_ids: Vec<u64> = Vec::new();
        for ev in events.iter() {
            let ev = ev.map_err(|e| ChainError::Decode(format!("event decode: {e}")))?;
            if ev.pallet_name() != "QuantumComputeMempool" || ev.event_name() != "JobProposed" {
                continue;
            }
            let fields = ev
                .decode_fields_unchecked_as::<Composite<()>>()
                .map_err(|e| ChainError::Decode(format!("JobProposed fields: {e}")))?;
            if let Some(oid) = order_id_from_fields(&fields) {
                order_ids.push(oid);
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

        // Chain state for signed extensions. `system_accountNextIndex` expects
        // an SS58-encoded address (the node rejects a hex account with a
        // "Base 58 requirement is violated" param error), so encode the
        // derived account with the default SS58 prefix.
        let account_ss58 =
            quip_transaction_crypto::account_id_from_public(&pair.public()).to_ss58check();
        let nonce_val = self
            .rpc_call(
                "system_accountNextIndex",
                Value::Array(vec![Value::String(account_ss58)]),
            )
            .await?;
        let account_nonce = nonce_val.as_u64().unwrap_or(0) as u32;

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
        let spec_version = rv.get("specVersion").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let transaction_version = rv
            .get("transactionVersion")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let signed_ctx = SignedExtensionContext {
            account_nonce,
            genesis_hash,
            spec_version,
            transaction_version,
            tip: 0,
        };
        let ext = build_hybrid_signed_extrinsic(&pair, &call, &signed_ctx);

        // Submit (watch preferred; fall back to fire-and-forget).
        let submit = self
            .rpc_call(
                "author_submitExtrinsic",
                Value::Array(vec![Value::String(hex_encode(&ext))]),
            )
            .await;

        match submit {
            Ok(_) => Ok(SubmitAction::Success),
            Err(ChainError::Submit(msg)) | Err(ChainError::Unavailable(msg)) => {
                Ok(classify_receipt(Some(&msg)))
            }
            Err(ChainError::Decode(msg)) => Err(ChainError::Decode(msg)),
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

/// Extract the `order_id` (u64) from a decoded `QuantumComputeMempool::
/// JobProposed` event's fields. Prefers the named `order_id` field (matching
/// the pallet's named event attributes); falls back to the first field of an
/// unnamed composite. `None` if absent or not an unsigned primitive. Extra
/// fields (proposer, reward, …) are ignored, so it survives event-shape growth.
fn order_id_from_fields(fields: &Composite<()>) -> Option<u64> {
    let value = match fields {
        Composite::Named(named) => named
            .iter()
            .find(|(name, _)| name == "order_id")
            .map(|(_, v)| v),
        Composite::Unnamed(vals) => vals.first(),
    }?;
    match &value.value {
        ValueDef::Primitive(Primitive::U128(n)) => u64::try_from(*n).ok(),
        _ => None,
    }
}

async fn rpc_request(url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
    // Support both ws:// and http(s):// via jsonrpsee.
    if url.starts_with("ws://") || url.starts_with("wss://") {
        rpc_ws(url, method, params).await
    } else {
        rpc_http(url, method, params).await
    }
}

async fn rpc_http(url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
    use jsonrpsee::core::client::ClientT;
    use jsonrpsee::http_client::HttpClientBuilder;

    let client = HttpClientBuilder::default()
        .build(url)
        .map_err(|e| ChainError::Unavailable(format!("http client: {e}")))?;
    let result: Value = client
        .request(method, rpc_params_from_value(params))
        .await
        .map_err(|e| ChainError::Unavailable(format!("rpc {method}: {e}")))?;
    Ok(result)
}

async fn rpc_ws(url: &str, method: &str, params: Value) -> Result<Value, ChainError> {
    use jsonrpsee::core::client::ClientT;
    use jsonrpsee::ws_client::WsClientBuilder;

    let client = WsClientBuilder::default()
        .build(url)
        .await
        .map_err(|e| ChainError::Unavailable(format!("ws client: {e}")))?;
    let result: Value = client
        .request(method, rpc_params_from_value(params))
        .await
        .map_err(|e| ChainError::Unavailable(format!("rpc {method}: {e}")))?;
    Ok(result)
}

fn rpc_params_from_value(params: Value) -> jsonrpsee::core::params::ArrayParams {
    match params {
        Value::Array(arr) => {
            let mut p = jsonrpsee::core::params::ArrayParams::new();
            for v in arr {
                let _ = p.insert(v);
            }
            p
        }
        other => {
            let mut p = jsonrpsee::core::params::ArrayParams::new();
            let _ = p.insert(other);
            p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::order_id_from_fields;
    use subxt::ext::scale_value::{Composite, Value};

    #[test]
    fn order_id_from_named_ignores_other_fields() {
        // Real JobProposed carries more than order_id (proposer, reward, …);
        // the named lookup must pick order_id regardless of position.
        let fields = Composite::Named(vec![
            ("proposer".to_string(), Value::u128(999)),
            ("order_id".to_string(), Value::u128(42)),
            ("reward".to_string(), Value::u128(7)),
        ]);
        assert_eq!(order_id_from_fields(&fields), Some(42));
    }

    #[test]
    fn order_id_from_unnamed_takes_first() {
        let fields = Composite::Unnamed(vec![Value::u128(7), Value::u128(99)]);
        assert_eq!(order_id_from_fields(&fields), Some(7));
    }

    #[test]
    fn order_id_absent_wrong_type_or_empty_is_none() {
        // No order_id field.
        assert_eq!(
            order_id_from_fields(&Composite::Named(vec![("x".to_string(), Value::u128(1))])),
            None
        );
        // order_id present but not an unsigned primitive.
        let nested = Value::named_composite(vec![("inner".to_string(), Value::u128(1))]);
        assert_eq!(
            order_id_from_fields(&Composite::Named(vec![("order_id".to_string(), nested)])),
            None
        );
        // Empty unnamed composite.
        assert_eq!(order_id_from_fields(&Composite::Unnamed(vec![])), None);
    }
}
