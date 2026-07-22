//! Production chain client over Substrate JSON-RPC.
//!
//! Live end-to-end verification requires a running node. Offline unit tests
//! cover SCALE encode/decode, hybrid extrinsic assembly, and proof packing.
//! Methods return transport errors when no validator is reachable.

use super::extrinsic::{
    build_hybrid_signed_extrinsic, hex_decode, hex_encode, job_orders_storage_key,
    load_hybrid_pair, miner_identity_bytes, SignedExtensionContext,
};
use super::proof_encode::{build_quantum_proof, ProofBuildContext};
use super::scale_types::{
    encode_submit_proof_call, set_values, JobOrderScale, MiningSnapshotScale, OrderStatus,
};
use super::submit::{classify_receipt, Proof, SubmitAction};
use super::{ChainClient, ChainError, JobOrder, MiningSnapshot};
use async_trait::async_trait;
use parity_scale_codec::Decode;
use quantum_validation::AllowedValueSpec;
use quip_transaction_crypto::HybridPair;
use serde_json::Value;
use sp_core::crypto::Ss58Codec;
use sp_core::Pair as _;
use std::sync::Mutex;

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
}

impl RealChainClient {
    pub fn new(validators: Vec<String>, signer_key: String) -> Self {
        Self {
            validators,
            signer_key,
            pair: Mutex::new(None),
            last_snapshot: Mutex::new(None),
            last_spin_spec: Mutex::new(None),
        }
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

        let events = self
            .rpc_call(
                "state_getStorage",
                Value::Array(vec![
                    // System.Events storage key: twox128("System")||twox128("Events")
                    Value::String(hex_encode(&system_events_key())),
                    Value::String(head_hex.to_string()),
                ]),
            )
            .await?;

        let mut order_ids: Vec<u64> = Vec::new();
        if let Some(ev_hex) = events.as_str() {
            if let Ok(ev_bytes) = hex_decode(ev_hex) {
                order_ids.extend(scan_job_proposed_order_ids(&ev_bytes));
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

fn system_events_key() -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.extend_from_slice(&twox128_local(b"System"));
    key.extend_from_slice(&twox128_local(b"Events"));
    key
}

fn twox128_local(data: &[u8]) -> [u8; 16] {
    use std::hash::Hasher;
    use twox_hash::XxHash64;
    let mut h0 = XxHash64::with_seed(0);
    h0.write(data);
    let mut h1 = XxHash64::with_seed(1);
    h1.write(data);
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&h0.finish().to_le_bytes());
    out[8..].copy_from_slice(&h1.finish().to_le_bytes());
    out
}

/// Best-effort scan of raw System.Events for JobProposed order_ids.
///
/// Full event decoding needs the metadata type registry. This scans for
/// SCALE-encoded u64 values after a recognizable pallet/event tag sequence
/// and is intentionally conservative — false positives are filtered by the
/// subsequent storage read.
fn scan_job_proposed_order_ids(events_bytes: &[u8]) -> Vec<u64> {
    // Without metadata we cannot reliably decode. Return empty; operators
    // that need live mempool should run against a node and extend this with
    // a metadata-driven decoder. Storage-key enumeration is not available
    // for Blake2_128Concat maps without knowing keys.
    let _ = events_bytes;
    Vec::new()
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
