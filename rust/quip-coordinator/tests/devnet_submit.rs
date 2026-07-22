//! Live devnet integration test for the v0.3 coordinator chain client.
//!
//! Gated on `QUIP_DEVNET=<ws-url>`; a no-op (passes) when unset so the normal
//! `cargo test` run stays offline. Run against a live node with:
//!
//! ```text
//! QUIP_DEVNET=ws://localhost:9944 \
//!   cargo test -p quip-coordinator --test devnet_submit -- --nocapture --ignored
//! ```
//!
//! Milestone 1 (read path): `RealChainClient::fetch_mining_snapshot` decodes a
//! real `MiningSnapshot` (runtime API `state_call` + SCALE decode) and reports
//! the topology size and default difficulty.
//!
//! Milestone 2 (submit path): register `//Alice` as a miner, derive the PoW
//! Ising from the snapshot, solve it locally (greedy spin-glass descent),
//! re-validate with `quantum_validation` exactly as the pallet does, then drive
//! the coordinator's `RealChainClient::submit_proof` and assert the pallet
//! accepts it (persistent `Miners.proofs_submitted` increment + `ProofAccepted`
//! event), or report the exact pallet error.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use parity_scale_codec::Decode;
use serde_json::{json, Value};

use quantum_validation::{
    calculate_diversity, derive_nonce, energy_of_solution, generate_ising_model, select_diverse,
    MilliValue,
};
use quip_coordinator::chain::extrinsic::{
    build_hybrid_signed_extrinsic, hex_decode, hex_encode, miner_identity_bytes,
    SignedExtensionContext,
};
use quip_coordinator::chain::scale_types::MiningSnapshotScale;
use quip_coordinator::chain::submit::SubmitAction;
use quip_coordinator::chain::{ChainClient, RealChainClient};
use quip_proto::v1::Solution;
use quip_protocol::wire::encode_spins;
use quip_transaction_crypto::{account_id_from_public, HybridPair};
use sp_core::Pair;

const QUANTUM_POW_PALLET: u8 = 10;
const REGISTER_MINER_CALL: u8 = 0;

/// QuantumPow `Error` variants in declaration order (module error index).
const POW_ERRORS: &[&str] = &[
    "MinerAlreadyRegistered",
    "MinerNotRegistered",
    "TopologyAlreadyRegistered",
    "TopologyNotRegistered",
    "InvalidCurve",
    "GraphTooSmall",
    "InvalidTopology",
    "ProofLimitReached",
    "InvalidNonce",
    "NoSolutionsSubmitted",
    "InvalidSpinValues",
    "SolutionLengthMismatch",
    "InsufficientEnergy",
    "InsufficientDiversity",
    "InsufficientSolutions",
    "ArithmeticOverflow",
    "EmptyAllowedValues",
    "EncodingTooWide",
    "PackedSolutionLengthMismatch",
    "InvalidEncodedSpin",
    "PackedSolutionTooLarge",
    "TopologyNotMineable",
    "TopologyIsDefault",
    "MineableTopologyConflict",
];

// ----------------------------------------------------------------------------
// RPC
// ----------------------------------------------------------------------------

async fn rpc_try(url: &str, method: &str, params: Vec<Value>) -> Result<Value, String> {
    use jsonrpsee::core::client::ClientT;
    use jsonrpsee::ws_client::WsClientBuilder;
    let client = WsClientBuilder::default()
        .build(url)
        .await
        .map_err(|e| format!("ws build: {e}"))?;
    let mut p = jsonrpsee::core::params::ArrayParams::new();
    for v in params {
        p.insert(v).map_err(|e| e.to_string())?;
    }
    client
        .request::<Value, _>(method, p)
        .await
        .map_err(|e| e.to_string())
}

async fn rpc(url: &str, method: &str, params: Vec<Value>) -> Value {
    rpc_try(url, method, params)
        .await
        .unwrap_or_else(|e| panic!("rpc {method}: {e}"))
}

async fn get_storage(url: &str, key: &[u8], at: Option<&str>) -> Option<Vec<u8>> {
    let mut params = vec![json!(hex_encode(key))];
    if let Some(h) = at {
        params.push(json!(h));
    }
    let v = rpc(url, "state_getStorage", params).await;
    v.as_str().map(|s| hex_decode(s).expect("hex storage"))
}

// ----------------------------------------------------------------------------
// Storage keys
// ----------------------------------------------------------------------------

fn twox128(data: &[u8]) -> [u8; 16] {
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

fn blake2_128(data: &[u8]) -> [u8; 16] {
    use blake2::digest::{Update, VariableOutput};
    use blake2::Blake2bVar;
    let mut hasher = Blake2bVar::new(16).expect("16-byte blake2b");
    hasher.update(data);
    let mut out = [0u8; 16];
    hasher.finalize_variable(&mut out).expect("finalize");
    out
}

/// `Blake2_128Concat` storage-map key for `pallet.item(account)`.
fn map_key(pallet: &[u8], item: &[u8], acct: &[u8; 32]) -> Vec<u8> {
    let mut k = Vec::new();
    k.extend_from_slice(&twox128(pallet));
    k.extend_from_slice(&twox128(item));
    k.extend_from_slice(&blake2_128(acct));
    k.extend_from_slice(acct);
    k
}

fn events_key() -> Vec<u8> {
    let mut k = Vec::new();
    k.extend_from_slice(&twox128(b"System"));
    k.extend_from_slice(&twox128(b"Events"));
    k
}

/// SCALE-lite mirror of `frame_system::AccountInfo` (only the leading nonce).
#[derive(Decode)]
struct AccountNonce {
    nonce: u32,
    #[allow(dead_code)]
    consumers: u32,
    #[allow(dead_code)]
    providers: u32,
    #[allow(dead_code)]
    sufficients: u32,
}

/// SCALE mirror of `pallet_quantum_pow::MinerInfo<u128, u32>`.
#[derive(Decode)]
struct MinerInfoLite {
    #[allow(dead_code)]
    registered_at: u32,
    #[allow(dead_code)]
    deposit: u128,
    proofs_submitted: u32,
    #[allow(dead_code)]
    proofs_won: u32,
    #[allow(dead_code)]
    rewards_earned: u128,
}

async fn account_nonce(url: &str, acct: &[u8; 32]) -> u32 {
    let key = map_key(b"System", b"Account", acct);
    match get_storage(url, &key, None).await {
        Some(bytes) => AccountNonce::decode(&mut &bytes[..])
            .map(|a| a.nonce)
            .unwrap_or(0),
        None => 0,
    }
}

async fn miner_proofs_submitted(url: &str, acct: &[u8; 32]) -> Option<u32> {
    let key = map_key(b"QuantumPow", b"Miners", acct);
    let bytes = get_storage(url, &key, None).await?;
    MinerInfoLite::decode(&mut &bytes[..])
        .map(|m| m.proofs_submitted)
        .ok()
}

// ----------------------------------------------------------------------------
// Signed-extension context (genesis / spec / tx version)
// ----------------------------------------------------------------------------

async fn fetch_ext_ctx(url: &str, nonce: u32) -> SignedExtensionContext {
    let genesis_hex = rpc(url, "chain_getBlockHash", vec![json!(0)])
        .await
        .as_str()
        .expect("genesis hash")
        .to_string();
    let gb = hex_decode(&genesis_hex).expect("genesis hex");
    let mut genesis_hash = [0u8; 32];
    genesis_hash.copy_from_slice(&gb);
    let rv = rpc(url, "state_getRuntimeVersion", vec![]).await;
    SignedExtensionContext {
        account_nonce: nonce,
        genesis_hash,
        spec_version: rv["specVersion"].as_u64().unwrap() as u32,
        transaction_version: rv["transactionVersion"].as_u64().unwrap() as u32,
        tip: 0,
    }
}

// ----------------------------------------------------------------------------
// Snapshot (full scale form, for the exact allowed-value specs)
// ----------------------------------------------------------------------------

async fn fetch_scale_snapshot(url: &str) -> MiningSnapshotScale {
    let head = rpc(url, "chain_getBlockHash", vec![])
        .await
        .as_str()
        .expect("head")
        .to_string();
    // Runtime API arg: Option<H256>::None.
    let result = rpc(
        url,
        "state_call",
        vec![
            json!("QuantumPowApi_mining_snapshot"),
            json!(hex_encode(&[0u8])),
            json!(head),
        ],
    )
    .await;
    let bytes = hex_decode(result.as_str().expect("snapshot hex")).expect("hex");
    let decoded: Option<MiningSnapshotScale> =
        Decode::decode(&mut &bytes[..]).expect("decode MiningSnapshotScale");
    decoded.expect("snapshot is Some")
}

// ----------------------------------------------------------------------------
// Extrinsic submit + confirmation
// ----------------------------------------------------------------------------

async fn head_number(url: &str) -> u64 {
    let header = rpc(url, "chain_getHeader", vec![]).await;
    let n = header["number"].as_str().expect("number");
    u64::from_str_radix(n.trim_start_matches("0x"), 16).unwrap()
}

/// Submit a raw extrinsic and wait until `ready(state)` holds (or timeout).
/// Returns the observed head block number when the condition first held.
async fn submit_and_confirm<F, Fut>(
    url: &str,
    ext_hex: &str,
    label: &str,
    ready: F,
) -> Result<u64, String>
where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let tx = rpc_try(url, "author_submitExtrinsic", vec![json!(ext_hex)]).await;
    match &tx {
        Ok(h) => println!("  [{label}] injected tx {}", h.as_str().unwrap_or("?")),
        Err(e) => return Err(format!("author_submitExtrinsic rejected: {e}")),
    }
    for _ in 0..40 {
        let url_owned = url.to_string();
        if ready(url_owned).await {
            return Ok(head_number(url).await);
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
    }
    Err(format!("[{label}] condition not met within timeout"))
}

/// Scan raw `System.Events` across recent blocks for a QuantumPow module error
/// or a QuantumPow event naming `acct`. Returns a human-readable finding.
async fn scan_recent_events(url: &str, acct: &[u8; 32], depth: u64) -> Option<String> {
    let head = head_number(url).await;
    let mut seen = HashSet::new();
    for n in (head.saturating_sub(depth)..=head).rev() {
        let hash = rpc(url, "chain_getBlockHash", vec![json!(n)]).await;
        let Some(hash) = hash.as_str() else { continue };
        if !seen.insert(hash.to_string()) {
            continue;
        }
        let Some(bytes) = get_storage(url, &events_key(), Some(hash)).await else {
            continue;
        };
        if let Some(f) = scan_events_blob(&bytes, acct) {
            return Some(format!("block #{n}: {f}"));
        }
    }
    None
}

fn scan_events_blob(bytes: &[u8], acct: &[u8; 32]) -> Option<String> {
    // System(0).ExtrinsicFailed(1) with DispatchError::Module(3){index,error[4]}.
    for i in 0..bytes.len().saturating_sub(6) {
        if bytes[i] == 0x00 && bytes[i + 1] == 0x01 && bytes[i + 2] == 0x03 {
            let module = bytes[i + 3];
            let err = bytes[i + 4];
            let name = if module == QUANTUM_POW_PALLET {
                POW_ERRORS.get(err as usize).copied().unwrap_or("Unknown")
            } else {
                "(non-QuantumPow pallet)"
            };
            return Some(format!(
                "ExtrinsicFailed Module{{ index: {module}, error: {err} }} => {name}"
            ));
        }
    }
    // QuantumPow(10) event naming our account (MinerRegistered / ProofAccepted:
    // both encode the miner AccountId immediately after pallet+variant byte).
    for i in 0..bytes.len().saturating_sub(2 + 32) {
        if bytes[i] == QUANTUM_POW_PALLET && bytes[i + 2..i + 2 + 32] == acct[..] {
            return Some(format!(
                "QuantumPow event variant {} for miner (matches account)",
                bytes[i + 1]
            ));
        }
    }
    None
}

// ----------------------------------------------------------------------------
// Local spin-glass solver + pallet-mirror validation
// ----------------------------------------------------------------------------

/// Greedy single-flip descent to a local minimum from a random start.
fn solve_one(
    rng: &mut impl rand::Rng,
    n: usize,
    h: &[MilliValue],
    adj: &[Vec<(usize, i32)>],
) -> Vec<i8> {
    let mut s: Vec<i8> = (0..n)
        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
        .collect();
    let mut lf: Vec<i64> = (0..n)
        .map(|p| {
            h[p] as i64
                + adj[p]
                    .iter()
                    .map(|&(q, j)| j as i64 * s[q] as i64)
                    .sum::<i64>()
        })
        .collect();
    loop {
        let mut improved = false;
        for p in 0..n {
            // Flip lowers energy iff s[p] * local_field[p] > 0.
            if (s[p] as i64) * lf[p] > 0 {
                let ds = (-2 * s[p]) as i64;
                s[p] = -s[p];
                for &(q, j) in &adj[p] {
                    lf[q] += j as i64 * ds;
                }
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    s
}

fn build_adjacency(
    nodes: &[u32],
    edges: &[(u32, u32)],
    j: &[MilliValue],
) -> Vec<Vec<(usize, i32)>> {
    let pos: HashMap<u32, usize> = nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut adj = vec![Vec::new(); nodes.len()];
    for (e, &(u, v)) in edges.iter().enumerate() {
        let (pu, pv) = (pos[&u], pos[&v]);
        adj[pu].push((pv, j[e]));
        adj[pv].push((pu, j[e]));
    }
    adj
}

// ----------------------------------------------------------------------------
// Test
// ----------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires a live devnet; set QUIP_DEVNET=ws://host:port"]
async fn devnet_submit_proof_end_to_end() {
    let Ok(url) = std::env::var("QUIP_DEVNET") else {
        eprintln!("QUIP_DEVNET unset; skipping live devnet test");
        return;
    };

    let alice = HybridPair::from_string("//Alice", None).expect("//Alice");
    let alice_acct = account_id_from_public(&alice.public());
    let alice_bytes: [u8; 32] = *AsRef::<[u8; 32]>::as_ref(&alice_acct);

    let client = RealChainClient::new(vec![url.clone()], "//Alice".to_string());

    // ---------------- Milestone 1: read path ----------------
    let snap = client
        .fetch_mining_snapshot(None, alice_bytes, None)
        .await
        .expect("fetch_mining_snapshot RPC")
        .expect("snapshot present");
    println!(
        "M1 snapshot: nodes={} edges={} min_solutions={} max_energy_milli={} min_diversity_milli={} last_proof_block_hash={}",
        snap.nodes.len(),
        snap.edges.len(),
        snap.min_solutions,
        snap.max_energy_milli,
        snap.min_diversity_milli,
        hex_encode(&snap.last_proof_block_hash),
    );
    assert!(
        (4000..5000).contains(&snap.nodes.len()),
        "expected ~4578 nodes, got {}",
        snap.nodes.len()
    );
    assert_eq!(snap.min_solutions, 5, "default min_solutions");
    assert_eq!(
        snap.max_energy_milli, -1_200_000,
        "default max_energy_milli"
    );
    assert_eq!(snap.min_diversity_milli, 200, "default min_diversity_milli");
    println!("M1 PASS: real MiningSnapshot decoded from live v0.2 chain");

    // ---------------- Milestone 2: submit path ----------------
    // Full-precision snapshot for the exact allowed-value specs (needed to
    // regenerate the same Ising the pallet will).
    let scale = fetch_scale_snapshot(&url).await;
    let nodes = &scale.nodes;
    let edges = &scale.edges;
    let last_proof = snap.last_proof_block_hash;

    // (1) Register //Alice as a miner (idempotent).
    if miner_proofs_submitted(&url, &alice_bytes).await.is_none() {
        let nonce = account_nonce(&url, &alice_bytes).await;
        let ctx = fetch_ext_ctx(&url, nonce).await;
        let call = vec![QUANTUM_POW_PALLET, REGISTER_MINER_CALL];
        let ext = build_hybrid_signed_extrinsic(&alice, &call, &ctx);
        let block = submit_and_confirm(&url, &hex_encode(&ext), "register_miner", |u| async move {
            miner_proofs_submitted(&u, &alice_bytes).await.is_some()
        })
        .await
        .expect("register_miner not confirmed");
        println!("  register_miner: Miners[//Alice] present by block #{block}");
    } else {
        println!("  register_miner: //Alice already registered, skipping");
    }
    let proofs_before = miner_proofs_submitted(&url, &alice_bytes)
        .await
        .expect("miner registered");

    // (2) Derive the PoW nonce + Ising, exactly as the pallet does.
    let salt: [u8; 32] = rand::random();
    let miner_id = miner_identity_bytes(&alice);
    let nonce = derive_nonce(&last_proof, &miner_id, &salt);
    let (h, j) = generate_ising_model(
        nonce,
        nodes,
        edges,
        &scale.allowed_h_values.as_slice(),
        &scale.allowed_j_values.as_slice(),
    )
    .expect("generate_ising_model");
    println!(
        "  Ising derived: |h|={} |j|={} nonce=0x{:x}",
        h.len(),
        j.len(),
        nonce
    );

    // (3) Solve locally: collect >= min_solutions distinct qualifying solutions.
    let adj = build_adjacency(nodes, edges, &j);
    let ceiling = snap.max_energy_milli;
    let mut rng = rand::thread_rng();
    let mut valid: Vec<(Vec<i8>, i64)> = Vec::new();
    let want = (snap.min_solutions as usize + 3).max(6);
    for _ in 0..80 {
        let s = solve_one(&mut rng, nodes.len(), &h, &adj);
        let e = energy_of_solution(&s, &h, edges, &j, nodes).expect("energy");
        if e < ceiling && !valid.iter().any(|(o, _)| *o == s) {
            valid.push((s, e));
            if valid.len() >= want {
                break;
            }
        }
    }
    assert!(
        valid.len() >= snap.min_solutions as usize,
        "solver found only {} solutions below {ceiling}",
        valid.len()
    );

    // (4) Re-validate exactly as pallet `validate_proof` (energy + diversity).
    let valid_slices: Vec<&[i8]> = valid.iter().map(|(s, _)| s.as_slice()).collect();
    let target = valid_slices.len().min((snap.min_solutions.max(1)) as usize);
    let selected_idx = select_diverse(&valid_slices, target).expect("select_diverse");
    let selected: Vec<&[i8]> = selected_idx.iter().map(|&i| valid_slices[i]).collect();
    let diversity = calculate_diversity(&selected).expect("diversity");
    let best_energy = valid.iter().map(|(_, e)| *e).min().unwrap();
    println!(
        "  local validation: valid_count={} best_energy_milli={} diversity_milli={} (need count>={}, energy<{}, diversity>={})",
        valid.len(),
        best_energy,
        diversity,
        snap.min_solutions,
        ceiling,
        snap.min_diversity_milli
    );
    assert!(best_energy < ceiling, "best energy not below ceiling");
    assert!(
        diversity >= snap.min_diversity_milli,
        "diversity {diversity} below {}",
        snap.min_diversity_milli
    );

    // (5) Drive the coordinator's real submit path.
    let proof = quip_coordinator::chain::submit::Proof {
        job_id: vec![],
        best_energy_milli: best_energy,
        diversity_milli: diversity,
        n_valid: valid.len() as u32,
        solutions: valid
            .iter()
            .map(|(s, e)| Solution {
                spins_bytes: encode_spins(s),
                energy_milli: *e,
            })
            .collect(),
        is_pow: true,
        order_id: vec![],
        generation: 0,
        salt: salt.to_vec(),
        device_access_time_us: 0,
    };
    let action = client.submit_proof(&proof).await;
    println!("  RealChainClient::submit_proof -> {action:?}");
    assert!(
        matches!(action, Ok(SubmitAction::Success)),
        "submit_proof RPC injection failed: {action:?}"
    );

    // (6) Assert on-chain acceptance: persistent proofs_submitted increment.
    let mut accepted_at = None;
    for _ in 0..40 {
        if let Some(now) = miner_proofs_submitted(&url, &alice_bytes).await {
            if now > proofs_before {
                accepted_at = Some(now);
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
    }
    let event = scan_recent_events(&url, &alice_bytes, 8).await;

    match accepted_at {
        Some(now) => {
            println!(
                "M2 PASS: submit_proof ACCEPTED (proofs_submitted {proofs_before} -> {now}); event: {}",
                event.as_deref().unwrap_or("(not decoded)")
            );
        }
        None => {
            panic!(
                "M2 FAIL: submit_proof not accepted on-chain. Latest event finding: {}",
                event.as_deref().unwrap_or("(none found)")
            );
        }
    }
}
