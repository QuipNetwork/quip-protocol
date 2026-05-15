# quip-miner

> **Experimental software.** Use at your own risk. No production warranties.

A Python mining stack for the [quip-protocol-rs](https://gitlab.com/quip.network/quip-protocol-rs) Substrate chain. Drives CPU SA, GPU (CUDA / Metal / Modal), and QPU (D-Wave) miners against the chain's `QuantumPow` pallet — fetch the mining snapshot at each new chain head, search for valid Ising solutions, submit `QuantumPow.submit_proof` extrinsics, repeat.

This is the `v0.2` line of the repository (formerly `quip-protocol`). In `v0.1` this codebase shipped its own consensus, P2P (QUIC), block store, REST API, and SPHINCS+ block signer. `v0.2` removes all of that — the chain is the source of truth, miners attach to it.

## Architecture

```
                              chain (substrate)
                                     │
                              ws://localhost:9944
                                     │
                  ┌──────────────────┴──────────────────┐
                  │       SubstrateClient (read)         │
                  │   - get_mining_snapshot              │
                  │   - subscribe_new_heads              │
                  │   - submit_extrinsic                 │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │  SubstrateMinerController            │
                  │   - on new head: cancel + fetch +    │
                  │     dispatch                         │
                  │   - on result: encode + submit       │
                  │   - classify receipts                │
                  └──────┬────────────────────┬──────────┘
                         │                    │
                  ┌──────┴──────┐      ┌──────┴──────┐
                  │ MinerCore   │      │ TelemetryApi│
                  │ - handles[] │      │ /api/v1/*   │
                  │ - stats     │      └─────────────┘
                  │ - descriptor│
                  └──────┬──────┘
                         │
                  MinerHandle (per worker process)
                         │
                  BaseMiner.mine_work_item
                  (CPU SA / GPU CUDA|Metal|Modal / QPU)
```

Component responsibilities (`shared/`):

| module | role |
|--------|------|
| `signer.py` | Abstract `Signer` + `Sr25519Signer`. Phase 7 adds `HybridSigner` (sr25519 + ML-DSA-44). |
| `keystore.py` | sr25519 keystore (`0o600` JSON; plaintext seed for dev). |
| `substrate_client.py` | `py-substrate-interface` async wrapper; `state_call` for the mining snapshot. |
| `substrate_types.py` | `SubstrateMiningContext`, `SubstrateDifficulty`, `MinerInfo`, `ExtrinsicReceipt`. |
| `substrate_submitter.py` | `MiningResult` → `QuantumProof` SCALE encoding + submission. |
| `substrate_miner_controller.py` | Head subscription, snapshot fetch, dispatch, receipt classification. |
| `miner_core.py` | Owns persistent `MinerHandle` workers, hardware descriptor cache, aggregate stats. |
| `miner_bootstrap.py` | Idempotent fund + register pipeline. |
| `telemetry_api.py` | HTTP REST surface (`/api/v1/status`, `/system`, `/stats`, `/block/*`). |
| `base_miner.py` | Protocol-neutral `mine_work_item(context, stop_event)` loop. |
| `miner_worker.py` | 2-process worker scaffolding (parent ↔ child mp.Queue + stop_event). |
| `quantum_proof_of_work.py` | `derive_nonce`, `generate_ising_model_from_nonce`, `evaluate_sampleset`. |

Standalone scripts at repo root:

| file | role |
|------|------|
| `faucet_bot.py` | Dev-only HTTP faucet (self-contained; no `shared/` imports). |
| `quip_cli.py` | `quip-miner` CLI dispatch (keygen / bootstrap / cpu / gpu / qpu). |

## Installation

```bash
python3 -m venv .quip
source .quip/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

Dependencies pulled in by `pyproject.toml`:
- `substrate-interface>=1.7.4`, `scalecodec>=1.2` — chain RPC + SCALE
- `dwave-ocean-sdk>=9.0.0,<10`, `numpy>=1.24.0` — Ising sampling
- `aiohttp>=3.9.0` — telemetry server + faucet
- `click>=8.1.7` — CLI
- `blake3>=1.0.5` — nonce derivation

D-Wave QPU access requires `DWAVE_API_KEY` in `.env` (loaded via `python-dotenv`).

## Quick start

In one terminal, bring up the chain:

```bash
cd ../quip-protocol-rs
docker compose up -d
docker compose logs -f node1   # confirm blocks being produced
```

In another terminal, start the dev faucet (standalone script — does not require `pip install -e .`):

```bash
python faucet_bot.py --node-url ws://localhost:9944 --faucet-key //Alice
```

Bootstrap a miner account (generates a keystore, funds it via the faucet, sudo-seeds `Difficulty` + `DefaultTopology` on a fresh chain, then submits `register_miner`):

```bash
quip-miner bootstrap \
    --node-url ws://localhost:9944 \
    --faucet-url http://127.0.0.1:8087 \
    --seed-chain
```

Run the miner:

```bash
quip-miner cpu \
    --node-url ws://localhost:9944 \
    --num-cpus 4 \
    --topology zephyr:9,2 \
    --rest-port 8086
```

In a third terminal, watch chain events for `QuantumPow.ProofAccepted` (via [polkadot.js](https://polkadot.js.org/apps/#/explorer) pointed at `ws://localhost:9944`), or hit the local telemetry API:

```bash
curl http://localhost:8086/api/v1/status   | jq
curl http://localhost:8086/api/v1/stats    | jq
curl http://localhost:8086/api/v1/system   | jq
```

## CLI reference

### `quip-miner keygen`

Generate a fresh sr25519 signing key. Writes a `0o600` JSON keystore with the seed in plaintext (passphrase-encrypted keystores ship in Phase 7).

```
quip-miner keygen --out ~/.quip-miner/signing.json
```

### `quip-miner bootstrap`

Idempotent setup: generate keystore (if missing) → request funds from the faucet → submit `register_miner`. With `--seed-chain`, also sudo-submits `set_difficulty` + `register_topology` if missing.

```
quip-miner bootstrap \
    --node-url ws://localhost:9944 \
    --signer-key ~/.quip-miner/signing.json \
    --faucet-url http://127.0.0.1:8087 \
    --seed-chain \
    --seed-topology 9,2
```

Re-runs are no-ops that just verify state.

### `quip-miner cpu | gpu | qpu`

Run the mining controller. All three subcommands share these flags:

- `--node-url ws://...` (required) — substrate WS endpoint
- `--signer-key ~/.quip-miner/signing.json` — keystore path
- `--topology zephyr:M,T` — sampler topology (defaults to `zephyr:9,2`)
- `--rest-port 8086` — HTTP telemetry port (`-1` disables)

The `cpu` subcommand adds `--num-cpus N`; `gpu` adds `--gpu-backend {local,metal,modal}`; `qpu` adds `--qpu-type` and `--daily-budget`.

Topology binding is enforced at startup: the CLI hashes the configured topology with the same `blake2_256(SCALE((sorted_nodes, canonical_edges)))` recipe the chain uses, and refuses to start if the hash doesn't match the chain's registered topology.

### `faucet_bot.py` (standalone)

Dev-only HTTP faucet, deployable independently of the rest of the repo. Refuses to bind unless the connected chain matches a known dev-chain prefix (`Development`, `Local Testnet`, `quip-local`). Rate-limited per destination.

```
python faucet_bot.py \
    --node-url ws://localhost:9944 \
    --faucet-key //Alice \
    --listen 127.0.0.1 \
    --port 8087
```

## Telemetry REST API

```
GET  /health
GET  /api/v1/status                        chain head + miner identity + is_mining
GET  /api/v1/system                        hardware descriptor (cached)
GET  /api/v1/stats                         aggregate MinerCore + controller stats
GET  /api/v1/block/latest                  substrate-fetched chain head
GET  /api/v1/block/{n}                     substrate-fetched block by number
GET  /api/v1/block/{n}/header              header subset
POST /api/v1/solve                         disabled in v0.2 (was direct DWave sample)
```

Response envelope: `{"success": bool, "data": ..., "error": ..., "timestamp": int}`.

The legacy `/api/v1/peers`, `/api/v1/join`, `/api/v1/gossip`, `/api/v1/heartbeat`, and `POST /api/v1/block` paths are removed — they were P2P / consensus surfaces with no equivalent in substrate mode. The legacy `/telemetry/*` SSE stream and per-peer aggregator also moved out; consumers should switch to substrate-side events and Prometheus (`http://localhost:9615/metrics`).

## Topology

Mining works against any `BoundedVec`-bounded graph registered on chain (`QuantumPow.RegisteredTopologies`). The CLI's `--topology zephyr:M,T` constructs a [dwave-networkx](https://docs.ocean.dwavesys.com/projects/dwave-networkx/) Zephyr graph; the chain's `pallets/quantum-pow/src/topology.rs::hash_topology` canonicalizes and `blake2_256`-hashes the result.

`bootstrap --seed-chain --seed-topology 9,2` registers Zephyr Z(9,2) (1368 nodes / 7692 edges — the legacy default that matches the chain's difficulty calibration of `max_energy_milli=-2_500_000`). Smaller graphs (Z(2,2), Z(3,2)) work too but need their own difficulty calibration since their ground-state energy range is much narrower.

## Running tests

```bash
python -m pytest tests/ -v
```

Integration tests against the docker chain auto-skip if `ws://localhost:9944` isn't reachable. The end-to-end controller test (`test_controller_submits_proof_end_to_end`) bootstraps inline and asserts at least one `QuantumPow.ProofAccepted` event lands within 120 seconds.

The cross-language nonce-parity test (`test_derive_nonce_parity.py`) reads `crates/quantum-validation/tests/fixtures/python_parity.json` from a sibling `quip-protocol-rs` checkout. Set `QUIP_RUST_FIXTURE_DIR` if your checkout is elsewhere.

## What changed from v0.1

Removed entirely:

- The local blockchain stack: `shared/block.py`, `shared/node.py`, `shared/network_node.py`, `shared/block_store.py`, `shared/block_synchronizer.py`, `shared/block_requirements.py`, `genesis_block_public.json`.
- The P2P stack: QUIC client/server, SWIM failure detector, peer scorer / ban list, gossip telemetry aggregator, sync wire codecs.
- The legacy signing path: SPHINCS+ block signer + certificate manager.
- CLI: `quip-network-node` and `quip-network-simulator` are gone. Use `quip-miner` instead.

Kept (with rewired backends):

- `/api/v1/status`, `/system`, `/stats`, `/block/*` (now substrate-backed)
- The 2-process worker model (`MinerHandle` ↔ child mp.Queue + stop_event)
- The Ising sampling code (`BaseMiner.mine_work_item`, `quantum_proof_of_work.*`, CPU/GPU/QPU subclasses)
- The hardware descriptor / aggregate stats (now exposed via `MinerCore`)

Migration:

- A `v0.1` miner that ran a single mining node now runs a chain node (quip-protocol-rs docker compose) + `quip-miner cpu|gpu|qpu` against it.
- Mining rewards now accrue on chain to the registered sr25519 account, not as v0.1 block-proposer credit.

## License

AGPL-3.0-or-later. See LICENSE.
