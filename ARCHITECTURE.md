# QUIP Protocol Architecture

This document describes the runtime architecture of `quip-network-node` as
of v0.2 (post Plans 1–4). It covers the four pillars the system runs on:

1. **The controller main loop** (PoW + mempool)
2. **The miner layer** (CPU / CUDA / Metal / QPU)
3. **The validator connection pool** (hot-active swap, per-URL child process)
4. **The chain event manager** (adaptive polling, watchdog)

Everything below describes code that exists in `main` today. Items
that exist but are vestigial are flagged in §9 Cleanup candidates.

---

## 1. Process topology

A running miner node is up to **four** OS processes:

```
┌─ controller process (asyncio) ─────────────────────────────────┐
│   quip_cli._run_concurrent_miner()                             │
│     ├─ SubstrateMinerController   (PoW, optional)              │
│     ├─ MempoolMinerController     (mempool, optional)          │
│     ├─ ValidatorPool              (shared by both controllers) │
│     │     └─ one ValidatorHandle  (active URL only)            │
│     ├─ ChainEventManager          (PoW only)                   │
│     └─ in-process TelemetryApiServer (legacy, default-on)      │
└────────────────────────────────────────────────────────────────┘
        │ mp.Process per miner backend         │ mp.Process per active URL
        ▼                                       ▼
┌─ MinerHandle child (×N) ──┐         ┌─ validator_main child ──┐
│   miner_worker_main()     │         │  SubstrateClient(url)   │
│   owns one BaseMiner      │         │  asyncio loop           │
│   (SA / CUDA / Metal /    │         │  serves req_q → resp_q  │
│    DWave / Modal)         │         └─────────────────────────┘
└───────────────────────────┘
        │ resp queue                           │ resp queue
        ▼                                       ▼
                  controller main loop

┌─ telemetry sibling (optional, opt-in) ──┐
│  shared/telemetry_process.telemetry_main │
│  aiohttp app                              │
│  reads ${runtime_dir}/telemetry-stats.json│
└───────────────────────────────────────────┘
```

Cancellation, errors, and lifecycle:

- Every miner child registers a SIGTERM handler for hardware cleanup.
- Long-lived asyncio tasks (drainers, event manager loops, snapshot
  writer, mempool subscription, claim loop) are wrapped in
  `supervise(coro, name, on_failure=...)`. An unhandled exception
  triggers controller shutdown — silent task death is the original
  bug class this design eliminates.
- The validator child is killed and respawned on connection-class
  errors (hot-active swap, §4).

---

## 2. Controller main loop

### 2.1 PoW: `SubstrateMinerController` (`substrate/miner_controller.py`)

Entry: `run()` → `_main_loop()`.

**Wait set** (`_main_loop`, ~`miner_controller.py:695`):
- `_result_queue.get()` — mining results from drainer tasks
- `_shutdown_event.wait()` — graceful shutdown

Long-lived supervised tasks:

| Task | Source | Purpose |
|---|---|---|
| `_drain_handle_loop(handle)` (×N) | `miner_controller.py:524` | Pull `resp_q` from each miner child, post `_ResultEnvelope` onto `_result_queue` |
| `ChainEventManager.run()` | `miner_controller.py:620` | Poll snapshot, fire `on_new_head` |
| Stats snapshot writer | `miner_controller.py:553` | Write `telemetry-stats.json` every 1s |
| Telemetry sibling (`mp.Process`) | `miner_controller.py:519` | Opt-in via `telemetry_port=` constructor arg |

**`on_new_head(ctx)`** (`miner_controller.py:748`) — the event-driven
work dispatcher. Guards in order:

1. `None` snapshot → bump `stats.none_snapshots_seen`, return
2. Topology hash mismatch → raise `_OperatorFailLoud`
3. Threshold changed → `handle.set_live_threshold_milli(...)` on each handle
4. Zero seed with `_highest_handled_block > 0` → drop (transient)
5. Work key in `_closed_work_keys` → return (already won)
6. All handles idle on current key → return (same-key skip)
7. Cancel prior dispatch, await `done` sentinel per handle (500ms timeout)
8. `handle.mine_work_item(ctx)` on each handle; record context in `_dispatch_contexts[(handle_id, dispatch_id)]`

**`_handle_result(envelope)`** (`miner_controller.py:882`):

- Duplicate-drop if work key already closed (sibling won)
- Stale-drop if envelope's work key ≠ current
- Encode proof → `submit_proof()` → `pool.send("submit_extrinsic", ...)`
- Classify receipt: `ok` (verify recorded on-chain, mark closed,
  cancel siblings) / `stale` (drop) / `fatal` (shutdown)

### 2.2 Mempool: `MempoolMinerController` (`shared/mempool_miner_controller.py`)

Entry: `run()` → `_main_loop()`.

**Wait set** (`_wait_for_event`, ~`mempool_miner_controller.py:524`):
- `_head_signal.wait()`
- `_result_queue.get()`
- `_shutdown_event.wait()`

Long-lived supervised tasks:

| Task | Purpose |
|---|---|
| `_drain_handle_loop(handle)` (×N) | Same shape as PoW |
| `_subscribe_heads()` | **Still uses `client.subscribe_new_heads()`** — has NOT moved to ChainEventManager |
| `_periodic_claim_loop()` | Submit `claim_reward` extrinsic for expired orders every 30s |

On new head: query `get_events_at(block_hash)`, route `JobProposed`
events through `_consider_order()` → enqueue eligible orders →
`_maybe_dispatch_next()`.

### 2.3 CLI orchestration (`quip_cli.py:_run_concurrent_miner`)

Sequence (~`quip_cli.py:901`):

1. Load keystore, connect `ValidatorPool`, ensure funded, auto-identify
2. Query snapshot, validate topology
3. Split `miner_handles` by `--mode`:
   - `pow` → all to PoW controller
   - `mempool` → all to mempool controller
   - `both` → `floor(n/2)` to PoW, rest to mempool
4. Construct controllers; share the same pool
5. Spawn in-process `TelemetryApiServer` (legacy, default-on)
6. Install SIGINT/SIGTERM → `shutdown()` on both controllers
7. `asyncio.wait(..., FIRST_COMPLETED)` until one exits, then drain the
   other with a 15s grace period, then `pool.close()`

---

## 3. Miner layer

### 3.1 Class hierarchy

```
BaseMiner                          shared/base_miner.py
├── SimulatedAnnealingMiner        CPU/sa_miner.py
├── GPUMiner
│   ├── CudaMiner                  GPU/cuda_miner.py
│   ├── MetalMiner                 GPU/metal_miner.py
│   └── ModalMiner                 GPU/modal_miner.py
└── DWaveMiner                     QPU/dwave_miner.py
```

Each subclass implements:

- `_sample(h, J, num_reads, num_sweeps, **kwargs) → dimod.SampleSet`
- `_adapt_mining_params(requirements, nodes, edges) → dict`

Optional hooks: `_pre_mine_setup`, `_post_sample`, `_post_mine_cleanup`,
`_on_sampling_error`. The QPU implementation uses `_pre_mine_setup` to
gate the daily budget; `_post_mine_cleanup` releases GPU resources.

The protocol-neutral entry is `BaseMiner.mine_work_item(context,
stop_event)` (`base_miner.py:339`). It accepts a `WorkContext` Protocol
satisfied by both `SubstrateMiningContext` and `MempoolJobContext`. The
loop iterates:

```
while not stop_event.is_set():
    salt = fresh_salt()
    h, J, nonce = context.resolve_ising(salt)
    sampleset = self._sample(h, J, num_reads=..., num_sweeps=...)
    sampleset = self._post_sample(sampleset)
    result = self.evaluate_sampleset(sampleset, requirements, ..., strict_energy=mempool)
    if result: return result
return None
```

PoW uses **ratchet mode** (lenient eval; tracks best-so-far; reads
`live_threshold_milli` from shared `mp.Value` every iteration). Mempool
uses **strict mode** (energy gate always applied).

### 3.2 `MinerHandle` (`shared/miner_worker.py:263`)

The controller's view of one miner child. IPC primitives:

- `req: mp.Queue` — controller → worker (RPC-style ops)
- `resp: mp.Queue` — worker → controller (results + sentinels)
- `stop_event: mp.Event` — worker polls each iteration
- `live_max_energy_milli: mp.Value('q')` — shared i64 ratchet threshold

Operations:

| Method | Effect |
|---|---|
| `mine_work_item(ctx) → dispatch_id` | Clear stop_event, enqueue `{op: "mine_work_item", context, dispatch_id}`, return id |
| `cancel()` | Set stop_event directly (worker observes within one iteration) |
| `set_live_threshold_milli(milli)` | Atomic write to shared `mp.Value` |
| `get_stats()` | Request/reply via queues |

The worker (`miner_worker_main`) is a simple op dispatcher
(`miner_worker.py:131`). Mining results land on `resp_q` as either
`{"op": "mine_result", "dispatch_id", "result"}` or
`{"op": "work_item_done", "dispatch_id"}` (sentinel for cancelled
dispatches).

### 3.3 Dispatch correlation

The controller stores
`_dispatch_contexts[(handle_id, dispatch_id)] → context`. The drainer
pairs each response with the originating context using `dispatch_id`,
so late results from cancelled dispatches are dropped before reaching
`_handle_result`.

---

## 4. Validator pool

### 4.1 `ValidatorPool` (`substrate/pool.py`)

Owns one **active** `ValidatorHandle` at a time. Routes every RPC
through it. On a connection-class error
(`ConnectionError`, `TimeoutError`, …) the active handle is killed and
the next URL is spawned.

Retry policy:

- **Idempotent ops** (`get_*`, `query_*`, `get_mining_snapshot`,
  `get_events_at`) — auto-retry on the new handle up to
  `max_swap_retries` (default 3).
- **Non-idempotent ops** (notably `submit_extrinsic`) — raise
  `ValidatorSwapped`. Caller has domain knowledge and decides.

`force_swap()` is the watchdog escape hatch — used by the event
manager when the chain looks frozen on the current validator.

### 4.2 `ValidatorHandle` (`substrate/validator_handle.py`)

Parent-side proxy. Spawns one `mp.Process(target=validator_main)`,
maintains `_inflight[request_id] → asyncio.Future`, and runs a
`_drain_responses` task that pulls from `resp_q` and resolves futures.

Picklability is checked proactively (`ForkingPickler.dumps()`) before
queueing requests/responses, because `mp.Queue` serializes in a
background thread where exceptions are lost.

### 4.3 `validator_main` (`substrate/validator_handle.py:68`)

The child process entry point. Owns one `SubstrateClient(url)`,
creates a persistent asyncio loop, and serves RPC ops with a
per-call timeout (`rpc_call_timeout_s`, default 10s).

### 4.4 `PoolClient` (`substrate/pool_client.py`)

A SubstrateClient-shaped shim that routes each method through
`pool.send(op_name, kwargs_dict)`. Read-only — `submit_extrinsic`
raises `NotImplementedError` because key material can't cross IPC.

### 4.5 `SubstrateUrlFailover` (`substrate/url_failover.py`)

Round-robin URL rotation with all-down exponential backoff
(1s → 2s → 4s → … → 60s). `confirm_success()` resets on the next
successful call.

### 4.6 Legacy slot-client shim

`pool.get(role)` (`substrate/pool.py:124`) returns a per-role
`SubstrateClient` that does **not** participate in hot-active swap.
This exists because submitters need a direct client to sign and
submit extrinsics, and `PoolClient.submit_extrinsic` is not
supported. See §9.

---

## 5. Chain event manager (PoW only)

`ChainEventManager` (`substrate/event_manager.py`) replaced the
substrate-interface WS subscription that was the root cause of the
original 90+ minute silent-stall bug.

Two concurrent loops:

- **`_poll_loop`** — every `settled_poll_pct × blocktime_s` (default
  5.1s) in steady state, or `catch_up_poll_pct × blocktime_s` (0.6s)
  when overdue. Calls `pool.send(snapshot_op, snapshot_args)`,
  computes `state_key(snapshot)`, fires a `new_head` event only on
  key change (source-side dedup).
- **`_dispatch_loop`** — pulls from internal `event_q`, invokes
  subscriber callbacks (sync or async), per-callback try/except so a
  bad subscriber can't kill the loop.

Watchdog (in `_poll_loop`):

- `stale_blocktime_multiplier × blocktime_s` (default 6s) → log warning
- `dead_blocktime_multiplier × blocktime_s` (default 18s) → call
  `pool.force_swap()` wrapped in `asyncio.wait_for(...)` so a hung
  pool can't permanently disarm the watchdog.

The mempool controller does **not** use the event manager — it still
runs its own `_subscribe_heads()` loop against
`SubstrateClient.subscribe_new_heads()`. This asymmetry is the largest
remaining piece of pre-Plan-3 architecture and is the highest-impact
cleanup target.

---

## 6. Supervision

`supervise(coro, name, on_failure)` in `shared/asyncio_supervise.py`:

```python
async def supervise(coro, name, on_failure):
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("supervised task %s crashed", name)
        await maybe_await(on_failure())
        raise
```

Every long-lived task wires `on_failure=self._shutdown_event.set` so a
crash is loud and the controller exits, instead of going silent.
**This is the load-bearing guarantee against the original bug class.**

---

## 7. Submission path

PoW submission still happens through a directly-held `SubstrateClient`
obtained via `pool.get("rpc")`, not through `PoolClient` — because
signing requires the parent process to hold the keypair and sign
locally before sending raw bytes over the wire. `PoolClient` cannot
ship a `Signer` across IPC.

The flow for PoW:

```
_handle_result(envelope)
  → encode_quantum_proof(...)
  → submit_proof(signer, client, ...)
       → client.submit_extrinsic(call, signer)     # client = pool.get("rpc")
  → receipt classification
       → ok      → mark work key closed, cancel siblings, notify on_proof_submitted
       → stale   → drop
       → fatal   → raise → controller shutdown
```

Mempool submission is analogous (`submit_solution`, `claim_reward`).

A future refactor would move signing into the parent and ship signed
bytes through `pool.send("submit_extrinsic_bytes", ...)`. That removes
the last consumer of `pool.get(role)`.

---

## 8. Telemetry

Two telemetry surfaces coexist today:

- **In-process** `shared/telemetry_api.py` (`TelemetryApiServer`) —
  default-on, runs an aiohttp app on the controller's event loop.
  This was scheduled for deletion in Plan 4 task 9 and deferred.
- **Sibling process** `shared/telemetry_process.py`
  (`telemetry_main`) — opt-in, spawned only if `telemetry_port=...`
  is passed to the controller. Reads
  `${runtime_dir}/telemetry-stats.json` (written every 1s by the
  controller's `StatsSnapshotWriter`) and serves `/api/v1/stats`,
  `/api/v1/status` (stub), `/health`. Other endpoints
  (`/api/v1/system`, `/api/v1/miner/survey`, `/api/v1/block/*`,
  `/api/v1/mining/attempts`, `/api/v1/solve`) are not yet ported.

Telemetry sibling exists so a slow telemetry handler can't starve the
controller's event loop — the original bug class once more.

---

## 9. Cleanup candidates

These are surfaces that exist in `main` today but are either redundant
or vestigial after Plans 1–4.

### High-confidence removal

| Item | Location | Why dead |
|---|---|---|
| `_consecutive_none_snapshots` counter | `substrate/miner_controller.py:475` | Bumped on every None snapshot but no code path escalates on the configured `_NONE_SNAPSHOT_FAIL_THRESHOLD` |
| `_reentrant_failover` guard | `substrate/client.py:182` | Pre-pool multi-URL safety belt; SubstrateClient is single-URL inside the validator child now |
| `urls` and `current_url` properties on `SubstrateClient` | `substrate/client.py:172, 176` | Only read by the legacy `pool.get(role)` callers; SubstrateClient does not rotate |
| `SubstrateClient.reconnect(target_url=...)` multi-URL walk | `substrate/client.py:215` | `validator_main` never calls multi-URL reconnect; pool handles rotation |
| Old test docstring references to `_handle_head` | `tests/test_miner_controller_on_new_head.py:6–10`, `tests/test_substrate_miner_controller.py` | `_handle_head` was deleted in Plan 3; comments are historical |

### Medium-confidence (architectural, needs discussion)

| Item | Location | Why questionable |
|---|---|---|
| Mempool controller still uses `client.subscribe_new_heads` | `shared/mempool_miner_controller.py` (`_subscribe_heads`, `_on_head`, `_head_signal`, `_latest_head`) | Same WS subscription pattern the PoW path moved away from. Should migrate to `ChainEventManager` with a `new_mempool_event` event type |
| In-process `TelemetryApiServer` | `shared/telemetry_api.py`, wired in `quip_cli.py` | Plan 4 task 9 scheduled deletion once sibling has feature parity. Currently both surfaces exist |
| `pool.get(role)` legacy slot client | `substrate/pool.py:124–142` | Only kept because submitters need direct SubstrateClient to sign. Removable once `submit_extrinsic` migrates to "sign in parent, ship bytes" |

### Tests to retire / rewrite alongside the above

- `tests/test_substrate_client_failover.py` — covers `SubstrateClient`'s
  own URL-walk failover, which becomes unreachable code once we delete
  `reconnect(target_url=...)`. Retires with the production code.
- `tests/test_mempool_miner_controller.py` — heavy mocking of
  `_subscribe_heads`. Needs significant rewrite when mempool migrates
  to `ChainEventManager`.
- `tests/test_telemetry_api.py` — retired when in-process telemetry is
  removed; sibling coverage is in `test_telemetry_process.py`.

### Suspected dead, needs verification

- `_last_pushed_threshold_milli` initialized to `0` causes the first
  head to always trigger a threshold push. Probably intentional, but
  worth confirming. (`substrate/miner_controller.py:443`)
- `IsingFeeder` (`shared/ising_feeder.py`) — `BaseMiner.mine_work_item`
  intentionally does not use the batch/streaming path
  (`base_miner.py:369` comment). Feeder is still wired into
  `_pre_mine_setup` for QPU and GPU but its output may not be reached.
  Needs a closer look before declaring dead.
