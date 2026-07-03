# QUIP Protocol Architecture

This document describes the runtime architecture of `quip-network-node` as
of v0.2 (post Plans 1–4 and the mempool-priority cutover). It covers the
four pillars the system runs on:

1. **The scheduler stack** (one `WorkScheduler` over all handles: PoW as
   idle filler, mempool jobs as priority)
2. **The miner layer** (CPU / CUDA / Metal / QPU)
3. **The validator connection pool** (hot-active swap, per-URL child process)
4. **The chain event manager** (adaptive polling, watchdog; shared by the
   PoW controller and the mempool producer)

Everything below describes code that exists in this tree (v0.2, post
mempool-priority cutover). Items that exist but are vestigial are
flagged in §9 Cleanup candidates.

---

## 1. Process topology

A running miner node is one controller process plus its children:

```
┌─ controller process (asyncio) ─────────────────────────────────┐
│   quip_cli._run_concurrent_miner()                             │
│     ├─ SubstrateMinerController   (the PoW brain)              │
│     ├─ WorkScheduler              (owns ALL miner handles)     │
│     ├─ MempoolStack               (only when [miner] mempool   │
│     │                              resolves on)                │
│     ├─ ValidatorPool                                           │
│     │     └─ one ValidatorHandle  (active URL only)            │
│     └─ ChainEventManager          (shared: PoW + mempool)      │
└────────────────────────────────────────────────────────────────┘
        │ mp.Process per miner backend         │ mp.Process per active URL
        ▼                                       ▼
┌─ MinerHandle child (×N) ─────┐      ┌─ validator_main child ──┐
│   miner_worker_main()        │      │  SubstrateClient(url)   │
│   owns one BaseMiner + one   │      │  asyncio loop           │
│   stream-driver subprocess   │      │  serves req_q → resp_q  │
│   (sampling; see             │      └─────────────────────────┘
│   docs/miner-architecture.md)│
│   QPU adds a D-Wave submitter│
│   child (QPU/dwave_submitter)│
└──────────────────────────────┘
        │ resp queue (one WorkScheduler drainer per handle)
        ▼
                  controller main loop

┌─ telemetry sibling (default-on) ─────────────┐
│  substrate/telemetry_process.telemetry_main   │
│  aiohttp app, full /api/v1 surface            │
│  reads ${runtime_dir}/<kind> stats snapshot   │
└───────────────────────────────────────────────┘
```

Cancellation, errors, and lifecycle:

- Every miner child registers a SIGTERM handler for hardware cleanup.
- Controller-side long-lived tasks (event manager, fire timer, stats
  snapshot writer) are wrapped in
  `supervise(coro, name, on_failure=...)`. An unhandled exception
  triggers controller shutdown — silent task death is the original
  bug class this design eliminates.
- WorkScheduler tasks (per-handle drainers, job pump) fail loud on
  their own: a crashed drainer or a dead worker escalates to scheduler
  shutdown plus `on_fatal` → pow controller shutdown.
- MempoolStack loops (feed / submit / claim) are containment-wrapped:
  a crash **parks** the mempool side (producer, feed loop, job queue)
  and pow mining continues — mempool failure never takes pow down.
- The validator child is killed and respawned on connection-class
  errors (hot-active swap, §4).

---

## 2. The scheduler stack

### 2.1 PoW brain: `SubstrateMinerController` (`substrate/miner_controller.py`)

Entry: `run()` → `_main_loop()`. Requires an attached `WorkScheduler`
(`attach_scheduler`, two-phase init) — the scheduler owns every handle's
drainer and all dispatch operations; the controller keeps the pow brain
(submit_proof, receipt classification, anticipatory fire,
verify-recorded, decay schedules, closed-work-keys).

**Wait set** (`_main_loop`, ~`miner_controller.py:857`):
- `_result_queue.get()` — pow `WorkResult`s, queue-put by the
  scheduler's `on_pow_result` callback (`enqueue_pow_result`)
- `_shutdown_event.wait()` — graceful shutdown

Long-lived supervised tasks:

| Task | Source | Purpose |
|---|---|---|
| `ChainEventManager.run()` | `miner_controller.py:756` | Poll snapshot, fire `new_head` to `on_new_head` + `head_subscribers` |
| Fire-timer loop | `miner_controller.py:764` | Anticipatory decay-target fire authority |
| Stats snapshot writer | `miner_controller.py:674` | Write the per-kind stats snapshot every 1s |
| Telemetry sibling (`mp.Process`) | `miner_controller.py:800` | Default-on; skipped when `QUIP_TELEMETRY_EXTERNAL=1` |

**`on_new_head(ctx)`** (`miner_controller.py:910`) — the event-driven
work dispatcher. Guards in order:

1. `None` snapshot → bump `stats.none_snapshots_seen`, return
2. Chain `DefaultTopology` changed → set `rebind_requested`, shut down
   gracefully (the CLI rebuilds the stack against the new topology)
3. Threshold changed → `handle.set_live_threshold_milli(...)` on each handle
4. Zero seed with `_highest_handled_block > 0` → drop (transient)
5. Work key in `_closed_work_keys` → return (already won)
6. Same key with any handle busy → return (same-key skip)
7. Attach the round-constant decay schedule, resolve the chain-global
   solution number, then `scheduler.dispatch_pow(ctx, ...)` — the
   scheduler's atomic preempt (cancel → mandatory done sentinel →
   dispatch), excluding handles owned by an active mempool job

**`_handle_result(envelope)`** (`miner_controller.py:1272`):

- Duplicate-drop if work key already closed (sibling won)
- Stale-drop if envelope's work key ≠ current
- Encode proof → `submit_proof()` (sign on the parent `build_client`,
  submit through the swap-aware `pool_client`, §7)
- Classify receipt: `ok` (verify recorded on-chain, mark closed,
  `scheduler.cancel_pow_siblings(...)`) / `stale` (drop) / `fatal`
  (shutdown)

### 2.2 `WorkScheduler` (`substrate/work_scheduler.py`)

ONE scheduler owns ALL of the process's miner handles: one drainer task
per handle (a handle's resp queue admits exactly one consumer), the
preempt lock, and all dispatch bookkeeping. Every handle mines PoW
continuously; mempool jobs are the priority source. It has no chain
dependencies — every collaborator is a duck-typed handle or callback.

The core protocol is the atomic preemption in `preempt_and_dispatch`:

```
cancel() → MANDATORY await of the victim's work_item_done sentinel → dispatch
```

`MinerHandle.cancel()` only sets the shared stop_event and the NEXT
`mine_work_item()` clears it, so dispatching before the worker acks
*wipes the cancel* (priority inversion: the old item keeps mining and
the new one queues behind it). There is no timeout-then-dispatch-anyway;
a handle that never acks is a dead worker, which drainer death
detection escalates (unblocking any pending sentinel wait).

Dispatch policy:

- A mempool job fans out to ALL eligible handles — idle ones
  immediately, busy non-QPU ones via the preemption protocol. First
  `mine_result` wins; siblings are cancelled; terminal accounting
  compares done handles against the fanned set only.
- QPU handles are **never preempted by jobs** (idle-only job dispatch):
  the split D-Wave submitter has no ctl_q, so a preemption would strand
  already-paid samples. Pow work-key-change broadcasts do preempt busy
  QPU handles — their in-flight pow work is dead either way.
- A job whose every fanned dispatch terminates result-less (e.g. the
  QPU budget gate aborted it) is requeued once, then dropped.
- Pow is the idle filler: when a handle frees and no job wants it, the
  `provide_pow_context` callback (the pow controller) supplies the
  current context, or `None` to leave the handle idle.

The controller's only handle operations are `dispatch_pow` (work-key
change broadcast), `fill_idle` (verify-fail re-dispatch; never cancels),
and `cancel_pow_siblings` (the submission-storm fix). Result consumers
(`on_pow_result` / `on_job_result`) run inline on the delivering
drainer task and must queue-put only — never RPC.

### 2.3 Mempool: `MempoolStack` (`substrate/mempool_stack.py`)

Thin composition of producer + submitter over the scheduler:

- **`MempoolJobProducer`** (`substrate/mempool_producer.py`) —
  subscribed to the pow controller's ONE `ChainEventManager` via
  `head_subscribers`; polls `System.Events` per block and filters
  `JobProposed` orders through: exact topology-hash eligibility, the
  pallet's Bid OR-semantics (account OR solver_type), a 2-block
  deadline margin, and `[miner] mempool_min_reward` (0 = accept all).
- **Feed loop** — turns accepted order ids into `MempoolJobContext`s
  and calls `scheduler.submit_job(...)` with a dispatch-time
  revalidation callback (order re-fetch + `OPENED` check).
- **`MempoolSubmitter`** (`substrate/mempool_submitter.py`) —
  `submit_solution` carries a small tip (`tip_plancks` = 2e9 =
  0.002 UNIT) so it outranks the same account's tip-0 pow traffic in
  the txpool; retries txpool nonce races with block-spanning backoff;
  caps each watch at 90s. The claim loop claims proactively once a
  submitted order passes its COMPUTED expiry — the pallet expires
  lazily (no `on_initialize` sweep), so an event-only claim loop would
  wait forever on a quiet mempool.
- **Park semantics** — a mempool-fatal receipt (`SolverNotRegistered`
  / `BadSignature` / `BadProof` / anything unrecognized) classifies as
  `SubmitOutcome.MEMPOOL_DISABLE` and parks the whole mempool side
  (producer, feed loop, job queue) while pow mining continues.
  `MempoolStack.run()` returns only on shutdown — under the CLI's
  FIRST_COMPLETED orchestration an early return would tear pow down.

Mempool participation is config-only: `[miner] mempool` (defaults:
cpu/gpu ON, qpu OFF — paid samples are opt-in) and
`[miner] mempool_min_reward`. There is no mempool-only operation mode
and no CLI mode flag. `QUIP_MEMPOOL=0` force-disables mempool in a child; the
supervisor's owner election sets it on every non-owner child of a
multi-backend config (first non-qpu mode in canonical cpu,gpu,qpu
order) because one substrate account can register only ONE solver type
on chain — it is not an operator knob. Guard D+
(`substrate/solver_registration.py:ensure_solver_registered`)
auto-registers the solver at startup: query-first (idempotent),
race-tolerant, and it NEVER auto-deregisters — switching solver type
requires an explicit `quip-miner deregister-solver` and restart. Guard
failure is non-fatal: mempool is disabled for the run and pow proceeds.

### 2.4 CLI orchestration (`quip_cli.py:_run_concurrent_miner`)

Sequence:

1. Announce + load keystore (Guard A), connect a direct setup client,
   run guards C/D/D+/E (funded / registered / solver-registered /
   descriptor). Guard D+ can only flip mempool off for the run.
2. Bind-and-run loop: fetch the chain's `DefaultTopology`,
   `_prepare_core` builds one `MinerCore` (no handle split — the
   scheduler serves both work sources, so 1-handle nodes are fully
   supported), `_build_scheduler_stack` builds the pow controller, ONE
   `WorkScheduler` over ALL handles, and the optional `MempoolStack`
   (when `[miner] mempool` resolves on), then attaches the scheduler to
   both.
3. `_orchestrate_controllers`: `scheduler.start()` first (drainers must
   be live before the first dispatch), SIGINT/SIGTERM → `shutdown()` on
   all three, `asyncio.wait(..., FIRST_COMPLETED)` over the pow
   controller (+ mempool stack), a 15s grace drain, then
   `scheduler.stop()`.
4. On `rebind_requested` (chain topology changed) the loop rebuilds the
   whole stack without a process restart; otherwise the pow
   controller's exit code becomes the process's.

---

## 3. Miner layer

### 3.1 Class hierarchy

```
BaseMiner                          shared/base_miner.py
├── SimulatedAnnealingMiner        CPU/sa_miner.py
├── GPUMiner                       GPU/gpu_miner.py
│   └── CudaMiner                  GPU/cuda_miner.py
├── MetalMiner                     GPU/metal_miner.py
├── ModalMiner                     GPU/modal_miner.py
└── DWaveMiner                     QPU/dwave_miner.py
```

There is no inline sampling path: every backend exposes a
`build_persistent_context(...)` factory that runs in a **stream-driver
subprocess** (feeder + sampler), writing samplesets into a
shared-memory ring the worker consumes. The full producer→ring→consumer
design is in `docs/miner-architecture.md`. The base class supplies a
default `_adapt_mining_params(...)` (forwarding to `adapt_parameters`
with subclass-declared calibration bounds; only CUDA and D-Wave
override it) plus optional hooks (`_pre_mine_setup`,
`_post_mine_cleanup`, ...); the QPU implementation uses
`_pre_mine_setup` to gate the daily budget, and its `USES_SUBMITTER_SPLIT`
isolates the D-Wave SDK into a separate submitter process.

The protocol-neutral entry is `BaseMiner.mine_work_item(context,
stop_event)` (`base_miner.py:699`). It accepts either work-source
flavor (`SubstrateMiningContext` or `MempoolJobContext`) and loops until
the stop event fires, sourcing one `(nonce, salt, sampleset)` per
iteration off the stream-driver's descriptor queue and evaluating it.

PoW uses **ratchet mode** (`_run_substrate_ratchet`: bounded top-K
stash, decay-aware submit gate; reads `live_threshold_milli` from a
shared `mp.Value` each iteration). Mempool uses **strict mode**
(`_run_mempool_eval`: energy gate always applied).

### 3.2 `MinerHandle` (`shared/miner_worker.py:368`)

The parent's view of one miner child. IPC primitives:

- `req: mp.Queue` — parent → worker (RPC-style ops)
- `resp: mp.Queue` — worker → parent (results + sentinels; drained by
  the WorkScheduler, one drainer per handle)
- `stop_event: mp.Event` — worker polls each iteration
- `live_max_energy_milli: mp.Value('q')` — shared i64 ratchet threshold

Operations:

| Method | Effect |
|---|---|
| `mine_work_item(ctx, solution_number=...) → dispatch_id` | Clear stop_event, enqueue `{op: "mine_work_item", context, dispatch_id, solution_number}`, return id |
| `cancel()` | Set stop_event directly (worker observes within one iteration; the NEXT `mine_work_item()` clears it — hence the scheduler's mandatory sentinel wait) |
| `set_live_threshold_milli(milli)` | Atomic write to shared `mp.Value` |
| `get_stats()` | Request/reply via queues |

The worker (`miner_worker_main`) is a simple op dispatcher
(`miner_worker.py:163`). Mining results land on `resp_q` as either
`{"op": "mine_result", "dispatch_id", "result"}` or
`{"op": "work_item_done", "dispatch_id"}` (sentinel for cancelled or
result-less dispatches).

### 3.3 Dispatch correlation

The **scheduler** stores
`_dispatch_contexts[(handle_id, dispatch_id)] → context`
(retention-pruned per handle). Its drainer pairs each response with the
originating context using `dispatch_id`, so late results from cancelled
dispatches are dropped, and `dispatch_context()` lets the pow
controller pair non-terminal worker messages (previews) with the exact
context they were produced against.

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
- **Non-idempotent ops** (notably `submit_signed_extrinsic`) — raise
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

### 4.3 `validator_main` (`substrate/validator_handle.py:99`)

The child process entry point. Owns one `SubstrateClient(url)`,
creates a persistent asyncio loop, and serves RPC ops with a
per-call timeout (`rpc_call_timeout_s`, default 10s).

### 4.4 `PoolClient` (`substrate/pool_client.py`)

A SubstrateClient-shaped shim that routes each method through
`pool.send(op_name, kwargs_dict)`: read ops (idempotent, auto-retried
across swaps) plus `submit_signed_extrinsic` — the one non-idempotent
write, which surfaces `ValidatorSwapped` to the caller. Signing stays
in the calling process (`SubstrateClient.build_signed_extrinsic`); key
material never crosses IPC (§7).

### 4.5 `SubstrateUrlFailover` (`substrate/url_failover.py`)

Round-robin URL rotation with all-down exponential backoff
(1s → 2s → 4s → … → 60s). `confirm_success()` resets on the next
successful call.

---

## 5. Chain event manager

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

There is ONE event manager per process, owned by the pow controller.
Its `state_key` includes `block_hash`, so it fires on every block —
which the mempool producer needs (each block may carry mempool events)
and the pow path absorbs cheaply via its same-key short-circuits. The
producer's per-block `System.Events` poll rides the same manager as a
`head_subscribers` entry; the pre-T7 mempool controller's separate
`subscribe_new_heads()` WS loop is gone.

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

Signing happens in the controller process: a parent-side
`SubstrateClient` (the `build_client`, connected in
`SubstrateMinerController.run()`) composes and signs each extrinsic via
`build_signed_extrinsic`, and the signed bytes go out through the
swap-aware `PoolClient.submit_signed_extrinsic`. Key material never
crosses the mp.Queue IPC boundary; a mid-flight validator swap raises
`ValidatorSwapped` and the retry wrapper (`submit_with_retry`)
re-composes from scratch, which also reads a fresh nonce.

The flow for PoW:

```
_handle_result(envelope)
  → encode_quantum_proof(...)
  → submit_proof(...)
       → build_client.build_signed_extrinsic("QuantumPow", "submit_proof", ...)
       → pool_client.submit_signed_extrinsic(extrinsic_hex, ...)
  → receipt classification
       → ok      → mark work key closed, cancel pow siblings, verify recorded
       → stale   → drop
       → fatal   → raise → controller shutdown
```

Mempool submission (`MempoolSubmitter.submit_solution` /
`claim_reward`) uses the same compose-in-parent + submit-via-pool
shape, adding a txpool tip and block-spanning nonce-race retries
(§2.3). Its fatal receipts park mempool instead of shutting the
controller down.

---

## 8. Telemetry

One telemetry surface: the sibling process
`substrate/telemetry_process.py` (`telemetry_main`), spawned by the
controller by default. The in-process `TelemetryApiServer` is deleted.
Under the config-driven supervisor the controller skips the sibling
spawn (`QUIP_TELEMETRY_EXTERNAL=1`) and a single `quip-miner telemetry`
aggregator child owns the port instead, reading every backend's
snapshot.

The sibling reads the per-kind stats snapshot the controller's
`StatsSnapshotWriter` writes to `${runtime_dir}` every 1s and serves
the full `/api/v1/*` surface (`status`, `system`, `miner/survey`,
`stats`, `block/latest`, `block/{n}`, `block/{n}/header`, `solve`,
`mining/attempts`, `mining/solutions`, `/health`). There is no live IPC
between miner and telemetry — the snapshot file is the channel, so a
slow telemetry handler can't starve the controller's event loop (the
original bug class once more).

---

## 9. Cleanup candidates

The Plans 1–4 cleanup list is done: the pre-T7 mempool controller and
its `subscribe_new_heads()` WS loop were replaced by the producer riding
the shared `ChainEventManager`, the in-process `TelemetryApiServer` and
the `pool.get(role)` slot client are deleted (submission now signs in
the parent and ships bytes through the pool, §7), and
`SubstrateClient`'s own multi-URL failover machinery is gone.

### Suspected dead / needs verification

- `_last_pushed_threshold_milli` initialized to `0` causes the first
  head to always trigger a threshold push. Probably intentional, but
  worth confirming. (`substrate/miner_controller.py:513`)

### Test suites covering this document

- `tests/test_work_scheduler.py` — preemption protocol, job fan-out,
  first-result-wins, requeue-once, pow idle filler.
- `tests/test_mempool_producer.py` / `tests/test_mempool_submitter.py`
  — discovery guards; receipt classification, tip, claim loop.
- `tests/test_concurrent_mode.py` — scheduler-stack wiring (one
  scheduler over all handles, shared event manager, park semantics).
- `tests/test_mempool_priority_integration.py` — live dev-chain
  integration; the dev-chain start procedure and the manual QPU /
  multi-backend smoke procedures are in its module docstring, and the
  failure-isolation park test is opt-in via `QUIP_T9_PARK=1`.
- `tests/test_miner_controller_on_new_head.py` /
  `tests/test_substrate_miner_controller.py` — the pow brain.
