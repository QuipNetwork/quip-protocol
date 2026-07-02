# Mempool-Priority Scheduling Plan (corrected)

Goal (operator-stated): pow and mempool are not separate worker pools or
modes. Every miner runs pow continuously; when a mempool job appears that
this node can fulfill, it takes priority — the job preempts pow work on the
same workers, runs, and pow resumes. Mempool participation defaults ON
(with backend-specific exceptions, below) and is disabled with a single
config key.

This revision replaces the original plan after subsystem verification and
adversarial review. Every mechanism description below has been verified
against the code with file:line anchors; the original plan's factual errors
are corrected inline and called out in **Corrections** notes.

## Config surface (replaces `[miner] mode`)

```toml
[miner]
mempool = true              # cpu/gpu default; qpu default is false (opt-in)
mempool_min_reward = 0      # optional; drop orders paying less (default 0 = accept all)
```

- `mode = "pow"|"mempool"|"both"`, `_split_handles_for_mode`, the exit-5
  both-guard, `--mode`, `--mine-mode`, `_MODE_HELP`,
  `_MINING_DEFAULTS['mode']`, and the `multi-backend-not-allowed-in-
  mempool-mode` guard are all removed (replace, don't deprecate). This
  **deletes mempool-only operation** — pow-as-idle-filler always runs.
  Chain-side that is cost-neutral: Guard D already pow-registers every
  mode with the MinerDeposit reserve (quip_cli.py:1738-1742; runtime
  configs/mod.rs:283).
- **Effective-default resolution is per backend group**: when the merged
  config's `mempool` key is `None`, cpu/gpu processes default `True`,
  qpu processes default `False`. The `is None` default machinery already
  preserves explicit falsy TOML values (quip_cli.py:594-600 — its comment
  names `auto_mine = false` as the protected case), so `mempool = false`
  and an explicit `mempool = true` on a QPU node both work.
- The `[miner]` table currently has **zero per-key validation**, so
  `mempool = "false"` (a TOML string) would be truthy and silently enable
  mempool. A bool type-check is added following the
  `_validate_validators_field` pattern (shared/miner_config.py:552-566;
  bool-rejection idiom at 170-201), and the key is added to the schema
  docstring (miner_config.py:17-41).
- **Correction:** the supervisor does NOT need to \"pass the flag to each
  worker child\" — `_plan_processes` passes no work-source flag or env
  today (child argv is `[<backend>, --config, <path>]`,
  quip_cli.py:803-810); children re-derive everything from the shared
  config file. The boolean transits for free. The only NEW plumbing is
  the multi-backend owner-election env var (`QUIP_MEMPOOL=0` for
  non-owner children, see \"Multi-backend policy\").

## Current architecture (verified, corrected)

- `_run_concurrent_miner` builds a pow controller and a mempool
  controller. **Correction:** the handle partition happens in
  `_prepare_core` via `_split_handles_for_mode` (quip_cli.py:1889-1891,
  2215-2238), not in `_build_controllers` — the latter only consumes the
  pre-split `pow_handles`/`mempool_handles` lists (quip_cli.py:1903-1912).
  mode=both gives pow `floor(n/2)` handles; n=1 yields ([], [h]) and the
  exit-5 guard fires (quip_cli.py:1892-1899). **Correction:** exclusivity
  is per bind cycle, not process lifetime — a topology rebind rebuilds
  core + split + controllers in-process (quip_cli.py:2145-2201).
- Both controllers share ONE asyncio loop (`asyncio.run` at
  quip_cli.py:2345; `_orchestrate_controllers` at 1759-1852). A fatal
  error in either controller tears down the other via FIRST_COMPLETED
  (quip_cli.py:1804-1845); the production supervisor treats the first
  child exit as fatal and terminates all siblings (quip_cli.py:843-859).
- Worker handles: one persistent spawned worker process + req/resp
  mp.Queues + shared stop_event, plus 1-2 persistent driver processes and
  a shm ring (shared/miner_worker.py:368-406; base_miner.py:1267-1384).
  `mine_work_item(context)` accepts either `SubstrateMiningContext` or
  `MempoolJobContext` through the same op (miner_worker.py:434-436;
  base_miner.py:840-852) — the worker layer is already job-type agnostic.
- **Correction (cancel semantics):** `cancel()` only sets the shared
  per-handle stop_event (miner_worker.py:464-477). Three consequences:
  1. The **next `mine_work_item()` clears it** (miner_worker.py:453) —
     cancel → clear → dispatch can wipe a cancel before the worker
     observes it (the controller's own warning, miner_controller.py:
     1044-1047), leaving the old item running and the new item queued
     behind it on the worker's **serial FIFO req_q**
     (miner_worker.py:204-208).
  2. `cancel()` stops only the **consumer**. The driver's stop is set
     only at shutdown (`_close_driver`, base_miner.py:1386-1402); backend
     compute continues on the stale generation until the NEXT dispatch's
     `('switch', ...)` reaches the driver ctl_q
     (stream_context.py:100-140). A cancel without an immediate follow-on
     dispatch just burns compute into the generation filters.
  3. On the QPU split path the submitter has **no ctl_q at all**
     (QPU/dwave_submitter.py:186-249): up to queue_depth=30 already-paid
     D-Wave jobs run to completion per preemption
     (dwave_sampler.py:983-1061), and `_pre_mine_setup`'s reservoir gate
     can abort an incoming dispatch entirely (base_miner.py:1046-1069).
- **Correction (cancel latency):** worker-side cancel-ack is uniform
  across backends and fast in the streaming stack — ~0.1s stop_event poll
  on `desc_q.get(timeout=0.1)` plus one eval pass (base_miner.py:
  1500-1533, 815-823, 884-889). The 30-45s \"uninterruptible
  sample_ising\" claim in miner_controller.py:1509-1519 and the \"on every
  new best head: cancel\" module docstring (miner_controller.py:26-27)
  are **stale** (pre-MR-!111); today cancel fires only on work-key change
  (miner_controller.py:1033-1061). What differs per backend is
  producer-side **switch** granularity (batch boundaries on
  CPU/CUDA/Metal; never for QPU in-flight), not cancel granularity.
- **Correction (job discovery):** there is no JobProposed subscription.
  ChainEventManager POLLS `get_mining_snapshot` per block and the mempool
  controller then polls `System.Events` via `get_events_at(block_hash)`
  (mempool_miner_controller.py:264-325, 434-451; event_manager.py:41-77).
  Any producer refactor inherits a poll loop.
- **Correction (job filtering):** `_should_accept_job` checks ONLY
  topology hash and JobMode (mempool_miner_controller.py:497-531).
  `order.timing.deadline_blocks` is never read; `order.reward` is only
  logged; min-reward is explicitly deferred (lines 193-194). Deadline and
  profitability guards are NEW work, not existing behavior to relocate.
- Mempool dispatch today fans ONE order out to ALL of the mempool
  controller's handles, first result wins, siblings cancelled, gate
  released when every handle is terminal (mempool_miner_controller.py:
  574-636, 651-657).
- **Correction (registration):** `register_solver` is NOT idempotent
  on-chain — a repeat call fails with `SolverAlreadyRegistered`
  (quantum-compute-mempool/src/lib.rs:429-432); the client explicitly
  makes idempotency \"the caller's concern via query_solver\"
  (substrate/client.py:1126-1127). The idempotent pattern lives only in
  the `register-solver` CLI command (quip_cli.py:1366-1395, exit 4 on
  type mismatch). No deposit is taken (fee-scale funding only,
  lib.rs:434-443); an idle registration has zero ongoing on-chain cost
  (Hooks = on_runtime_upgrade only, lib.rs:883-924). The
  MempoolMinerController never registers — it verify-or-raises
  (mempool_miner_controller.py:408-421).
- **Correction (solver types):** \"qpu\" is not one MinerType. The
  registered type is vendor-resolved (QpuDwave vs QpuIbm/QpuIonq/
  QpuPasqal via `_qpu_miner_kind_from_backends` + `qpu_miner_kind`,
  quip_cli.py:1666-1682, 2562, 2568; mempool_types.py:98-108). The
  one-solver-type constraint binds the **account** (`Solvers` is keyed by
  AccountId, lib.rs:295), and all supervisor children share one account
  (single `[miner] signer_key`, miner_config.py:21, 518-522) — process
  separation partitions nothing.
- Mempool-fatal receipt errors (SolverNotRegistered/BadSignature/
  BadProof) currently raise RuntimeError out of the controller
  (mempool_miner_controller.py:89-93, 724-730), which via FIRST_COMPLETED
  kills the pow controller too.

## Design

### 1. Single WorkScheduler per process (new: `substrate/work_scheduler.py`)

Owns ALL of `core.miner_handles`. Structural rationale: each handle's
resp mp.Queue admits exactly one drainer, so handles cannot be shared
between two controllers — a single owner is the only structure that ends
the partition, and it turns the 1-handle mode=both case from
impossible (exit 5) into supported.

The scheduler absorbs and unifies:
- **Drainers** — one per handle, ported from `_drain_handle_loop`
  (miner_controller.py:2517-2637) + the mempool drainer
  (mempool_miner_controller.py:801-887). Result envelopes are routed by
  the **context type** stored in the `(handle_id, dispatch_id)` immutable
  context map (pow submit path vs mempool submit path). Per-handle
  done-sentinel queues (`_done_queues`, miner_controller.py:556-564)
  cover ALL handles.
- **One ChainEventManager** — the two managers poll identical state keys
  today (miner_controller.py:715-736 vs mempool_miner_controller.py:
  289-301); one manager fans out to the pow `on_new_head` callback and
  the mempool producer's per-block `System.Events` poll. This halves
  poller load on the shared asyncio loop (helps the known
  event-loop-starvation history).
- **Dispatch invariant:** the scheduler NEVER calls `mine_work_item()`
  on a handle whose `_active_dispatch_id != 0`.

The pow controller keeps its brain (submit_proof, classify, anticipatory
fire timer, verify-recorded, decay schedules, closed-work-keys) but
delegates all handle operations to the scheduler: `on_new_head`'s
dispatch/cancel paths call scheduler ops instead of touching handles;
its drainer startup is removed. The pow **idle filler**: when a handle
frees and no job is queued, the scheduler asks the pow source for the
current context; the pow source returns None when the work key is closed
or anticipatory-fired (`_should_drop_result` guards,
miner_controller.py:1174-1243), leaving the handle idle until the next
head — replacing today's up-to-one-block idle gap after job completion
with immediate backfill.

### 2. Preemption protocol (atomic; fixes the priority inversion)

`preempt_and_dispatch(victims, job_ctx)` is ONE scheduler operation:

1. `handle.cancel()` on every victim (parallel, mirrors the gather shape
   at miner_controller.py:1049-1061).
2. **Mandatory await** of each victim's `work_item_done` sentinel — no
   timeout-then-dispatch-anyway. Do NOT copy `_await_handle_done`'s 0.5s
   best-effort proceed (miner_controller.py:1493-1554): on the priority
   path, dispatching before the sentinel lets `mine_work_item()`'s
   `stop_event.clear()` (miner_worker.py:453) un-cancel the pow item and
   wedge the job behind it on the serial req_q — priority inversion.
   The wait is affordable: ack is ~0.1s + one eval pass in the streaming
   stack. Log a progress WARNING every 5s while waiting; a handle that
   never acks means a dead worker, which the drainer's death detection
   already escalates.
3. Dispatch `job_ctx` immediately after the sentinel. The dispatch IS
   the driver `switch` — this is what actually stops backend compute, so
   cancel+await+dispatch must never be split across scheduler decisions.

### 3. Dispatch policy (fan-out preserved; victim selection simplified)

- **Width:** a fulfillable job is dispatched to ALL eligible handles —
  idle ones immediately, busy non-QPU ones via the preemption protocol.
  This preserves today's fan-out-all/first-result-wins/cancel-siblings
  semantics (mempool_miner_controller.py:599-601, 651-657) and its solve
  redundancy (relevant to the known ~0.10-0.19 diversity-yield ceiling).
  Width is an internal scheduler policy parameter, not a config key.
- **Victim selection:** \"youngest/least-progressed\" is dropped — no
  per-handle progress signal exists (only `_active_dispatch_id`
  busy/idle) and all pow handles mine the same work key. The victim set
  is simply every busy non-QPU handle.
- **First result wins:** scheduler cancels siblings, hands the result to
  the mempool submitter; terminal accounting compares done-handles
  against the fanned set (not `len(miner_handles)` — reworks
  mempool_miner_controller.py:615-636's gate).
- **Dequeue-time re-validation** moves into the scheduler: re-fetch the
  order and re-check `OrderStatus.OPENED` immediately before dispatch
  (ported from mempool_miner_controller.py:581-594).

### 4. QPU policy (resolves the deferred budget question NOW)

- QPU handles are **never preempted** (idle-only dispatch): the split
  submitter has no ctl_q, so each preemption strands up to
  queue_depth=30 paid D-Wave jobs and a num_reads flip forces a D-Wave
  reconnect + handshake.
- QPU processes default `mempool = false`; explicit `mempool = true`
  opts in (per-backend default resolution, see Config surface).
- **Budget-abort tolerance:** `_pre_mine_setup`'s reservoir gate can
  return None → the worker emits `work_item_done` with no `mine_result`
  (base_miner.py:1046-1069). The scheduler must treat a job whose every
  fanned dispatch terminated result-less as not-executed: requeue once,
  then drop with a log.

### 5. Driver-respawn mitigation for pow↔mempool flips

Two costs the original plan omitted:
- **Feeder-kind flip** rebuilds the driver-side feeder each direction and
  mempool→pow respawns RandomIsingFeeder's process pool
  (stream_context.py:124-140; base_miner.py:1314-1317). Accepted cost;
  documented.
- **Ring dims:** `_ensure_driver` reuses the driver only on exact
  `dims == (num_reads, len(nodes))` (base_miner.py:1282-1292), while
  num_reads is re-adapted per dispatch (base_miner.py:1078-1082) — a
  mempool job with different adapted num_reads forces a FULL driver
  respawn per flip (Metal kernel recompile, CUDA re-init, D-Wave
  reconnect, new shm ring), twice per job on a 1-handle node.
  **Fix:** relax the reuse gate to `len(nodes)` match AND
  `num_reads <= ring max_rows` (SampleView already stores `max_rows`
  capacity, base_miner.py:1294-1296), clamping the dispatch's num_reads
  to the ring capacity when larger; log every forced respawn with its
  reason so the residual cost is measurable. Preemption hysteresis
  (minimum pow run time) is deferred until job arrival rates demand it.

### 6. Mempool producer (extraction + NEW guards)

The evaluation pipeline is verified handle-free (`on_new_block` →
`_process_head` → `_handle_mempool_event` → `_consider_order` →
`_should_accept_job` → deque; mempool_miner_controller.py:327-531) and is
extracted into `substrate/mempool_producer.py`, registered as a callback
on the single ChainEventManager. It keeps the polling discovery model
(there is no subscription to reuse) and gains:
- **Deadline check (mandatory, new):** read `order.timing.deadline_blocks`
  (mempool_types.py:359-366) against the current snapshot block; drop
  orders already expired or with fewer than a small margin (2 blocks)
  remaining — mandatory once preemption latency (sentinel wait + possible
  driver respawn) can exceed short deadlines.
- **Min-reward guard (minimal, new):** `order.reward >=
  merged['mempool_min_reward']` (default 0 = accept-all). No cost model
  is attempted; this just replaces the deferred Phase-9 TODO
  (mempool_miner_controller.py:193-194) with the operator knob.
- Existing checks retained: topology-hash equality, JobMode OR-semantics
  eligibility, `_pending_seen` dedup, OPENED status on discovery.

### 7. Submission, claims, and failure containment

`substrate/mempool_submitter.py` absorbs `_handle_result`'s build/submit
(solutions_to_scale, 20-solution clip, parent-process build_client,
swap-aware pool submit) and the claim loop. **Behavior change:** receipt
classification returns an outcome enum; the currently-fatal class
(SolverNotRegistered/BadSignature/BadProof, mempool_miner_controller.py:
89-93, 724-730) maps to `MEMPOOL_DISABLE` — the scheduler parks the
producer, logs loudly, and pow continues. Raising RuntimeError out of the
shared stack is forbidden: under one process it would kill pow (and via
the supervisor's first-child-exit rule, the whole node). An operator
deregistering mid-run must not stop pow mining.

### 8. Guard D+ — solver auto-registration (query-first, non-fatal)

New `ensure_solver_registered(client, signer, miner_kind)` in
`substrate/solver_registration.py`, run at startup when mempool is
effectively enabled, after funding (Guard C), alongside Guard D:
- **Query-first:** `query_solver`; already-registered with matching type
  → success, no extrinsic (a blind per-boot `register_solver` would fail
  with SolverAlreadyRegistered and still burn a fee). Copies the CLI
  pattern (quip_cli.py:1371-1387).
- **Race-tolerant:** if the extrinsic fails with SolverAlreadyRegistered
  (two sibling children on one account racing), re-query; matching type
  → success.
- **Vendor-resolved type:** register `MinerType.from_kind(miner_kind)`
  with the vendor-resolved kind, exactly as `_build_controllers` does at
  quip_cli.py:1997 — never the backend-group name. A QPU vendor switch on
  the same account requires an explicit `deregister-solver` (documented;
  today's hard error stands).
- **Never auto-deregister on mismatch:** it resets on-chain solver stats
  (lib.rs:436-460) and two children with different types would ping-pong
  registrations every boot. Mismatch → mempool-off for this process.
- **Non-fatal, unlike the Guard D it structurally mirrors**
  (`_ensure_registered_or_fail` raises, quip_cli.py:701-734): ANY guard
  failure (mismatch, unfunded fee, mempool pallet absent on the target
  chain) degrades to mempool-disabled-for-this-run with a loud log —
  a fatal child exit would trigger supervisor terminate-all-siblings
  (quip_cli.py:843-859) and take pow down node-wide.
- Funding needed is fee-scale only (no deposit, lib.rs:434-443); an idle
  registration persists at zero ongoing cost.

### 9. Multi-backend policy (launch blocker, decided now)

The `multi-backend-not-allowed-in-mempool-mode` guard
(miner_config.py:294, 358-373) is deleted with the mode key, but its
rationale (one solver type per shared account) survives. Replacement:
**supervisor owner election.** `_plan_processes` designates the FIRST
non-QPU backend group (in resolved group order) as the sole mempool
owner and sets `QUIP_MEMPOOL=0` in every other child's env (the env dict
at quip_cli.py:803-810 already carries telemetry vars). Child-side
effective-enable = config bool AND env != \"0\". Single-backend nodes
(the common case) are unaffected. Per-backend accounts are rejected for
now as a much larger schema/funding/descriptor change. This ships in T8
and MUST land before any release tag that flips the default — with
default-ON it is the default path for every multi-backend node, not a
deferrable edge. (Guard D+'s non-fatal race handling makes the T7→T8
window safe: the registration race loser degrades to mempool-off.)

### 10. Sequencing constraint

The boolean default flips ONLY in the same change (T7) that deletes
`_split_handles_for_mode` and the two-controller layout. If
`mempool = true` were ever interpreted by the existing split, every
1-handle node (typical single-QPU) would exit code 5 (n//2 = 0 pow
handles) and every node would double its ChainEventManager pollers on
the one asyncio loop.

## Deletions (replace, don't deprecate)

- `_split_handles_for_mode` + exit-5 guard (quip_cli.py:2215-2238,
  1892-1899); `--mode`, `_MODE_HELP`, `_MINING_DEFAULTS['mode']`
  (quip_cli.py:2241-2246, 2261, 2286-2291); mode validation/banner
  plumbing in `_run_concurrent_miner` (2080-2083, 2040); the
  two-controller `_build_controllers`/`_orchestrate_controllers`
  signatures (1903-1915, 1759-1761); node_id becomes `quip-miner`.
- `MempoolMinerController` (substrate/mempool_miner_controller.py) —
  its pipeline splits into producer/submitter/scheduler;
  `_verify_solver_registered` is replaced by Guard D+.
- `--mine-mode` / `_effective_mine_mode` (quip_cli.py:1031-1043) and the
  `mine_mode` parameter + `_MEMPOOL_MINE_MODES` guard in
  `resolve_modes`/`resolve_mode` (shared/miner_config.py:294, 358-373);
  `_plan_processes`' `miner_toml.get(\"mode\")` read (quip_cli.py:781-787).
- Stale docstrings: miner_controller.py:26-27 (\"on every new best head:
  cancel\") and 1509-1519 (30-45s sample_ising) — fixed early (T3) so
  implementers don't design around dead constraints.

## Test migration (rewrite, not patch)

- tests/test_concurrent_mode.py — entirely about removed machinery
  (split unit tests 35-119, --mode forwarding 128-191): replaced by
  scheduler wiring tests.
- tests/test_miner_config.py:565-634 — the mine_mode guard block:
  deleted with the guard; new tests for the `mempool` bool validator and
  `mempool_min_reward`.
- tests/test_quip_cli.py — mode plumbing (474-536), --mine-mode guard
  tests (1649-1725), `mode==\"pow\"` kwarg assertions (113, 133,
  1843-1854): rewritten against the boolean + scheduler;
  `_plan_processes` tests (1352-1389) survive and gain owner-election
  env assertions.
- New scheduler unit suite (fake handles): enqueue job → pow cancelled →
  sentinel awaited → job dispatched → pow resumes; **busy-handle
  un-cancel regression** (fake handle models stop_event-clear-on-
  dispatch; assert the scheduler never dispatches before the sentinel);
  QPU exclusion; budget-abort requeue; first-wins sibling cancel.

## Verification

- Unit: the scheduler suite above; Guard D+ branch matrix (unregistered,
  registered-matching, mismatch, extrinsic race, RPC failure — all
  non-fatal); producer deadline/min-reward/dedup; ring-reuse gate.
- Integration (minertest, local chain): post a mempool job while
  pow-mining on a 1-handle node (previously exit-5) — assert the job
  solution lands and pow submissions continue after; multi-backend
  config boots with exactly one mempool owner; operator deregisters
  mid-run → mempool parks, pow continues.

## Staged implementation order (suite green at every step)

T1 config keys → T2 Guard D+ helper → T3 stale-docstring fixes →
T4 ring-reuse relaxation → T5 producer/submitter extraction
(behavior-preserving) → T6 WorkScheduler module + fake-handle tests →
T7 atomic switchover (rewire quip_cli, delete mode machinery, flip
default, rewrite tests) → T8 supervisor owner election + guard removal →
T9 live integration → T10 docs. T1-T6 are additive or
behavior-preserving; T7 is the single atomic cutover; T8 must land
before any release tag.