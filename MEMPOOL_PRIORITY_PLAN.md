# Mempool-Priority Scheduling Plan

Goal (operator-stated): pow and mempool are not separate worker pools or
modes. Every miner runs pow continuously; when a mempool job appears that
this node can fulfill, it takes priority — the job preempts (or is queued
ahead of) pow work on the same workers, runs, and pow resumes. Mempool
participation defaults ON and is disabled with a single config key.

## Config surface (replaces `[miner] mode`)

```toml
[miner]
mempool = true    # default; false = pow only
```

- `mode = "pow"|"mempool"|"both"` and the worker-splitting "both"
  implementation are removed once this lands (replace, don't deprecate).
- `quip-miner --config` (the production supervisor) passes the flag to
  each worker child; the per-mode subcommands stay as test tooling.

## Current architecture (what has to change)

- `_run_concurrent_miner` builds a pow controller and a mempool
  controller and **partitions `core.miner_handles`** between them
  (`_build_controllers`, quip_cli.py). Handles are exclusive to one
  controller for the process lifetime.
- Worker handles expose `mine_work_item(context)` + `cancel()` — cancel
  is already exercised on new-head/rebind, so mid-work interruption is a
  supported operation.
- Mempool solving requires the account registered as a solver
  (`register_solver`, idempotent) and one solver type per account —
  which is why multi-backend + mempool is currently rejected.

## Design sketch

1. **Single work scheduler per process** owning all handles. Work items
   carry a priority: `MEMPOOL_JOB > POW`. Pow is the idle filler: a
   handle with nothing better to do mines the current head.
2. **Mempool listener** (existing JobProposed subscription) becomes a
   producer only: it evaluates each job (topology/type match, deadline,
   profitability guard) and enqueues fulfillable ones.
3. **Preemption:** when a mempool job is enqueued and no handle is free,
   the scheduler `cancel()`s one pow work item (choose the youngest /
   least-progressed) and dispatches the job to that handle. Pow re-enters
   automatically because it is the idle filler.
4. **Auto solver registration:** startup Guard D+ — when `mempool = true`,
   `register_solver(<kind of this process's backend>)` idempotently,
   after funding (mirrors `_ensure_registered_or_fail`). The one-solver-
   type-per-account constraint maps cleanly because each backend group is
   its own process; multi-backend nodes need either per-backend accounts
   or mempool restricted to the first group — decide at implementation.
5. **Failure isolation:** a mempool job that errors must not kill the pow
   loop — log, mark the job failed, resume filler.

## Open questions for implementation

- Preempt in-flight pow sweeps vs. wait-for-chunk-boundary: `cancel()`
  granularity differs per backend (CPU SA vs Metal chunked sweeps vs QPU
  in-flight submits). Start with cancel(); refine per backend.
- Multi-backend + mempool account constraint (see 4).
- Whether the mempool evaluation step needs its own budget guard for QPU
  backends (daily_budget interplay).

## Verification

- Unit: scheduler priority/preemption with fake handles (enqueue job →
  pow item cancelled → job dispatched → pow resumes).
- Integration: minertest with a local chain — post a mempool job while
  pow-mining, assert the job solution lands and pow submissions continue
  after.
