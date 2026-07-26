# Coordinator — how the QuIP v0.3 coordinator works

`quip-coordinator` is the one process that touches the chain. It follows the
chain head and stages work for the miners. Each miner runs as a subprocess and
speaks the gRPC miner protocol to the coordinator over a local socket; the
coordinator collects the solutions and submits the winning proofs. The miners
never see the chain.

The crate is `crates/quip-coordinator`. This document describes the modules and
how they fit together. For build and run commands, see `AGENTS.md`.

## Two run modes

`main.rs` parses one CLI:

- `quip-coordinator --config <config.toml>` — the production runtime
  (`run_config_path` → `runtime::run_runtime`). Connects to the validators,
  spawns and supervises the configured miners, feeds them chain work, submits
  proofs.
- `quip-coordinator drive …` — a synthetic driver (`drive` module). Spawns one
  miner and feeds it generated or replayed problems with no chain and no submit.
  Used for benchmarking and matched-condition parity runs.

## Runtime wiring

`run_runtime` (in `runtime.rs`) is the production entry point. It:

1. Binds a Unix-domain socket and serves the gRPC `MinerService` on it
   (`CoordinatorService`). Miners connect back over this socket.
2. Seeds each miner's `Configure` from the launch plan so the server can answer
   the handshake.
3. Spawns one `supervise_miner` task per configured miner.
4. Spawns the `feeder_loop`, which follows the chain head and keeps every
   miner's staged queue full.
5. Waits for `shutdown` (SIGINT/SIGTERM). On stop it flips a `watch` channel,
   drains the feeder and supervisors (so in-band `Shutdown` and any kill-after-
   grace finish), aborts the server, and removes the socket.

`CoordinatorState` (in `session.rs`) is the shared state behind an
`Arc<Mutex<…>>`: the `Router`, the in-flight map, the current `Topology` and
difficulty `target`, per-miner `Configure`, and the salt→job bookkeeping.

## The blockchain seam

All chain access sits behind one trait, `ChainClient` (`chain/mod.rs`), with
three methods:

- `fetch_mining_snapshot` — the mining inputs at a block: topology
  (nodes/edges, allowed h/J/spin ranges), difficulty gates
  (`max_energy_milli`, `min_solutions`, `min_diversity_milli`), and the round
  anchor (`last_proof_block_hash`).
- `fetch_mempool_orders` — open mempool orders eligible for this miner.
- `submit_proof` — hybrid-sign and submit a proof extrinsic, then classify the
  receipt.

`RealChainClient` (`chain/real.rs`) is the live client over Substrate
JSON-RPC and subxt; `FakeChain` (`chain/fake.rs`) backs the tests. Supporting
modules: `extrinsic` (hybrid sr25519 + ML-DSA-44 signing, `load_hybrid_pair`,
`miner_identity_bytes`), `snapshot` (`MiningSnapshot`, `DecayParams`),
`mempool` (`JobOrder`), `submit` (`Proof`, `classify_receipt`, `SubmitAction`),
plus `proof_encode` and `scale_types` for SCALE codec. Nothing outside
`chain/` talks to the node directly.

## Job production — the feeder

`feeder_loop` (in `runtime.rs`) runs one poll every `poll_interval_ms`:

- **Follow the head.** It fetches the snapshot. When `last_proof_block_hash`
  changes, a new round has started: it bumps a generation counter, cancels the
  prior generation's staged jobs, refreshes the topology and difficulty target
  in `CoordinatorState`, and clears the salt map.
- **Stage PoW work.** For each registered miner it derives fresh jobs
  (`producer::derive_pow_job`) from the snapshot, the miner account, and a
  unique salt per attempt, and stages them until the miner's queue reaches its
  adaptive depth. Each distinct salt derives a distinct nonce, so every attempt
  is a fresh draw.

Mempool jobs carry generation `0` (`producer::mempool::job_order_to_job`) and
are preserved across reseeds; PoW jobs carry the live generation and are
dropped when their round ends.

### Adaptive staging window

The staged depth per miner tracks how fast that miner drains work, so a
many-core backend stays saturated while a slow one doesn't hoard jobs. Each
poll samples the miner's jobs-consumed counter (`Router::take_consumed`,
read-and-reset), smooths it with an EMA (`α = 0.3`), and sizes the window to
`ceil(ema × 2.0)` — about two poll-intervals of drain — with `buffer_depth` as
the floor and no ceiling. The EMA is anchored to real completions, so the depth
self-bounds to roughly headroom × actual throughput. The default floor is 256
(`main.rs`), generous enough to keep every miner fed from the first poll.

## Routing and credits

`Router` (`router.rs`) indexes miners by capability and holds a staged queue
per miner. `route` first-fits a job to a capable miner (by `supported_kinds`
and `max_nodes`/`max_edges` from the miner's `Hello`); `stage_on` targets a
specific miner, which is what the feeder uses.

Dispatch is gated by a **credit pool**, not a queue depth. The miner grants
`width` credits up front and one more per terminal event (`Result` or
`Reject`); each dispatch spends one credit (`next_job`). Dispatch stays
one-to-one with completion, and in-flight is bounded by the credits the miner
has actually issued. In-flight jobs live in `CoordinatorState`, so the
router keeps no separate outstanding counter.

On a reject with `UnsupportedKind`, the router records that the miner can't
serve that job kind and re-routes the job to another capable miner. On a miner
crash, `reclaim` returns its staged and in-flight jobs for re-queue.

## The miner session

`CoordinatorService<C>` is the server side of the gRPC `MinerService`
(`session.rs`) — the streaming miner protocol carried over the Unix socket. Per
session it runs the handshake (`Hello` → `Welcome`), applies `Configure`, then
services the bidirectional stream: a `JobRequest` grants credits, the
coordinator dispatches staged `Job`s, and each `Result` or `Reject` is a
terminal event that frees a credit and advances accounting. `Status` reports how
busy the miner is; `Shutdown` ends the session in-band. The harness owns
credits, reject reasons, and the idle timeout, so each backend only has to
sample.

## Supervision and shutdown

`supervise_miner` (`supervisor.rs`) owns one miner for the run. It spawns the
binary with the session token in the `QUIP_SESSION_TOKEN` environment variable
(never argv) and two arguments — `--quip-coordinator <socket-uri>` and
`--miner-id <id>` — sets `kill_on_drop`, and merges the child's stderr (miners
emit JSON log lines) into the coordinator's log tagged by miner id.

Restarts follow the child's exit code (`restart_policy`):

- `0` — clean exit; respawn on demand, reset backoff.
- `64` / `69` / `77` — operator, environment, or token error; don't respawn.
- `70` or a signal — crash; respawn after exponential backoff
  (`2^consecutive × base`, capped at `max`) under a failure budget. Too many
  crashes inside the window marks the miner unhealthy and stops respawning.

On a crash the miner's staged and in-flight jobs are re-queued. When the runtime
stops, each live miner gets an in-band `Shutdown` first, then a hard kill after
the grace period (`grace_ms`).

## Config to launch plan

`parse_config` (`config.rs`) turns `config.toml` into a `CoordinatorConfig`:
the `[miner]` section gives `validators` and `signer_key`; each backend section
becomes one `LaunchEntry`. Section names map to miner ids `[cpu]`→`cpu-0`,
`[cuda.N]`→`cuda-N`, `[metal]`→`metal-0`, `[dwave]`/`[qpu]`→`qpu-0`.

`binary` selects the executable, defaulting to `quip-<backend>-sa` (and
`quip-dwave-qa` for D-Wave). The coordinator-owned keys (`binary`,
`queue_depth`, `idle_timeout_s`, `heartbeat_s`, `reconnect_window_s`) become
fields of the `Configure` message; every other key passes through verbatim in
`Configure.backend_toml`, which the coordinator forwards to the miner. This is
how a backend receives its own settings without the coordinator knowing them.

## Drive mode

`quip-coordinator drive` (the `drive` module and `DriveService` in `session.rs`)
spawns one miner and feeds it synthetic work with no chain and no submit. Jobs
come from a golden random draw (`--source random`, optionally against a topology
preset under `crates/quip-coordinator/fixtures/drive/`) or a JSONL replay
(`--source list`). It reports per-job and aggregate timing, and can pin
`num_reads`/`num_sweeps` through the `SetTarget` control-plane override to run
matched-condition throughput and parity comparisons. This is the path used for
CPU-versus-CUDA benchmarking.
