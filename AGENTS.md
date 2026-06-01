# AGENTS.md — QuIP Protocol Project Instructions

Cross-tool instructions for AI coding assistants (Claude Code, Codex, Cursor, Gemini CLI).

For the **runtime architecture**, read [`ARCHITECTURE.md`](./ARCHITECTURE.md).
This file is the developer-facing how-to: commands, dependencies, code
style. Anything about how the system *runs* belongs in
`ARCHITECTURE.md`.

## Environment

The `.quip` virtualenv is already active in development shells —
don't prefix shell commands with `source .quip/bin/activate`.

```bash
# Fresh install (only if .quip doesn't exist yet)
python3 -m venv .quip
source .quip/bin/activate
pip install -U pip setuptools wheel
pip install -e .            # core + CPU
pip install -e .[cuda]      # CUDA backend
pip install -e .[metal]     # Apple Silicon backend
pip install -e .[dev]       # pytest + pytest-asyncio
```

`.env` holds `DWAVE_API_KEY` and other credentials. **Never read or
display its contents.**

## Running the miner

The CLI is `quip-miner` (defined in `quip_cli.py`, entry point in
`pyproject.toml`). It attaches to a substrate validator over WS or
HTTP; there is no longer any in-process P2P node to run.

```bash
# Generate a hybrid sr25519 + ML-DSA-44 keystore
quip-miner keygen --out ~/.quip-miner/signing.json

# Bootstrap (one-shot reachability + funding check against a validator)
quip-miner bootstrap --validator ws://127.0.0.1:9944 \
  --signer-key ~/.quip-miner/signing.json

# CPU PoW miner
quip-miner cpu --validator ws://127.0.0.1:9944 \
  --signer-key ~/.quip-miner/signing.json

# CUDA / Metal / D-Wave
quip-miner gpu --validator ws://127.0.0.1:9944 --gpu-backend local --signer-key ...
quip-miner gpu --validator ws://127.0.0.1:9944 --gpu-backend metal --signer-key ...
quip-miner qpu --validator ws://127.0.0.1:9944 --daily-budget 30s --signer-key ...

# Concurrent PoW + mempool in one process
quip-miner cpu --validator ws://... --mode both --num-cpus 4 --signer-key ...

# TOML config (see docker/quip-miner.cpu.toml, docker/quip-miner.cuda.toml)
quip-miner cpu --config ./docker/quip-miner.cpu.toml
```

Live integration uses the docker-compose validator under `docker/`
(`docker compose up quip-validator`); the validator listens on
`ws://127.0.0.1:9944` by default.

**Metal interactive cap (Apple Silicon):** the `[metal]` section runs an
adaptive governor when `yielding` is on (default). It senses HID-idle /
thermal / battery / displays and caps **GPU occupancy** — the jank lever is
concurrent threads per command buffer (`problems × reads`), not core count or
duty cycle. While you're present it splits reads so each command buffer stays
under `active_util` % of the GPU's max thread capacity
(`maxTotalThreadsPerThreadgroup × cores`; default 85); idle/headless runs
uncapped (full speed); thermal-serious halves it; battery / critical thermal
pause. Total reads and sweeps are always preserved. (On an M4 Max steady-state
mining is smooth even at full saturation, so the cap is mainly insurance for
weaker GPUs / sustained thermal load.) This path
is **independent of the CUDA util monitor** — it lives in
`GPU/metal_scheduler.py` + `GPU/macos_sensors.py` and shares no utilization
machinery with `GPU/util_monitor.py`. See `docs/metal-gpu-governor.md`.

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Single file / single test
python -m pytest tests/test_pool_client.py -v
python -m pytest tests/test_pool_client.py::test_get_head_forwards_empty_args -v
```

There are no `smoke_node_*.py` scripts in `tests/` anymore — the old
in-process P2P node smoke tests are gone. Live integration is via
the docker-compose validator described above.

## Benchmarking and tools

```bash
# CPU baseline
python tools/cpu_baseline.py --quick
python tools/cpu_baseline.py --quick \
  --topology dwave_topologies/topologies/advantage2_system1.json.gz

# Topology analysis
python tools/analyze_topology_sizes.py --configs "8,2" --samples 10
python tools/validate_mined_topology.py --all

# Precompute embedding for QPU (slow)
python tools/analyze_topology_sizes.py --configs "9,2" \
  --precompute-embedding --embedding-timeout 1w

# GPU benchmarks (Modal Labs)
modal run benchmarks/gpu_benchmark_modal.py
```

**Never run QPU benchmarks in the background.** Provide the command;
let the operator execute it.

### Modal Labs (cloud GPU)

```bash
pip install modal
modal token new  # opens a browser for authentication
```

## Dependencies

From `pyproject.toml`:

- **Core**: `dwave-ocean-sdk`, `numpy`, `aiohttp`, `click`, `blake3`,
  `substrate-interface`, `scalecodec`, `dilithium-py` (ML-DSA-44),
  `python-dotenv`, `tomli` (on 3.10).
- **CUDA** (optional): `cupy-cuda12x`, `nvidia-ml-py`.
- **Metal** (optional): `pyobjc-framework-Metal`,
  `pyobjc-framework-MetalPerformanceShaders`.
- **fast** (optional): `pyzmq`, `uvloop`.
- **dev**: `pytest`, `pytest-asyncio`.

Python 3.10+ required.

**Removed in v0.2**: `aioquic`, `hashsigs` (legacy SPHINCS+),
`cryptography` (was for self-signed TLS). The QUIC P2P stack went
with them.

## Topology management (`dwave_topologies/`)

- `topologies/*.json.gz` — hardware topology files (Advantage2, Chimera, Pegasus, Zephyr).
- `embeddings/*.json.gz` — precomputed embeddings for QPU hardware mapping.
- `embedded_topology.py`, `embedding_loader.py`, `smart_embedding.py` — loaders and embedding utilities.
- **Default**: Zephyr Z(9,2) — 1,368 nodes, 7,692 edges.
- For full Advantage2 hardware: `advantage2_system1.json.gz` (~4,579 qubits).

**QPU solver revision updates** (when D-Wave recalibrates):
1. Replace the topology file in `topologies/`.
2. Verify existing `embeddings/` still match the new graph; copy over compatible ones.
3. Delete stale topology files and incompatible embeddings.

## Critical parameters

**Genesis block defaults** (`genesis_block.json`):
```python
difficulty_energy = -2500.0
min_diversity = 0.2
min_solutions = 5
h_values = [-1.0, 0.0, 1.0]
```

**Production Z(9,2) targets:**
```python
difficulty_energy = -4100.0
min_diversity = 0.15
min_solutions = 5
```

**Energy ranges by topology (GSE):**

| Topology | Range | Size |
|---|---|---|
| Z(8,2) | -2869 to -2677 | 192 |
| Z(9,2) | -4100 to -3870 | 230 (default) |
| Z(10,2) | -5470 to -5200 | 270 |
| Z(11,4) | -15170 to -14158 | 1012 |
| Advantage2 (full) | similar to Z(11,4) | ~4579 qubits |

**QPU h/J ranges:** See [`docs/dwave-solver-ranges.md`](docs/dwave-solver-ranges.md)
for per-solver `h_range`, `j_range`, `extended_j_range`, and
`per_qubit_coupling_range`. Regenerate with
`python tools/dump_solver_ranges.py`.

**Per-miner adaptive parameters:**

| Backend | num_sweeps | num_reads |
|---|---|---|
| CPU/SA | 64–4096 | 64–512 |
| GPU/CUDA | 256–2048 | (adaptive) |
| GPU/Metal | 64–512 | (adaptive) |
| Modal | 128–4096 | (adaptive) |
| QPU | annealing 5–20µs | 32–64 |

## Identifiers: solution number vs `dispatch_id`

Two distinct identifiers exist; **never conflate them**, and never persist
on `dispatch_id`.

- **Solution number** — the chain-global *ordinal* of the QPoW solution
  being mined: `count(QuantumPow.WinningSolutions) + 1`. The chain has no
  stored solution counter — solutions are keyed in `WinningSolutions` by
  the block number they won at (`submitted_at == key`), so the ordinal is
  derived by counting that map's keys (cheap: one paged `state_getKeysPaged`
  over the storage prefix, keys only). It is **determinable at round
  start** — a round mines toward a specific upcoming solution number; the
  round's `last_proof_block_hash` (= `block_hash(LastProofBlock)`) stays
  constant until a proof wins and advances it. Compute the count **once per
  round and cache it** (recount only when `last_proof_block_hash` changes);
  it is global, monotonic, and never repeats.
  - **Not the same as a block number.** `LastProofBlock` (e.g. 52507) is
    the block number of the most recent winning proof — the round anchor,
    not the solution ordinal (e.g. 196). Don't conflate them.
  - **This is the on-disk key for the mining-attempts archive**
    (`{base}/{solution_number}/…`). Because it tracks the logical
    solution, a controller/worker restart mid-round correctly *resumes*
    writing into the same solution dir — that is not stale accretion, it
    is the same solution. Different solution number ⇒ different dir.

- **`dispatch_id`** — an **internal-only** controller↔worker coordination
  handle: `_dispatch_contexts[(handle_id, dispatch_id)]`, cancel-ack
  (`_await_handle_done`), and the preview channel (see `ARCHITECTURE.md`
  §3.3). It is a process-local monotonic counter that **resets to 0 on
  restart**, so it collides across runs and MUST NOT be used as a durable
  or on-disk identity. Keep it in memory for pairing responses to
  contexts; never name persisted artifacts after it.

The old `SubmissionLogger` `next_solution_id` local counter (persisted in
`{base}/next_solution_id`) was a stand-in for the chain solution number
and is misaligned with this model — replace it with the chain-derived
solution number, do not extend it.

## Code style

- **Imports** at top of file, in stdlib → third-party → local order. Absolute imports only. No inline imports inside functions/methods. Exception: optional-dependency `try/except` at module level.
- **Concurrency**: NEVER use threads. Multiprocessing only. Async via `asyncio` for network operations. Mining runs in worker processes (`shared/miner_worker.py:MinerHandle`).
- **Type hints** on public APIs. Google-style docstrings on non-trivial public functions.
- **Logging** via `logging.getLogger(__name__)` at module level. The custom formatter and component classification live in `shared/logging_config.py`.
- **No `Co-Authored-By: <assistant> ...` lines in commits.**

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) §9 for the cleanup-candidates
list (vestigial code flagged for removal).

## Deployment

- **Docker** (`docker/`): `Dockerfile.cpu`, `Dockerfile.cuda`, `docker-compose.yml`, `entrypoint.sh`, plus the example TOML configs (`quip-miner.cpu.toml`, `quip-miner.cuda.toml`).
- **Cloud**: `akash/` and `aws/` contain deployment configs.
- **GitLab CI** (`.gitlab-ci.yml`): builds CPU + CUDA Docker images on main/tags.

## Concurrency: processes, not threads

Use `multiprocessing` for concurrency. Do **not** introduce `threading.Thread`
in our own code. Threads share the GIL; a CPU-bound thread starves its
siblings. This stalled the QPU stream pump in v0.2 — the per-attempt consumer
held the GIL ~1.2 s and the pump thread could not drive the sampler, so the
QPU pipeline drained (throughput 0.6 vs ~8 subs/sec).

Rules:
- New background work → a `multiprocessing` process (use `spawn`).
- Share scalars via `multiprocessing.Value`; share large numpy buffers via
  `multiprocessing.shared_memory.SharedMemory` (zero-copy), never by pickling
  per item on a hot path.
- `threading.Lock`/`RLock` for intra-process state is fine (a lock is not a
  thread). Cross-process correctness uses `os.replace`/PIPE_BUF atomicity or
  `mp` primitives, not locks.
- Exceptions we don't control: third-party internal threads (D-Wave SDK,
  asyncio executors, stdlib `QueueListener` if ever reused). Document any such
  exception inline with the reason.

## Debugging a hung process (get a traceback)

`SIGINT` (Ctrl-C) only helps when the **main thread is running Python
bytecode** — it raises `KeyboardInterrupt` at the next bytecode boundary. A
process wedged in a C-level call (a lock/`join`/`wait`, an `mp.Queue`
feeder-thread join at interpreter shutdown, a blocking syscall) won't unwind,
so `SIGINT` just kills it with **no traceback**. Don't reach for it on a hang.

Use **`faulthandler`** + **`SIGABRT`** — the repo convention (CI already runs
`timeout --signal=ABRT … python -X faulthandler -m pytest`):

```bash
# 1. Start the process with faulthandler enabled (installs handlers for the
#    fatal signals SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL).
PYTHONFAULTHANDLER=1 python tools/whatever.py …      # or: python -X faulthandler …

# 2. When it hangs, dump every thread's Python (and C) stack to stderr:
kill -ABRT <pid>          # SIGABRT -> faulthandler dumps, then the process aborts (exit 134)
```

This prints the exact frame each thread is stuck in (e.g. `threading.py:wait`
→ an unjoined queue feeder thread). Notes:

- **Pre-arm it.** Faulthandler must be enabled *before* the hang. For tools we
  expect to run interactively against the QPU/long pipelines, prefer enabling
  it (env var or `faulthandler.enable()` at startup) so a hang is debuggable.
- **Dump without killing:** `faulthandler.register(signal.SIGUSR1)` in code,
  then `kill -USR1 <pid>` dumps and **continues** (repeatable). `SIGABRT` is
  fatal; `SIGUSR1` (registered) is not.
- **No instrumentation available?** `py-spy dump --pid <pid>` attaches to any
  running CPython and prints all-thread tracebacks without a signal or restart.
- **Inspect the tree first:** `ps -o pid,ppid,stat,command -p <pid>` and
  `pgrep -P <pid>` reveal stuck children / orphans (`PPID 1` = reparented after
  a parent crash). An interpreter that won't exit is usually blocked joining a
  non-daemon child or an `mp.Queue` feeder thread (call `cancel_join_thread()`
  on queues whose buffered data is worthless at teardown).
