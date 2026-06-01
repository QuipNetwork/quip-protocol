# Miner Architecture — Unified Streaming Stack

Every backend (QPU, Metal, CUDA, CPU, Modal) and both job types (PoW and
mempool) mine through a single path: a **stream-driver subprocess** produces
samplesets into a **shared-memory ring**, and the **worker** consumes the ring
and runs evaluate / ratchet / stash. There is no synchronous/inline sampling
path and no mode toggle.

## Layers

```
L1  Orchestrator ── chain head + submit
      │  dispatch: (job kind, feeder spec, round params) via ctl_q "switch"
      ▼
L2  Coordinator / worker ── spawns ONE stream-driver process per device;
      │   owns the SampleView ring; consumes descriptors off desc_q;
      │   runs the evaluator (ratchet for PoW, mempool-eval for mempool)
      ▼
L3  Stream-driver process (StreamContext) ── feeder + sampler
      │   iter_results(): for (model, ss, gen) in
      │       sampler.sample_ising_streaming(feeder, ...): write ring slot
      └── desc_q.put((slot, ...)) ──▶ back to L2 (ring carries bulk arrays)
```

- `QPU/stream_driver.py:stream_driver_main` is the L3 entry point. It resolves a
  backend `build_persistent_context` factory, builds a `StreamContext`, then
  loops `apply_command` (ctl_q) ↔ `iter_results` (ring writes) until shutdown.
- The ring is `shared/ring_views.py:SampleView` over `shared/shared_ring.py:SharedRing`.
  The driver writes `(sample, energy)` into a slot; metadata rides the
  descriptor queue. The descriptor is an 8-tuple:
  `(slot, n_rows, n_cols, nonce, salt, qpu_us, generation, defect_pickle)`.
  `defect_pickle` is `None` for every non-QPU backend. A trailing `None` on
  `desc_q` is the end-of-stream sentinel.

## The invariant

The sampler is the only per-backend difference. The feeder (`pow` / `mempool`)
is the only per-job difference. `shared/stream_context.py:StreamContext` is
backend- and job-agnostic: it owns the feeder, generation counter, and pause
flag, and drives whatever sampler it was handed.

`tools/lint_no_inline_sampling.py` (run in CI and as a pytest check) fails if
the deleted `_sample` / `_sample_batch` / `STREAMING_PUMP` / `DRIVER_OWNS_FEEDER`
symbols reappear in source.

## Per-backend factories

Each backend exposes a `build_persistent_context(...)` factory that builds its
sampler and returns a `StreamContext` wired with it:

| Backend | Factory |
| ------- | ------- |
| QPU     | `QPU/dwave_miner.py` |
| Metal   | `GPU/metal_stream.py` |
| CUDA    | `GPU/cuda_stream.py` |
| CPU SA  | `CPU/sa_stream.py` |
| Modal   | `GPU/modal_stream.py` |

The factory runs ONLY in the stream-driver process. Metal init failure (no
Metal device, kernel compile failure) raises `RuntimeError` and CRASHES the
driver — there is no CPU fallback.

## The sampler contract

```python
sample_ising_streaming(feeder, *, num_reads, num_sweeps, **sampler_kwargs)
    -> Iterator[(IsingModel, SampleSet)]
```

A generator that pops models from `feeder` and yields `(model, sampleset)` per
completion. It must guard `feeder.pop_blocking()` against `StopIteration` (PEP
479: a bare `StopIteration` escaping a generator body becomes a `RuntimeError`)
— catch it and `return`. CPU SA and Modal are thin generators (pop one model,
`sample_ising`, yield); Metal and CUDA already streamed. `StreamContext` passes
`num_reads` / `num_sweeps` explicitly and forwards backend kwargs from
`sampler_kwargs`.

## Feeder spec (carried on the `switch` ctl_q command)

The `switch` command carries a picklable `feeder_spec`; `StreamContext` builds
the feeder from it via `shared/ising_feeder.py:build_feeder`:

- PoW: `("pow", last_proof_block_hash, miner_bytes)` → `RandomIsingFeeder`
  (a `"pow"`→`"pow"` switch reseeds the existing feeder rather than rebuilding).
- Mempool: `("mempool", problemview_attach_args, slot)` → `FixedIsingFeeder`.

Mempool transports the fixed `(h, J)` zero-copy: the worker writes the order's
problem into a one-slot `ProblemView` and sends its `attach_args()` in the spec.
Unlike `SampleView`, `ProblemView.attach_args()` is **free_q-less**, so the spec
is small enough to ride the ctl_q. The driver attaches the `ProblemView` as a
non-owner and reconstructs the `IsingModel`. QPU mines mempool identically — its
sampler simply streams reads of the fixed model.

## QPU specifics

The QPU sampler (`QPU/dwave_sampler.py:sample_ising_streaming`) runs an async
pump: it keeps `queue_depth` submissions in flight via `sample_ising_async`,
polls futures, and yields each completion as a **raw reduced** sampleset (only
the qubits this problem uses) with a per-model `DefectInfo` on
`sampleset.info['defect_info']`. The producer does NO reconstruction gating; it
ships the reduced sample plus the pickled `DefectInfo` (the descriptor's 8th
element). The consumer (`shared/base_miner.py:_acquire_result`) runs the cheap
decay `_stash_pre_check` on the corrected energy and reconstructs only survivors
to full topology — reconstruction is pure compute in the worker, no D-Wave
connection. In-flight futures are cancelled on `GeneratorExit` when a chain-head
switch closes the generator.
