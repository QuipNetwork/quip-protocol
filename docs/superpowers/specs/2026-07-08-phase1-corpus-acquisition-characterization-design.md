<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 QUIP Protocol Contributors
-->

# Phase 1 — `hardest_models` corpus acquisition & characterization — design

**Date:** 2026-07-08
**Status:** design (awaiting review)
**Part 1 of 2.** Phase 2 (SA-vs-QPU testing framework) is a separate spec
(`2026-07-08-sa-vs-qpu-same-instances-hardness-edge-design.md`) and is
brainstormed after this one ships. This phase produces the `hardest_models`
corpus Phase 2 consumes.

## Purpose

Acquire and characterize **`hardest_models`** — a corpus of Ising model
instances at the edge of the hardness regime, **organized by topology**, sourced
from all of `test_results` plus the chain, prioritized by **lowest achieved
energy**. Phase 2 uses it to run paired SA-vs-QPU comparisons on identical
instances. Along the way, determine whether the hard/edge instances carry a
**structural signature** we could filter on.

## Scope (locked by review)

- **Dataset name: `hardest_models`.** One **bucket per topology**; buckets are
  never cross-pooled.
- **Topology = `topology_hash`** = `shared/topology_hash.py:topology_hash(nodes,
  edges, allowed_h, allowed_j, allowed_spin)` — the graph **and** the allowed
  h/J value sets. So h0-Advantage2 and ternary-Advantage2 on the same physical
  graph are **different topologies** (different hash, different energy scale).
  **Bucket by `topology_hash`:** the harvest computes each source's hash from its
  spec and groups on it; any source whose hash equals the chain default folds
  into the priority bucket. Membership is then self-verified by regeneration (the
  instance's (h, J) must reproduce its recorded energy under that hash's spec).
- **Size: up to 10k hardest instances per topology** (fewer if that's all that's
  available; **~5k expected** for the current default network topology — the
  harvest reports the real per-bucket counts).
- **Keep any regenerable topology with sufficient hardness**, including non-h0
  ones. **Drop** topologies that are un-regenerable (retired
  `Advantage2_system1.7`, topology file removed) or under-populated below the
  usable threshold (e.g. the 3-instance `Zephyr Z9_T2`).
- **Characterization = descriptive + hardness-correlation + filter test** vs a
  random-nonce baseline. **Mempool `JobOrders` are out of scope** for this study.

## Key facts (from the design investigation)

- An instance is a deterministic function of its nonce and topology:
  `generate_ising_arrays_from_nonce(nonce, nodes, edges, allowed_h, allowed_j)`
  → numpy `(h_arr, j_arr)` (`shared/quantum_proof_of_work.py`). The **nonce is
  the instance key**; (h, J) regenerates on load and self-verifies against a
  recorded solution's energy.
- The **graph is fixed** within a topology; only (h, J) *values* vary. At h0,
  h ≡ 0, so h0 instances differ **only in their J couplings** — each is a random
  frustration pattern on the fixed hardware graph.
- **All PoW instances are i.i.d. from one generator.** "Chain vs random" is the
  same distribution; the meaningful structural questions are whether there is a
  per-instance hardness signature *within* the distribution.
- **Nonce retention is the gating asymmetry:** only QPU-side captures and chain
  wins retain nonces. SA-side `attempts.jsonl` and the `reads_diversity_count`
  aggregates discarded them → not harvestable.
- The on-chain `topology_hash 0x6eae38a9…` did **not** reproduce from the bundled
  topology files under either allowed-h spec, so the current default topology's
  exact spec is resolved by **one read-only snapshot fetch** — and the ~31k
  archived h0 instances only join the default-topology bucket if they verify
  against that hash.

## Harvest map (nonce-bearing sources, bucketed by topology)

| source | topology bucket | scale | ~raw count | disposition |
|---|---|---|---|---|
| `.../qpu_reads_diversity/data/*/meta.jsonl` (+ `shards/*.npz` energies) | Advantage2_system1 / h0 spec | h0 | ~31k | **keep** (verify vs default-topology hash) |
| `test_results/qpu_tts_test/raw/*.stored.jsonl` | Advantage2_system1 / ternary spec | ternary | ~10⁴ | **keep** if bucket meets the hardness threshold |
| `test_results/metal_{sa,gibbs}_tts_*/raw/*.stored.jsonl` | Advantage2_system1 / ternary spec | ternary | 5 energies ea. | **keep** (merges into the ternary bucket) |
| root `quip_wins.wins.jsonl`, `quip_recent.wins.jsonl` + chain `--dump-bqm` | `0x6eae38a9…` (current default) | resolved by snapshot | 288 + 80 + pull | **keep** (the priority bucket) |
| `qpu-test-full-topology/*.json` | Advantage2_system1.7 (retired) | — | 3 | **drop** — topology file gone, un-regenerable |
| `qpu-test-z9_t2/*.json` | Zephyr Z9_T2 | — | 3 | **drop** — under-populated |

`reads_diversity_count` aggregates and SA `*.attempts.jsonl` carry **no nonce** —
excluded. Mempool `JobOrders` — excluded per review.

## "Sufficient hardness" (bucket inclusion + per-instance selection)

- **Per instance:** rank by achieved best energy (most negative first); take the
  hardest up to 10k.
- **Per bucket:** keep a topology only if it yields a usable number of
  edge-band instances (proposed floor to confirm: ≥ a few hundred), else drop as
  under-populated. The edge band is the topology's own winnable frontier (its
  deepest achievable energies) — computed from the harvested achieved energies,
  not hardcoded across topologies (the h0 4577-node band is −14490…−14520 on the
  h0 scale; other topologies get their own band from their data).

## Components

New code under `test_results/hardest_models/` (a new sibling of the phase-B
bundle), reusing shared primitives.

| unit | responsibility | reuses |
|---|---|---|
| `harvest_nonces.py` | scan every source; compute each source's `topology_hash` (`shared/topology_hash.py`) from its (graph, allowed_h, allowed_j); dedup by `(nonce, topology_hash)`; record achieved best energy + provenance + on-chain `difficulty` where present; group by hash; emit `hardest_models/_raw/<topology_hash>.jsonl` | `shared/topology_hash.py`, `qpu_archive_lib.py` |
| chain pull (script wrapper) | `download_and_validate_wins.py --dump-bqm` on the lowest-energy wins; **one read-only** `get_mining_snapshot(0x6eae38a9…)` to resolve the default topology's `(nodes, edges, allowed_h, allowed_j)` and hence its bucket + scale | `download_and_validate_wins.py`, `substrate/client.py` |
| `regen_verify.py` | nonce → (h, J) via `generate_ising_arrays_from_nonce` against each candidate topology's spec; assign the instance to the topology whose spec **reproduces its recorded energy**; flag unverifiable instances | `shared/quantum_proof_of_work.py`, `dwave_topologies` |
| `rank_select.py` | per topology, rank by lowest achieved energy; take up to 10k; drop under-populated buckets; emit `hardest_models/<topology_id>/instances.jsonl` | — |
| `characterize.py` | per topology: structural features per instance + a random-nonce baseline; hardness-correlation; filter test; emit `hardest_models/<topology_id>/characterization.{json,md}` + figures | `numpy`, `scipy.sparse.linalg` |

## `hardest_models` layout & per-instance schema

```
test_results/hardest_models/
  <topology_hash>/          # bucket dir IS the topology hash; readable alias in manifest
    instances.jsonl         # ranked, hardest first
    characterization.{json,md}
    figures/
  manifest.json             # topology_hash → {alias, scale, allowed_h/j, n, energy band}
```
`instances.jsonl`, one line per instance:
```
{ "instance_id", "topology_hash", "topology_alias", "scale",
  "nonce_hex", "salt_hex",
  "best_energy_milli_seen",                 # lowest-energy ranking key
  "difficulty": {"max_energy_milli", "min_solutions", "min_diversity_milli"} | null,
  "provenance": {...} }                      # archive cell+idx, or block_number
```

## Structural characterization (point 3)

Per instance (graph fixed within a topology, so features are on the sampled
weights):

- **Coupler-sign balance:** fractions of negative (AFM) / zero / positive J.
- **|J| magnitude distribution:** moments + histogram (richer where `allowed_j`
  has multiple magnitudes).
- **Frustration density:** fraction of frustrated short cycles over a fixed cycle
  basis / native short cycles (Zephyr has few triangles → sampled shortest-cycle
  basis; document the estimator + variance). h0's only hardness lever.
- **Spectral summary:** extreme eigenvalues + spectral gap of the signed,
  J-weighted adjacency (sparse Lanczos on the ~4577-node graph — fast).
- **(non-h0 topologies)** local-field h distribution and h–J alignment.

Analysis, **per topology**:

- **Hardness correlation:** Spearman rank-correlation of each feature vs achieved
  best energy → which features (if any) predict depth.
- **Filter test:** can a simple threshold / logistic on the features separate the
  deep (edge) instances from typical draws? Report AUC / precision.
- **Baseline:** a random-nonce sample from the same generator; confirm the harvested
  instances match the baseline distribution and locate where the edge instances
  sit in feature space.

Deliverable: per-topology `characterization.md` (the hardness-signature finding +
a clear filter verdict) + figures (feature-vs-depth scatter, distribution
overlays).

## Data flow

```
test_results/* + root quip_*.wins.jsonl ─▶ harvest_nonces.py ─▶ _raw/<topology>.jsonl
chain: download_and_validate_wins.py --dump-bqm
       + 1 read-only get_mining_snapshot(0x6eae38a9…)   ─┘ (resolves default topology spec)
_raw/* ─▶ regen_verify.py (assign+verify by topology_hash)
       ─▶ rank_select.py  ─▶ hardest_models/<topology>/instances.jsonl (up to 10k, hardest first)
       ─▶ characterize.py ─▶ hardest_models/<topology>/characterization.* + figures/
```

## Testing

- **Harvest/dedup unit test:** synthetic files across topologies → correct
  bucketing, counts, and **no** cross-topology merge.
- **Regen/verify unit test:** a known nonce reproduces a known energy under the
  right topology and **fails** under the wrong one (proves bucket assignment).
- **Characterization unit test:** features on a tiny hand-built instance with a
  known frustration pattern (e.g. one frustrated triangle) return expected values.
- **Smoke:** harvest a few real files + 3 chain wins → regen/verify →
  rank_select → characterize → one figure, end-to-end.

## Provenance & caveats

- **No cross-pool** across topology buckets — enforced by `topology_hash`.
- Use the nested `.../h0_gpu_vs_qpu_phaseB/h0_gpu_vs_qpu_phaseB/...` copies; the
  top-level `test_results/reads_diversity_count_h0/` is stale.
- Two `phaseA_e14950_*` dirs under the h0 tree are mislabeled ternary and lack
  nonces — excluded anyway.
- The one live call is a **read-only** RPC snapshot fetch; the rest of harvest is
  offline. No live/QPU work is backgrounded.
- Dropped: retired `Advantage2_system1.7` (un-regenerable) and `Zephyr Z9_T2`
  (under-populated); mempool `JobOrders` (out of scope).

## Open items for the operator

- Confirm the **under-populated floor** (min edge-band instances to keep a
  topology bucket) — proposed: a few hundred.
- Confirm the **10k cap** per topology and how to break ties at the cap (e.g.
  strictly by energy, or stratified across the edge band).

## Phase 1 exit criteria (hand-off to Phase 2)

1. `hardest_models/` — per-topology `instances.jsonl`, ranked by lowest achieved
   energy (≤10k each), + `manifest.json`, + a regenerate/self-verify loader.
2. Per-topology `characterization.md` — the hardness-signature finding + filter
   verdict.
3. The current default `topology_hash 0x6eae38a9…` resolved (spec + scale) and
   cached; the priority bucket populated (~5k expected).
