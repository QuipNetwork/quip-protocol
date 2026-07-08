<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 QUIP Protocol Contributors
-->

# Phase 1 — `hardest_models` corpus — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans. **Tasks live in beads**, not as
> checkboxes here — run `bd ready` and `bd show <id>`; each issue carries its own
> Files / Consumes / Produces / TDD steps.

**Goal:** Build `hardest_models` — Ising instances at the hardness edge, bucketed by
`topology_hash`, ranked by lowest achieved energy, with a structural hardness-signature /
filter analysis — to feed the Phase 2 SA-vs-QPU testing framework.

**Architecture:** Harvest nonces from every source that kept them (QPU archives + chain
wins), regenerate (h, J) deterministically from each nonce, verify + bucket by
`topology_hash`, rank by depth, and characterize. Nonce = instance key;
`generate_ising_arrays_from_nonce` regenerates instances offline and self-verifies.

**Tech stack:** Python (.quip venv / uv), numpy, scipy.sparse.linalg; reuses
`shared/quantum_proof_of_work.py`, `shared/topology_hash.py`, `qpu_archive_lib.py`,
`tools/download_and_validate_wins.py`, `substrate/client.py`.

**Spec:** `docs/superpowers/specs/2026-07-08-phase1-corpus-acquisition-characterization-design.md`

## Global constraints

- `.quip` venv (uv); ruff + ty clean; AGPL header on new files; absolute imports;
  ≤100 lines/function; Google docstrings; no LLM co-author trailers.
- Scripts under `test_results/hardest_models/tools/`; data under
  `test_results/hardest_models/<topology_hash>/` (test_results is gitignored).
- `topology_hash = shared/topology_hash.py:topology_hash(nodes,edges,allowed_h,allowed_j,allowed_spin)`.
  **Never cross-pool topology buckets** (h0 vs ternary differ in hash and scale).
- Any live chain/QPU call is **read-only and operator-launched, never backgrounded.**

## File structure

```
test_results/hardest_models/tools/
  harvest_nonces.py    regen_verify.py    rank_select.py    characterize.py
  chain_pull.py        run_phase1_smoke.sh    tests/
test_results/hardest_models/
  _raw/<topology_hash>.jsonl        _snapshots/<hash>.json
  <topology_hash>/instances.jsonl   <topology_hash>/characterization.{json,md}   <topology_hash>/figures/
  manifest.json
```

## Task index (beads epic `quip-protocol-h6d`)

| bead | task | blocked by |
|---|---|---|
| `h6d.1` | `harvest_nonces.py` — sweep sources, bucket by `topology_hash` | — |
| `h6d.2` | `chain_pull.py` — rank chain wins by energy + resolve/cache snapshot | — |
| `h6d.3` | `regen_verify.py` — regenerate (h,J), assign+verify bucket, export `load_instance` | h6d.1, h6d.2 |
| `h6d.4` | `rank_select.py` — per-topology rank ≤10k, drop under-populated, manifest | h6d.3 |
| `h6d.5` | `characterize.py` — features + hardness-correlation + filter test | h6d.4 |
| `h6d.6` | end-to-end smoke + exit-criteria verification | h6d.5 |

Ready to start: `h6d.1` and `h6d.2` (independent). Run `bd ready`.

## Self-review

- **Spec coverage:** harvest-all-families → h6d.1; chain pull + one read-only snapshot →
  h6d.2; regen/verify + loader → h6d.3; rank/cap-10k/floor/manifest + "sufficient hardness"
  → h6d.4; structural features + correlation + filter + baseline → h6d.5; layout + exit
  criteria + smoke → h6d.6. The two open operator items (floor value, 10k tie-break) are
  h6d.4 CLI flags. No gaps.
- **Placeholders:** none — each bead names concrete files, functions, and test assertions.
- **Type consistency:** `load_instance` defined in h6d.3, consumed in h6d.5; `HarvestRecord`
  from h6d.1 flows through; `topology_hash` bucketing consistent throughout.

## Open items (operator, folded into h6d.4)

- Under-populated floor (min edge-band instances to keep a bucket) — proposed a few hundred.
- 10k cap tie-break — strict-by-energy (default) vs stratified across the edge band.
