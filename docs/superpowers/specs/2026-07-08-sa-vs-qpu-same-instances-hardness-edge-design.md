<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 QUIP Protocol Contributors
-->

# Phase 2 — SA vs QPU testing framework on identical Ising models — design

**Date:** 2026-07-08
**Status:** design (awaiting review)
**Part 2 of 2.** Consumes the Phase 1 `hardest_models` corpus
(`2026-07-08-phase1-corpus-acquisition-characterization-design.md`).
**Supersedes:** `2026-07-06-gpu-vs-qpu-per-attempt-difference-design.md` (the
Jack-correction framing is retired — shipped as tables in the Notion page and
`test_results/h0_gpu_vs_qpu_phaseB/gpu_vs_qpu_offsets_for_jack.md`).

## Purpose

On the `hardest_models` corpus (edge of the hardness regime), **re-calibrate
both engines for the extreme end**, then run a **paired, equal-attempts
SA-vs-QPU comparison on identical instances** — same models, same order, plus
multiple orderings as a drift control — analyzed as **McNemar + pooled by
hardness bin**. An initial **baseline with the existing (shipped) params**
quantifies what re-calibration buys.

## The question

At the deepest end of the range, on identical instances: after re-calibrating
each engine for that regime, how often does SA win where the QPU does not (and
vice versa), and how do win rate, time-to-solution, and per-attempt success
probability compare as a function of instance hardness — and did re-calibration
change the answer versus the shipped params?

## Consumes (from Phase 1)

- `test_results/hardest_models/<topology_hash>/instances.jsonl` (ranked, ≤10k),
  `manifest.json`, cached topology snapshots.
- `regen_verify.load_instance(record) -> (h_arr, j_arr)` — the reusable,
  self-verifying instance loader.
- The per-topology characterization / filter verdict (may inform stratification).

## Corpus split (prevents overfitting the calibration)

Per topology, split `hardest_models` into a **calibration slice** (held out,
default 20%) and a **comparison slice** (default 80%), **stratified by hardness
bin** so both slices span the edge band. Stage A (calibration + baseline) touches
**only** the calibration slice; Stage B (the head-to-head) touches **only** the
comparison slice. The pinned parameters are therefore never tuned on the
instances they are ultimately judged on.

## Stage A — calibration & baseline (point 1)

Everything in Stage A runs on the **calibration slice**.

1. **Baseline (shipped params):** run the Stage-B equal-attempts protocol with
   the **existing** params (SA shipped adapt; QPU r=112 / 80 µs; gate k=5,
   D=0.2). Records baseline per-instance win rates — the control the re-derived
   params must beat.
2. **SA re-derivation:** re-fit the SA reach/cost model at the extreme end
   (`analyze_h0_opt.py`-style: `sec/attempt = a + b·R·S`; reach
   `ln q = α(E) + γ·ln S`), search the (R, S) grid for max wins/GPU-hour, and
   **regenerate `ADAPT_MAX_SWEEPS` + the adapt_params curve** (phase B flagged
   the shipped adapt ~3.4× suboptimal and `ADAPT_MAX_SWEEPS` ~16× too low).
3. **QPU re-scan:** archive-mode sweep of `num_reads` and `annealing_time`,
   re-scored offline, to locate r\*/anneal at the extreme end and confirm/adjust
   the QPU policy (phase B found r\* interior and anneal flat at h0 — re-verify
   here).
4. **Gate check:** confirm the count gate binds and diversity is ≈free at extreme
   depth on the calibration slice (may not extrapolate from phase B).
5. **Pin + compare:** pin one **(SA config, QPU config, gate)** set; report the
   signed delta vs the baseline (what calibration bought). Output:
   `calibration_report.md` + `pinned_params.json`.

## Stage B — paired comparison (point 2)

Runs on the **comparison slice**, per topology, with the params pinned in A5.

- **Equal attempts:** each instance gets **N ≈ 30** attempts (SA) / submissions
  (QPU) — enough to stabilize each per-instance estimate past small-sample noise.
- **Same models, same order:** both engines process the identical instance list
  in the identical canonical order `O0`; the pairing is per instance.
- **Per instance, per engine:** binary **win-within-N** (≥1 chain-valid solution
  at the instance's own threshold — all k selected solutions clear it, diversity
  ≥ D) and the raw **#wins/N**.
- **Per-instance threshold:** the instance's own difficulty — chain instances use
  the on-chain `difficulty.max_energy_milli`; archive instances use a threshold
  at/near their bucket's edge band. Both engines are judged at the **same**
  per-instance threshold.
- **Ordering control (drift detector):** on a subsample (default 200 instances),
  re-run each engine under **K ≈ 3** random orderings `{O1..OK}`. Under pinned
  params the per-instance win rates should be order-invariant; a shift exposes
  session drift (GPU thermal, QPU inter-call calibration) or stateful
  contamination — pool only if stable, else report the drift.

## Analysis

1. **Paired McNemar 2×2 per hardness bin:** both-win / SA-only / QPU-only /
   neither on identical instances, with the McNemar statistic — the core result.
2. **Pooled win rate & throughput by bin:** SA wins/GPU-hour vs QPU
   wins/QPU-hour (QPU including full-networking wall), CIs, vs achieved-energy
   depth.
3. **Per-attempt & per-second success-probability ratio by bin** (the shipped
   Notion per-attempt view, now on paired instances, with the ~92× attempt-time
   annotation).
4. **Baseline-vs-recalibrated delta** (from Stage A) and the **ordering-control**
   readout.
5. **Cross-topology overlay** — does the result hold across topology buckets.

Wilson/bootstrap CIs throughout; `is_soft` where either engine has <5 events in
a bin (do not quote as a constant).

## Engine configs

- **SA:** the topology's graph; (R, S) + adapt from Stage A (baseline uses the
  shipped adapt). Real-card wall time recorded (re-fits `0.032 + 5.41e-6·R·S`).
- **QPU:** r\*/anneal from Stage A (baseline r=112 / 80 µs). Instance-pinned
  submissions.
- **Gate:** from Stage A (default k=5, D=0.2).

## Components

New code under `test_results/hardest_models/tools/phase2/` (data under
`test_results/sa_vs_qpu_paired/`), reusing the phase-B + Phase-1 pipeline.

| unit | responsibility | reuses |
|---|---|---|
| `corpus_split.py` | stratified calibration/comparison split of `hardest_models` per topology | Phase 1 `manifest.json` |
| SA instance-pinned mode | `cuda_tts_canary.py` mode: consume a corpus + fixed params + fixed order → per-instance per-attempt records keyed by instance_id | `cuda_tts_canary.py`, `regen_verify.load_instance` |
| QPU instance-pinned mode | `qpu_throughput_canary.py` mode: same, fixed nonces/order → per-read energies keyed by instance_id | `qpu_throughput_canary.py`, `QPU/dwave_submitter.py` |
| `calibrate_sa.py` | re-fit SA reach/cost on the calibration slice; regenerate `ADAPT_MAX_SWEEPS` + adapt curve | `analyze_h0_opt.py` |
| `calibrate_qpu.py` | QPU archive-mode sweep on the calibration slice; rescore; r\*/anneal | `rescore.py`, `tts_policy.py` |
| `baseline_and_pin.py` | run baseline (shipped params) + compare vs re-derived; emit `pinned_params.json` + `calibration_report.md` | the two runners |
| `pair_consolidate.py` | per instance win/#wins per engine; McNemar per bin; pooled rates/throughput/per-attempt; ordering-control; CIs → `paired_frontier.json` | `phase_b_consolidate.py`, `estimators.py` |
| `plot_paired.py` | render the figures | `plot_gpu_qpu_tts.py` |

## Data flow

```
hardest_models/<topo>/instances.jsonl ─▶ corpus_split.py ─▶ {calibration, comparison} slices
calibration slice ─▶ calibrate_sa.py / calibrate_qpu.py ─▶ re-derived curves
                  ─▶ baseline_and_pin.py (shipped-params baseline + compare) ─▶ pinned_params.json + calibration_report.md
comparison slice + pinned_params ─▶ SA-pinned + QPU-pinned runners (equal-attempts, O0 + K orderings)
   └─▶ raw_paired/<engine>/<instance_id>/  ─▶ pair_consolidate.py ─▶ paired_frontier.json ─▶ plot_paired.py ─▶ figures/
```

## Figures

1. McNemar 2×2 per hardness bin. 2. Pooled win-rate/throughput vs depth. 3.
Per-attempt & per-second ratio vs depth. 4. Baseline-vs-recalibrated delta. 5.
Ordering-control (win-rate vs ordering). 6. Cross-topology overlay.

## QPU budget (operator-launched, never backgrounded)

Equal-attempts footprint ≈ `N × |comparison slice| × 0.062 s` per topology, plus
the calibration sweep and the K-ordering subsample. For the default topology
(~5k models, N=30) the comparison alone is ~155 min QPU ≈ ~5 daily Leap quotas;
the operator sizes N / slice / K to the quota, runs the ordering control on a
subsample (not the full slice), and launches every QPU run manually.

## Testing

- **Split determinism:** stratified split is reproducible from a seed; slices are
  disjoint and both span the edge band.
- **McNemar/binning unit:** synthetic paired outcomes → correct 2×2 + statistic;
  Wilson on sparse bins.
- **Ordering-control unit:** identical per-instance outcomes under permuted order
  → "stable"; injected drift → "unstable" flag.
- **Regression anchor:** the re-derived SA on the comparison slice reproduces the
  phase-B `gpu_yield` band within CI at matching depth.
- **Smoke:** tiny split → short baseline + short calibrate + short comparison (3
  instances, N=3, K=2) → consolidate → one figure, end-to-end.

## Provenance & caveats

- No cross-pool across topology buckets; held-out split prevents overfit.
- Ordering is tested precisely because QPU inter-call drift and GPU thermal
  throttling are plausible session contaminants.
- Deep bins are rare-event → Wilson bounds, not point estimates.
- QPU runs are operator-launched and never backgrounded.

## Open items for the operator

- **N** per instance (default ~30) and the **calibration/comparison split
  fraction** (default 20/80).
- **K orderings** + ordering-control **subsample size** (default 3 × 200).
- The **per-instance threshold rule** for archive instances (fixed edge-band
  ladder vs each instance's near-best) — chain instances use their on-chain
  threshold.
