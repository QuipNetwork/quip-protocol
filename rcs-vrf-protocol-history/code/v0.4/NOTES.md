# v0.4 — Liu structure refactor (current working version, calibrated as of v0.4.1)

**The substantive structural change in the protocol's history.** Switched from Aaronson-Hung (one circuit, many shots) to Liu et al. (many distinct circuits, one shot each).

> **v0.4.1 update (June 2026):** Q_min calibration completed against Liu's reference notebook; end-to-end verification against Liu's experimental data added in the `verification/` subdirectory. See CHANGELOG.md for the v0.4 → v0.4.1 narrative.

## Files in this version

- `rcs_vrf.py` — main protocol with Liu structure
- `leg5_qmin_liu.py` — **calibrated** Q_min implementation, exactly reproduces Liu's notebook (eff=0.5, A=4×Frontier_theoretical)
- `leg5_certified_entropy.py` — entropy chain implementation (collision entropy, certified bits, Toeplitz extraction)
- `test_rcs_vrf.py` — pipeline tests
- `test_leg5.py` — Leg 5 entropy/extraction tests
- `test_qmin_calibrated.py` — five-test regression suite for the calibrated Q_min (replaces former xfail'd `test_qmin_regression.py`)
- `visualize_circuit.py` — utility for visualizing the random circuits
- `verification/` — end-to-end verification package against Liu's published experimental data

## What was new here (v0.4)

The structural change:
- **Before (AH):** one circuit per round, m shots. Shots correlated through shared circuit. Bound handled via EAT (Entropy Accumulation Theorem). Correct but loose.
- **After (Liu):** M distinct circuits per round, 1 shot each. Rounds independent. Standard i.i.d. concentration applies. Cleaner argument, tighter bound.

Result: "rounds" — the unit the certified bound counts — became meaningful. Q_min is now a count of distinct quantum circuits, not shots.

## What was new in v0.4.1

Calibration and end-to-end verification:

- **Reconciled Q_min algorithm with Liu's reference code** (`src/entropy.py` from Zenodo). Replaced numerical δ-grid with closed-form Chernoff δ; changed threshold convention from `4*eps_s` to `eps_sou/2`.
- **Identified Liu's adversary parameters** from their notebook (`reproduce_figures/Table2-bounds-on-extractable-entropy.ipynb`): `eff = 0.5`, `A = 4 × FRONTIER_theoretical = 8e18 FLOPS`. Effective adversary compute is 4 exaFLOPS — roughly the aggregate of the world's top four supercomputers.
- **Exact reproduction**: Q_min = 1297, certified entropy = 71,313.068431 bits, extractable = 71,273.205294 bits — all match Liu's notebook to six decimal places.
- **End-to-end verification against experimental data**: `verification/verify_against_liu_data.py` loads Liu's published `aggregated_probs.npy` (1,522 precomputed validation probabilities) and reproduces F_XEB = 0.32 along with the full certification chain.

## Verification status

```
F_XEB                = 0.319725  (Liu's published: 0.32)        ✓
Q_min                = 1297      (Liu's notebook: 1297)         ✓ exact
Certified entropy    = 71313.07  (Liu's notebook: 71313.07)     ✓ exact
Extractable bits     = 71273.21  (Liu's notebook: 71273.21)     ✓ exact
```

All match to the digit Liu reports.

## Acknowledged simplification

Liu uses an edge-coloured random graph for two-qubit gate scheduling to match Quantinuum H2's all-to-all connectivity. Our toy implementation (n=8) uses a simpler random perfect matching per layer. Functionally equivalent at toy scale; would need to be upgraded for real-hardware connectivity matching.

## What's still owed (post-v0.4.1)

The math layer is closed. Remaining items are not math:

1. **Timing enforcement** — currently records timestamps but doesn't enforce a deadline. Gated on Mehdi's hash-graph timing design.
2. **Real QPU integration** — currently uses `cirq` simulation with a fidelity model. Gated on Colton-led vendor conversations.
3. **Leg 1 redesign implementation** — hash-based commit-reveal + beacons + puzzle + chained transcript. Design documented; implementation planned for v0.5.
4. **Choice of adversary model for our deployment** — currently inherits Liu's `4 × Frontier × 50%`. See design notes §4.8 for the decision space.

## How to run

```bash
cd code/v0.4

# Pipeline tests + entropy chain tests + Q_min calibration tests
python -m pytest test_rcs_vrf.py test_leg5.py test_qmin_calibrated.py -v

# End-to-end verification against Liu's experimental data
cd verification
python verify_against_liu_data.py
```

Expected: all tests pass cleanly. The verification script outputs six pass/fail checks and exits with code 0.

## Migration from v0.4 → v0.4.1

If you have code that imports from v0.4's old layout:

| Old (v0.4) | New (v0.4.1) |
|---|---|
| `from leg5_liu_certified import compute_q_min` | `from leg5_qmin_liu import compute_q_min` |
| `test_qmin_regression.py` (xfail'd test) | `test_qmin_calibrated.py` (five passing tests) |

The function signature for `compute_q_min` has also changed to match Liu's reference exactly: parameters are now `(epsilon_sou, chi, m, M, t_tot, eff, A, B_val)` instead of the older mixed-naming convention.
