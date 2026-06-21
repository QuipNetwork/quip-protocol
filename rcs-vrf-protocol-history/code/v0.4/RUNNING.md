# Running RCS-VRF v0.4.1

Quick instructions for running the v0.4.1 release — the math-layer-closed
milestone that reproduces Liu et al.'s published numbers end-to-end.

For full project context, see `README.md`.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install cirq numpy scipy cryptography requests
```

Python 3.10+ recommended.

---


## Three Things to Run

### 1. End-to-end verification against Liu — *the headline demo*

```bash
python verify_against_liu_data.py
```

**What it does:** loads Liu's published probabilities from
`aggregated_probs.npy`, runs them through the full math chain
(F_XEB → Q_min → certified min-entropy → Toeplitz extraction), and prints
the reproduced values against Liu's published numbers.

**Runtime:** ~1 second.

### 2. Toy-scale end-to-end protocol run

```bash
python rcs_vrf_v0_4.py
```

**What it does:** runs the full 7-leg pipeline at toy scale (n = 8 qubits).
Generates a seed, expands via the Leg 2 PRF, constructs Liu-style
multi-circuit batches, samples via cirq's noisy simulator, runs F_XEB /
Q_min / certified-min-entropy, extracts via Toeplitz/LHL, and writes a
hash-chained audit log.

**Expected output:** a sequence of per-leg log lines ending with a
summary of the certified bits produced.

**Runtime:** ~10-30 seconds (most time is cirq simulation).

**Caveat — toy numbers won't match Liu's.** F_XEB will differ from 0.32,
Q_min will differ from 1297. The toy is for confirming the pipeline runs
end-to-end with sensible numbers at small scale, not for matching Liu's
experimental values.

### 3. Test suites

```bash
python test_qmin_calibrated.py
python test_fxeb_limiting_cases.py
```

Or with pytest:

```bash
pytest test_qmin_calibrated.py test_fxeb_limiting_cases.py
```

**`test_qmin_calibrated.py`:** 5 regression tests confirming Q_min
reproduces Liu (1297), behaves monotonically with adversary strength, and
collapses gracefully (Q_min → 0) at sufficient adversary compute.

**`test_fxeb_limiting_cases.py`:** 7 sanity tests on F_XEB across the full
input range (perfect quantum φ = 1, pure noise φ = 0, fixed bitstring,
argmax adversary, single-shot, LLN scaling).

**Expected output:** all 12 tests pass.



## What v0.4.1 Establishes

If all three runs succeed:

- The math layer is closed (Liu's published numbers reproduce exactly).
- The math implementation is regression-safe (tests pass).
- The end-to-end pipeline runs at toy scale.

This is the "math layer closed" claim made concrete.

---

## Honest Hedges

A few things worth knowing before drawing conclusions from the runs:

**1. The verification reproduces Liu's numbers using Liu's data as input.**
The harness loads Liu's published `aggregated_probs.npy` (their measured
probabilities) and runs them through our math chain. This verifies our
math is correctly implemented. It does *not* verify that our toy circuits
would produce probabilities like Liu's if run on real hardware at
production scale.

**2. Toy-scale construction uses random SU(2) + simple matchings.** Depth
(10 layers) matches Liu's; gate set and matching strategy differ from
Liu's production recipe.

**3. The audit log is local and not externally anchored.** Hash-chained
JSON files demonstrate the audit-log mechanism but are not yet anchored
to a tamper-evident external structure.

**4. Timing enforcement is not yet implemented.** The protocol records
timestamps but does not cryptographically bind them to an external time
source.

**5. Real QPU integration is not yet implemented.** Leg 4 uses cirq's
noisy simulator, not real quantum hardware.

These are open items, not bugs. Each one is on the roadmap for v0.5 and
later versions.

---

## Reference

Liu, M.-Z. et al. *Certified randomness using a trapped-ion quantum
processor.* Nature 640:343–348 (2025). arXiv:2503.20498.
