# Changelog — RCS VRF Protocol Evolution

Each version captures a substantive design decision, not just a bug fix. The changelog walks through the protocol's evolution from initial AH-style implementation (v0) through the Liu-structure refactor (v0.4) and the calibration milestone (v0.4.1), explaining at each step *what changed and why*.

For the design rationale behind individual decisions, see `docs/design/`.

---

## v0.4.1 — Calibration milestone (math layer closed)

**Date:** Mid-June 2026
**Files:** `code/v0.4/leg5_qmin_liu.py`, `code/v0.4/test_qmin_calibrated.py`, `code/v0.4/verification/`
**Status:** Current. End-to-end verified against Liu et al.'s published experimental data.

### What changed

Three substantive changes, all in service of closing the math layer:

1. **Q_min algorithm reconciled with Liu's reference.** Replaced the numerical δ-grid optimisation (400 points) with the closed-form Chernoff δ from Liu's `src/entropy.py`. Changed the soundness-budget convention from a single combined threshold (`4 * eps_s = eps_sou`) to Liu's half-budget split (`eps_sou/2` on each tail separately).

2. **Adversary parameters identified from Liu's notebook.** The previous calibration attempts had used Table I's "Frontier sustained" value (`A = 0.897e18`) as the adversary compute. The actual parameters Liu uses in their reference notebook (`reproduce_figures/Table2-bounds-on-extractable-entropy.ipynb`) are `A = 4 × FRONTIER_theoretical = 8e18 FLOPS` with `eff = 0.5` — i.e., four supercomputers' worth of theoretical peak compute at 50% sustained efficiency, giving 4 exaFLOPS effective.

3. **End-to-end verification against experimental data.** Added `code/v0.4/verification/`, a self-contained verification package that loads Liu's published `aggregated_probs.npy` (1,522 precomputed validation probabilities from their Quantinuum H2 run) and reproduces the entire chain: F_XEB = 0.32, Q_min = 1297, certified entropy = 71,313.07 bits, extractable bits = 71,273.21 bits. All match to the digit Liu reports.

### Why

The previous v0.4 was structurally correct but unable to derive Q_min from first principles. The regression test had an explicit `xfail` marker stating "DO NOT fudge parameters to make this pass." After the Liu Zenodo notebook surfaced, the actual parameter conventions were identifiable and the calibration could be done principled rather than empirically.

### Verification

| Quantity | v0.4.1 result | Liu's reference | Match |
|---|---|---|---|
| F_XEB | 0.319725 | 0.32 | ✓ to printed precision |
| Q_min | 1297 | 1297 | ✓ exact |
| Certified entropy | 71,313.07 bits | 71,313.07 | ✓ exact |
| Extractable bits | 71,273.21 bits | 71,273.21 | ✓ exact |

### Files added / replaced

- **Added:** `code/v0.4/leg5_qmin_liu.py` (calibrated Q_min)
- **Added:** `code/v0.4/test_qmin_calibrated.py` (five passing tests, replaces former xfail)
- **Added:** `code/v0.4/verification/verify_against_liu_data.py` (end-to-end script)
- **Added:** `code/v0.4/verification/reference_data/aggregated_probs.npy` (Liu's data, 12 KB)
- **Added:** `code/v0.4/verification/README.md` and `reference_data/README.md` (provenance + usage)
- **Added:** `docs/math/ah_vs_liu_reference.pdf` (10-page side-by-side variance analysis explaining why v0.4's structural choice was correct)
- **Added:** `docs/meeting_prep/meeting_prep_mehdi.md` (Notion-style meeting prep doc)
- **Updated:** `docs/design/quip_design_notes.{md,pdf}` (now 19 pages, with new §4.8 on Q_min ↔ adversary power dependence)
- **Removed:** `code/v0.4/leg5_liu_certified.py` (superseded by `leg5_qmin_liu.py`)
- **Removed:** `code/v0.4/test_qmin_regression.py` (xfail retired; replaced by `test_qmin_calibrated.py`)

### Honest framings preserved

- The "global supercomputer aggregate" interpretation of the calibration constant — which was given as a post-hoc rationalisation of an earlier curve-fit attempt at `c_eff = 4.46` — was wrong as stated. The correct factorisation (per Liu's notebook) is `4 × FRONTIER_theoretical × 0.5 sustained efficiency`, not a single opaque constant. The product is the same (4 exaFLOPS effective), but the principled factorisation makes the two distinct modelling choices ("how many supercomputers" and "how efficiently used") explicit.
- The two-sided F_XEB check remains our addition, not Liu's. It's strictly more conservative than Liu's one-sided check.
- This milestone closes the *math* layer of the protocol. The remaining items (real QPU access, timing enforcement, Leg 1 implementation, choice of our own adversary model) are hardware, engineering, and product decisions — not math.

---

## v0.4 — Liu structure (the structural refactor)

**Date:** Late May 2026  
**Files:** `code/v0.4/`  
**Status:** Superseded by v0.4.1 (calibration milestone). The structural choice introduced here — many distinct circuits, one shot each — is unchanged; v0.4.1 added principled calibration on top.

### What changed

Switched the protocol structure from Aaronson-Hung (one circuit, many shots per round) to Liu et al. (many distinct circuits, one shot each per round). This is *the* substantive structural change in the entire history.

### Why

Both AH and Liu produce correct certified-entropy bounds, but they differ in how rounds relate statistically:

- **AH** has correlated shots within a round (all shots share one circuit). Handles correlation via the Entropy Accumulation Theorem (EAT). Bound is correct but looser.
- **Liu** has independent rounds (each round a fresh distinct circuit). Standard i.i.d. concentration applies. Tighter bound for the same parameters, simpler proof.

The switch made "rounds" — the unit the certified bound counts — *meaningful*. Under Liu, $Q_\text{min}$ is a number of distinct quantum circuits that had to be honest, not a number of shots.

### Verification

`leg5_liu_certified.py` *(renamed to `leg5_qmin_liu.py` in v0.4.1)* reproduces Liu et al.'s published numbers exactly:
- $H_\text{min}^\text{cert} = 71{,}313$ bits (Theorem 1) ✓
- $\ell = 71{,}273$ extracted bits (Corollary 7) ✓
- Per-round collision entropy $H_2 = n - 1 = 55$ bits ✓

`test_qmin_regression.py` *(retired in v0.4.1; replaced by `test_qmin_calibrated.py`)* codifies these reproductions with explicit honesty about what's still owed (see below).

### Known issues introduced/persisting (at v0.4 release; resolved status as of v0.4.1)

- **$Q_\text{min}$ derivation is uncalibrated.** Our `compute_q_min` returns ~1700–3700 vs. the paper's 1297, depending on which $A$ (compute power) is plugged in. The structural formula is correct; three numerical conventions ($c_\text{eff}$, $\delta$-grid, $\varepsilon_2$ tail) need reconciliation against the authors' Zenodo code (DOI 10.5281/zenodo.12952178). The regression test marks this as `xfail` — intentionally not fudged to pass. **[RESOLVED in v0.4.1 — see entry above.]**
- **Timing enforcement remains stubbed.** Same as v0.3a; deferred to v0.5+. *[Still open as of v0.4.1.]*

### Acknowledged simplification

Edge-coloured random graph for two-qubit gate scheduling (Liu's approach for matching Quantinuum H2's hardware connectivity) is approximated here by a simpler random perfect matching. Fine at $n = 8$; would need to be upgraded for production scale.

---

## v0.3a (revision) — Certified min-entropy + Toeplitz/LHL extraction

**Date:** Mid May 2026  
**Files:** `code/v0.3a/`  
**Status:** Superseded by v0.4 structurally, but the certified-entropy + extraction math was carried over intact.

### What changed

Added the **certified min-entropy formula** (Liu Theorem 1) and the **Toeplitz extractor with Leftover Hash Lemma bound** (Liu Corollary 7) on top of the v0.2 audit-log infrastructure:

- $H_\text{min}^\text{cert} = Q_\text{min}(n-1) - \log_2(1/\varepsilon_s)$
- $\ell \leq Q_\text{min}(n-1) - 3\log_2(1/\varepsilon_\text{sou}) - 2$

Implementation included a working Toeplitz extractor with fresh drand-pulse-derived seed.

### Why

Up through v0.2, the protocol had verification (F_XEB + timing) but no entropy quantification. v0.3a added the formulas that convert "passed verification" into "$X$ bits of certified entropy" and then into "$\ell$ bits of clean extracted output." This is the math chain Liu et al. specifies.

The `_rev` revision was a structural cleanup of v0.3a — same formulas, cleaner code organization — and is what v0.4 inherited.

### Verification

Implementation reproduces 71,313 / 71,273 exactly given Liu's $Q_\text{min}$. The per-round entropy ($n - 1$) is derived from Porter-Thomas's second moment from first principles, not transcribed.

### Limitations at this stage

- Still using AH structure (one circuit, many shots). The amortization-prone setup that v0.4 fixes.
- Q_min derivation was not yet attempted; the formula used the paper's published value directly.
- Timing still stubbed.

---

## v0.2 — Real Ed25519 VRF + audit log

**Date:** Mid May 2026  
**Files:** `code/v0.2/`  
**Status:** Superseded by ongoing Leg 1 redesign (commit-reveal + beacons). The audit-log infrastructure carries forward.

### What changed

Replaced the stub VRF (a placeholder hash function in v0–v0.1) with a real Ed25519 EC-VRF implementation conforming to RFC 9381. Added a tamper-evident audit log that hash-chains every round's events.

### Why

The protocol's claim to be "publicly verifiable" requires a cryptographically real VRF plus an audit log that anyone can replay. v0.2 made both real:
- VRF: full EC-VRF Prove/Verify with curve operations, Schnorr-style proof.
- Audit log: JSON with hash-chained entries; `verify_audit_log.py` replays an audit log to confirm the round was honest end-to-end.

### Subsequently questioned (but not undone here)

This was the right move *given the textbook approach*. On closer reading, EC-VRF carries three problems for the RCS VRFsetting:

1. Not quantum-secure (Shor breaks elliptic-curve discrete log).
2. Structurally single-keyholder (forces centralised trust or full validator-network).
3. MEV exposure in request-then-fulfill workflow.

These motivated the Leg 1 redesign currently in progress (see `docs/design/leg1_redesign_story.pdf`). v0.2's EC-VRF will be replaced in v0.5 by hash-based commit-reveal + external beacons.

### What carries forward

The audit-log design is structurally sound and carries forward to v0.3a, v0.4, and beyond. The hash-chained ledger is the foundation of public verifiability regardless of which seed-generation primitive is used.

---

## v0.1 — Two-sided F_XEB check

**Date:** Mid May 2026  
**Files:** `code/v0.1/`  
**Status:** The two-sided check carries forward to all later versions.

### What changed

Replaced the one-sided F_XEB acceptance check ($F_\text{XEB} \geq \chi$) with a two-sided check ($\chi_\text{low} \leq F_\text{XEB} \leq \chi_\text{high}$). Added a `spoofing_experiment` that demonstrates the upper bound catches heavy-output-generation attacks that one-sided thresholds miss.

### Why

The original AH/Liu acceptance is one-sided — the protocol cares only that the score is high enough to confirm quantum behaviour. We added an upper bound as a *defence-in-depth* measure against an adversary class outside the standard model: one that scores anomalously high (e.g., pure cherry-picking of high-probability strings). The two-sided check is strictly more conservative than the paper requires.

### Honest framing

This is *our addition*, not Liu's prescription. The lower bound is load-bearing in the security argument; the upper bound is our conservative belt-and-suspenders. Documented explicitly in `docs/presentations/leg5_verification_design.pdf` slide 14.

---

## v0 — Initial RCS+VRF implementation (Aaronson-Hung structure)

**Date:** Early-mid May 2026  
**Files:** `code/v0/`  
**Status:** Historical. Superseded structurally by Liu (v0.4).

### What it contains

The first working implementation of the seven-leg protocol:
- Leg 1: stub VRF (placeholder hash, made real in v0.2)
- Leg 2: parameter precommitment (basic)
- Leg 3: circuit generation from seed via SHAKE-256
- Leg 4: simulated quantum sampling (cirq, fidelity model)
- Leg 5: F_XEB verification (one-sided, made two-sided in v0.1)
- Leg 6: bit extraction (stub, made real in v0.3a with Toeplitz/LHL)
- Leg 7: anchoring (basic, made real in v0.2 with hash-chained audit log)

Used the **Aaronson-Hung structure** — one circuit per round, many shots. This was the published certification protocol at the time of initial implementation.

### Supplementary files

- `spoofing_experiment.py`: demonstrates how a classical-sim spoofer's F_XEB compares to honest sampling.
- `depth_sweep.py`: explores how F_XEB varies with circuit depth (validates the 2-design approximation kicks in around $d \approx n$).

### Why it's preserved

Useful historical reference for:
- What the original AH-style implementation looked like (the protocol we built first).
- The seven-leg structure introduced here carries through all subsequent versions.
- The spoofing experiment is pedagogical: shows why F_XEB > $\chi$ is non-trivial.

---

## Cross-cutting changes by topic

### Leg 5 verification chain

| Version | Status |
|---|---|
| v0 | F_XEB only, one-sided |
| v0.1 | Two-sided F_XEB |
| v0.2 | + audit log integration |
| v0.3a | + certified min-entropy formula (Liu Theorem 1) |
| v0.4 | + Liu structure (many circuits, one shot each); reproduces 71,313 given paper's Q_min |
| v0.4.1 | + principled Q_min calibration; end-to-end verified against Liu's experimental data; math layer closed |

### Leg 6 extraction

| Version | Status |
|---|---|
| v0 - v0.2 | Stub (placeholder bit extraction) |
| v0.3a | Real Toeplitz extractor with LHL bound (Liu Corollary 7) |
| v0.4 | Inherited from v0.3a; reproduces 71,273 exactly |

### Leg 1 seed generation

| Version | Status |
|---|---|
| v0 - v0.1 | Stub VRF (placeholder) |
| v0.2 - v0.4 | Real Ed25519 EC-VRF (RFC 9381) |
| v0.5 (next) | Replacing EC-VRF with hash-based commit-reveal + external beacons + PoW puzzle + chained transcript |

### Audit log

| Version | Status |
|---|---|
| v0 - v0.1 | Minimal (in-memory) |
| v0.2 | Hash-chained JSON with replay verifier |
| v0.3a - v0.4 | Inherited from v0.2 |

### Tests

| Version | Status |
|---|---|
| v0 - v0.1 | Spoofing-experiment scripts (not formal tests) |
| v0.2 | First formal test suite (`test_v0_2.py`) |
| v0.3a | Extended test suite covering entropy chain |
| v0.4 | Three test files: pipeline tests, Leg 5 entropy tests, $Q_\text{min}$ regression tests (with explicit `xfail` marker) |
| v0.4.1 | xfail retired (`test_qmin_regression.py` removed). New `test_qmin_calibrated.py` with five passing tests covering exact-reproduction, protocol-breakdown, and adversary-power monotonicity. End-to-end script `verify_against_liu_data.py` against Liu's experimental data. |

---

## Looking forward

### v0.5 (next, in progress)

Planned changes:
- Replace EC-VRF with hash-based commit-reveal + external beacons (drand + NIST) + PoW puzzle + chained transcript per Leg 1 redesign.
- Pin the hash primitive (SHA-3-256 vs SHAKE-256 — currently open). First make sure if it is required at all.
- Resolve timing enforcement design (hash-graph propagation vs drand pulse ordering vs cryptographic timestamping).
- Pin down commit and reveal ordering rules (avoid last-revealer bias and adaptive-commit attacks).


### Beyond v0.5

- Real QPU integration (Quantinuum H2 or IonQ Forte/Tempo via cloud).
- Production-scale parameters ($n = 56$ vs current $n = 8$).
- Hash-graph multi-party anchoring for Leg 7.
- NIST STS uniformity sanity-test on extractor output.
