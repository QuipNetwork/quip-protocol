# RCS VRF Protocol — Design History

This is the audit-trail repository for the **RCS VRF** protocol — a subnet that combines quantum random circuit sampling with classical verification to produce *certified randomness*. The repo documents the protocol's evolution through five iterations (v0 → v0.4), with the code, tests, design artifacts, and supporting math at each stage.

It is **internal-only** and intended for the project team (and selectively-invited reviewers like collaborators or external auditors). It is not yet a polished public release; some pieces are intentionally still in flight (see [Known limitations](#known-limitations) below).

## What you'll find here

```
code/                         The implementation, one directory per version
  v0/                         Initial AH-style protocol
  v0.1/                       + two-sided F_XEB check
  v0.2/                       + real Ed25519 VRF + audit log
  v0.3a/                      + certified min-entropy + Toeplitz/LHL extraction
  v0.4/                       Liu-structure refactor (current working version)
    verification/             End-to-end verification against Liu's experimental data (v0.4.1)
      reference_data/         Liu's published aggregated_probs.npy

docs/                         All documentation
  design/                     Story-paced design walkthroughs (PDF + LaTeX source)
  math/                       Mathematical references and proofs
  presentations/              Slide decks and speaker notes
  architecture/               Pipeline diagrams
 

CHANGELOG.md                  Version-by-version evolution narrative
LICENSE                       Apache 2.0
README.md                     This file
```

## Where to start (for different readers)

- **Strategic / non-technical reviewer**: open `docs/design/quip_design_notes.pdf` — the 19-page consolidated overview of every substantive design decision.
- **Mathematical reviewer**: open `docs/math/porter_thomas_report.pdf` for the certification chain's math derivation, then `docs/design/certified_randomness_story.pdf` for the story-paced version.
- **Protocol designer / reviewer of Leg 1 redesign**: open `docs/design/leg1_redesign_story.pdf` and `docs/presentations/leg5_verification_design.pdf`.
- **Code reviewer**: start at `code/v0.4/`, the current working version, then walk backward through `CHANGELOG.md` to see how it got there.
- **Auditor**: read `CHANGELOG.md` end-to-end, then `code/v0.4/test_qmin_calibrated.py` (the five regression tests covering the calibrated math layer), and run `code/v0.4/verification/verify_against_liu_data.py` to reproduce Liu's published numbers end-to-end.

## What this codebase is and isn't

**Is:**
- A research/development codebase for the RCS VRFprotocol.
- A faithful implementation of Liu et al.'s certification chain (Theorem 1, Lemma 2, Corollary 7), reproducing the paper's headline numbers (71,313 certified bits, 71,273 extracted bits) exactly given the paper's stated $Q_\text{min}$.
- A working pipeline at toy scale ($n = 8$ qubits) suitable for design validation and as a reference for production scaling.

**Is not:**
- Production-ready. Several engineering items are still owed (see below).
- Connected to real quantum hardware. The current implementation simulates the QPU using `cirq` with a fidelity model.
- A standalone product. Some legs (timing enforcement, real QPU integration, customer interface) are deliberately stubbed pending design completion.

## Known limitations

This repo is honest about what is and isn't yet finished. The reviewer should be able to see these from the code, but here they are explicitly:

1. **$Q_\text{min}$ derivation is calibrated and end-to-end verified.** *(Was: "uncalibrated"; closed in v0.4.1.)* The `compute_q_min` function in `code/v0.4/leg5_qmin_liu.py` exactly reproduces Liu's reference notebook output (Q_min = 1297, certified entropy = 71,313.07 bits, extractable = 71,273.21 bits) using their published parameters (`eff = 0.5`, `A = 4 × FRONTIER_theoretical`, etc.). End-to-end verification against Liu's experimental data is in `code/v0.4/verification/` — run `python verify_against_liu_data.py` to reproduce F_XEB = 0.32 and the full chain on Liu's published `aggregated_probs.npy`. The math layer is settled.

2. **Timing enforcement is stubbed.** The protocol records sample timestamps in the audit log but does not currently enforce the per-circuit response deadline. The math chain assumes this deadline holds (it's load-bearing in Liu's security argument). Closing this is bounded engineering, design pending — see `docs/design/quip_design_notes.pdf` Section 10.4.

3. **Leg 1 (seed generation) is mid-redesign.** The v0.2-and-later versions use a real Ed25519 EC-VRF. We're replacing this with hash-based commit-reveal + external beacons (drand + NIST) + a PoW puzzle + chained transcript. The redesign is documented in `docs/design/leg1_redesign_story.pdf`; the implementation lands in the next version (v0.5).

4. **Two-sided F_XEB check is our addition, not Liu's.** Liu uses a one-sided acceptance check ($F_\text{XEB} \geq \chi$); our v0.1+ uses a two-sided check ($\chi_\text{low} \leq F_\text{XEB} \leq \chi_\text{high}$) as a defence-in-depth choice. This is explicitly flagged in the code and the presentation deck.

5. **Edge-coloured graph simplification.** Liu et al. uses a sophisticated edge-coloured random graph for two-qubit gate scheduling. Our toy implementation uses a simpler random perfect matching. Functionally equivalent at toy scale; would need to be upgraded for real-hardware connectivity.

6. **Toy scale only ($n = 8$).** Production scale would be $n = 56$ matching Liu's Quantinuum H2 setup. Toy scale is fine for design validation; the math chain extrapolates.

## How to run

Each version has its own implementation in `code/v$VERSION/`. The current working version is `v0.4` (with v0.4.1 calibration applied in place). From there:

```bash
cd code/v0.4
python -m pytest test_rcs_vrf.py test_leg5.py test_qmin_calibrated.py
```

All tests should pass cleanly. The earlier `test_qmin_regression.py` with the intentional `xfail` was retired in v0.4.1 once calibration closed; see `code/v0.4/NOTES.md` for the migration notes.

To reproduce Liu et al.'s published results end-to-end on their experimental data:

```bash
cd code/v0.4/verification
python verify_against_liu_data.py
```

Expected output: six pass/fail checks, all green, exit code 0.

Dependencies (rough; varies by version):
- Python 3.9+
- `numpy`, `scipy`, `cirq` (quantum simulation)
- `cryptography` (Ed25519 in v0.2+)
- `pytest` (test runner)

## Conventions and notation

- **Per-round collision entropy** is exactly $n - 1$, derived from Porter-Thomas's second moment ($2/N^2$ where $N = 2^n$). See `docs/math/porter_thomas_report.pdf` for the full derivation.
- **Certified min-entropy formula**: $H_\text{min}^\text{cert} = Q_\text{min}(n-1) - \log_2(1/\varepsilon_s)$ (Liu Theorem 1).
- **Toeplitz/LHL extraction bound**: $\ell \leq Q_\text{min}(n-1) - 3\log_2(1/\varepsilon_\text{sou}) - 2$ (Liu Corollary 7).
- **Paper parameters**: $n = 56$, $M = 30{,}010$, $\chi = 0.3$, $\varepsilon_s = 2.5 \times 10^{-7}$, $\varepsilon_\text{sou} = 10^{-6}$, $A = 3.6 \times 10^{18}$ FLOPS, $t_\text{threshold} \approx 2.2$ s.

## References

- Liu et al. (2025), *Certified randomness using a trapped-ion quantum processor*, Nature. The protocol we implement.
- Aaronson & Hung (STOC 2023), *Certified Randomness from Quantum Supremacy*. The predecessor structure (v0–v0.3a implemented this).
- Boixo et al. (Nature Physics 2018), *Characterizing quantum supremacy in near-term devices*. F_XEB foundations.
- Dupuis, Fawzi & Renner (Communications in Mathematical Physics 2020), *Entropy Accumulation*. The EAT machinery AH relied on.

## License

Apache 2.0. See `LICENSE`.

## Status

Active development. Last substantive update: 2026. Critical-path items (real QPU access, timing enforcement, Leg 1 implementation, Q_min calibration) tracked in `docs/design/quip_design_notes.pdf` Section 10.
