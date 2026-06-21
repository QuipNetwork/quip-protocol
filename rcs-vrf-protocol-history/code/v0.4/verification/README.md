# Verification: End-to-End Against Liu et al. Experimental Data

This directory contains the verification of the Quip certification chain against the published experimental data from Liu et al.'s Nature 2025 paper.

## What this verifies

Loading Liu's 1,522 precomputed validation probabilities and running them through our complete chain produces:

```
F_XEB                = 0.319725  (just above χ = 0.3)
Audit accepted       = True
Q_min                = 1297              (matches paper)
Certified entropy    = 71313.068431      (matches paper)
Extractable bits     = 71273.205294      (matches paper)
```

All values match Liu et al.'s published numbers to the digit.

## What this validates and what it doesn't

**Validates:**
- Our F_XEB formula implementation is bit-for-bit identical to Liu's
- The audit-acceptance logic is correct
- The full chain (F_XEB → Q_min → H_certified → extractable bits) operates correctly on real experimental data

**Does not validate:**
- The hardware integrity of Liu's QPU (we trust their attestation)
- The probability-computation step (Liu did this on Frontier; we use their output)
- Bitstring-handling code (we never touch raw bitstrings here)
- Properties of our protocol beyond Liu's framework (timing, Leg 1, real QPU, etc.)

For the broader scope of validation status across the protocol, see the top-level `README.md` of the audit-trail repo.

## Contents

```
.
├── README.md                       This file
├── verify_against_liu_data.py      Standalone verification script
├── leg5_qmin_liu.py                The verified-calibrated chain
└── reference_data/
    ├── README.md                   Data provenance and licensing
    └── aggregated_probs.npy        Liu's 1,522 precomputed probabilities
```

## How to run

From this directory:

```bash
python verify_against_liu_data.py
```

The script prints a step-by-step pass/fail report and exits with code 0 if all checks pass, 1 otherwise. Expected runtime: < 1 second on any modern machine.

## Dependencies

- Python 3.9+
- numpy
- scipy (for `leg5_qmin_liu.compute_q_min`)

## Provenance

The reference data file (`reference_data/aggregated_probs.npy`) is taken directly from Liu et al.'s Zenodo archive:

> [zenodo.org/records/15192591](https://zenodo.org/records/15192591), file `reproduce_verification/data/aggregated_probs.npy`

See `reference_data/README.md` for full citation, provenance, and licensing details.

## Verification milestone

This verification closes the math-layer validation work for Quip v0.4. Every numerical claim in the protocol's certification math is now backed by either:

- Liu's reference algorithm + parameters (algorithm verification, in `leg5_qmin_liu.py`)
- Liu's experimental data + our chain (data verification, in this directory)

The remaining open items in the protocol (real QPU integration, timing enforcement, Leg 1 redesign implementation, choice of adversary model) are not math items.
