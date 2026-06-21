# v0.3a (revision) — Certified min-entropy + Toeplitz/LHL extraction

**Two substantive additions vs v0.2:**
1. Certified min-entropy formula (Liu Theorem 1).
2. Real Toeplitz extractor with LHL bound (Liu Corollary 7).

## Files in this version

- `rcs_vrf.py` — protocol with full certified-entropy chain (revision of original v0.3a)
- `test_rcs_vrf.py` — test suite extended to cover entropy chain
- `audit_log_example.json` — sample audit log including the extraction stage

## What was new here

- **Certified min-entropy formula**: `H_min_cert = Q_min × (n-1) - log₂(1/ε_s)`
- **Per-round collision entropy**: `H₂ = n-1`, derived from Porter-Thomas's second moment
- **Toeplitz extractor**: deterministic linear map over GF(2), seeded with a fresh drand pulse
- **LHL bound**: `ℓ ≤ Q_min × (n-1) - 3·log₂(1/ε_sou) - 2` for the extractable bit count

## Verification

Reproduces Liu et al.'s headline numbers exactly given the paper's Q_min:
- 71,313 certified bits (Theorem 1) ✓
- 71,273 extracted bits (Corollary 7) ✓

## Limitations at this stage

- Still using Aaronson-Hung structure (one circuit, many shots) — the amortization-prone setup. The structural switch to Liu happens in v0.4.
- Q_min derivation was not yet attempted; uses the paper's published value directly.
- Timing enforcement still stubbed.

## About the `_rev` revision

The original v0.3a was structurally tangled in places. This revision cleaned up the organization without changing the math. v0.4 inherited this cleaned-up version.
