#!/usr/bin/env python3
"""
verify_against_liu_data.py
==========================

End-to-end verification of the Quip certification chain against Liu et al.'s
experimental data, as published in their Nature 2025 paper.

This script loads Liu's precomputed validation probabilities (the 1,522
aggregated probabilities from their Quantinuum H2 experimental run),
feeds them through our verification chain (F_XEB → Q_min → H_certified
→ extractable bits), and confirms each step reproduces Liu's published
values exactly.

The verification covers:
  (1) F_XEB formula implementation
  (2) Audit acceptance check (F_XEB >= chi)
  (3) Q_min computation with Liu's adversary model
  (4) Certified min-entropy (Liu Theorem 1)
  (5) Extractable bits (Liu Corollary 7, Toeplitz/LHL)

Usage:
    python verify_against_liu_data.py

Exit code:
    0 if all verifications pass
    1 if any verification fails

Reference:
    Liu et al., "Certified randomness using a trapped-ion quantum processor,"
    Nature 640:343 (2025); arXiv:2503.20498.
    Zenodo archive: zenodo.org/records/15192591
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import leg5_qmin_liu.
# This makes the script runnable from anywhere, not just the parent dir.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# Import the verified-calibrated chain
from leg5_qmin_liu import (
    compute_q_min,
    certified_min_entropy,
    extractable_bits,
    PAPER,
)


# Liu's published reference values (from the paper and their notebook)
REFERENCE_VALUES = dict(
    n_validation_circuits=1522,
    fxeb_min=0.30,           # must exceed chi = 0.3 for audit to pass
    fxeb_published=0.3197,   # value computed from Liu's aggregated_probs.npy
    fxeb_tol=1e-3,           # tolerance when comparing against published
    q_min=1297,
    h_certified=71313.068431,
    extractable_bits=71273.205294,
    tolerance=1e-3,
)


def banner(text: str, char: str = "=", width: int = 70) -> None:
    """Print a banner header."""
    print(char * width)
    print(text)
    print(char * width)


def check(label: str, computed, expected, tol: float = 1e-3) -> bool:
    """Compare a computed value against expected; print pass/fail line."""
    if isinstance(expected, int) and isinstance(computed, int):
        match = (computed == expected)
        diff_str = f"diff: {computed - expected:+d}"
    else:
        match = abs(computed - expected) < tol
        diff_str = f"diff: {computed - expected:+.6f}"
    status = "PASS" if match else "FAIL"
    marker = "✓" if match else "✗"
    print(f"  {marker} {label:35s} {status:4s}  ({diff_str})")
    return match


def verify(data_path: Path) -> bool:
    """
    Run the full end-to-end verification.

    Returns True if all checks pass, False otherwise.
    """
    n = PAPER['n']
    N = 2 ** n
    chi = PAPER['chi']
    M = PAPER['M']
    m_expected = PAPER['m']

    banner("Quip Protocol — End-to-End Verification on Liu et al. Data")
    print()
    print(f"Reference: Liu et al., Nature 640:343 (2025); arXiv:2503.20498")
    print(f"Data source: zenodo.org/records/15192591")
    print(f"Reference data file: {data_path}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load Liu's experimental data
    # ------------------------------------------------------------------
    banner("Step 1: Load Liu's experimental probabilities", char="-")
    if not data_path.exists():
        print(f"  ✗ ERROR: reference data not found at {data_path}")
        return False
    prob_s = np.load(data_path)
    print(f"  shape:    {prob_s.shape}")
    print(f"  dtype:    {prob_s.dtype}")
    print(f"  min:      {prob_s.min():.6e}")
    print(f"  max:      {prob_s.max():.6e}")
    print(f"  mean:     {prob_s.mean():.6e}")
    print(f"  1/N:      {1.0/N:.6e}  (size-unbiased Porter-Thomas mean)")
    print(f"  2/N:      {2.0/N:.6e}  (perfect quantum sampling target)")
    print()
    all_pass = True
    all_pass &= check(
        "validation circuit count",
        len(prob_s),
        REFERENCE_VALUES['n_validation_circuits'],
    )
    print()

    # ------------------------------------------------------------------
    # Step 2: Compute F_XEB on Liu's data
    # ------------------------------------------------------------------
    banner("Step 2: Compute F_XEB", char="-")
    fxeb = N * np.mean(prob_s) - 1.0
    print(f"  F_XEB computed:   {fxeb:.6f}")
    print(f"  F_XEB threshold:  {chi}")
    print()
    all_pass &= check(
        "F_XEB matches Liu's value",
        fxeb,
        REFERENCE_VALUES['fxeb_published'],
        tol=REFERENCE_VALUES['fxeb_tol'],
    )
    accepted = fxeb >= chi
    print(f"  {'✓' if accepted else '✗'} F_XEB >= chi (audit acceptance): "
          f"{'PASS' if accepted else 'FAIL'}  "
          f"(margin: {fxeb - chi:+.6f})")
    all_pass &= accepted
    print()

    # ------------------------------------------------------------------
    # Step 3: Run certified-entropy chain
    # ------------------------------------------------------------------
    banner("Step 3: Certified-entropy chain (Liu Theorem 1, Corollary 7)", char="-")
    Q = compute_q_min(
        epsilon_sou=PAPER['epsilon_sou'],
        chi=PAPER['chi'],
        m=PAPER['m'],
        M=PAPER['M'],
        t_tot=PAPER['M'] * PAPER['t_threshold_per_circuit'],
        eff=PAPER['eff'],
        A=PAPER['A'],
        B_val=PAPER['B_val_per_circuit'],
    )
    H = certified_min_entropy(Q, n, PAPER['epsilon_sou'])
    ell = extractable_bits(Q, n, PAPER['epsilon_sou'])
    print(f"  Q_min computed:        {Q}")
    print(f"  H_certified computed:  {H:.6f}")
    print(f"  Extractable computed:  {ell:.6f}")
    print()
    all_pass &= check(
        "Q_min matches paper",
        Q,
        REFERENCE_VALUES['q_min'],
    )
    all_pass &= check(
        "Certified entropy matches",
        H,
        REFERENCE_VALUES['h_certified'],
        tol=REFERENCE_VALUES['tolerance'],
    )
    all_pass &= check(
        "Extractable bits matches",
        ell,
        REFERENCE_VALUES['extractable_bits'],
        tol=REFERENCE_VALUES['tolerance'],
    )
    print()

    # ------------------------------------------------------------------
    # Step 4: Adversary model (informational)
    # ------------------------------------------------------------------
    banner("Step 4: Adversary model in use (informational)", char="-")
    eff_a = PAPER['eff'] * PAPER['A']
    n_frontier = PAPER['A'] / 2.0e18
    print(f"  A (raw):              {PAPER['A']:.2e} FLOPS")
    print(f"                        ({n_frontier:.0f}x Frontier theoretical peak)")
    print(f"  eff (efficiency):     {PAPER['eff']}")
    print(f"  eff*A (effective):    {eff_a:.2e} FLOPS")
    print(f"                        ({eff_a/1e18:.1f} exaFLOPS effective)")
    print(f"  B (per circuit):      {PAPER['B_val_per_circuit']:.2e} FLOPs")
    print(f"  epsilon_sou:          {PAPER['epsilon_sou']:.0e}")
    print()
    print(f"  This is Liu et al.'s adversary model. Substitute different values")
    print(f"  for our own deployment if/when we choose to.")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    banner("Verification Summary", char="=")
    if all_pass:
        print(f"  ✓ ALL CHECKS PASSED")
        print(f"  ")
        print(f"  The Quip certification chain reproduces Liu et al.'s")
        print(f"  experimental result exactly: F_XEB = {fxeb:.4f}, Q_min = {Q},")
        print(f"  H = {H:.2f} bits, extractable = {ell:.2f} bits.")
    else:
        print(f"  ✗ ONE OR MORE CHECKS FAILED — investigate before relying on result")
    print()
    return all_pass


def main() -> int:
    # Default data path: alongside this script, in reference_data/
    script_dir = Path(__file__).resolve().parent
    default_data_path = script_dir / "reference_data" / "aggregated_probs.npy"

    # Allow command-line override
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        data_path = default_data_path

    success = verify(data_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
