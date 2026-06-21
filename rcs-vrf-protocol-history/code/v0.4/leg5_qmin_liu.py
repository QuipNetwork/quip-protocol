"""
leg5_qmin_liu.py
================

EXACT reproduction of Liu et al.'s Q_min computation, using the parameters
from their actual reference notebook (Zenodo entropy.py + Table2-bounds-on-
extractable-entropy.ipynb).

This module supersedes the earlier curve-fit-style attempts. It uses the
actual parameter values from the authors' notebook:

    FRONTIER = 2e18 FLOPS           (Frontier theoretical peak)
    A = 4 * FRONTIER = 8e18 FLOPS   (adversary has 4 supercomputers worth)
    eff = 0.5                       (numerical efficiency — running at 50% peak)
    B_val = 90e18 FLOPs/circuit     (per-circuit simulation cost)

VERIFICATION (exact reproduction of Liu's notebook output):
    Q_min      = 1297
    H_min^cert = 71313.07 bits
    ell        = 71273.21 extracted bits

PHYSICAL INTERPRETATION:
The adversary model assumes 4 supercomputers worth of theoretical peak
compute (~8 exaFLOPS), running at 50% sustained efficiency. The effective
compute is 4 exaFLOPS — roughly matching the actual sustained capacity of
the world's top four supercomputers (Frontier, Aurora, El Capitan, Fugaku)
combined. This is a deliberate worst-case adversary specification.

HONEST NOTE ABOUT THE PRIOR "c_eff = 4.46" CALIBRATION:
An earlier version of this module used c_eff = 4.46 with A = 0.897e18
(Frontier sustained from Table I). That reproduced the paper's numbers
because the *product* eff * A is nearly identical (2934 vs 2934) — but
the factorization was wrong, and the c_eff = 4.46 was found by curve-fit,
not derived from the source code. The principled factorization
(eff=0.5, A=4*Frontier_theoretical) makes the modelling assumptions
explicit and is what the authors actually use.

Reference:
  Liu et al., "Certified randomness using a trapped-ion quantum processor,"
  Nature 640:343 (2025); arXiv:2503.20498.
  Zenodo: zenodo.org/records/15192591
    - src/entropy.py (Q_FP_exact_hypergeometric_chernoff)
    - reproduce_figures/Table2-bounds-on-extractable-entropy.ipynb
"""

from __future__ import annotations
import numpy as np
from math import ceil, log2
from scipy.special import gammainc as P
from scipy.stats import hypergeom


# Liu's actual parameters from Table2-bounds-on-extractable-entropy.ipynb
FRONTIER_THEORETICAL = 2.0e18  # Frontier theoretical peak FLOPS

PAPER = dict(
    # Protocol parameters
    n=56,
    M=30010,
    m=1522,
    chi=0.3,
    t_threshold_per_circuit=2.2,
    epsilon_sou=1e-6,

    # Adversary model parameters (from notebook cell 3)
    A=4 * FRONTIER_THEORETICAL,           # 8e18 FLOPS: 4 supercomputers at peak
    eff=0.5,                              # numerical efficiency factor
    B_val_per_circuit=90.0e18,            # 90 exaFLOPs per circuit

    # Expected outputs (from notebook cell 3 stdout)
    Q_min_paper=1297,
    H_paper=71313.068431,
    ell_paper=71273.205294,
)


def compute_q_min(epsilon_sou, chi, m, M, t_tot, eff, A, B_val,
                  allow_negative=False):
    """
    Liu et al.'s Q_min computation, exactly reproducing
    Q_FP_exact_hypergeometric_chernoff from their Zenodo src/entropy.py.

    Args:
      epsilon_sou: soundness parameter (paper uses 1e-6).
      chi: F_XEB acceptance threshold (paper uses 0.3).
      m: audit subset size (paper uses 1522).
      M: total batch size (paper uses 30010).
      t_tot: total wall-clock time available across all M circuits, seconds.
             For Liu's setup: t_tot = M * t_threshold_per_circuit = 66022s.
      eff: numerical efficiency factor (paper uses 0.5).
      A: adversary's classical compute, FLOPS (paper uses 4*FRONTIER = 8e18).
      B_val: cost to simulate one circuit, FLOPs (paper uses 90e18).
      allow_negative: search Q in [-M, M] instead of [0, M]; used only for
                      diagnostic edge cases.

    Returns:
      int: Q_min (the boundary value where adversary's pass probability
           crosses epsilon_sou/2).
    """
    high = M
    low = -M if allow_negative else 0
    previous_Q = None

    while True:
        Q = (high + low) // 2
        if Q == previous_Q:
            return Q
        previous_Q = Q

        # Closed-form Chernoff delta
        exp_Lc = min(M - Q, eff * A * t_tot / B_val)
        delta = np.sqrt(np.log(1 / (epsilon_sou / 2)) * (3 / exp_Lc))
        Lmax = ceil(Q + (1 + delta) * exp_Lc)

        # Hypergeometric tail: Pr[adversary passes XEB | <= Lmax PT samples]
        pmf = hypergeom.pmf(np.arange(m + 1), M, Lmax, m)
        ans = 1 - np.dot(pmf, P(np.arange(m + 1) + m, m * (chi + 1)))

        # Binary search converges to boundary: largest Q with ans >= eps/2
        if ans < epsilon_sou / 2:
            low = Q
        else:
            high = Q


def certified_min_entropy(q_min: int, n: int, epsilon_sou: float) -> float:
    """
    Liu's certified min-entropy formula, matching the notebook cell 3:
        H = Q_min * (n - 1) + log2(epsilon_sou) - 2
    Note: log2(eps_sou) - 2 = -log2(1/eps_sou) - log2(4) = -log2(4/eps_sou)
    which is the standard form H = Q_min*(n-1) - log2(1/eps_s) with
    eps_s = eps_sou/4 (Liu's smoothing split).
    """
    return q_min * (n - 1) + log2(epsilon_sou) - 2


def extractable_bits(q_min: int, n: int, epsilon_sou: float) -> float:
    """
    Liu Corollary 7 extractable bits (Toeplitz extractor, quantum-proof):
        ell = Q_min * (n - 1) + 3*log2(epsilon_sou) - 2
    From the notebook: ell = H + 2*log2(epsilon_sou), which simplifies to
    the standard form ell = Q_min*(n-1) - 3*log2(1/eps_sou) - 2.
    """
    return q_min * (n - 1) + 3 * log2(epsilon_sou) - 2


def validate():
    """Verify exact reproduction of Liu's notebook output."""
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
    H = certified_min_entropy(Q, PAPER['n'], PAPER['epsilon_sou'])
    ell = extractable_bits(Q, PAPER['n'], PAPER['epsilon_sou'])

    print("EXACT reproduction of Liu et al. notebook cell 3:")
    print(f"  Q_min:           {Q}              (paper: {PAPER['Q_min_paper']})")
    print(f"  H_certified:     {H:.6f}    (paper: {PAPER['H_paper']:.6f})")
    print(f"  Extractable:     {ell:.6f}    (paper: {PAPER['ell_paper']:.6f})")
    print()
    assert Q == PAPER['Q_min_paper'], f"Q_min mismatch: {Q} vs {PAPER['Q_min_paper']}"
    assert abs(H - PAPER['H_paper']) < 0.01, f"H mismatch: {H} vs {PAPER['H_paper']}"
    assert abs(ell - PAPER['ell_paper']) < 0.01, f"ell mismatch: {ell} vs {PAPER['ell_paper']}"
    print("✓ All three quantities reproduced exactly.")


if __name__ == "__main__":
    validate()
