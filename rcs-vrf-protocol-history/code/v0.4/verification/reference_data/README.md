# Reference Data — Liu et al. Experimental Probabilities

## Contents

- **`aggregated_probs.npy`** — A numpy array of 1,522 float64 values, each representing the ideal probability $p_{C_i}(x_i)$ that the $i$-th validation circuit assigns to the bitstring actually observed during Liu et al.'s Quantinuum H2 experimental run.

## Provenance

This data is taken directly from Liu et al.'s Zenodo archive accompanying their Nature 2025 paper, at:

> [zenodo.org/records/15192591](https://zenodo.org/records/15192591)
> File path within archive: `reproduce_verification/data/aggregated_probs.npy`

The full citation:

> Liu, M., Hoke, S.D., Sherrill, M., Lukin, M., Choi, S., Shi, Y., et al. (2025). *Certified randomness using a trapped-ion quantum processor.* Nature **640**, 343–349. arXiv:2503.20498.

## How the data was produced (by Liu et al.)

1. **Sampling step (Quantinuum H2):** For each of 30,010 successful protocol rounds, the Quantinuum H2-1 trapped-ion processor executed a 56-qubit random quantum circuit and produced one measurement bitstring.

2. **Validation set selection:** From the 30,010 rounds, Liu sampled $m = 1{,}522$ rounds at random to form the audit set. (Liu's notebook indicates `valIdx` ranges from 0 to 1531 with indices 1390–1399 skipped — these exclusions explain the 1,522 count.)

3. **Ideal probability computation (Frontier supercomputer):** For each of the 1,522 audit circuits, Liu used tensor-network contraction on the Frontier supercomputer to classically compute the ideal probability $p_{C_i}(x_i)$ that the noiseless quantum execution would have assigned to the observed bitstring. This is the expensive step — each 56-qubit circuit requires substantial supercomputer time.

4. **Aggregation:** The 1,522 computed probabilities are aggregated into the array stored in `aggregated_probs.npy`.

## How we use this data

The verification script `verify_against_liu_data.py` loads this array and computes:

- F_XEB = $\frac{2^{56}}{1522} \sum_i p_{C_i}(x_i) - 1$

The expected result is F_XEB ≈ 0.3197 (just above the acceptance threshold $\chi = 0.3$), which then feeds through the certified-entropy chain to reproduce Liu's published numbers exactly.

## Licensing

Liu et al.'s Zenodo archive is published under an open license. This file is reproduced here under that license for the purpose of independent verification, with full attribution as above.

## What this data does *not* include

Worth being explicit about what's *not* in this file:

- **Raw bitstrings.** The 1,522 measurement outcomes from the QPU are not in this array — they've already been folded into the probability values via $p_{C_i}(x_i)$.

- **Circuit definitions.** The circuits themselves (in QASM/JSON format) are in Liu's Zenodo archive at `data/circuits/`. They are not needed for F_XEB verification because the ideal probabilities have already been computed.

- **Timing data.** The per-circuit response times are tracked in Liu's archive but are not in this file. Timing enforcement is a separate protocol concern.

## Verification status

| Quantity | Reproduced exactly | Reference value |
|---|---|---|
| Validation circuit count | ✓ | 1,522 |
| F_XEB | ✓ | 0.3197 (just above $\chi = 0.3$) |
| Q_min | ✓ | 1,297 |
| Certified min-entropy | ✓ | 71,313.07 bits |
| Extractable bits (Toeplitz/LHL) | ✓ | 71,273.21 bits |

Run `python verify_against_liu_data.py` from the parent directory to reproduce.
