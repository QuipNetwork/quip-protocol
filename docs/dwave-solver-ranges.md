# D-Wave QPU h/J Ranges

Per-solver linear-bias (`h`) and coupler (`J`) range limits, plus extended-J
and per-qubit coupling bounds. Generated from `solver.properties` for every
QPU your D-Wave API key can reach.

## Field meanings

| Field | Meaning |
|---|---|
| `h_range` | Allowed range for each single-qubit linear bias `h_i`. Typically `[-4, 4]` (Advantage) or `[-6, 6]` (Advantage2). |
| `j_range` | Allowed range for each single-coupler quadratic bias `J_ij` in **standard** mode. Almost always `[-1, 1]`. |
| `extended_j_range` | Wider `J` bounds available when you opt into extended-J mode (sets `extended_j_range_mode` / `flux_drift_compensation` on the sample call). Usually `[-2, 1]`. Trades a small calibration penalty for room beyond `-1`. |
| `per_qubit_coupling_range` | Cumulative bound on `Σ_j |J_ij|` for each qubit `i` (i.e. the qubit's coupling budget). Independent of `j_range`; often the binding constraint for dense problems. |
| `Qubits (active/total)` | Total qubits on the chip vs. how many are working at calibration time. Defects vary per revision; offline qubits should be clamped to deterministic spins (the miner already does this). |

See the [D-Wave solver-properties reference](https://docs.dwavequantum.com/en/latest/quantum_research/solver_properties_specifics.html) for authoritative definitions.

## Solver ranges

<!-- BEGIN GENERATED -->
_Last generated: not yet run. Regenerate with `python tools/dump_solver_ranges.py`._

_(Run the tool to populate this table — it queries SAPI metadata, no QPU time is consumed.)_
<!-- END GENERATED -->

## Current QPoW usage vs. headroom

The quantum proof-of-work in this repo deliberately lives well inside the
conservative envelope. The table above shows what the hardware allows; this
section shows what the code actually exercises today, so a reader can tell
which numbers are load-bearing and which represent headroom.

### Currently uses

- `generate_ising_model_from_nonce()` (`shared/quantum_proof_of_work.py:72`,
  `:90`) emits **ternary h ∈ {-1, 0, +1}** and **binary J ∈ {-1, +1}**. Both
  fit inside every solver's standard `h_range` and `j_range`, so no scaling is
  required for the current PoW.
- Dynamic `chain_strength` derived from problem magnitude
  (`QPU/dwave_sampler.py:560-563`, `:640-643`).
- Fixed `annealing_time` and `num_reads` on the sample call
  (`QPU/dwave_sampler.py:236-241`, `:286-290`).

### Does NOT use (headroom)

- **`extended_j_range`** — the `extended_j_range_mode` and
  `flux_drift_compensation` parameters are never set on
  `solver.sample_ising()`. Mining could opt in if a future PoW pushes `J`
  past `-1`, at the cost of a small calibration penalty.
- **`per_qubit_coupling_range`** — never computed or validated. No
  per-qubit `Σ |J_ij|` check exists anywhere in the repo. The current binary
  J with bounded degree stays well inside, but a denser model would need
  this check.
- **`h_range` / `j_range` validation** — `solver.properties` is copied into
  the sampler at `QPU/dwave_sampler.py:322` but no pre-submission check
  reads it back. The default ±1 problem fits everywhere, so this never
  bites today.
- **Anneal-schedule tuning, reverse anneal, h-gain schedules** — fixed
  schedule only.
- **`normalize_for_qpu` / `to_bqm_normalized`** — referenced in older docs
  but no such functions exist in the codebase; `h` and `J` go straight to
  `BinaryQuadraticModel.from_ising()` (`QPU/dwave_sampler.py:547`, `:636`).

If a future PoW variant pushes outside ternary-h / binary-J, the first
features to wire in would be `extended_j_range` and a per-qubit
coupling-sum guard.
