# IBM QAOA Baseline Benchmarking Tool

Benchmarks the QAOA solver on Qiskit's AerSimulator — no IBM account or API keys required. Proves the solver works, measures performance, and validates solution quality against problems with known optimal energies.

## Dependencies

```bash
pip install qiskit qiskit-aer dimod numpy scipy python-dotenv psutil
```

## Quick Start

```bash
# From the project root:
python tools/ibm_qaoa_baseline.py --quick            # fast smoke test (seconds)
python tools/ibm_qaoa_baseline.py                    # standard run
python tools/ibm_qaoa_baseline.py --known-problems   # solution quality check
python tools/ibm_qaoa_baseline.py --extended         # thorough parameter sweep
python tools/ibm_qaoa_baseline.py --stress           # memory stress test (28 qubits)
python tools/ibm_qaoa_baseline.py --quick -v         # show QAOA solver steps
```

## Modes

### Pipeline Mode (`--quick` / standard / `--extended` / `--stress`)

Proves the mining pipeline works end-to-end using protocol-style problems:

```
nonce → generate_ising_model_from_nonce → solve_ising → evaluate_sampleset → MiningResult
```

Each solve generates a random nonce, builds an Ising problem on a subgraph of the protocol topology, runs the QAOA solver, and checks whether `evaluate_sampleset` returns a valid `MiningResult`. The full ~4,580-qubit protocol topology can't run on AerSimulator (2^n state vector), so all modes extract a connected subgraph:

| Flag | Subgraph | p | Optimizers | Shots | Solves/config |
|------|----------|---|------------|-------|---------------|
| `--quick` | 8 nodes | 1 | COBYLA | 512 | 1 |
| (standard) | 10 nodes | 1, 2 | COBYLA | 512, 1024 | 3 |
| `--extended` | 14 nodes | 1, 2, 3 | COBYLA, SPSA | 512, 1024, 2048 | 5 |
| `--stress` | 28 nodes | 1 | COBYLA | 1024 | 1 |

We don't know the optimal energy for random problems, so this mode doesn't measure solution quality — it proves the pipeline runs without crashing and produces valid MiningResults.

**What `evaluate_sampleset` checks:**
1. Energy threshold — is the best energy below the difficulty target?
2. Solution count — are there enough unique valid solutions?
3. Diversity — are the solutions different enough from each other?

If all three pass → returns a `MiningResult` (valid block proof). If any fail → returns `None`.

### Known-Problems Mode (`--known-problems`)

Tests solution quality against Ising problems with known optimal energies. Reports approximation ratios: `found_energy / optimal_energy`.

Problems come from two files:
- `tools/basic_ising_problems.py` — easy problems (2-16 qubits, simple structures like chains, grids, complete graphs). QAOA should hit ratio 1.0 on these.
- `tools/hard_ising_problems.py` — hard problems (16-20 qubit spin glasses with dense connectivity, random couplings, and frustration). QAOA at low depth may fall below 90%.

This mode does NOT go through `evaluate_sampleset` — it just checks how close to optimal the solver gets.

**Why the hard problems are hard:**
At 16 qubits, 4,096 final shots cover only ~6% of the state space (2^16 = 65,536 states). At 20 qubits, coverage drops to ~0.4% (2^20 = 1,048,576 states). The solver can't stumble onto the optimum by random sampling — it has to actually find good solutions. Dense spin glass problems with random ±1 couplings create many local minima and high frustration, which low-depth QAOA circuits struggle to navigate.

## Metrics

Both modes measure:

| Metric | Pipeline Mode | Known-Problems Mode |
|--------|--------------|-------------------|
| Energy levels | min and avg per solve | min per solve |
| Execution time | wall-clock seconds | wall-clock seconds |
| Memory usage | peak memory in MB | peak memory in MB |
| Diversity | top-10 solution diversity | — |
| Pipeline pass rate | MiningResult / None | — |
| Approximation ratio | — | found / optimal |

## All Flags

```
Mode selection (pick one):
  --quick              8-node subgraph, p=1, COBYLA, 512 shots, 1 solve
  (no flag)            10-node subgraph, p=1+2, COBYLA, 512+1024 shots, 3 solves
  --extended           14-node subgraph, p=1+2+3, COBYLA+SPSA, 512-2048 shots, 5 solves
  --stress             28-node subgraph, p=1, COBYLA, 1024 shots, 1 solve (memory test)
  --known-problems     Solution quality check against known optima

Overrides:
  -p 1 2 3             QAOA circuit depths
  --optimizer COBYLA SPSA   Classical optimizers
  --shots 512 1024     Shot counts per optimizer evaluation
  -n 5                 Solves per config/problem
  --subgraph-size 12   Override default subgraph node count
  --max-qubits 20      Max qubit count for --known-problems (default: 20)
  -t 60                Timeout in minutes (pipeline mode)
  -o results.json      Output filename
  -v, --verbose        Show QAOA solver steps (circuit building, optimization, sampling)
```

## What Success Looks Like

### `--quick` / standard / `--extended`
- Completes without errors
- `evaluate_sampleset` returns a `MiningResult` (not `None`)
- Energy scale depends on subgraph size (smaller graph = smaller energies)
- `--quick` (8 nodes) solves in seconds
- `--extended` (14 nodes) may take a few minutes per solve

### `--stress`
- 28-node subgraph — AerSimulator uses ~4 GB for the state vector
- Peak memory should reach several GB
- May take several minutes per solve
- Proves the pipeline handles larger problem sizes without crashing

### `--known-problems`
- Easy problems (basic_ising_problems.py) should hit 1.000 — 🎉 Excellent!
- Hard problems (hard_ising_problems.py) may drop — look for the verdict:
  - 🎉 **Excellent** (≥ 90%) — solver nailed it
  - 🌈 **Very Good** (80-90%) — strong result for a hard problem
  - 🎖️ **Good** (70-80%) — decent, expected for dense spin glasses at low p
  - ⚡ **Fair** (60-70%) — solver is struggling but finding reasonable solutions
  - 🥶 **Poor** (< 60%) — solver needs higher p, more shots, or a better optimizer
- Small problems solve in seconds; 20-qubit problems may take longer

## Example Output

### Pipeline mode (`--quick`)

```
🔬 IBM QAOA Pipeline Test (protocol-style problems)
=======================================================
⏰ Timeout: 15.0 min | 🔁 Solves per config: 1
✂️ Subgraph: 8 nodes, 14 edges (from 4580-node topology)
📐 Expected GSE (empirical): -12.8
💾 Memory before solver init: 45.2 MB

🧪 1 configs x 1 solves = 1 total solves

--- Config 1/1: p=1, optimizer=COBYLA, shots=512 ---
  💾 Memory after solver init: 48.3 MB
  Solve 1/1: nonce=3827194522 ... energy=-10.0, avg=-5.2,
    diversity=0.450, solutions=3, time=0.4s ✅ MiningResult

  📊 Config summary:
    Best energy:       -10.0
    Avg solve time:    0.4s (0.0 min)
    Avg diversity:     0.450
    Pipeline pass:     1/1
    💾 Peak memory:    52.1 MB

📊 Pipeline Test Summary (total: 0.0 min)
=======================================================
🏆 Best energy: -10.0
✅ Pipeline pass rate: 1/1
💾 Results saved to ibm_qaoa_pipeline_12345.json
✅ IBM QAOA baseline test complete!
```

### Known-problems mode

```
🔬 IBM QAOA Known-Problems Test (solution quality)
=======================================================
🔧 Solver: p=1, optimizer=COBYLA, shots=1024
🔁 Solves per problem: 3

📋 Loading problems from basic_ising_problems.py + hard_ising_problems.py...
  17 problems loaded (filtered to <= 20 qubits)

--- Problem 1/17: problem_0_2q ---
  2-spin ferromagnetic chain
  2 nodes, 1 edge, optimal = -1.0
  Solve 1/3 ... energy=-1.0, optimal=-1.0, ratio=1.000, gap=+0.0, time=0.22s
  ...

📊 Known-Problems Summary (total: 25.3s)
=======================================================
Problem                        Nodes  Optimal    Found   Ratio    Time
problem_0_2q                       2     -1.0     -1.0   1.000   0.22s
...
hard_0_16q                        16    -52.3    -46.0   0.880   1.85s
hard_5_20q                        20    -78.1    -64.0   0.819   4.22s

🏆 Overall avg approx ratio: 0.962
📉 Worst ratio:             0.819
🎉 Excellent (>= 90%):      14/17
🌈 Very Good (80-90%):      2/17
🎖️ Good (70-80%):           1/17
⚡ Fair (60-70%):           0/17
🥶 Poor (< 60%):            0/17
```

## Output Files

Results are saved as JSON with all raw data:
- `ibm_qaoa_pipeline_<timestamp>.json` — pipeline mode results
- `ibm_qaoa_known_problems_<timestamp>.json` — known-problems mode results

Files are saved in the directory you run the command from (usually the project root).

## How It Works Under the Hood

For each solve in pipeline mode:

1. `_extract_subgraph()` — cuts a connected subgraph from the protocol topology via BFS
2. `generate_ising_model_from_nonce()` — random nonce → h, J coefficients on the subgraph
3. `QAOASolverWrapper.solve_ising()` — the 5-step QAOA pipeline:
   - Step 1: Build cost Hamiltonian (h, J → SparsePauliOp)
   - Step 2: Build QAOA circuit (QAOAAnsatz with p layers)
   - Step 3: Transpile circuit for AerSimulator
   - Step 4: Variational optimization (~100 iterations of circuit execution)
   - Step 5: Final sampling with optimized angles
4. `evaluate_sampleset()` — checks energy, diversity, solution count → MiningResult or None

For known-problems mode, steps 1-2 are replaced by loading h, J from `basic_ising_problems.py` / `hard_ising_problems.py`, and step 4 is replaced by comparing the found energy against the known optimal.

Use `--verbose` / `-v` to see each step logged in real time.