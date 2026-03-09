# IBM QAOA Miner for Quantum Proof-of-Work

Gate-based quantum mining using the Quantum Approximate Optimization Algorithm (QAOA) on IBM Quantum hardware or local simulation via Qiskit Aer.

## Dependencies

```bash
pip install qiskit qiskit-aer dimod numpy scipy
```

## Architecture

Three modules work together:

```
shared/
└── quantum_proof_of_work.py  ← Problem generation + solution evaluation (shared by all miners)

QPU/IBM/
├── ibm_qaoa_solver.py        ← QAOA engine: (h, J) → dimod.SampleSet
└── ibm_qaoa_miner.py         ← Mining loop: generates problems, submits solves, evaluates results
```

`quantum_proof_of_work.py` defines the protocol — how problems are generated from block data, how solutions are scored, and how mining results are validated. Both the D-Wave miner and the IBM QAOA miner depend on it.

`ibm_qaoa_solver.py` is analogous to `dwave_sampler.py` — it wraps hardware-specific details behind a clean interface. `ibm_qaoa_miner.py` is analogous to `dwave_miner.py` — it implements the mining loop on top of that interface.

### Outline
The solver's job is: "Give me an Ising problem (h, J), I'll give you back solutions as a dimod.SampleSet." That's the interface. How it gets those solutions — quantum annealing, QAOA circuits, simulated annealing — doesn't matter to anything above it.
The miner doesn't know or care that QAOA builds circuits, runs an optimizer 100 times, transpiles for a backend. It just calls solve_ising_async(h, J) and gets back a future that eventually contains a SampleSet. Same as D-Wave's miner calls sample_ising_async(h, J) and gets back a future that eventually contains a SampleSet.
The miner handles everything around the solve: where do the problems come from (block headers → nonces → Ising models), when to solve (budget checks, stop signals), what to do with results (evaluate against requirements, cache near-misses, check difficulty decay), and when to stop (another miner won, budget exhausted).
So "building on top of" means the miner orchestrates the full mining workflow and treats the solver as a black box that converts problems into solutions. Swap the solver, the miner logic stays the same.


## How It Works

### The Problem

The blockchain generates an Ising optimization problem for each mining attempt. The protocol defines a fixed graph — currently 4,582 nodes and 41,596 edges, extracted from a D-Wave Advantage2 chip topology. This graph never changes; it's the same for every miner, every attempt, every block.

What changes is the **weights** on that graph. Each mining attempt generates a random nonce from the block header, miner ID, and a random salt. That nonce seeds a deterministic random number generator that assigns:

- A **field bias** (h) to each node: randomly chosen from {-1, 0, +1}
- A **coupling value** (J) to each edge: randomly ±1

The result is an Ising problem: find the spin assignment (s = +1 or −1 per node) that minimizes total energy:

```
E = Σ hᵢ·sᵢ  +  Σ Jᵢⱼ·sᵢ·sⱼ
    ─────────    ──────────────
    linear       quadratic
    (field)      (coupling)
```

Lower energy = better solution. Typical energies are around −14,000 to −16,000 for this graph. The blockchain sets a difficulty threshold (e.g., −15,000). Solutions with energy below (more negative than) that threshold count as valid. The miner needs to find enough valid, diverse solutions to earn the right to mine a block.

### Why This Is Hard

The coupling values create **frustration** — conflicts where satisfying one edge forces another to be unsatisfied. With 41,596 edges, the energy landscape has an enormous number of local minima. Finding the global minimum is NP-hard. The protocol deliberately includes zero-field nodes (h = 0 for ~1/3 of nodes) so that the graph structure dominates, preventing simple greedy solutions and favoring quantum hardware.

### QAOA Approach

Unlike D-Wave's quantum annealing (which runs on specialized hardware), QAOA is a variational quantum algorithm that runs on gate-based quantum computers. It works in five steps:

1. **Build cost operator** — Translate the Ising (h, J) into a quantum Hamiltonian using Pauli-Z operators
2. **Build QAOA circuit** — Create a parameterized circuit with `p` layers, each containing a cost layer (driven by gammas) and a mixer layer (driven by betas), giving `2p` tunable angle parameters
3. **Transpile** — Compile the abstract circuit for the target backend (maps logical gates to native hardware gates)
4. **Optimize** — Run a classical optimizer (COBYLA, SPSA, etc.) that repeatedly executes the circuit with different angles, measures the output, computes average energy, and searches for the angles that minimize it
5. **Final sample** — Run the optimized circuit with many shots to collect diverse low-energy solutions, then convert measurement results to a `dimod.SampleSet`

### Mining Loop

The miner wraps this in a sequential loop:

```
while not stopped:
    1. Generate random salt → nonce → Ising problem (h, J)
    2. Compute adaptive parameters (circuit depth, shots, iterations)
    3. Submit QAOA solve asynchronously
    4. Poll every 200ms:
       - Check if blockchain says stop (another miner won)
       - Check for difficulty decay (threshold relaxes over time)
       - Re-evaluate cached near-miss results against relaxed difficulty
       - If cached result now qualifies → cancel solve, return it
    5. Process result → evaluate against requirements
    6. If valid → return. If not → cache and continue.
```

### What Makes a Valid Mining Result

A mining result must satisfy three requirements simultaneously (checked by `evaluate_sampleset`):

1. **Energy threshold** — At least `min_solutions` solutions must have energy below `difficulty_energy`
2. **Diversity** — The selected solutions must be sufficiently different from each other (measured by average normalized Hamming distance, accounting for global spin-flip symmetry)
3. **Solution count** — Must have at least `min_solutions` valid, unique solutions

If all three are met, a `MiningResult` is created and returned to the blockchain.


## Code Walkthrough: quantum_proof_of_work.py

This module is the protocol layer — shared by all miners (D-Wave, IBM, simulated annealing). It defines how Ising problems are generated and how results are evaluated.

### `ising_nonce_from_block(prev_hash, miner_id, cur_index, salt) → int`

Generates a deterministic 32-bit nonce from block parameters. Concatenates the previous block hash, miner ID, block index, and a random salt, then BLAKE3-hashes the result. The same inputs always produce the same nonce, which is critical for validation — any node can regenerate the exact same problem to verify a solution.

### `generate_ising_model_from_nonce(nonce, nodes, edges, h_values) → (h, J)`

Creates a random Ising problem on the protocol graph, seeded by the nonce. Uses numpy's `default_rng` (thread-safe, no global state) to generate:

- **J values**: Random ±1 for each of the 41,596 edges. `2 * rng.integers(2) - 1` maps {0,1} to {-1,+1}.
- **h values**: Random selection from `[-1, 0, +1]` for each of the 4,582 nodes. The default ternary distribution means ~1/3 of nodes have no field bias, keeping the problem graph-structured.

Same nonce always produces the same (h, J). Different salt → different nonce → different problem.

### `energy_of_solution(solution, h, J, nodes) → float`

The core scoring function. Computes the Ising energy for a given spin assignment:

```
E = Σ hᵢ·sᵢ + Σ Jᵢⱼ·sᵢ·sⱼ
```

Maps solution values to {-1, +1} spins, then sums the linear (h) and quadratic (J) contributions. Used everywhere — inside the QAOA optimizer loop (called ~100 times per solve), in final sample evaluation, and in block validation.

### `evaluate_sampleset(sampleset, requirements, ...) → MiningResult or None`

The gatekeeper. Takes a `dimod.SampleSet` (collection of spin assignments with energies) and checks whether it meets all three blockchain requirements.

Pipeline:
1. **Fast exit** — If best energy doesn't beat the threshold, return None immediately
2. **Count check** — If fewer than `min_solutions` samples meet the threshold, return None
3. **Filter** — Extract unique solutions that meet the energy threshold
4. **Diversity selection** — Use farthest-point sampling to pick the most diverse subset of `min_solutions`
5. **Diversity check** — Compute average pairwise Hamming distance; reject if below `min_diversity`
6. **Build result** — Package as `MiningResult` with solutions, energy, diversity, timing

Has two modes: fast path (`skip_validation=True`, trusts sampler output during mining) and slow path (`skip_validation=False`, full per-solution validation for verifying other miners' blocks).

### `calculate_hamming_distance(s1, s2) → int`

Symmetric Hamming distance between two spin arrays. Accounts for global spin-flip symmetry: `distance(s, -s) = 0`, because flipping every spin in an Ising solution doesn't change the physics. Takes the minimum of normal distance and inverted distance.

### `calculate_diversity(solutions) → float`

Average normalized Hamming distance across all solution pairs. Returns a value between 0 (all identical) and ~0.5 (maximally diverse). The blockchain requires this to be above `min_diversity` to prevent miners from submitting trivial variations of the same solution.

### `select_diverse_solutions(solutions, target_count) → indices`

Given more valid solutions than needed, selects the most diverse subset. Uses farthest-point sampling (greedy: always pick the point farthest from the already-selected set) followed by local search refinement (try swapping elements to improve total diversity). Pre-computes a vectorized distance matrix for performance.

### `validate_quantum_proof(quantum_proof, miner_id, requirements, block_index, previous_hash) → bool`

Full validation for blocks received from other miners. Regenerates the Ising problem from block data, validates every solution (format, spin values, topology consistency), recomputes energies, checks count, selects diverse subset, and checks diversity. This is the slow path that ensures no one can fake a mining result.

### `validate_solution(spins, h, J, nodes, edges) → dict`

Per-solution validation: checks spin array length matches topology, all values are in {-1, +1}, topology is consistent (h and J match expected nodes/edges), computes energy, and measures coupling satisfaction rate.


## Code Walkthrough: ibm_qaoa_solver.py

### OPTIMIZER_CONFIGS

Lookup table mapping optimizer names to `scipy.optimize.minimize` configurations. Four classical optimizers are available: COBYLA (gradient-free, default), Nelder-Mead (robust but slow), Powell (direction-set method), and L-BFGS-B (gradient-based, fast on smooth landscapes).

### SPSAOptimizer

Custom Simultaneous Perturbation Stochastic Approximation implementation. Not in scipy, so it's built from scratch. Preferred for real quantum hardware because it only needs 2 circuit evaluations per iteration regardless of how many parameters the circuit has.

Algorithm per iteration:
- Shrink step size: `ak = a / k^alpha` (large early steps, fine-tuning later)
- Shrink perturbation: `ck = c / k^gamma`
- Pick random direction: `delta = random ±1 per parameter`
- Evaluate objective at `params + ck·delta` and `params - ck·delta`
- Estimate gradient: `(f+ − f−) / (2·ck·delta)`
- Update: `params = params - ak · gradient`

### QAOAFuture

Lightweight wrapper around Python's `concurrent.futures.Future` that matches the interface of D-Wave's `EmbeddedFuture`. This lets the miner use the same polling pattern for both backends.

Key interface: `future.sampleset` (blocks until done), `future.done()` (non-blocking check), `future.cancel()`, `future.elapsed` (wall-clock time since submission).

### QAOASolverWrapper

The main solver class. Stores the topology (nodes, edges), configures the backend (AerSimulator for local simulation or a real IBM QPU), sets QAOA defaults, and creates a single-threaded executor for async solves.

#### `_build_cost_operator(h, J) → SparsePauliOp`

Translates the Ising problem into a quantum operator. Each linear bias `hᵢ` becomes a single-qubit Z term; each coupling `Jᵢⱼ` becomes a two-qubit ZZ term. For 4,582 qubits, each Pauli string is 4,582 characters of 'I's with 'Z's at the relevant positions.

#### `_build_circuit(cost_op, p) → QAOAAnsatz`

Creates the parameterized QAOA circuit using Qiskit's built-in `QAOAAnsatz`. The circuit has `2p` parameters: `p` gamma angles (controlling cost layer strength) and `p` beta angles (controlling mixer layer strength). Accepts an optional `p` override for per-solve adaptive depth.

#### `_transpile(circuit) → compiled circuit`

Compiles the abstract circuit for the specific backend. On AerSimulator this is mostly a no-op. On real IBM hardware it decomposes gates to the native gate set, maps logical qubits to physical qubits, and optimizes gate count.

#### `_evaluate_circuit(circuit, params) → float`

The objective function called ~100 times by the optimizer. Binds angle parameters to the circuit, runs it, and computes the expectation energy: `⟨E⟩ = Σ P(x)·E(x)` where P(x) is the probability of measuring bitstring x and E(x) is its Ising energy. Calls `energy_of_solution` from quantum_proof_of_work.

Handles Qiskit's little-endian bitstring convention (qubit 0 is the rightmost character) and converts 0/1 measurements to +1/−1 Ising spins.

#### `_optimize(circuit, stop_event) → best_params or None`

Drives the classical optimization loop. Creates a closure that checks `stop_event` on every evaluation (for responsiveness), calls `_evaluate_circuit`, and logs every 10 iterations. Starts with random angles uniformly distributed in [0, 2π].

For SPSA: creates an `SPSAOptimizer` instance and calls `minimize()`.
For others: calls `scipy.optimize.minimize()` with the configured method.

Catches `StopIteration` (raised when `stop_event` fires mid-evaluation) and returns `None` to signal interruption.

#### `_final_sample(circuit, best_params) → counts`

After optimization finds the best angles, runs the circuit one final time with many more shots to collect a diverse set of solutions. Returns raw Qiskit measurement counts.

#### `_counts_to_sampleset(counts, h, J) → dimod.SampleSet`

Converts Qiskit measurement results into the `dimod.SampleSet` format used throughout the blockchain codebase. For each unique bitstring: reverses for endianness, converts 0/1 → +1/−1 spins, computes energy via `energy_of_solution`, and packages everything with variable labels matching the topology node IDs.

#### `solve_ising(h, J, stop_event, params) → SampleSet or None`

Public interface. Runs the full five-step pipeline. Accepts optional per-solve parameter overrides (`p`, `shots`, `final_shots`, `max_iter`) from `adapt_parameters()`, falling back to constructor defaults for any not provided.

#### `solve_ising_async(h, J, stop_event, params) → QAOAFuture`

Submits `solve_ising` to a background thread and returns a `QAOAFuture` immediately. This is what the miner calls — it needs the solve to run in the background so it can poll for `stop_event` and difficulty decay.


## Code Walkthrough: ibm_qaoa_miner.py

### IBMQAOAMiner

Subclass of `BaseMiner`. Creates a `QAOASolverWrapper` with the configured parameters, registers SIGTERM handler for cleanup, and stores topology references.

#### `_bridge_stop_event(mp_event)`

The blockchain uses `multiprocessing.Event` for inter-process signaling, but QAOA runs in-process using threads. This method creates a `threading.Event` and spawns a daemon thread that polls the multiprocessing event every 100ms, propagating the stop signal when detected.

#### `mine_block(...) → MiningResult or None`

The main mining loop. Setup phase computes current difficulty (with decay), extracts topology from the solver, and calls `adapt_parameters()` to configure QAOA parameters based on difficulty.

The sequential solve loop generates a random salt and nonce via `ising_nonce_from_block`, creates the Ising problem via `generate_ising_model_from_nonce`, submits an async QAOA solve, then enters the poll loop. The poll loop checks every 200ms for: completion, external stop signal, and difficulty decay. On difficulty decay, it recomputes adaptive parameters and re-checks cached near-miss attempts — if one now qualifies under relaxed difficulty, it cancels the in-progress solve and returns immediately.

When a solve completes, the result is evaluated via `evaluate_sampleset` from quantum_proof_of_work. If it meets all three requirements (energy, diversity, solution count), a `MiningResult` is returned. Otherwise the result is cached via `update_top_samples` for potential re-evaluation on difficulty decay.

### adapt_parameters(difficulty_energy, ...) → dict

Module-level function that maps blockchain difficulty to QAOA-specific parameters. Normalizes difficulty to a 0–1 scale using `energy_to_difficulty()` from energy_utils, then computes:

- **Circuit depth (p)**: 1 for easy (< 0.3), 2 for medium (0.3–0.7), 3 for hard (> 0.7)
- **Optimizer iterations**: Linear from 50 (easiest) to 200 (hardest)
- **Shots per evaluation**: Linear from 512 to 2,048
- **Final sampling shots**: Scales with both difficulty and `min_solutions` requirement


## Optimizer Comparison

| Optimizer | Evals/Iter | Best For | Trade-off |
|-----------|-----------|----------|-----------|
| COBYLA | 1 | Simulator (low noise) | Fast convergence, sensitive to noise |
| SPSA | 2 | Real QPU (high noise) | Noise-robust, slower convergence |
| Nelder-Mead | varies | Robust exploration | Slowest, most reliable |
| Powell | varies | Smooth landscapes | Fast, fragile with noise |
| L-BFGS-B | ~2N | Very smooth landscapes | Fastest if gradients exist, rare in practice |

COBYLA is the default. Switch to SPSA for real IBM hardware.


## Usage

### Local Simulation

```python
from QPU.IBM import IBMQAOAMiner

miner = IBMQAOAMiner(
    miner_id="ibm-qaoa-1",
    nodes=protocol_nodes,       # from protocol topology
    edges=protocol_edges,       # from protocol topology
    backend=None,               # defaults to AerSimulator
    p=1,                        # circuit depth
    optimizer='COBYLA',         # classical optimizer
    shots=1024,                 # shots per optimizer evaluation
)

result = miner.mine_block(prev_block, node_info, requirements, prev_timestamp, stop_event)
```

### Real IBM Hardware

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

miner = IBMQAOAMiner(
    miner_id="ibm-qaoa-1",
    nodes=protocol_nodes,
    edges=protocol_edges,
    backend=backend,
    optimizer='SPSA',           # noise-robust optimizer for real hardware
    shots=512,                  # fewer shots (hardware is expensive)
)
```


## Data Flow

```
Blockchain
  │
  ├── prev_block, requirements, stop_event
  │
  ▼
IBMQAOAMiner.mine_block()
  │
  ├── compute_current_requirements()               → current difficulty (with decay)
  ├── adapt_parameters()                           → {p, shots, final_shots, max_iter}
  │
  │   ┌─── loop ──────────────────────────────────────────────────────────┐
  │   │                                                                   │
  │   │ quantum_proof_of_work:                                            │
  │   ├── ising_nonce_from_block(prev_hash, id, index, salt)  → nonce     │
  │   ├── generate_ising_model_from_nonce(nonce, nodes, edges) → (h, J)   │
  │   │                                                                   │
  │   │ ibm_qaoa_solver:                                                  │
  │   ▼ QAOASolverWrapper.solve_ising_async(h, J, stop, params)           │
  │   │                                                                   │
  │   ├── _build_cost_operator(h, J)             → SparsePauliOp          │
  │   ├── _build_circuit(cost_op, p)             → QAOAAnsatz             │
  │   ├── _transpile(circuit)                    → compiled circuit        │
  │   ├── _optimize(circuit, stop)               → best angles             │
  │   │     └── _evaluate_circuit() ×N                                    │
  │   │           └── energy_of_solution() ×shots  ← quantum_proof_of_work│
  │   ├── _final_sample(circuit, angles)         → counts                  │
  │   └── _counts_to_sampleset(counts)           → dimod.SampleSet        │
  │         └── energy_of_solution() per bitstring ← quantum_proof_of_work│
  │                                                                       │
  │   ◄── poll 200ms: stop? decay? cached hit? ──────────────────────────┤
  │                                                                       │
  │   quantum_proof_of_work:                                              │
  │   evaluate_sampleset(sampleset, requirements)                         │
  │   ├── energy threshold check                                          │
  │   ├── solution count check                                            │
  │   ├── select_diverse_solutions() → farthest-point sampling            │
  │   ├── calculate_diversity() → Hamming distance                        │
  │   └── → MiningResult or None                                          │
  │                                                                       │
  │   └─── if valid → return. else cache and repeat ──────────────────────┘
  │
  ▼
MiningResult → Blockchain
```
