# IBM QAOA Miner

QAOA-based quantum miner for the QUIP proof-of-work protocol. Currently runs on Qiskit's AerSimulator (local simulation) — no IBM account or API keys needed.

## Setup

```bash
# Install project (registers quip-network-node on PATH)
pip install -e .

# Install IBM QAOA dependencies
pip install qiskit qiskit-aer qiskit-ibm-runtime dimod numpy scipy psutil

# If quip-network-node is not recognized after install, add Scripts to PATH:
# PowerShell:
$env:PATH += ";C:\Users\<you>\AppData\Local\Python\<version>\Scripts"
```

## Usage

### Baseline Testing (no node infrastructure needed)

```bash
# Quick smoke test — 8-node subgraph, ~1 min
python tools/ibm_qaoa_baseline.py --quick -v

# Standard test — 10-node subgraph, parameter sweep
python tools/ibm_qaoa_baseline.py -v

# Solution quality check against known optimal energies
python tools/ibm_qaoa_baseline.py --known-problems -v

# Memory stress test — 28-node subgraph, ~4 GB RAM
python tools/ibm_qaoa_baseline.py --stress -v
```

### Running a Node

```bash
# AerSimulator (default, no API key needed)
quip-network-node --config quip.network.ibm_qaoa.example.toml ibm_qaoa --auto-mine

# IBM hardware
quip-network-node --config quip.network.ibm_qaoa.example.toml ibm_qaoa --backend ibm --ibm-api-token TOKEN

# With custom QAOA parameters
quip-network-node --config quip.network.ibm_qaoa.example.toml ibm_qaoa --auto-mine --subgraph-size 10 --p 2 --optimizer SPSA --shots 512
```

### TOML Configuration

See `quip.network.ibm_qaoa.example.toml` for all options. Key settings:

```toml
[ibm_qaoa]
backend = "aer"           # "aer" (local) or "ibm" (real hardware)
subgraph_size = 14        # nodes extracted from protocol topology
p = 1                     # QAOA circuit depth
optimizer = "COBYLA"      # COBYLA, NELDER_MEAD, POWELL, L_BFGS_B, SPSA
shots = 1024              # shots per optimizer evaluation

# IBM hardware (only when backend = "ibm")
# ibm_api_token = "your-token"
# ibm_backend_name = "ibm_brisbane"
```

## Why QAOA

The protocol's proof-of-work is an Ising optimization problem defined on a fixed graph (4,579 nodes, 41,549 edges, from the D-Wave Advantage2 topology). D-Wave solves this natively via quantum annealing. QAOA is the gate-based alternative — it runs on IBM-style quantum hardware (or simulators) and uses a variational approach: build a parameterized quantum circuit, optimize its parameters classically, then sample solutions.

**Current limitation:** No existing gate-based hardware or simulator can handle the full 4,579-node graph. AerSimulator caps out at ~31 qubits due to exponential memory scaling. The baseline tool proves the solver works on subgraphs (8–28 nodes) — full-scale mining would require future hardware with thousands of qubits, or a protocol change to use smaller graphs.

## How It Works

### The Problem

Each mining attempt generates a random Ising problem on the protocol graph. A nonce (derived from the block header, miner ID, and a random salt) seeds a deterministic RNG that assigns field biases h ∈ {-1, 0, +1} to each node and couplings J ∈ {-1, +1} to each edge. The miner must find spin assignments (s = ±1 per node) that minimize the total energy:

```
E = Σ hᵢ·sᵢ + Σ Jᵢⱼ·sᵢ·sⱼ
```

Lower energy = better solution. The blockchain sets a difficulty threshold — solutions below that threshold count as valid. The miner needs enough valid, diverse solutions to earn the right to mine a block.

### QAOA Solve Pipeline

The solver takes an Ising problem (h, J) and returns solutions as a `dimod.SampleSet` in five steps:

1. **Build cost operator** — translate (h, J) into a quantum Hamiltonian (SparsePauliOp with Pauli-Z terms)
2. **Build QAOA circuit** — parameterized circuit with `p` layers, giving `2p` tunable angles
3. **Transpile** — compile the circuit for the target backend
4. **Optimize** — classical optimizer (COBYLA by default) repeatedly runs the circuit with different angles (~100 iterations), searching for angles that minimize expected energy
5. **Final sample** — run the optimized circuit with 4× shots to collect diverse low-energy solutions, convert to `dimod.SampleSet`

### Mining Loop

The miner wraps the solver in a loop:

1. Generate random salt → nonce → Ising problem (h, J)
2. Submit QAOA solve asynchronously
3. Poll for completion — check for stop signals (another miner won) and difficulty decay
4. Evaluate result — does it meet energy, diversity, and solution count requirements?
5. If valid → return `MiningResult`. If not → cache and retry.

### What Makes a Valid Mining Result

`evaluate_sampleset` checks three things:

1. **Energy threshold** — enough solutions below the difficulty target
2. **Diversity** — solutions must be sufficiently different (Hamming distance)
3. **Solution count** — enough unique valid solutions

All three must pass to produce a `MiningResult`.

## Limitations

IBM's current hardware — 156-qubit Heron R3 and 120-qubit Nighthawk — cannot run the full 4,579-node protocol graph. Neither can IonQ's 100-qubit Tempo (#AQ 64). For the IBM QAOA miner, AerSimulator is the only viable backend, and it's limited to subgraphs.

AerSimulator uses a statevector method that stores one complex amplitude for every possible quantum state. With n qubits, there are 2^n possible states — adding one qubit doubles the number of amplitudes, doubling the RAM. AerSimulator uses double precision (`complex128`, 16 bytes per amplitude), so the memory requirement is:

```
Memory = 2^n × 16 bytes
```

For example, 28 qubits: 2^28 × 16 = 4,294,967,296 bytes = 4 GB. Measured results match this precisely:

| Qubits | Amplitudes | Measured RAM | Theoretical SV | Solve time |
|--------|-----------|-------------|----------------|------------|
| 23     | 2^23 = 8M  | 288 MB      | 128 MB         | ~27s       |
| 26     | 2^26 = 64M | 1.2 GB      | 1 GB           | ~2.4 min   |
| 28     | 2^28 = 268M | 4.3 GB     | 4 GB           | ~6.9 min   |
| 31     | 2^31 = 2B  | 32.9 GB     | 32 GB          | ~42.5 min  |

Measured RAM includes ~153 MB of baseline overhead (Python, Qiskit imports). Subtracting that, the simulation memory matches the theoretical state vector size within 1%.

To verify memory scaling on your own machine:

```bash
python tools/ibm_qaoa_baseline.py --quick --subgraph-size 20
python tools/ibm_qaoa_baseline.py --quick --subgraph-size 23
python tools/ibm_qaoa_baseline.py --quick --subgraph-size 26
python tools/ibm_qaoa_baseline.py --quick --subgraph-size 28
```

Each 3-qubit jump should roughly 8× the state vector memory (2^3 = 8). At smaller qubit counts the ~153 MB baseline overhead makes the peak memory ratio appear lower, but at 28+ qubits the state vector dominates and the 8× scaling holds.

The default transpiler rejects circuits at ≥32 qubits. This may be bypassable with `set_max_qubits()` (untested). To check the default limit on your machine:

```bash
python -c "from qiskit_aer import AerSimulator; print(AerSimulator().num_qubits)"
```

The practical limit for consumer hardware (64 GB) is around 28–31 qubits with the default configuration. Possible ways to push further (not yet implemented):

- **Single precision** — `AerSimulator(precision='single')` uses `complex64` (8 bytes) instead of `complex128` (16 bytes), halving memory. 32 qubits would need ~32 GB instead of ~64 GB. Trades accuracy for reach.
- **GPU acceleration** — `AerSimulator(device='GPU')` offloads statevector computation to an NVIDIA GPU via CUDA. Limited by VRAM (e.g., 8 GB VRAM ≈ 28 qubits at double precision). Requires `qiskit-aer-gpu`.
- **Matrix Product State (MPS)** — `AerSimulator(method='matrix_product_state')` avoids storing the full 2^n statevector. Memory scales with entanglement, not qubit count. Can handle 50+ qubits for low-entanglement circuits, but QAOA circuits may have high entanglement, limiting the benefit.

The baseline tool uses subgraphs of 8–28 nodes to prove the code works. Real mining on the full topology would require future quantum hardware with thousands of qubits, or a protocol change to use smaller graphs.