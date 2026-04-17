# IonQ QAOA Miner

QAOA-based quantum miner for the QUIP proof-of-work protocol using IonQ's trapped-ion quantum hardware. Runs on Qiskit's AerSimulator (local), IonQ's cloud simulator, or IonQ QPU hardware.

## Setup

```bash
# Install project (registers quip-network-node on PATH)
pip install -e .

# Install IonQ QAOA dependencies
pip install qiskit qiskit-aer qiskit-ionq dimod numpy scipy psutil

# If quip-network-node is not recognized after install, add Scripts to PATH:
# PowerShell:
$env:PATH += ";C:\Users\<you>\AppData\Local\Python\<version>\Scripts"
```

### IonQ Cloud Access (optional)

For IonQ cloud simulator or real hardware, you need an API key:

1. Create a free account at [cloud.ionq.com](https://cloud.ionq.com)
2. Go to API Keys under your profile and generate a key
3. Save the key immediately — it can only be viewed once
4. Set it as an environment variable:

```bash
# PowerShell
$env:IONQ_API_KEY = "your-key-here"

# Linux/macOS
export IONQ_API_KEY="your-key-here"
```

The free account includes unlimited access to IonQ's cloud simulator (up to 29 qubits). QPU access requires credits.

## Usage

### Baseline Testing (no node infrastructure needed)

```bash
# Quick smoke test — 8-node subgraph, AerSimulator, ~1 min
python tools/ionq_qaoa_baseline.py --quick -v

# Same test on IonQ cloud simulator (~12 min due to network overhead)
python tools/ionq_qaoa_baseline.py --quick -v --ionq-backend ionq_simulator

# Solution quality check against known optimal energies
python tools/ionq_qaoa_baseline.py --known-problems -v

# Standard test — 10-node subgraph, parameter sweep
python tools/ionq_qaoa_baseline.py -v

# Memory stress test — 28-node subgraph, ~4 GB RAM
python tools/ionq_qaoa_baseline.py --stress -v
```

### Running a Node

```bash
# AerSimulator (default, no API key needed)
quip-network-node --config quip.network.ionq_qaoa.example.toml ionq_qaoa --auto-mine

# IonQ cloud simulator
quip-network-node --config quip.network.ionq_qaoa.example.toml ionq_qaoa --backend ionq --ionq-backend-name ionq_simulator

# IonQ real hardware
quip-network-node --config quip.network.ionq_qaoa.example.toml ionq_qaoa --backend ionq --ionq-api-token TOKEN --ionq-backend-name ionq_qpu

# With custom QAOA parameters
quip-network-node --config quip.network.ionq_qaoa.example.toml ionq_qaoa --auto-mine --subgraph-size 10 --p 2 --optimizer SPSA --shots 512
```

### TOML Configuration

See `quip.network.ionq_qaoa.example.toml` for all options. Key settings:

```toml
[ionq_qaoa]
backend = "aer"                        # "aer" (local) or "ionq" (IonQ cloud)
subgraph_size = 14                     # nodes extracted from protocol topology
p = 1                                  # QAOA circuit depth
optimizer = "COBYLA"                   # COBYLA, NELDER_MEAD, POWELL, L_BFGS_B, SPSA
shots = 1024                           # shots per optimizer evaluation

# IonQ cloud (only when backend = "ionq")
# ionq_api_token = "your-token"
# ionq_backend_name = "ionq_simulator"  # or "ionq_qpu" for real hardware
```

## Why QAOA on IonQ

The protocol's proof-of-work is an Ising optimization problem defined on a fixed graph (4,579 nodes, 41,549 edges, from the D-Wave Advantage2 topology). D-Wave solves this natively via quantum annealing. QAOA is the gate-based alternative — it builds a parameterized quantum circuit, optimizes its parameters classically, then samples solutions.

IonQ's trapped-ion architecture offers some advantages for QAOA: all-to-all qubit connectivity (no SWAP overhead for arbitrary interactions), high gate fidelities, and long coherence times. The trade-off is slower gate speeds compared to superconducting qubits.

**Current limitation:** No existing gate-based hardware can handle the full 4,579-node graph. IonQ's largest system (Forte, #AQ 36) has far fewer qubits than needed. The IonQ cloud simulator supports up to 29 qubits. The miner uses BFS subgraph extraction to mine on smaller subgraphs (8–28 nodes).

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
3. **Transpile** — compile the circuit for the target backend (local transpilation for AerSimulator, IonQ handles its own transpilation server-side for cloud backends)
4. **Optimize** — classical optimizer (COBYLA by default) repeatedly runs the circuit with different angles (~100 iterations), searching for angles that minimize expected energy
5. **Final sample** — run the optimized circuit with 4× shots to collect diverse low-energy solutions, convert to `dimod.SampleSet`

### Mining Loop

The miner wraps the solver in a loop:

1. Generate random salt → nonce → Ising problem (h, J)
2. Submit QAOA solve asynchronously (multiprocessing)
3. Poll for completion — check for stop signals (another miner won) and difficulty decay
4. Evaluate result — does it meet energy, diversity, and solution count requirements?
5. If valid → return `MiningResult`. If not → cache and retry.

### What Makes a Valid Mining Result

`evaluate_sampleset` checks three things:

1. **Energy threshold** — enough solutions below the difficulty target
2. **Diversity** — solutions must be sufficiently different (Hamming distance)
3. **Solution count** — enough unique valid solutions

All three must pass to produce a `MiningResult`.

## Backends

### AerSimulator (default)

Local Qiskit simulator. No API key, no network calls, fast. Same simulator used by the IBM QAOA miner. Good for development and testing.

### IonQ Cloud Simulator (`ionq_simulator`)

IonQ's cloud-hosted simulator accessed via their REST API. Free with a free IonQ account, supports up to 29 qubits, includes hardware noise models for Aria and Forte. Each circuit evaluation is a network round-trip, so solves are significantly slower than local AerSimulator.

### IonQ QPU (`ionq_qpu`)

Real trapped-ion hardware (Aria #AQ 25, Forte #AQ 36). Requires IonQ credits. Same API, same code path — just swap the backend name.

## Limitations

IonQ's current hardware — Aria (#AQ 25) and Forte (#AQ 36) — cannot run the full 4,579-node protocol graph. The IonQ cloud simulator supports up to 29 qubits. AerSimulator caps out at ~31 qubits due to exponential memory scaling.




