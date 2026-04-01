# AGENTS.md — QuIP Protocol Project Instructions

Cross-tool instructions for AI coding assistants (Claude Code, Codex, Cursor, Gemini CLI).

## Environment Setup

```bash
python3 -m venv .quip
source .quip/bin/activate
pip install -U pip setuptools wheel
pip install -e .

# D-Wave credentials (optional, for QPU access)
# .env contains sensitive credentials — never read or display its contents
```

## Commands

### Running Network Nodes

```bash
# CPU node (bootstrap, standalone single-node mode)
quip-network-node cpu --listen 127.0.0.1 --port 8085 --public-host 127.0.0.1:8085 \
  --auto-mine --peer 127.0.0.1:8085 --genesis-config genesis_block_public.json

# GPU node (CUDA)
quip-network-node gpu --listen 127.0.0.1 --port 8085 --public-host 127.0.0.1:8085 \
  --auto-mine --peer 127.0.0.1:8085 --gpu-backend local --genesis-config genesis_block_public.json

# Mac Metal node
quip-network-node gpu --gpu-backend mps --listen 127.0.0.1 --port 8085 \
  --auto-mine --peer 127.0.0.1:8085 --genesis-config genesis_block_public.json

# TOML config (see QUIP-node.example.toml)
quip-network-node --config ./QUIP-node.example.toml

# Network simulator
quip-network-simulator --scenario mixed
quip-network-simulator --scenario cpu --base-port 9000
```

### Testing

```bash
# All tests
python -m pytest tests/ -v

# Single file / single test
python -m pytest tests/test_block_signer.py -v
python -m pytest tests/test_block_signer.py::test_sign_and_verify -v

# Smoke tests (run as scripts, not pytest)
python tests/smoke_node_cpu_only.py
python tests/smoke_node_gpu_metal.py   # Mac MPS
python tests/smoke_node_gpu_local.py   # CUDA
python tests/smoke_node_qpu.py         # D-Wave QPU
```

### Benchmarking and Tools

```bash
python tools/cpu_baseline.py --quick
python tools/cpu_baseline.py --quick \
  --topology dwave_topologies/topologies/advantage2_system1_13.json.gz
python tools/analyze_topology_sizes.py --configs "8,2" --samples 10
python tools/validate_mined_topology.py --all
python reference/test_quantum_pow.py
modal run benchmarks/gpu_benchmark_modal.py
```

## Architecture

### CLI Entry Points (`quip_cli.py`)

- `quip-network-node`: Run a single P2P node (cpu/gpu/qpu subcommands)
- `quip-network-simulator`: Launch multiple connected nodes for testing

### Shared Module (`shared/`)

**Core data structures:**
- `block.py`: Block, BlockHeader, MinerInfo, QuantumProof dataclasses with binary serialization
- `block_requirements.py`: BlockRequirements dataclass, `compute_current_requirements()` difficulty adjustment
- `miner_types.py`: MiningResult, IsingSample, Sampler protocol

**Mining & PoW:**
- `base_miner.py`: Abstract BaseMiner — template method pattern for `mine_block()`
- `quantum_proof_of_work.py`: Ising model generation (`generate_ising_model_from_nonce()`), diversity calculation, `evaluate_sampleset()`
- `miner_worker.py`: MinerHandle — 2-process mining orchestration
- `energy_utils.py`: Expected solution energy calculations, topology parameters
- `beta_schedule.py`: Temperature scheduling for simulated annealing

**P2P network:**
- `network_node.py`: Main NetworkNode class — P2P server with mining coordination
- `node.py`: Node state management and miner lifecycle
- `node_client.py` / `quic_client.py`: Async QUIC client (aioquic)
- `quic_server.py`: QUIC server implementation
- `quic_protocol.py`: QUIC message types and protocol definitions
- `block_store.py`: Persistent block storage
- `block_synchronizer.py`: Chain synchronization with peers

**Cryptography & trust:**
- `block_signer.py`: SPHINCS+ post-quantum signatures
- `certificate_manager.py`: TLS certificate management (auto-generated self-signed for dev)
- `trust_store.py`: Peer trust/reputation tracking

**Utilities:**
- `time_utils.py`: Network time synchronization
- `address_utils.py`: Host:port parsing
- `logging_config.py`: Unified logging setup
- `version.py`: Version and protocol version management
- `rest_api.py`: HTTP REST endpoints (legacy)

### Miner Implementations

```
BaseMiner (abstract, shared/base_miner.py)
├── SimulatedAnnealingMiner (CPU/sa_miner.py)
├── GPUMiner (GPU/metal_miner.py, GPU/cuda_miner.py, GPU/modal_miner.py)
└── DWaveMiner (QPU/dwave_miner.py)
```

**CPU** (`CPU/`): `sa_miner.py` (pure Python SA), `sa_sampler.py` (dimod-compatible sampler)

**GPU** (`GPU/`):
- Metal: `metal_miner.py`, `metal_sa.py`, `metal_gibbs_sa.py`, `metal_splash_sa.py` + `.metal` shaders
- CUDA: `cuda_miner.py`, `cuda_sa.py`, `cuda_kernel.py`
- Modal: `modal_miner.py`, `modal_sampler.py` (cloud GPU T4/A10G/A100)

**QPU** (`QPU/`): `dwave_miner.py`, `dwave_sampler.py`, `qpu_time_manager.py` (daily budget tracking)

### Topology Management (`dwave_topologies/`)

- `topologies/*.json.gz`: Pregenerated hardware topology files
- `embeddings/`: Precomputed embeddings for QPU hardware mapping
- `embedded_topology.py`, `embedding_loader.py`, `smart_embedding.py`
- Default topology: Zephyr Z(9,2) — 1,368 nodes, 7,692 edges

### Key Architecture Concepts

**Quantum Proof-of-Work:**
1. Each block generates a unique Ising problem from: previous block hash, miner ID, block index, and salt
2. Solutions must meet energy threshold, diversity requirement (Hamming distance), and minimum solution count
3. First miner to find valid solutions wins the block

**2-Process Mining Architecture** (`shared/miner_worker.py`):
```
Parent Process (Controller via MinerHandle)
├── Spawns child process for each mining attempt
├── Monitors stop_event every 100ms
└── Sends SIGTERM/SIGKILL to cancel mining

Child Process (Mining Worker)
├── Runs mine_block() in isolation
├── Handles SIGTERM for hardware cleanup
└── Returns MiningResult via IPC
```
Each miner must register a SIGTERM handler in `__init__()` for hardware-specific cleanup.

**P2P Network Protocol (QUIC):**
- QUIC with datagrams for low-latency, streams for larger blocks
- Built-in TLS 1.3 encryption (self-signed certs for dev)
- Message types: JOIN, HEARTBEAT, GOSSIP, BLOCK, STATUS, STATS
- Heartbeat: 15s interval, 60s timeout
- Default port: 20049, ALPN: "quip-v1"

**Difficulty Adjustment** (`block_requirements.py`):
- `compute_current_requirements()` adjusts energy threshold based on mining time
- Hardening (fast blocks): 35%±30%; Easing (slow blocks): 15%±14%

**TOML Configuration** (`--config`):
- `[global]`: listen, port, node_name, secret, auto_mine, peer list, heartbeat, log_level
- `[cpu]`: num_cpus
- `[gpu]`: backend (local/modal), devices, types
- `[qpu]`: dwave_api_key, dwave_api_solver, dwave_region_url

### Critical Parameters

**Genesis block** (`genesis_block_public.json`):
```python
difficulty_energy = -2500.0   # Genesis default (relaxed)
min_diversity = 0.2
min_solutions = 5
h_values = [-1.0, 0.0, 1.0]  # Ternary Ising distribution
```

**Production Z(9,2) targets:**
```python
difficulty_energy = -4100.0
min_diversity = 0.15
min_solutions = 5
```

**Energy Ranges by Topology (GSE):**
- Z(8,2): -2869 to -2677 (192 unit range)
- Z(9,2): -4100 to -3870 (230 unit range) — DEFAULT
- Z(10,2): -5470 to -5200 (270 unit range)
- Z(11,4): -15170 to -14158 (1012 unit range)

**Miner-Specific Configurations:**
- CPU/SA: num_sweeps=64–4096, reads=64–512 (adaptive)
- GPU/CUDA: num_sweeps=256–2048
- GPU/Metal: num_sweeps=64–512
- Modal: num_sweeps=128–4096
- QPU: annealing_time=5–20μs, reads=32–64

## Code Style

**Imports:** All at top of file (no inline imports). Order: stdlib → third-party → local. Absolute imports only. Exception: optional dependency try/except at module level; Modal remote functions.

**Concurrency:** NEVER use threads — multiprocessing only. Async via asyncio for network operations. Mining runs in separate processes.

**Dependencies:**
- Core: dwave-ocean-sdk, numpy, aioquic, click, blake3, hashsigs (SPHINCS+), cryptography
- GPU: torch, cupy (Linux/Windows), pyobjc-Metal (macOS)
- Optional: modal (cloud GPU)
- Python 3.10+

## CI/CD and Deployment

**GitLab CI** (`.gitlab-ci.yml`): Builds Docker CPU and CUDA images on main/tags

**Docker** (`docker/`): Dockerfile.cpu, Dockerfile.cuda, docker-compose.yml, entrypoint.sh

**Systemd** (`quip-network-node.service`): Production deployment, config at `/etc/quip.network/config.toml`

**Cloud**: `akash/` and `aws/` directories for cloud deployment configs

## QPU Solver Updates

When D-Wave updates a solver to a new revision:
1. Download the new topology file and replace the old one in `topologies/`
2. Check if existing embeddings in `embeddings/` are still compatible
3. Delete all references to the old solver revision (stale topology files, incompatible embeddings)
