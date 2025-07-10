# Quantum Proof-of-Work Blockchain Implementation

This project implements a quantum blockchain using quantum annealing for proof-of-work consensus. It features competitive mining between quantum computers (QPU) and classical simulated annealing (SA) with a dynamic difficulty adjustment mechanism.

## Overview

The blockchain demonstrates:
- **Quantum Annealing PoW**: Using Ising model optimization as the mining puzzle
- **Competitive Mining**: Multiple miners (QPU and SA) compete to mine blocks
- **Multi-Miner Support**: Configure any number of QPU and SA miners
- **Dynamic Difficulty**: Inverted difficulty mechanism that prevents miner monopolization
- **Streak Rewards**: Consecutive wins increase block rewards
- **Solution Diversity**: Requires multiple diverse solutions to prevent trivial mining
- **Individual Miner Tracking**: Each miner has unique ID and performance stats

## Setup

1. Create and activate a Python 3.13 virtual environment:
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up D-Wave API credentials (optional, for QPU access):
   ```bash
   echo "DWAVE_API_KEY=your_api_key_here" > .env
   ```

## Project Structure

```
quip-protocol/
├── quantum_blockchain.py       # Main blockchain implementation
├── reference/                  # Reference implementation tests
│   ├── test_quantum_pow.py    # Tests showing optimal SA parameters
│   └── reference_test_results.png
├── benchmarks/                 # Performance benchmarking suite
│   ├── benchmark_quantum_pow.py
│   ├── energy_distributions.png
│   ├── performance_metrics.png
│   ├── blockchain_benchmark_comprehensive.png
│   ├── blockchain_benchmark_timing.png
│   └── benchmark_results.json
└── venv/                      # Python virtual environment
```

## Usage

### Run the Quantum Blockchain Demo

```bash
# Basic competitive mining (1 QPU vs 1 SA miner)
python quantum_blockchain.py --competitive

# Multiple miners (2 QPU miners vs 4 SA miners)
python quantum_blockchain.py --competitive --num-qpu 2 --num-sa 4

# Custom number of blocks
python quantum_blockchain.py --competitive --num-qpu 2 --num-sa 3 --blocks 10

# Non-competitive mode (single miner)
python quantum_blockchain.py
```

Parameters:
- `--competitive`: Enable competitive mining mode
- `--num-qpu N`: Number of QPU miners (default: 1)
- `--num-sa N`: Number of SA miners (default: 1)
- `--blocks N`: Number of blocks to mine (default: 20)

### Run Reference Implementation Tests

```bash
python reference/test_quantum_pow.py
```

Tests the reference implementation with different `num_sweeps` values. Results show SA achieves optimal performance at `num_sweeps=4096`.

![Reference Test Results](reference/reference_test_results.png)

### Run Performance Benchmarks

```bash
python benchmarks/benchmark_quantum_pow.py
```

Generates comprehensive benchmarks comparing QPU vs SA performance:

![Performance Metrics](benchmarks/performance_metrics.png)
![Energy Distributions](benchmarks/energy_distributions.png)

## Quantum Proof-of-Work Mechanism

### Core Concepts

1. **Ising Model Generation**: Each block generates a unique Ising problem based on:
   - Block header hash
   - Mining nonce
   - Deterministic random seed

2. **Solution Requirements**:
   - **Energy Threshold**: Solutions must have energy < difficulty_energy
   - **Solution Diversity**: Multiple solutions with minimum Hamming distance
   - **Minimum Solutions**: At least N valid solutions required

3. **Mining Process**:
   - Miners iterate through nonces
   - For each nonce, sample the quantum annealer
   - Check if solutions meet all criteria
   - First miner to find valid solutions wins

### Dynamic Difficulty (Inverted Mechanism)

The blockchain implements an inverted difficulty adjustment:

```
Initial State: HARD (QPU-favored)
├── Energy: -1150
├── Diversity: 0.45
└── Solutions: 15

Consecutive Wins → EASIER
└── Reduces requirements progressively

New Winner → HARDER
└── Increases difficulty based on previous streak
```

This mechanism:
- Starts with QPU-favorable difficulty
- Makes mining easier for consecutive winners
- Immediately hardens when a new miner wins
- Prevents long-term monopolization

### Competitive Mining Results

The inverted difficulty mechanism produces balanced mining distribution:

![Blockchain Mining Results](benchmarks/blockchain_benchmark_comprehensive.png)

Key outcomes:
- **QPU**: ~70% of blocks (leverages quantum advantage initially)
- **SA**: ~30% of blocks (catches up as difficulty eases)
- **Streak Rewards**: Up to 5x multiplier for consecutive wins
- **Dynamic Balance**: Self-adjusting difficulty maintains competition

![Mining Time Analysis](benchmarks/blockchain_benchmark_timing.png)

## Technical Parameters

### Shared Mining Parameters
```python
base_difficulty_energy = -1150  # Energy threshold
min_diversity = 0.45           # Solution diversity requirement
min_solutions = 15             # Minimum valid solutions
```

### Miner-Specific Settings
- **QPU**: Uses D-Wave quantum processor (when available)
- **SA**: num_sweeps=4096 for optimal performance
- **Both**: 64 reads per mining attempt

### Difficulty Adjustment
```python
energy_adjustment_rate = 0.10  # 10% change per streak level
max_streak_multiplier = 5      # Maximum reward multiplier
```

## Key Features

1. **Decentralized Consensus**: All miners use identical difficulty parameters
2. **Quantum-Classical Competition**: Fair competition between QPU and SA
3. **Anti-Monopolization**: Dynamic difficulty prevents single miner dominance
4. **Performance Monitoring**: Comprehensive metrics and visualizations
5. **Solution Quality**: Enforces diversity to prevent trivial solutions

## GPU Benchmarking (Optional)

The project includes GPU benchmarking capabilities using Modal Labs for cost-effective cloud GPU access:

```bash
# Install Modal
pip install modal

# Authenticate (get $30/month free credits)
modal token new

# Run GPU benchmarks
modal run gpu_benchmark_modal.py
```

GPU options with Modal Labs:
- **T4 GPU**: ~$0.10/hour (good for testing)
- **A10G GPU**: ~$0.30/hour (balanced performance)
- **A100 GPU**: ~$1.00/hour (maximum performance)

The GPU benchmark compares simulated annealing performance across different GPU types against QPU results.

## Future Enhancements

- Network layer for distributed mining
- Persistent blockchain storage
- Transaction validation and smart contracts
- Multiple QPU support
- Advanced difficulty algorithms
- Real-time mining pool statistics
- Full GPU miner integration
- Mining pool protocol

## License

MIT License - See LICENSE file for details