# Metal P-bit Implementation for Quantum Blockchain Mining

This document describes the Metal-accelerated P-bit (Probabilistic Bit) implementation used for quantum blockchain mining on Apple Silicon devices.

## Overview

The Metal P-bit implementation provides native GPU acceleration for simulated annealing optimization using Apple's Metal compute framework. This replaces the previous PyTorch MPS implementation with a more efficient, lower-level approach specifically optimized for P-bit quantum annealing algorithms.

### Key Features

- **Native Metal Kernels**: Direct Metal compute shaders for maximum performance
- **P-bit Algorithm Support**: Implements probabilistic bit simulated annealing with device variability modeling
- **Energy Minimization**: Optimized for finding low-energy solutions for Ising model problems

## P-bit Algorithm Background

### What are P-bits?

P-bits (Probabilistic Bits) are computational units that can be in a probabilistic superposition of states {-1, +1}, unlike classical bits that are deterministic. 

### Algorithm Variants

The implementation supports multiple P-bit algorithm variants:

- **pbit_parallel_update**: Basic parallel P-bit updates
- **pbit_sequential_update**: Sequential updates to avoid race conditions
- **pbit_optimized_parallel_update**: Optimized parallel version with better performance
- **pbit_research_simplified_update**: Simplified research version with TApSA and SpSA

## Technical Implementation

### Metal Kernels

The implementation uses 9 Metal compute kernels:

#### Core Kernels (Required for basic operation)
1. **fused_metropolis_update**: Vectorized Metropolis acceptance criterion
2. **optimized_coupling_field**: Efficient computation of coupling field contributions
3. **compute_energies**: Final energy calculation for solution evaluation
4. **compute_local_fields**: Combines external and coupling field contributions
5. **initialize_random_spins**: Random initialization of spin configurations

#### P-bit Kernels (Advanced probabilistic algorithms)
6. **pbit_parallel_update**: Basic P-bit parallel updates with device variability
7. **pbit_sequential_update**: Race-condition safe sequential updates
8. **pbit_optimized_parallel_update**: Enhanced performance parallel version
9. **pbit_research_simplified_update**: Research-optimized with TApSA/SpSA

### Performance Characteristics

- **Energy Range**: Typically achieves energies in the range -400 to -2000 (vs CPU baseline ~-15000)
- **Speed**: Optimized for fast convergence with 64-256 sweeps (vs CPU 4096+ sweeps)
- **Strategy**: Multiple fast attempts rather than single slow convergence
- **Memory**: Efficient Metal buffer management with minimal GPU-CPU transfers

### Device Variability Parameters

The P-bit implementation models realistic hardware variations:

```metal
float timing_variance = 0.1;      // P-bit timing variability (±10%)
float intensity_variance = 0.05;  // P-bit intensity variability (±5%)
float offset_variance = 0.02;     // P-bit offset variability (±2%)
```

These parameters are based on experimental measurements of actual P-bit hardware devices.

## Usage

### Basic Usage

```python
from GPU.metal_kernel_sampler import MetalKernelDimodSampler

# Initialize the sampler
sampler = MetalKernelDimodSampler("mps")

# Run P-bit sampling
sampleset = sampler.sample_ising(
    h=h_fields,
    J=coupling_dict,
    num_reads=64,
    num_sweeps=128
)
```

### Mining Integration

The Metal P-bit sampler is automatically used by the Metal miner:

```python
from GPU.metal_miner import MetalMiner

# Creates a miner using P-bit Metal acceleration
miner = MetalMiner(miner_id="metal-1")
```

### Benchmark Testing

Use the baseline testing tool to evaluate P-bit performance:

```bash
python tools/metal_baseline.py --timeout 10 --output results.json
```

## Kernel Implementation Details

### Energy Minimization Strategy

The P-bit kernels implement energy minimization through the Metropolis criterion:

```metal
// P-bit Metropolis acceptance (energy minimization)
float delta_e = 2.0f * float(current_spin) * modified_field;
bool accept = (delta_e < 0.0f) || (random_val < exp(-beta * delta_e));
```

Key points:
- `delta_e < 0` always accepts moves that decrease energy
- Probabilistic acceptance for energy-increasing moves based on temperature
- Modified fields incorporate device variability for realistic P-bit behavior

### Memory Optimization

- Uses `int8` for spin storage (compact {-1, +1} representation)
- `float32` for all field calculations (Metal-optimized precision)
- Minimal CPU-GPU transfers (only at start and end)
- Atomic operations for coupling field accumulation

## Architecture Benefits

### vs. CPU Implementation
- **Parallel Processing**: Simultaneous updates across multiple spin chains
- **GPU Memory Bandwidth**: Fast field computations and energy evaluations
- **Vectorized Operations**: Metal's SIMD capabilities for bulk operations

## Debugging and Diagnostics

### Solution Validation

The implementation includes built-in validation:

```python
# Validate P-bit solutions
valid = sampler.validate_solutions(samples, energies, h, J)
if not valid:
    print("❌ P-BIT SOLUTION FORMAT CORRUPTED")
```

### Logging and Timing

Detailed timing breakdown available with debug logging:
- Buffer creation time
- Kernel execution time per sweep
- Energy computation time
- Memory transfer overhead


## References and Further Reading

### Academic Papers
- [Probabilistic computing with p-bits](https://arxiv.org/abs/2108.09836)
- [https://ewh.ieee.org/r6/oeb/mag/KCamsari_abstract.pdf](https://ewh.ieee.org/r6/oeb/mag/KCamsari_abstract.pdf)
- [https://www.nature.com/articles/s41467-024-48152-0](https://www.nature.com/articles/s41467-024-48152-0) 
- [Accelerating Simulated Quantum Annealing with GPU and Tensor Cores](https://yi-huaaa.github.io/2022/05/10/Accelerating%20Simulated%20Quantum%20Annealing%20with%20GPU%20and%20Tensor%20Cores/) -- NOTE: this is the most useful resource.

### Technical Documentation
- [Apple Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [Quantum Annealing Theory](https://arxiv.org/abs/quant-ph/0001106)

### Implementation References
- [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/) - Quantum annealing reference
- [D-Wave Topology/Graph Structure](https://docs.dwavesys.com/docs/latest/c_gs_4.html) - Problem topology
- [Simulated Annealing Algorithms](https://en.wikipedia.org/wiki/Simulated_annealing) - Classical optimization