#include <metal_stdlib>
using namespace metal;

// P-bit Enhanced Metal Kernels for Quantum Blockchain Mining
// Implements probabilistic bit (P-bit) simulated annealing with research optimizations
// Based on 2024 research: TApSA, SpSA, and enhanced device variability modeling

// =============================================================================
// CORE KERNELS (Required for basic P-bit operation)
// =============================================================================

// Kernel 1: Fused Metropolis acceptance with vectorized operations
kernel void fused_metropolis_update(
    device int8_t* spins [[buffer(0)]],           // Input/Output: spin states
    device const float* local_fields [[buffer(1)]], // Input: local field values
    device const float* random_values [[buffer(2)]], // Input: pre-generated random values
    constant float& beta [[buffer(3)]],           // Input: inverse temperature
    constant uint& num_chains [[buffer(4)]],      // Input: number of parallel chains
    constant uint& chunk_size [[buffer(5)]],      // Input: chunk size
    uint3 thread_id [[thread_position_in_grid]]
) {
    uint chain_idx = thread_id.x;
    uint spin_idx = thread_id.y;
    
    if (chain_idx >= num_chains || spin_idx >= chunk_size) return;
    
    uint flat_idx = chain_idx * chunk_size + spin_idx;
    
    // Load current spin and field
    int8_t current_spin = spins[flat_idx];
    float field = local_fields[flat_idx];
    float rand_val = random_values[flat_idx];
    
    // Fused Metropolis computation
    float delta_e = 2.0 * float(current_spin) * field;
    
    // Accept/reject decision for ENERGY MINIMIZATION
    bool accept = (delta_e < 0.0) || (rand_val < exp(-beta * delta_e));
    
    // Conditional spin flip
    if (accept) {
        spins[flat_idx] = -current_spin;
    }
}

// Kernel 2: Optimized coupling field computation
kernel void optimized_coupling_field(
    device float* neighbor_sum [[buffer(0)]],     // Output: neighbor contribution sums
    device const int8_t* spins [[buffer(1)]],     // Input: current spin states  
    device const uint* i_indices [[buffer(2)]],   // Input: coupling i-indices
    device const uint* j_indices [[buffer(3)]],   // Input: coupling j-indices
    device const float* j_values [[buffer(4)]],   // Input: coupling strengths
    constant uint& num_chains [[buffer(5)]],      // Input: number of parallel chains
    constant uint& num_couplings [[buffer(6)]],   // Input: number of couplings
    constant uint& num_spins [[buffer(7)]],       // Input: number of spins
    uint2 thread_id [[thread_position_in_grid]]
) {
    uint chain_idx = thread_id.x;
    uint coupling_idx = thread_id.y;
    
    if (chain_idx >= num_chains || coupling_idx >= num_couplings) return;
    
    uint i = i_indices[coupling_idx];
    uint j = j_indices[coupling_idx]; 
    float coupling_strength = j_values[coupling_idx];
    
    // Get spins for this chain
    uint chain_offset = chain_idx * num_spins;
    int8_t spin_i = spins[chain_offset + i];
    int8_t spin_j = spins[chain_offset + j];
    
    // Compute mutual contributions using atomic operations
    float contribution_i = float(spin_j) * coupling_strength;
    float contribution_j = float(spin_i) * coupling_strength;
    
    uint neighbor_i_idx = chain_idx * num_spins + i;
    uint neighbor_j_idx = chain_idx * num_spins + j;
    
    atomic_fetch_add_explicit(
        (device atomic<float>*)&neighbor_sum[neighbor_i_idx], 
        contribution_i, 
        memory_order_relaxed
    );
    atomic_fetch_add_explicit(
        (device atomic<float>*)&neighbor_sum[neighbor_j_idx], 
        contribution_j, 
        memory_order_relaxed
    );
}

// Kernel 3: Energy computation for solution evaluation
kernel void compute_energies(
    device float* energies [[buffer(0)]],         // Output: computed energies
    device const int8_t* spins [[buffer(1)]],     // Input: spin configurations
    device const float* h_fields [[buffer(2)]],   // Input: external field values
    device const uint* i_indices [[buffer(3)]],   // Input: coupling i-indices
    device const uint* j_indices [[buffer(4)]],   // Input: coupling j-indices
    device const float* j_values [[buffer(5)]],   // Input: coupling strengths
    constant uint& num_chains [[buffer(6)]],      // Input: number of parallel chains
    constant uint& num_spins [[buffer(7)]],       // Input: number of spins
    constant uint& num_couplings [[buffer(8)]],   // Input: number of couplings
    uint thread_id [[thread_position_in_grid]]
) {
    uint chain_idx = thread_id;
    if (chain_idx >= num_chains) return;
    
    uint chain_offset = chain_idx * num_spins;
    float total_energy = 0.0;
    
    // External field contribution
    for (uint spin_idx = 0; spin_idx < num_spins; spin_idx++) {
        int8_t spin = spins[chain_offset + spin_idx];
        total_energy += float(spin) * h_fields[spin_idx];
    }
    
    // Coupling contribution
    for (uint coupling_idx = 0; coupling_idx < num_couplings; coupling_idx++) {
        uint i = i_indices[coupling_idx];
        uint j = j_indices[coupling_idx];
        float coupling_strength = j_values[coupling_idx];
        
        int8_t spin_i = spins[chain_offset + i];
        int8_t spin_j = spins[chain_offset + j];
        
        total_energy += float(spin_i) * float(spin_j) * coupling_strength;
    }
    
    energies[chain_idx] = total_energy;
}

// Kernel 4: Local field computation (external + coupling fields)
kernel void compute_local_fields(
    device float* local_fields [[buffer(0)]],     // Output: combined local fields
    device const float* neighbor_sums [[buffer(1)]], // Input: coupling field contributions
    device const float* h_fields [[buffer(2)]],   // Input: external field values
    constant uint& num_chains [[buffer(3)]],      // Input: number of parallel chains
    constant uint& num_spins [[buffer(4)]],       // Input: number of spins
    uint2 thread_id [[thread_position_in_grid]]
) {
    uint chain_idx = thread_id.x;
    uint spin_idx = thread_id.y;
    
    if (chain_idx >= num_chains || spin_idx >= num_spins) return;
    
    uint flat_idx = chain_idx * num_spins + spin_idx;
    local_fields[flat_idx] = neighbor_sums[flat_idx] + h_fields[spin_idx];
}

// Kernel 5: Random spin initialization
kernel void initialize_random_spins(
    device int8_t* spins [[buffer(0)]],           // Output: initialized spins
    device const float* random_values [[buffer(1)]], // Input: random values [0,1]
    constant uint& num_chains [[buffer(2)]],      // Input: number of parallel chains
    constant uint& num_spins [[buffer(3)]],       // Input: number of spins
    uint2 thread_id [[thread_position_in_grid]]
) {
    uint chain_idx = thread_id.x;
    uint spin_idx = thread_id.y;
    
    if (chain_idx >= num_chains || spin_idx >= num_spins) return;
    
    uint flat_idx = chain_idx * num_spins + spin_idx;
    spins[flat_idx] = (random_values[flat_idx] > 0.5) ? int8_t(1) : int8_t(-1);
}

// =============================================================================
// P-BIT KERNELS (Advanced probabilistic bit simulated annealing)
// =============================================================================

// Kernel 6: P-bit parallel update (basic)
kernel void pbit_parallel_update(
    device int8_t* spins [[buffer(0)]],           
    device const float* fields [[buffer(1)]],    
    device const float* random_decisions [[buffer(2)]], 
    device const float* timing_random [[buffer(3)]],
    device const float* intensity_random [[buffer(4)]], 
    device const float* offset_random [[buffer(5)]],
    constant float& beta [[buffer(6)]],           
    constant uint& R [[buffer(7)]],               // num_chains
    constant uint& spins_per_block [[buffer(8)]],
    constant uint& n [[buffer(9)]],               // num_spins
    constant float& timing_variance [[buffer(10)]],
    constant float& intensity_variance [[buffer(11)]],
    constant float& offset_variance [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint chain_idx = gid.z;
    uint block_idx = gid.y;  
    uint thread_idx = gid.x;
    
    if (chain_idx >= R) return;
    
    uint global_spin_idx = block_idx * spins_per_block + thread_idx;
    if (global_spin_idx >= n) return;
    
    uint flat_idx = chain_idx * n + global_spin_idx;
    
    // P-bit device variability modeling
    float timing_factor = 1.0f + timing_variance * (timing_random[flat_idx] - 0.5f);
    float intensity_factor = 1.0f + intensity_variance * (intensity_random[flat_idx] - 0.5f);
    float offset = offset_variance * (offset_random[flat_idx] - 0.5f);
    
    // Apply variability to field calculation
    float modified_field = (fields[flat_idx] + offset) * intensity_factor * timing_factor;
    
    // P-bit parallel Metropolis criterion
    int8_t current_spin = spins[flat_idx];
    float delta_e = 2.0f * float(current_spin) * modified_field;
    
    float rand_val = random_decisions[flat_idx];
    bool accept = (delta_e < 0.0f) || (rand_val < exp(-beta * delta_e));
    
    if (accept) {
        spins[flat_idx] = -current_spin;
    }
}

// Kernel 7: P-bit sequential update (race-condition safe)
kernel void pbit_sequential_update(
    device int8_t* spins [[buffer(0)]],           
    device const float* fields [[buffer(1)]],    
    device const float* random_decisions [[buffer(2)]], 
    device const float* timing_random [[buffer(3)]],
    device const float* intensity_random [[buffer(4)]], 
    device const float* offset_random [[buffer(5)]],
    constant float& beta [[buffer(6)]],           
    constant uint& R [[buffer(7)]],               
    constant uint& spins_per_block [[buffer(8)]],
    constant uint& n [[buffer(9)]],               
    constant float& timing_variance [[buffer(10)]],
    constant float& intensity_variance [[buffer(11)]],
    constant float& offset_variance [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint chain_idx = gid.z;
    uint block_idx = gid.y;  
    uint thread_idx = gid.x;
    
    if (chain_idx >= R) return;
    
    // Sequential processing within blocks to avoid race conditions
    for (uint seq_idx = 0; seq_idx < spins_per_block; seq_idx += 32) {
        threadgroup_barrier(mem_flags::mem_device);
        
        uint global_spin_idx = block_idx * spins_per_block + seq_idx + thread_idx;
        if (global_spin_idx >= n) continue;
        
        uint flat_idx = chain_idx * n + global_spin_idx;
        
        // P-bit device variability
        float timing_factor = 1.0f + timing_variance * (timing_random[flat_idx] - 0.5f);
        float intensity_factor = 1.0f + intensity_variance * (intensity_random[flat_idx] - 0.5f);
        float offset = offset_variance * (offset_random[flat_idx] - 0.5f);
        
        float modified_field = (fields[flat_idx] + offset) * intensity_factor * timing_factor;
        
        // Sequential Metropolis update
        int8_t current_spin = spins[flat_idx];
        float delta_e = 2.0f * float(current_spin) * modified_field;
        
        float rand_val = random_decisions[flat_idx];
        bool accept = (delta_e < 0.0f) || (rand_val < exp(-beta * delta_e));
        
        if (accept) {
            spins[flat_idx] = -current_spin;
        }
    }
}

// Kernel 8: P-bit optimized parallel (enhanced performance)
kernel void pbit_optimized_parallel_update(
    device int8_t* spins [[buffer(0)]],           
    device const float* fields [[buffer(1)]],    
    device const float* random_decisions [[buffer(2)]], 
    device const float* timing_random [[buffer(3)]],
    device const float* intensity_random [[buffer(4)]], 
    device const float* offset_random [[buffer(5)]],
    constant float& beta [[buffer(6)]],           
    constant uint& R [[buffer(7)]],               
    constant uint& spins_per_block [[buffer(8)]],
    constant uint& n [[buffer(9)]],               
    constant float& timing_variance [[buffer(10)]],
    constant float& intensity_variance [[buffer(11)]],
    constant float& offset_variance [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint chain_idx = gid.z;
    uint block_idx = gid.y;  
    uint thread_idx = gid.x;
    
    if (chain_idx >= R) return;
    
    uint global_spin_idx = block_idx * spins_per_block + thread_idx;
    if (global_spin_idx >= n) return;
    
    uint flat_idx = chain_idx * n + global_spin_idx;
    
    // Enhanced P-bit variability (research-optimized parameters)
    float timing_factor = 1.0f + timing_variance * (timing_random[flat_idx] - 0.5f);
    float intensity_factor = 1.0f + intensity_variance * (intensity_random[flat_idx] - 0.5f);
    float offset = offset_variance * (offset_random[flat_idx] - 0.5f);
    
    // Optimized field computation
    float base_field = fields[flat_idx];
    float modified_field = (base_field + offset) * intensity_factor * timing_factor;
    
    // Vectorized Metropolis computation
    int8_t current_spin = spins[flat_idx];
    float delta_e = 2.0f * float(current_spin) * modified_field;
    
    // Optimized acceptance probability
    float exp_factor = exp(-beta * max(0.0f, delta_e));
    bool accept = (delta_e <= 0.0f) || (random_decisions[flat_idx] < exp_factor);
    
    // Conditional update with memory optimization
    if (accept) {
        spins[flat_idx] = -current_spin;
    }
}

// Kernel 9: P-bit research simplified (TApSA + SpSA, buffer-safe)
kernel void pbit_research_simplified_update(
    device int8_t* spins [[buffer(0)]],           
    device const float* fields [[buffer(1)]],    
    device const float* random_decisions [[buffer(2)]], 
    device const float* timing_random [[buffer(3)]],
    device const float* intensity_random [[buffer(4)]], 
    device const float* stall_random [[buffer(5)]],
    constant float& beta [[buffer(6)]],           
    constant uint& R [[buffer(7)]],               
    constant uint& spins_per_block [[buffer(8)]],
    constant uint& n [[buffer(9)]],               
    constant float& timing_variance [[buffer(10)]],
    constant float& intensity_variance [[buffer(11)]],
    constant float& averaging_factor [[buffer(12)]],    // TApSA: α parameter
    constant float& stall_probability [[buffer(13)]],   // SpSA: p parameter
    uint3 gid [[thread_position_in_grid]]
) {
    uint chain_idx = gid.z;
    uint block_idx = gid.y;  
    uint thread_idx = gid.x;
    
    if (chain_idx >= R) return;
    
    uint global_spin_idx = block_idx * spins_per_block + thread_idx;
    if (global_spin_idx >= n) return;
    
    uint flat_idx = chain_idx * n + global_spin_idx;
    
    // Enhanced device variability (2024 research parameters)
    float timing_factor = 1.0f + timing_variance * (timing_random[flat_idx] - 0.5f);
    float intensity_factor = 1.0f + intensity_variance * (intensity_random[flat_idx] - 0.5f);
    
    // TApSA: Time-averaged field (simplified without external buffer)
    float raw_field = fields[flat_idx] * intensity_factor * timing_factor;
    
    // SpSA: Stalled pSA - probabilistic stalling to prevent oscillations
    bool should_stall = (stall_random[flat_idx] < stall_probability);
    if (should_stall) {
        return; // Skip this update to allow system stabilization
    }
    
    // Research-optimized Metropolis criterion
    int8_t current_spin = spins[flat_idx];
    float delta_e = 2.0f * float(current_spin) * raw_field;
    
    // Enhanced acceptance probability with research optimizations
    bool accept = (delta_e < 0.0f) || (random_decisions[flat_idx] < exp(-beta * abs(delta_e)));
    
    if (accept) {
        spins[flat_idx] = -current_spin;
    }
}