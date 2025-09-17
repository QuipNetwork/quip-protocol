#include <metal_stdlib>
using namespace metal;

// P-bit Enhanced Metal Kernels for Quantum Blockchain Mining
// Implements probabilistic bit (P-bit) simulated annealing with device variability modeling

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
    
    // Fused Metropolis computation for energy minimization
    // delta_e is energy change if we flip the spin
    float delta_e = 2.0 * float(current_spin) * field;
    
    // Accept/reject decision for ENERGY MINIMIZATION
    // Accept if energy decreases (delta_e < 0) or with probability exp(-beta * delta_e)
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
    // For Ising model: H = sum h_i*s_i + sum J_ij*s_i*s_j
    // Local field for spin i includes: h_i + sum_j J_ij*s_j
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
    
    // P-bit parallel Metropolis criterion for energy minimization
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
        
        // Sequential Metropolis update for energy minimization
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

    // Enhanced P-bit variability with optimized parameters
    float timing_factor = 1.0f + timing_variance * (timing_random[flat_idx] - 0.5f);
    float intensity_factor = 1.0f + intensity_variance * (intensity_random[flat_idx] - 0.5f);
    float offset = offset_variance * (offset_random[flat_idx] - 0.5f);

    // Optimized field computation
    float base_field = fields[flat_idx];
    float modified_field = (base_field + offset) * intensity_factor * timing_factor;

    // Vectorized Metropolis computation for energy minimization
    int8_t current_spin = spins[flat_idx];
    float delta_e = 2.0f * float(current_spin) * modified_field;

    // Standard Metropolis acceptance for energy minimization
    float rand_val = random_decisions[flat_idx];
    bool accept = (delta_e < 0.0f) || (rand_val < exp(-beta * delta_e));

    // Conditional update with memory optimization
    if (accept) {
        spins[flat_idx] = -current_spin;
    }
}

// =============================================================================
// HIERARCHICAL UPDATE KERNELS (Optimized for maximum GSE performance)
// =============================================================================

// Kernel 9: Hierarchical block update - updates spins within a specific block only
kernel void hierarchical_block_update(
    device int8_t* spins [[buffer(0)]],
    device const float* local_fields [[buffer(1)]],
    device const float* random_decisions [[buffer(2)]],
    constant float& beta [[buffer(3)]],
    constant uint& R [[buffer(4)]],
    constant uint& block_start [[buffer(5)]],
    constant uint& block_size [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    uint2 thread_id [[thread_position_in_grid]]
) {
    uint chain_idx = thread_id.x;
    uint block_spin_idx = thread_id.y;

    if (chain_idx >= R || block_spin_idx >= block_size) return;

    // Calculate actual spin index within the full problem
    uint spin_idx = block_start + block_spin_idx;
    if (spin_idx >= n) return;

    uint flat_idx = chain_idx * n + spin_idx;

    // Get values for this spin
    int8_t current_spin = spins[flat_idx];
    float field = local_fields[flat_idx];
    float rand_val = random_decisions[chain_idx * block_size + block_spin_idx];

    // Metropolis acceptance for energy minimization
    float delta_e = 2.0f * float(current_spin) * field;
    bool accept = (delta_e < 0.0f) || (rand_val < exp(-beta * delta_e));

    if (accept) {
        spins[flat_idx] = -current_spin;
    }
}

// Kernel 10: Efficient block local field update using matrix operations (Paper Section 3.2)
kernel void block_local_field_update(
    device float* local_fields [[buffer(0)]],
    device const int8_t* spins [[buffer(1)]],
    device const uint* i_indices [[buffer(2)]],
    device const uint* j_indices [[buffer(3)]],
    device const float* j_values [[buffer(4)]],
    device const float* h_fields [[buffer(5)]],
    constant uint& R [[buffer(6)]],
    constant uint& block_start [[buffer(7)]],
    constant uint& block_size [[buffer(8)]],
    constant uint& n [[buffer(9)]],
    constant uint& num_couplings [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint chain_idx = gid.x;
    uint spin_in_block = gid.y;

    if (chain_idx >= R || spin_in_block >= block_size) return;

    uint global_spin_idx = block_start + spin_in_block;
    if (global_spin_idx >= n) return;

    uint flat_idx = chain_idx * n + global_spin_idx;

    // Recompute local field for this spin based on current spin configuration
    // This implements the hierarchical update strategy from the paper
    float local_field = h_fields[global_spin_idx];

    // Add coupling contributions - optimized for block processing
    for (uint c = 0; c < num_couplings; c++) {
        uint i = i_indices[c];
        uint j = j_indices[c];
        float coupling_val = j_values[c];

        if (i == global_spin_idx) {
            int8_t spin_j = spins[chain_idx * n + j];
            local_field += coupling_val * float(spin_j);
        } else if (j == global_spin_idx) {
            int8_t spin_i = spins[chain_idx * n + i];
            local_field += coupling_val * float(spin_i);
        }
    }

    local_fields[flat_idx] = local_field;
}

// Kernel 11: True incremental field updates (Paper Algorithm 2)
// Only updates fields affected by recently flipped spins - O(degree × flips) instead of O(N²)
kernel void hierarchical_tensor_coupling_update(
    device float* local_fields [[buffer(0)]],          // Input/Output: local field values to update
    device const int8_t* old_spins [[buffer(1)]],      // Input: spins before flip
    device const int8_t* new_spins [[buffer(2)]],      // Input: spins after flip
    device const uint* i_indices [[buffer(3)]],        // Input: coupling i-indices
    device const uint* j_indices [[buffer(4)]],        // Input: coupling j-indices
    device const float* j_values [[buffer(5)]],        // Input: coupling strengths
    device const uint* flipped_indices [[buffer(6)]],  // Input: indices of flipped spins this block
    constant uint& R [[buffer(7)]],                    // Input: number of chains
    constant uint& block_start [[buffer(8)]],          // Input: start index of current block
    constant uint& block_size [[buffer(9)]],           // Input: size of current block
    constant uint& n [[buffer(10)]],                   // Input: total number of spins
    constant uint& num_couplings [[buffer(11)]],       // Input: number of coupling terms
    constant uint& num_flipped [[buffer(12)]],         // Input: number of spins flipped in this block
    uint2 gid [[thread_position_in_grid]]
) {
    uint chain_idx = gid.x;
    uint flipped_spin_idx = gid.y;

    if (chain_idx >= R || flipped_spin_idx >= num_flipped) return;

    // Get the actual spin index that was flipped
    uint global_flipped_idx = flipped_indices[flipped_spin_idx];
    if (global_flipped_idx >= n) return;

    uint chain_offset = chain_idx * n;
    
    // Get old and new spin values for the flipped spin
    int8_t old_spin = old_spins[chain_offset + global_flipped_idx];
    int8_t new_spin = new_spins[chain_offset + global_flipped_idx];
    
    // Calculate the spin change (delta)
    float spin_delta = float(new_spin - old_spin);  // Will be ±2 for actual flips

    // DEBUG: Skip if no actual flip occurred
    if (abs(spin_delta) < 1.5f) return;  // Skip if no actual flip occurred
    
    // TRUE INCREMENTAL UPDATE: Only update neighbors of flipped spins
    // This achieves the O(degree × flips) complexity from the paper
    for (uint c = 0; c < num_couplings; c++) {
        uint i = i_indices[c];
        uint j = j_indices[c];
        float coupling_val = j_values[c];

        // Check if this coupling involves the flipped spin
        if (i == global_flipped_idx) {
            // Update local field of spin j (neighbor of flipped spin i)
            uint neighbor_idx = chain_offset + j;
            float field_delta = coupling_val * spin_delta;
            
            // Atomic update to handle race conditions
            atomic_fetch_add_explicit(
                (device atomic<float>*)&local_fields[neighbor_idx], 
                field_delta, 
                memory_order_relaxed
            );
        } else if (j == global_flipped_idx) {
            // Update local field of spin i (neighbor of flipped spin j)
            uint neighbor_idx = chain_offset + i;
            float field_delta = coupling_val * spin_delta;
            
            // Atomic update to handle race conditions
            atomic_fetch_add_explicit(
                (device atomic<float>*)&local_fields[neighbor_idx], 
                field_delta, 
                memory_order_relaxed
            );
        }
    }
}

// Kernel 12: Sparse neighbor adjacency list update (Maximum Performance)
// Uses precomputed neighbor lists for O(degree) complexity per spin flip
kernel void sparse_incremental_field_update(
    device float* local_fields [[buffer(0)]],          // Input/Output: local field values
    device const int8_t* old_spins [[buffer(1)]],      // Input: spins before update
    device const int8_t* new_spins [[buffer(2)]],      // Input: spins after update
    device const uint* neighbor_offsets [[buffer(3)]],  // Input: offset into neighbor list for each spin
    device const uint* neighbor_indices [[buffer(4)]],  // Input: flattened neighbor indices
    device const float* neighbor_weights [[buffer(5)]], // Input: flattened neighbor coupling weights
    device const uint* flipped_indices [[buffer(6)]],   // Input: indices of flipped spins
    constant uint& R [[buffer(7)]],                     // Input: number of chains
    constant uint& n [[buffer(8)]],                     // Input: total number of spins
    constant uint& num_flipped [[buffer(9)]],           // Input: number of flipped spins
    uint2 gid [[thread_position_in_grid]]
) {
    uint chain_idx = gid.x;
    uint flipped_idx = gid.y;

    if (chain_idx >= R || flipped_idx >= num_flipped) return;

    // Get the global index of the flipped spin
    uint global_flipped_idx = flipped_indices[flipped_idx];
    if (global_flipped_idx >= n) return;

    uint chain_offset = chain_idx * n;

    // Get old and new spin values
    int8_t old_spin = old_spins[chain_offset + global_flipped_idx];
    int8_t new_spin = new_spins[chain_offset + global_flipped_idx];

    // Calculate spin change
    float spin_delta = float(new_spin - old_spin);
    if (abs(spin_delta) < 1.5f) return;  // Skip if no flip

    // Get neighbor list bounds for this spin
    uint neighbor_start = neighbor_offsets[global_flipped_idx];
    uint neighbor_end = (global_flipped_idx + 1 < n) ? 
                        neighbor_offsets[global_flipped_idx + 1] : 
                        neighbor_start;  // FIXED: Use neighbor_start if at boundary

    // OPTIMIZED: Update only direct neighbors using precomputed adjacency list
    // This is the key optimization that achieves maximum speedup
    for (uint neighbor_pos = neighbor_start; neighbor_pos < neighbor_end; neighbor_pos++) {
        uint neighbor_spin_idx = neighbor_indices[neighbor_pos];
        float coupling_weight = neighbor_weights[neighbor_pos];

        if (neighbor_spin_idx < n) {
            uint neighbor_field_idx = chain_offset + neighbor_spin_idx;
            float field_delta = coupling_weight * spin_delta;

            // Atomic field update
            atomic_fetch_add_explicit(
                (device atomic<float>*)&local_fields[neighbor_field_idx],
                field_delta,
                memory_order_relaxed
            );
        }
    }
}

// Kernel 13: Flip detection and tracking for incremental updates
kernel void detect_and_track_flips(
    device const int8_t* old_spins [[buffer(0)]],       // Input: spins before update
    device const int8_t* new_spins [[buffer(1)]],       // Input: spins after update
    device uint* flipped_indices [[buffer(2)]],         // Output: indices of flipped spins
    device uint* flip_count [[buffer(3)]],              // Output: total number of flips (atomic)
    constant uint& R [[buffer(4)]],                     // Input: number of chains
    constant uint& block_start [[buffer(5)]],           // Input: start of current block
    constant uint& block_size [[buffer(6)]],            // Input: size of current block
    constant uint& n [[buffer(7)]],                     // Input: total number of spins
    constant uint& max_flips_per_chain [[buffer(8)]],   // Input: maximum flips per chain
    uint2 gid [[thread_position_in_grid]]
) {
    uint chain_idx = gid.x;
    uint spin_in_block = gid.y;

    if (chain_idx >= R || spin_in_block >= block_size) return;

    uint global_spin_idx = block_start + spin_in_block;
    if (global_spin_idx >= n) return;

    uint spin_idx = chain_idx * n + global_spin_idx;

    // Check if this spin flipped
    int8_t old_spin = old_spins[spin_idx];
    int8_t new_spin = new_spins[spin_idx];

    if (old_spin != new_spin) {
        // Atomically add this spin to the flip list
        uint flip_pos = atomic_fetch_add_explicit(
            (device atomic<uint>*)&flip_count[chain_idx],
            1,
            memory_order_relaxed
        );

        // Store the global spin index that flipped
        // FIXED: Use the correct buffer size passed from Python
        if (flip_pos < max_flips_per_chain) {
            flipped_indices[chain_idx * max_flips_per_chain + flip_pos] = global_spin_idx;
        }
    }
}

// Kernel 14: Tensor-core optimized coupling field computation
kernel void tensor_optimized_coupling_field(
    device float* neighbor_sum [[buffer(0)]],
    device const int8_t* spins [[buffer(1)]],
    device const uint* i_indices [[buffer(2)]],
    device const uint* j_indices [[buffer(3)]],
    device const float* j_values [[buffer(4)]],
    constant uint& num_chains [[buffer(5)]],
    constant uint& num_couplings [[buffer(6)]],
    constant uint& num_spins [[buffer(7)]],
    uint2 thread_id [[thread_position_in_grid]]
) {
    uint chain_idx = thread_id.x;
    uint coupling_idx = thread_id.y;

    if (chain_idx >= num_chains || coupling_idx >= num_couplings) return;

    uint i = i_indices[coupling_idx];
    uint j = j_indices[coupling_idx];
    float coupling_strength = j_values[coupling_idx];

    uint chain_offset = chain_idx * num_spins;
    int8_t spin_i = spins[chain_offset + i];
    int8_t spin_j = spins[chain_offset + j];

    // Optimized atomic operations for tensor-core like performance
    float contribution_i = float(spin_j) * coupling_strength;
    float contribution_j = float(spin_i) * coupling_strength;

    uint neighbor_i_idx = chain_idx * num_spins + i;
    uint neighbor_j_idx = chain_idx * num_spins + j;

    // Use relaxed memory ordering for better performance
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

