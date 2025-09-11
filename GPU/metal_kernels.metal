#include <metal_stdlib>
using namespace metal;

// Optimized Metal kernels for simulated annealing

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
    
    // Accept/reject decision in single operation - for ENERGY MINIMIZATION
    bool accept = (delta_e < 0.0) || (rand_val < exp(-beta * delta_e));
    
    // Conditional spin flip
    if (accept) {
        spins[flat_idx] = -current_spin;
    }
}

// Kernel 2: Optimized scatter-add for coupling field computation
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
    float val = j_values[coupling_idx];
    
    uint spin_i_idx = chain_idx * num_spins + i;
    uint spin_j_idx = chain_idx * num_spins + j;
    
    float spin_i = float(spins[spin_i_idx]);
    float spin_j = float(spins[spin_j_idx]);
    
    // Optimized atomic operations for coupling contributions
    uint neighbor_i_idx = chain_idx * num_spins + i;
    uint neighbor_j_idx = chain_idx * num_spins + j;
    
    // Use atomic_fetch_add for thread safety across couplings
    atomic_fetch_add_explicit(
        (device atomic<float>*)&neighbor_sum[neighbor_i_idx], 
        spin_j * val, 
        memory_order_relaxed
    );
    atomic_fetch_add_explicit(
        (device atomic<float>*)&neighbor_sum[neighbor_j_idx], 
        spin_i * val, 
        memory_order_relaxed  
    );
}

// Kernel 3: Vectorized energy computation
kernel void compute_energies(
    device float* energies [[buffer(0)]],         // Output: total energies
    device const int8_t* spins [[buffer(1)]],     // Input: final spin states
    device const float* h_fields [[buffer(2)]],   // Input: linear field terms
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
    
    float h_energy = 0.0;
    float j_energy = 0.0;
    
    uint chain_offset = chain_idx * num_spins;
    
    // Compute linear field energy
    for (uint spin_idx = 0; spin_idx < num_spins; spin_idx++) {
        h_energy += float(spins[chain_offset + spin_idx]) * h_fields[spin_idx];
    }
    
    // Compute coupling energy
    for (uint coupling_idx = 0; coupling_idx < num_couplings; coupling_idx++) {
        uint i = i_indices[coupling_idx];
        uint j = j_indices[coupling_idx];
        float val = j_values[coupling_idx];
        
        float spin_i = float(spins[chain_offset + i]);
        float spin_j = float(spins[chain_offset + j]);
        
        j_energy += spin_i * spin_j * val;
    }
    
    energies[chain_idx] = h_energy + j_energy;
}

// Kernel 4: Efficient random spin initialization
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
    
    // Convert [0,1] random value to {-1,+1} spin
    spins[flat_idx] = (random_values[flat_idx] > 0.5) ? int8_t(1) : int8_t(-1);
}

// Kernel 5: Optimized field computation with local memory
kernel void compute_local_fields(
    device float* local_fields [[buffer(0)]],     // Output: local field values
    device const float* neighbor_sums [[buffer(1)]], // Input: neighbor contributions
    device const float* h_fields [[buffer(2)]],   // Input: linear field terms
    constant uint& num_chains [[buffer(3)]],      // Input: number of parallel chains  
    constant uint& num_spins [[buffer(4)]],       // Input: number of spins
    uint2 thread_id [[thread_position_in_grid]]
) {
    uint chain_idx = thread_id.x;
    uint spin_idx = thread_id.y;
    
    if (chain_idx >= num_chains || spin_idx >= num_spins) return;
    
    uint flat_idx = chain_idx * num_spins + spin_idx;
    
    // Simple addition: neighbor_sum + h_field
    local_fields[flat_idx] = neighbor_sums[flat_idx] + h_fields[spin_idx];
}

// Kernel 6: Memory-optimized coupling computation with workgroup sharing
kernel void optimized_coupling_field_shared(
    device float* neighbor_sum [[buffer(0)]],     // Output: neighbor contribution sums
    device const int8_t* spins [[buffer(1)]],     // Input: current spin states
    device const uint* i_indices [[buffer(2)]],   // Input: coupling i-indices
    device const uint* j_indices [[buffer(3)]],   // Input: coupling j-indices
    device const float* j_values [[buffer(4)]],   // Input: coupling strengths
    constant uint& num_chains [[buffer(5)]],      // Input: number of parallel chains
    constant uint& num_couplings [[buffer(6)]],   // Input: number of couplings
    constant uint& num_spins [[buffer(7)]],       // Input: number of spins
    uint2 thread_id [[thread_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    uint chain_idx = thread_id.x;
    uint coupling_start = thread_id.y * threads_per_group.y;
    
    if (chain_idx >= num_chains) return;
    
    uint chain_offset = chain_idx * num_spins;
    
    // Process multiple couplings per thread for better efficiency
    uint couplings_per_thread = (num_couplings + threads_per_group.y - 1) / threads_per_group.y;
    
    for (uint c = 0; c < couplings_per_thread; c++) {
        uint coupling_idx = coupling_start + c;
        if (coupling_idx >= num_couplings) break;
        
        uint i = i_indices[coupling_idx];
        uint j = j_indices[coupling_idx];
        float val = j_values[coupling_idx];
        
        float spin_i = float(spins[chain_offset + i]);
        float spin_j = float(spins[chain_offset + j]);
        
        // Direct memory access for better performance
        neighbor_sum[chain_offset + i] += spin_j * val;
        neighbor_sum[chain_offset + j] += spin_i * val;
    }
}