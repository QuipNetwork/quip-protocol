#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// PURE SIMULATED ANNEALING - Exact D-Wave Implementation
// ==============================================================================
// This kernel exactly mimics D-Wave's cpu_sa.cpp implementation:
// 1. Delta energy array optimization (pre-compute, update incrementally)
// 2. xorshift32 RNG
// 3. Sequential variable ordering (spins 0..N-1)
// 4. Metropolis criterion with threshold optimization (skip if delta_E > 22.18/beta)
// 5. Efficient neighbor update after each flip

typedef unsigned int uint;

// Simple xorshift32 RNG
inline uint xorshift32(thread uint &state) {
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Compute delta energy for flipping a single variable
// Uses aggressive loop unrolling for Zephyr topology (degrees 8-20)
inline int get_flip_energy(
    int var,
    device const int8_t* state,
    device const int* csr_row_ptr,
    device const int* csr_col_ind,
    device const int8_t* csr_J_vals
) {
    int start = csr_row_ptr[var];
    int end = csr_row_ptr[var + 1];

    int energy = 0;

    // Aggressive unrolling for typical Zephyr degrees (18-20)
    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = csr_col_ind[p];
        int8_t Jij = csr_J_vals[p];
        energy += state[neighbor] * Jij;
    }

    // Delta energy = -2 * state[var] * energy
    return -2 * state[var] * energy;
}

// Pure SA kernel with delta energy optimization
// Each thread runs ONE independent SA chain (num_threads = num_reads)
kernel void pure_simulated_annealing(
    // Problem CSR representation
    device const int* csr_row_ptr [[buffer(0)]],          // [N+1]
    device const int* csr_col_ind [[buffer(1)]],          // [nnz]
    device const int8_t* csr_J_vals [[buffer(2)]],        // [nnz]

    // Scalar parameters
    constant int& N [[buffer(3)]],                         // number of spins
    constant int& num_betas [[buffer(4)]],                 // number of beta values
    constant int& sweeps_per_beta [[buffer(5)]],           // sweeps per beta
    constant uint& base_seed [[buffer(6)]],                // RNG seed

    // Beta schedule
    device const float* beta_schedule [[buffer(7)]],       // [num_betas]

    // Working memory
    device int8_t* working_states [[buffer(8)]],           // [num_reads * N]

    // Outputs
    device int8_t* final_samples [[buffer(9)]],            // [num_reads * N]
    device int* final_energies [[buffer(10)]],             // [num_reads]

    // Delta energy array
    device int* delta_energies [[buffer(11)]],             // [num_reads * N]

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    // Compute global thread ID
    uint thread_id = threadgroup_pos.x * threads_per_group.x + thread_pos_in_group.x;

    int n = N;
    int num_beta_values = num_betas;
    int sweeps_per_beta_val = sweeps_per_beta;

    // Each thread gets its own state and delta_energy array
    device int8_t* state = &working_states[thread_id * n];
    device int* delta_energy = &delta_energies[thread_id * n];

    // Initialize RNG state with unique seed per thread
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        state[var] = (rand_val & 1) ? 1 : -1;  // Random ±1
    }

    // Build initial delta_energy array
    for (int var = 0; var < n; var++) {
        delta_energy[var] = get_flip_energy(var, state, csr_row_ptr, csr_col_ind, csr_J_vals);
    }

    // Compute initial energy
    int current_energy = 0;
    for (int i = 0; i < n; i++) {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {  // Count each edge once
                int8_t Jij = csr_J_vals[p];
                current_energy += Jij * state[i] * state[j];
            }
        }
    }

    // Perform sweeps across beta schedule
    for (int beta_idx = 0; beta_idx < num_beta_values; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        // D-Wave optimization: threshold to skip impossible flips
        // log(1/2^32) ≈ -22.18, so if delta_E > 22.18/beta, probability is < 2^-32
        float threshold = 22.18f / beta;

        for (int sweep = 0; sweep < sweeps_per_beta_val; sweep++) {
            // Sequential variable ordering (matching D-Wave default)
            for (int var = 0; var < n; var++) {
                // Skip if delta energy too large (D-Wave optimization)
                if (delta_energy[var] >= threshold) continue;

                bool flip_spin = false;

                // Metropolis-Hastings acceptance rule
                if (delta_energy[var] <= 0) {
                    // Always accept energy-lowering flips
                    flip_spin = true;
                } else {
                    // Get random number
                    uint rand_val = xorshift32(rng_state);

                    // Accept with probability exp(-delta_energy * beta)
                    float prob = exp(-float(delta_energy[var]) * beta);
                    float rand_normalized = float(rand_val) / 4294967295.0f;  // 2^32 - 1

                    if (prob > rand_normalized) {
                        flip_spin = true;
                    }
                }

                if (flip_spin) {
                    // Track energy change
                    current_energy += delta_energy[var];

                    // Update delta energies of all neighbors
                    int8_t multiplier = 4 * state[var];
                    int start = csr_row_ptr[var];
                    int end = csr_row_ptr[var + 1];

                    // Aggressive unrolling for typical Zephyr degrees (18-20)
                    #pragma unroll 20
                    for (int p = start; p < end; ++p) {
                        int neighbor = csr_col_ind[p];
                        int8_t Jij = csr_J_vals[p];
                        delta_energy[neighbor] += multiplier * Jij * state[neighbor];
                    }

                    // Flip the spin and negate its delta energy
                    state[var] *= -1;
                    delta_energy[var] *= -1;
                }
            }
        }
    }

    // Write final state to output
    device int8_t* output = &final_samples[thread_id * n];
    for (int i = 0; i < n; i++) {
        output[i] = state[i];
    }

    // Use tracked energy
    final_energies[thread_id] = current_energy;
}
