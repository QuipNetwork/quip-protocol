// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// METAL BLOCK GIBBS SAMPLER
// ==============================================================================
// Implements block Gibbs sampling based on dwave-pytorch-plugin BlockSampler:
// 1. Graph coloring partitions nodes into independent blocks (4 colors for Zephyr)
// 2. All nodes in a color block can be updated simultaneously (no edges between them)
// 3. Gibbs update: P(spin=+1) = 1 / (1 + exp(2 * beta * h_eff))
// 4. Metropolis update: P(flip) = min(1, exp(-beta * delta))
// 5. Effective field: h_eff = h_i + sum_j(J_ij * x_j)

typedef unsigned int uint;

// ==============================================================================
// RNG - xorshift32 (same as metal_kernels.metal)
// ==============================================================================

inline uint xorshift32(thread uint &state) {
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// ==============================================================================
// Bit-packing helpers (same as metal_kernels.metal)
// Pack 8 spins into 1 byte: bit i stores spin i (0=+1, 1=-1)
// ==============================================================================

inline int8_t get_spin_packed(int var, thread const int8_t* packed_state) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    int bit = (packed_state[byte_idx] >> bit_idx) & 1;
    return bit ? -1 : 1;  // 0 -> +1, 1 -> -1
}

inline void set_spin_packed(int var, int8_t spin, thread int8_t* packed_state) {
    int byte_idx = var >> 3;
    int bit_idx = var & 7;
    int8_t bit = (spin < 0) ? 1 : 0;  // -1 -> 1, +1 -> 0
    int8_t mask = 1 << bit_idx;

    if (bit) {
        packed_state[byte_idx] |= mask;   // Set bit
    } else {
        packed_state[byte_idx] &= ~mask;  // Clear bit
    }
}

// ==============================================================================
// Effective field computation
// h_eff = h_i + sum_j(J_ij * x_j)
// ==============================================================================

inline float compute_effective_field(
    int var,
    thread const int8_t* packed_state,
    device const int* csr_row_ptr,
    device const int* csr_col_ind,
    device const int8_t* csr_J_vals,
    device const int8_t* h_vals
) {
    int start = csr_row_ptr[var];
    int end = csr_row_ptr[var + 1];

    // Start with linear bias
    float h_eff = float(h_vals[var]);

    // Sum over neighbors: h_eff += sum_j(J_ij * x_j)
    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = csr_col_ind[p];
        int8_t Jij = csr_J_vals[p];
        int8_t neighbor_spin = get_spin_packed(neighbor, packed_state);
        h_eff += float(Jij) * float(neighbor_spin);
    }

    return h_eff;
}

// ==============================================================================
// Gibbs update: Sample new spin from conditional distribution
// P(spin = +1) = 1 / (1 + exp(2 * beta * h_eff))
// ==============================================================================

inline int8_t gibbs_sample(float h_eff, float beta, thread uint &rng_state) {
    // Compute probability of spin = +1
    // P(+1) = 1 / (1 + exp(2 * beta * h_eff))
    float prob_plus = 1.0f / (1.0f + exp(2.0f * beta * h_eff));

    // Sample from Bernoulli
    uint rand_val = xorshift32(rng_state);
    float rand_normalized = float(rand_val) / 4294967295.0f;

    return (rand_normalized < prob_plus) ? 1 : -1;
}

// ==============================================================================
// Metropolis update: Accept flip with probability min(1, exp(-beta * delta))
// delta = -2 * spin * h_eff (energy change from flipping)
// ==============================================================================

inline int8_t metropolis_update(
    int8_t current_spin,
    float h_eff,
    float beta,
    thread uint &rng_state
) {
    // Energy change from flipping: delta = E_new - E_old = -2 * spin * h_eff
    float delta = -2.0f * float(current_spin) * h_eff;

    if (delta <= 0) {
        // Always accept energy-lowering flip
        return -current_spin;
    }

    // Accept with probability exp(-beta * delta)
    float prob = exp(-delta * beta);
    uint rand_val = xorshift32(rng_state);
    float rand_normalized = float(rand_val) / 4294967295.0f;

    return (rand_normalized < prob) ? -current_spin : current_spin;
}

// ==============================================================================
// Main Block Gibbs Kernel
// ==============================================================================

kernel void block_gibbs_sampler(
    // CSR graph structure (concatenated for all problems)
    device const int* csr_row_ptr [[buffer(0)]],
    device const int* csr_col_ind [[buffer(1)]],
    device const int8_t* csr_J_vals [[buffer(2)]],
    device const int* row_ptr_offsets [[buffer(3)]],
    device const int* col_ind_offsets [[buffer(4)]],

    // Scalar parameters
    constant int& N [[buffer(5)]],
    constant int& num_betas [[buffer(6)]],
    constant int& sweeps_per_beta [[buffer(7)]],
    constant uint& base_seed [[buffer(8)]],

    // Beta schedule
    device const float* beta_schedule [[buffer(9)]],

    // Output buffers
    device int8_t* final_samples [[buffer(10)]],
    device int* final_energies [[buffer(11)]],

    // Batch parameters
    constant int& num_threads [[buffer(12)]],
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

    // h field values (concatenated for all problems)
    device const int8_t* csr_h_vals [[buffer(15)]],

    // Color block arrays
    device const int* color_block_starts [[buffer(16)]],   // [4] start indices into color_node_indices
    device const int* color_block_counts [[buffer(17)]],   // [4] node counts per color
    device const int* color_node_indices [[buffer(18)]],   // [N] nodes sorted by color

    // Update mode and num_colors
    constant int& update_mode [[buffer(19)]],              // 0=Gibbs, 1=Metropolis
    constant int& num_colors [[buffer(20)]],               // typically 4 for Zephyr

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    // Compute global thread ID
    uint thread_id = threadgroup_pos.x * threads_per_group.x + thread_pos_in_group.x;

    if (thread_id >= num_threads) {
        return;
    }

    // Determine which problem this thread is working on
    uint problem_id = thread_id / num_reads;

    // Get CSR offsets for this specific problem
    int row_ptr_start = row_ptr_offsets[problem_id];
    int col_ind_start = col_ind_offsets[problem_id];

    // Point to this problem's CSR data
    device const int* my_csr_row_ptr = &csr_row_ptr[row_ptr_start];
    device const int* my_csr_col_ind = &csr_col_ind[col_ind_start];
    device const int8_t* my_csr_J_vals = &csr_J_vals[col_ind_start];
    device const int8_t* my_h_vals = &csr_h_vals[problem_id * N];

    int n = N;
    int num_beta_values = num_betas;
    int sweeps_per_beta_val = sweeps_per_beta;
    int packed_size = (n + 7) / 8;

    // Thread-local memory for state
    // Support up to ~4800 nodes (600 bytes packed)
    thread int8_t packed_state[600];

    // Initialize RNG with unique seed per thread
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state (bit-packed)
    for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
        packed_state[byte_idx] = 0;
    }
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        int8_t spin = (rand_val & 1) ? -1 : 1;
        set_spin_packed(var, spin, packed_state);
    }

    // Block Gibbs annealing
    for (int beta_idx = 0; beta_idx < num_beta_values; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        for (int sweep = 0; sweep < sweeps_per_beta_val; sweep++) {
            // Process each color block
            // Key insight: All nodes in a color block are independent
            // (no edges between them due to valid graph coloring)
            for (int color = 0; color < num_colors; color++) {
                int block_start = color_block_starts[color];
                int block_count = color_block_counts[color];

                // Update all nodes in this color block
                for (int i = 0; i < block_count; i++) {
                    int var = color_node_indices[block_start + i];

                    // Compute effective field: h_eff = h[var] + sum_j(J[var,j] * x[j])
                    float h_eff = compute_effective_field(
                        var, packed_state,
                        my_csr_row_ptr, my_csr_col_ind, my_csr_J_vals, my_h_vals
                    );

                    // Update spin based on mode
                    int8_t new_spin;
                    if (update_mode == 0) {
                        // Gibbs: Sample from conditional distribution
                        new_spin = gibbs_sample(h_eff, beta, rng_state);
                    } else {
                        // Metropolis: Accept/reject flip
                        int8_t current_spin = get_spin_packed(var, packed_state);
                        new_spin = metropolis_update(current_spin, h_eff, beta, rng_state);
                    }

                    set_spin_packed(var, new_spin, packed_state);
                }
            }
        }
    }

    // Compute final energy: E = sum_i(h_i * x_i) + sum_{i<j}(J_ij * x_i * x_j)
    int current_energy = 0;
    for (int i = 0; i < n; i++) {
        int8_t spin_i = get_spin_packed(i, packed_state);

        // Add h field contribution
        current_energy += my_h_vals[i] * spin_i;

        // Add J coupling contribution (count each edge once: j > i)
        int start = my_csr_row_ptr[i];
        int end = my_csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = my_csr_col_ind[p];
            if (j > i) {
                int8_t Jij = my_csr_J_vals[p];
                int8_t spin_j = get_spin_packed(j, packed_state);
                current_energy += Jij * spin_i * spin_j;
            }
        }
    }

    // Write final state to output (bit-packed)
    device int8_t* output = &final_samples[thread_id * packed_size];
    for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
        output[byte_idx] = packed_state[byte_idx];
    }

    final_energies[thread_id] = current_energy;
}
