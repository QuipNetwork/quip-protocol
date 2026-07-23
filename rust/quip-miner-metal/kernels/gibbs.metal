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
// RNG - xoshiro128** (128-bit state, passes BigCrush)
// ==============================================================================
// Upgraded from xorshift32 for better statistical quality in Gibbs sampling.
// Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number Generators"

struct RngState {
    uint s0, s1, s2, s3;
};

inline uint rotl(uint x, int k) {
    return (x << k) | (x >> (32 - k));
}

inline uint xoshiro128starstar(thread RngState &state) {
    uint result = rotl(state.s1 * 5, 7) * 9;
    uint t = state.s1 << 9;
    state.s2 ^= state.s0;
    state.s3 ^= state.s1;
    state.s1 ^= state.s2;
    state.s0 ^= state.s3;
    state.s2 ^= t;
    state.s3 = rotl(state.s3, 11);
    return result;
}

// SplitMix32 for seeding xoshiro state from a single uint
inline uint splitmix32(thread uint &z) {
    z += 0x9e3779b9u;
    uint r = z;
    r = (r ^ (r >> 16)) * 0x85ebca6bu;
    r = (r ^ (r >> 13)) * 0xc2b2ae35u;
    return r ^ (r >> 16);
}

inline RngState seed_rng(uint seed) {
    uint z = seed;
    RngState state;
    state.s0 = splitmix32(z);
    state.s1 = splitmix32(z);
    state.s2 = splitmix32(z);
    state.s3 = splitmix32(z);
    return state;
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
// Effective field computation (thread-local state version)
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
// Threadgroup memory helpers for parallel kernel
// ==============================================================================

inline int8_t get_spin_shared(int var, threadgroup const int8_t* shared_state) {
    int byte_idx = var >> 3;
    int bit_idx = var & 7;
    int bit = (shared_state[byte_idx] >> bit_idx) & 1;
    return bit ? -1 : 1;
}

inline void set_spin_shared(int var, int8_t spin, threadgroup int8_t* shared_state) {
    int byte_idx = var >> 3;
    int bit_idx = var & 7;
    int8_t bit = (spin < 0) ? 1 : 0;
    int8_t mask = 1 << bit_idx;

    if (bit) {
        shared_state[byte_idx] |= mask;
    } else {
        shared_state[byte_idx] &= ~mask;
    }
}

inline float compute_effective_field_shared(
    int var,
    threadgroup const int8_t* shared_state,
    device const int* csr_row_ptr,
    device const int* csr_col_ind,
    device const int8_t* csr_J_vals,
    device const int8_t* h_vals
) {
    int start = csr_row_ptr[var];
    int end = csr_row_ptr[var + 1];

    float h_eff = float(h_vals[var]);

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = csr_col_ind[p];
        int8_t Jij = csr_J_vals[p];
        int8_t neighbor_spin = get_spin_shared(neighbor, shared_state);
        h_eff += float(Jij) * float(neighbor_spin);
    }

    return h_eff;
}

// ==============================================================================
// Gibbs update: Sample new spin from conditional distribution
// P(spin = +1) = 1 / (1 + exp(2 * beta * h_eff))
// ==============================================================================

inline int8_t gibbs_sample(float h_eff, float beta, thread RngState &rng_state) {
    // Compute probability of spin = +1
    // P(+1) = 1 / (1 + exp(2 * beta * h_eff))
    float prob_plus = 1.0f / (1.0f + exp(2.0f * beta * h_eff));

    // Sample from Bernoulli
    uint rand_val = xoshiro128starstar(rng_state);
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
    thread RngState &rng_state
) {
    // Energy change from flipping: delta = E_new - E_old = -2 * spin * h_eff
    float delta = -2.0f * float(current_spin) * h_eff;

    if (delta <= 0) {
        // Always accept energy-lowering flip
        return -current_spin;
    }

    // Accept with probability exp(-beta * delta)
    float prob = exp(-delta * beta);
    uint rand_val = xoshiro128starstar(rng_state);
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

    // Initialize RNG with unique seed per thread (xoshiro128** with splitmix32 seeding)
    RngState rng_state = seed_rng((base_seed ? base_seed : 1u) ^ (thread_id * 2654435761u));

    // Generate random initial state (bit-packed)
    for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
        packed_state[byte_idx] = 0;
    }
    for (int var = 0; var < n; var++) {
        uint rand_val = xoshiro128starstar(rng_state);
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

// ==============================================================================
// PARALLEL Block Gibbs Kernel
// ==============================================================================
// Uses threadgroup memory and barriers to parallelize within color blocks.
// One threadgroup per sample, threads divide nodes within each color.
// This enables TRUE parallel updates within each color block.
//
// IMPORTANT: Uses UNPACKED state (1 byte per spin) to avoid race conditions.
// Bit-packing would cause data races when multiple threads update spins
// that share the same byte.
//
// Architecture:
//   - threadgroup_id = sample_id (which sample we're working on)
//   - thread_in_group divides work within each color block
//   - threadgroup_barrier synchronizes between color blocks
//
// For Z(9,2) with ~1368 nodes and 4 colors (~342 nodes/color):
//   - With 256 threads/group: each thread handles ~1-2 nodes per color
//   - All 342 nodes in a color updated truly in parallel

// Helper functions for unpacked state (1 byte per spin, no bit packing)
inline int8_t get_spin_unpacked(int var, threadgroup const int8_t* state) {
    return state[var];
}

inline void set_spin_unpacked(int var, int8_t spin, threadgroup int8_t* state) {
    state[var] = spin;
}

inline float compute_effective_field_unpacked(
    int var,
    threadgroup const int8_t* state,
    device const int* csr_row_ptr,
    device const int* csr_col_ind,
    device const int8_t* csr_J_vals,
    device const int8_t* h_vals
) {
    int start = csr_row_ptr[var];
    int end = csr_row_ptr[var + 1];

    float h_eff = float(h_vals[var]);

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = csr_col_ind[p];
        int8_t Jij = csr_J_vals[p];
        int8_t neighbor_spin = state[neighbor];
        h_eff += float(Jij) * float(neighbor_spin);
    }

    return h_eff;
}

kernel void block_gibbs_parallel(
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
    constant int& num_threadgroups [[buffer(12)]],
    constant int& num_problems [[buffer(13)]],
    constant int& num_reads [[buffer(14)]],

    // h field values (concatenated for all problems)
    device const int8_t* csr_h_vals [[buffer(15)]],

    // Color block arrays
    device const int* color_block_starts [[buffer(16)]],
    device const int* color_block_counts [[buffer(17)]],
    device const int* color_node_indices [[buffer(18)]],

    // Update mode and num_colors
    constant int& update_mode [[buffer(19)]],
    constant int& num_colors [[buffer(20)]],

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one sample
    uint sample_id = threadgroup_pos.x;
    uint thread_in_group = thread_pos_in_group.x;
    uint group_size = threads_per_group.x;

    if (sample_id >= num_threadgroups) {
        return;
    }

    // Determine which problem this sample belongs to
    uint problem_id = sample_id / num_reads;

    // Get CSR offsets for this problem
    int row_ptr_start = row_ptr_offsets[problem_id];
    int col_ind_start = col_ind_offsets[problem_id];

    device const int* my_csr_row_ptr = &csr_row_ptr[row_ptr_start];
    device const int* my_csr_col_ind = &csr_col_ind[col_ind_start];
    device const int8_t* my_csr_J_vals = &csr_J_vals[col_ind_start];
    device const int8_t* my_h_vals = &csr_h_vals[problem_id * N];

    int n = N;
    int packed_size = (n + 7) / 8;

    // Threadgroup shared memory for spin state - UNPACKED (1 byte per spin)
    // Support up to ~4800 nodes
    threadgroup int8_t shared_state[4800];

    // Initialize RNG with unique seed per thread (xoshiro128** with splitmix32 seeding)
    RngState rng_state = seed_rng(
        (base_seed ? base_seed : 1u) ^ (sample_id * 2654435761u) ^ (thread_in_group * 2246822519u)
    );

    // Collaboratively generate random initial state (unpacked)
    for (uint var = thread_in_group; var < n; var += group_size) {
        uint rand_val = xoshiro128starstar(rng_state);
        shared_state[var] = (rand_val & 1) ? -1 : 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block Gibbs annealing with TRUE parallel updates
    for (int beta_idx = 0; beta_idx < num_betas; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
            // Process each color block
            for (int color = 0; color < num_colors; color++) {
                int block_start = color_block_starts[color];
                int block_count = color_block_counts[color];

                // PARALLEL: Each thread handles multiple nodes in this color
                // All nodes in a color are independent - no data races with unpacked state!
                for (uint i = thread_in_group; i < block_count; i += group_size) {
                    int var = color_node_indices[block_start + i];

                    // Compute effective field from shared state
                    float h_eff = compute_effective_field_unpacked(
                        var, shared_state,
                        my_csr_row_ptr, my_csr_col_ind, my_csr_J_vals, my_h_vals
                    );

                    // Update spin based on mode
                    int8_t new_spin;
                    if (update_mode == 0) {
                        // Gibbs sampling
                        float prob_plus = 1.0f / (1.0f + exp(2.0f * beta * h_eff));
                        uint rand_val = xoshiro128starstar(rng_state);
                        float rand_normalized = float(rand_val) / 4294967295.0f;
                        new_spin = (rand_normalized < prob_plus) ? 1 : -1;
                    } else {
                        // Metropolis update
                        int8_t current_spin = shared_state[var];
                        float delta = -2.0f * float(current_spin) * h_eff;
                        if (delta <= 0) {
                            new_spin = -current_spin;
                        } else {
                            float prob = exp(-delta * beta);
                            uint rand_val = xoshiro128starstar(rng_state);
                            float rand_normalized = float(rand_val) / 4294967295.0f;
                            new_spin = (rand_normalized < prob) ? -current_spin : current_spin;
                        }
                    }

                    shared_state[var] = new_spin;
                }

                // CRITICAL: Barrier between colors to ensure all updates complete
                // before next color reads the state
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Compute final energy collaboratively
    // Each thread computes partial energy for its subset of nodes
    int partial_energy = 0;
    for (uint i = thread_in_group; i < n; i += group_size) {
        int8_t spin_i = shared_state[i];

        // h field contribution
        partial_energy += my_h_vals[i] * spin_i;

        // J coupling contribution (count each edge once: j > i)
        int start = my_csr_row_ptr[i];
        int end = my_csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = my_csr_col_ind[p];
            if (j > i) {
                int8_t Jij = my_csr_J_vals[p];
                int8_t spin_j = shared_state[j];
                partial_energy += Jij * spin_i * spin_j;
            }
        }
    }

    // Reduce partial energies via shared array
    threadgroup int partial_energies[256];  // Max threadgroup size
    partial_energies[thread_in_group] = partial_energy;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 sums all partial energies
    if (thread_in_group == 0) {
        int total_energy = 0;
        for (uint t = 0; t < group_size; t++) {
            total_energy += partial_energies[t];
        }
        final_energies[sample_id] = total_energy;
    }

    // Write final state to output - convert unpacked to packed format
    device int8_t* output = &final_samples[sample_id * packed_size];

    // First, collaboratively zero out the output
    for (uint i = thread_in_group; i < packed_size; i += group_size) {
        output[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread packs its variables - use atomic OR to avoid races
    // Since we're writing to device memory (not threadgroup), this is safe
    // as long as each thread handles different bits
    for (uint var = thread_in_group; var < n; var += group_size) {
        int byte_idx = var >> 3;
        int bit_idx = var & 7;
        int8_t spin = shared_state[var];
        if (spin < 0) {
            // Set bit (spin = -1)
            // Note: This could race if two threads write to same byte
            // But since we stride by group_size, threads handle vars
            // that are group_size apart, so multiple vars may share a byte
            // To be safe, we use thread 0 to do all packing
        }
    }

    // Thread 0 does all the bit packing to avoid races
    if (thread_in_group == 0) {
        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            output[byte_idx] = 0;
        }
        for (int var = 0; var < n; var++) {
            int byte_idx = var >> 3;
            int bit_idx = var & 7;
            if (shared_state[var] < 0) {
                output[byte_idx] |= (1 << bit_idx);
            }
        }
    }
}
