// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// CUDA BLOCK GIBBS SAMPLER
// ==============================================================================
// Two kernel variants:
//
// 1. cuda_block_gibbs_parallel (chromatic parallel):
//    - 1 CUDA block per sample, 256 threads
//    - Colors processed sequentially (Gauss-Seidel across colors)
//    - Nodes within each color updated in parallel (independent set)
//    - __syncthreads() between colors for visibility
//
// 2. cuda_block_gibbs_sequential (validation baseline):
//    - 1 thread per sample, fully sequential
//    - For correctness verification against parallel variant

// ==============================================================================
// xoshiro128** RNG (better quality than xorshift32)
// ==============================================================================

__device__ __forceinline__ unsigned int rotl(unsigned int x, int k) {
    return (x << k) | (x >> (32 - k));
}

struct Xoshiro128 {
    unsigned int s[4];
};

__device__ __forceinline__ unsigned int xoshiro128ss(Xoshiro128 &state) {
    unsigned int result = rotl(state.s[1] * 5, 7) * 9;
    unsigned int t = state.s[1] << 9;

    state.s[2] ^= state.s[0];
    state.s[3] ^= state.s[1];
    state.s[1] ^= state.s[2];
    state.s[0] ^= state.s[3];
    state.s[2] ^= t;
    state.s[3] = rotl(state.s[3], 11);

    return result;
}

__device__ __forceinline__ void xoshiro128_init(
    Xoshiro128 &state, unsigned int seed,
    unsigned int block_id, unsigned int thread_id
) {
    // SplitMix64-style seed expansion for each unique (seed, block, thread)
    unsigned long long z = (unsigned long long)seed + block_id * 65537ULL
                           + thread_id * 2654435761ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    state.s[0] = (unsigned int)(z & 0xFFFFFFFF);
    state.s[1] = (unsigned int)(z >> 32);

    z = (unsigned long long)seed + block_id * 65537ULL
        + thread_id * 2654435761ULL + 1;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    state.s[2] = (unsigned int)(z & 0xFFFFFFFF);
    state.s[3] = (unsigned int)(z >> 32);

    // Ensure non-zero state
    if (state.s[0] == 0 && state.s[1] == 0
        && state.s[2] == 0 && state.s[3] == 0) {
        state.s[0] = 1;
    }
}

// ==============================================================================
// Spin read/write helpers (unpacked int8 state)
// ==============================================================================

__device__ __forceinline__ int read_spin(
    const signed char* state, int sample_id, int var, int N
) {
    return (int)__ldg(&state[sample_id * N + var]);
}

__device__ __forceinline__ void write_spin(
    signed char* state, int sample_id, int var, int N, int spin
) {
    state[sample_id * N + var] = (signed char)spin;
}

// ==============================================================================
// Bit-packing helpers (for output)
// ==============================================================================

__device__ __forceinline__ void set_spin_packed(
    int var, int spin, signed char* packed
) {
    int byte_idx = var >> 3;
    int bit_idx = var & 7;
    signed char bit = (spin < 0) ? 1 : 0;
    signed char mask = 1 << bit_idx;
    if (bit) {
        packed[byte_idx] |= mask;
    } else {
        packed[byte_idx] &= ~mask;
    }
}

// ==============================================================================
// Effective field computation
// h_eff = h_i + sum_j(J_ij * x_j)
// ==============================================================================

__device__ float compute_effective_field(
    int var, int sample_id, int N,
    const signed char* state_buf,
    const int* csr_row_ptr,
    const int* csr_col_ind,
    const signed char* csr_J_vals,
    const signed char* h_vals
) {
    int start = __ldg(&csr_row_ptr[var]);
    int end = __ldg(&csr_row_ptr[var + 1]);

    float h_eff = (float)__ldg(&h_vals[var]);

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = __ldg(&csr_col_ind[p]);
        int Jij = (int)__ldg(&csr_J_vals[p]);
        int neighbor_spin = read_spin(state_buf, sample_id, neighbor, N);
        h_eff += (float)(Jij * neighbor_spin);
    }

    return h_eff;
}

// ==============================================================================
// Gibbs update: P(spin = +1) = 1 / (1 + exp(2 * beta * h_eff))
// ==============================================================================

__device__ __forceinline__ int gibbs_sample(
    float h_eff, float beta, Xoshiro128 &rng
) {
    float prob_plus = 1.0f / (1.0f + __expf(2.0f * beta * h_eff));
    unsigned int rand_val = xoshiro128ss(rng);
    float rand_norm = (float)rand_val / 4294967295.0f;
    return (rand_norm < prob_plus) ? 1 : -1;
}

// ==============================================================================
// Metropolis update: accept flip with P = min(1, exp(-beta * delta))
// ==============================================================================

__device__ __forceinline__ int metropolis_update(
    int current_spin, float h_eff, float beta, Xoshiro128 &rng
) {
    float delta = -2.0f * (float)current_spin * h_eff;

    if (delta <= 0.0f) {
        return -current_spin;
    }

    float prob = __expf(-delta * beta);
    unsigned int rand_val = xoshiro128ss(rng);
    float rand_norm = (float)rand_val / 4294967295.0f;
    return (rand_norm < prob) ? -current_spin : current_spin;
}

// ==============================================================================
// KERNEL 1: Chromatic parallel block Gibbs
// Grid: num_samples blocks, 256 threads per block
// Colors processed sequentially (Gauss-Seidel across colors)
// Nodes within each color updated in parallel (independent set)
// ==============================================================================

extern "C" __global__ void cuda_block_gibbs_parallel(
    // CSR graph (single problem, shared across all samples)
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const signed char* __restrict__ csr_J_vals,
    const signed char* __restrict__ h_vals,
    int N,

    // Color block partitioning
    const int* __restrict__ color_block_starts,
    const int* __restrict__ color_block_counts,
    const int* __restrict__ color_node_indices,
    int num_colors,

    // Annealing schedule
    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,

    // State (single buffer, one per sample)
    signed char* state,

    // Output
    signed char* final_samples,
    int* final_energies,

    // Config
    int num_samples,
    unsigned int base_seed,
    int update_mode  // 0=Gibbs, 1=Metropolis
) {
    int sample_id = blockIdx.x;
    if (sample_id >= num_samples) return;

    // Init RNG per thread
    Xoshiro128 rng;
    xoshiro128_init(rng, base_seed, blockIdx.x, threadIdx.x);

    // Phase 1: Cooperative initialization
    for (int var = threadIdx.x; var < N; var += blockDim.x) {
        unsigned int r = xoshiro128ss(rng);
        int spin = (r & 1) ? -1 : 1;
        write_spin(state, sample_id, var, N, spin);
    }
    __syncthreads();

    // Phase 2: Chromatic Gibbs annealing
    // Colors processed sequentially, nodes parallel within color
    for (int beta_idx = 0; beta_idx < num_betas; beta_idx++) {
        float beta = __ldg(&beta_schedule[beta_idx]);

        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
            for (int color = 0; color < num_colors; color++) {
                int block_start = __ldg(
                    &color_block_starts[color]
                );
                int block_count = __ldg(
                    &color_block_counts[color]
                );

                // Parallel update within this color
                for (int i = threadIdx.x; i < block_count;
                     i += blockDim.x) {
                    int var = __ldg(
                        &color_node_indices[block_start + i]
                    );

                    float h_eff = compute_effective_field(
                        var, sample_id, N, state,
                        csr_row_ptr, csr_col_ind,
                        csr_J_vals, h_vals
                    );

                    int new_spin;
                    if (update_mode == 0) {
                        new_spin = gibbs_sample(
                            h_eff, beta, rng
                        );
                    } else {
                        int cur = read_spin(
                            state, sample_id, var, N
                        );
                        new_spin = metropolis_update(
                            cur, h_eff, beta, rng
                        );
                    }

                    write_spin(
                        state, sample_id, var, N, new_spin
                    );
                }

                // Sync between colors: make updates visible
                __syncthreads();
            }
        }
    }

    // Phase 3: Energy computation (all threads cooperate)
    float thread_energy = 0.0f;

    for (int var = threadIdx.x; var < N; var += blockDim.x) {
        int spin_i = read_spin(state, sample_id, var, N);

        // h contribution
        thread_energy += (float)__ldg(&h_vals[var])
                         * (float)spin_i;

        // J contribution (j > var to count each edge once)
        int start = __ldg(&csr_row_ptr[var]);
        int end = __ldg(&csr_row_ptr[var + 1]);
        for (int p = start; p < end; ++p) {
            int j = __ldg(&csr_col_ind[p]);
            if (j > var) {
                int Jij = (int)__ldg(&csr_J_vals[p]);
                int spin_j = read_spin(
                    state, sample_id, j, N
                );
                thread_energy += (float)(
                    Jij * spin_i * spin_j
                );
            }
        }
    }

    // Warp-level reduction
    unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_energy += __shfl_down_sync(
            mask, thread_energy, offset
        );
    }

    // Lane 0 of each warp does atomicAdd
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(
            &final_energies[sample_id], (int)thread_energy
        );
    }

    __syncthreads();

    // Phase 4: Bit-pack final state (thread 0 only)
    if (threadIdx.x == 0) {
        int packed_size = (N + 7) / 8;
        signed char* output = &final_samples[
            sample_id * packed_size
        ];

        for (int b = 0; b < packed_size; b++) {
            output[b] = 0;
        }

        for (int var = 0; var < N; var++) {
            int spin = read_spin(
                state, sample_id, var, N
            );
            set_spin_packed(var, spin, output);
        }
    }
}

// ==============================================================================
// KERNEL 2: Sequential block Gibbs (validation baseline)
// Grid: num_samples blocks, 1 thread per block
// Standard Gauss-Seidel: colors processed sequentially
// ==============================================================================

extern "C" __global__ void cuda_block_gibbs_sequential(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const signed char* __restrict__ csr_J_vals,
    const signed char* __restrict__ h_vals,
    int N,

    const int* __restrict__ color_block_starts,
    const int* __restrict__ color_block_counts,
    const int* __restrict__ color_node_indices,
    int num_colors,

    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,

    // Single state buffer (no double buffering needed)
    signed char* state,

    // Output
    signed char* final_samples,
    int* final_energies,

    int num_samples,
    unsigned int base_seed,
    int update_mode
) {
    int sample_id = blockIdx.x;
    if (sample_id >= num_samples) return;
    if (threadIdx.x != 0) return;

    Xoshiro128 rng;
    xoshiro128_init(rng, base_seed, blockIdx.x, 0);

    // Initialize random state
    for (int var = 0; var < N; var++) {
        unsigned int r = xoshiro128ss(rng);
        int spin = (r & 1) ? -1 : 1;
        write_spin(state, sample_id, var, N, spin);
    }

    // Gauss-Seidel annealing
    for (int beta_idx = 0; beta_idx < num_betas; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
            for (int c = 0; c < num_colors; c++) {
                int block_start = color_block_starts[c];
                int block_count = color_block_counts[c];

                for (int i = 0; i < block_count; i++) {
                    int var = color_node_indices[block_start + i];

                    float h_eff = compute_effective_field(
                        var, sample_id, N, state,
                        csr_row_ptr, csr_col_ind,
                        csr_J_vals, h_vals
                    );

                    int new_spin;
                    if (update_mode == 0) {
                        new_spin = gibbs_sample(h_eff, beta, rng);
                    } else {
                        int cur = read_spin(
                            state, sample_id, var, N
                        );
                        new_spin = metropolis_update(
                            cur, h_eff, beta, rng
                        );
                    }

                    write_spin(
                        state, sample_id, var, N, new_spin
                    );
                }
            }
        }
    }

    // Compute energy
    int energy = 0;
    for (int i = 0; i < N; i++) {
        int spin_i = read_spin(state, sample_id, i, N);
        energy += (int)h_vals[i] * spin_i;

        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {
                int Jij = (int)csr_J_vals[p];
                int spin_j = read_spin(state, sample_id, j, N);
                energy += Jij * spin_i * spin_j;
            }
        }
    }

    final_energies[sample_id] = energy;

    // Bit-pack final state
    int packed_size = (N + 7) / 8;
    signed char* output = &final_samples[sample_id * packed_size];
    for (int b = 0; b < packed_size; b++) {
        output[b] = 0;
    }
    for (int var = 0; var < N; var++) {
        int spin = read_spin(state, sample_id, var, N);
        set_spin_packed(var, spin, output);
    }
}
