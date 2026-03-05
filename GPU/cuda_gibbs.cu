// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// CUDA BLOCK GIBBS SAMPLER
// ==============================================================================
// Two kernel variants:
//
// 1. cuda_block_gibbs_parallel (Jacobi-style):
//    - 4 CUDA blocks per sample (one per color)
//    - All colors update simultaneously from previous sweep's state
//    - Double-buffered global state with sense-reversing barrier
//    - 4x parallelism vs sequential, trades per-sweep info flow
//
// 2. cuda_block_gibbs_sequential (validation baseline):
//    - 1 thread per sample, processes colors sequentially
//    - Gauss-Seidel style: each color sees previous colors' updates
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
// Sense-reversing barrier for inter-block synchronization
// 4 blocks per sample must synchronize after each sweep
// ==============================================================================

__device__ void sample_barrier(
    int sample_id,
    volatile int* sync_counters,
    volatile int* sync_sense
) {
    __threadfence();
    if (threadIdx.x == 0) {
        int local_sense = sync_sense[sample_id];
        int arrived = atomicAdd(
            (int*)&sync_counters[sample_id], 1
        );
        if (arrived == 3) {
            // Last of 4 blocks: reset counter and flip sense
            sync_counters[sample_id] = 0;
            __threadfence();
            sync_sense[sample_id] = 1 - local_sense;
        } else {
            // Spin-wait for sense to flip
            while (sync_sense[sample_id] == local_sense) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(100);
#endif
            }
        }
    }
    __syncthreads();
}

// ==============================================================================
// KERNEL 1: Jacobi-style parallel block Gibbs
// Grid: num_samples * 4 blocks, 256 threads per block
// Each block handles one color of one sample
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

    // Double-buffered state
    signed char* state_A,
    signed char* state_B,

    // Inter-block sync
    int* sync_counters,
    int* sync_sense,

    // Output
    signed char* final_samples,
    int* final_energies,

    // Config
    int num_samples,
    unsigned int base_seed,
    int update_mode  // 0=Gibbs, 1=Metropolis
) {
    int sample_id = blockIdx.x / num_colors;
    int color = blockIdx.x % num_colors;

    if (sample_id >= num_samples) return;

    int my_block_start = __ldg(&color_block_starts[color]);
    int my_block_count = __ldg(&color_block_counts[color]);

    // Init RNG per thread
    Xoshiro128 rng;
    xoshiro128_init(rng, base_seed, blockIdx.x, threadIdx.x);

    // Phase 1: Cooperative initialization
    // Each block initializes its color's nodes in state_A
    for (int i = threadIdx.x; i < my_block_count; i += blockDim.x) {
        int var = __ldg(&color_node_indices[my_block_start + i]);
        unsigned int r = xoshiro128ss(rng);
        int spin = (r & 1) ? -1 : 1;
        write_spin(state_A, sample_id, var, N, spin);
    }

    // Barrier: wait for all 4 blocks to finish init
    sample_barrier(sample_id, sync_counters, sync_sense);

    // Phase 2: Jacobi annealing sweeps
    int sweep_parity = 0;  // 0: read A write B, 1: read B write A

    for (int beta_idx = 0; beta_idx < num_betas; beta_idx++) {
        float beta = __ldg(&beta_schedule[beta_idx]);

        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
            signed char* read_buf = (sweep_parity == 0)
                                    ? state_A : state_B;
            signed char* write_buf = (sweep_parity == 0)
                                     ? state_B : state_A;

            // Update this color's nodes
            for (int i = threadIdx.x; i < my_block_count;
                 i += blockDim.x) {
                int var = __ldg(
                    &color_node_indices[my_block_start + i]
                );

                float h_eff = compute_effective_field(
                    var, sample_id, N, read_buf,
                    csr_row_ptr, csr_col_ind, csr_J_vals, h_vals
                );

                int new_spin;
                if (update_mode == 0) {
                    new_spin = gibbs_sample(h_eff, beta, rng);
                } else {
                    int cur = read_spin(
                        read_buf, sample_id, var, N
                    );
                    new_spin = metropolis_update(
                        cur, h_eff, beta, rng
                    );
                }

                write_spin(write_buf, sample_id, var, N, new_spin);
            }

            // Barrier: wait for all 4 blocks before next sweep
            sample_barrier(sample_id, sync_counters, sync_sense);

            sweep_parity = 1 - sweep_parity;
        }
    }

    // Phase 3: Energy computation (partial per block)
    // Determine which buffer holds the final state
    signed char* final_state = (sweep_parity == 0)
                               ? state_A : state_B;

    // Each block computes partial energy for its color's nodes
    // Use warp-level reduction, then atomicAdd to global
    float thread_energy = 0.0f;

    for (int i = threadIdx.x; i < my_block_count; i += blockDim.x) {
        int var = __ldg(&color_node_indices[my_block_start + i]);
        int spin_i = read_spin(final_state, sample_id, var, N);

        // h contribution
        thread_energy += (float)__ldg(&h_vals[var]) * (float)spin_i;

        // J contribution (j > var to count each edge once)
        int start = __ldg(&csr_row_ptr[var]);
        int end = __ldg(&csr_row_ptr[var + 1]);
        for (int p = start; p < end; ++p) {
            int j = __ldg(&csr_col_ind[p]);
            if (j > var) {
                int Jij = (int)__ldg(&csr_J_vals[p]);
                int spin_j = read_spin(
                    final_state, sample_id, j, N
                );
                thread_energy += (float)(Jij * spin_i * spin_j);
            }
        }
    }

    // Warp-level reduction
    unsigned mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_energy += __shfl_down_sync(mask, thread_energy, offset);
    }

    // Lane 0 of each warp does atomicAdd
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&final_energies[sample_id], (int)thread_energy);
    }

    // Phase 4: Bit-pack final state (only color 0, thread 0)
    // Barrier to ensure energy computation is done
    sample_barrier(sample_id, sync_counters, sync_sense);

    if (color == 0 && threadIdx.x == 0) {
        int packed_size = (N + 7) / 8;
        signed char* output = &final_samples[sample_id * packed_size];

        // Zero output first
        for (int b = 0; b < packed_size; b++) {
            output[b] = 0;
        }

        for (int var = 0; var < N; var++) {
            int spin = read_spin(final_state, sample_id, var, N);
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
