// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// CUDA BLOCK GIBBS SAMPLER - PERSISTENT KERNEL WITH WORK QUEUE
// ==============================================================================
// Architecture: single persistent kernel, blocks grab work from atomic queue.
// Each work unit = (model_id, read_chunk). Block processes all sweeps/colors
// for its reads, then grabs the next unit.
//
// Color loop inside kernel with __syncthreads() between colors.
// Spin state in shared memory (unpacked, 1 byte/spin) - same as Metal.
//
// Two kernel variants:
//   1. cuda_gibbs_persistent: work-queue persistent kernel (256 threads)
//   2. cuda_block_gibbs_sequential: validation baseline (1 thread)

// ==============================================================================
// xoshiro128** RNG
// ==============================================================================

__device__ __forceinline__ unsigned int rotl(
    unsigned int x, int k
) {
    return (x << k) | (x >> (32 - k));
}

struct Xoshiro128 {
    unsigned int s[4];
};

__device__ __forceinline__ unsigned int xoshiro128ss(
    Xoshiro128 &state
) {
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
    unsigned long long z =
        (unsigned long long)seed + block_id * 65537ULL
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

    if (state.s[0] == 0 && state.s[1] == 0
        && state.s[2] == 0 && state.s[3] == 0) {
        state.s[0] = 1;
    }
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
// Effective field from shared memory state
// h_eff = h_i + sum_j(J_ij * x_j)
// ==============================================================================

__device__ float compute_effective_field_shared(
    int var,
    volatile signed char* shared_state,
    const int* csr_row_ptr, int rp_off,
    const int* csr_col_ind, int ci_off,
    const signed char* csr_J_vals,
    const signed char* h_vals, int h_off
) {
    int start = __ldg(&csr_row_ptr[rp_off + var]);
    int end = __ldg(&csr_row_ptr[rp_off + var + 1]);

    float h_eff = (float)__ldg(&h_vals[h_off + var]);

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = __ldg(&csr_col_ind[ci_off + p]);
        int Jij = (int)__ldg(&csr_J_vals[ci_off + p]);
        int neighbor_spin = (int)shared_state[neighbor];
        h_eff += (float)(Jij * neighbor_spin);
    }

    return h_eff;
}

// ==============================================================================
// Effective field from global memory state (for sequential kernel)
// ==============================================================================

__device__ float compute_effective_field(
    int var,
    const signed char* my_state,
    const int* csr_row_ptr, int rp_off,
    const int* csr_col_ind, int ci_off,
    const signed char* csr_J_vals,
    const signed char* h_vals, int h_off
) {
    int start = __ldg(&csr_row_ptr[rp_off + var]);
    int end = __ldg(&csr_row_ptr[rp_off + var + 1]);

    float h_eff = (float)__ldg(&h_vals[h_off + var]);

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = __ldg(&csr_col_ind[ci_off + p]);
        int Jij = (int)__ldg(&csr_J_vals[ci_off + p]);
        int neighbor_spin = (int)__ldg(&my_state[neighbor]);
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
    float prob_plus =
        1.0f / (1.0f + __expf(2.0f * beta * h_eff));
    unsigned int rand_val = xoshiro128ss(rng);
    float rand_norm = (float)rand_val / 4294967295.0f;
    return (rand_norm < prob_plus) ? 1 : -1;
}

// ==============================================================================
// Metropolis update: accept flip with P = min(1, exp(-beta * delta))
// ==============================================================================

__device__ __forceinline__ int metropolis_update(
    int current_spin, float h_eff, float beta,
    Xoshiro128 &rng
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
// KERNEL 1: Persistent work-queue Gibbs kernel
// Grid: (num_blocks,), Block: (256,)
// Blocks grab work units from atomic queue. Each unit = one read
// for one model. Block processes all sweeps/colors using shared memory.
// ==============================================================================

extern "C" __global__ void cuda_gibbs_persistent(
    // Concatenated CSR graph
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const signed char* __restrict__ csr_J_vals,
    const signed char* __restrict__ h_vals,

    // Per-problem metadata
    const int* __restrict__ problem_N,
    const int* __restrict__ problem_rp_offsets,
    const int* __restrict__ problem_ci_offsets,
    const int* __restrict__ problem_h_offsets,

    // Color blocks (flattened, global indices)
    const int* __restrict__ all_block_starts,
    const int* __restrict__ all_block_counts,
    const int* __restrict__ all_color_nodes,
    int num_colors,

    // Annealing schedule
    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,

    // Output
    signed char* final_samples,
    int* final_energies,

    // Config
    int num_reads,
    int max_N,
    int max_packed_size,
    int num_problems,

    // Work queue
    int* work_queue_counter,
    int chunks_per_model,
    int reads_per_chunk,
    int total_work_units,

    unsigned int base_seed,
    int update_mode
) {
    // Shared memory for spin state (unpacked, 1 byte per spin)
    __shared__ signed char shared_state[4800];
    __shared__ int s_work_unit;

    // Work-stealing loop
    while (true) {
        // Thread 0 grabs next work unit from queue
        if (threadIdx.x == 0) {
            s_work_unit = atomicAdd(work_queue_counter, 1);
        }
        __syncthreads();

        if (s_work_unit >= total_work_units) break;

        int model_id = s_work_unit / chunks_per_model;
        int chunk_id = s_work_unit % chunks_per_model;

        // Read range for this chunk
        int read_start = chunk_id * reads_per_chunk;
        int read_end = read_start + reads_per_chunk;
        if (read_end > num_reads) read_end = num_reads;

        // Load problem metadata
        int N = __ldg(&problem_N[model_id]);
        int rp_off = __ldg(&problem_rp_offsets[model_id]);
        int ci_off = __ldg(&problem_ci_offsets[model_id]);
        int h_off = __ldg(&problem_h_offsets[model_id]);
        int color_base = model_id * num_colors;

        // Unique seed per model
        unsigned int seed =
            base_seed + (unsigned int)model_id;

        // Process each read in this chunk
        for (int read_idx = read_start;
             read_idx < read_end; read_idx++) {

            int global_idx =
                model_id * num_reads + read_idx;

            // Init RNG per (read, thread)
            Xoshiro128 rng;
            xoshiro128_init(
                rng, seed,
                (unsigned int)read_idx, threadIdx.x
            );

            // Phase 1: Init random spins in shared memory
            for (int var = threadIdx.x; var < N;
                 var += blockDim.x) {
                unsigned int r = xoshiro128ss(rng);
                int spin = (r & 1) ? -1 : 1;
                shared_state[var] = (signed char)spin;
            }
            __syncthreads();

            // Phase 2: Chromatic Gibbs annealing
            for (int beta_idx = 0; beta_idx < num_betas;
                 beta_idx++) {
                float beta =
                    __ldg(&beta_schedule[beta_idx]);

                for (int sweep = 0;
                     sweep < sweeps_per_beta; sweep++) {
                    for (int color = 0;
                         color < num_colors; color++) {
                        int bstart = __ldg(
                            &all_block_starts[
                                color_base + color]
                        );
                        int bcount = __ldg(
                            &all_block_counts[
                                color_base + color]
                        );

                        // 256 threads divide nodes
                        for (int i = threadIdx.x;
                             i < bcount;
                             i += blockDim.x) {
                            int var = __ldg(
                                &all_color_nodes[
                                    bstart + i]
                            );

                            float h_eff =
                                compute_effective_field_shared(
                                    var, shared_state,
                                    csr_row_ptr, rp_off,
                                    csr_col_ind, ci_off,
                                    csr_J_vals,
                                    h_vals, h_off
                                );

                            int new_spin;
                            if (update_mode == 0) {
                                new_spin = gibbs_sample(
                                    h_eff, beta, rng
                                );
                            } else {
                                int cur = (int)
                                    shared_state[var];
                                new_spin =
                                    metropolis_update(
                                        cur, h_eff,
                                        beta, rng
                                    );
                            }

                            shared_state[var] =
                                (signed char)new_spin;
                        }

                        // Barrier between colors
                        __syncthreads();
                    }
                }
            }

            // Phase 3: Energy (warp reduction)
            float thread_energy = 0.0f;
            for (int var = threadIdx.x; var < N;
                 var += blockDim.x) {
                int spin_i = (int)shared_state[var];

                thread_energy +=
                    (float)__ldg(&h_vals[h_off + var])
                    * (float)spin_i;

                int start =
                    __ldg(&csr_row_ptr[rp_off + var]);
                int end =
                    __ldg(&csr_row_ptr[rp_off + var + 1]);
                for (int p = start; p < end; ++p) {
                    int j = __ldg(
                        &csr_col_ind[ci_off + p]);
                    if (j > var) {
                        int Jij = (int)__ldg(
                            &csr_J_vals[ci_off + p]);
                        int spin_j =
                            (int)shared_state[j];
                        thread_energy += (float)(
                            Jij * spin_i * spin_j);
                    }
                }
            }

            // Warp reduction
            unsigned mask = __activemask();
            for (int offset = 16; offset > 0;
                 offset >>= 1) {
                thread_energy += __shfl_down_sync(
                    mask, thread_energy, offset);
            }

            if ((threadIdx.x & 31) == 0) {
                atomicAdd(
                    &final_energies[global_idx],
                    (int)thread_energy);
            }

            __syncthreads();

            // Phase 4: Bit-pack (thread 0)
            if (threadIdx.x == 0) {
                int packed_size = (N + 7) / 8;
                signed char* output =
                    &final_samples[
                        global_idx * max_packed_size];

                for (int b = 0; b < packed_size; b++) {
                    output[b] = 0;
                }
                for (int var = 0; var < N; var++) {
                    int spin = (int)shared_state[var];
                    set_spin_packed(
                        var, spin, output);
                }
            }

            __syncthreads();
        }
    }
}

// ==============================================================================
// KERNEL 2: Sequential block Gibbs (validation baseline)
// Grid: (num_reads, num_problems), 1 thread per block
// ==============================================================================

extern "C" __global__ void cuda_block_gibbs_sequential(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const signed char* __restrict__ csr_J_vals,
    const signed char* __restrict__ h_vals,

    const int* __restrict__ problem_N,
    const int* __restrict__ problem_rp_offsets,
    const int* __restrict__ problem_ci_offsets,
    const int* __restrict__ problem_h_offsets,

    const int* __restrict__ all_block_starts,
    const int* __restrict__ all_block_counts,
    const int* __restrict__ all_color_nodes,
    int num_colors,

    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,

    signed char* state,
    signed char* final_samples,
    int* final_energies,

    int num_samples,
    int max_N,
    int max_packed_size,
    int num_problems,
    int reads_per_block,
    unsigned int base_seed,
    int update_mode
) {
    int read_group = blockIdx.x;
    int prob_id = blockIdx.y;
    if (prob_id >= num_problems) return;
    if (threadIdx.x != 0) return;

    int N = problem_N[prob_id];
    int rp_off = problem_rp_offsets[prob_id];
    int ci_off = problem_ci_offsets[prob_id];
    int h_off = problem_h_offsets[prob_id];
    int color_base = prob_id * num_colors;

    unsigned int seed = base_seed + (unsigned int)prob_id;

    int read_start = read_group * reads_per_block;
    int read_end = read_start + reads_per_block;
    if (read_end > num_samples) read_end = num_samples;

    for (int read_idx = read_start; read_idx < read_end;
         read_idx++) {

        int global_idx = prob_id * num_samples + read_idx;
        signed char* my_state = state + global_idx * max_N;

        Xoshiro128 rng;
        xoshiro128_init(
            rng, seed, (unsigned int)read_idx, 0
        );

        // Initialize random state
        for (int var = 0; var < N; var++) {
            unsigned int r = xoshiro128ss(rng);
            int spin = (r & 1) ? -1 : 1;
            my_state[var] = (signed char)spin;
        }

        // Gauss-Seidel annealing
        for (int beta_idx = 0; beta_idx < num_betas;
             beta_idx++) {
            float beta = beta_schedule[beta_idx];

            for (int sweep = 0; sweep < sweeps_per_beta;
                 sweep++) {
                for (int c = 0; c < num_colors; c++) {
                    int bstart =
                        all_block_starts[color_base + c];
                    int bcount =
                        all_block_counts[color_base + c];

                    for (int i = 0; i < bcount; i++) {
                        int var = all_color_nodes[
                            bstart + i
                        ];

                        float h_eff =
                            compute_effective_field(
                                var, my_state,
                                csr_row_ptr, rp_off,
                                csr_col_ind, ci_off,
                                csr_J_vals, h_vals, h_off
                            );

                        int new_spin;
                        if (update_mode == 0) {
                            new_spin = gibbs_sample(
                                h_eff, beta, rng
                            );
                        } else {
                            int cur = (int)my_state[var];
                            new_spin = metropolis_update(
                                cur, h_eff, beta, rng
                            );
                        }

                        my_state[var] =
                            (signed char)new_spin;
                    }
                }
            }
        }

        // Compute energy
        int energy = 0;
        for (int i = 0; i < N; i++) {
            int spin_i = (int)my_state[i];
            energy += (int)h_vals[h_off + i] * spin_i;

            int start = csr_row_ptr[rp_off + i];
            int end = csr_row_ptr[rp_off + i + 1];
            for (int p = start; p < end; ++p) {
                int j = csr_col_ind[ci_off + p];
                if (j > i) {
                    int Jij =
                        (int)csr_J_vals[ci_off + p];
                    int spin_j = (int)my_state[j];
                    energy += Jij * spin_i * spin_j;
                }
            }
        }

        final_energies[global_idx] = energy;

        // Bit-pack final state
        int packed_size = (N + 7) / 8;
        signed char* output =
            &final_samples[global_idx * max_packed_size];
        for (int b = 0; b < packed_size; b++) {
            output[b] = 0;
        }
        for (int var = 0; var < N; var++) {
            int spin = (int)my_state[var];
            set_spin_packed(var, spin, output);
        }
    }
}
