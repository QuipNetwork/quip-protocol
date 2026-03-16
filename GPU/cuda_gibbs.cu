// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// CUDA BLOCK GIBBS SAMPLER - SELF-FEEDING PERSISTENT KERNEL
// ==============================================================================
// Architecture: self-feeding persistent kernel with 3-slot rotating buffers.
// Each nonce owns blocks_per_nonce blocks (static assignment).
// Blocks process reads via per-nonce work queue, then barrier.
// After barrier, last block marks slot COMPLETE and finds next
// READY slot. Host uploads new models to free slots while kernel runs.
//
// Color loop inside kernel with __syncthreads() between colors.
// Spin state in shared memory (unpacked, 1 byte/spin) - same as Metal.

// ==============================================================================
// NonceControl layout (flat int array, CTRL_STRIDE ints per nonce)
// ==============================================================================
// Slot states
#define SLOT_EMPTY    0
#define SLOT_READY    1
#define SLOT_ACTIVE   2
#define SLOT_COMPLETE 3

// Field offsets within each nonce's control block
#define CTRL_STRIDE       8
#define CTRL_SLOT_STATE_0 0
#define CTRL_SLOT_STATE_1 1
#define CTRL_SLOT_STATE_2 2
#define CTRL_ACTIVE_SLOT  3
#define CTRL_BLOCKS_DONE  4
#define CTRL_WORK_QUEUE   5
#define CTRL_EXIT_NOW     6
#define CTRL_GENERATION   7

// Profiling macros (zero overhead when PROFILE_REGIONS is not defined)
#ifdef PROFILE_REGIONS
#define PROF_T(var) var = clock64()
#define PROF_ACCUM(arr, idx, start_var) arr[idx] += clock64() - start_var
#define PROF_INC(arr, idx) arr[idx]++
// Thread-0-only variants for Gibbs kernel (representative measurement)
#define PROF_T0(cond, var) if (cond) var = clock64()
#define PROF_ACCUM0(cond, arr, idx, sv) if (cond) arr[idx] += clock64() - sv
#define PROF_INC0(cond, arr, idx) if (cond) arr[idx]++
#define GIBBS_NUM_REGIONS 12
#else
#define PROF_T(var)
#define PROF_ACCUM(arr, idx, start_var)
#define PROF_INC(arr, idx)
#define PROF_T0(cond, var)
#define PROF_ACCUM0(cond, arr, idx, sv)
#define PROF_INC0(cond, arr, idx)
#endif

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
    const signed char* csr_J_vals, int j_off,
    const signed char* h_vals, int h_off
) {
    int start = __ldg(&csr_row_ptr[rp_off + var]);
    int end = __ldg(&csr_row_ptr[rp_off + var + 1]);

    float h_eff = (float)__ldg(&h_vals[h_off + var]);

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = __ldg(&csr_col_ind[ci_off + p]);
        int Jij = (int)__ldg(&csr_J_vals[j_off + p]);
        int neighbor_spin = (int)shared_state[neighbor];
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
// KERNEL: Self-feeding persistent kernel with 3-slot rotating buffers
// ==============================================================================
// Each nonce owns blocks_per_nonce blocks (static assignment).
// Blocks process reads via per-nonce work queue, then barrier.
// After barrier, last block marks slot COMPLETE and finds next
// READY slot. Host uploads new models to free slots while kernel
// runs. Kernel exits when no READY slot found or exit_now set.

extern "C" __global__ void cuda_gibbs_self_feeding(
    // Shared topology (same for all nonces)
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,

    // Color blocks (shared topology, tiled per nonce)
    const int* __restrict__ all_block_starts,
    const int* __restrict__ all_block_counts,
    const int* __restrict__ all_color_nodes,
    int num_colors,

    // Shared annealing schedule
    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,

    // Per-slot flat buffers (stride: see below)
    // J: indexed by (nonce_id * 3 + slot_id) * nnz + p
    signed char* slot_J_vals,
    // h: indexed by (nonce_id * 3 + slot_id) * N + var
    signed char* slot_h_vals,
    // samples: indexed by (nonce_id * 3 + slot_id)
    //          * reads_per_nonce * max_packed_size
    //          + read_idx * max_packed_size
    signed char* slot_samples,
    // energies: indexed by (nonce_id * 3 + slot_id)
    //           * reads_per_nonce + read_idx
    int* slot_energies,

    // Per-nonce control (flat int array, CTRL_STRIDE per nonce)
    volatile int* nonce_ctrl,

    // Config
    int num_nonces,
    int blocks_per_nonce,
    int reads_per_nonce,
    int N,
    int nnz,
    int max_packed_size,
    int chunks_per_model,
    int reads_per_chunk,
    unsigned int base_seed,
    int update_mode
#ifdef PROFILE_REGIONS
    , long long* profile_output
#endif
) {
    __shared__ signed char shared_state[4800];
    __shared__ int s_chunk;
    __shared__ int s_arrival;

    int nonce_id = blockIdx.x / blocks_per_nonce;
    if (nonce_id >= num_nonces) return;

    int ctrl_base = nonce_id * CTRL_STRIDE;
    int color_base = nonce_id * num_colors;
    int local_block = blockIdx.x % blocks_per_nonce;

    // Lead block (local_block==0) claims the initial READY
    // slot; other blocks spin until leader signals via
    // generation counter.  This prevents multiple blocks
    // from claiming different slots.
    int active_slot;
    if (local_block == 0) {
        active_slot = -1;
        if (threadIdx.x == 0) {
            for (int s = 0; s < 3; s++) {
                int old = atomicCAS(
                    (int*)&nonce_ctrl[ctrl_base + s],
                    SLOT_READY, SLOT_ACTIVE
                );
                if (old == SLOT_READY) {
                    active_slot = s;
                    nonce_ctrl[
                        ctrl_base + CTRL_ACTIVE_SLOT
                    ] = s;
                    nonce_ctrl[
                        ctrl_base + CTRL_WORK_QUEUE
                    ] = 0;
                    nonce_ctrl[
                        ctrl_base + CTRL_BLOCKS_DONE
                    ] = 0;
                    __threadfence();
                    atomicExch(
                        (int*)&nonce_ctrl[
                            ctrl_base + CTRL_GENERATION],
                        1
                    );
                    break;
                }
            }
            if (active_slot < 0) {
                atomicExch(
                    (int*)&nonce_ctrl[
                        ctrl_base + CTRL_GENERATION],
                    -1
                );
            }
            s_chunk = active_slot;
        }
        __syncthreads();
        active_slot = s_chunk;
        if (active_slot < 0) return;
    } else {
        // Non-lead: thread 0 spins on generation, then
        // broadcasts active_slot to other threads.
        if (threadIdx.x == 0) {
            while (true) {
                int gen = nonce_ctrl[
                    ctrl_base + CTRL_GENERATION];
                if (gen > 0) { s_chunk = 1; break; }
                if (gen < 0) { s_chunk = -1; break; }
                __nanosleep(100);
            }
        }
        __syncthreads();
        if (s_chunk < 0) return;
        active_slot = nonce_ctrl[
            ctrl_base + CTRL_ACTIVE_SLOT];
    }

    int my_gen = 1;

    // === Model loop: process models until no READY slot ===
    while (true) {
        // Compute buffer offsets for this slot
        int slot_idx = nonce_id * 3 + active_slot;
        int j_off = slot_idx * nnz;
        int h_off = slot_idx * N;
        int sample_base = slot_idx * reads_per_nonce
                          * max_packed_size;
        int energy_base = slot_idx * reads_per_nonce;

        unsigned int seed =
            base_seed + (unsigned int)(nonce_id * 3
                                       + active_slot);

        // === Work-steal loop: grab chunks of reads ===
        while (true) {
            if (threadIdx.x == 0) {
                s_chunk = atomicAdd(
                    (int*)&nonce_ctrl[
                        ctrl_base + CTRL_WORK_QUEUE],
                    1
                );
            }
            __syncthreads();

            if (s_chunk >= chunks_per_model) break;

            int read_start = s_chunk * reads_per_chunk;
            int read_end = read_start + reads_per_chunk;
            if (read_end > reads_per_nonce)
                read_end = reads_per_nonce;

            // Process each read in this chunk
            for (int read_idx = read_start;
                 read_idx < read_end; read_idx++) {

                // Init RNG
                Xoshiro128 rng;
                xoshiro128_init(
                    rng, seed,
                    (unsigned int)read_idx, threadIdx.x
                );

                // Phase 1: Init random spins
                for (int var = threadIdx.x; var < N;
                     var += blockDim.x) {
                    unsigned int r = xoshiro128ss(rng);
                    int spin = (r & 1) ? -1 : 1;
                    shared_state[var] =
                        (signed char)spin;
                }
                __syncthreads();

                // Phase 2: Chromatic Gibbs annealing
#ifdef PROFILE_REGIONS
                long long prof[GIBBS_NUM_REGIONS] = {0};
                long long _t0, _t1, _t2, _t3, _t4;
                int is_t0 = (threadIdx.x == 0);
#endif

                PROF_T0(is_t0, _t0);  // ANNEALING_TOTAL
                for (int beta_idx = 0;
                     beta_idx < num_betas;
                     beta_idx++) {
                    PROF_T0(is_t0, _t1);  // BETA_ITER
                    float beta = __ldg(
                        &beta_schedule[beta_idx]);

                    for (int sweep = 0;
                         sweep < sweeps_per_beta;
                         sweep++) {
                        PROF_T0(is_t0, _t2);  // SWEEP_ITER

                        for (int color = 0;
                             color < num_colors;
                             color++) {
                            PROF_T0(is_t0, _t3);  // COLOR_ITER

                            PROF_T0(is_t0, _t4);  // COLOR_SETUP
                            int bstart = __ldg(
                                &all_block_starts[
                                    color_base + color]);
                            int bcount = __ldg(
                                &all_block_counts[
                                    color_base + color]);
                            PROF_ACCUM0(is_t0, prof, 4, _t4);

                            PROF_T0(is_t0, _t4);  // NODE_LOOP
                            for (int i = threadIdx.x;
                                 i < bcount;
                                 i += blockDim.x) {
                                int var = __ldg(
                                    &all_color_nodes[
                                        bstart + i]);

#ifdef PROFILE_REGIONS
                                long long ft = 0;
                                if (is_t0) ft = clock64();
#endif
                                float h_eff =
                                    compute_effective_field_shared(
                                        var,
                                        shared_state,
                                        csr_row_ptr, 0,
                                        csr_col_ind, 0,
                                        slot_J_vals,
                                        j_off,
                                        slot_h_vals,
                                        h_off
                                    );
                                PROF_ACCUM0(is_t0, prof, 6, ft);

#ifdef PROFILE_REGIONS
                                long long st = 0;
                                if (is_t0) st = clock64();
#endif
                                int new_spin;
                                if (update_mode == 0) {
                                    new_spin =
                                        gibbs_sample(
                                            h_eff,
                                            beta, rng);
                                } else {
                                    int cur = (int)
                                        shared_state[
                                            var];
                                    new_spin =
                                        metropolis_update(
                                            cur, h_eff,
                                            beta, rng);
                                }
                                PROF_ACCUM0(is_t0, prof, 7, st);

                                shared_state[var] =
                                    (signed char)
                                    new_spin;
                            }
                            PROF_ACCUM0(is_t0, prof, 5, _t4);

                            PROF_T0(is_t0, _t4);  // SYNC_COLOR
                            __syncthreads();
#ifdef PROFILE_REGIONS
                            if (is_t0) {
                                prof[8] += clock64() - _t4;
                                prof[3] += clock64() - _t3;
                                prof[11]++;
                            }
#endif
                        }
#ifdef PROFILE_REGIONS
                        if (is_t0) {
                            prof[2] += clock64() - _t2;
                            prof[10]++;
                        }
#endif
                    }
#ifdef PROFILE_REGIONS
                    if (is_t0) {
                        prof[1] += clock64() - _t1;
                        prof[9]++;
                    }
#endif
                }
#ifdef PROFILE_REGIONS
                if (is_t0) {
                    prof[0] += clock64() - _t0;
                }
                // Write profile for this work unit
                if (is_t0) {
                    int wu_id = nonce_id * 10
                        + (my_gen - 1);
                    for (int r = 0;
                         r < GIBBS_NUM_REGIONS; r++)
                        profile_output[
                            wu_id * GIBBS_NUM_REGIONS + r
                        ] = prof[r];
                }
#endif

                // Phase 3: Energy (warp reduction)
                float thread_energy = 0.0f;
                for (int var = threadIdx.x; var < N;
                     var += blockDim.x) {
                    int spin_i =
                        (int)shared_state[var];

                    thread_energy +=
                        (float)slot_h_vals[
                            h_off + var]
                        * (float)spin_i;

                    int start = __ldg(
                        &csr_row_ptr[var]);
                    int end = __ldg(
                        &csr_row_ptr[var + 1]);
                    for (int p = start; p < end;
                         ++p) {
                        int j = __ldg(
                            &csr_col_ind[p]);
                        if (j > var) {
                            int Jij = (int)
                                slot_J_vals[
                                    j_off + p];
                            int spin_j = (int)
                                shared_state[j];
                            thread_energy +=
                                (float)(Jij
                                    * spin_i
                                    * spin_j);
                        }
                    }
                }

                unsigned mask = __activemask();
                for (int offset = 16; offset > 0;
                     offset >>= 1) {
                    thread_energy +=
                        __shfl_down_sync(
                            mask, thread_energy,
                            offset);
                }

                if ((threadIdx.x & 31) == 0) {
                    atomicAdd(
                        &slot_energies[
                            energy_base + read_idx],
                        (int)thread_energy);
                }

                __syncthreads();

                // Phase 4: Bit-pack (thread 0)
                if (threadIdx.x == 0) {
                    int packed_size = (N + 7) / 8;
                    signed char* output =
                        &slot_samples[
                            sample_base
                            + read_idx
                              * max_packed_size];

                    for (int b = 0;
                         b < packed_size; b++) {
                        output[b] = 0;
                    }
                    for (int var = 0;
                         var < N; var++) {
                        int spin = (int)
                            shared_state[var];
                        set_spin_packed(
                            var, spin, output);
                    }
                }

                __syncthreads();
            }
        }

        // === Inter-block barrier (generation-based) ===
        if (threadIdx.x == 0) {
            s_arrival = atomicAdd(
                (int*)&nonce_ctrl[
                    ctrl_base + CTRL_BLOCKS_DONE],
                1
            );
        }
        __syncthreads();

        if (s_arrival + 1 == blocks_per_nonce) {
            // Last block: transition state
            if (threadIdx.x == 0) {
                // Mark current slot COMPLETE
                nonce_ctrl[
                    ctrl_base + active_slot
                ] = SLOT_COMPLETE;

                // Reset barrier counter
                nonce_ctrl[
                    ctrl_base + CTRL_BLOCKS_DONE
                ] = 0;

                // Check exit
                if (nonce_ctrl[
                        ctrl_base + CTRL_EXIT_NOW]) {
                    __threadfence();
                    atomicExch(
                        (int*)&nonce_ctrl[
                            ctrl_base
                            + CTRL_GENERATION],
                        -1
                    );
                } else {
                    // Find next READY slot
                    int next_slot = -1;
                    for (int s = 0; s < 3; s++) {
                        int old = atomicCAS(
                            (int*)&nonce_ctrl[
                                ctrl_base + s],
                            SLOT_READY,
                            SLOT_ACTIVE
                        );
                        if (old == SLOT_READY) {
                            next_slot = s;
                            break;
                        }
                    }

                    if (next_slot >= 0) {
                        nonce_ctrl[
                            ctrl_base
                            + CTRL_ACTIVE_SLOT
                        ] = next_slot;
                        nonce_ctrl[
                            ctrl_base
                            + CTRL_WORK_QUEUE
                        ] = 0;
                        __threadfence();
                        atomicAdd(
                            (int*)&nonce_ctrl[
                                ctrl_base
                                + CTRL_GENERATION],
                            1
                        );
                    } else {
                        // No more work
                        __threadfence();
                        atomicExch(
                            (int*)&nonce_ctrl[
                                ctrl_base
                                + CTRL_GENERATION],
                            -1
                        );
                    }
                }
            }
        }

        // All blocks: spin on generation
        while (true) {
            int gen = nonce_ctrl[
                ctrl_base + CTRL_GENERATION];
            if (gen > my_gen) break;
            if (gen < 0) return;
            __nanosleep(100);
        }
        __syncthreads();
        my_gen++;

        // Read new active slot
        active_slot = nonce_ctrl[
            ctrl_base + CTRL_ACTIVE_SLOT];
    }
}
