// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

// ==============================================================================
// CUDA SIMULATED ANNEALING - SELF-FEEDING PERSISTENT KERNEL
// ==============================================================================
// Architecture: self-feeding persistent kernel with 3-slot rotating buffers.
// Each nonce owns exactly 1 block (1 SM). Each thread processes one read
// independently using thread-local unpacked state + delta_energy workspace.
//
// No inter-block coordination needed (blocks_per_nonce = 1).
// No shared memory spin state (thread-local unpacked_state[5000]).
// No color loop (SA sweeps all vars sequentially per thread).

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
// [4] unused (blocks_done not needed for 1 block/nonce)
// [5] unused (work_queue not needed for 1 block/nonce)
#define CTRL_EXIT_NOW     6
// [7] unused (generation not needed for 1 block/nonce)

// Debug flags (can be overridden via -D compiler flags)
#ifndef DEBUG_KERNEL
#define DEBUG_KERNEL 0
#endif
#ifndef DEBUG_VERBOSE
#define DEBUG_VERBOSE 0
#endif

// Profiling macros (zero overhead when PROFILE_REGIONS is not defined)
#ifdef PROFILE_REGIONS
#define PROF_T(var) var = clock64()
#define PROF_ACCUM(arr, idx, start_var) \
    arr[idx] += clock64() - start_var
#define PROF_INC(arr, idx) arr[idx]++
#define SA_NUM_REGIONS 10
#else
#define PROF_T(var)
#define PROF_ACCUM(arr, idx, start_var)
#define PROF_INC(arr, idx)
#endif

// Fast math constants
#define RNG_SCALE 2.32830643653869628906e-10f  // 1.0f / 2^32

extern "C" {

// Simple xorshift32 RNG
__device__ unsigned int xorshift32(unsigned int &state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Bit-packing helpers for output
__device__ void set_spin_packed(
    int var, signed char spin, signed char* packed
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

// Compute delta energy for flipping variable (unpacked state)
__device__ signed char get_flip_energy_unpacked(
    int var,
    const signed char* unpacked_state,
    const int* csr_row_ptr,
    const int* csr_col_ind,
    const signed char* csr_J_vals,
    int n,
    const signed char* h
) {
    const int start = __ldg(&csr_row_ptr[var]);
    const int end = __ldg(&csr_row_ptr[var + 1]);

    int energy = 0;

    if (h != NULL) {
        energy += (int)__ldg(&h[var]);
    }

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        const int neighbor = __ldg(&csr_col_ind[p]);
        const signed char Jij = __ldg(&csr_J_vals[p]);
        energy += unpacked_state[neighbor] * Jij;
    }

    return (signed char)(-2 * unpacked_state[var] * energy);
}


// ==============================================================================
// KERNEL: Self-feeding persistent SA with 3-slot rotating buffers
// ==============================================================================
// Each nonce owns 1 block = 1 SM. Each thread processes one read.
// Thread 0 manages slot transitions via atomicCAS.
// No inter-block sync needed.

__global__ void cuda_sa_self_feeding(
    // Shared topology (constant across all slots)
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,

    // Per-slot flat buffers (slot_idx = nonce_id * 3 + slot_id)
    const signed char* __restrict__ slot_J_vals,
    const signed char* __restrict__ slot_h_vals,
    signed char* slot_samples,
    int* slot_energies,

    // Shared beta schedule
    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,

    // Control array (CTRL_STRIDE ints per nonce)
    volatile int* nonce_ctrl,

    // Config
    int num_nonces,
    int num_reads,
    int N,
    int nnz,
    int max_packed_size,
    unsigned int base_seed,

    // Workspace (per global thread)
    signed char* delta_energy_workspace,
    int max_N
#ifdef PROFILE_REGIONS
    , long long* profile_output
#endif
) {
    int tid = threadIdx.x;
    int nonce_id = blockIdx.x;
    if (nonce_id >= num_nonces) return;

    int ctrl_base = nonce_id * CTRL_STRIDE;

    // Shared memory for slot coordination
    __shared__ int s_active_slot;

    // Thread 0: claim first READY slot via atomicCAS
    int active_slot = -1;
    if (tid == 0) {
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
                __threadfence();
                break;
            }
        }
        s_active_slot = active_slot;
    }
    __syncthreads();
    active_slot = s_active_slot;
    if (active_slot < 0) return;

    // === Model loop: process slots until none READY ===
    while (true) {
        int slot_idx = nonce_id * 3 + active_slot;
        const signed char* my_J = &slot_J_vals[
            (long long)slot_idx * nnz];
        const signed char* my_h = &slot_h_vals[
            (long long)slot_idx * N];
        int sample_base = slot_idx * num_reads
                          * max_packed_size;
        int energy_base = slot_idx * num_reads;

        unsigned int slot_seed =
            base_seed + (unsigned int)(
                nonce_id * 3 + active_slot);

        // Each thread processes one read
        if (tid < num_reads) {
            int packed_size = (N + 7) / 8;

            // Thread-local state
            signed char unpacked_state[5000];

            // Delta energy workspace (unique per global thread)
            int global_tid = blockIdx.x * blockDim.x + tid;
            signed char* delta_energy =
                &delta_energy_workspace[
                    (long long)global_tid * max_N];

            // Init RNG (unique per thread + slot)
            unsigned int rng_state =
                (slot_seed ^ ((tid + 1) * 12345u));
            if (rng_state == 0) rng_state = 0xdeadbeef;

            // Random initial state
            for (int var = 0; var < N; var++) {
                unsigned int r = xorshift32(rng_state);
                unpacked_state[var] =
                    (r & 1) ? (signed char)-1
                            : (signed char)1;
            }

            // Build initial delta_energy including h
            for (int var = 0; var < N; var++) {
                delta_energy[var] =
                    get_flip_energy_unpacked(
                        var, unpacked_state,
                        csr_row_ptr, csr_col_ind,
                        my_J, N, my_h);
            }

            // Initial energy (h + J terms)
            int current_energy = 0;
            for (int i = 0; i < N; i++) {
                signed char spin_i = unpacked_state[i];
                current_energy +=
                    (int)__ldg(&my_h[i]) * spin_i;
                const int start =
                    __ldg(&csr_row_ptr[i]);
                const int end =
                    __ldg(&csr_row_ptr[i + 1]);
                for (int p = start; p < end; ++p) {
                    const int j =
                        __ldg(&csr_col_ind[p]);
                    if (j > i) {
                        const signed char Jij =
                            __ldg(&my_J[p]);
                        const signed char spin_j =
                            unpacked_state[j];
                        current_energy +=
                            Jij * spin_i * spin_j;
                    }
                }
            }

            // === SA sweep loop ===
#ifdef PROFILE_REGIONS
            long long prof[SA_NUM_REGIONS] = {0};
            long long _t0, _t1, _t2, _t3, _t4;
#endif

            PROF_T(_t0);  // SA_TOTAL start
            for (int beta_idx = 0;
                 beta_idx < num_betas;
                 beta_idx++) {
                PROF_T(_t1);  // BETA_OVERHEAD
                float beta = __ldg(
                    &beta_schedule[beta_idx]);
                float threshold = 22.18f / beta;
                PROF_ACCUM(prof, 1, _t1);

                for (int sweep = 0;
                     sweep < sweeps_per_beta;
                     sweep++) {
                    PROF_T(_t2);  // SWEEP_TOTAL
                    for (int var = 0; var < N; var++) {
                        PROF_T(_t3);  // Per-var
                        signed char de =
                            delta_energy[var];

                        if (de >= threshold) {
                            PROF_ACCUM(
                                prof, 4, _t3);
                            PROF_INC(prof, 9);
                            continue;
                        }

                        PROF_T(_t4);  // ACCEPT_DECIDE
                        bool flip_spin = false;

                        if (de <= 0) {
                            flip_spin = true;
                        } else {
                            const float accept_prob =
                                __expf(
                                    -__int2float_rn(de)
                                    * beta);
                            const float rand_uniform =
                                __uint2float_rn(
                                    xorshift32(
                                        rng_state))
                                * RNG_SCALE;
                            flip_spin =
                                (accept_prob
                                 > rand_uniform);
                        }
                        PROF_ACCUM(prof, 5, _t4);

                        if (flip_spin) {
                            PROF_T(_t4);  // FLIP_TOTAL
                            current_energy += de;

                            const signed char
                                var_spin =
                                    unpacked_state[
                                        var];
                            const signed char
                                multiplier =
                                    4 * var_spin;
                            const int start =
                                __ldg(
                                    &csr_row_ptr[
                                        var]);
                            const int end =
                                __ldg(
                                    &csr_row_ptr[
                                        var + 1]);

                            PROF_T(_t3);  // NEIGHBOR
                            for (int p = start;
                                 p < end; ++p) {
                                const int neighbor =
                                    __ldg(
                                        &csr_col_ind[
                                            p]);
                                const signed char
                                    Jij = __ldg(
                                        &my_J[p]);
                                const signed char
                                    ns =
                                        unpacked_state[
                                            neighbor];
                                delta_energy[
                                    neighbor] +=
                                    multiplier
                                    * Jij * ns;
                            }
                            PROF_ACCUM(
                                prof, 7, _t3);

                            unpacked_state[var] =
                                -var_spin;
                            delta_energy[var] = -de;
                            PROF_ACCUM(
                                prof, 6, _t4);
                            PROF_INC(prof, 8);
                        }
                    }
                    PROF_ACCUM(prof, 2, _t2);
                }
            }
            PROF_ACCUM(prof, 0, _t0);  // SA_TOTAL end

#ifdef PROFILE_REGIONS
            // Write profile for this thread
            int gid = blockIdx.x * blockDim.x + tid;
            for (int r = 0; r < SA_NUM_REGIONS; r++)
                profile_output[
                    gid * SA_NUM_REGIONS + r
                ] = prof[r];
#endif

            // Pack final state to bit format
            signed char packed_state[640];
            for (int b = 0; b < packed_size; b++)
                packed_state[b] = 0;
            for (int i = 0; i < N; i++) {
                set_spin_packed(
                    i, unpacked_state[i],
                    packed_state);
            }

            // Write packed samples to output
            signed char* out_sample =
                &slot_samples[
                    sample_base
                    + tid * max_packed_size];
            for (int b = 0; b < packed_size; b++)
                out_sample[b] = packed_state[b];

            // Write energy to output
            slot_energies[energy_base + tid] =
                current_energy;
        }

        // All threads done with this slot
        __syncthreads();

        // Thread 0: mark COMPLETE, find next READY
        if (tid == 0) {
            nonce_ctrl[ctrl_base + active_slot] =
                SLOT_COMPLETE;

            // Check exit flag
            if (nonce_ctrl[
                    ctrl_base + CTRL_EXIT_NOW]) {
                s_active_slot = -1;
            } else {
                // Find next READY slot
                int next_slot = -1;
                for (int retry = 0;
                     retry < 10000; retry++) {
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
                    if (next_slot >= 0) break;
                    __nanosleep(10000);  // 10us
                }

                if (next_slot >= 0) {
                    nonce_ctrl[
                        ctrl_base + CTRL_ACTIVE_SLOT
                    ] = next_slot;
                    __threadfence();
                }
                s_active_slot = next_slot;
            }
        }
        __syncthreads();
        active_slot = s_active_slot;
        if (active_slot < 0) return;
    }
}

}  // extern "C"
