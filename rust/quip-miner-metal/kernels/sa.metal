#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// METAL SIMULATED ANNEALING - Based on the D-Wave Implementation
// ==============================================================================
// This kernel mimics D-Wave's cpu_sa.cpp implementation:
// 1. Delta energy array optimization (pre-compute, update incrementally)
// 2. xorshift32 RNG
// 3. Sequential variable ordering (spins 0..N-1)
// 4. Metropolis criterion with threshold optimization (skip if delta_E > 22.18/beta)
// 5. Efficient neighbor update after each flip

typedef unsigned int uint;

inline uint xorshift32(thread uint &state) {
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Bit-packing helpers for thread-local state
// Pack 8 spins into 1 byte: bit i stores spin i (0=+1, 1=-1)
inline int8_t get_spin_packed(int var, thread const int8_t* packed_state) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    int bit = (packed_state[byte_idx] >> bit_idx) & 1;
    return bit ? -1 : 1;  // 0 -> +1, 1 -> -1
}

inline void set_spin_packed(int var, int8_t spin, thread int8_t* packed_state) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    int8_t bit = (spin < 0) ? 1 : 0;  // -1 -> 1, +1 -> 0
    int8_t mask = 1 << bit_idx;

    if (bit) {
        packed_state[byte_idx] |= mask;   // Set bit
    } else {
        packed_state[byte_idx] &= ~mask;  // Clear bit
    }
}

inline void flip_spin_packed(int var, thread int8_t* packed_state) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    packed_state[byte_idx] ^= (1 << bit_idx);  // Toggle bit
}

// Compute delta energy for flipping a single variable
// using the bit-packed state in thread-local memory
inline int8_t get_flip_energy(
    int var,
    thread const int8_t* packed_state,
    device const int* csr_row_ptr,
    device const int* csr_col_ind,
    device const int8_t* csr_J_vals,
    device const int8_t* h_vals
) {
    int start = csr_row_ptr[var];
    int end = csr_row_ptr[var + 1];

    int energy = 0;

    // Aggressive unrolling for typical Zephyr degrees (18-20)
    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = csr_col_ind[p];
        int8_t Jij = csr_J_vals[p];
        int8_t neighbor_spin = get_spin_packed(neighbor, packed_state);
        energy += neighbor_spin * Jij;
    }

    // Add h field contribution
    int8_t h_var = h_vals[var];
    energy += h_var;

    // Delta energy = -2 * state[var] * energy
    // Cast to int8_t (safe because -42 <= delta_E ≤ 42 for all topologies)
    // Remember, it's a single flip, and J values are only ±1, so max energy is degree (~20 for Zephyr) + h (±1)
    // Therefore, max delta_E = 2 * (degree + |h|) = 2 * 21 = 42, well within int8_t range
    int8_t var_spin = get_spin_packed(var, packed_state);
    return (int8_t)(-2 * var_spin * energy);
}

// Pure SA kernel with delta energy optimization
// Batched multi-problem evaluation for reduced kernel overhead
// Each problem gets its own CSR structure, threads process different problems in parallel
//
// Supports chunked dispatch for GPU yielding: when beta_start > 0,
// state is loaded from persistent buffers instead of random init.
// This lets the Python side break the full beta schedule into small
// chunks (~10 betas each) with duty-cycle sleeps between dispatches,
// keeping the GPU responsive on Apple Silicon's shared architecture.
kernel void pure_simulated_annealing(
    device const int* csr_row_ptr [[buffer(0)]],          // Concatenated [N+1] for all problems
    device const int* csr_col_ind [[buffer(1)]],          // Concatenated [nnz] for all problems
    device const int8_t* csr_J_vals [[buffer(2)]],        // Concatenated [nnz] for all problems
    device const int* row_ptr_offsets [[buffer(3)]],      // [num_problems+1] offset into csr_row_ptr
    device const int* col_ind_offsets [[buffer(4)]],      // [num_problems+1] offset into csr_col_ind

    constant int& N [[buffer(5)]],                         // number of spins (same for all problems)
    constant int& num_betas [[buffer(6)]],                 // number of beta values in full schedule
    constant int& sweeps_per_beta [[buffer(7)]],           // sweeps per beta
    constant uint& base_seed [[buffer(8)]],                // RNG seed

    constant int& num_threads [[buffer(12)]],              // total number of threads
    constant int& num_problems [[buffer(13)]],             // number of batched problems
    constant int& num_reads [[buffer(14)]],                // reads per individual problem

    device const float* beta_schedule [[buffer(9)]],       // [num_betas]
    device const int8_t* csr_h_vals [[buffer(15)]],        // Concatenated [N] for all problems - h field values

    // Outputs only (no working memory needed - using thread-local)
    device int8_t* final_samples [[buffer(10)]],           // [num_threads * (N+7)/8] - bit-packed
    device int* final_energies [[buffer(11)]],             // [num_threads]

    // Chunked dispatch parameters for GPU yielding
    constant int& beta_start [[buffer(16)]],               // first beta index this chunk (0 = init)
    constant int& beta_count [[buffer(17)]],               // betas to process this chunk
    device int8_t* persistent_state [[buffer(18)]],        // [num_threads * packed_size] persisted packed state
    device int8_t* persistent_delta_energy [[buffer(19)]], // [num_threads * N] persisted delta_energy
    device uint* persistent_rng [[buffer(20)]],            // [num_threads] persisted RNG state
    device int* persistent_energy [[buffer(21)]],          // [num_threads] persisted current_energy

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    // Compute global thread ID
    uint thread_id = threadgroup_pos.x * threads_per_group.x + thread_pos_in_group.x;

    // Early exit if thread_id is out of bounds
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
    device const int8_t* my_h_vals = &csr_h_vals[problem_id * N];  // h values for this problem

    int n = N;
    int sweeps_per_beta_val = sweeps_per_beta;
    int packed_size = (n + 7) / 8;  // bytes needed for bit-packed state

    // Thread-local memory for state and delta_energy
    // Bit-packed state: (N+7)/8 bytes
    // Delta energy: N bytes (int8)
    // Total: ~5 KiB for N=4593 (fits in thread-local memory)
    thread int8_t packed_state[4593 / 8 + 1];  // Bit-packed state
    thread int8_t delta_energy[4593];           // Delta energy (int8)

    uint rng_state;
    int current_energy;

    if (beta_start == 0) {
        // ── First chunk: random initialization ──────────────
        rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

        // Generate random initial state (bit-packed)
        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            packed_state[byte_idx] = 0;
        }
        for (int var = 0; var < n; var++) {
            uint rand_val = xorshift32(rng_state);
            int8_t spin = (rand_val & 1) ? -1 : 1;
            set_spin_packed(var, spin, packed_state);
        }

        // Build initial delta_energy array
        for (int var = 0; var < n; var++) {
            delta_energy[var] = get_flip_energy(var, packed_state, my_csr_row_ptr, my_csr_col_ind, my_csr_J_vals, my_h_vals);
        }

        // Compute initial energy
        current_energy = 0;
        for (int i = 0; i < n; i++) {
            int8_t spin_i = get_spin_packed(i, packed_state);
            current_energy += my_h_vals[i] * spin_i;
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
    } else {
        // ── Continuation chunk: load from persistent buffers ──
        device const int8_t* src_state = &persistent_state[thread_id * packed_size];
        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            packed_state[byte_idx] = src_state[byte_idx];
        }

        device const int8_t* src_de = &persistent_delta_energy[thread_id * n];
        for (int var = 0; var < n; var++) {
            delta_energy[var] = src_de[var];
        }

        rng_state = persistent_rng[thread_id];
        current_energy = persistent_energy[thread_id];
    }

    // Perform sweeps for this chunk's beta range
    int chunk_end = min(beta_start + beta_count, num_betas);
    for (int beta_idx = beta_start; beta_idx < chunk_end; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        // D-Wave optimization: threshold to skip impossible flips
        // log(1/2^32) ≈ -22.18, so if delta_E > 22.18/beta, probability is < 2^-32
        float threshold = 22.18f / beta;

        for (int sweep = 0; sweep < sweeps_per_beta_val; sweep++) {
            // Sequential variable ordering (matching D-Wave default)
            for (int var = 0; var < n; var++) {
                int8_t de = delta_energy[var];

                // Skip if delta energy too large (D-Wave optimization)
                if (de >= threshold) continue;

                bool flip_spin = false;

                // Metropolis-Hastings acceptance rule
                if (de <= 0) {
                    // Always accept energy-lowering flips
                    flip_spin = true;
                } else {
                    // Get random number
                    uint rand_val = xorshift32(rng_state);

                    // Accept with probability exp(-delta_energy * beta)
                    float prob = exp(-float(de) * beta);
                    float rand_normalized = float(rand_val) / 4294967295.0f;  // 2^32 - 1

                    if (prob > rand_normalized) {
                        flip_spin = true;
                    }
                }

                if (flip_spin) {
                    current_energy += de;

                    // Get current spin value
                    int8_t var_spin = get_spin_packed(var, packed_state);

                    // Update delta energies of all neighbors
                    int8_t multiplier = 4 * var_spin;
                    int start = my_csr_row_ptr[var];
                    int end = my_csr_row_ptr[var + 1];

                    // Aggressive unrolling for typical Zephyr degrees (18-20)
                    #pragma unroll 20
                    for (int p = start; p < end; ++p) {
                        int neighbor = my_csr_col_ind[p];
                        int8_t Jij = my_csr_J_vals[p];
                        int8_t neighbor_spin = get_spin_packed(neighbor, packed_state);
                        delta_energy[neighbor] += multiplier * Jij * neighbor_spin;
                    }

                    // Flip the spin and negate its delta energy
                    flip_spin_packed(var, packed_state);
                    delta_energy[var] = -de;
                }
            }
        }
    }

    // Persist state for next chunk (always write — the Python side
    // decides whether there will be a next chunk or not)
    {
        device int8_t* dst_state = &persistent_state[thread_id * packed_size];
        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            dst_state[byte_idx] = packed_state[byte_idx];
        }

        device int8_t* dst_de = &persistent_delta_energy[thread_id * n];
        for (int var = 0; var < n; var++) {
            dst_de[var] = delta_energy[var];
        }

        persistent_rng[thread_id] = rng_state;
        persistent_energy[thread_id] = current_energy;
    }

    // Write final output (always — the Python side reads only after
    // the last chunk, but writing every time is cheaper than branching)
    device int8_t* output = &final_samples[thread_id * packed_size];
    for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
        output[byte_idx] = packed_state[byte_idx];
    }

    final_energies[thread_id] = current_energy;
}
