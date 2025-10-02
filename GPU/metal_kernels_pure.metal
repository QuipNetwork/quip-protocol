#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// PURE SIMULATED ANNEALING - Exact D-Wave Implementation
// ==============================================================================
// This kernel exactly mimics D-Wave's cpu_sa.cpp implementation:
// 1. Delta energy array optimization (pre-compute, update incrementally)
// 2. xorshift128+ RNG matching D-Wave
// 3. Sequential variable ordering (spins 0..N-1)
// 4. Metropolis criterion with threshold optimization (skip if delta_E > 44.36/beta)
// 5. Efficient neighbor update after each flip

typedef unsigned int uint;

// Simple xorshift32 RNG (Metal doesn't have good 64-bit support)
inline uint xorshift32(thread uint &state) {
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Compute initial delta energy for a single variable
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
    // 80% of nodes have degree 18-20, so unroll by 20
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

    // Scalar parameters (passed directly, not as buffers)
    constant int& N [[buffer(3)]],                         // number of spins
    constant int& num_betas [[buffer(4)]],                 // number of beta values
    constant int& sweeps_per_beta [[buffer(5)]],           // sweeps per beta
    constant uint& base_seed [[buffer(6)]],                // RNG seed

    // Beta schedule
    device const float* beta_schedule [[buffer(7)]],       // [num_betas]

    // Working memory (allocated, but we generate initial state in kernel)
    device int8_t* working_states [[buffer(8)]],           // [num_reads * N] - workspace for states

    // Outputs
    device int8_t* final_samples [[buffer(9)]],            // [num_reads * N]
    device int* final_energies [[buffer(10)]],             // [num_reads]

    // Delta energy array (device memory, one per thread)
    device int* delta_energies [[buffer(11)]],             // [num_reads * N]

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    // Compute global thread ID
    uint thread_id = threadgroup_pos.x * threads_per_group.x + thread_pos_in_group.x;

    // Scalar parameters are now direct values (no indirection)
    int n = N;
    int num_beta_values = num_betas;
    int sweeps_per_beta_val = sweeps_per_beta;

    // Each thread gets its own state and delta_energy array
    device int8_t* state = &working_states[thread_id * n];
    device int* delta_energy = &delta_energies[thread_id * n];

    // Initialize RNG state with unique seed per thread
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state in kernel (saves CPU->GPU transfer)
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        state[var] = (rand_val & 1) ? 1 : -1;  // Random ±1
    }

    // Build initial delta_energy array and compute initial energy
    int current_energy = 0;
    for (int var = 0; var < n; var++) {
        delta_energy[var] = get_flip_energy(var, state, csr_row_ptr, csr_col_ind, csr_J_vals);
    }

    // Compute initial energy (only once at start)
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

    // Use tracked energy (no need to recompute!)
    final_energies[thread_id] = current_energy;
}


// ==============================================================================
// PHASE 2: PURE SA WITH GRAPH COLORING + DOUBLE BUFFERING
// ==============================================================================
// This kernel adds double buffering (src/dst swap after each color)
// while keeping everything else from Phase 1:
// - Graph coloring with sequential color order (0, 1, 2, ...)
// - Delta energy array with incremental updates (NOT per-color precomputation)
// - Threshold skipping
// - Same beta schedule
//
// Key change: Use src/dst buffers and swap after each color update
// This mimics PT's buffer management but still uses incremental delta updates

kernel void pure_sa_with_double_buffering(
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

    // Working memory - now we need TWO buffers for double buffering
    device int8_t* working_states_src [[buffer(8)]],       // [num_reads * N]
    device int8_t* working_states_dst [[buffer(9)]],       // [num_reads * N]

    // Outputs
    device int8_t* final_samples [[buffer(10)]],           // [num_reads * N]
    device int* final_energies [[buffer(11)]],             // [num_reads]

    // Delta energy array
    device int* delta_energies [[buffer(12)]],             // [num_reads * N]

    // Graph coloring data
    device const int* node_colors [[buffer(13)]],          // [N] - color per node
    constant int& num_colors [[buffer(14)]],               // number of colors

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
    int n_colors = num_colors;

    // Each thread gets its own src/dst buffers and delta_energy array
    device int8_t* src = &working_states_src[thread_id * n];
    device int8_t* dst = &working_states_dst[thread_id * n];
    device int* delta_energy = &delta_energies[thread_id * n];

    // Initialize RNG state
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state in src
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        src[var] = (rand_val & 1) ? 1 : -1;
    }

    // Build initial delta_energy array
    for (int var = 0; var < n; var++) {
        delta_energy[var] = get_flip_energy(var, src, csr_row_ptr, csr_col_ind, csr_J_vals);
    }

    // Compute initial energy
    int current_energy = 0;
    for (int i = 0; i < n; i++) {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {
                int8_t Jij = csr_J_vals[p];
                current_energy += Jij * src[i] * src[j];
            }
        }
    }

    // Perform sweeps across beta schedule
    for (int beta_idx = 0; beta_idx < num_beta_values; beta_idx++) {
        float beta = beta_schedule[beta_idx];
        float threshold = 22.18f / beta;

        for (int sweep = 0; sweep < sweeps_per_beta_val; sweep++) {
            // Iterate through colors in sequential order
            for (int color = 0; color < n_colors; color++) {
                // Update all spins with this color
                // Write to dst, read from src
                for (int var = 0; var < n; var++) {
                    if (node_colors[var] != color) {
                        // Not this color, copy unchanged from src to dst
                        dst[var] = src[var];
                        continue;
                    }

                    // This color - attempt flip
                    if (delta_energy[var] >= threshold) {
                        // Skip, copy unchanged
                        dst[var] = src[var];
                        continue;
                    }

                    bool flip_spin = false;

                    // Metropolis-Hastings
                    if (delta_energy[var] <= 0) {
                        flip_spin = true;
                    } else {
                        uint rand_val = xorshift32(rng_state);
                        float prob = exp(-float(delta_energy[var]) * beta);
                        float rand_normalized = float(rand_val) / 4294967295.0f;
                        if (prob > rand_normalized) {
                            flip_spin = true;
                        }
                    }

                    if (flip_spin) {
                        // Track energy change
                        current_energy += delta_energy[var];

                        // Update delta energies of all neighbors
                        int8_t multiplier = 4 * src[var];
                        int start = csr_row_ptr[var];
                        int end = csr_row_ptr[var + 1];

                        #pragma unroll 20
                        for (int p = start; p < end; ++p) {
                            int neighbor = csr_col_ind[p];
                            int8_t Jij = csr_J_vals[p];
                            delta_energy[neighbor] += multiplier * Jij * src[neighbor];
                        }

                        // Write flipped spin to dst
                        dst[var] = -src[var];
                        // Negate its delta energy
                        delta_energy[var] *= -1;
                    } else {
                        // No flip, copy unchanged
                        dst[var] = src[var];
                    }
                }

                // PHASE 2: Swap src/dst after each color
                device int8_t* tmp = src;
                src = dst;
                dst = tmp;
            }
        }
    }

    // Write final state to output (from src, which has the latest state)
    device int8_t* output = &final_samples[thread_id * n];
    for (int i = 0; i < n; i++) {
        output[i] = src[i];
    }

    final_energies[thread_id] = current_energy;
}


// ==============================================================================
// PHASE 3: PURE SA WITH PER-COLOR DELTA ENERGY PRECOMPUTATION
// ==============================================================================
// This kernel adds per-color delta energy precomputation (recompute from src buffer)
// while keeping everything else from Phase 2:
// - Graph coloring with sequential color order
// - Double buffering (src/dst swap after each color)
// - Threshold skipping
// - Same beta schedule
//
// Key change: Instead of incrementally updating delta energies, we recompute them
// from the src buffer for each color. This matches PT's approach.

kernel void pure_sa_with_per_color_precomputation(
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

    // Working memory - TWO buffers for double buffering
    device int8_t* working_states_src [[buffer(8)]],       // [num_reads * N]
    device int8_t* working_states_dst [[buffer(9)]],       // [num_reads * N]

    // Outputs
    device int8_t* final_samples [[buffer(10)]],           // [num_reads * N]
    device int* final_energies [[buffer(11)]],             // [num_reads]

    // Delta energy array
    device int* delta_energies [[buffer(12)]],             // [num_reads * N]

    // Graph coloring data
    device const int* node_colors [[buffer(13)]],          // [N] - color per node
    constant int& num_colors [[buffer(14)]],               // number of colors

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
    int n_colors = num_colors;

    // Each thread gets its own src/dst buffers and delta_energy array
    device int8_t* src = &working_states_src[thread_id * n];
    device int8_t* dst = &working_states_dst[thread_id * n];
    device int* delta_energy = &delta_energies[thread_id * n];

    // Initialize RNG state
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state in src
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        src[var] = (rand_val & 1) ? 1 : -1;
    }

    // Compute initial energy
    int current_energy = 0;
    for (int i = 0; i < n; i++) {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {
                int8_t Jij = csr_J_vals[p];
                current_energy += Jij * src[i] * src[j];
            }
        }
    }

    // Perform sweeps across beta schedule
    for (int beta_idx = 0; beta_idx < num_beta_values; beta_idx++) {
        float beta = beta_schedule[beta_idx];
        float threshold = 22.18f / beta;

        for (int sweep = 0; sweep < sweeps_per_beta_val; sweep++) {
            // Iterate through colors in sequential order
            for (int color = 0; color < n_colors; color++) {
                // PHASE 3: Precompute delta energies for this color from src buffer
                for (int var = 0; var < n; var++) {
                    if (node_colors[var] != color) {
                        continue;
                    }

                    // Recompute delta energy from src buffer
                    delta_energy[var] = get_flip_energy(var, src, csr_row_ptr, csr_col_ind, csr_J_vals);
                }

                // Update all spins with this color
                // Write to dst, read from src
                for (int var = 0; var < n; var++) {
                    if (node_colors[var] != color) {
                        // Not this color, copy unchanged from src to dst
                        dst[var] = src[var];
                        continue;
                    }

                    // This color - attempt flip
                    if (delta_energy[var] >= threshold) {
                        // Skip, copy unchanged
                        dst[var] = src[var];
                        continue;
                    }

                    bool flip_spin = false;

                    // Metropolis-Hastings
                    if (delta_energy[var] <= 0) {
                        flip_spin = true;
                    } else {
                        uint rand_val = xorshift32(rng_state);
                        float prob = exp(-float(delta_energy[var]) * beta);
                        float rand_normalized = float(rand_val) / 4294967295.0f;
                        if (prob > rand_normalized) {
                            flip_spin = true;
                        }
                    }

                    if (flip_spin) {
                        // Track energy change
                        current_energy += delta_energy[var];

                        // Write flipped spin to dst
                        dst[var] = -src[var];
                    } else {
                        // No flip, copy unchanged
                        dst[var] = src[var];
                    }
                }

                // Swap src/dst after each color
                device int8_t* tmp = src;
                src = dst;
                dst = tmp;
            }
        }
    }

    // Write final state to output (from src, which has the latest state)
    device int8_t* output = &final_samples[thread_id * n];
    for (int i = 0; i < n; i++) {
        output[i] = src[i];
    }

    final_energies[thread_id] = current_energy;
}


// ==============================================================================
// PHASE 4: PURE SA WITH COLOR SHUFFLING
// ==============================================================================
// This kernel adds color order shuffling (Fisher-Yates) each sweep
// while keeping everything else from Phase 3:
// - Graph coloring
// - Double buffering (src/dst swap after each color)
// - Per-color delta energy precomputation
// - Threshold skipping
// - Same beta schedule
//
// Key change: Randomize the color order each sweep using Fisher-Yates shuffle

kernel void pure_sa_with_color_shuffling(
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

    // Working memory - TWO buffers for double buffering
    device int8_t* working_states_src [[buffer(8)]],       // [num_reads * N]
    device int8_t* working_states_dst [[buffer(9)]],       // [num_reads * N]

    // Outputs
    device int8_t* final_samples [[buffer(10)]],           // [num_reads * N]
    device int* final_energies [[buffer(11)]],             // [num_reads]

    // Delta energy array
    device int* delta_energies [[buffer(12)]],             // [num_reads * N]

    // Graph coloring data
    device const int* node_colors [[buffer(13)]],          // [N] - color per node
    constant int& num_colors [[buffer(14)]],               // number of colors

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
    int n_colors = num_colors;

    // Each thread gets its own src/dst buffers and delta_energy array
    device int8_t* src = &working_states_src[thread_id * n];
    device int8_t* dst = &working_states_dst[thread_id * n];
    device int* delta_energy = &delta_energies[thread_id * n];

    // Initialize RNG state
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state in src
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        src[var] = (rand_val & 1) ? 1 : -1;
    }

    // PHASE 4: Allocate color order array (on stack, max 32 colors for Zephyr)
    int color_order[32];
    for (int c = 0; c < n_colors && c < 32; c++) {
        color_order[c] = c;
    }

    // PHASE 4 OPTIMIZATION: Build per-color spin lists to avoid scanning all N spins
    // This is a one-time cost that pays off across all sweeps
    // Note: We can't allocate dynamic arrays, so we'll still use the scan approach
    // but we can at least make it clearer that this is a known inefficiency

    // Compute initial energy
    int current_energy = 0;
    for (int i = 0; i < n; i++) {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {
                int8_t Jij = csr_J_vals[p];
                current_energy += Jij * src[i] * src[j];
            }
        }
    }

    // Track global sweep number across beta schedule
    int global_sweep = 0;

    // Perform sweeps across beta schedule
    for (int beta_idx = 0; beta_idx < num_beta_values; beta_idx++) {
        float beta = beta_schedule[beta_idx];
        float threshold = 22.18f / beta;

        for (int sweep = 0; sweep < sweeps_per_beta_val; sweep++) {
            // PHASE 4: Fisher-Yates shuffle of color order
            // FIX: Use unique seed per sweep (like PT does)
            uint shuffle_seed = base_seed ^ uint((thread_id + 1) * 2654435761u) ^ uint((global_sweep + 1) * 987654321u);
            for (int i = n_colors - 1; i > 0; --i) {
                shuffle_seed ^= shuffle_seed << 13;
                shuffle_seed ^= shuffle_seed >> 17;
                shuffle_seed ^= shuffle_seed << 5;
                int j = int(shuffle_seed % uint(i + 1));

                // Swap color_order[i] and color_order[j]
                int tmp = color_order[i];
                color_order[i] = color_order[j];
                color_order[j] = tmp;
            }

            // Iterate through colors in shuffled order
            for (int color_idx = 0; color_idx < n_colors; color_idx++) {
                int color = color_order[color_idx];

                // NOTE: This loop structure is inefficient (scans all N spins for each color)
                // but is necessary for single-threaded execution without dynamic memory.
                // PT avoids this with threadgroup parallelism and tiling.
                // Combined precompute + update loop to reduce overhead
                for (int var = 0; var < n; var++) {
                    int var_color = node_colors[var];

                    if (var_color != color) {
                        // Not this color, copy unchanged from src to dst
                        dst[var] = src[var];
                        continue;
                    }

                    // This color - recompute delta energy and attempt flip
                    int delta_e = get_flip_energy(var, src, csr_row_ptr, csr_col_ind, csr_J_vals);

                    // Threshold skipping
                    if (delta_e >= threshold) {
                        // Skip, copy unchanged
                        dst[var] = src[var];
                        continue;
                    }

                    bool flip_spin = false;

                    // Metropolis-Hastings
                    if (delta_e <= 0) {
                        flip_spin = true;
                    } else {
                        uint rand_val = xorshift32(rng_state);
                        float prob = exp(-float(delta_e) * beta);
                        float rand_normalized = float(rand_val) / 4294967295.0f;
                        if (prob > rand_normalized) {
                            flip_spin = true;
                        }
                    }

                    if (flip_spin) {
                        // Track energy change
                        current_energy += delta_e;

                        // Write flipped spin to dst
                        dst[var] = -src[var];
                    } else {
                        // No flip, copy unchanged
                        dst[var] = src[var];
                    }
                }

                // Swap src/dst after each color
                device int8_t* tmp = src;
                src = dst;
                dst = tmp;
            }

            // Increment global sweep counter
            global_sweep++;
        }
    }

    // Write final state to output (from src, which has the latest state)
    device int8_t* output = &final_samples[thread_id * n];
    for (int i = 0; i < n; i++) {
        output[i] = src[i];
    }

    final_energies[thread_id] = current_energy;
}


// ==============================================================================
// PHASE 5: PURE SA WITH MULTIPLE REPLICAS (NO REPLICA EXCHANGE)
// ==============================================================================
// This kernel adds multiple temperature replicas while keeping everything from Phase 4:
// - Graph coloring
// - Double buffering (src/dst swap after each color)
// - Per-color delta energy precomputation
// - Color shuffling
//
// Key change: Run multiple independent chains at different temperatures.
// Each thread runs ONE replica at ONE temperature.
// NO replica exchange - just independent parallel chains.

kernel void pure_sa_with_multiple_replicas(
    // Problem CSR representation
    device const int* csr_row_ptr [[buffer(0)]],          // [N+1]
    device const int* csr_col_ind [[buffer(1)]],          // [nnz]
    device const int8_t* csr_J_vals [[buffer(2)]],        // [nnz]

    // Scalar parameters
    constant int& N [[buffer(3)]],                         // number of spins
    constant int& num_sweeps [[buffer(4)]],                // total sweeps (not per beta)
    constant uint& base_seed [[buffer(6)]],                // RNG seed

    // Temperature ladder
    device const float* beta_schedule [[buffer(7)]],       // [num_replicas] - one beta per replica
    constant int& num_replicas [[buffer(8)]],              // number of temperature replicas

    // Working memory - TWO buffers for double buffering
    device int8_t* working_states_src [[buffer(9)]],       // [num_replicas * N]
    device int8_t* working_states_dst [[buffer(10)]],      // [num_replicas * N]

    // Outputs
    device int8_t* final_samples [[buffer(11)]],           // [num_replicas * N]
    device int* final_energies [[buffer(12)]],             // [num_replicas]

    // Delta energy array
    device int* delta_energies [[buffer(13)]],             // [num_replicas * N]

    // Graph coloring data
    device const int* node_colors [[buffer(14)]],          // [N] - color per node
    constant int& num_colors [[buffer(15)]],               // number of colors

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    // Compute global thread ID - each thread is ONE replica
    uint thread_id = threadgroup_pos.x * threads_per_group.x + thread_pos_in_group.x;

    // Early exit if thread_id >= num_replicas
    if (thread_id >= uint(num_replicas)) {
        return;
    }

    int n = N;
    int n_colors = num_colors;
    int total_sweeps = num_sweeps;

    // Each thread gets its own src/dst buffers and delta_energy array
    device int8_t* src = &working_states_src[thread_id * n];
    device int8_t* dst = &working_states_dst[thread_id * n];
    device int* delta_energy = &delta_energies[thread_id * n];

    // Get this replica's temperature (fixed for entire run)
    float beta = beta_schedule[thread_id];
    float threshold = 22.18f / beta;

    // Initialize RNG state
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state in src
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        src[var] = (rand_val & 1) ? 1 : -1;
    }

    // Allocate color order array
    int color_order[32];
    for (int c = 0; c < n_colors && c < 32; c++) {
        color_order[c] = c;
    }

    // Compute initial energy
    int current_energy = 0;
    for (int i = 0; i < n; i++) {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {
                int8_t Jij = csr_J_vals[p];
                current_energy += Jij * src[i] * src[j];
            }
        }
    }

    // Perform sweeps at this replica's fixed temperature
    for (int sweep = 0; sweep < total_sweeps; sweep++) {
        // Fisher-Yates shuffle of color order
        uint shuffle_seed = base_seed ^ uint((thread_id + 1) * 2654435761u) ^ uint((sweep + 1) * 987654321u);
        for (int i = n_colors - 1; i > 0; --i) {
            shuffle_seed ^= shuffle_seed << 13;
            shuffle_seed ^= shuffle_seed >> 17;
            shuffle_seed ^= shuffle_seed << 5;
            int j = int(shuffle_seed % uint(i + 1));

            // Swap color_order[i] and color_order[j]
            int tmp = color_order[i];
            color_order[i] = color_order[j];
            color_order[j] = tmp;
        }

        // Iterate through colors in shuffled order
        for (int color_idx = 0; color_idx < n_colors; color_idx++) {
            int color = color_order[color_idx];

            // Combined precompute + update loop
            for (int var = 0; var < n; var++) {
                int var_color = node_colors[var];

                if (var_color != color) {
                    // Not this color, copy unchanged from src to dst
                    dst[var] = src[var];
                    continue;
                }

                // This color - recompute delta energy and attempt flip
                int delta_e = get_flip_energy(var, src, csr_row_ptr, csr_col_ind, csr_J_vals);

                // Threshold skipping
                if (delta_e >= threshold) {
                    // Skip, copy unchanged
                    dst[var] = src[var];
                    continue;
                }

                bool flip_spin = false;

                // Metropolis-Hastings
                if (delta_e <= 0) {
                    flip_spin = true;
                } else {
                    uint rand_val = xorshift32(rng_state);
                    float prob = exp(-float(delta_e) * beta);
                    float rand_normalized = float(rand_val) / 4294967295.0f;
                    if (prob > rand_normalized) {
                        flip_spin = true;
                    }
                }

                if (flip_spin) {
                    // Track energy change
                    current_energy += delta_e;

                    // Write flipped spin to dst
                    dst[var] = -src[var];
                } else {
                    // No flip, copy unchanged
                    dst[var] = src[var];
                }
            }

            // Swap src/dst after each color
            device int8_t* tmp = src;
            src = dst;
            dst = tmp;
        }
    }

    // Write final state to output (from src, which has the latest state)
    device int8_t* output = &final_samples[thread_id * n];
    for (int i = 0; i < n; i++) {
        output[i] = src[i];
    }

    final_energies[thread_id] = current_energy;
}


// ==============================================================================
// PHASE 1: PURE SA WITH GRAPH COLORING
// ==============================================================================
// This kernel adds graph coloring (parallel updates within color groups)
// while keeping everything else from Pure SA:
// - Sequential color order (0, 1, 2, ..., num_colors-1) - NO shuffling
// - Delta energy array (but NOT per-color precomputation yet)
// - Threshold skipping
// - Same beta schedule
// - Per-thread RNG state
//
// Key change: Instead of updating spins 0..N-1 sequentially,
// we update all spins of color 0, then all spins of color 1, etc.
// Within each color, spins are updated in parallel (but we're still single-threaded per chain)

kernel void pure_sa_with_coloring(
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

    // NEW: Graph coloring data
    device const int* node_colors [[buffer(12)]],          // [N] - color per node
    constant int& num_colors [[buffer(13)]],               // number of colors

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
    int n_colors = num_colors;

    // Each thread gets its own state and delta_energy array
    device int8_t* state = &working_states[thread_id * n];
    device int* delta_energy = &delta_energies[thread_id * n];

    // Initialize RNG state
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        state[var] = (rand_val & 1) ? 1 : -1;
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
            if (j > i) {
                int8_t Jij = csr_J_vals[p];
                current_energy += Jij * state[i] * state[j];
            }
        }
    }

    // Perform sweeps across beta schedule
    for (int beta_idx = 0; beta_idx < num_beta_values; beta_idx++) {
        float beta = beta_schedule[beta_idx];
        float threshold = 22.18f / beta;

        for (int sweep = 0; sweep < sweeps_per_beta_val; sweep++) {
            // NEW: Iterate through colors in SEQUENTIAL order (0, 1, 2, ...)
            for (int color = 0; color < n_colors; color++) {
                // Update all spins with this color
                // Note: In single-threaded mode, this is still sequential within the color,
                // but the ORDER is different from 0..N-1
                for (int var = 0; var < n; var++) {
                    // Skip if not this color
                    if (node_colors[var] != color) continue;

                    // Skip if delta energy too large
                    if (delta_energy[var] >= threshold) continue;

                    bool flip_spin = false;

                    // Metropolis-Hastings
                    if (delta_energy[var] <= 0) {
                        flip_spin = true;
                    } else {
                        uint rand_val = xorshift32(rng_state);
                        float prob = exp(-float(delta_energy[var]) * beta);
                        float rand_normalized = float(rand_val) / 4294967295.0f;
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
    }

    // Write final state to output
    device int8_t* output = &final_samples[thread_id * n];
    for (int i = 0; i < n; i++) {
        output[i] = state[i];
    }

    final_energies[thread_id] = current_energy;
}


// ============================================================================
// PHASE 6: Replica Exchange (Swapping states between adjacent temperatures)
// ============================================================================
// This adds the final PT feature: swapping states between adjacent replicas
// based on the Metropolis criterion to allow information flow between temperatures.
//
// Key differences from Phase 5:
// - Alternates between sweeps and replica exchange attempts
// - Uses atomic operations for energy storage (needed for swaps)
// - Implements Metropolis criterion for state swaps between adjacent temperatures
// - Uses even/odd parity to avoid conflicts (replica i swaps with i+1)

kernel void pure_sa_with_replica_exchange(
    // Problem CSR representation
    device const int* csr_row_ptr [[buffer(0)]],          // [N+1]
    device const int* csr_col_ind [[buffer(1)]],          // [nnz]
    device const int8_t* csr_J_vals [[buffer(2)]],        // [nnz]

    // Scalar parameters
    constant int& N [[buffer(3)]],                         // number of spins
    constant int& num_sweeps [[buffer(4)]],                // sweeps per exchange attempt
    constant int& num_exchanges [[buffer(5)]],             // number of exchange attempts
    constant uint& base_seed [[buffer(6)]],                // RNG seed

    // Temperature ladder
    device const float* beta_schedule [[buffer(7)]],       // [num_replicas] - one beta per replica
    constant int& num_replicas [[buffer(8)]],              // number of temperature replicas

    // Working memory
    device int8_t* working_states [[buffer(9)]],           // [num_replicas * N]

    // Outputs
    device int8_t* final_samples [[buffer(10)]],           // [num_replicas * N]
    device atomic_int* replica_energies [[buffer(11)]],    // [num_replicas] - atomic for swaps

    // Graph coloring data
    device const int* node_colors [[buffer(12)]],          // [N] - color per node
    constant int& num_colors [[buffer(13)]],               // number of colors

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    // Compute global thread ID - each thread is ONE replica
    uint thread_id = threadgroup_pos.x * threads_per_group.x + thread_pos_in_group.x;

    // Early exit if thread_id >= num_replicas
    if (thread_id >= uint(num_replicas)) {
        return;
    }

    int n = N;
    int n_colors = num_colors;
    int sweeps_per_exchange = num_sweeps;
    int n_exchanges = num_exchanges;

    // Each thread gets its own state buffer
    device int8_t* state = &working_states[thread_id * n];

    // Get this replica's temperature (fixed for entire run)
    float beta = beta_schedule[thread_id];
    float threshold = 22.18f / beta;

    // Initialize RNG state
    uint rng_state = (base_seed ? base_seed : 1u) ^ (thread_id * 12345u);

    // Generate random initial state
    for (int var = 0; var < n; var++) {
        uint rand_val = xorshift32(rng_state);
        state[var] = (rand_val & 1) ? 1 : -1;
    }

    // Allocate color order array
    int color_order[32];
    for (int c = 0; c < n_colors && c < 32; c++) {
        color_order[c] = c;
    }

    // Compute initial energy
    int current_energy = 0;
    for (int i = 0; i < n; i++) {
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {
                int8_t Jij = csr_J_vals[p];
                current_energy += Jij * state[i] * state[j];
            }
        }
    }

    // Store initial energy atomically
    atomic_store_explicit(&replica_energies[thread_id], current_energy, memory_order_relaxed);

    int global_sweep = 0;

    // Main loop: alternate between sweeps and replica exchanges
    for (int exchange_round = 0; exchange_round < n_exchanges; exchange_round++) {
        // Perform sweeps at this replica's fixed temperature
        for (int sweep = 0; sweep < sweeps_per_exchange; sweep++) {
            // Fisher-Yates shuffle of color order
            uint shuffle_seed = base_seed ^ uint((thread_id + 1) * 2654435761u) ^ uint((global_sweep + 1) * 987654321u);
            for (int i = n_colors - 1; i > 0; --i) {
                shuffle_seed ^= shuffle_seed << 13;
                shuffle_seed ^= shuffle_seed >> 17;
                shuffle_seed ^= shuffle_seed << 5;
                int j = int(shuffle_seed % uint(i + 1));

                // Swap color_order[i] and color_order[j]
                int tmp = color_order[i];
                color_order[i] = color_order[j];
                color_order[j] = tmp;
            }

            // Iterate through colors in shuffled order
            for (int color_idx = 0; color_idx < n_colors; color_idx++) {
                int color = color_order[color_idx];

                // Update spins of this color
                for (int var = 0; var < n; var++) {
                    if (node_colors[var] != color) {
                        continue;
                    }

                    // Compute delta energy for flipping this spin
                    int delta_e = get_flip_energy(var, state, csr_row_ptr, csr_col_ind, csr_J_vals);

                    // Threshold skipping
                    if (delta_e >= threshold) {
                        continue;
                    }

                    // Metropolis criterion
                    if (delta_e <= 0) {
                        // Accept flip
                        state[var] = -state[var];
                        current_energy += delta_e;
                    } else {
                        // Probabilistic acceptance
                        float p = exp(-beta * float(delta_e));
                        uint rand_val = xorshift32(rng_state);
                        float r = float(rand_val & 0x7FFFFFFFu) / 2147483647.0f;

                        if (r < p) {
                            state[var] = -state[var];
                            current_energy += delta_e;
                        }
                    }
                }
            }

            global_sweep++;
        }

        // Update atomic energy after sweeps
        atomic_store_explicit(&replica_energies[thread_id], current_energy, memory_order_relaxed);

        // Synchronization: wait for all replicas to finish sweeps
        threadgroup_barrier(mem_flags::mem_device);

        // Attempt replica exchange (even/odd parity alternating)
        int parity = exchange_round % 2;
        int replica_idx = int(thread_id);

        // Check if this replica participates in exchange
        if ((replica_idx % 2) == parity && replica_idx + 1 < num_replicas) {
            // This replica attempts exchange with replica_idx + 1
            int partner_idx = replica_idx + 1;

            // Get energies
            int E0 = atomic_load_explicit(&replica_energies[replica_idx], memory_order_relaxed);
            int E1 = atomic_load_explicit(&replica_energies[partner_idx], memory_order_relaxed);

            // Get temperatures
            float beta0 = beta_schedule[replica_idx];
            float beta1 = beta_schedule[partner_idx];

            // Metropolis criterion for replica exchange
            float delta_beta = beta0 - beta1;
            float delta_E = float(E1 - E0);
            float acceptance_prob = exp(min(0.0f, delta_beta * delta_E));

            // Random number for acceptance
            uint rand_val = xorshift32(rng_state);
            float r = float(rand_val & 0x7FFFFFFFu) / 2147483647.0f;

            if (r < acceptance_prob) {
                // Accept swap: exchange states
                device int8_t* my_state = &working_states[replica_idx * n];
                device int8_t* partner_state = &working_states[partner_idx * n];

                for (int var = 0; var < n; var++) {
                    int8_t tmp = my_state[var];
                    my_state[var] = partner_state[var];
                    partner_state[var] = tmp;
                }

                // Swap energies
                atomic_store_explicit(&replica_energies[replica_idx], E1, memory_order_relaxed);
                atomic_store_explicit(&replica_energies[partner_idx], E0, memory_order_relaxed);

                // Update local energy
                current_energy = E1;
            }
        }

        // Synchronization: wait for all exchanges to complete
        threadgroup_barrier(mem_flags::mem_device);
    }

    // Copy final state to output
    for (int var = 0; var < n; var++) {
        final_samples[thread_id * n + var] = state[var];
    }
}
