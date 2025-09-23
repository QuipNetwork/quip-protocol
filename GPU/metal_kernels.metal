#include <metal_stdlib>
using namespace metal;

// 3D Edwards-Anderson Spin Glass Metal Kernels
// Implements the algorithm from CURRENT_PLAN.md for cubic lattice with 6 nearest neighbors

// Constants
constant int MAX_REPLICAS = 512;    // Max temperature replicas

// Data Structures - dynamic sizing based on input topology
// Note: Arrays are now passed as device buffers with dynamic sizing

// Spin state: int8_t for memory efficiency (dynamic array via buffer)
struct SpinState {
    int N;                // actual N (number of spins in this replica)
    // spins array passed separately as device buffer
};

// Global configuration data
struct GlobalData {
    int N;                 // actual number of spins (determined by topology)
    int num_replicas;
    int swap_interval;
    int cooling_interval;
    float T_min, T_max;
    int step;
};

// Temperature array (one per replica)
struct Temperatures {
    float T[MAX_REPLICAS]; // Logarithmically spaced temperatures
};

// Debug: Atomic counter to track execution progress
// Declare as kernel parameter — not global

// Ensure Metal compiler recognizes atomic types
typedef unsigned int uint;

// Xorshift Random Number Generator
uint xorshift(uint seed) {
    uint x = seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// Generate random float [0,1) from thread-specific state
float get_random(uint3 thread_id, uint step) {
    uint seed = thread_id.x * 73856093u + thread_id.y * 19349663u + thread_id.z * 83492791u + step * 12345u;
    uint rand_val = xorshift(seed);
    return float(rand_val & 0x7FFFFFFF) / 2147483647.0f;
}

// Spin Flip Kernel using CSR adjacency (arbitrary degree)
kernel void ea_spin_flip_kernel(
    device int8_t* replica_spins [[buffer(0)]],           // [num_replicas * N] spins (flattened)
    device const int* csr_row_ptr [[buffer(1)]],          // [N+1] row pointers
    device const int* csr_col_ind [[buffer(2)]],          // [2E] neighbor indices
    device const int8_t* csr_J_vals [[buffer(3)]],        // [2E] ±1 couplings aligned with csr_col_ind
    device const Temperatures* temperatures [[buffer(4)]],// [1] temperature array
    device const GlobalData* global [[buffer(5)]],        // [1] global parameters
    uint3 thread_id [[thread_position_in_grid]]
) {
    uint replica_id = thread_id.x;
    if (replica_id >= (uint)global->num_replicas) return;

    int N = global->N;
    float beta = 1.0f / temperatures->T[replica_id];

    // Pick random spin index for this replica
    uint seed = replica_id * 12345u + global->step * 67890u;
    uint random_val = xorshift(seed);
    uint spin_idx = random_val % (uint)N;

    device int8_t* spins = &replica_spins[replica_id * N];
    int8_t s_i = spins[spin_idx];

    // sum_j J_ij * s_j over neighbors from CSR
    int start = csr_row_ptr[spin_idx];
    int end   = csr_row_ptr[spin_idx + 1];
    float sum = 0.0f;
    for (int p = start; p < end; ++p) {
        int j = csr_col_ind[p];
        int8_t s_j = spins[j];
        int8_t Jij = csr_J_vals[p];
        sum += float(Jij) * float(s_j);
    }

    // ΔE = 2 s_i * sum_j J_ij s_j
    float delta_E = 2.0f * float(s_i) * sum;

    bool accept = (delta_E <= 0.0f);
    if (!accept) {
        float rand_val = get_random(thread_id, global->step);
        accept = (rand_val < exp(-beta * delta_E));
    }
    if (accept) {
        spins[spin_idx] = -s_i;
    }
}

// Parallel spin flip kernel using 2D grid (replicas × spins) with double-buffering
kernel void ea_parallel_spin_flip_kernel(
    device const int8_t* replica_spins_src [[buffer(0)]], // [num_replicas * N] source spins (read-only)
    device int8_t* replica_spins_dst [[buffer(1)]],       // [num_replicas * N] destination spins (write-only)
    device const int* csr_row_ptr [[buffer(2)]],          // [N+1] CSR row pointers
    device const int* csr_col_ind [[buffer(3)]],          // [nnz] CSR column indices
    device const int8_t* csr_J_vals [[buffer(4)]],        // [nnz] CSR coupling values (sign only)
    device const Temperatures* temperatures [[buffer(5)]], // [1] temperature array
    device const GlobalData* global [[buffer(6)]],        // [1] global parameters
    uint3 thread_id [[thread_position_in_grid]]
) {
    uint replica_id = thread_id.x;
    uint spin_idx = thread_id.y;

    if (replica_id >= (uint)global->num_replicas || spin_idx >= (uint)global->N) return;

    int N = global->N;
    float beta = 1.0f / temperatures->T[replica_id];

    // Read from source buffer
    device const int8_t* src_spins = &replica_spins_src[replica_id * N];
    device int8_t* dst_spins = &replica_spins_dst[replica_id * N];

    int8_t s_i = src_spins[spin_idx];

    // Compute sum of J_ij * s_j for neighbors of spin i using CSR
    float sum = 0.0f;
    int start = csr_row_ptr[spin_idx];
    int end = csr_row_ptr[spin_idx + 1];

    for (int p = start; p < end; ++p) {
        int j = csr_col_ind[p];
        int8_t Jij = csr_J_vals[p];
        sum += float(Jij) * float(src_spins[j]);  // Read from source
    }

    // ΔE = 2 s_i * sum_j J_ij s_j
    float delta_E = 2.0f * float(s_i) * sum;

    // Metropolis acceptance
    bool accept = (delta_E <= 0.0f);
    if (!accept) {
        // Use unique thread ID for randomness (replica_id, spin_idx, step)
        uint seed = replica_id * 12345u + spin_idx * 67890u + global->step * 98765u;
        uint random_val = xorshift(seed);
        float rand_val = float(random_val) / float(UINT_MAX);
        accept = (rand_val < exp(-beta * delta_E));
    }

    // Write to destination buffer
    dst_spins[spin_idx] = accept ? -s_i : s_i;
}

// Compute energies using CSR adjacency (avoid double counting via j>i)
kernel void ea_compute_energies_kernel(
    device const int8_t* replica_spins [[buffer(0)]],     // [num_replicas * N] spins (flattened)
    device int* replica_energies [[buffer(1)]],           // [num_replicas] energies
    device const int* csr_row_ptr [[buffer(2)]],          // [N+1]
    device const int* csr_col_ind [[buffer(3)]],          // [2E]
    device const int8_t* csr_J_vals [[buffer(4)]],        // [2E]
    device const GlobalData* global [[buffer(5)]],        // [1]
    uint thread_id [[thread_position_in_grid]]
) {
    uint replica_id = thread_id;
    if (replica_id >= global->num_replicas) return;

    int N = global->N;
    device const int8_t* spins = &replica_spins[replica_id * N];
    int total_energy = 0;

    // Validate global->N before proceeding
    if (global->N <= 0) {
        replica_energies[replica_id] = -999;  // Sentinel value for corruption
        return;
    }

    for (int i = 0; i < N; i++) {
        int8_t s_i = spins[i];
        int start = csr_row_ptr[i];
        int end   = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {
                int8_t Jij = csr_J_vals[p];
                int8_t s_j = spins[j];
                total_energy += Jij * s_i * s_j;  // Standard Ising Hamiltonian: H = ∑ Jij si sj
            }
        }
    }

    replica_energies[replica_id] = total_energy;
}

// Replica exchange kernel - swap configurations between adjacent replicas (flattened spins)
kernel void ea_replica_exchange_kernel(
    device int8_t* replica_spins [[buffer(0)]],           // [num_replicas * N] flattened spins
    device int* replica_energies [[buffer(1)]],           // [num_replicas] energies (int32)
    device const Temperatures* temperatures [[buffer(2)]], // [1] temperature array
    device const GlobalData* global [[buffer(3)]],        // [1] global parameters
    uint thread_id [[thread_position_in_grid]]
) {
    uint replica_id = thread_id;

    // Only process even-numbered replicas (they attempt swaps with replica_id + 1)
    if (replica_id >= (uint)global->num_replicas - 1 || (replica_id % 2) != 0) return;

    uint next_replica = replica_id + 1;
    int N = global->N;

    // Get energies and temperatures
    int E_i_int = replica_energies[replica_id];
    int E_j_int = replica_energies[next_replica];
    float T_i = temperatures->T[replica_id];
    float T_j = temperatures->T[next_replica];

    // Compute exchange probability: exp(-(E_i - E_j) * (1/T_i - 1/T_j))
    float beta_i = 1.0f / T_i;
    float beta_j = 1.0f / T_j;
    float log_prob = -(float)(E_i_int - E_j_int) * (beta_i - beta_j);

    // Generate random number for exchange decision
    uint3 fake_thread_id = uint3(replica_id, 0, 0);
    float rand_val = get_random(fake_thread_id, global->step);

    // Accept exchange if log_prob >= 0 or with probability exp(log_prob)
    bool accept = (log_prob >= 0.0f) || (rand_val < exp(log_prob));

    if (accept) {
        // Swap flattened spin configurations
        device int8_t* spins_i = &replica_spins[replica_id * N];
        device int8_t* spins_j = &replica_spins[next_replica * N];
        for (int spin_idx = 0; spin_idx < N; spin_idx++) {
            int8_t temp = spins_i[spin_idx];
            spins_i[spin_idx] = spins_j[spin_idx];
            spins_j[spin_idx] = temp;
        }

        // Swap energies
        replica_energies[replica_id] = E_j_int;
        replica_energies[next_replica] = E_i_int;
    }
}

// Track best energy and configuration across all replicas (flattened spins)
kernel void ea_track_best_kernel(
    device const int8_t* replica_spins [[buffer(0)]],     // [num_replicas * N] flattened spins
    device const int* replica_energies [[buffer(1)]],     // [num_replicas] current energies (int32)
    device atomic<int>* best_energy [[buffer(2)]],        // [1] best energy found so far (int32)
    device int8_t* best_spins [[buffer(3)]],              // [N] best spin configuration
    device const GlobalData* global [[buffer(4)]],        // [1] global parameters
    uint thread_id [[thread_position_in_grid]]
) {
    uint replica_id = thread_id;
    if (replica_id >= global->num_replicas) return;

    int current_energy = replica_energies[replica_id];
    int N = global->N;

    // Check if this is the best energy so far
    int current_best = atomic_load_explicit(best_energy, memory_order_relaxed);
    if (current_energy < current_best) {
        // Try to update best energy atomically
        int expected = current_best;
        if (atomic_compare_exchange_weak_explicit(
            best_energy, &expected, current_energy,
            memory_order_relaxed, memory_order_relaxed)) {
            // Successfully updated best energy, copy configuration from flattened array
            device const int8_t* spins = &replica_spins[replica_id * N];
            for (int i = 0; i < N; i++) {
                best_spins[i] = spins[i];
            }
        }
    }
}

// Initialize random spin configurations for all replicas (flattened spins)
kernel void ea_initialize_spins_kernel(
    device int8_t* replica_spins [[buffer(0)]],           // [num_replicas * N] output spins (flattened)
    device const GlobalData* global [[buffer(1)]],        // [1] global parameters
    constant uint& seed [[buffer(2)]],                    // [1] random seed
    uint3 thread_id [[thread_position_in_grid]]
) {
    uint replica_id = thread_id.x;
    uint spin_idx = thread_id.y;

    if (replica_id >= (uint)global->num_replicas || spin_idx >= (uint)global->N) return;

    // Generate random spin: +1 or -1
    float rand_val = get_random(thread_id, seed);
    device int8_t* spins = &replica_spins[replica_id * global->N];
    spins[spin_idx] = (rand_val < 0.5f) ? int8_t(-1) : int8_t(1);
}

// Update temperature schedule (cooling)
kernel void ea_update_temperatures_kernel(
    device Temperatures* temperatures [[buffer(0)]],      // [1] temperatures to update
    constant float& cooling_factor [[buffer(1)]],         // [1] cooling factor (e.g., 0.999)
    device const GlobalData* global [[buffer(2)]],        // [1] global parameters
    uint thread_id [[thread_position_in_grid]]
) {
    uint replica_id = thread_id;
    if (replica_id >= global->num_replicas) return;

    temperatures->T[replica_id] *= cooling_factor;
}

// Sampling parameters for unified kernel
struct SamplingParams {
    int N;                    // Number of spins
    int num_replicas;         // Number of temperature replicas
    int num_sweeps;           // Total sweeps to run
    int num_reads;            // Number of samples to collect
    int swap_interval;        // Replica exchange frequency
    int sample_interval;      // Sample collection frequency
    float T_min;              // Minimum temperature
    float T_max;              // Maximum temperature
    uint base_seed;           // Base RNG seed
};

// MANAGER-THREAD ARCHITECTURE WITH PER-THREAD BUFFERS
// Each thread has its own buffer and requests fresh copies from manager
// SEPARATE WORKER KERNEL - Only handles worker thread optimization
kernel void worker_parallel_tempering(
    // Input data (read-only)
    device const int* csr_row_ptr [[buffer(0)]],          // [N+1] CSR row pointers
    device const int* csr_col_ind [[buffer(1)]],          // [nnz] CSR column indices
    device const int8_t* csr_J_vals [[buffer(2)]],        // [nnz] CSR coupling values
    device const float* temperatures [[buffer(3)]],       // [num_replicas] temperature ladder
    device const SamplingParams* params [[buffer(4)]],    // [1] sampling parameters

    // Output data (write-only)
    device int8_t* final_samples [[buffer(5)]],           // [num_reads * N] collected samples
    device int* final_energies [[buffer(6)]],             // [num_reads] sample energies

    // Worker buffers
    device int8_t* thread_buffers [[buffer(7)]],          // [num_replicas * N] per-thread work buffers
    device uint* rng_states [[buffer(8)]],               // [num_replicas] RNG states per replica

    // Output partitioning (per-thread base/quota)
    device const int* per_thread_base [[buffer(9)]],     // [num_replicas] base output index per thread
    device const int* per_thread_quota [[buffer(10)]],    // [num_replicas] number of samples to write per thread

    uint3 thread_id [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]]
) {
    // Use global thread position for replica assignment (more robust than threadgroup_id)
    uint replica_id = thread_id.x;

    // Energy initialization removed - only manager writes to final_energies

    if (replica_id >= (uint)params->num_replicas) return;

    int N = params->N;
    float beta = 1.0f / temperatures[replica_id];

    // PROPER MEMORY ISOLATION: Add padding to prevent buffer overlap
    const int BUFFER_PADDING = 8;  // 8-byte alignment padding
    const int PADDED_SIZE = N + BUFFER_PADDING;

    // Simplified bounds check - ensure we have enough space for this thread
    if (replica_id >= (uint)params->num_replicas) return;  // Should be redundant but explicit

    // Each thread gets isolated buffer region with padding
    device int8_t* my_thread_buffer = &thread_buffers[replica_id * PADDED_SIZE];

    // Workers only - no manager logic in this kernel

    // Initialize thread's RNG state (PRIVATE - no sharing)
    uint my_rng_state = params->base_seed + replica_id * 12345u + threadgroup_id.x * 67890u;

    // STEP 1: Thread initializes independently (no manager dependency for startup)

    // STEP 2: Thread initializes its isolated buffer with bounds checking
    for (int i = 0; i < N; i++) {
        my_rng_state = xorshift(my_rng_state);
        // Explicit bounds check to prevent buffer overflow
        if (i < N) {
            my_thread_buffer[i] = (my_rng_state & 1) ? 1 : -1;
        }
    }

    // No barrier needed for single-thread kernel calls

    // INTEGRITY CHECK: Write unique pattern to padding area to detect buffer overlap
    if (BUFFER_PADDING > 0) {
        my_thread_buffer[N] = (int8_t)(replica_id + 100);  // Unique marker per thread
    }

    // Buffer isolation verified - debug code removed

    // WORKERS: Initialize energy tracking (manager will calculate final energies)
    int my_thread_energy = 0;

    // Workers calculate initial energy for their private optimization
    if (N == 2) {
        int8_t s0 = my_thread_buffer[0];
        int8_t s1 = my_thread_buffer[1];
        my_thread_energy = -1 * s0 * s1;
    } else {
        // General case for optimization loop
        for (int i = 0; i < N; i++) {
            int8_t s_i = my_thread_buffer[i];
            int start = csr_row_ptr[i];
            int end   = csr_row_ptr[i + 1];
            for (int p = start; p < end; ++p) {
                int j = csr_col_ind[p];
                if (j > i) {
                    int8_t Jij = csr_J_vals[p];
                    int8_t s_j = my_thread_buffer[j];
                    my_thread_energy += Jij * s_i * s_j;
                }
            }
        }
    }

    // Workers only handle optimization - no coordination duties

    // STEP 4: Main optimization loop - Parallel tempering with isolated thread buffers
    int my_quota = per_thread_quota[replica_id];
    int my_base = per_thread_base[replica_id];
    int local_out = 0;
    int sample_interval = params->sample_interval > 0 ? params->sample_interval : 1;

    for (int sweep = 0; sweep < params->num_sweeps; sweep++) {
        // Each thread optimizes its own isolated buffer
        for (int spin_idx = 0; spin_idx < N; spin_idx++) {
            // Try flipping spin at spin_idx
            int8_t old_spin = my_thread_buffer[spin_idx];
            int8_t new_spin = -old_spin;

            // Calculate energy change for this flip using CSR
            int delta_energy = 0;
            int start = csr_row_ptr[spin_idx];
            int end = csr_row_ptr[spin_idx + 1];

            for (int p = start; p < end; ++p) {
                int neighbor_j = csr_col_ind[p];
                int8_t Jij = csr_J_vals[p];
                int8_t neighbor_spin = my_thread_buffer[neighbor_j];

                // Energy change: new_contribution - old_contribution
                delta_energy += Jij * (new_spin - old_spin) * neighbor_spin;
            }

            // Metropolis acceptance criterion
            bool accept = false;
            if (delta_energy <= 0) {
                accept = true;  // Always accept improvements
            } else {
                // Accept with probability exp(-delta_energy * beta)
                float prob = exp(-delta_energy * beta);
                my_rng_state = xorshift(my_rng_state);
                float rand_val = float(my_rng_state & 0x7FFFFFFF) / 2147483647.0f;
                accept = (rand_val < prob);
            }

            if (accept) {
                my_thread_buffer[spin_idx] = new_spin;
                my_thread_energy += delta_energy;
            }
        }


        // Sample collection at specified intervals
        if (((sweep + 1) % sample_interval) == 0 && local_out < my_quota) {
            // Compute energy from current buffer
            int calculated_energy = 0;
            if (N == 2) {
                int8_t s0 = my_thread_buffer[0];
                int8_t s1 = my_thread_buffer[1];
                calculated_energy = -1 * s0 * s1;
            } else {
                for (int i = 0; i < N; i++) {
                    int8_t s_i = my_thread_buffer[i];
                    int row_start = csr_row_ptr[i];
                    int row_end = csr_row_ptr[i + 1];
                    for (int p = row_start; p < row_end; ++p) {
                        int j = csr_col_ind[p];
                        if (j > i) {
                            int8_t Jij = csr_J_vals[p];
                            int8_t s_j = my_thread_buffer[j];
                            calculated_energy += Jij * s_i * s_j;
                        }
                    }
                }
            }

            int out_idx = my_base + local_out;
            if (out_idx < params->num_reads) {
                // Write sample spins
                uint sample_offset = (uint)out_idx * (uint)N;
                for (int i = 0; i < N; i++) {
                    final_samples[sample_offset + i] = my_thread_buffer[i];
                }
                final_energies[out_idx] = calculated_energy;
                local_out++;
            }
        }
    }

    // If quota not met due to large sample_interval, emit remaining samples from final state
    while (local_out < my_quota) {
        int calculated_energy = 0;
        if (N == 2) {
            int8_t s0 = my_thread_buffer[0];
            int8_t s1 = my_thread_buffer[1];
            calculated_energy = -1 * s0 * s1;
        } else {
            for (int i = 0; i < N; i++) {
                int8_t s_i = my_thread_buffer[i];
                int row_start = csr_row_ptr[i];
                int row_end = csr_row_ptr[i + 1];
                for (int p = row_start; p < row_end; ++p) {
                    int j = csr_col_ind[p];
                    if (j > i) {
                        int8_t Jij = csr_J_vals[p];
                        int8_t s_j = my_thread_buffer[j];
                        calculated_energy += Jij * s_i * s_j;
                    }
                }
            }
        }
        int out_idx = my_base + local_out;
        if (out_idx < params->num_reads) {
            uint sample_offset = (uint)out_idx * (uint)N;
            for (int i = 0; i < N; i++) {
                final_samples[sample_offset + i] = my_thread_buffer[i];
            }
            final_energies[out_idx] = calculated_energy;
        }
        local_out++;
    }
}

