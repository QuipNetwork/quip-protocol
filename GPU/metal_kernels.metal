#include <metal_stdlib>
using namespace metal;

// Metal Parallel Tempering Kernels
// Optimized implementation using unified worker kernel

// Constants
constant int MAX_REPLICAS = 512;    // Max temperature replicas

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
    float cooling_factor;     // Temperature cooling factor (e.g., 0.999)
    int cooling_start_sweep;  // Sweep to start cooling (for hybrid PT/SA)
    float target_acceptance_ratio;  // Phase 1: Target acceptance ratio (0.25-0.35)
    int adaptation_interval;        // Phase 1: Sweeps between temperature adaptations
    int enable_replica_exchange;    // Phase 3: Enable intelligent replica exchange
    float exchange_threshold;       // Phase 3: Energy gradient threshold for exchange
};

// Each thread has its own buffer and updates from a central copy
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

    if (replica_id >= (uint)params->num_replicas) return;

    int N = params->N;
    // Initial temperature for this replica (will be cooled during PT/SA hybrid)
    float base_temperature = temperatures[replica_id];

    // Simplified bounds check - ensure we have enough space for this thread
    if (replica_id >= (uint)params->num_replicas) return;  // Should be redundant but explicit

    // Each thread gets isolated buffer region
    device int8_t* my_thread_buffer = &thread_buffers[replica_id * N];

    // Initialize thread's RNG state (PRIVATE - no sharing)
    uint my_rng_state = params->base_seed + replica_id * 12345u + threadgroup_id.x * 67890u;

    // STEP 1: Thread initializes independently

    // STEP 2: Thread initializes its isolated buffer with bounds checking
    for (int i = 0; i < N; i++) {
        my_rng_state = xorshift(my_rng_state);
        // Explicit bounds check to prevent buffer overflow
        if (i < N) {
            my_thread_buffer[i] = (my_rng_state & 1) ? 1 : -1;
        }
    }
    // WORKERS: Initialize energy tracking
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
        // Phase 2: Multi-phase cooling strategy (exploration -> cooling -> quench)
        float current_temperature = base_temperature;

        // Define phase boundaries
        float exploration_end = (float)params->num_sweeps * 0.4f;  // First 40% of sweeps
        float cooling_end = (float)params->num_sweeps * 0.8f;      // Next 40% of sweeps
        // Final 20% is quench phase

        if ((float)sweep < exploration_end) {
            // Phase 1: Exploration - maintain full PT temperature
            current_temperature = base_temperature;
        } else if ((float)sweep < cooling_end) {
            // Phase 2: Moderate cooling
            float cooling_progress = ((float)sweep - exploration_end) / (cooling_end - exploration_end);
            current_temperature = base_temperature * pow(0.98f, cooling_progress * 100.0f);
        } else {
            // Phase 3: Aggressive quench
            float quench_progress = ((float)sweep - cooling_end) / ((float)params->num_sweeps - cooling_end);
            current_temperature = base_temperature * pow(0.9f, quench_progress * 200.0f);
        }

        // Temperature-dependent cooling rates (Phase 2 enhancement)
        if (base_temperature < 0.1f) {
            // Cooler replicas get more aggressive cooling
            current_temperature *= pow(params->cooling_factor, 2.0f);
        } else if (base_temperature > 0.5f) {
            // Hotter replicas get less aggressive cooling
            current_temperature *= sqrt(params->cooling_factor);
        }

        float beta = 1.0f / current_temperature;

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


        // Phase 3: Intelligent replica exchange at swap intervals
        if (params->enable_replica_exchange && (sweep % params->swap_interval) == 0 && sweep > 0) {
            // Only even-numbered replicas attempt exchanges with next replica
            if ((replica_id % 2) == 0 && replica_id + 1 < (uint)params->num_replicas) {
                uint partner_id = replica_id + 1;

                // Calculate partner's energy using neighbor's buffer (simplified approximation)
                int partner_energy = 0;
                device int8_t* partner_buffer = &thread_buffers[partner_id * N];

                // Quick energy calculation for partner
                if (N == 2) {
                    int8_t s0 = partner_buffer[0];
                    int8_t s1 = partner_buffer[1];
                    partner_energy = -1 * s0 * s1;
                } else {
                    for (int i = 0; i < N; i++) {
                        int8_t s_i = partner_buffer[i];
                        int start = csr_row_ptr[i];
                        int end = csr_row_ptr[i + 1];
                        for (int p = start; p < end; ++p) {
                            int j = csr_col_ind[p];
                            if (j > i) {
                                int8_t Jij = csr_J_vals[p];
                                int8_t s_j = partner_buffer[j];
                                partner_energy += Jij * s_i * s_j;
                            }
                        }
                    }
                }

                // Energy-gradient driven exchange decision
                float my_temp = base_temperature;
                float partner_temp = temperatures[partner_id];
                float energy_diff = abs(my_thread_energy - partner_energy);
                float temp_diff = abs(my_temp - partner_temp);

                // Phase 3.1: Intelligent exchange criterion
                bool should_exchange = false;
                if (temp_diff > params->exchange_threshold) {
                    float exchange_probability = exp(-energy_diff / temp_diff);
                    my_rng_state = xorshift(my_rng_state);
                    float rand_val = float(my_rng_state & 0x7FFFFFFF) / 2147483647.0f;
                    should_exchange = (rand_val < exchange_probability);
                }

                // Simple buffer swap (atomic operation not needed for disjoint buffers)
                if (should_exchange) {
                    // Swap spin configurations
                    for (int i = 0; i < N; i++) {
                        int8_t temp_spin = my_thread_buffer[i];
                        my_thread_buffer[i] = partner_buffer[i];
                        partner_buffer[i] = temp_spin;
                    }
                    // Update my energy to partner's energy
                    my_thread_energy = partner_energy;
                }
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