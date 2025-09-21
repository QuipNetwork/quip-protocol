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
