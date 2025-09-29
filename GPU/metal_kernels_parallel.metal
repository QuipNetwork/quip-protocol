#include <metal_stdlib>
using namespace metal;

// Unified Metal Kernels - 1:1 mapping with Python functions
// Each kernel corresponds to a specific Python preprocessing function

typedef unsigned int uint;

// Input structures
struct ProblemData {
    int num_nodes;
    int num_edges;
    int max_node_index;
};

// 1. Build CSR from h,J dictionaries - matches _convert_problem_to_buffers()
kernel void build_csr_from_hJ(
    device const ProblemData* problem [[buffer(0)]],
    device const int* J_edges [[buffer(1)]],        // [num_edges * 2] as (i,j) pairs
    device const float* J_values [[buffer(2)]],     // [num_edges] coupling values
    device int* csr_row_ptr [[buffer(3)]],          // [N+1] output row pointers
    device int* csr_col_ind [[buffer(4)]],          // [nnz] output column indices
    device int8_t* csr_J_vals [[buffer(5)]],        // [nnz] output coupling values
    device int* nnz_out [[buffer(6)]],              // [1] output total non-zeros
    device int* row_ptr_working [[buffer(7)]],      // [N] working copy for fill
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id != 0) return;  // Single thread does all work

    int N = problem->max_node_index + 1;

    // Initialize row_ptr to zero
    for (int i = 0; i <= N; i++) {
        csr_row_ptr[i] = 0;
    }

    // Count edges per node (undirected: both i->j and j->i)
    for (int e = 0; e < problem->num_edges; e++) {
        int i = J_edges[e * 2];
        int j = J_edges[e * 2 + 1];
        if (i < N && j < N) {
            csr_row_ptr[i + 1]++;
            csr_row_ptr[j + 1]++;
        }
    }

    // Convert counts to cumulative (prefix sum)
    for (int i = 1; i <= N; i++) {
        csr_row_ptr[i] += csr_row_ptr[i - 1];
    }

    int nnz = csr_row_ptr[N];
    *nnz_out = nnz;

    // Build adjacency lists - need working copy of row pointers
    for (int i = 0; i < N; i++) {
        row_ptr_working[i] = csr_row_ptr[i];
    }

    // Fill edges
    for (int e = 0; e < problem->num_edges; e++) {
        int i = J_edges[e * 2];
        int j = J_edges[e * 2 + 1];
        float coupling = J_values[e];
        int8_t coupling_sign = (coupling > 0) ? 1 : ((coupling < 0) ? -1 : 0);

        if (i < N && j < N) {
            // Add i->j
            int pos_i = row_ptr_working[i]++;
            csr_col_ind[pos_i] = j;
            csr_J_vals[pos_i] = coupling_sign;

            // Add j->i
            int pos_j = row_ptr_working[j]++;
            csr_col_ind[pos_j] = i;
            csr_J_vals[pos_j] = coupling_sign;
        }
    }
}

// 2. Calculate optimal replica count - matches _calculate_optimal_replicas()
kernel void calculate_optimal_replicas(
    device const int* problem_size [[buffer(0)]],     // [1] N
    device const int* num_couplings [[buffer(1)]],    // [1] number of J edges
    device const float* coupling_density [[buffer(2)]], // [1] computed density
    device const int* requested_replicas [[buffer(3)]], // [1] user request (0=auto)
    device int* optimal_replicas [[buffer(4)]],       // [1] output
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id != 0) return;

    if (*requested_replicas > 0) {
        *optimal_replicas = min(*requested_replicas, 256);
        return;
    }

    int N = *problem_size;
    int num_edges = *num_couplings;
    float density = *coupling_density;

    // Base replica count from problem size (logarithmic scaling)
    int base_replicas = max(8, int(log2(float(N))) * 2);

    // Adjust based on coupling density
    float density_factor = 1.0f;
    if (density > 0.1f) {
        density_factor = 1.5f;  // Dense problems need more replicas
    } else if (density < 0.01f) {
        density_factor = 0.8f;  // Sparse problems can use fewer
    }

    // Adjust based on energy scale
    float energy_factor = 1.0f;
    if (num_edges > 1000) {
        energy_factor = 1.3f;   // Large energy scale
    } else if (num_edges < 100) {
        energy_factor = 0.9f;   // Small energy scale
    }

    int result = int(float(base_replicas) * density_factor * energy_factor);
    *optimal_replicas = min(max(result, 8), 256);
}

// 3. Create temperature ladder - matches _create_adaptive_temperature_ladder()
kernel void create_temperature_ladder(
    device const float* T_min [[buffer(0)]],          // [1] minimum temperature
    device const float* T_max [[buffer(1)]],          // [1] maximum temperature
    device const int* num_replicas [[buffer(2)]],     // [1] number of replicas
    device float* temperatures [[buffer(3)]],         // [num_replicas] output ladder
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id != 0) return;

    float t_min = *T_min;
    float t_max = *T_max;
    int n_reps = *num_replicas;

    if (n_reps == 1) {
        temperatures[0] = t_min;
        return;
    }

    // Geometric spacing with adaptive adjustment
    float base_ratio = pow(t_max / t_min, 1.0f / float(n_reps - 1));

    // Adaptive ratio adjustment for target acceptance
    float adjustment_factor = 1.0f;
    if (n_reps <= 4) {
        adjustment_factor = 1.15f;      // Tighter spacing for few replicas
    } else if (n_reps <= 8) {
        adjustment_factor = 1.05f;      // Balanced spacing
    } else if (n_reps <= 16) {
        adjustment_factor = 0.98f;      // Slightly tighter
    } else {
        adjustment_factor = 0.92f;      // Much tighter for many replicas
    }

    float optimal_ratio = base_ratio * adjustment_factor;

    // Generate geometric ladder
    for (int i = 0; i < n_reps; i++) {
        temperatures[i] = t_min * pow(optimal_ratio, float(i));
    }

    // Ensure we hit T_max exactly
    temperatures[n_reps - 1] = t_max;
}

// 4. Partition reads across replicas - matches per_thread_base/quota calculation
kernel void calculate_read_partitioning(
    device const int* num_reads [[buffer(0)]],        // [1] total reads wanted
    device const int* num_replicas [[buffer(1)]],     // [1] number of replicas
    device int* per_replica_base [[buffer(2)]],       // [num_replicas] base index
    device int* per_replica_quota [[buffer(3)]],      // [num_replicas] read count
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id != 0) return;

    int total_reads = *num_reads;
    int n_reps = *num_replicas;

    // Calculate base quota and remainder
    int base_quota = total_reads / n_reps;
    int remainder = total_reads % n_reps;

    // Assign quotas
    for (int i = 0; i < n_reps; i++) {
        per_replica_quota[i] = base_quota + (i < remainder ? 1 : 0);
    }

    // Calculate base indices (prefix sum)
    per_replica_base[0] = 0;
    for (int i = 1; i < n_reps; i++) {
        per_replica_base[i] = per_replica_base[i-1] + per_replica_quota[i-1];
    }
}

// Device helper functions for preprocess
inline int clamp_int(int v, int lo, int hi) { return max(lo, min(hi, v)); }

inline int compute_optimal_replicas_device(int N, int num_edges, float density, int requested) {
    if (requested > 0) return clamp_int(requested, 1, 256);
    int base = max(8, int(log2(float(N))) * 2);
    float density_factor = (density > 0.1f) ? 1.5f : ((density < 0.01f) ? 0.8f : 1.0f);
    float energy_factor = (num_edges > 1000) ? 1.3f : ((num_edges < 100) ? 0.9f : 1.0f);
    int res = int(float(base) * density_factor * energy_factor);
    return clamp_int(res, 1, 256);
}

inline void create_temperature_ladder_device(float T_min, float T_max, int n_reps, device float* temps) {
    if (n_reps <= 1) { temps[0] = T_min; return; }
    float ratio = pow(T_max / T_min, 1.0f / float(n_reps - 1));
    if (n_reps <= 4) ratio *= 1.15f;
    else if (n_reps <= 8) ratio *= 1.05f;
    else if (n_reps <= 16) ratio *= 0.98f;
    else ratio *= 0.92f;
    for (int i = 0; i < n_reps; i++) temps[i] = T_min * pow(ratio, float(i));
    temps[n_reps - 1] = T_max;
}

inline void partition_reads_device(int total_reads, int n_reps, device int* base_idx, device int* quota) {
    int base_q = total_reads / n_reps;
    int rem = total_reads % n_reps;
    int acc = 0;
    for (int i = 0; i < n_reps; i++) {
        quota[i] = base_q + (i < rem ? 1 : 0);
        base_idx[i] = acc;
        acc += quota[i];
    }
}

inline uint mix_seed(uint x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

inline void seed_and_init_spins_device(uint base_seed, int n_spins, int n_reps,
                                       device uint* rng_states, device int8_t* thread_buffers) {
    for (int r = 0; r < n_reps; r++) {
        uint s = base_seed ^ uint(r * 12345u);
        rng_states[r] = s;
        device int8_t* spins = &thread_buffers[r * n_spins];
        for (int i = 0; i < n_spins; i++) { s = mix_seed(s); spins[i] = (s & 1) ? 1 : -1; }
    }
}

// 5a. Unified preprocess kernel (single-dispatch)
kernel void preprocess_all(
    // Problem
    device const ProblemData* problem [[buffer(0)]],
    device const int* J_edges [[buffer(1)]],        // [E*2]
    device const float* J_values [[buffer(2)]],     // [E]
    // CSR outputs
    device int* csr_row_ptr [[buffer(3)]],          // [N+1]
    device int* csr_col_ind [[buffer(4)]],          // [<=2E]
    device int8_t* csr_J_vals [[buffer(5)]],        // [<=2E]
    device int* nnz_out [[buffer(6)]],              // [1]
    device int* row_ptr_working [[buffer(7)]],      // [N]
    // Scheduling inputs
    device const int* requested_replicas [[buffer(8)]], // [1]
    device const int* total_reads [[buffer(9)]],         // [1]
    device const float* T_min [[buffer(10)]],            // [1]
    device const float* T_max [[buffer(11)]],            // [1]
    device const uint* base_seed [[buffer(12)]],         // [1]
    // Scheduling outputs
    device int* num_replicas [[buffer(13)]],         // [1]
    device float* temperatures [[buffer(14)]],       // [cap]
    device int* per_replica_base [[buffer(15)]],     // [cap]
    device int* per_replica_quota [[buffer(16)]],    // [cap]
    // Initialization outputs
    device uint* rng_states [[buffer(17)]],          // [cap]
    device int8_t* thread_buffers [[buffer(18)]],    // [cap * N]
    // Additional inputs/outputs
    device const int* num_sweeps_in [[buffer(19)]],  // [1] may be <=0 to keep provided sample_interval
    device int* sample_interval_out [[buffer(20)]],  // [1] if <=0 on entry, computed here
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0u) return;  // single-thread orchestrator

    int N = problem->max_node_index + 1;
    int E = problem->num_edges;

    // 1) Build CSR
    for (int i = 0; i <= N; i++) csr_row_ptr[i] = 0;
    for (int e = 0; e < E; e++) {
        int i = J_edges[e * 2]; int j = J_edges[e * 2 + 1];
        if (i < N && j < N) { csr_row_ptr[i + 1]++; csr_row_ptr[j + 1]++; }
    }
    for (int i = 1; i <= N; i++) csr_row_ptr[i] += csr_row_ptr[i - 1];
    int nnz = csr_row_ptr[N]; *nnz_out = nnz;
    for (int i = 0; i < N; i++) row_ptr_working[i] = csr_row_ptr[i];
    for (int e = 0; e < E; e++) {
        int i = J_edges[e * 2]; int j = J_edges[e * 2 + 1]; float coupling = J_values[e];
        int8_t s = (coupling > 0) ? 1 : ((coupling < 0) ? -1 : 0);
        if (i < N && j < N) { int pi = row_ptr_working[i]++; csr_col_ind[pi] = j; csr_J_vals[pi] = s;
                               int pj = row_ptr_working[j]++; csr_col_ind[pj] = i; csr_J_vals[pj] = s; }
    }

    // 2) Optimal replicas
    float density = (N > 1) ? (2.0f * float(E)) / (float(N) * float(N - 1)) : 0.0f;
    int req = *requested_replicas;
    int n_reps = compute_optimal_replicas_device(N, E, density, req);
    *num_replicas = n_reps;

    // 3) Temperatures
    create_temperature_ladder_device(*T_min, *T_max, n_reps, temperatures);

    // 4) Read partitioning
    partition_reads_device(*total_reads, n_reps, per_replica_base, per_replica_quota);

    // 4b) Sample interval (only compute if not preset by host)
    int existing = *sample_interval_out;
    if (existing <= 0) {
        int max_quota = 1;
        for (int i = 0; i < n_reps; i++) max_quota = max(max_quota, per_replica_quota[i]);
        int sweeps = (*num_sweeps_in > 0) ? *num_sweeps_in : 1000;
        int s_int = max(1, sweeps / max_quota);
        *sample_interval_out = s_int;
    }

    // 5) RNG + spin initialization
    seed_and_init_spins_device(*base_seed, N, n_reps, rng_states, thread_buffers);
}

// 5. Main sampling kernel - uses pre-initialized buffers
kernel void parallel_tempering_sampling(
    // CSR data (pre-built)
    device const int* csr_row_ptr [[buffer(0)]],      // [N+1]
    device const int* csr_col_ind [[buffer(1)]],      // [nnz]
    device const int8_t* csr_J_vals [[buffer(2)]],    // [nnz]
    device const int* N [[buffer(3)]],                // [1] number of spins

    // Temperature ladder (pre-built)
    device const float* temperatures [[buffer(4)]],   // [num_replicas]
    device const int* num_replicas [[buffer(5)]],     // [1]

    // Sampling parameters
    device const int* num_sweeps [[buffer(6)]],       // [1]
    device const int* sample_interval [[buffer(7)]],  // [1]
    device const uint* base_seed [[buffer(8)]],       // [1]

    // Per-replica read partitioning
    device const int* per_replica_base [[buffer(9)]],  // [num_replicas]
    device const int* per_replica_quota [[buffer(10)]], // [num_replicas]

    // Output
    device int8_t* final_samples [[buffer(11)]],      // [total_reads * N]
    device int* final_energies [[buffer(12)]],        // [total_reads]

    // Per-replica persistent buffers and RNG state (device memory)
    device int8_t* thread_buffers [[buffer(13)]],     // [num_replicas * N]
    device uint* rng_states [[buffer(14)]],           // [num_replicas]

    uint replica_id [[thread_position_in_grid]]
) {
    int n_reps = *num_replicas;
    if (replica_id >= uint(n_reps)) return;

    int n_spins = *N;
    float temperature = temperatures[replica_id];
    int n_sweeps = *num_sweeps;
    int samp_interval = *sample_interval;

    int my_base = per_replica_base[replica_id];
    int my_quota = per_replica_quota[replica_id];

    // Each replica's private spin buffer lives in device memory
    device int8_t* spins = &thread_buffers[replica_id * n_spins];

    // Use pre-initialized RNG state and spins from preprocess_all
    uint rng_state = rng_states[replica_id];

    // Calculate initial energy
    int energy = 0;
    if (n_spins == 2) {
        energy = -1 * spins[0] * spins[1];
    } else {
        for (int i = 0; i < n_spins; i++) {
            int8_t s_i = spins[i];
            int start = csr_row_ptr[i];
            int end = csr_row_ptr[i + 1];
            for (int p = start; p < end; p++) {
                int j = csr_col_ind[p];
                if (j > i) {
                    int8_t Jij = csr_J_vals[p];
                    int8_t s_j = spins[j];
                    energy += Jij * s_i * s_j;
                }
            }
        }
    }

    // Persist updated RNG state
    rng_states[replica_id] = rng_state;

    // Sampling loop
    float beta = 1.0f / temperature;
    int samples_written = 0;

    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        // Metropolis updates
        for (int spin_idx = 0; spin_idx < n_spins; spin_idx++) {
            int8_t old_spin = spins[spin_idx];
            int8_t new_spin = -old_spin;

            // Calculate energy change
            int delta_energy = 0;
            int start = csr_row_ptr[spin_idx];
            int end = csr_row_ptr[spin_idx + 1];
            for (int p = start; p < end; p++) {
                int j = csr_col_ind[p];
                int8_t Jij = csr_J_vals[p];
                int8_t neighbor_spin = spins[j];
                delta_energy += Jij * (new_spin - old_spin) * neighbor_spin;
            }

            // Metropolis acceptance
            bool accept = (delta_energy <= 0);
            if (!accept) {
                float prob = exp(-float(delta_energy) * beta);
                rng_state = rng_state ^ (rng_state << 13);
                rng_state = rng_state ^ (rng_state >> 17);
                rng_state = rng_state ^ (rng_state << 5);
                float rand_val = float(rng_state & 0x7FFFFFFF) / 2147483647.0f;
                accept = (rand_val < prob);
            }

            if (accept) {
                spins[spin_idx] = new_spin;
                energy += delta_energy;
            }
        }

        // Sample collection
        if (((sweep + 1) % samp_interval) == 0 && samples_written < my_quota) {
            int out_idx = my_base + samples_written;

            // Write sample
            for (int i = 0; i < n_spins; i++) {
                final_samples[out_idx * n_spins + i] = spins[i];
            }
            final_energies[out_idx] = energy;
            samples_written++;
        }
    }

    // Fill remaining quota if needed
    while (samples_written < my_quota) {
        int out_idx = my_base + samples_written;
        for (int i = 0; i < n_spins; i++) {
            final_samples[out_idx * n_spins + i] = spins[i];
        }
        final_energies[out_idx] = energy;
        samples_written++;
    }
}