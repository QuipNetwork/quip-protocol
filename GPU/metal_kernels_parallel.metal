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


// 6. 2D skeleton kernel: initialize double buffers with deterministic RNG per (replica, spin)
kernel void worker_parallel_updates(
    // Sizes
    device const int* N [[buffer(0)]],                 // [1]
    device const int* num_replicas [[buffer(1)]],      // [1]
    device const uint* base_seed [[buffer(2)]],        // [1]
    // Double buffers
    device int8_t* thread_buffers_src [[buffer(3)]],   // [num_replicas * N]
    device int8_t* thread_buffers_dst [[buffer(4)]],   // [num_replicas * N]

    uint3 tid3 [[thread_position_in_grid]]
) {
    int n = *N;
    int rcount = *num_replicas;

    int sid = int(tid3.x);
    int rid = int(tid3.y);
    if (rid >= rcount || sid >= n) return;

    // Stateless RNG from (rid, sid, base_seed)
    uint s = *base_seed ^ uint(rid * 2654435761u) ^ uint((sid + 1) * 974238197u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    int8_t spin = (s & 1u) ? int8_t(1) : int8_t(-1);

    int idx = rid * n + sid;
    thread_buffers_src[idx] = spin;
    thread_buffers_dst[idx] = spin; // same initial content; ready for double-buffering
}


// 7. 2D synchronous Metropolis updates with double-buffering (Step 2)
kernel void metropolis_2d_synchronous(
    // CSR data
    device const int* csr_row_ptr [[buffer(0)]],      // [N+1]
    device const int* csr_col_ind [[buffer(1)]],      // [nnz]
    device const int8_t* csr_J_vals [[buffer(2)]],    // [nnz]
    device const int* N [[buffer(3)]],                // [1]

    // Temperatures and replicas
    device const float* temperatures [[buffer(4)]],   // [num_replicas]
    device const int* num_replicas [[buffer(5)]],     // [1]

    // Parameters
    device const int* num_sweeps [[buffer(6)]],       // [1]
    device const uint* base_seed [[buffer(7)]],       // [1]

    // Double buffers
    device int8_t* thread_buffers_src [[buffer(8)]],  // [num_replicas * N]
    device int8_t* thread_buffers_dst [[buffer(9)]],  // [num_replicas * N]
    // Debug: accepted flip counts per replica
    device atomic_int* flip_counts [[buffer(10)]],     // [num_replicas]
    device atomic_int* pos_delta_counts [[buffer(11)]], // [num_replicas]
    device atomic_int* neg_delta_counts [[buffer(12)]], // [num_replicas]
    device int* energy_debug [[buffer(13)]],             // [num_replicas]

    // Thread indices
    uint3 tid3_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    int n = *N;
    int rid = int(tg_pos.y);                // one threadgroup per replica
    int n_reps = *num_replicas;
    if (rid >= n_reps) return;

    float temperature = temperatures[rid];
    float beta = 1.0f / max(temperature, 1e-6f);

    // Local pointers to current source/dest buffers for this replica
    device int8_t* src = &thread_buffers_src[rid * n];
    device int8_t* dst = &thread_buffers_dst[rid * n];

    // Sweep loop
    int sweeps = *num_sweeps;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        // Randomize starting parity per sweep to break patterns
        threadgroup int start_parity;
        if (tid3_tg.x == 0) {
            uint s = *base_seed ^ uint((rid + 1) * 2654435761u) ^ uint((sweep + 1) * 362437u);
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            start_parity = int(s & 1u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Two-phase Gauss-Seidel style update to avoid bipartite oscillations
        for (int phase = 0; phase < 2; ++phase) {
            int parity = start_parity ^ phase;
            for (int tile = 0; tile < n; tile += int(tg_size.x)) {
                int sid = tile + int(tid3_tg.x);
                if (sid < n) {
                    int8_t old_spin = src[sid];
                    int8_t new_spin = -old_spin;
                    if ((sid & 1) == parity) {
                        // Compute energy change using neighbors from current source
                        int delta_energy = 0;
                        int start = csr_row_ptr[sid];
                        int end = csr_row_ptr[sid + 1];
                        for (int p = start; p < end; ++p) {
                            int j = csr_col_ind[p];
                            int8_t Jij = csr_J_vals[p];
                            int8_t neighbor_spin = src[j];
                            delta_energy += Jij * (new_spin - old_spin) * neighbor_spin;
                        }
                        // Debug counts for delta sign
                        if (delta_energy > 0) {
                            atomic_fetch_add_explicit(&pos_delta_counts[rid], 1, memory_order_relaxed);
                        } else if (delta_energy < 0) {
                            atomic_fetch_add_explicit(&neg_delta_counts[rid], 1, memory_order_relaxed);
                        }
                        // Metropolis accept
                        bool accept = (delta_energy <= 0);
                        if (!accept) {
                            uint s2 = *base_seed ^ uint((rid + 1) * 2654435761u) ^ uint((sid + 1) * 974238197u) ^ uint((sweep + 1) * 362437u);
                            s2 ^= s2 << 13; s2 ^= s2 >> 17; s2 ^= s2 << 5;
                            float r = float(s2 & 0x7FFFFFFFu) / 2147483647.0f;
                            float prob = exp(-float(delta_energy) * beta);
                            accept = (r < prob);
                        }
                        if (accept) {
                            dst[sid] = new_spin;
                            atomic_fetch_add_explicit(&flip_counts[rid], 1, memory_order_relaxed);
                        } else {
                            dst[sid] = old_spin;
                        }
                    } else {
                        // Carry forward unchanged spin for the opposite parity
                        dst[sid] = old_spin;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
            // Make updates visible to next parity and use them as source
            device int8_t* tmp = src; src = dst; dst = tmp;
            threadgroup_barrier(mem_flags::mem_device);
        }
        // After two parity phases, 'src' already points to the latest configuration
    }

    // Ensure the latest configuration resides in thread_buffers_src
    if (src != &thread_buffers_src[rid * n]) {
        for (int tile = 0; tile < n; tile += int(tg_size.x)) {
            int sid = tile + int(tid3_tg.x);
            if (sid < n) {
                thread_buffers_src[rid * n + sid] = src[sid];
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    // Debug: compute per-replica energy using j>i convention on src buffer
    if (tid3_tg.x == 0) {
        int e = 0;
        for (int i = 0; i < n; ++i) {
            int start = csr_row_ptr[i];
            int end = csr_row_ptr[i + 1];
            int8_t s_i = src[i];
            for (int p = start; p < end; ++p) {
                int j = csr_col_ind[p];
                if (j > i) {
                    int8_t Jij = csr_J_vals[p];
                    e += Jij * s_i * src[j];
                }
            }
        }
        energy_debug[rid] = e;
    }

}


// 8. Energy reduction per replica (parallel over replicas, tiled over spins)
kernel void reduce_energies_2d(
    device const int* csr_row_ptr [[buffer(0)]],      // [N+1]
    device const int* csr_col_ind [[buffer(1)]],      // [nnz]
    device const int8_t* csr_J_vals [[buffer(2)]],    // [nnz]
    device const int* N [[buffer(3)]],                // [1]
    device int8_t* thread_buffers_src [[buffer(4)]],  // [num_replicas * N]
    device int* energies_out [[buffer(5)]],           // [num_replicas]
    device const int* num_replicas [[buffer(6)]],     // [1]

    uint3 tid3_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    int n = *N;
    int rid = int(tg_pos.y);
    if (rid >= *num_replicas) return;

    if (tid3_tg.x != 0) return; // single lane computes energy serially for simplicity

    device int8_t* spins = &thread_buffers_src[rid * n];

    int e = 0;
    for (int i = 0; i < n; ++i) {
        int8_t s_i = spins[i];
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {
                int8_t Jij = csr_J_vals[p];
                e += Jij * s_i * spins[j];
            }
        }
    }
    energies_out[rid] = e;
}

// 9. Adjacent replica swaps (even/odd parity pairs)
kernel void pt_swap_adjacent(
    device const int* N [[buffer(0)]],                // [1]
    device const float* temperatures [[buffer(1)]],   // [num_replicas]
    device const int* num_replicas [[buffer(2)]],     // [1]
    device int8_t* thread_buffers_src [[buffer(3)]],  // [num_replicas * N]
    device int* energies [[buffer(4)]],               // [num_replicas]
    device const uint* base_seed [[buffer(5)]],       // [1]
    device const int* swap_parity [[buffer(6)]],      // [1] 0=even pairs, 1=odd pairs
    device atomic_int* swap_attempts [[buffer(7)]],   // [num_replicas-1]
    device atomic_int* swap_accepts [[buffer(8)]],    // [num_replicas-1]

    uint3 tid3_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    int n = *N;
    int parity = *swap_parity;
    int pair_idx = int(tg_pos.y);
    int rid0 = parity + 2 * pair_idx;
    int rid1 = rid0 + 1;
    int n_reps = *num_replicas;
    if (rid1 >= n_reps) return;

    // Compute acceptance in lane 0, broadcast via threadgroup var
    threadgroup int do_swap;
    do_swap = 0;
    if (tid3_tg.x == 0) {
        float beta0 = 1.0f / max(temperatures[rid0], 1e-6f);
        float beta1 = 1.0f / max(temperatures[rid1], 1e-6f);
        int E0 = energies[rid0];
        int E1 = energies[rid1];
        float x = (beta0 - beta1) * float(E1 - E0);
        float p = exp(min(0.0f, x));
        uint s = *base_seed ^ uint((pair_idx + 1) * 2654435761u) ^ uint((parity + 1) * 974238197u);
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        float r = float(s & 0x7FFFFFFFu) / 2147483647.0f;
        int idx = rid0; // index into attempts/accepts arrays aligns with lower rid
        atomic_fetch_add_explicit(&swap_attempts[idx], 1, memory_order_relaxed);
        do_swap = (r < p) ? 1 : 0;
        if (do_swap) {
            atomic_fetch_add_explicit(&swap_accepts[idx], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel swap across spins if accepted
    if (do_swap) {
        device int8_t* A = &thread_buffers_src[rid0 * n];
        device int8_t* B = &thread_buffers_src[rid1 * n];
        for (int sid = int(tid3_tg.x); sid < n; sid += int(tg_size.x)) {
            int8_t tmp = A[sid];
            A[sid] = B[sid];
            B[sid] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid3_tg.x == 0) {
            int tmpE = energies[rid0];
            energies[rid0] = energies[rid1];
            energies[rid1] = tmpE;
        }
    }
}



// 11. Pack selected replicas' spins into a compact buffer [num_reads * N]
kernel void pack_selected_replicas(
    device const int* N [[buffer(0)]],                 // [1]
    device const int* sel_indices [[buffer(1)]],       // [num_reads]
    device const int* num_reads [[buffer(2)]],         // [1]
    device const int8_t* thread_buffers_src [[buffer(3)]], // [num_replicas * N]
    device int8_t* packed_out [[buffer(4)]],          // [num_reads * N]

    uint3 tid3_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    int n = *N;
    int k = int(tg_pos.y);             // output sample index
    if (k >= *num_reads) return;
    int rid = sel_indices[k];          // replica to pack

    for (int sid = int(tid3_tg.x); sid < n; sid += int(tg_size.x)) {
        int8_t v = thread_buffers_src[rid * n + sid];
        packed_out[k * n + sid] = v;
    }
}


// 10. Compute swap acceptance rates per adjacent pair
kernel void compute_swap_stats(
    device const int* num_replicas [[buffer(0)]],
    device const atomic_int* swap_attempts [[buffer(1)]],
    device const atomic_int* swap_accepts [[buffer(2)]],
    device float* rates_out [[buffer(3)]],
    uint3 tid3_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]
) {
    if (tid3_tg.x != 0) return;
    int n_reps = *num_replicas;
    int idx = int(tg_pos.y);
    if (idx >= max(0, n_reps - 1)) return;
    int a = atomic_load_explicit(&swap_attempts[idx], memory_order_relaxed);
    int b = atomic_load_explicit(&swap_accepts[idx], memory_order_relaxed);
    rates_out[idx] = (a > 0) ? (float(b) / float(a)) : 0.0f;
}

// 12. Greedy graph coloring (distance-1) for multi-color updates
kernel void compute_graph_coloring(
    device const int* csr_row_ptr [[buffer(0)]],    // [N+1]
    device const int* csr_col_ind [[buffer(1)]],    // [nnz]
    device const int* N [[buffer(2)]],              // [1]
    device const int* max_colors [[buffer(3)]],     // [1] maximum allowed colors
    device int* node_colors [[buffer(4)]],          // [N] output: color per node
    device int* num_colors_out [[buffer(5)]],       // [1] output: number of colors used
    device int* use_coloring [[buffer(6)]],         // [1] output: 1=success, 0=fallback to parity
    device int* node_degrees [[buffer(7)]],         // [N] temp: degree per node
    uint tid [[thread_position_in_grid]]
) {
    int n = *N;
    int max_col = *max_colors;

    // Single-threaded greedy coloring for simplicity and correctness
    if (tid != 0) return;

    // Initialize: compute degrees
    for (int i = 0; i < n; i++) {
        node_colors[i] = -1;  // uncolored
        node_degrees[i] = csr_row_ptr[i + 1] - csr_row_ptr[i];
    }

    int num_colors = 0;

    // Greedy coloring in degree-descending order
    // Process nodes by finding max degree uncolored node each iteration
    for (int iter = 0; iter < n; iter++) {
        // Find uncolored node with highest degree
        int best_node = -1;
        int best_degree = -1;
        for (int i = 0; i < n; i++) {
            if (node_colors[i] < 0 && node_degrees[i] > best_degree) {
                best_node = i;
                best_degree = node_degrees[i];
            }
        }

        if (best_node < 0) break;  // all colored

        // Find smallest available color for this node
        bool neighbor_has_color[16];  // Support up to 16 colors (max_colors)
        for (int c = 0; c < max_col; c++) {
            neighbor_has_color[c] = false;
        }

        int start = csr_row_ptr[best_node];
        int end = csr_row_ptr[best_node + 1];
        for (int p = start; p < end; p++) {
            int neighbor = csr_col_ind[p];
            if (neighbor < n && node_colors[neighbor] >= 0 && node_colors[neighbor] < max_col) {
                neighbor_has_color[node_colors[neighbor]] = true;
            }
        }

        // Find first available color
        int chosen_color = -1;
        for (int c = 0; c < max_col; c++) {
            if (!neighbor_has_color[c]) {
                chosen_color = c;
                break;
            }
        }

        if (chosen_color < 0) {
            // Exceeded max_colors, fallback to parity mode
            *use_coloring = 0;
            *num_colors_out = 0;
            return;
        }

        node_colors[best_node] = chosen_color;
        num_colors = max(num_colors, chosen_color + 1);
    }

    // Verify coloring is valid
    for (int i = 0; i < n; i++) {
        int my_color = node_colors[i];
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; p++) {
            int neighbor = csr_col_ind[p];
            if (neighbor < n && node_colors[neighbor] == my_color) {
                // Invalid coloring detected
                *use_coloring = 0;
                *num_colors_out = 0;
                return;
            }
        }
    }

    // Success
    *num_colors_out = num_colors;
    *use_coloring = 1;
}

// 13. Color-phase Metropolis updates (Phase 2 - multi-color updates)
kernel void metropolis_color_phase(
    // CSR data
    device const int* csr_row_ptr [[buffer(0)]],      // [N+1]
    device const int* csr_col_ind [[buffer(1)]],      // [nnz]
    device const int8_t* csr_J_vals [[buffer(2)]],    // [nnz]
    device const int* N [[buffer(3)]],                // [1]

    // Temperatures and replicas
    device const float* temperatures [[buffer(4)]],   // [num_replicas]
    device const int* num_replicas [[buffer(5)]],     // [1]

    // Coloring data
    device const int* node_colors [[buffer(6)]],      // [N] color per node
    device const int* num_colors [[buffer(7)]],       // [1] number of colors
    device const int* color_order [[buffer(8)]],      // [num_colors] randomized color order for this sweep

    // Parameters
    device const int* num_sweeps [[buffer(9)]],       // [1]
    device const uint* base_seed [[buffer(10)]],      // [1]

    // Double buffers
    device int8_t* thread_buffers_src [[buffer(11)]], // [num_replicas * N]
    device int8_t* thread_buffers_dst [[buffer(12)]], // [num_replicas * N]

    // Debug counters
    device atomic_int* flip_counts [[buffer(13)]],         // [num_replicas]
    device atomic_int* pos_delta_counts [[buffer(14)]],    // [num_replicas]
    device atomic_int* neg_delta_counts [[buffer(15)]],    // [num_replicas]
    device int* energy_debug [[buffer(16)]],               // [num_replicas]

    // Thread indices
    uint3 tid3_tg [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    int n = *N;
    int rid = int(tg_pos.y);                // one threadgroup per replica
    int n_reps = *num_replicas;
    if (rid >= n_reps) return;

    float temperature = temperatures[rid];
    float beta = 1.0f / max(temperature, 1e-6f);

    // Local pointers to current source/dest buffers for this replica
    device int8_t* src = &thread_buffers_src[rid * n];
    device int8_t* dst = &thread_buffers_dst[rid * n];

    int sweeps = *num_sweeps;
    int n_colors = *num_colors;

    // Sweep loop
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        // Iterate through colors in randomized order (color_order is pre-shuffled per sweep)
        for (int color_idx = 0; color_idx < n_colors; ++color_idx) {
            int current_color = color_order[color_idx];

            // Update all spins with this color in parallel (they're independent!)
            for (int tile = 0; tile < n; tile += int(tg_size.x)) {
                int sid = tile + int(tid3_tg.x);
                if (sid < n && node_colors[sid] == current_color) {
                    int8_t old_spin = src[sid];
                    int8_t new_spin = -old_spin;

                    // Compute energy change using neighbors from current source
                    int delta_energy = 0;
                    int start = csr_row_ptr[sid];
                    int end = csr_row_ptr[sid + 1];
                    for (int p = start; p < end; ++p) {
                        int j = csr_col_ind[p];
                        int8_t Jij = csr_J_vals[p];
                        int8_t neighbor_spin = src[j];
                        delta_energy += Jij * (new_spin - old_spin) * neighbor_spin;
                    }

                    // Debug counts for delta sign
                    if (delta_energy > 0) {
                        atomic_fetch_add_explicit(&pos_delta_counts[rid], 1, memory_order_relaxed);
                    } else if (delta_energy < 0) {
                        atomic_fetch_add_explicit(&neg_delta_counts[rid], 1, memory_order_relaxed);
                    }

                    // Metropolis accept
                    bool accept = (delta_energy <= 0);
                    if (!accept) {
                        uint s2 = *base_seed ^ uint((rid + 1) * 2654435761u) ^ uint((sid + 1) * 974238197u)
                                             ^ uint((sweep + 1) * 362437u) ^ uint((color_idx + 1) * 987654321u);
                        s2 ^= s2 << 13; s2 ^= s2 >> 17; s2 ^= s2 << 5;
                        float r = float(s2 & 0x7FFFFFFFu) / 2147483647.0f;
                        float prob = exp(-float(delta_energy) * beta);
                        accept = (r < prob);
                    }

                    if (accept) {
                        dst[sid] = new_spin;
                        atomic_fetch_add_explicit(&flip_counts[rid], 1, memory_order_relaxed);
                    } else {
                        dst[sid] = old_spin;
                    }
                } else if (sid < n) {
                    // Not this color, carry forward unchanged
                    dst[sid] = src[sid];
                }
                threadgroup_barrier(mem_flags::mem_device);
            }

            // Swap src/dst for next color
            device int8_t* tmp = src; src = dst; dst = tmp;
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    // Ensure the latest configuration resides in thread_buffers_src
    if (src != &thread_buffers_src[rid * n]) {
        for (int tile = 0; tile < n; tile += int(tg_size.x)) {
            int sid = tile + int(tid3_tg.x);
            if (sid < n) {
                thread_buffers_src[rid * n + sid] = src[sid];
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }

    // Debug: compute per-replica energy
    if (tid3_tg.x == 0) {
        int e = 0;
        for (int i = 0; i < n; ++i) {
            int start = csr_row_ptr[i];
            int end = csr_row_ptr[i + 1];
            int8_t s_i = src[i];
            for (int p = start; p < end; ++p) {
                int j = csr_col_ind[p];
                if (j > i) {
                    int8_t Jij = csr_J_vals[p];
                    e += Jij * s_i * src[j];
                }
            }
        }
        energy_debug[rid] = e;
    }
}
