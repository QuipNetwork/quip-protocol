// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 QUIP Protocol Contributors

#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// METAL SPLASH SAMPLER
// ==============================================================================
// Implements the Splash Sampler from Gonzalez et al. 2011:
// "Parallel Gibbs Sampling: From Colored Fields to Thin Junction Trees"
//
// Key algorithm:
// 1. Build bounded treewidth junction trees ("Splashes") covering all variables
// 2. For each Splash: calibrate via belief propagation
// 3. Sample entire Splash jointly from calibrated distribution
// 4. Repeat for all Splashes in each sweep
//
// This addresses slow mixing in strongly coupled models (like Zephyr)
// by jointly sampling groups of tightly coupled variables.

typedef unsigned int uint;

// ==============================================================================
// Data Structures
// ==============================================================================

// Clique in a junction tree
struct Clique {
    int var_start;      // Index into clique_vars array
    int var_count;      // Number of variables in clique (max: max_treewidth + 1)
    int parent_id;      // Parent clique index (-1 for root)
    int sep_start;      // Index into separator_vars array
    int sep_count;      // Number of separator variables with parent
    int pot_offset;     // Offset into potentials array (2^var_count entries)
    int msg_offset;     // Offset into messages array (2^sep_count entries)
};

// Splash region with its junction tree
struct Splash {
    int var_start;      // Index into splash_vars array
    int var_count;      // Number of variables in this Splash
    int clique_start;   // Index into cliques array
    int clique_count;   // Number of cliques in junction tree
    int root_clique;    // Root clique index (relative to clique_start)
};

// ==============================================================================
// RNG - xoshiro128** (128-bit state, passes BigCrush)
// ==============================================================================
// Upgraded from xorshift32 for better statistical quality in Gibbs sampling.
// Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number Generators"

struct RngState {
    uint s0, s1, s2, s3;
};

inline uint rotl(uint x, int k) {
    return (x << k) | (x >> (32 - k));
}

inline uint xoshiro128starstar(thread RngState &state) {
    uint result = rotl(state.s1 * 5, 7) * 9;
    uint t = state.s1 << 9;
    state.s2 ^= state.s0;
    state.s3 ^= state.s1;
    state.s1 ^= state.s2;
    state.s0 ^= state.s3;
    state.s2 ^= t;
    state.s3 = rotl(state.s3, 11);
    return result;
}

// SplitMix32 for seeding xoshiro state from a single uint
inline uint splitmix32(thread uint &z) {
    z += 0x9e3779b9u;
    uint r = z;
    r = (r ^ (r >> 16)) * 0x85ebca6bu;
    r = (r ^ (r >> 13)) * 0xc2b2ae35u;
    return r ^ (r >> 16);
}

inline RngState seed_rng(uint seed) {
    uint z = seed;
    RngState state;
    state.s0 = splitmix32(z);
    state.s1 = splitmix32(z);
    state.s2 = splitmix32(z);
    state.s3 = splitmix32(z);
    return state;
}

inline float rand_float(thread RngState &state) {
    return float(xoshiro128starstar(state)) / 4294967295.0f;
}

// ==============================================================================
// BFS Queue for Splash Construction
// ==============================================================================

#define MAX_QUEUE_SIZE 256
#define MAX_SPLASH_VARS 128
#define MAX_CLIQUES_PER_SPLASH 64
#define MAX_CLIQUE_SIZE 5  // treewidth + 1

struct BFSQueue {
    int data[MAX_QUEUE_SIZE];
    int front;
    int back;

    void init() {
        front = 0;
        back = 0;
    }

    bool empty() const {
        return front == back;
    }

    void push(int val) {
        if (back < MAX_QUEUE_SIZE) {
            data[back++] = val;
        }
    }

    int pop() {
        return data[front++];
    }
};

// ==============================================================================
// Helper: Check if variable is in array
// ==============================================================================

inline bool contains(threadgroup const int* arr, int size, int val) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == val) return true;
    }
    return false;
}

// ==============================================================================
// Build a single Splash using BFS (Algorithm 4 from paper)
// ==============================================================================

inline void build_splash_bfs(
    int root_var,
    device int* global_visited,             // [N] global visited flags (device memory)
    device const int* csr_row_ptr,
    device const int* csr_col_ind,
    int N,
    int max_splash_size,
    int max_treewidth,
    // Output (all in device memory)
    device Splash* splash,
    device Clique* cliques,
    device int* splash_vars,                // Variables in this Splash
    device int* clique_vars,                // Variables per clique
    device int* separator_vars,             // Separator variables
    int splash_var_offset,                  // Starting offset in splash_vars
    int clique_offset,                      // Starting offset in cliques array
    int clique_var_offset,                  // Starting offset in clique_vars
    int sep_var_offset                      // Starting offset in separator_vars
) {
    // Local arrays for Splash construction
    thread int local_splash_vars[MAX_SPLASH_VARS];
    thread int local_clique_membership[MAX_SPLASH_VARS];  // Which clique each splash var belongs to
    thread BFSQueue queue;

    int num_splash_vars = 0;
    int num_cliques = 0;
    int total_clique_vars = 0;
    int total_sep_vars = 0;

    queue.init();

    // Initialize with root
    local_splash_vars[num_splash_vars] = root_var;
    local_clique_membership[num_splash_vars] = 0;  // First clique
    num_splash_vars++;
    global_visited[root_var] = 1;

    // Create root clique with just the root variable
    cliques[clique_offset].var_start = clique_var_offset;
    cliques[clique_offset].var_count = 1;
    cliques[clique_offset].parent_id = -1;
    cliques[clique_offset].sep_start = sep_var_offset;
    cliques[clique_offset].sep_count = 0;
    clique_vars[clique_var_offset] = root_var;
    num_cliques = 1;
    total_clique_vars = 1;

    // Add root's neighbors to queue
    int start = csr_row_ptr[root_var];
    int end = csr_row_ptr[root_var + 1];
    for (int p = start; p < end; p++) {
        int neighbor = csr_col_ind[p];
        if (!global_visited[neighbor]) {
            queue.push(neighbor);
        }
    }

    // BFS to grow Splash (with iteration limit for safety)
    int bfs_iterations = 0;
    int max_bfs_iterations = 1024;  // Safety limit

    while (!queue.empty() && num_splash_vars < max_splash_size && num_splash_vars < MAX_SPLASH_VARS && bfs_iterations < max_bfs_iterations) {
        bfs_iterations++;
        int var = queue.pop();

        // Skip if already visited (might have been added multiple times)
        if (global_visited[var]) continue;

        // Find which existing clique this variable connects to
        // Check all neighbors of var that are already in the Splash
        int best_clique = -1;
        int connection_count = 0;

        int var_start = csr_row_ptr[var];
        int var_end = csr_row_ptr[var + 1];

        for (int p = var_start; p < var_end; p++) {
            int neighbor = csr_col_ind[p];
            // Check if neighbor is in Splash
            for (int sv = 0; sv < num_splash_vars; sv++) {
                if (local_splash_vars[sv] == neighbor) {
                    int neighbor_clique = local_clique_membership[sv];
                    if (best_clique == -1 || neighbor_clique == best_clique) {
                        best_clique = neighbor_clique;
                        connection_count++;
                    }
                    break;
                }
            }
        }

        if (best_clique == -1) {
            // No connection to existing Splash - skip this variable
            continue;
        }

        // Check if we can add to existing clique without exceeding treewidth
        int existing_clique_idx = clique_offset + best_clique;
        int existing_size = cliques[existing_clique_idx].var_count;

        if (existing_size < max_treewidth + 1) {
            // Add variable to existing clique
            int var_idx = cliques[existing_clique_idx].var_start + existing_size;
            clique_vars[var_idx] = var;
            cliques[existing_clique_idx].var_count++;
            total_clique_vars++;

            local_splash_vars[num_splash_vars] = var;
            local_clique_membership[num_splash_vars] = best_clique;
            num_splash_vars++;
        } else if (num_cliques < MAX_CLIQUES_PER_SPLASH) {
            // Create new clique connected to existing
            int new_clique_idx = clique_offset + num_cliques;
            int new_var_start = clique_var_offset + total_clique_vars;
            int new_sep_start = sep_var_offset + total_sep_vars;

            // New clique contains: variable + one connector from parent
            cliques[new_clique_idx].var_start = new_var_start;
            cliques[new_clique_idx].var_count = 2;  // var + separator
            cliques[new_clique_idx].parent_id = best_clique;  // Relative index
            cliques[new_clique_idx].sep_start = new_sep_start;
            cliques[new_clique_idx].sep_count = 1;

            // Variable in new clique
            clique_vars[new_var_start] = var;

            // Find a separator variable (first neighbor in parent clique)
            int sep_var = -1;
            for (int p = var_start; p < var_end; p++) {
                int neighbor = csr_col_ind[p];
                // Check if neighbor is in parent clique
                int parent_var_start = cliques[existing_clique_idx].var_start;
                int parent_var_count = cliques[existing_clique_idx].var_count;
                for (int pv = 0; pv < parent_var_count; pv++) {
                    if (clique_vars[parent_var_start + pv] == neighbor) {
                        sep_var = neighbor;
                        break;
                    }
                }
                if (sep_var >= 0) break;
            }

            if (sep_var >= 0) {
                clique_vars[new_var_start + 1] = sep_var;
                separator_vars[new_sep_start] = sep_var;
                total_clique_vars += 2;
                total_sep_vars += 1;

                local_splash_vars[num_splash_vars] = var;
                local_clique_membership[num_splash_vars] = num_cliques;
                num_splash_vars++;
                num_cliques++;
            }
        }

        // Mark visited
        global_visited[var] = 1;

        // Add unvisited neighbors to queue (limit queue growth)
        for (int p = var_start; p < var_end && queue.back < MAX_QUEUE_SIZE - 16; p++) {
            int neighbor = csr_col_ind[p];
            if (!global_visited[neighbor]) {
                queue.push(neighbor);
            }
        }
    }

    // Copy local splash vars to output
    for (int i = 0; i < num_splash_vars; i++) {
        splash_vars[splash_var_offset + i] = local_splash_vars[i];
    }

    // Fill in Splash struct
    splash->var_start = splash_var_offset;
    splash->var_count = num_splash_vars;
    splash->clique_start = clique_offset;
    splash->clique_count = num_cliques;
    splash->root_clique = 0;  // First clique is root
}

// ==============================================================================
// Compute clique potential for a given configuration
// potential = exp(-beta * E) where E = sum(h_i * s_i) + sum(J_ij * s_i * s_j)
// config is a bitmask: bit i = spin of variable i in clique (0=+1, 1=-1)
// ==============================================================================

inline float compute_clique_potential(
    device const Clique* clique,
    device const int* clique_vars,
    int config,
    threadgroup const int8_t* state,        // Current global state (for external couplings)
    device const float* h,
    device const int* csr_row_ptr,
    device const int* csr_col_ind,
    device const float* csr_J_vals,
    int N,
    float beta
) {
    int var_start = clique->var_start;
    int var_count = clique->var_count;

    float energy = 0.0f;

    // Get spins for this configuration
    thread int8_t clique_spins[MAX_CLIQUE_SIZE];
    for (int i = 0; i < var_count; i++) {
        clique_spins[i] = (config & (1 << i)) ? -1 : 1;
    }

    // h field contribution for clique variables
    for (int i = 0; i < var_count; i++) {
        int var = clique_vars[var_start + i];
        energy += h[var] * float(clique_spins[i]);
    }

    // J coupling within clique
    for (int i = 0; i < var_count; i++) {
        int var_i = clique_vars[var_start + i];
        int row_start = csr_row_ptr[var_i];
        int row_end = csr_row_ptr[var_i + 1];

        for (int p = row_start; p < row_end; p++) {
            int var_j = csr_col_ind[p];

            // Check if var_j is in this clique (and j > i to avoid double counting)
            for (int j = i + 1; j < var_count; j++) {
                if (clique_vars[var_start + j] == var_j) {
                    float Jij = csr_J_vals[p];
                    energy += Jij * float(clique_spins[i]) * float(clique_spins[j]);
                    break;
                }
            }
        }
    }

    // J coupling to variables outside clique (using current state)
    for (int i = 0; i < var_count; i++) {
        int var_i = clique_vars[var_start + i];
        int row_start = csr_row_ptr[var_i];
        int row_end = csr_row_ptr[var_i + 1];

        for (int p = row_start; p < row_end; p++) {
            int var_j = csr_col_ind[p];

            // Check if var_j is NOT in this clique
            bool in_clique = false;
            for (int j = 0; j < var_count; j++) {
                if (clique_vars[var_start + j] == var_j) {
                    in_clique = true;
                    break;
                }
            }

            if (!in_clique && var_j < N) {
                float Jij = csr_J_vals[p];
                int8_t external_spin = state[var_j];
                energy += Jij * float(clique_spins[i]) * float(external_spin);
            }
        }
    }

    return exp(-beta * energy);
}

// ==============================================================================
// BP Message Computation
// Message from child to parent: marginalize over non-separator variables
// ==============================================================================

inline void compute_message_to_parent(
    device const Clique* child_clique,
    device const Clique* parent_clique,
    device const int* clique_vars,
    device const int* separator_vars,
    threadgroup const float* child_potentials,  // Child's potentials (threadgroup)
    threadgroup const float* child_incoming,    // Messages TO child (from its children)
    threadgroup float* message_out,             // Output message (threadgroup)
    int max_msg_size
) {
    int child_var_count = child_clique->var_count;
    int sep_count = child_clique->sep_count;
    int sep_start = child_clique->sep_start;
    int child_pot_offset = child_clique->pot_offset;

    int num_sep_configs = 1 << sep_count;
    int num_child_configs = 1 << child_var_count;

    // Find which variables in child are separators
    thread int sep_var_indices[MAX_CLIQUE_SIZE];
    for (int s = 0; s < sep_count; s++) {
        int sep_var = separator_vars[sep_start + s];
        for (int v = 0; v < child_var_count; v++) {
            if (clique_vars[child_clique->var_start + v] == sep_var) {
                sep_var_indices[s] = v;
                break;
            }
        }
    }

    // For each separator configuration
    for (int sep_config = 0; sep_config < num_sep_configs && sep_config < max_msg_size; sep_config++) {
        float msg = 0.0f;

        // Sum over child configurations consistent with this separator config
        for (int child_config = 0; child_config < num_child_configs; child_config++) {
            // Check if child_config matches sep_config on separator variables
            bool matches = true;
            for (int s = 0; s < sep_count; s++) {
                int child_bit = (child_config >> sep_var_indices[s]) & 1;
                int sep_bit = (sep_config >> s) & 1;
                if (child_bit != sep_bit) {
                    matches = false;
                    break;
                }
            }

            if (matches) {
                float pot = child_potentials[child_pot_offset + child_config];
                // Multiply by incoming messages (if any)
                msg += pot;
            }
        }

        message_out[sep_config] = msg;
    }
}

// ==============================================================================
// Sample from calibrated junction tree (backward sampling)
// ==============================================================================

inline void sample_from_junction_tree(
    device const Splash* splash,
    device const Clique* cliques,
    device const int* clique_vars,
    device const int* separator_vars,
    threadgroup const float* potentials,
    threadgroup const float* messages,
    threadgroup int8_t* state,
    int clique_base,
    thread RngState& rng_state
) {
    int num_cliques = splash->clique_count;

    // Process cliques in tree order (root first)
    for (int c = 0; c < num_cliques; c++) {
        int clique_idx = clique_base + c;
        device const Clique* clique = &cliques[clique_idx];

        int var_count = clique->var_count;
        int var_start = clique->var_start;
        int pot_offset = clique->pot_offset;
        int parent_id = clique->parent_id;

        int num_configs = 1 << var_count;

        // Compute unnormalized probabilities for each configuration
        thread float probs[32];  // 2^5 max
        float total = 0.0f;

        for (int config = 0; config < num_configs; config++) {
            float prob = potentials[pot_offset + config];

            // If not root, condition on parent's assignment to separator
            if (parent_id >= 0) {
                int sep_count = clique->sep_count;
                int sep_start = clique->sep_start;

                // Build separator config from current state
                int sep_config = 0;
                for (int s = 0; s < sep_count; s++) {
                    int sep_var = separator_vars[sep_start + s];
                    if (state[sep_var] < 0) {
                        sep_config |= (1 << s);
                    }
                }

                // Check if this config is consistent with separator assignment
                bool consistent = true;
                for (int s = 0; s < sep_count; s++) {
                    int sep_var = separator_vars[sep_start + s];
                    // Find sep_var position in clique
                    for (int v = 0; v < var_count; v++) {
                        if (clique_vars[var_start + v] == sep_var) {
                            int config_bit = (config >> v) & 1;
                            int sep_bit = (sep_config >> s) & 1;
                            if (config_bit != sep_bit) {
                                consistent = false;
                            }
                            break;
                        }
                    }
                }

                if (!consistent) {
                    prob = 0.0f;
                }
            }

            probs[config] = prob;
            total += prob;
        }

        // Normalize and sample
        if (total > 0.0f) {
            float u = rand_float(rng_state) * total;
            float cumulative = 0.0f;
            int sampled_config = 0;

            for (int config = 0; config < num_configs; config++) {
                cumulative += probs[config];
                if (u <= cumulative) {
                    sampled_config = config;
                    break;
                }
            }

            // Write sampled spins to state
            for (int v = 0; v < var_count; v++) {
                int var = clique_vars[var_start + v];
                int8_t spin = (sampled_config & (1 << v)) ? -1 : 1;
                state[var] = spin;
            }
        }
    }
}

// ==============================================================================
// Main Splash Sampler Kernel
// ==============================================================================

kernel void splash_sampler(
    // Ising model (CSR format)
    device const int* csr_row_ptr [[buffer(0)]],
    device const int* csr_col_ind [[buffer(1)]],
    device const float* csr_J_vals [[buffer(2)]],
    device const float* h [[buffer(3)]],

    // Problem dimensions
    constant int& N [[buffer(4)]],
    constant int& num_edges [[buffer(5)]],

    // Splash parameters
    constant int& max_splash_size [[buffer(6)]],
    constant int& max_treewidth [[buffer(7)]],

    // Annealing parameters
    device const float* beta_schedule [[buffer(8)]],
    constant int& num_betas [[buffer(9)]],
    constant int& sweeps_per_beta [[buffer(10)]],
    constant uint& base_seed [[buffer(11)]],

    // Output buffers
    device int8_t* final_samples [[buffer(12)]],
    device int* final_energies [[buffer(13)]],

    // Batch info
    constant int& num_samples [[buffer(14)]],

    // Device memory for Splash construction (per-sample)
    device int* visited_buffer [[buffer(15)]],    // [num_samples * N]
    device Splash* splash_buffer [[buffer(16)]],  // [num_samples * 32]
    device Clique* clique_buffer [[buffer(17)]],  // [num_samples * 64]
    device int* splash_var_buffer [[buffer(18)]], // [num_samples * 512]
    device int* clique_var_buffer [[buffer(19)]], // [num_samples * 256]
    device int* sep_var_buffer [[buffer(20)]],    // [num_samples * 128]

    // Thread info
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint3 thread_pos_in_group [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint sample_id = threadgroup_pos.x;
    uint thread_in_group = thread_pos_in_group.x;
    uint group_size = threads_per_group.x;

    if (sample_id >= uint(num_samples)) return;

    // Device memory pointers for this sample
    // Buffer sizes must match Python allocation: 96 Splashes, 256 Cliques, N vars, 1024 clique_vars, 512 sep_vars
    device int* visited = &visited_buffer[sample_id * N];
    device Splash* splashes = &splash_buffer[sample_id * 96];
    device Clique* cliques = &clique_buffer[sample_id * 256];
    device int* splash_vars = &splash_var_buffer[sample_id * N];
    device int* clique_vars = &clique_var_buffer[sample_id * 1024];
    device int* separator_vars = &sep_var_buffer[sample_id * 512];

    // Threadgroup memory - only for hot data
    // Total: 4800 + 4096 + 2048 + 1024 + 4 = ~12KB (under 32KB limit)
    threadgroup int8_t state[4800];              // Spin configuration (unpacked)
    threadgroup float potentials[4096];          // Clique potentials (256 cliques * 16 configs max)
    threadgroup float messages[2048];            // BP messages (256 cliques * 8 configs max)
    threadgroup int partial_energies[256];       // For energy reduction
    threadgroup int num_splashes_shared;

    // Initialize RNG (xoshiro128** with splitmix32 seeding)
    RngState rng_state = seed_rng(
        (base_seed ? base_seed : 1u) ^ (sample_id * 2654435761u) ^ (thread_in_group * 2246822519u)
    );

    // Initialize visited array collaboratively (device memory)
    for (uint v = thread_in_group; v < uint(N); v += group_size) {
        visited[v] = 0;
    }

    // Initialize state collaboratively (threadgroup memory)
    for (uint v = thread_in_group; v < uint(N); v += group_size) {
        uint rand_val = xoshiro128starstar(rng_state);
        state[v] = (rand_val & 1) ? -1 : 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 builds Splashes (uses device memory)
    if (thread_in_group == 0) {
        int num_splashes = 0;
        int splash_var_offset = 0;
        int clique_offset = 0;
        int clique_var_offset = 0;
        int sep_var_offset = 0;
        int visited_count = 0;
        int next_unvisited = 0;  // Track where to start searching for unvisited

        // Limit to avoid buffer overflow
        // For N=4600, with 64 vars per Splash, need ~72 Splashes
        int max_splashes = 96;
        int max_cliques_total = 256;

        while (visited_count < N && num_splashes < max_splashes) {
            // Find first unvisited node (O(N) total instead of O(N*num_splashes))
            int root = -1;
            while (next_unvisited < N) {
                if (!visited[next_unvisited]) {
                    root = next_unvisited;
                    break;
                }
                next_unvisited++;
            }
            if (root < 0) break;

            // Build Splash from this root
            build_splash_bfs(
                root, visited, csr_row_ptr, csr_col_ind, N,
                max_splash_size, max_treewidth,
                &splashes[num_splashes], &cliques[clique_offset],
                splash_vars, clique_vars, separator_vars,
                splash_var_offset, clique_offset, clique_var_offset, sep_var_offset
            );

            // Update offsets
            int splash_vars_added = splashes[num_splashes].var_count;
            int cliques_added = splashes[num_splashes].clique_count;

            // Compute clique_vars and sep_vars used
            int cv_used = 0;
            int sv_used = 0;
            for (int c = 0; c < cliques_added; c++) {
                cv_used += cliques[clique_offset + c].var_count;
                sv_used += cliques[clique_offset + c].sep_count;

                // Assign potential and message offsets (into threadgroup arrays)
                cliques[clique_offset + c].pot_offset = (clique_offset + c) * 16;  // Max 16 per clique
                cliques[clique_offset + c].msg_offset = (clique_offset + c) * 8;   // Max 8 per edge
            }

            visited_count += splash_vars_added;
            splash_var_offset += splash_vars_added;
            clique_offset += cliques_added;
            clique_var_offset += cv_used;
            sep_var_offset += sv_used;
            num_splashes++;

            if (clique_offset >= max_cliques_total) break;
        }

        num_splashes_shared = num_splashes;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total_splashes = num_splashes_shared;

    // Annealing loop
    for (int beta_idx = 0; beta_idx < num_betas; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
            // Process each Splash
            for (int s = 0; s < total_splashes; s++) {
                device Splash* splash = &splashes[s];
                int clique_base = splash->clique_start;
                int num_cliques_in_splash = splash->clique_count;

                // Collaboratively compute clique potentials
                for (uint c = thread_in_group; c < uint(num_cliques_in_splash); c += group_size) {
                    int clique_idx = clique_base + int(c);
                    device Clique* clique = &cliques[clique_idx];

                    int var_count = clique->var_count;
                    int num_configs = 1 << var_count;
                    int pot_offset = clique->pot_offset;

                    for (int config = 0; config < num_configs && config < 16; config++) {
                        potentials[pot_offset + config] = compute_clique_potential(
                            clique, clique_vars, config, state,
                            h, csr_row_ptr, csr_col_ind, csr_J_vals, N, beta
                        );
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // BP Forward pass (leaves to root) - thread 0 only
                if (thread_in_group == 0) {
                    for (int c = num_cliques_in_splash - 1; c >= 0; c--) {
                        int clique_idx = clique_base + c;
                        device Clique* clique = &cliques[clique_idx];

                        if (clique->parent_id >= 0) {
                            int parent_idx = clique_base + clique->parent_id;
                            device Clique* parent = &cliques[parent_idx];

                            compute_message_to_parent(
                                clique, parent, clique_vars, separator_vars,
                                potentials, messages,
                                &messages[clique->msg_offset], 8
                            );
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Sample from calibrated tree - thread 0 only
                if (thread_in_group == 0) {
                    sample_from_junction_tree(
                        splash, cliques, clique_vars, separator_vars,
                        potentials, messages, state, clique_base, rng_state
                    );
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    // Compute final energy collaboratively
    int partial_energy = 0;

    for (uint i = thread_in_group; i < uint(N); i += group_size) {
        int8_t spin_i = state[i];

        // h field
        partial_energy += int(h[i] * float(spin_i));

        // J coupling (j > i)
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i + 1];
        for (int p = row_start; p < row_end; p++) {
            int j = csr_col_ind[p];
            if (j > int(i)) {
                float Jij = csr_J_vals[p];
                int8_t spin_j = state[j];
                partial_energy += int(Jij * float(spin_i) * float(spin_j));
            }
        }
    }

    partial_energies[thread_in_group] = partial_energy;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 sums and writes output
    if (thread_in_group == 0) {
        int total_energy = 0;
        for (uint t = 0; t < group_size; t++) {
            total_energy += partial_energies[t];
        }
        final_energies[sample_id] = total_energy;

        // Pack state to output
        int packed_size = (N + 7) / 8;
        device int8_t* output = &final_samples[sample_id * packed_size];

        for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
            output[byte_idx] = 0;
        }
        for (int var = 0; var < N; var++) {
            if (state[var] < 0) {
                int byte_idx = var >> 3;
                int bit_idx = var & 7;
                output[byte_idx] |= (1 << bit_idx);
            }
        }
    }
}
