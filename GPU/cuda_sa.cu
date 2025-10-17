typedef signed char int8_t;
typedef unsigned int uint;

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

// Bit-packing helpers for thread-local state
// Pack 8 spins into 1 byte: bit i stores spin i (0=+1, 1=-1)
__device__ int8_t get_spin_packed(int var, const int8_t* packed_state) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    int bit = (packed_state[byte_idx] >> bit_idx) & 1;
    return bit ? -1 : 1;  // 0 -> +1, 1 -> -1
}

__device__ void set_spin_packed(int var, int8_t spin, int8_t* packed_state) {
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

__device__ void flip_spin_packed(int var, int8_t* packed_state) {
    int byte_idx = var >> 3;  // var / 8
    int bit_idx = var & 7;    // var % 8
    packed_state[byte_idx] ^= (1 << bit_idx);  // Toggle bit
}

// Compute delta energy for flipping a single variable
// Delta energy = -2 * state[var] * energy
// NOTE: CSR structure is validated on host side, so no bounds checks needed here
__device__ int8_t get_flip_energy(
    int var,
    const int8_t* packed_state,
    const int* csr_row_ptr,
    const int* csr_col_ind,
    const int8_t* csr_J_vals,
    int n
) {
    int start = csr_row_ptr[var];
    int end = csr_row_ptr[var + 1];

    int energy = 0;

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        int neighbor = csr_col_ind[p];
        // CSR col_ind is validated on host to be in [0, n)
        int8_t Jij = csr_J_vals[p];
        int8_t neighbor_spin = get_spin_packed(neighbor, packed_state);
        energy += neighbor_spin * Jij;
    }

    // Delta energy = -2 * state[var] * energy
    int8_t var_spin = get_spin_packed(var, packed_state);
    return (int8_t)(-2 * var_spin * energy);
}

// CUDA Simulated Annealing kernel - with delta_energy array in global memory
// Batched multi-problem evaluation for reduced kernel overhead
//
// BATCHED DISPATCH:
// Each problem is assigned to a contiguous range of threads:
// - problem_id = global_thread_id / num_reads
// - read_id_in_problem = global_thread_id % num_reads
// All problems are processed in parallel in a single kernel launch
//
// GUARDS:
// 1. thread_id >= num_threads: Prevents out-of-bounds writes to output arrays.
//    The kernel launch rounds up to full blocks, so total_threads may exceed num_threads.
//    This guard is essential and always active.
//
// 2. thread_id >= workspace_capacity: Protects the delta_energy_workspace.
//    Python allocates workspace based on GPU SM count, this is a safety fallback.
//
// CSR VALIDATION:
// The CSR structure (row_ptr, col_ind, J_vals) is validated on the host side,
// so the kernel can skip bounds checks for performance.
//
__global__ void cuda_simulated_annealing(
    const int* __restrict__ all_csr_row_ptr,     // Concatenated row pointers for all problems
    const int* __restrict__ all_csr_col_ind,     // Concatenated column indices for all problems
    const int8_t* __restrict__ all_csr_J_vals,   // Concatenated J values for all problems
    const int* __restrict__ row_ptr_offsets,     // [num_problems+1] offset into all_csr_row_ptr
    const int* __restrict__ col_ind_offsets,     // [num_problems+1] offset into all_csr_col_ind
    int num_problems,                             // Total number of problems in batch
    const float* __restrict__ beta_schedule,
    int N,
    int num_betas,
    int sweeps_per_beta,
    int num_reads,                                // Reads per individual problem
    unsigned int base_seed,
    int8_t* __restrict__ output_samples,
    float* __restrict__ output_energies,
    int8_t* __restrict__ delta_energy_workspace,  // Global memory: workspace_capacity * N bytes
    int workspace_capacity  // Maximum threads workspace can support
) {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = num_problems * num_reads;  // Total threads needed for all problems

    // Guard 1: Prevent out-of-bounds writes to output arrays
    if (global_thread_id >= num_threads) {
        return;
    }

    // Guard 2: Protect delta_energy_workspace from out-of-bounds access
    // Workspace allocated in Python with size = workspace_capacity × N bytes
    // This should never trigger if Python assertion works, but prevents GPU crash if it fails
    if (global_thread_id >= workspace_capacity) {
        return;
    }

    // Determine which problem this thread is working on
    int problem_id = global_thread_id / num_reads;
    int read_id_in_problem = global_thread_id % num_reads;

    // Get CSR offsets for this specific problem
    int row_ptr_start = row_ptr_offsets[problem_id];
    int col_ind_start = col_ind_offsets[problem_id];

    // Point to this problem's CSR data
    const int* csr_row_ptr = &all_csr_row_ptr[row_ptr_start];
    const int* csr_col_ind = &all_csr_col_ind[col_ind_start];
    const int8_t* csr_J_vals = &all_csr_J_vals[col_ind_start];

    int n = N;
    int num_beta_values = num_betas;
    int sweeps_per_beta_val = sweeps_per_beta;
    int packed_size = (n + 7) / 8;  // bytes needed for bit-packed state

    // Thread-local memory for state only (small enough to fit)
    int8_t packed_state[576];  // ~4600 bits / 8 = 575 bytes (for N=4600)

    // Delta energy workspace for this thread
    int8_t* delta_energy = &delta_energy_workspace[global_thread_id * n];

    // Initialize delta_energy array to zero
    for (int i = 0; i < n; i++) {
        delta_energy[i] = 0;
    }

    // Initialize RNG state with unique seed per thread
    unsigned int rng_state = (base_seed ? base_seed : 1u) ^ (global_thread_id * 12345u);

    // Generate random initial state (bit-packed)
    for (int byte_idx = 0; byte_idx < packed_size; byte_idx++) {
        packed_state[byte_idx] = 0;  // Clear all bits
    }
    for (int var = 0; var < n; var++) {
        unsigned int rand_val = xorshift32(rng_state);
        int8_t spin = (rand_val & 1) ? -1 : 1;  // Random ±1
        set_spin_packed(var, spin, packed_state);
    }

    // Build initial delta_energy array
    for (int var = 0; var < n; var++) {
        delta_energy[var] = get_flip_energy(var, packed_state, csr_row_ptr, csr_col_ind, csr_J_vals, n);
    }

    // Compute initial energy (J contribution only - h is constant per spin)
    int current_energy = 0;
    for (int i = 0; i < n; i++) {
        int8_t spin_i = get_spin_packed(i, packed_state);
        int start = csr_row_ptr[i];
        int end = csr_row_ptr[i + 1];
        for (int p = start; p < end; ++p) {
            int j = csr_col_ind[p];
            if (j > i) {  // Count each edge once
                int8_t Jij = csr_J_vals[p];
                int8_t spin_j = get_spin_packed(j, packed_state);
                current_energy += Jij * spin_i * spin_j;
            }
        }
    }

    // Perform sweeps across beta schedule
    for (int beta_idx = 0; beta_idx < num_beta_values; beta_idx++) {
        float beta = beta_schedule[beta_idx];

        // D-Wave optimization: threshold to skip impossible flips
        float threshold = 22.18f / beta;

        for (int sweep = 0; sweep < sweeps_per_beta_val; sweep++) {
            // Sequential variable ordering (matching D-Wave default)
            for (int var = 0; var < n; var++) {
                // Use cached delta_energy value
                int8_t de = delta_energy[var];

                // Skip if delta energy too large (D-Wave optimization)
                if (de >= threshold) continue;

                bool flip_spin = false;

                // Metropolis-Hastings acceptance rule
                if (de <= 0) {
                    flip_spin = true;
                } else {
                    unsigned int rand_val = xorshift32(rng_state);
                    float prob = exp(-float(de) * beta);
                    float rand_normalized = float(rand_val) / 4294967295.0f;

                    if (prob > rand_normalized) {
                        flip_spin = true;
                    }
                }

                if (flip_spin) {
                    current_energy += de;

                    // Get current spin value
                    int8_t var_spin = get_spin_packed(var, packed_state);

                    // Update delta energies of all neighbors
                    // CSR structure is validated on host, so no bounds checks needed
                    int multiplier = 4 * var_spin;
                    int start = csr_row_ptr[var];
                    int end = csr_row_ptr[var + 1];

                    for (int p = start; p < end; ++p) {
                        int neighbor = csr_col_ind[p];
                        int8_t Jij = csr_J_vals[p];
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
    
    // Store output
    for (int i = 0; i < packed_size; i++) {
        output_samples[global_thread_id * packed_size + i] = packed_state[i];
    }
    output_energies[global_thread_id] = (float)current_energy;
}

}  // extern "C"

