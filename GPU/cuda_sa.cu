typedef signed char int8_t;
typedef unsigned int uint;

// Control flag values for persistent kernel
#define CONTROL_RUNNING 0
#define CONTROL_STOP 1
#define CONTROL_DRAIN 2

// Kernel state values (tracked by output controller)
#define STATE_RUNNING 0
#define STATE_IDLE 1

// Debug flags (can be overridden via -D compiler flags from Python)
// These defaults are used when not specified at compile time.
// To enable debug output, pass debug_verbose=1, debug_kernel=1, or debug_workers=1
// to CudaKernelRealSA constructor in cuda_kernel.py
#ifndef DEBUG_KERNEL
#define DEBUG_KERNEL 0
#endif
#ifndef DEBUG_WORKERS
#define DEBUG_WORKERS 0  // Disabled in hot paths for performance
#endif
#ifndef DEBUG_VERBOSE
#define DEBUG_VERBOSE 0 // Disabled - very expensive printf in SA loop
#endif

// Profiling macros (zero overhead when PROFILE_REGIONS is not defined)
#ifdef PROFILE_REGIONS
#define PROF_T(var) var = clock64()
#define PROF_ACCUM(arr, idx, start_var) arr[idx] += clock64() - start_var
#define PROF_INC(arr, idx) arr[idx]++
#define SA_NUM_REGIONS 10
#else
#define PROF_T(var)
#define PROF_ACCUM(arr, idx, start_var)
#define PROF_INC(arr, idx)
#endif

// Fast math constants
#define RNG_SCALE 2.32830643653869628906e-10f  // 1.0f / 2^32

// Job descriptor for ring buffer
struct JobDesc {
    int job_id;
    int num_reads;
    int num_betas;             // Number of temperature steps (renamed from num_sweeps)
    int num_sweeps_per_beta;

    // CSR pointers (per-job CSR data)
    const int* csr_row_ptr;    // Pointer to CSR row pointer array for this job
    const int* csr_col_ind;    // Pointer to CSR column indices for this job
    int N;

    // Input arrays (device pointers)
    float* h;                  // Linear bias array
    int h_size;                // Size of h array
    const int8_t* csr_J_vals;  // CSR values array (moved from J pointer)
    int J_size;                // Size of J array
    float* beta_schedule;      // Per-job beta schedule

    // Output buffers
    int8_t* output_samples;    // Output buffer for samples (num_reads * packed_size)
    float* output_energies;    // Output buffer for energies (num_reads)

    // RNG seed for reproducibility/randomness
    unsigned int seed;         // Base seed for RNG initialization
};

// Output slot for per-block results (no ring buffer needed)
struct OutputSlot {
    volatile int ready;      // 0 = empty, 1 = has result, 2 = collected
    int job_id;
    float min_energy;
    float avg_energy;
    int num_reads;           // Number of samples/energies
    int N;                   // Variables per sample
    int samples_offset;      // Offset into samples pool (in floats)
    int energies_offset;     // Offset into energies pool (in floats)
};

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

// Compute delta energy for flipping a single variable (unpacked state version)
__device__ int8_t get_flip_energy_unpacked(
    int var,
    const int8_t* unpacked_state,
    const int* csr_row_ptr,
    const int* csr_col_ind,
    const int8_t* csr_J_vals,
    int n,
    const float* h = NULL
) {
    // Use read-only cache for CSR reads (constant data)
    const int start = __ldg(&csr_row_ptr[var]);
    const int end = __ldg(&csr_row_ptr[var + 1]);

    int energy = 0;

    // Add linear bias term if provided
    if (h != NULL) {
        energy += (int)__ldg(&h[var]);
    }

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        const int neighbor = __ldg(&csr_col_ind[p]);
        const int8_t Jij = __ldg(&csr_J_vals[p]);
        energy += unpacked_state[neighbor] * Jij;
    }

    // Delta energy = -2 * state[var] * energy
    return (int8_t)(-2 * unpacked_state[var] * energy);
}

// Compute delta energy for flipping a single variable (packed state version - legacy)
__device__ int8_t get_flip_energy(
    int var,
    const int8_t* packed_state,
    const int* csr_row_ptr,
    const int* csr_col_ind,
    const int8_t* csr_J_vals,
    int n,
    const float* h = NULL
) {
    // Use read-only cache for CSR reads (constant data)
    const int start = __ldg(&csr_row_ptr[var]);
    const int end = __ldg(&csr_row_ptr[var + 1]);

    int energy = 0;

    // Add linear bias term if provided
    if (h != NULL) {
        energy += (int)__ldg(&h[var]);
    }

    #pragma unroll 20
    for (int p = start; p < end; ++p) {
        const int neighbor = __ldg(&csr_col_ind[p]);
        // CSR col_ind is validated on host to be in [0, n)
        const int8_t Jij = __ldg(&csr_J_vals[p]);
        const int8_t neighbor_spin = get_spin_packed(neighbor, packed_state);
        energy += neighbor_spin * Jij;
    }

    // Delta energy = -2 * state[var] * energy
    int8_t var_spin = get_spin_packed(var, packed_state);
    return (int8_t)(-2 * var_spin * energy);
}


// Persistent kernel with real simulated annealing
// ARCHITECTURE:
// - Thread 0: Input controller (dequeues jobs)
// - Thread 1: Output controller (collects results, tracks state)
// - Threads 2+: Worker threads (run real SA algorithm)
__global__ void cuda_sa_persistent_real(
    unsigned long long* input_ring_ptrs,  // Ring buffer of pointers to JobDesc
    int input_ring_size,
    volatile int* input_head,
    volatile int* input_tail,
    volatile int* host_writing_mutex,   // Mutex: 1 when host is writing, 0 otherwise
    OutputSlot* output_slots,           // Per-block output slots
    volatile int* control_flag,
    volatile int* kernel_state,         // STATE_RUNNING or STATE_IDLE
    float* samples_buffer_pool,         // Pre-allocated output buffer (float32)
    float* energies_buffer_pool,        // Pre-allocated output buffer (float32)
    int max_samples_per_job,            // Max floats for samples per job
    int max_energies_per_job,           // Max floats for energies per job
    int8_t* __restrict__ delta_energy_workspace,
    int max_N                           // Max problem size (workspace capacity per thread)
#ifdef PROFILE_REGIONS
    , long long* profile_output         // Per-thread profiling counters
#endif
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // This block's dedicated output slot
    OutputSlot* my_output = &output_slots[bid];

    // Startup info (gated to block 0 only, once)
    if (tid == 0 && bid == 0) {
#if DEBUG_KERNEL
        printf("[KERNEL] Persistent SA kernel started (num_blocks=%d, threads_per_block=%d)\n", gridDim.x, blockDim.x);
#endif
        *kernel_state = STATE_IDLE;
        __threadfence_system();
    }

    __shared__ JobDesc shared_job;
    __shared__ bool has_job;
    __shared__ int worker_results[1024];
    __shared__ int exit_flag;
    __shared__ int dequeue_mutex;  // Mutex for ring buffer dequeue

    int job_output_offset = bid;
    int loop_count = 0;

    if (tid == 0) {
        exit_flag = 0;
        has_job = false;
        dequeue_mutex = 0;
        *kernel_state = STATE_IDLE;
        __threadfence_system();
    }
  
    while (true) {
        // Make sure we all see the same shared memory.
        __syncthreads();
        if (exit_flag) break;

        // INPUT CONTROLLER (Thread 0 on every block): each block claims jobs via atomic head
        if (tid == 0 && !has_job) {
            loop_count++;

            // Exit or drain the queue.
            int control_top = *control_flag;
            if (control_top == CONTROL_STOP) {
                exit_flag = 1;
            }

            // Try dequeue using atomic CAS
            int slot = -1;

            // Wait for host to signal batch ready (host_writing_mutex == 1)
            while (true) {
                __threadfence_system();  // Ensure we see host writes

                // Force volatile read from memory
                int signal = *(volatile int*)host_writing_mutex;
                if (signal == 1) {
                    break;  // Batch ready
                }

                // Check if we should exit
                int control = *(volatile int*)control_flag;
                if (control == CONTROL_STOP) {
                    exit_flag = 1;
                    break;
                }

                __nanosleep(1000);  // Wait for host to finish writing batch
            }

            if (!exit_flag) {
                // Acquire GPU-side dequeue mutex
                while (atomicCAS(&dequeue_mutex, 0, 1) != 0) {
                    __nanosleep(1000);  // Spin with small sleep
                }

                // Critical section: read head/tail and atomicCAS
                __threadfence_system();  // Ensure we see host writes
                int head = *(volatile int*)input_head;
                int tail = *(volatile int*)input_tail;

                if (head != tail) {
                    // Jobs available - claim one atomically
                    int observed = atomicCAS((int*)input_head, head, head + 1);
                    if (observed == head) {
                        // Successfully claimed job at position head
                        slot = head % input_ring_size;

                        // Check if this was the last job in the batch
                        if (head + 1 == tail) {
                            // We just picked up the last job - signal host it can write next batch
                            atomicCAS((int*)host_writing_mutex, 1, 0);
#if DEBUG_KERNEL
                            printf("[KERNEL] Block %d picked up last job (head=%d, tail=%d), signaling host\n", bid, head, tail);
#endif
                        }
                    }
                }

                // Release GPU-side mutex
                dequeue_mutex = 0;
                __threadfence_system();
            }

            if (slot != -1) {
                // Read pointer from ring buffer
                // Safe because host_writing_mutex ensures Python isn't writing
                unsigned long long jobdesc_ptr = input_ring_ptrs[slot];

                // Dereference pointer to get JobDesc
                JobDesc* job_ptr = (JobDesc*)jobdesc_ptr;
                shared_job = *job_ptr;

                if (shared_job.num_reads > 0) {
                    has_job = true;
                    for (int i = 0; i < shared_job.num_reads; i++) {
                        worker_results[i] = 0;
                    }

                    *kernel_state = STATE_RUNNING;
                    __threadfence_system();

                    // Event-driven debug: print when job is successfully dequeued
#if DEBUG_KERNEL
                    printf("[KERNEL] Block %d dequeued job_id=%d with num_reads=%d, num_betas=%d\n",
                           bid, shared_job.job_id, shared_job.num_reads, shared_job.num_betas);
#endif

                    // One-time debug: verify CSR and h metadata for this job (on the block that dequeued)
#if DEBUG_KERNEL
                    int n = shared_job.N;
                    const int* csr_row_ptr = shared_job.csr_row_ptr;
                    const int* csr_col_ind = shared_job.csr_col_ind;
                    const int8_t* csr_J_vals = shared_job.csr_J_vals;
                    int nnz = csr_row_ptr[n] - csr_row_ptr[0];
                    printf("[KERNEL] Dequeued job debug: h_ptr=%p h_size=%d N=%d | row_ptr[0]=%d row_ptr[n]=%d nnz=%d\n",
                           shared_job.h, shared_job.h_size, n, csr_row_ptr[0], csr_row_ptr[n], nnz);
                    if (n > 0) {
                        int deg0 = csr_row_ptr[1] - csr_row_ptr[0];
                        int deg_last = csr_row_ptr[n] - csr_row_ptr[n-1];
                        printf("[KERNEL] Dequeued job debug: deg(first)=%d deg(last)=%d | first J=%d first col=%d\n",
                               deg0, deg_last, (int)csr_J_vals[0], csr_col_ind[0]);
                    }
#endif
                }
            }
        }
        // Ensure all threads see a new job (the zeroed worker_results and has_job) before they start
        __syncthreads();

        // WORKER THREADS: Run real SA (on every block)
        int worker_id = tid;

        // Only run if this worker hasn't completed yet
        if (has_job && worker_id < shared_job.num_reads && worker_results[worker_id] == 0) {
            int n = shared_job.N;
            int num_reads = shared_job.num_reads;
            int num_betas = shared_job.num_betas;
            int sweeps_per_beta = shared_job.num_sweeps_per_beta;

            #if DEBUG_VERBOSE
            printf("[KERNEL] Block %d worker %d starting SA (job_id=%d, num_reads=%d)\n",
                    bid, worker_id, shared_job.job_id, num_reads);
            #endif

            // Get CSR pointers from job (per-job CSR data)
            const int* csr_row_ptr = shared_job.csr_row_ptr;
            const int* csr_col_ind = shared_job.csr_col_ind;
            const int8_t* csr_J_vals = shared_job.csr_J_vals;

            int packed_size = (n + 7) / 8;
            int8_t packed_state[640];  // Keep for final output only

            // OPTIMIZATION: Use unpacked state during SA (much faster than bit operations)
            int8_t unpacked_state[5000];  // Thread-local dense spin array

            // Delta energy workspace for this thread (unique per global thread)
            // Use max_N (workspace capacity) not n (actual problem size) to avoid overlapping workspaces
            int global_thread_id = bid * blockDim.x + tid;
            int8_t* delta_energy = &delta_energy_workspace[global_thread_id * max_N];

            // Initialize RNG (ensure non-zero seed)
            // Combine job seed with job_id and worker_id for uniqueness
            unsigned int rng_state = (shared_job.seed ^ ((shared_job.job_id + 1) ^ (worker_id + 1))) * 12345u;
            if (rng_state == 0) rng_state = 0xdeadbeef;

            // Generate random initial state (directly to unpacked array)
            for (int var = 0; var < n; var++) {
                unsigned int rand_val = xorshift32(rng_state);
                unpacked_state[var] = (rand_val & 1) ? -1 : 1;
            }

            // Build initial delta_energy array using unpacked state
            // Include h field in delta energy calculation
            for (int var = 0; var < n; var++) {
                delta_energy[var] = get_flip_energy_unpacked(var, unpacked_state, csr_row_ptr, csr_col_ind, csr_J_vals, n, shared_job.h);
            }

            // Compute initial energy (h + J terms)
            int current_energy = 0;
            for (int i = 0; i < n; i++) {
                int8_t spin_i = unpacked_state[i];

                // Add h term
                if (shared_job.h != NULL && i < shared_job.h_size) {
                    current_energy += (int)(__ldg(&shared_job.h[i]) * spin_i);
                }

                // Add J terms (only count each edge once with j > i)
                const int start = __ldg(&csr_row_ptr[i]);
                const int end = __ldg(&csr_row_ptr[i + 1]);
                for (int p = start; p < end; ++p) {
                    const int j = __ldg(&csr_col_ind[p]);
                    if (j > i) {  // Count each edge once
                        const int8_t Jij = __ldg(&csr_J_vals[p]);
                        const int8_t spin_j = unpacked_state[j];
                        current_energy += Jij * spin_i * spin_j;
                    }
                }
            }


            // Perform SA sweeps

#ifdef PROFILE_REGIONS
            long long prof[SA_NUM_REGIONS] = {0};
            long long _t0, _t1, _t2, _t3, _t4;
#endif

            PROF_T(_t0);  // SA_TOTAL start
            for (int beta_idx = 0; beta_idx < num_betas; beta_idx++) {
                PROF_T(_t1);  // BETA_OVERHEAD start
                float beta = shared_job.beta_schedule[beta_idx];
                float threshold = 22.18f / beta;
                PROF_ACCUM(prof, 1, _t1);  // BETA_OVERHEAD end

                for (int sweep = 0; sweep < sweeps_per_beta; sweep++) {
                    PROF_T(_t2);  // SWEEP_TOTAL start
                    for (int var = 0; var < n; var++) {
                        PROF_T(_t3);  // Per-var start
                        int8_t de = delta_energy[var];

                        if (de >= threshold) {
                            PROF_ACCUM(prof, 4, _t3);  // THRESHOLD_SKIP
                            PROF_INC(prof, 9);          // SKIP_COUNT
                            continue;
                        }

                        PROF_T(_t4);  // ACCEPT_DECIDE start
                        bool flip_spin = false;

                        if (de <= 0) {
                            flip_spin = true;
                        } else {
                            // Fast math: use __expf() and optimized RNG conversion
                            const float accept_prob = __expf(-__int2float_rn(de) * beta);
                            const float rand_uniform = __uint2float_rn(xorshift32(rng_state)) * RNG_SCALE;
                            flip_spin = (accept_prob > rand_uniform);
                        }
                        PROF_ACCUM(prof, 5, _t4);  // ACCEPT_DECIDE end

                        if (flip_spin) {
                            PROF_T(_t4);  // FLIP_TOTAL start
                            current_energy += de;

                            // OPTIMIZATION: Use unpacked state (matching Metal logic)
                            const int8_t var_spin = unpacked_state[var];  // BEFORE flip
                            const int8_t multiplier = 4 * var_spin;  // Matching Metal
                            const int start = __ldg(&csr_row_ptr[var]);
                            const int end = __ldg(&csr_row_ptr[var + 1]);

                            PROF_T(_t3);  // NEIGHBOR_LOOP start (reuse _t3)
                            for (int p = start; p < end; ++p) {
                                const int neighbor = __ldg(&csr_col_ind[p]);
                                const int8_t Jij = __ldg(&csr_J_vals[p]);
                                const int8_t neighbor_spin = unpacked_state[neighbor];
                                delta_energy[neighbor] += multiplier * Jij * neighbor_spin;
                            }
                            PROF_ACCUM(prof, 7, _t3);  // NEIGHBOR_LOOP end

                            // Flip spin (just negate, no bit packing)
                            unpacked_state[var] = -var_spin;
                            delta_energy[var] = -de;
                            PROF_ACCUM(prof, 6, _t4);  // FLIP_TOTAL end
                            PROF_INC(prof, 8);          // FLIP_COUNT
                        }
                    }
                    PROF_ACCUM(prof, 2, _t2);  // SWEEP_TOTAL end
                }
            }
            PROF_ACCUM(prof, 0, _t0);  // SA_TOTAL end

#ifdef PROFILE_REGIONS
            // Write profiling data to global buffer
            int gid = bid * blockDim.x + tid;
            for (int r = 0; r < SA_NUM_REGIONS; r++)
                profile_output[gid * SA_NUM_REGIONS + r] = prof[r];
#endif

            // Pack final state back to bit format (for compatibility with existing code)
            for (int i = 0; i < n; i++) {
                set_spin_packed(i, unpacked_state[i], packed_state);
            }

            // Write results to output buffer
            int base_samples_offset = job_output_offset * max_samples_per_job;
            int base_energies_offset = job_output_offset * max_energies_per_job;

            // Store final state as samples (directly from unpacked state)
            int sample_idx = base_samples_offset + worker_id * n;
            for (int i = 0; i < n; i++) {
                samples_buffer_pool[sample_idx + i] = (float)unpacked_state[i];
            }

            // Store final energy
            int energy_idx = base_energies_offset + worker_id;
            energies_buffer_pool[energy_idx] = (float)current_energy;

            #if DEBUG_VERBOSE
            printf("[KERNEL] Block %d worker %d finished: energy=%.1f (job_id=%d)\n", bid, worker_id, (float)current_energy, shared_job.job_id);
            #endif

            worker_results[worker_id] = 1;
        }

        __syncthreads();

        // OUTPUT CONTROLLER (Thread 0 on every block)
        if (has_job && tid == 0) {
            // Event-driven debug: print when output controller starts waiting
#if DEBUG_KERNEL
            printf("[KERNEL] Block %d output controller: waiting for %d workers\n", bid, shared_job.num_reads);
#endif

            // Wait for all workers to complete
            bool all_done = false;
            int wait_count = 0;
            while (!all_done) {
                all_done = true;
                int completed = 0;
                for (int i = 0; i < shared_job.num_reads; i++) {
                    // Use volatile read to ensure we see latest value from workers
                    int result_val = *(volatile int*)&worker_results[i];
                    if (result_val == 1) {
                        completed++;
                    } else {
                        all_done = false;
                    }
                }
                if (!all_done) {
                    wait_count++;
                    __nanosleep(100000);  // 100us sleep between checks
                }
            }

            // Compute min/avg energy
            int base_energies_offset = job_output_offset * max_energies_per_job;
            float min_e = 0.0f, sum_e = 0.0f;
            for (int i = 0; i < shared_job.num_reads; i++) {
                float e = energies_buffer_pool[base_energies_offset + i];
                if (i == 0 || e < min_e) min_e = e;
                sum_e += e;
            }

            // Wait for host to acknowledge previous result (ready==2 means host read it)
            // or ready==0 means slot is fresh
            while (my_output->ready == 1) {
                __threadfence_system();
                __nanosleep(100000);  // 100us
            }

            // Write to output slot
            my_output->job_id = shared_job.job_id;
            my_output->min_energy = min_e;
            my_output->avg_energy = sum_e / shared_job.num_reads;
            my_output->num_reads = shared_job.num_reads;
            my_output->N = shared_job.N;
            my_output->samples_offset = job_output_offset * max_samples_per_job;
            my_output->energies_offset = job_output_offset * max_energies_per_job;

            // Mark ready (host will poll this)
            my_output->ready = 1;
            __threadfence_system();

#if DEBUG_KERNEL
            printf("[KERNEL] Block %d wrote result for job_id=%d, min_e=%.1f\n", bid, shared_job.job_id, min_e);
#endif

            // Wait for host to collect (ready==2), then reset
            while (my_output->ready != 2) {
                __threadfence_system();
                int control = *control_flag;
                if (control == CONTROL_STOP) break;
                __nanosleep(1000000);  // 1ms polling - was 100ms causing serialization
            }
            my_output->ready = 0;
            has_job = false;
            *kernel_state = STATE_IDLE;
            __threadfence_system();
            continue;
        }

        // OPTIMIZATION: Idle unused threads to reduce GPU resource contention
        if (!has_job || worker_id >= shared_job.num_reads) {
            // Thread is not needed - sleep to free up GPU resources
            __nanosleep(10000000);  // 10ms - was 500ms
        }
    }
}

// Per-job oneshot SA kernel: no ring buffer, no polling, no control flags.
// Each thread runs one SA read independently.
// Grid=(num_blocks,) block=(threads_per_block,).
// Threads beyond num_reads are idle.
__global__ void cuda_sa_oneshot(
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_ind,
    const int8_t* __restrict__ csr_J_vals,
    const float* __restrict__ h,
    const float* __restrict__ beta_schedule,
    int N,
    int num_reads,
    int num_betas,
    int num_sweeps_per_beta,
    unsigned int base_seed,
    int8_t* __restrict__ delta_energy_workspace,
    float* __restrict__ out_samples,
    float* __restrict__ out_energies
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= num_reads) return;

    // Thread-local unpacked state
    int8_t unpacked_state[5000];

    // Delta energy workspace for this thread
    int8_t* delta_energy = &delta_energy_workspace[global_tid * N];

    // Initialize RNG
    unsigned int rng_state = (base_seed ^ (global_tid + 1)) * 12345u;
    if (rng_state == 0) rng_state = 0xdeadbeef;

    // Random initial state
    for (int var = 0; var < N; var++) {
        unsigned int rv = xorshift32(rng_state);
        unpacked_state[var] = (rv & 1) ? -1 : 1;
    }

    // Build initial delta_energy array
    for (int var = 0; var < N; var++) {
        delta_energy[var] = get_flip_energy_unpacked(
            var, unpacked_state, csr_row_ptr, csr_col_ind,
            csr_J_vals, N, h
        );
    }

    // Compute initial energy
    int current_energy = 0;
    for (int i = 0; i < N; i++) {
        int8_t spin_i = unpacked_state[i];
        current_energy += (int)(__ldg(&h[i]) * spin_i);
        const int start = __ldg(&csr_row_ptr[i]);
        const int end = __ldg(&csr_row_ptr[i + 1]);
        for (int p = start; p < end; ++p) {
            const int j = __ldg(&csr_col_ind[p]);
            if (j > i) {
                const int8_t Jij = __ldg(&csr_J_vals[p]);
                current_energy += Jij * spin_i * unpacked_state[j];
            }
        }
    }

    // SA sweeps
    for (int beta_idx = 0; beta_idx < num_betas; beta_idx++) {
        float beta = beta_schedule[beta_idx];
        float threshold = 22.18f / beta;

        for (int sweep = 0; sweep < num_sweeps_per_beta; sweep++) {
            for (int var = 0; var < N; var++) {
                int8_t de = delta_energy[var];
                if (de >= threshold) continue;

                bool flip_spin = false;
                if (de <= 0) {
                    flip_spin = true;
                } else {
                    const float accept_prob = __expf(
                        -__int2float_rn(de) * beta
                    );
                    const float rand_uniform = (
                        __uint2float_rn(xorshift32(rng_state))
                        * RNG_SCALE
                    );
                    flip_spin = (accept_prob > rand_uniform);
                }

                if (flip_spin) {
                    current_energy += de;
                    const int8_t var_spin = unpacked_state[var];
                    const int8_t multiplier = 4 * var_spin;
                    const int start = __ldg(&csr_row_ptr[var]);
                    const int end = __ldg(&csr_row_ptr[var + 1]);

                    for (int p = start; p < end; ++p) {
                        const int neighbor = __ldg(&csr_col_ind[p]);
                        const int8_t Jij = __ldg(&csr_J_vals[p]);
                        const int8_t ns = unpacked_state[neighbor];
                        delta_energy[neighbor] += multiplier * Jij * ns;
                    }

                    unpacked_state[var] = -var_spin;
                    delta_energy[var] = -de;
                }
            }
        }
    }

    // Write results: unpacked spins as float and energy
    int sample_offset = global_tid * N;
    for (int i = 0; i < N; i++) {
        out_samples[sample_offset + i] = (float)unpacked_state[i];
    }
    out_energies[global_tid] = (float)current_energy;
}

}  // extern "C"