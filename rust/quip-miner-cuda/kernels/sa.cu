// SPDX-License-Identifier: AGPL-3.0-or-later
// CUDA simulated annealing — explicit per-job (no nonce/slot economy).
// One thread = one independent read. Host uploads one Ising problem (CSR + h)
// and a beta schedule; kernel writes final spins for each read.
//
// Ported algorithm semantics from GPU/cuda_sa.cu (Metropolis, geometric beta
// ladder, xorshift32 RNG) without the self-feeding 3-slot control plane.

#define RNG_SCALE 2.32830643653869628906e-10f  // 1.0f / 2^32

extern "C" {

__device__ unsigned int xorshift32(unsigned int &state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Effective field h_i + sum_j J_ij s_j (float couplings).
__device__ float effective_field(
    int var,
    const signed char* __restrict__ state,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ j_vals,
    const float* __restrict__ h_vals
) {
    float heff = __ldg(&h_vals[var]);
    int start = __ldg(&row_ptr[var]);
    int end = __ldg(&row_ptr[var + 1]);
    #pragma unroll 8
    for (int p = start; p < end; ++p) {
        int nbr = __ldg(&col_ind[p]);
        heff += __ldg(&j_vals[p]) * (float)state[nbr];
    }
    return heff;
}

// ΔE for flipping var under positive-sign Ising convention.
__device__ float flip_delta(
    int var,
    const signed char* __restrict__ state,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ j_vals,
    const float* __restrict__ h_vals
) {
    float s = (float)state[var];
    return -2.0f * s * effective_field(
        var, state, row_ptr, col_ind, j_vals, h_vals);
}

// SA sample: grid covers num_reads (one thread per read).
// workspace: signed char[num_reads * N] annealing state
// out_spins: signed char[num_reads * N] final spins in {-1,+1}
__global__ void cuda_sa_sample(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ j_vals,
    const float* __restrict__ h_vals,
    signed char* __restrict__ workspace,
    signed char* __restrict__ out_spins,
    const float* __restrict__ beta_schedule,
    int num_betas,
    int sweeps_per_beta,
    int num_reads,
    int N,
    unsigned int base_seed
) {
    int read = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (read >= num_reads) return;

    signed char* state = workspace + (size_t)read * (size_t)N;
    signed char* out = out_spins + (size_t)read * (size_t)N;

    unsigned int rng = base_seed
        ^ (unsigned int)((read + 1) * 12345u)
        ^ 0x9E3779B9u;
    if (rng == 0u) rng = 0xdeadbeefu;

    // Random initial state
    for (int var = 0; var < N; ++var) {
        unsigned int r = xorshift32(rng);
        state[var] = (r & 1u) ? (signed char)-1 : (signed char)1;
    }

    if (N == 0) return;

    for (int bi = 0; bi < num_betas; ++bi) {
        float beta = __ldg(&beta_schedule[bi]);
        // Skip high-ΔE flips cheaply (same constant as v0.2 SA kernel).
        float threshold = 22.18f / fmaxf(beta, 1e-12f);

        for (int sweep = 0; sweep < sweeps_per_beta; ++sweep) {
            for (int var = 0; var < N; ++var) {
                float de = flip_delta(
                    var, state, row_ptr, col_ind, j_vals, h_vals);
                if (de >= threshold) {
                    continue;
                }
                bool flip = false;
                if (de <= 0.0f) {
                    flip = true;
                } else {
                    float accept = __expf(-de * beta);
                    float u = __uint2float_rn(xorshift32(rng)) * RNG_SCALE;
                    flip = (accept > u);
                }
                if (flip) {
                    state[var] = (signed char)(-state[var]);
                }
            }
        }
    }

    for (int var = 0; var < N; ++var) {
        out[var] = state[var];
    }
}

}  // extern "C"
