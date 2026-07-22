// SPDX-License-Identifier: AGPL-3.0-or-later
// CUDA single-site heat-bath Gibbs — explicit per-job (no nonce/slot economy).
// One thread = one independent read. Sequential site updates match the CPU
// miner algorithm (no chromatic coloring). Ported conditional from
// GPU/cuda_gibbs.cu: P(s=+1) = 1 / (1 + exp(2 * beta * h_eff)).

#define RNG_SCALE 2.32830643653869628906e-10f

extern "C" {

__device__ unsigned int xorshift32(unsigned int &state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

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

// Heat-bath sample: return +1 or -1.
__device__ signed char gibbs_sample_spin(
    float heff, float beta, unsigned int &rng
) {
    // Clamp exponent for numerical stability (matches CPU clamp range spirit).
    float arg = 2.0f * beta * heff;
    arg = fminf(fmaxf(arg, -80.0f), 80.0f);
    float p_plus = 1.0f / (1.0f + __expf(arg));
    float u = __uint2float_rn(xorshift32(rng)) * RNG_SCALE;
    return (u < p_plus) ? (signed char)1 : (signed char)-1;
}

__global__ void cuda_gibbs_sample(
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
        ^ (unsigned int)((read + 1) * 2654435761u)
        ^ 0xA5A5A5A5u;
    if (rng == 0u) rng = 0xC001D00Du;

    for (int var = 0; var < N; ++var) {
        unsigned int r = xorshift32(rng);
        state[var] = (r & 1u) ? (signed char)-1 : (signed char)1;
    }

    if (N == 0) return;

    for (int bi = 0; bi < num_betas; ++bi) {
        float beta = __ldg(&beta_schedule[bi]);
        for (int sweep = 0; sweep < sweeps_per_beta; ++sweep) {
            for (int var = 0; var < N; ++var) {
                float heff = effective_field(
                    var, state, row_ptr, col_ind, j_vals, h_vals);
                state[var] = gibbs_sample_spin(heff, beta, rng);
            }
        }
    }

    for (int var = 0; var < N; ++var) {
        out[var] = state[var];
    }
}

}  // extern "C"
